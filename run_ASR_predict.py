import pandas as pd 
from numpy import vectorize
import decimal, yaml, argparse, time, glob
import torch
from functools import partial 
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments
from datasets import Dataset
from helper import DataArguments, ModelArguments, print_time
from run_finetune import load_speech, extract_features
from typing import Dict, Any

def get_prediction(processor: Wav2Vec2Processor, 
                   model: Wav2Vec2ForCTC,
                   device: torch.device, 
                   example: Dict[str, Any]) -> Dict[str, Any]:
    model.to(device)
    with torch.no_grad():
        input_values = example["input_values"].unsqueeze(dim=0)
        logits = model(input_values.to(device)).logits
    pred_ids = torch.argmax(logits, dim=-1)
    predictions = processor.batch_decode(pred_ids)[0]
    example["ASR_transcript"] = predictions
    return example

def get_config(path):
    with open(path, 'r') as file:
        train_config = yaml.safe_load(file)
    return train_config

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"ASR prediction running on {device}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=None, help="Fold number, 0-3")
    parser.add_argument("--partial_model_path", type=str, default=None, help="Path to model")
    args = parser.parse_args()
    assert args.fold in range(4), f"Expect fold 0-3, got {args.fold}"

    # This line needs to be change for every run 
    model_path = f"{args.partial_model_path}{args.fold}"

    # Load arguments 
    try:
        train_config = get_config("config.yml")
    except OSError or TypeError:
        train_config = get_config("../config.yml")
    
    data_args = DataArguments(**train_config["data_args"])
    model_args = ModelArguments(**train_config["model_args"])
    if device != torch.device("cuda"):
        train_config["training_args"]["fp16"] = False
    training_args = TrainingArguments(**train_config["training_args"])

    # 1. Load processor and model 
    print("Load processor and model.")
    processor = Wav2Vec2Processor.from_pretrained(model_args.fi_pretrained)
    try:
        model = Wav2Vec2ForCTC.from_pretrained(model_path)
    except OSError:
        model_path = glob.glob(f"{model_path}/checkpoint*")[0]
        model = Wav2Vec2ForCTC.from_pretrained(model_path)

    # 2. Load df from csv
    print("Load df from csv")
    try:
        df = pd.read_csv(data_args.csv_fi)
    except FileNotFoundError:
        df = pd.read_csv(f"../{data_args.csv_fi}")
        
    df = df.rename(columns={"recording_path":"file_path", 
                            "transcript_normalized":"text"})
    
    # 3. Get valiation set, load speech and extract features 
    val_dataset = Dataset.from_pandas(df[df.split==args.fold])
    val_dataset.set_format("pt")

    print("Load speech")
    val_dataset = load_speech("val", val_dataset, processor, data_args, remove_columns=[])
    print("Extract features")
    val_dataset = extract_features("val", val_dataset, processor, data_args, training_args)

    # 4. Get predictions
    print("Get predictions")
    start = time.time()
    get_prediction_partial = partial(get_prediction, processor, model, device)
    val_dataset = val_dataset.map(
        get_prediction_partial,
        remove_columns=["__index_level_0__", "speech", "sampling_rate", "input_values", "labels"], 
        num_proc=1 if device == torch.device("cuda") else 6)
    print(f"Finnished in {print_time(start)}")
    
    # 5. Save dataset as csv file 
    print("Save csv")
    val_dataset.to_csv(f"finnish_ASR_transcrip_fold{args.fold}.csv")
    print("All done")