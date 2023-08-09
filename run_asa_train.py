import torch, torchaudio, evaluate, yaml, glob, argparse
import pandas as pd
from functools import partial

from datasets import Dataset
from transformers import Wav2Vec2Processor, AutoModelForAudioClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, recall_score, f1_score

from helper import ModelArguments, DataArguments

# for type annotation 
from typing import List, Dict
from transformers import EvalPrediction

# Helper functions 
def get_df(csv_path, drop_classes: List[int]) -> pd.DataFrame:
    """Process data frame before
    Args:
        df (pd.DataFrame)
        drop_classes (List[int]): sparse classes to remove 
    """
    # load df from csv
    df = pd.read_csv(csv_path, usecols=['recording_path', 'cefr_mean', 'split'])
    df = df.rename(columns={'recording_path': 'file_path', 'cefr_mean': 'label'})

    # drop sparse classes
    for c in drop_classes:
        df = df[df.label!=c]

    # shift classes so they start from 0
    df["label"] = df["label"] - df.label.min()

    return df

def load_speech(target_sr: int, example: Dict) -> Dict:
    """Used in the .map method to prepare each example in datasets"""
    example["speech"], example["sampling_rate"] = torchaudio.load(example["file_path"])
    example["speech"] = example["speech"].squeeze()

    example["duration_seconds"] = len(example["speech"])/target_sr
    return example

def extract_features(processor: Wav2Vec2Processor, batch: Dict) -> Dict:
    return processor(audio=batch["speech"], 
                     sampling_rate=processor.feature_extractor.sampling_rate)

def compute_metrics(pred: EvalPrediction) -> Dict:
    pred_logits = pred.predictions
    pred_ids = torch.argmax(pred_logits, axis=-1)
    
    print(len(pred.label_ids[pred.label_ids == -100]))
    
    accuracy = accuracy_metric(predictions=pred_ids, references=pred.label_ids)
    recall = recall_metric(predictions=pred_ids, references=pred.label_ids)
    f1 = f1_metric(predictions=pred_ids, references=pred.label_ids)
    spearman = spearmanr_metric(prediction=pred_ids, reference=pred.label_ids)
    
    accuracy_v2 = accuracy_score(pred.label_ids, pred_ids)
    print(f'evaluate: {accuracy}, sklearn: {accuracy_v2}')
    
    return {'accuracy':accuracy, 'recall':recall, 'f1':f1, 'spearman': spearman}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default="fi", help="Either fi or sv")
    parser.add_argument("--partial_model_path", type=str, defalut=None, help="Partial path to trained model")
    parser.add_argument("--fold", type=int, default=None, help="Fold number, [0,3]")
    args = parser.parse_args()
    assert args.fold in range(4), f"Expected fold [0, 3], got {args.fold}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load metrics
    f1_metric = evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")
    spearmanr_metric = evaluate.load("spearmanr")
    recall_metric = evaluate.load("recall")

    # Load config
    with open("config.yml") as f:
        config = yaml.safe_load(f)
    data_args = DataArguments(**config["data_args"])
    model_args = ModelArguments(**config["model_args"])
    asa_args = config["asa_args"]
    
    if args.lang == "fi":
        csv_path = data_args.csv_fi
        pretrained_path = model_args.fi_pretrained
    elif args.lang == "sv":
        csv_path = data_args.csv_sv
        pretrained_path = model_args.sv_pretrained

    # Load df
    drop_classes = asa_args.get("drop_classes", [])
    df = get_df(csv_path=csv_path, drop_classes=drop_classes)

    print(f"******** Training fold {args.fold} ********")
    
    # Load pretrained processor
    processor = Wav2Vec2Processor.from_pretrained(
        pretrained_path, 
        cache_dir=model_args.cache_dir).to(device)

    # Load fine-tuned model
    model_path = glob.glob(f"{args.partial_model_path}{args.fold}/*")[0]
    model = AutoModelForAudioClassification.from_pretrained(
        model_path, 
        cache_dir=model_args.cache_dir,
        num_labels=len(df.groupby('label'))).to(device)

    # Create and processes train and eval datasets
    df_train = df[df.split != args.fold].drop("split", axis=1)
    df_eval = df[df.split == args.fold].drop("split", axis=1)

    load_speech_partial = partial(load_speech, data_args.target_feature_extractor_sampling_rate)
    extract_features_partial = partial(extract_features, processor)

    train_dataset = Dataset.from_pandas(df_train)
    train_dataset = train_dataset.map(load_speech_partial, remove_columns=["file_path"])
    train_dataset = train_dataset.map(extract_features_partial, batched=True, batch_size=1)

    eval_dataset = Dataset.from_pandas(df_eval)
    eval_dataset = eval_dataset.map(load_speech_partial, remove_columns=["file_path"])
    eval_dataset = eval_dataset.map(extract_features_partial, batched=True, batch_size=1)

    # Define metrics 
    f1_metric = evaluate.load('f1')
    accuracy_metric = evaluate.load('accuracy')
    spearmanr_metric = evaluate.load('spearmanr')
    recall_metric = evaluate.load('recall')


    # Train
    training_args = TrainingArguments(
        output_dir=f'asa_output_fold_{args.fold}',
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=3e-5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=2,
        num_train_epochs=10,
        warmup_ratio=0.1,
        load_best_model_at_end=False,
        save_total_limit=1,
        metric_for_best_model='f1',
        push_to_hub=False,
        gradient_checkpointing=True
        )

    if model_args.freeze_feature_encoder:
        model.freeze_feature_encoder()

    trainer = Trainer(
        model=model, 
        args=training_args,
        train_dataset=train_dataset, 
        eval_dataset=eval_dataset,
        tokenizer=processor.feature_extractor,
        compute_metrics=compute_metrics
    )

    trainer.train()