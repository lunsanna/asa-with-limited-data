import yaml
import re
import logging
from functools import partial
import multiprocessing
import time 

import pandas as pd
import torch
import evaluate

from datasets import Dataset
from transformers import (
    Wav2Vec2ForCTC, 
    Wav2Vec2Processor, 
    TrainingArguments, 
    EvalPrediction)

# For typing 
from typing import Literal, Optional, Pattern, Union, Dict, Tuple, Any, Callable
from evaluate import Metric

from helper import (
    prepare_example, 
    prepare_dataset, 
    compute_metrics, 
    DataCollatorCTCWithPadding, 
    CTCTrainer,
    configure_logger, 
    print_time)


################
# Start here

# 1. Check device and initiate logger
device:torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)

# 2. Check training language

# TODO: use HfArgumentParser for this
lang: Literal["fi", "sv"] = "fi"

if lang not in ["fi","sv"]:
    raise ValueError(f"Lang must be either fi or sv, got {lang}")

# 3. Definte constants
# Keys used to fetch the path of the csv file and the pretrained model from config.yml
if lang == "fi":
    CSV_KEY: str = "csv_fi" 
    PRETRAINED_KEY: str = "fi_pretrained"
elif lang == "sv":
    CSV_KEY: str = "csv_sv"
    PRETRAINED_KEY: str = "sv_pretrained"

################
# functions used in main()

def load_processor_and_model(path:str, 
                             model_args:Dict[str, Any]) -> Tuple[Wav2Vec2Processor, Wav2Vec2ForCTC]:
    """Loads the processor and model from pre-trained

    Args:
        path (str): path to the pre-trained model
        model_args (Dict[str, Any]): model arguments loaded from config.yml

    Returns:
        Tuple[Wav2Vec2Procesor, Wav2Vec2ForCTC]: processor and model for training
    """
    # 1. Load pre-trained processor, Wav2Vec2Processor
    start = time.time()
    processor = Wav2Vec2Processor.from_pretrained(
        path, 
        cache_dir=model_args.get("cache_dir", "./cache")
    )
    logger.debug(f"Pre-trained processor loaded. Time taken: {print_time(start)}.")

    # 2. Load pre-trained model, Wav2Vec2ForCTC
    start = time.time()
    model = Wav2Vec2ForCTC.from_pretrained(
            path, 
            cache_dir=model_args.get("cache_dir", "./cache"), 
            pad_token_id=processor.tokenizer.pad_token_id,
            vocab_size=len(processor.tokenizer)
        )
    logger.debug(f"Pre-trained model loaded. Time taken: {print_time(start)}.")

    if model_args.get("freeze_feature_encoder", True):
        model.freeze_feature_encoder()
    
    model.to(device)

    return processor, model

def load_data(df:pd.DataFrame, 
              data_args:Dict[str,Any]) -> Tuple[Dataset, Dataset]:
    """Split data into train and val, then process data for training

    Args:
        df (pd.DataFrame): data summary, contain file_path, split and normalised text
        data_args (Dict[str, Any]): data config loaded from config.yml

    Returns:
        Tuple[Dataset, Dataset]: dataset for training and validation
    """

    # 1. Create Dataset object, split the dataset into train and validation
    train_dataset: Dataset = Dataset.from_pandas(df[df.split!=i])
    val_dataset: Dataset = Dataset.from_pandas(df[df.split==i])
    logger.info(
        f"{len(train_dataset)} training samples, {len(val_dataset)} validation samples.")

    # 2. Process data with the prepare_example function 
    target_sr: int = data_args.get("target_feature_extractor_sampling_rate", 16000)
    vocab_chars: str = "".join(t for t in processor.tokenizer.get_vocab().keys() if len(t) == 1)
    text_cleaner_re: Pattern[str] = re.compile(f"[^\s{re.escape(vocab_chars)}]", flags=re.IGNORECASE)

    # -- Pass the first two arguments to the function
    prepare_example_partial = partial(prepare_example, target_sr, text_cleaner_re)

    # -- dataset columns after this step: speech, text, sampling_rate, duration_seconds
    start = time.time()
    train_dataset = train_dataset.map(
        prepare_example_partial, 
        remove_columns=["file_path","split"])
    logger.debug(
        f"Training set: speech signal successfully loaded. Time taken: {print_time(start)}.")

    start = time.time()
    val_dataset = val_dataset.map(prepare_example_partial, remove_columns=["file_path", "split"])
    logger.debug(
        f"Validation set: speech signal successfully loaded. Time taken: {print_time(start)}.")

    # 3. Process data with the prepare_dataset function
    # -- Pass the first two arguments to the function
    prepare_dataset_partial = partial(prepare_dataset, processor, target_sr)

    # -- dataset columns after this step: speech, text, sampling_rate, duration_seconds, input_values, labels
    start = time.time()
    train_dataset:Dataset = train_dataset.map(
        prepare_dataset_partial,
        num_proc=4,                 # TODO
        batched=True,
        batch_size=training_args.get("per_device_train_batch_size", 1),)
    logger.debug(
        f"Training set: features and labels sucessfully extracted. Time taken: {print_time(start)}.")
    
    start = time.time()
    val_dataset:Dataset = val_dataset.map(
        prepare_dataset_partial, 
        num_proc=4,                 # TODO
        batched=True,
        batch_size=training_args.get("per_device_eval_batch_size", 1))
    logger.debug(
        f"Validation set: features and labels sucessfully extracted. Time taken: {print_time(start)}.")

    return train_dataset, val_dataset

def run_train(fold:int, 
              processor: Wav2Vec2Processor, 
              model: Wav2Vec2ForCTC, 
              train_dataset: Dataset, 
              val_dataset: Dataset, 
              training_args: Dict[str, Any]) -> None:
    """Run training

    Args:
        fold (int)
        processor (Wav2Vec2Processor)
        model (Wav2Vec2ForCTC)
        train_dataset (Dataset)
        val_dataset (Dataset)
        training_args (Dict[str, Any]): args used to create TrainingArgument
    """

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    wer_metric: Metric = evaluate.load("wer")
    cer_metric: Metric = evaluate.load("cer")
    compute_metrics_partical:Callable[[EvalPrediction], Dict] = partial(compute_metrics, processor, wer_metric, cer_metric)

    training_args["output_dir"] = f"output_fold_{fold}"
    training_args = TrainingArguments(**training_args)

    trainer = CTCTrainer(
        model=model,
        data_collator=data_collator, 
        args=training_args,
        compute_metrics=compute_metrics_partical, 
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=processor.feature_extractor
    )

    logger.debug("Training starts now.")
    start = time.time()
    trainer.train()
    logger.debug(f"Training done in {print_time(start)}.")

    if training_args.load_best_model_at_end:
        print("Make prediction for the validation set.")
        predictions = trainer.predict(val_dataset[:10])
        print(compute_metrics_partical(predictions))


if __name__ == "__main__":
    # 1. Configs
    with open('config.yml', 'r') as file:
        train_config = yaml.safe_load(file)

    data_args: Dict[str, Union[bool, str, int]] = train_config["data_args"]
    model_args: Dict[str, Union[bool, str, int]] = train_config["model_args"]
    training_args: Dict[str, Union[bool, str, int]] = train_config["training_args"]

    # -- mix precision training only avaible for cuda
    if device==torch.device("cpu"):
        training_args["fp16"] = False

    # -- configure logger, log cuda info
    verbose_logging: bool = model_args.get("verbose_logging", True)
    configure_logger(verbose_logging)

    logger.debug(f"Running on {device}")
    if device != torch.device("cuda"):
        logger.warning("Cuda is not available!")
        logger.debug(f"Training {lang} model.")
    else:
        logger.debug(f"Working on {torch.cuda.device_count()} cudas.")

    # 2. Load csv file containing data summary
    # -- columns: file_path, split, normalised transcripts 

    csv_path: Optional[str] = data_args.get(CSV_KEY, None)
    assert csv_path, f"Trying to train {lang} model but {CSV_KEY} is not defined in config.yml."

    try:
        df = pd.read_csv(csv_path, 
                    encoding="utf-8",
                    usecols=["recording_path", "transcript_normalized", "split"])
    except:
        FileNotFoundError(
            f"Please copy the csv file to this directory from /scratch/work/lunt1/wav2vec2-finetune/{csv_path}")

    df.columns = ["file_path", "split", "text"]

    # 3. Run k-fold 
    k = 1
    for i in range(k):
        print(f"********** Runing fold {i} ********** ")

        pretrained_model_name_or_path: Optional[str] = model_args.get(PRETRAINED_KEY, None)
        assert pretrained_model_name_or_path, f"Trying to train {lang} model but {PRETRAINED_KEY} is not found in config.yml."

        print("LOAD PRE-TRAINED PROCESSOR AND MODEL")
        processor, model = load_processor_and_model(pretrained_model_name_or_path, model_args)

        print("LOAD DATA")
        train_dataset, val_dataset = load_data(df, data_args)

        print("TRAIN")
        run_train(i, processor, model, train_dataset, val_dataset, training_args)