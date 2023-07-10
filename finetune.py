#!/usr/bin/env python3
import yaml
import re
import logging
from functools import partial
from multiprocessing import cpu_count
import time
import argparse

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

from augmentations import apply_tranformations, transform_names
from helper import (
    prepare_example,
    prepare_dataset,
    compute_metrics,
    DataCollatorCTCWithPadding,
    CTCTrainer,
    MetricCallback,
    configure_logger,
    print_time,
    print_memory_usage)

# Check device and initiate logger
device: torch.device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)

################
# functions related to data handling


def get_df(lang: Literal["fi", "sv"],
           data_args: Dict[str, Any]) -> pd.DataFrame:
    """Load csv file based on lang

    Args:
        lang (str): either fi or sv
        data_args (Dict[str, Any]): data args defined in config.yml

    Returns:
        pd.DataFrame: summary of the training and val data
    """
    csv_key = "csv_fi" if lang == "fi" else "csv_sv"
    csv_path: Optional[str] = data_args[csv_key]
    try:
        df = pd.read_csv(csv_path,
                         encoding="utf-8",
                         usecols=["recording_path", "transcript_normalized", "split"])
    except:
        FileNotFoundError(f"The data summary file {csv_path} not found, please check the file path in config.yml.")
    df.columns = ["file_path", "split", "text"]
    return df

def load_speech(train_dataset: Dataset,
                val_dataset: Dataset,
                processor: Wav2Vec2Processor,
                data_args: Dict[str, Any]) -> Tuple[Dataset, Dataset]:
    """Load speech data with prepare_example function. 
    New columns after this step: speech, sampling_rate, duration_seconds
    Existing text column will be pre-processed a well. 

    Args:
        train_dataset (Dataset)
        val_dataset (Dataset)
        processor (Wav2Vec2Processor): pre-trained processor, used to get vocab
        data_args (Dict[str, Any]): data config loaded from config.yml

    Returns:
        Tuple[Dataset, Dataset]: dataset for training and validation
    """
    target_sr: int = data_args["target_feature_extractor_sampling_rate"]

    # define data regex text cleaner to process text
    vocab_chars: str = "".join(t for t in processor.tokenizer.get_vocab().keys() if len(t) == 1)
    text_cleaner_re: Pattern[str] = re.compile(f"[^\s{re.escape(vocab_chars)}]", flags=re.IGNORECASE)

    # Pass the first two arguments to the function
    prepare_example_partial = partial(prepare_example, target_sr, text_cleaner_re)

    # Apply the prepare example function too all examples 
    # -- training set
    start = time.time()
    train_dataset = train_dataset.map(
        prepare_example_partial,
        remove_columns=["file_path", "split"])
    logger.debug(f"Training set (N={len(train_dataset)}): speech successfully loaded. {print_time(start)}")
    logger.debug(f"{print_memory_usage()}")

    # -- validation set
    start = time.time()
    val_dataset = val_dataset.map(
        prepare_example_partial, remove_columns=["file_path", "split"])
    logger.debug(f"Validation set (N={len(val_dataset)}): speech successfully loaded. {print_time(start)}")
    logger.debug(f"{print_memory_usage()}")

    return train_dataset, val_dataset

def extract_features(train_dataset: Dataset, 
                     val_dataset: Dataset, 
                     processor: Wav2Vec2Processor, 
                     data_args: Dict[str, Any], 
                     training_args: Dict[str, Any]) -> Tuple[Dataset, Dataset]:
    """Process data with the prepare_dataset function
    New columns after this step: input_values, labels

    Args:
        train_dataset (Dataset)
        val_dataset (Dataset)
        processor (Wav2Vec2Processor): pre-trained processor, used to process speech and text
        data_args (Dict[str, Any]): data args read from config.yml
        training_args (Dict[str, Any]): training args read from config.yml

    Returns:
        Tuple[Dataset, Dataset]: train and val data set ready for training
    """
    
    target_sr: int = data_args["target_feature_extractor_sampling_rate"]

    # Pass the first two arguments to the function
    prepare_dataset_partial = partial(prepare_dataset, processor, target_sr)

    # Training set 
    start = time.time()
    num_proc = 6 if cpu_count() >= 6 else cpu_count()
    train_dataset: Dataset = train_dataset.map(
        prepare_dataset_partial,
        num_proc=num_proc,
        batched=True,
        batch_size=training_args["per_device_train_batch_size"])
    logger.debug(f"Training set (N={len(train_dataset)}): features and labels sucessfully extracted. {print_time(start)}")
    logger.debug(f"{print_memory_usage()}")

    # Validation set
    start = time.time()
    val_dataset: Dataset = val_dataset.map(
        prepare_dataset_partial,
        num_proc=num_proc,
        batched=True,
        batch_size=training_args["per_device_eval_batch_size"])
    logger.debug(f"Validation set (N={len(val_dataset)}): features and labels sucessfully extracted. {print_time(start)}")
    logger.debug(f"{print_memory_usage()}")

    return train_dataset, val_dataset

################
# functions related to processor and model

def load_processor_and_model(path: str,
                             model_args: Dict[str, Any]
                             ) -> Tuple[Wav2Vec2Processor, Wav2Vec2ForCTC]:
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
    logger.debug(f"Pre-trained processor loaded. {print_time(start)}")
    logger.debug(f"{print_memory_usage()}")

    # 2. Load pre-trained model, Wav2Vec2ForCTC
    start = time.time()
    model = Wav2Vec2ForCTC.from_pretrained(
        path,
        cache_dir=model_args.get("cache_dir", "./cache"),
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer)
    )
    logger.debug(f"Pre-trained model loaded. {print_time(start)}")
    logger.debug(f"{print_memory_usage()}")

    if model_args.get("freeze_feature_encoder", True):
        model.freeze_feature_encoder()

    return processor, model

################
# function related to training


def run_train(fold: int,
              processor: Wav2Vec2Processor,
              model: Wav2Vec2ForCTC,
              train_dataset: Dataset,
              val_dataset: Dataset,
              training_args: Dict[str, Any]) -> None:
    """Initialise trainer and  run training

    Args:
        fold (int)
        processor (Wav2Vec2Processor)
        model (Wav2Vec2ForCTC)
        train_dataset (Dataset)
        val_dataset (Dataset)
        training_args (Dict[str, Any]): args used to create TrainingArgument
    """

    data_collator = DataCollatorCTCWithPadding(
        processor=processor, padding=True)

    # Set up compute metric function
    wer_metric: Metric = evaluate.load("wer")
    cer_metric: Metric = evaluate.load("cer")
    compute_metrics_partical: Callable[[EvalPrediction], Dict] = partial(
        compute_metrics, processor, wer_metric, cer_metric)

    # Update output dir based on fold
    output_dir = training_args.get("output_dir", "output")
    training_args["output_dir"] = f"{output_dir[:-1]}{fold}" if output_dir[-1].isnumeric() else f"{output_dir}_fold_{fold}"
    training_args = TrainingArguments(**training_args)

    # Set up trainer
    trainer = CTCTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics_partical,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=processor.feature_extractor,
        callbacks=[MetricCallback]
    )

    # Print metrics before training
    metrics_before_train = compute_metrics_partical(
        trainer.predict(val_dataset), print_examples=False)
    print({"eval_wer": metrics_before_train["wer"],
          "eval_cer": metrics_before_train["cer"]})

    # Train
    logger.debug(f"Training starts now. {print_memory_usage()}")
    start = time.time()
    trainer.train()
    logger.info(
        f"Trained {training_args.num_train_epochs} epochs. {print_time(start)}.")

    # Save the last checkpoint
    if training_args.load_best_model_at_end:
        print(f"Best model save at {trainer.state.best_model_checkpoint}")
    else:
        trainer.save_model(f"{training_args.output_dir}/final")
        print(f"Best model save at {training_args.output_dir}/final")

    predictions = trainer.predict(val_dataset)
    compute_metrics_partical(predictions, print_examples=True)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default="sv", help="Model language, either fi or sv.")
    parser.add_argument("--augment", type=str, default=None, help="Name of augmentation method")
    parser.add_argument("--test", help="Test run", action="store_true")
    parser.add_argument("--fold", type=int, default=None, help="Fold number, 0-3")
    args = parser.parse_args()

    assert args.lang in ["fi", "sv"], f"Lang must be either fi or sv, got {args.lang}."
    assert args.fold in range(4), f"Expect fold 0-3, got {args.fold}"

    # 1. Configs
    with open('config.yml', 'r') as file:
        train_config = yaml.safe_load(file)

    data_args: Dict[str, Union[bool, str, int]] = train_config["data_args"]
    model_args: Dict[str, Union[bool, str, int]] = train_config["model_args"]
    training_args: Dict[str, Union[bool, str, int]] = train_config["training_args"]
    # training_args["local_rank"] = int(os.environ["LOCAL_RANK"])

    # -- configure logger, log cuda info
    verbose_logging: bool = model_args.get("verbose_logging", True)
    configure_logger(verbose_logging)

    logger.debug(f"Running on {device}")
    if device != torch.device("cuda"):
        # mix precision training only avaible for cuda
        training_args["fp16"] = False
        logger.warning("Cuda is not available!")
        logger.debug(f"Training {args.lang} model.")
    else:
        logger.debug(f"Cuda count: {torch.cuda.device_count()}")

    # 2. Load csv file containing data summary
    # -- columns: file_path, split, normalised transcripts
    df: pd.DataFrame = get_df(args.lang, data_args)
    if args.test:
        df = df[:30]
        training_args["num_train_epochs"] = 1

    # 3. Fetch the path of the pre-trained model
    key = "fi_pretrained" if args.lang == "fi" else "sv_pretrained"
    pretrained_name_or_path: Optional[str] = model_args[key]

    # 4. Run k-fold
    print(f"********** Runing fold {args.fold} ********** ")

    print("LOAD PRE-TRAINED PROCESSOR AND MODEL")
    processor, model = load_processor_and_model(pretrained_name_or_path, model_args)

    print("LOAD DATA")
    # -- split dataset into train and validation
    train_dataset: Dataset = Dataset.from_pandas(df[df.split != args.fold])
    val_dataset: Dataset = Dataset.from_pandas(df[df.split == args.fold])
    train_dataset.set_format("pt")
    val_dataset.set_format("pt")

    # -- load speech and other info from path 
    train_dataset, val_dataset = load_speech(train_dataset, val_dataset, processor, data_args)
    
    # -- apply augmentations
    if args.augment:
        assert args.augment in transform_names, f"Expect {transform_names}, got {args.augment}"
        print(f"AUGMENT DATA, METHOD: {args.augment}")
        augment_args: Dict[str, Any] = train_config["augment_args"]
        train_dataset = apply_tranformations(train_dataset, data_args, augment_args, args.augment)
    
    print("EXTRACT FEATURES")
    train_dataset, val_dataset = extract_features(train_dataset, val_dataset, 
                                                  processor, data_args,training_args)
    
    print("TRAIN")
    run_train(args.fold, processor, model, train_dataset, val_dataset, training_args)
