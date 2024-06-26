import torch
import torchaudio
import evaluate
import yaml
import glob
import argparse
import logging
import time
import pandas as pd
from numpy import argmax
from functools import partial

from datasets import Dataset, concatenate_datasets
from transformers import Wav2Vec2Processor, AutoModelForAudioClassification, TrainingArguments, Trainer
from augmentations import apply_tranformations, transform_names, AugmentArguments, resample, ratings, CCLArguments

from helper import ModelArguments, DataArguments, configure_logger, print_memory_usage, print_time, MetricCallback

# for type annotation
from typing import List, Dict
from transformers import EvalPrediction

logger = logging.getLogger(__name__)

# Helper functions


def true_round(x):
    import decimal
    return int(decimal.Decimal(str(x)).quantize(decimal.Decimal("1"), rounding=decimal.ROUND_HALF_UP))


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
        df = df[df.label != c]
    if len(drop_classes) > 0:
        logger.warning(f"Dropped classes {drop_classes}")

    # do rounding if neccessary
    if df.label.dtype != int:
        logger.warning("Rounding required. Ensure that the correct dataset was loaded.")
        df["label"] = df["label"].apply(true_round)

    # shift classes so they start from 0
    df["label"] = df["label"] - df.label.min()

    return df


def load_speech(target_sr: int, example: Dict) -> Dict:
    """Used in the .map method to prepare each example in datasets"""
    example["speech"], example["sampling_rate"] = torchaudio.load(
        example["file_path"])
    example["speech"] = example["speech"].squeeze()

    example["duration_seconds"] = len(example["speech"])/target_sr
    return example


def extract_features(processor: Wav2Vec2Processor, batch: Dict) -> Dict:
    """Used in the .map method to prepare each example in datasets"""
    batch["input_values"] = processor(
        audio=batch["speech"],
        sampling_rate=processor.feature_extractor.sampling_rate,
        return_tensors="pt",
        padding="longest").input_values[0]
    return batch


def compute_metrics(pred: EvalPrediction, print_eval: bool = False) -> Dict:
    pred_logits = pred.predictions
    pred_ids = argmax(pred_logits, axis=-1)

    precision = precision_metric.compute(predictions=pred_ids, references=pred.label_ids, average="macro")
    recall = recall_metric.compute(predictions=pred_ids, references=pred.label_ids, average="macro")
    f1 = f1_metric.compute(predictions=pred_ids, references=pred.label_ids, average="macro")
    spearman = spearmanr_metric.compute(predictions=pred_ids, references=pred.label_ids)

    precision_weighted = precision_metric.compute(predictions=pred_ids, references=pred.label_ids, average="weighted")
    recall_weighted = recall_metric.compute(predictions=pred_ids, references=pred.label_ids, average="weighted")
    f1_weighted = f1_metric.compute(predictions=pred_ids, references=pred.label_ids, average="weighted")

    if print_eval and logger.isEnabledFor(logging.DEBUG):
        for pred, label in zip(pred_ids, pred.label_ids):
            logger.debug(f"label: {label}")
            logger.debug(f"pred: {pred}")

    return {**precision, **recall, **f1, **spearman,
            "precision_weighted": precision_weighted["precision"],
            "recall_weighted": recall_weighted["recall"],
            "f1_weighted": f1_weighted["f1"]}


def run_train(processor: Wav2Vec2Processor,
              model: AutoModelForAudioClassification,
              train_dataset: Dataset,
              eval_dataset: Dataset,
              training_args: TrainingArguments) -> None:
    """ Initialise trainer and run training."""

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.feature_extractor,
        compute_metrics=compute_metrics,
        callbacks=[MetricCallback]
    )

    logger.debug(f"Training starts now. {print_memory_usage()}")
    start = time.time()
    trainer.train()
    logger.info(f"Training completed. {print_time(start)}")

    predictions = trainer.predict(eval_dataset)
    compute_metrics(predictions, print_eval=True)


def uniform_mixing(datasets: List[Dataset], swap_proportion: float, num_of_unique_classes: int) -> List[Dataset]:
    """Swap part of the first dataset with the other datasets

    Args:
        datasets (List[Dataset]): Original list of datasets

    Returns:
        List[Dataset]: List of datasets after uniform mixing
    """
    assert len(datasets) == 3
    assert swap_proportion > 0 and swap_proportion < 1

    easy_set, medium_set, hard_set = datasets
    assert len(hard_set.unique("label")) != num_of_unique_classes, f"Do not combine the classes of different difficulty levels."
    print("Perform Uniform Mixing")
    # split the easy set 80% - 20%
    easy_set_split = easy_set.train_test_split(test_size=swap_proportion, seed=201123)

    # further split the 20% easy set into 50% - 50%
    easy_set_replace = easy_set_split["test"].train_test_split(test_size=0.5, seed=201123)

    # split the medium and hard set into 90% - 10%
    medium_set_split = medium_set.train_test_split(test_size=swap_proportion/2, seed=201123)
    hard_set_split = hard_set.train_test_split(test_size=swap_proportion/2, seed=201123)

    # final sets 
    easy_set = concatenate_datasets([easy_set_split["train"], medium_set_split["test"], hard_set_split["test"]]).shuffle()
    medium_set = concatenate_datasets([medium_set_split["train"], easy_set_replace["train"]]).shuffle()
    hard_set = concatenate_datasets([hard_set_split["train"], easy_set_replace["test"]]).shuffle()

    return [
        easy_set, 
        concatenate_datasets([easy_set, medium_set]).shuffle(),
        concatenate_datasets([easy_set, medium_set, hard_set]).shuffle()
    ]


def run_ccl_train(processor: Wav2Vec2Processor,
                  model: AutoModelForAudioClassification,
                  train_dataset: Dataset,
                  eval_dataset: Dataset,
                  training_args: TrainingArguments,
                  ccl_args: CCLArguments, 
                  um:bool=False) -> None:
    """Initialise trainer and run CCL training."""
    
    logger.debug(f"Class difficulty order: {ccl_args.difficulty_order}")
    logger.debug(f"n_epochs for each CCL phase: {ccl_args.n_epochs}")
    assert training_args.num_train_epochs == sum(ccl_args.n_epochs)
    # assert all([score in ccl_args.difficulty_order[-1] for score in train_dataset.unique("label")])

    train_datasets = [train_dataset.filter(
        lambda example: example["label"] in scores
    ) for scores in ccl_args.difficulty_order]
    
    if um:
        train_datasets = uniform_mixing(train_datasets, 0.2, len(train_dataset.unique("label")))

    for i, (n_epoch, current_train_set) in enumerate(zip(ccl_args.n_epochs, train_datasets)):
        print(f"Training with classes: {ccl_args.difficulty_order[i]}")

        # update epoch num
        training_args.num_train_epochs = n_epoch

        #  setup trainer
        trainer = Trainer(
            model=model if i == 0 else trainer.model,
            args=training_args,
            train_dataset=current_train_set,
            eval_dataset=eval_dataset,
            tokenizer=processor.feature_extractor,
            compute_metrics=compute_metrics,
            callbacks=[MetricCallback]
        )

        logger.debug(f"Training phase {i} starts now. {print_memory_usage()}")
        start = time.time()
        trainer.train()
        logger.info(f"Training completed. {print_time(start)}")

    predictions = trainer.predict(eval_dataset)
    compute_metrics(predictions, print_eval=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default="fi",help="Either fi or sv")
    parser.add_argument("--partial_model_path", type=str,default=None, help="Partial path to trained model")
    parser.add_argument("--resample", type=str, default=None,help="Resampling criteria. If set, augment arg will be ignored.")
    parser.add_argument("--augment", type=str, default=None,help="Augmentation method, ignored if resample is set.")
    parser.add_argument("--fold", type=int, default=None,help="Fold number, [0,3]")
    parser.add_argument("--resume_from", type=str,default=None, help="Checkpoint to resume from")
    parser.add_argument("--epoch", type=int, default=None, help="Set epoch to value different from config")
    parser.add_argument("--test", help="Test run", action="store_true")
    parser.add_argument("--ccl_training", help="Run class-wise curriculum learning or not", action="store_true")
    parser.add_argument("--um", help="Set uniform mixing or not", action="store_true")

    args = parser.parse_args()
    assert args.lang in ["fi", "sv"], f"Expected fi or sv, got {args.lang}"
    assert args.fold in range(4), f"Expected fold [0, 3], got {args.fold}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prep: Load config, setting params and config logger
    with open("config.yml") as f:
        config = yaml.safe_load(f)
    data_args = DataArguments(**config["data_args"])
    model_args = ModelArguments(**config["model_args"])
    augment_args = AugmentArguments(**config["augment_args"])
    asa_args = config["asa_args"]

    # Prep: Update training args based on arguments
    training_args = TrainingArguments(**config["asa_training_args"])
    training_args.output_dir = f"{training_args.output_dir}{args.fold}"
    training_args.resume_from_checkpoint = args.resume_from
    if args.epoch:
        training_args.num_train_epochs = args.epoch

    if args.lang == "fi":
        csv_path = data_args.csv_fi
        pretrained_path = model_args.fi_pretrained
    elif args.lang == "sv":
        csv_path = data_args.csv_sv
        pretrained_path = model_args.sv_pretrained

    configure_logger(model_args.verbose_logging)
    if device != torch.device('cuda'):
        logger.warning("Cuda is not available!!")

    # Prep: Load df
    drop_classes = asa_args.get("drop_classes", [])
    df = get_df(csv_path=csv_path, drop_classes=drop_classes)
    if args.test:
        df = df[:30]

    print(f"******** Training fold {args.fold} ********")

    # 1. Load pretrained processor
    processor = Wav2Vec2Processor.from_pretrained(
        pretrained_path,
        cache_dir=model_args.cache_dir)
    logger.debug(f"Processor loaded. {print_memory_usage()}")

    # 2. Load fine-tuned model
    model_path = args.resume_from if args.resume_from else glob.glob(
        f"{args.partial_model_path}{args.fold}/*")[0]
    model = AutoModelForAudioClassification.from_pretrained(
        model_path,
        cache_dir=model_args.cache_dir,
        num_labels=len(df.groupby('label')))
    logger.debug(f"Model loaded. {print_memory_usage()}")
    print(model)

    # 3. Create and processes train and eval datasets
    df_train = df[df.split != args.fold].drop("split", axis=1)
    df_eval = df[df.split == args.fold].drop("split", axis=1)
    train_dataset = Dataset.from_pandas(df_train)
    eval_dataset = Dataset.from_pandas(df_eval)
    train_dataset.set_format("pt")
    eval_dataset.set_format("pt")

    # -- load speech
    load_speech_partial = partial(
        load_speech, data_args.target_feature_extractor_sampling_rate)
    train_dataset = train_dataset.map(
        load_speech_partial, remove_columns=["file_path"])
    eval_dataset = eval_dataset.map(
        load_speech_partial, remove_columns=["file_path"])
    logger.debug(
        f"Speech loaded. N_train={len(train_dataset)}, N_eval={len(eval_dataset)}. {print_memory_usage()}")

    # -- extract features
    start = time.time()
    extract_features_partial = partial(extract_features, processor)
    train_dataset = train_dataset.map(
        extract_features_partial, batched=True, batch_size=1)
    eval_dataset = eval_dataset.map(
        extract_features_partial, batched=True, batch_size=1)
    logger.debug(
        f"Feature extracted. {print_time(start)} {print_memory_usage()}")

    # -- augment data
    if args.resample:
        assert args.resample in ratings, f"Expected {ratings}, got {args.resample}"
        train_dataset = resample(
            train_dataset, data_args, augment_args, criterion="label")
    elif args.augment:
        assert args.augment in transform_names, f"Check augment name"
        train_dataset = apply_tranformations(
            train_dataset, data_args, augment_args, args.augment)

    # 4. Define metrics
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")
    spearmanr_metric = evaluate.load("spearmanr")

    # 5. Train
    if model_args.freeze_feature_encoder:
        model.freeze_feature_encoder()
    if args.ccl_training:
        print("RUNNING CCL")
        ccl_args = CCLArguments(**config["ccl_args"])
        run_ccl_train(processor, model, train_dataset,eval_dataset, training_args, ccl_args, args.um)
    else:
        run_train(processor, model, train_dataset, eval_dataset, training_args)
