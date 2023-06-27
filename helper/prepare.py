#!/usr/bin/env python3
import torchaudio
from typing import Pattern, Any, Dict
from transformers import Wav2Vec2Processor

text_updates = []

def prepare_example(target_sr: int, 
                    vocab_cleaner: Pattern[str], 
                    example: Dict[str, Any]) -> Dict[str, Any]:
    """This function will be used when construct train_dataset and val_dataset.
    It loads the speech signals from the file paths, also save the sampling rate and 
    duration of the speech files. 

     Original columns: file_path, split, normalised text
     Final columns: text, speech, duration_seconds, sampling_rate

    Args:
        example (Dict[str, Any]): one example from the dataset
        target_sr (int): target sample rate defined in config.yml
        vocab_cleaner (Pattern[str]): a regular expression object 

    Returns:
        Dict[str, Any]: one example ready for training
    """
    # Load speech from file path
    example["speech"], example["sampling_rate"] = torchaudio.load(example["file_path"])
    example["speech"] = example["speech"].squeeze()
    
    # Resample the speech if necessary
    if example["sampling_rate"] != target_sr:
        resampler = torchaudio.transforms.Resample(example["sampling_rate"], target_sr)
        example["speech"] = resampler(example["speech"])

    example["duration_seconds"] = len(example["speech"])/target_sr

    # Remove out-of-vocab characters 
    updated_text = vocab_cleaner.sub("", example["text"])
    if updated_text != example["text"]:
        text_updates.append((example["text"], updated_text))
        example["text"] = updated_text

    return example

def prepare_dataset(processor:Wav2Vec2Processor, 
                    target_sr:int, 
                    batch:Dict[str, Any]) -> Dict[str, Any]:
    """This function will be used to prepare train_dataset and val_dataset. 
    It uses the processor to extract features from speech signals, and uses 
    the tokenizer to convert the text to labels. 

    Original columns: text, speech, duration_seconds, sampling_rate
    Final columns: speech, text, sampling_rate, duration_seconds, input_values, labels

    Args:
        processor (Wav2Vec2Processor): 
        target_sr (int): _description_
        batch (Dict[str, Any]): _description_

    Returns:
        Dict[str, Any]: _description_
    """
    # check sr
    sr = set(batch["sampling_rate"].tolist())
    assert len(sr)==1 and sr.pop()==target_sr, f"All sampling rate much match {target_sr}, got {sr}"

    # speech => feature representation, using feature extractor
    batch["input_values"] = processor(
        audio=batch["speech"], # (batch_size, seq_len)
        sampling_rate=target_sr, 
        return_tensors="pt", 
        padding="longest"
    ).input_values[0]

    # text => labels, using tokenizer
    batch["labels"] = processor(text=batch["text"]).input_ids
    
    return batch



