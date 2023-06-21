import torchaudio

text_updates = []

def prepare_example(target_sr, vocab_cleaner, example):
    """This function will be used to prepare each example when constructing
    train_dataset and val_dataset

    Args:
        example (Dataset.LazyRow): one example from the dataset
        target_sr (int): target sample rate defined in config.yml
        vocab_cleaner (Pattern[str]): a regular expression object 

    Returns:
        Dataset.Lazy: one example ready for training
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

def prepare_dataset(processor, target_sr, batch):
    """_summary_

    Args:
        processor (_type_): _description_
        target_sr (_type_): _description_
        batch (_type_): _description_

    Returns:
        _type_: _description_
    """
    # check sr
    sr = set(batch["sampling_rate"])
    assert len(sr)==1 and sr.pop()==target_sr, f"Ensure all sampling rate match {target_sr}"
    
    # speech => feature representation, using feature extractor
    batch["input_values"] = processor(
        audio=batch["speech"], 
        sampling_rate=target_sr, 
        return_tensors="pt", 
        padding=True
    ).input_values
    
    # text => labels, using tokenizer
    batch["labels"] = processor(text=batch["text"]).input_ids
    
    return batch



