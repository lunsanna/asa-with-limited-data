from WavAugment import augment
from datasets import Dataset, concatenate_datasets
from functools import partial 
import logging
import time 
import numpy as np
import copy

from helper import print_time

# for typing 
from torch import Tensor
from typing import Dict, Any

logger = logging.getLogger(__name__)

def time_masking(data_args: Dict[str, Any],
                 transform_args: Dict[str, Any],
                 example: Dict[str, Any]) -> Dict[str, Any]:
    
    # get deep copy of the example instead of altering the original dataset
    new_example = copy.deepcopy(example)
    # (seq_len) -> (1, seq_len) expected by WavAugment
    # Otherwise all speech are 1D arrays
    speech: Tensor = new_example["speech"].unsqueeze(dim=0) 

    # get hyperparams
    max_mask_proportion: float = transform_args["max_mask_proportion"]
    max_mask_size: float = transform_args["max_mask_size"]
    sampling_rate = data_args["target_feature_extractor_sampling_rate"]
    
    assert max_mask_proportion < 1 and max_mask_proportion > 0, f"Masking proportion should be positive and < 1, got {max_mask_proportion}"
    assert speech.ndim == 2, f"Expect 1-dim speech signal, got {speech.ndim-1}. Please ensure batched=False"

    max_frames = sampling_rate*max_mask_size  # max frames per mask
    effect_chain = augment.EffectChain().time_dropout(max_frames=max_frames)

    # num of masks apply proportional to the chosen max mask percentage
    n_masks = int(max_mask_proportion*speech.size(1)//max_frames)

    # apply masking n_masks times
    src_info = {'rate': sampling_rate} # resampled to target sr in earlier steps
    target_info = {'rate': sampling_rate}
    transformed_speech = speech.clone()
    for _ in range(n_masks):
        transformed_speech = effect_chain.apply(transformed_speech, src_info, target_info)

    new_example["speech"] = transformed_speech.squeeze()  # 1D speech array
    return new_example


def pitch_shift(data_args: Dict[str, Any],
                transform_args: Dict[str, Any], 
                example: Dict[str, Any]) -> Dict[str, Any]:
    
    new_example = copy.deepcopy(example)
    speech: Tensor = new_example["speech"].unsqueeze(dim=0)

    # Get hyperparams
    sigma: int = transform_args["sigma"]
    sampling_rate = data_args["target_feature_extractor_sampling_rate"]
    assert sigma, f"Sigma value must be set to use pitch shift"

    effect_chain = augment.EffectChain() \
        .pitch(lambda: np.random.normal(0, sigma)) \
        .rate(sampling_rate)

    src_info = {'rate': sampling_rate}
    target_info = {'rate': sampling_rate}
    transformed_speech = effect_chain.apply(speech, src_info, target_info)
    new_example["speech"] = transformed_speech.squeeze()
    
    return new_example 

transform_dict = {
    "time_masking": time_masking, 
    "pitch_shift": pitch_shift
}

def apply_tranformations(train_dataset: Dataset,  
                         data_args: Dict[str, Any], 
                         augment_args: Dict[str, Any], 
                         augment_name: str):
    start = time.time()
    transform = transform_dict[augment_name]
    transform_args = augment_args[augment_name] # params for the chosen transformation

    transform_partial = partial(transform, data_args, transform_args)
    augmented_train_dataset = train_dataset.map(transform_partial)
    
    if augment_args["copy"]:
        train_dataset = concatenate_datasets([train_dataset, augmented_train_dataset]).shuffle(seed=42)
    else:
        train_dataset = augmented_train_dataset

    logger.debug(f"Training set (N={len(train_dataset)}): augmented training data added. {print_time(start)}")
    return train_dataset

transform_names = transform_dict.keys()