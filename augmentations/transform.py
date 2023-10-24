# -*- coding: utf-8 -*-
from WavAugment import augment
from datasets import Dataset, concatenate_datasets
from functools import partial
import time, copy, glob, logging, random
import numpy as np
import torchaudio
from collections import Counter
from math import isclose

from helper import print_time, DataArguments
from .AugmentArguments import (AugmentArguments, 
                               TimeMaskingArgs, 
                               BandRejectArgs, 
                               PitchShiftArgs, 
                               AdditiveNoiseArgs, 
                               TempoPerturbArgs, 
                               ReverbArgs)

# for typing
from torch import Tensor
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

def time_masking(data_args: DataArguments,
                 transform_args: TimeMaskingArgs,
                 example: Dict[str, Any]) -> Dict[str, Any]:

    # (seq_len) -> (1, seq_len) expected by WavAugment
    # Otherwise all speech are 1D arrays
    speech: Tensor = example["speech"]
    speech = speech.unsqueeze(dim=0) if speech.ndim == 1 else speech
    assert speech.ndim == 2, f"Expect 1-dim speech signal, got {speech.ndim-1}. Please ensure batched=False"

    # get hyperparams
    max_mask_proportion: float = transform_args.max_mask_proportion
    max_mask_size: float = transform_args.max_mask_size
    sampling_rate = data_args.target_feature_extractor_sampling_rate

    max_frames = int(sampling_rate*max_mask_size)  # max frames per mask

    # num of masks apply proportional to the chosen max mask percentage
    n_masks = int(max_mask_proportion*speech.size(1)//max_frames)

    # apply masking n_masks times
    mask_lengths = np.random.randint(0, max_frames, n_masks)
    for length in mask_lengths:
        start = np.random.randint(0, speech.size(1) - length)
        end = start + length
        speech[:, start:end, ...].zero_()

    example["speech"] = speech.squeeze()  # 1D speech array
    return example


def pitch_shift(data_args: DataArguments,
                transform_args: PitchShiftArgs,
                example: Dict[str, Any]) -> Dict[str, Any]:

    speech: Tensor = example["speech"]
    speech = speech.unsqueeze(dim=0) if speech.ndim == 1 else speech

    # Get hyperparams from config.yml
    sigma: int = transform_args.sigma
    sampling_rate = data_args.target_feature_extractor_sampling_rate

    def random_pitch_shift():
        return np.random.normal(0, sigma)

    transformed_speech = augment.EffectChain() \
        .pitch(random_pitch_shift) \
        .rate(sampling_rate) \
        .apply(speech, {'rate': sampling_rate}, {'rate': sampling_rate})

    example["speech"] = transformed_speech.squeeze()

    return example


def reverberation(data_args: DataArguments,
                  transform_args: ReverbArgs,
                  example: Dict[str, Any]) -> Dict[str, Any]:
    speech: Tensor = example["speech"]
    speech = speech.unsqueeze(dim=0) if speech.ndim == 1 else speech

    # get hyperparams from config.yml
    sampling_rate = data_args.target_feature_extractor_sampling_rate
    room_size_sigma = transform_args.room_size_sigma

    def random_room_size():
        r = np.random.normal(0, room_size_sigma)
        return min(abs(r), 100)

    transformed_speech = augment.EffectChain()\
        .reverb(50, 50, random_room_size) \
        .channels(1) \
        .apply(speech, {'rate': sampling_rate}, {'rate': sampling_rate})

    example["speech"] = transformed_speech.squeeze()

    return example


def additive_noise(data_args: DataArguments,
                   transform_args: AdditiveNoiseArgs,
                   example: Dict[str, Any]) -> Dict[str, Any]:
    speech: Tensor = example["speech"]  # (seq_len, ), used by Wav2Vec2
    speech = speech if speech.ndim == 2 else speech.unsqueeze(dim=0)  # (1, seq_len), expected by WavAugment
    assert speech.ndim == 2 and speech.size(0) == 1
    
    snr_low: int = transform_args.snr_low
    snr_high: int = transform_args.snr_high
    sampling_rate: int = data_args.target_feature_extractor_sampling_rate

    def noise_generator():
        # choose random noise sample from Musan
        noise_paths = glob.glob(f"{transform_args.noise_dir}/*.wav")
        random_noise_path = np.random.choice(noise_paths)
        noise, noise_sr = torchaudio.load(random_noise_path)
        if noise_sr != sampling_rate:
            resampler = torchaudio.transforms.Resample(noise_sr, sampling_rate)
            noise = resampler(noise)

        # adjust length to match speech
        speech_len: int = speech.size(1)
        noise_len: int = noise.size(1)

        if speech_len > noise_len:
            noise = noise.repeat(1, speech_len // noise_len + 1 )

        return noise[:, :speech_len] 

    # add noise
    snr = np.random.randint(snr_low, snr_high+1)
    example["speech"] = augment.EffectChain() \
        .additive_noise(noise_generator, snr) \
        .apply(speech, {'rate':sampling_rate}, {'rate': sampling_rate})
    
    # adjust dim
    example["speech"] = example["speech"].squeeze()

    return example


def band_reject(data_args: DataArguments,
                transform_args: BandRejectArgs,
                example: Dict[str, Any]) -> Dict[str, Any]:

    speech = example["speech"]
    speech = speech if speech.ndim == 2 else speech.unsqueeze(dim=0)
    assert speech.ndim == 2 and speech.size(0) == 1

    sampling_rate: int = data_args.target_feature_extractor_sampling_rate

    mask_width = np.random.randint(0, transform_args.max_mask_width) # mel 
    n_mask = np.random.choice([1,2])

    # the following two lines are taken from Introduction to Speech Processing by Tom Bäckström et el
    # https://speechprocessingbook.aalto.fi/Representations/Melcepstrum.html?highlight=mel
    def freq2mel(f): return 2595*np.log10(1 + (f/700))
    def mel2freq(m): return 700*(10**(m/2595) - 1)

    mel_max = freq2mel(sampling_rate/2) - mask_width
    start_mel = np.random.uniform(0, mel_max, n_mask)
    end_mel = start_mel + mask_width
    start_f = mel2freq(start_mel).astype(int)
    end_f = mel2freq(end_mel).astype(int)
    bands_to_reject = list(zip(start_f, end_f))

    # mel_max = freq2mel(sampling_rate/2 - mask_width)
    # starts_mel = np.random.uniform(0, mel_max, n_mask)
    # starts_f = mel2freq(starts_mel).astype(int)
    # ends = starts_f + mask_width
    # bands_to_reject = list(zip(starts_f, ends))

    effects = [["sinc", "-a", "120", f"{f2}-{f1}"]
               for f1, f2 in bands_to_reject]
    augmented_speech, sr = torchaudio.sox_effects.apply_effects_tensor(
        speech,
        sampling_rate,
        effects,
        channels_first=True
    )
    assert sr == sampling_rate
    example["speech"] = augmented_speech.squeeze()

    # print(start_f, mask_width)
    return example


def tempo_perturbation(data_args: DataArguments,
                       transform_args: TempoPerturbArgs,
                       example: Dict[str, Any]) -> Dict[str, Any]:
    speech: Tensor = example["speech"]
    speech = speech if speech.ndim == 2 else speech.unsqueeze(dim=0)
    assert speech.ndim == 2 

    perturb_factors: List[int] = transform_args.perturbation_factors
    perturb_factor: int = np.random.choice(perturb_factors)
    sampling_rate: int = data_args.target_feature_extractor_sampling_rate


    # apply tempo perturbation
    effects = [["tempo", str(perturb_factor)]]
    speech, sr = torchaudio.sox_effects.apply_effects_tensor(
        speech, sampling_rate, effects, channels_first=True
    )
    assert sampling_rate == sr

    example["speech"] = speech.squeeze()
    return example

def duplicate(example: Dict[str, Any]) -> Dict[str, Any]:
    """Apply no transformations"""
    return example

def random_transforms(data_args: DataArguments, 
                     augment_args: AugmentArguments,
                     example: Dict[str, Any]) -> Dict[str, Any]:
    speech = example["speech"]
    speech = speech if speech.ndim == 2 else speech.unsqueeze(dim=0)

    # 1. decide how many augmentations to perform 
    n_augment = np.random.randint(1, augment_args.max_num_of_transforms+1)
    transform_names = np.random.choice(list(transform_dict.keys()), n_augment)

    # 2. apply augmentations
    for transform_name in transform_names:
        transform = transform_dict[transform_name]
        example = transform(data_args, getattr(augment_args, transform_name), example)

    return example


transform_dict = {
    "time_masking": time_masking,
    "pitch_shift": pitch_shift,
    "reverberation": reverberation,
    "additive_noise": additive_noise,
    "band_reject": band_reject,
    "tempo_perturbation": tempo_perturbation
}

transform_names = list(transform_dict.keys())
transform_names.extend(["random_transforms", "duplicate"]) # Pseudo transforms
ratings = ["cefr_mean", "pronunciation_mean","fluency_mean", "accuracy_mean","range_mean","task_completion_mean"]

def apply_tranformations(train_dataset: Dataset,
                         data_args: DataArguments,
                         augment_args: AugmentArguments,
                         augment_name: str) -> Dataset:
    start = time.time()
    if augment_name == "random_transforms":
        transform = random_transforms
        transform_partial = partial(transform, data_args, augment_args)
    elif augment_name == "duplicate":
        transform_partial = duplicate
    else:
        transform = transform_dict[augment_name]
        transform_args = getattr(augment_args, augment_name)
        transform_partial = partial(transform, data_args, transform_args)

    if augment_args.copy:
        augmented_set = copy.deepcopy(train_dataset)
        augmented_set = augmented_set.map(transform_partial)
        train_dataset = concatenate_datasets([train_dataset, augmented_set]).shuffle()
    else:
        train_dataset = train_dataset.map(transform_partial)

    logger.debug(f"Training set (N={len(train_dataset)}): augmented training data added. {print_time(start)}")
    return train_dataset

def resample(train_dataset: Dataset, 
             data_args: DataArguments, 
             augment_args: AugmentArguments, 
             criterion: str = "rating") -> Dataset:
    """Resample data to balanced out the data based on the chosen rating (default = cefr_mean)"""
    start = time.time()
    
    train_copy = copy.deepcopy(train_dataset) # always create a copy 
    ratings = train_copy[criterion].tolist()

    # calculate samlping rate
    group_counts = Counter(ratings)
    n_group = len(group_counts)
    n_copy = 2 # double the dataset
    avg_n_samples_per_gp = len(train_copy)*n_copy/n_group
    n_samples = [(group, avg_n_samples_per_gp - count) for group, count in group_counts.items()]
    assert all([n > 0 for _, n in n_samples]), f"This calculation does not work. Might have to re-design."
    weights = {group: round(100*n/group_counts[group]) for group, n in n_samples}
    resample_weights = [weights[r] for r in ratings]

    # resample data based on weights
    n = random.choices(range(len(train_copy)), weights=resample_weights, k=len(train_copy))
    train_copy = train_copy.select(n) 

    # augment only the copied dataset, using random_transforms
    augment_args.copy = False
    train_copy = apply_tranformations(train_copy, data_args, augment_args, "random_transforms")

    # concat 
    train_dataset = concatenate_datasets([train_dataset, train_copy]).shuffle()
    rating_avg = sum(group_counts.keys())/len(group_counts)
    actual_avg = sum(train_dataset[criterion])/len(train_dataset)
    assert isclose(rating_avg, actual_avg,  rel_tol=0.1), f"Expect {actual_avg}, got {rating_avg}"

    logger.debug(f"Training set (N={len(train_dataset)}): data resampled. {print_time(start)}")
    return train_dataset
