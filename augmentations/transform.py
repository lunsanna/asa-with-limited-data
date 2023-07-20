from WavAugment import augment
from datasets import Dataset, concatenate_datasets
from functools import partial
from random import choice
import logging
import time 
import numpy as np
import copy
import glob
import torchaudio

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

    # choose random noise sample from Musan
    noise_dir = transform_args.noise_dir
    noise_paths = glob.glob(f"{noise_dir}/*.wav")
    noise_paths = noise_paths if len(
        noise_paths) > 0 else glob.glob(f"../{noise_dir}/*")
    random_noise_path: str = choice(noise_paths)
    noise, noise_sr = torchaudio.load(random_noise_path)
    if noise_sr != sampling_rate:
        resampler = torchaudio.transforms.Resample(noise_sr, sampling_rate)
        noise = resampler(noise)

    # adjust length to match speech
    speech_len: int = speech.size(1)
    noise_len: int = noise.size(1)
    assert speech_len > 0 and noise_len > 0, f"Speech: {speech_len}, noise: {noise_len}"

    if speech_len > noise_len:
        repeat_factor: int = speech_len//noise_len + 1
        noise = noise.repeat(1, repeat_factor)
    noise = noise[:, :speech_len]
    assert speech.size() == noise.size()

    # add noise
    snr = np.random.randint(snr_low, snr_high+1)
    snr_linear = 10**(snr/10)
    speech_power = (speech**2).mean()
    noise_power = (noise**2).mean()
    noise_factor = np.sqrt(speech_power/(noise_power*snr_linear))
    example["speech"] = speech + noise_factor * noise
    # adjust dim
    example["speech"] = example["speech"].squeeze()

    return example


def band_reject(data_args: DataArguments,
                transform_args: BandRejectArgs,
                example: Dict[str, Any]) -> Dict[str, Any]:

    speech = example["speech"]
    speech = speech if speech.ndim == 2 else speech.unsqueeze(dim=0)
    assert speech.ndim == 2 and speech.size(0) == 1

    max_mask_proportion = transform_args.max_mask_proportion
    sampling_rate: int = data_args.target_feature_extractor_sampling_rate

    mask_width = np.random.randint(0, transform_args.max_mask_width)
    n_mask = int((sampling_rate/2)*max_mask_proportion//transform_args.max_mask_width)

    # the following two lines are taken from Introduction to Speech Processing by Tom Bäckström et el
    # https://speechprocessingbook.aalto.fi/Representations/Melcepstrum.html?highlight=mel
    def freq2mel(f): return 2595*np.log10(1 + (f/700))
    def mel2freq(m): return 700*(10**(m/2595) - 1)

    mel_max = freq2mel(sampling_rate/2 - mask_width)
    starts_mel = np.random.uniform(0, mel_max, n_mask)
    starts_f = mel2freq(starts_mel).astype(int)
    ends = starts_f + mask_width
    bands_to_reject = list(zip(starts_f, ends))

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

    return example


def tempo_perturbation(data_args: DataArguments,
                       transform_args: TempoPerturbArgs,
                       example: Dict[str, Any]) -> Dict[str, Any]:
    speech: Tensor = example["speech"]
    speech = speech if speech.ndim == 2 else speech.unsqueeze(dim=0)
    assert speech.ndim == 2 

    perturb_factors: List[int] = transform_args.perturbation_factors
    perturb_factor: int = choice(perturb_factors)
    sampling_rate: int = data_args.target_feature_extractor_sampling_rate


    # apply tempo perturbation
    effects = [["tempo", str(perturb_factor)]]
    speech, sr = torchaudio.sox_effects.apply_effects_tensor(
        speech, sampling_rate, effects, channels_first=True
    )
    assert sampling_rate == sr

    example["speech"] = speech.squeeze()
    return example

def random_transform(data_args: DataArguments, 
                     augment_args: AugmentArguments,
                     example: Dict[str, Any]) -> Dict[str, Any]:
    speech = example["speech"]
    speech = speech if speech.ndim == 2 else speech.unsqueeze(dim=0)
    example = time_masking(data_args, getattr(augment_args, "time_masking"), example)
    example = band_reject(data_args, getattr(augment_args, "band_reject"), example)
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
transform_names.append("random_transforms")

def apply_tranformations(train_dataset: Dataset,
                         data_args: DataArguments,
                         augment_args: AugmentArguments,
                         augment_name: str) -> Dataset:
    start = time.time()
    if augment_name == "random_transforms":
        transform = random_transform
        transform_partial = partial(transform, data_args, augment_args)
    else:
        transform = transform_dict[augment_name]
        transform_args = getattr(augment_args, augment_name)
        transform_partial = partial(transform, data_args, transform_args)

    augmented_set = copy.deepcopy(train_dataset)
    augmented_set = augmented_set.map(transform_partial)

    if augment_args.copy:
        train_dataset = concatenate_datasets([train_dataset, augmented_set]).shuffle()
    else:
        train_dataset = augmented_set

    logger.debug(f"Training set (N={len(train_dataset)}): augmented training data added. {print_time(start)}")
    return train_dataset


