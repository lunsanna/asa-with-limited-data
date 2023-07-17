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

from helper import print_time

# for typing 
from torch import Tensor
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

def time_masking(data_args: Dict[str, Any],
                 transform_args: Dict[str, Any],
                 example: Dict[str, Any]) -> Dict[str, Any]:
    
    # get deep copy of the example instead of altering the original dataset
    new_example = copy.deepcopy(example)
    # (seq_len) -> (1, seq_len) expected by WavAugment
    # Otherwise all speech are 1D arrays
    speech: Tensor = new_example["speech"]
    speech = speech.unsqueeze(dim=0) if speech.ndim == 1 else speech

    # get hyperparams
    max_mask_proportion: float = transform_args.get("max_mask_proportion")
    max_mask_size: float = transform_args.get("max_mask_size")
    min_mask_size: float = transform_args.get("min_mask_size")
    sampling_rate = data_args.get("target_feature_extractor_sampling_rate")
    
    assert max_mask_proportion < 1 and max_mask_proportion > 0, f"Masking proportion should be positive and < 1, got {max_mask_proportion}"
    assert speech.ndim == 2, f"Expect 1-dim speech signal, got {speech.ndim-1}. Please ensure batched=False"

    max_frames = int(sampling_rate*max_mask_size)  # max frames per mask
    min_frames = int(sampling_rate*min_mask_size) # min frames per mask

    # num of masks apply proportional to the chosen max mask percentage
    n_masks = int(max_mask_proportion*speech.size(1)//max_frames)

    # apply masking n_masks times
    mask_lengths = np.random.randint(min_frames, max_frames, n_masks)
    for length in mask_lengths:
        start = np.random.randint(0, speech.size(1) - length)
        end = start + length
        speech[:,start:end,...].zero_()

    new_example["speech"] = speech.squeeze()  # 1D speech array
    return new_example


def pitch_shift(data_args: Dict[str, Any],
                transform_args: Dict[str, Any], 
                example: Dict[str, Any]) -> Dict[str, Any]:
    
    new_example = copy.deepcopy(example)
    speech: Tensor = new_example["speech"]
    speech = speech.unsqueeze(dim=0) if speech.ndim == 1 else speech

    # Get hyperparams from config.yml
    sigma: int = transform_args.get("sigma")
    sampling_rate = data_args.get("target_feature_extractor_sampling_rate")
    assert sampling_rate 

    def random_pitch_shift():
        shift = 0 
        while shift == 0:
            shift = np.random.normal(0, sigma)
        return shift 

    transformed_speech = augment.EffectChain() \
        .pitch(random_pitch_shift) \
        .rate(sampling_rate) \
        .apply(speech, {'rate': sampling_rate}, {'rate': sampling_rate})

    new_example["speech"] = transformed_speech.squeeze()
    
    return new_example 

def reverberation(data_args: Dict[str, Any], 
                  transform_args: Dict[str, Any], 
                  example: Dict[str, Any]) -> Dict[str, Any]:
    # create a new copy of the example instead of mutating the original example
    new_example = copy.deepcopy(example)
    speech: Tensor = new_example["speech"]
    speech = speech.unsqueeze(dim=0) if speech.ndim == 1 else speech

    # get hyperparams from config.yml
    sampling_rate = data_args.get("target_feature_extractor_sampling_rate")
    room_size_sigma = transform_args.get("room_size_sigma")
    assert sampling_rate and room_size_sigma

    def random_room_size():
        r = np.random.normal(0,room_size_sigma)
        return min(abs(r), 100)
    
    transformed_speech = augment.EffectChain()\
        .reverb(50, 50, random_room_size) \
        .channels(1) \
        .apply(speech, {'rate': sampling_rate}, {'rate': sampling_rate})
    
    new_example["speech"] = transformed_speech.squeeze()

    return new_example

def additive_noise(data_args: Dict[str, Any], 
                  transform_args: Dict[str, Any], 
                  example: Dict[str, Any]) -> Dict[str, Any]:
    # copy example 
    new_example = copy.deepcopy(example)
    speech: Tensor = new_example["speech"] # (seq_len, ), used by Wav2Vec2
    speech = speech if speech.ndim == 2 else speech.unsqueeze(dim=0) # (1, seq_len), expected by WavAugment
    snr_low: int = transform_args["snr_low"]
    snr_high: int = transform_args["snr_high"]
    sampling_rate: int = data_args["target_feature_extractor_sampling_rate"]
    assert speech.ndim == 2 and speech.size(0) == 1

    # choose random noise sample from Musan
    noise_dir = transform_args["noise_dir"]
    noise_paths = glob.glob(f"{noise_dir}/*.wav")
    noise_paths = noise_paths if len(noise_paths) > 0 else glob.glob(f"../{noise_dir}/*")
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
        noise = noise.repeat(1,repeat_factor)
    noise = noise[:,:speech_len]
    assert speech.size() == noise.size()

    # add noise 
    snr = np.random.randint(snr_low, snr_high+1)
    snr_linear = 10**(snr/10)
    speech_power = (speech**2).mean()
    noise_power = (noise**2).mean()
    noise_factor = np.sqrt(speech_power/(noise_power*snr_linear))
    new_example["speech"] = speech + noise_factor * noise
    # adjust dim 
    new_example["speech"] = new_example["speech"].squeeze()

    return new_example

def band_reject(data_args: Dict[str, Any], 
                transform_args: Dict[str, Any], 
                example: Dict[str, Any]) -> Dict[str, Any]:
    
    new_example = copy.deepcopy(example)
    speech = new_example["speech"]
    speech = speech if speech.ndim == 2 else speech.unsqueeze(dim=0)
    assert speech.ndim == 2 and speech.size(0) == 1
    max_mask_proportion = transform_args["max_mask_proportion"]
    assert max_mask_proportion > 0 and max_mask_proportion < 1
    sampling_rate: int = data_args["target_feature_extractor_sampling_rate"]
    
    mask_width = transform_args["mask_width"] # constant
    n_mask = int((sampling_rate/2)*max_mask_proportion//transform_args["mask_width"])

    # the following two lines are taken from Introduction to Speech Processing by Tom Bäckström et el
    # https://speechprocessingbook.aalto.fi/Representations/Melcepstrum.html?highlight=mel
    def freq2mel(f): return 2595*np.log10(1 + (f/700))
    def mel2freq(m): return 700*(10**(m/2595) - 1)
    
    mel_upper = freq2mel(sampling_rate/2 - mask_width)
    starts_mel = np.random.uniform(0, mel_upper, n_mask)
    starts_f = mel2freq(starts_mel).astype(int)
    ends = starts_f + mask_width
    bands_to_reject = list(zip(starts_f, ends))

    effects = [["sinc", "-a", "120", f"{f2}-{f1}"] for f1, f2 in bands_to_reject]
    augmented_speech, sr = torchaudio.sox_effects.apply_effects_tensor(
        speech, 
        sampling_rate, 
        effects, 
        channels_first=True
    )
    assert sr == sampling_rate
    new_example["speech"] = augmented_speech.squeeze()

    return new_example


def tempo_perturbation(data_args: Dict[str, Any], 
                       transform_args: Dict[str, Any], 
                       example: Dict[str, Any]) -> Dict[str, Any]:
    new_example = copy.deepcopy(example)
    speech: Tensor = new_example["speech"]
    speech = speech if speech.ndim==2 else speech.unsqueeze(dim=0)
    perturb_factors: List[int] = transform_args["perturbation_factors"]
    perturb_factor: int = choice(perturb_factors)
    sampling_rate: int = data_args.get("target_feature_extractor_sampling_rate")

    assert speech.ndim==2 and sampling_rate

    # apply tempo perturbation
    effects = [["tempo", str(perturb_factor)]]
    speech, sr = torchaudio.sox_effects.apply_effects_tensor(
        speech, sampling_rate, effects, channels_first=True
    )
    assert sampling_rate==sr

    new_example["speech"] = speech.squeeze()
    return new_example
    

transform_dict = {
    "time_masking": time_masking, 
    "pitch_shift": pitch_shift, 
    "reverberation": reverberation,
    "additive_noise": additive_noise,
    "band_reject": band_reject, 
    "tempo_perturbation": tempo_perturbation
}

def apply_tranformations(train_dataset: Dataset,  
                         data_args: Dict[str, Any], 
                         augment_args: Dict[str, Any], 
                         augment_name: str) -> Dataset:
    start = time.time()
    transform = transform_dict[augment_name]
    transform_args = augment_args[augment_name] # params for the chosen transformation

    transform_partial = partial(transform, data_args, transform_args)
    augmented_train_dataset = train_dataset.map(transform_partial)
    
    if augment_args["copy"]:
        train_dataset = concatenate_datasets([train_dataset, augmented_train_dataset]).shuffle()
    else:
        train_dataset = augmented_train_dataset

    logger.debug(f"Training set (N={len(train_dataset)}): augmented training data added. {print_time(start)}")
    return train_dataset

transform_names = transform_dict.keys()