from WavAugment import augment
from torch import Tensor
from typing import Dict, Any


def time_masking(max_mask_proportion: float,
                 data_args: Dict[str, Any],
                 example: Dict[str, Any]
                 ) -> Dict[str, Any]:

    speech: Tensor = example["speech"]  # 1D speech array (seq_len)

    assert max_mask_proportion < 1 and max_mask_proportion > 0, f"Masking proportion should be positive and < 1, got {max_mask_proportion}"
    assert speech.ndim == 1, f"Expect 1-dim speech signal, got {speech.ndim}. Please ensure batched=False"

    # 2D speech array (1, seq_len) expected by WavAugment
    speech = speech.unsqueeze(dim=0)

    sampling_rate = data_args.get(
        "target_feature_extractor_sampling_rate", 16000)
    max_frames = sampling_rate/5  # max frames per mask
    effect_chain = augment.EffectChain().time_dropout(max_frames=max_frames)

    # num of masks apply proportional to the chosen max mask percentage
    n_masks = int(max_mask_proportion*speech.size(1)//max_frames)

    # apply masking n_masks times
    augmented_speech = speech.clone()
    for _ in range(n_masks):
        augmented_speech = effect_chain.apply(augmented_speech,
                                              {'rate': sampling_rate},
                                              {'rate': sampling_rate})

    example["speech"] = augmented_speech.squeeze()  # 1D speech array
    return example
