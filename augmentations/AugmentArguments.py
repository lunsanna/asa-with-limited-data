from typing import Dict, List 

class TimeMaskingArgs(object):
    def __init__(self, 
                 max_mask_proportion: float=0.2, 
                 max_mask_size:float=0.2) -> None:
        self.max_mask_proportion = max_mask_proportion
        self.max_mask_size = max_mask_size
        assert self.max_mask_proportion <= 1, f"Proportion must be <= 1, got {self.max_mask_proportion}"

class BandRejectArgs(object):
    def __init__(self, 
                 max_mask_width: int = 64) -> None:
        self.max_mask_width = max_mask_width

class PitchShiftArgs(object):
    def __init__(self, 
                 sigma: int = 50) -> None:
        self.sigma = sigma

class ReverbArgs(object):
    def __init__(self, 
                 room_size_sigma: int = 60) -> None:
        self.room_size_sigma = room_size_sigma

class AdditiveNoiseArgs(object):
    def __init__(self, 
                 noise_dir: str = "",
                 snr_low: int = 10, 
                 snr_high: int = 50) -> None:
        self.noise_dir = noise_dir
        self.snr_low = snr_low
        self.snr_high = snr_high
        assert self.snr_high >= self.snr_low, f"SNR upper bound must be larger than lower bound."

class TempoPerturbArgs(object):
    def __init__(self, perturbation_factors: List[int]=[0.9, 0.95, 1.05, 1.1]) -> None:
        self.perturbation_factors = perturbation_factors
        assert isinstance(self.perturbation_factors, list) and len(self.perturbation_factors) > 0

class AugmentArguments(object):
    def __init__(self, 
                 copy: bool = True,
                 max_num_of_transforms: int = 2,
                 time_masking: Dict = None, 
                 band_reject: Dict = None, 
                 pitch_shift: Dict = None, 
                 reverberation: Dict = None, 
                 additive_noise: Dict = None, 
                 tempo_perturbation: Dict = None):
        
        self.copy = copy 
        self.max_num_of_transforms = max_num_of_transforms
        
        self.time_masking = TimeMaskingArgs(**time_masking) if time_masking else TimeMaskingArgs()
        self.band_reject = BandRejectArgs(**band_reject) if band_reject else BandRejectArgs()
        self.pitch_shift = PitchShiftArgs(**pitch_shift) if pitch_shift else PitchShiftArgs()
        self.reverberation = ReverbArgs(**reverberation) if reverberation else ReverbArgs()
        self.additive_noise = AdditiveNoiseArgs(**additive_noise) if additive_noise else AdditiveNoiseArgs()
        self.tempo_perturbation = TempoPerturbArgs(**tempo_perturbation) if tempo_perturbation else TempoPerturbArgs()

        

