class DataArguments(object):
    def __init__(self, 
                 csv_fi: str = "finnish_df.csv", 
                 csv_sv: str = "swedish_df.csv", 
                 csv_fi_synth: str = "/scratch/work/lunt1/TTS/csv_fi_synth.csv",
                 target_feature_extractor_sampling_rate: int = 16000) -> None:
        self.csv_fi = csv_fi
        self.csv_sv = csv_sv
        self.csv_fi_synth = csv_fi_synth
        self.target_feature_extractor_sampling_rate = target_feature_extractor_sampling_rate

class ModelArguments(object):
    def __init__(self, 
                 sv_pretrained: str, 
                 fi_pretrained: str, 
                 cache_dir: str, 
                 freeze_feature_encoder: bool = True, 
                 verbose_logging: bool = True) -> None:
        self.sv_pretrained = sv_pretrained
        self.fi_pretrained = fi_pretrained
        self.cache_dir = cache_dir
        self.freeze_feature_encoder = freeze_feature_encoder
        self.verbose_logging = verbose_logging