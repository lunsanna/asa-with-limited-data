# Oversampling, Augmentation and Curriculum Learning for Speaking Assessment with Limited Training Data

This project is a refactored version of the `l2-speech-scoring-tools` developed by [Aalto-speech](https://github.com/aalto-speech/l2-speech-scoring-tools). This project explores methods includeing data augmentation, oversampling and curriculum learning to alleviate challenges related to training wav2vec-based Automatic Speaking Assessment models using **small** and **imbalanced** datasets.

The datasets can be downloaded from [https://www.kielipankki.fi/corpora/digitala/](https://www.kielipankki.fi/corpora/digitala/). 
The pre-trained Swedish model [KBLab/wav2vec2-large-voxrex-swedish](https://huggingface.co/KBLab/wav2vec2-large-voxrex-swedish) is publicly available on HuggingFace. 
The pre-trained Finnish model is unfortunately not publicly available. 

### Brief description
- `config.yml` contains all the model, data and training parameters.
- `environment.yml` defines the conda env that the code is run on.
- `run_finetune.py` fine-tunes the wav2vec2 model pre-trained on native Finnish/Finland Swedish speech. 
- `run_predict.py` use the fine-tuned models for prediction. 
- `run_finetune.sh` runs `run_finetune.py` on Triton.
- `run_predict.sh` runs `run_predict.py` on Triton.
- `augmentations` folder contains everything to do with data augmentation. 
- `helper` folder contains all the functions that are not directly run in main(). 
- `others` files that are reference and can be remove later. 

### Get started 
1. Clone this repo and cd into it
2. Create conda env 
```
conda env create --file environment.yml
```
3. Install WavAugment 
```
git clone git@github.com:facebookresearch/WavAugment.git && cd WavAugment && python setup.py develop
```
4. Run the code on Triton
- Check `config.yml` to see if all parameters look good. 
- Check `run.sh` and set `--lang` to the desired language (either `fi` or `sv`).
- And then run: 
```
sbatch run.sh
```
