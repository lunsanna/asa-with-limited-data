# wav2vec2-finetune

This projec is a refactored version of the `l2-speech-scoring-tools` developed by [Aalto-speech](https://github.com/aalto-speech/l2-speech-scoring-tools). The refactoring was done for my own understanding and practice. This project will be developed further for my master's thesis by adding data augmentation. The code will be shared in another repository. 

Since the data is not public as of the creating of project, you will need access to Aalto's database to reproduce the results.

### Brief description
- `config.yml` contains all the model, data and training parameters.
- `environment.yml` defines the conda env that the code is run on.
- `finetune.py` fine-tunes the model pre-trained on native speech. 
- `run.sh` runs `finetune.py` on Triton.
- `helper` folder contains all the functions that are not directly run in main(). 
- `others` files that are reference and can be remove later. 

### Get started 
1. Clone this repo and cd into it
2. Copy the csv files that contains the data summary to this directory
```
cp /scratch/work/lunt1/wav2vec2-finetune/finnish_df.csv .
cp /scratch/work/lunt1/wav2vec2-finetune/swedish_df.csv .
```
3. Create conda env 
```
conda env create --file environment.yml
```
4. Run the code on Triton
- Check `config.yml` to see if all parameters look good. 
- Check `run.sh` and set `--lang` to the desired language (either `fi` or `sv`).
- And then run: 
```
sbatch run.sh
```
