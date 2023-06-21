# wav2vec2-finetune

### Brief description
- `config.yml` contains all the model, data and training parameters.
- `environment.yml` defines the conda env that the code is run on.
- `finetune.py` fine-tunes the model pre-trained on native speech. 
- `run.sh` runs `finetune.py` on Triton.
- `helper` folder contains all the functions that are not directly run in main(). 
- `output.out` and `errors.err`contains outputs produced by runing `finetune.py`.
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
```
sbatch run.sh
```