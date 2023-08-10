#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH -p gpu-nvlink,dgx-spa
#SBATCH --time=0-09:00:00
#SBATCH --job-name=no_augment
#SBATCH --mem=30G
#SBATCH --array=0-3
#SBATCH --output=output_%a.out
#SBATCH --error=errors_%a.err

module load anaconda
module load cuda 
source activate w2v2

srun python -u /scratch/work/lunt1/wav2vec2-finetune/run_finetune.py \
--lang=fi \
--fold=$SLURM_ARRAY_TASK_ID \
# --augment=time_masking \
# --resample=cefr_mean \
# -- test

# torchrun --nproc_per_node=1 finetune.py --lang=fi
