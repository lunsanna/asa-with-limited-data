#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH -p gpu-nvlink,dgx-spa
#SBATCH --time=0-10:00:00
#SBATCH --job-name=no_augment
#SBATCH --mem=10G
#SBATCH --array=0-3
#SBATCH --output=ex0_output_%a.out
#SBATCH --error=ex0_errors_%a.err

module load anaconda
module load cuda 
source activate w2v2

srun python -u finetune.py \
--lang=fi \
--fold=$SLURM_ARRAY_TASK_ID \
# --augment=time_masking \
# -- test

# torchrun --nproc_per_node=1 finetune.py --lang=fi
