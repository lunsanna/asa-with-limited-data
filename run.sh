#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH -p gpu-nvlink,dgx-spa
#SBATCH --time=2-00:00:00
#SBATCH --job-name=digitala_fi
#SBATCH --mem=20G
#SBATCH --output=output.out
#SBATCH --error=errors.err

module load anaconda
module load cuda 
source activate w2v2

srun python -u finetune.py --lang=fi --augment=time_masking

# torchrun --nproc_per_node=1 finetune.py --lang=fi
