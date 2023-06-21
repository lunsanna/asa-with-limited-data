#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH --time=00:45:00
#SBATCH -J digitala_fi
#SBATCH --mem=20G
#SBATCH --output=output.out
#SBATCH --error=errors.err

module load anaconda
source activate w2v2

srun python -u finetune.py