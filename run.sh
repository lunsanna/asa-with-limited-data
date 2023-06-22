#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=6
#SBATCH --time=00:00:00
#SBATCH -J digitala_fi
#SBATCH --mem=20G
#SBATCH --output=output.out
#SBATCH --error=errors.err

module load anaconda
source activate w2v2

srun python -u finetune.py --lang=fi