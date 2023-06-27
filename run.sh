#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=01:00:00
#SBATCH -J digitala_fi
#SBATCH --mem=20G
#SBATCH --output=output.out
#SBATCH --error=errors.err

module load anaconda
source activate w2v2

srun torchrun finetune.py --lang=fi

# watch -n 5 nvidia-smi >> gpu_usage.txt
# srun python -u finetune.py --lang=fi

# srun python -u -m torch.distributed.launch \
    # --nproc_per_node 4 finetune.py --lang=fi