#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH -p gpu-nvlink,dgx-spa
#SBATCH --time=0-00:30:00
#SBATCH --job-name=no_augment
#SBATCH --mem=10G
#SBATCH --array=0
#SBATCH --output=asa_output_%a.out
#SBATCH --error=asa_errors_%a.err

module load anaconda
module load cuda 
source activate w2v2

srun python -u /scratch/work/lunt1/wav2vec2-finetune/run_asa_train.py \
--lang=fi \
--partial_model_path=output_fold_ \
--fold=$SLURM_ARRAY_TASK_ID \
# --resume_from=asa_output_fold_2/checkpoint-7056 \
# --epoch=2
# --test

# torchrun --nproc_per_node=1 finetune.py --lang=fi
