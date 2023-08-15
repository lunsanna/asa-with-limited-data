#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=0-00:15:00
#SBATCH --job-name=asr_no_augment
#SBATCH --mem=5G
#SBATCH --array=0-3
#SBATCH --output=asr_output_%a.out
#SBATCH --error=asr_errors_%a.err

module load anaconda
module load cuda 
source activate w2v2

srun python -u /scratch/work/lunt1/wav2vec2-finetune/run_predict.py \
--fold=$SLURM_ARRAY_TASK_ID \
--partial_model_path=output_fold_
