#!/bin/bash
#================

#SBATCH --partition gpu
#SBATCH --cpus-per-task 25
#SBATCH --mem-per-cpu 2G
#SBATCH --time 0-15:00:00
#SBATCH --nodes 1
#SBATCH --gpus-per-node=1
#SBATCH --array=1-4

#SBATCH --job-name="hypertune_drl_teacher"
#SBATCH --output="/home3/s4984218/Kenku/habrok/jobscripts/results/hypertune_drl_teacher/%j.out"

module purge

source ~/scripts/activate_venv.sh

echo "Shell Script Contents"
cat ~/Kenku/habrok/jobscripts/hypertune_drl_teacher.sh

echo "Running script"
date +"%H:%M:%S"

cd ~/Kenku
srun python -m train.hypertune ./train/configs/hypertune/drl_teacher ~/scratch/runs/hypertune_drl_teacher/$SLURM_ARRAY_JOB_ID/ $SLURM_ARRAY_TASK_ID --n-cores $SLURM_CPUS_PER_TASK 

echo "Finished training"