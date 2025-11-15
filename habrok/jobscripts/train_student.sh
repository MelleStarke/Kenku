#!/bin/bash
#================

#SBATCH --partition gpu
#SBATCH --cpus-per-task 25
#SBATCH --mem-per-cpu 2G
#SBATCH --time 2-12:00:00
#SBATCH --nodes 1
#SBATCH --gpus-per-node=1

#SBATCH --job-name="train_student_final"
#SBATCH --output="/home3/s4984218/Kenku/habrok/jobscripts/results/train_student/final/%j.out"

module purge

source ~/scripts/activate_venv.sh

echo "Script Contents"

cat ~/Kenku/habrok/jobscripts/train_student.sh

echo "Running script"
date +"%H:%M:%S"

# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64,garbage_collection_threshold:0.6

cd ~/Kenku
srun --export=ALL python -m train.train_model --config-dir /home3/s4984218/Kenku/train/configs/train/student --run-dir ~/scratch/runs/student/final/$SLURM_JOB_ID --n-cores $SLURM_CPUS_PER_TASK

echo "Finished training"

#================
