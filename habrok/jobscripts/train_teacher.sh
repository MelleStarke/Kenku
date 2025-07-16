#!/bin/bash
#================

#SBATCH --partition gpu
#SBATCH --cpus-per-task 15
#SBATCH --mem-per-cpu 2G
#SBATCH --time 0-0:30:00
#SBATCH --nodes 1
#SBATCH --gpus-per-node=a100:1

#SBATCH --job-name="train_teacher_max_size_test"
#SBATCH --output="/home3/s4984218/Kenku/habrok/jobscripts/results/train_teacher/max_size_test_%j.out"

module purge

source ~/scripts/activate_venv.sh

echo "Script Contents"

cat ~/Kenku/habrok/jobscripts/train_teacher.sh

echo "Running script"
date +"%H:%M:%S"

# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64,garbage_collection_threshold:0.6

cd ~/Kenku
srun --export=ALL python -m train.train_model --test-interval 50 --melspec-interval 50 --checkpoint-interval 100 --checkpoint-max 3 --dataset-dir ~/scratch/processed --n-cores $SLURM_CPUS_PER_TASK --min-samples 7 --stack-factor 8 --conv-ch 1024 --att-ch 1024 -dor 0.2 --epochs 50 --main-loss mse --batch-size 500 --max-test-batches 200 --run-dir ~/scratch/runs/teacher/max_size_test_$SLURM_JOB_ID

echo "Finished training"

#================
