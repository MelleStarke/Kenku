#!/bin/bash
#================

#SBATCH --partition gpu
#SBATCH --cpus-per-task 25
#SBATCH --mem-per-cpu 2G
#SBATCH --time 0:20:00
#SBATCH --nodes 1
#SBATCH --gpus-per-node=1

#SBATCH --job-name="train_teacher_no_wd"
#SBATCH --output="/home3/s4984218/Kenku/habrok/jobscripts/results/train_teacher/tuned/no_weight_decay/%j.out"

module purge

source ~/scripts/activate_venv.sh

echo "Script Contents"

cat ~/Kenku/habrok/jobscripts/train_teacher.sh

echo "Running script"
date +"%H:%M:%S"

# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64,garbage_collection_threshold:0.6

cd ~/Kenku
# srun --export=ALL python -m train.train_model -lr 1e-6 -wda 2000 -woa 2000 --test-interval 50 --melspec-interval 50 --checkpoint-interval 100 --checkpoint-max 5 --dataset-dir ~/scratch/processed --n-cores $SLURM_CPUS_PER_TASK --min-samples 7 --stack-factor 1 --conv-ch 512 --att-ch 256 -dor 0.2 --epochs 200 --main-loss mse --batch-size 200 --max-test-batches 200 --run-dir ~/scratch/runs/teacher/tuned/$SLURM_JOB_ID
srun --export=ALL python -m train.train_model --config-dir /home3/s4984218/Kenku/train/configs/best_teacher --run-dir ~/scratch/runs/teacher/tuned/no_weight_decay_$SLURM_JOB_ID --batch-size 800 --epochs 200 --checkpoint-max 2 --att-weight-decay 0.0

echo "Finished training"

#================
