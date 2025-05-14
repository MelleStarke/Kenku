#!/bin/bash
#================

#SBATCH --partition gpu
#SBATCH --cpus-per-task 24
#SBATCH --mem 12G
#SBATCH --time 0-0:30:00
#SBATCH --nodes 1
#SBATCH --gpus-per-node=v100:1

#SBATCH --job-name="train_kenkuteacher"
#SBATCH --output="/home3/s4984218/Kenku/habrok/jobscripts/results/train_teacher/%j.out"

module purge

source ~/scripts/activate_venv.sh

echo "Script Contents"

cat ~/Kenku/habrok/jobscripts/train_teacher.sh

echo "Running script"
date +"%H:%M:%S"

cd ~/Kenku
srun python -m train.train_model --test-interval 100 --melspec-interval 100 --checkpoint-interval 200 --dataset-dir ~/scratch/processed --n-cores $SLURM_CPUS_PER_TASK --min-samples 7 --stack-factor 4 --conv-ch 256 --att-ch 256 -dor 0.2 --epochs 40 --main-loss mse --batch-size 120 --max-test-batches 150 --run-dir ~/scratch/runs/teacher/$SLURM_JOB_ID
# srun python -m train.train_model --dataset-dir ~/scratch/processed --n-cores $SLURM_CPUS_PER_TASK --min-samples 7 --stack-factor 4 --conv-ch 80 --att-ch 80 -dor 0.2 --epochs 20 --batch-size 256 --max-test-batches 100 --run-dir ~/scratch/runs/$SLURM_JOB_ID

echo "Finished training"

#================
