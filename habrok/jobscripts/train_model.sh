#!/bin/bash

#SBATCH --partition gpu
#SBATCH --cpus-per-task 1
#SBATCH --mem 10G
#SBATCH --time 0-3:30:00
#SBATCH --nodes 1
#SBATCH --gpus-per-node=a100:1

#SBATCH --job-name="train_kenkuteacher"
#SBATCH --output="/home3/s4984218/Kenku/habrok/jobscripts/results/train_kenkuteacher_%j.out"

module purge

source ~/scripts/activate_venv.sh

echo "Running script"

cd ~/Kenku
srun python -m train.train_model --dataset-dir ~/scratch/processed --preload-melspecs --n-cores $SLURM_CPUS_PER_TASK --min-samples 7 --stack-factor 1 --conv-ch 512 --att-ch 512 -dor 0.2 --epochs 20 --batch-size 64 --max-test-batches 150 --run-dir ~/scratch/runs/$SLURM_JOB_ID

echo "Finished training"