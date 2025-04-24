#!/bin/bash
#================

#SBATCH --partition gpu
#SBATCH --cpus-per-task 2
#SBATCH --mem 10G
#SBATCH --time 0-5:00:00
#SBATCH --nodes 1
#SBATCH --gpus-per-node=a100:1

#SBATCH --job-name="train_kenkuteacher"
#SBATCH --output="/home3/s4984218/Kenku/habrok/jobscripts/results/train_kenkuteacher_%j.out"

module purge

source ~/scripts/activate_venv.sh

echo "Script Contents"

cat ~/Kenku/habrok/jobscripts/train_teacher.sh

echo "Running script"

cd ~/Kenku
srun python -m train.train_model --dataset-dir ~/scratch/processed --preload-melspecs --n-cores $SLURM_CPUS_PER_TASK --min-samples 7 --stack-factor 4 --conv-ch 512 --att-ch 512 -dor 0.2 --epochs 50 --main-loss mae --batch-size 200 --max-test-batches 150 --run-dir ~/scratch/runs/teacher/$SLURM_JOB_ID
# srun python -m train.train_model --dataset-dir ~/scratch/processed --n-cores $SLURM_CPUS_PER_TASK --min-samples 7 --stack-factor 4 --conv-ch 80 --att-ch 80 -dor 0.2 --epochs 20 --batch-size 256 --max-test-batches 100 --run-dir ~/scratch/runs/$SLURM_JOB_ID

echo "Finished training"

#================

