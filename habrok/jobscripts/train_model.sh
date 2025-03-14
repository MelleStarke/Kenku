#!/bin/bash

#SBATCH --partition gpu
#SBATCH --cpus-per-task 4
#SBATCH --mem 128G
#SBATCH --time 0-5:0:00
#SBATCH --nodes 1
#SBATCH --gpus-per-node=a100:1

#SBATCH --job-name="train_kenkuteacher"
#SBATCH --output="/home3/s4984218/Kenku/habrok/jobscripts/results/train_kenkuteacher_%j.out"

module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load scikit-learn/1.3.1-gfbf-2023a

echo "Finished loading modules"

source ~/venvs/py3.11.5/bin/activate

echo "Activated venv"

echo "Running script"

cd ~/Kenku
srun python -m train.train_model --dataset-dir ~/scratch/processed --preload-melspecs --n-cores $SLURM_CPUS_PER_TASK --min-samples 7 --conv-ch 512 --att-ch 512 -dor 0.2 --epochs 20 --batch-size 50 --max-test-batches 150 --run-dir ~/scratch/runs/$SLURM_JOB_ID

echo "Finished training"

deactivate