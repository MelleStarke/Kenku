#!/bin/bash
#================

#SBATCH --partition gpu
#SBATCH --cpus-per-task 15
#SBATCH --mem-per-cpu 2G
#SBATCH --time 0-15:00:00
#SBATCH --nodes 1
#SBATCH --gpus-per-node=1
#SBATCH --array=1-35

#SBATCH --job-name="retrain_stragglers"
#SBATCH --output="/home3/s4984218/Kenku/habrok/jobscripts/results/hypertune_teacher/%j.out"

module purge

source ~/scripts/activate_venv.sh

echo "Shell Script Contents"
cat ~/Kenku/habrok/jobscripts/retrain_stragglers.sh

echo "Running script"
date +"%H:%M:%S"

cd ~/Kenku
srun python -m train.retrain_stragglers $SLURM_ARRAY_TASK_ID

echo "Finished training"