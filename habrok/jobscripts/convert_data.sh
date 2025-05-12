#!/bin/bash

#SBATCH --time=8:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10GB
#SBATCH --partition=regular

#SBATCH --job-name="convert_data"
#SBATCH --output="/home3/s4984218/Kenku/habrok/jobscripts/results/convert_data_%j.out"

module purge

source ~/scripts/activate_venv.sh

echo "Script Contents"

cat ~/Kenku/habrok/jobscripts/train_teacher.sh

echo "Running script"
date +"%H:%M:%S"

cd ~/Kenku
srun python -m data.convert_audio ~/scratch/raw/wav48 ~/scratch/processed/melspec --calc-norm --apply-norm

echo "Finished audio conversion"

# srun python -m data.clean_transcripts ~/scratch/raw/txt ~/scratch/processed/transcript

echo "Finished transcript cleaning"
