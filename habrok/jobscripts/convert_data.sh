#!/bin/bash

#SBATCH --time=08:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16GB
#SBATCH --partition=regular

#SBATCH --job-name="convert_data"
#SBATCH --output="/home3/s4984218/Kenku/habrok/jobscripts/results/convert_data_%j.out"

module purge
module load PyTorch-bundle/2.1.2-foss-2023a
module load scikit-learn/1.3.1-gfbf-2023a

echo "Finished loading modules"

source ~/venvs/py3.11.5/bin/activate

echo "Activated venv"

echo "Running scripts"

cd ~/Kenku
srun python -m data.convert_audio ~/scratch/raw/wav48 ~/scratch/processed/melspec --calc-norm --apply-norm --trim-silence --top-db 20

echo "Finished audio conversion"

srun python -m data.clean_transcripts ~/scratch/raw/txt ~/scratch/processed/transcript

echo "Finished transcript cleaning"

deactivate