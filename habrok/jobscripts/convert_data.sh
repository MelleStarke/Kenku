#!/bin/bash

#SBATCH --time=00:20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8GB
#SBATCH --partition=regular

#SBATCH --job-name="convert_data"
#SBATCH --output="./results/convert_data_%j.out"

module purge
module load Python/3.11.3-GCCcore-12.3.0
module load scikit-learn/1.3.1-gfbf-2023a

source ~/venvs/py3.11.5/bin/activate

cd $TMPDIR
mkdir ./raw
mkdir ./processed

cp ~/Data/raw/VCTK-Corpus/* ./raw

ls -a

srun python -m data.convert_audio --src ./raw/wav48 --dst ./processed/melspec --conf ./processed/data_config.json --calc-norm --apply-norm --trim-silence --top-db 20
srun python -m data.clean_transcripts


mv ./processed ~/scratch

deactivate