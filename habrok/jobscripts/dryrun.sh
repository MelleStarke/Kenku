#!/bin/bash

#SBATCH --time=00:20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1

#SBATCH --job-name="dryrun"
#SBATCH --output="dryrun_%j.out"

echo "$PWD" >> log.txt
echo $'\r' >> log.txt

ls -a >> log.txt
echo $'\r' >> log.txt

ls -a ~ >> log.txt
echo $'\r' >> log.txt

ls -a $TMPDIR >> log.txt
echo $'\r' >> log.txt


echo date >> log.txt
echo "Starting to move Kenku" >> log.txt
echo $'\r' >> log.txt

mv ~/Kenku $TMPDIR/Kenku >> log.txt
echo $'\r' >> log.txt

echo date
echo "Finished moving Kenku"
echo $'\r' >> log.txt

python --version >> log.txt

echo "import torch;import os;print(torch.cuda.is_available());print(torch.cuda.device_count());print(os.cpu_count())" > test.py

srun python test.py

mv ./* ~/temp/