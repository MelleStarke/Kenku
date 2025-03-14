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

module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load scikit-learn/1.3.1-gfbf-2023a

source ~/venvs/py3.11.5/bin/activate


cd $TMPDIR

echo "$PWD" >> log.txt
echo $'\r' >> log.txt

ls -a >> log.txt
echo $'\r' >> log.txt

ls -a ~ >> log.txt
echo $'\r' >> log.txt

ls -a $TMPDIR >> log.txt
echo $'\r' >> log.txt


date >> log.txt
echo "Starting to move Kenku" >> log.txt
echo $'\r' >> log.txt

echo "This is only in the terminal output"

python --version >> log.txt

echo "import torch;import os;print(torch.cuda.is_available());print(torch.cuda.device_count());print(os.cpu_count())" > test.py

srun python test.py

mv ./* ~/temp/

deactivate
