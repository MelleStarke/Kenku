#!/bin/bash

module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load scikit-learn/1.3.1-gfbf-2023a 
module load matplotlib/3.7.2-gfbf-2023a
module load librosa/0.10.1-foss-2023a
echo "Finished loading modules"

source ~/venvs/py3.11.3/bin/activate
echo "Activated venv"
