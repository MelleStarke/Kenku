#!/bin/bash

module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load scikit-learn/1.3.1-gfbf-2023a 
module load matplotlib/3.7.2-gfbf-2023a
module load librosa/0.10.1-foss-2023a

python -m venv ~/venvs/py3.11.3
source ~/venvs/py3.11.3/bin/activate

pip install --upgrade pip wheel
pip install -r ~/Kenku/habrok/requirements.txt
