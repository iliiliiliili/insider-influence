#!/bin/bash -i

conda create -n iin python=3.10
conda activate iin

module load cuda/12.1

conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

pip install -r requirements.txt
