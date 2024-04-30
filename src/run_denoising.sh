#!/bin/bash

# CONDA_PATH="/disk2/soham/miniconda3/envs/denoiser"
# source "$CONDA_PATH/etc/profile.d/conda.sh"
# conda activate denoiser

source activate denoiser

noisy_dir="/raid/soham.pendurkar/workspace/Voxceleb_data_annotation/data/noisy_wavs"
cleaned_wavs_dir="/raid/soham.pendurkar/workspace/Voxceleb_data_annotation/data/cleaned_wavs"

python3 -m denoiser.enhance \
            --dns48 \
            --num_workers 8 \
            --noisy_dir=$noisy_dir \
            --out_dir=$cleaned_wavs_dir \
            --batch_size 32