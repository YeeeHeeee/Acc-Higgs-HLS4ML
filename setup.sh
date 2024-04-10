#!/usr/bin/env bash

export MAMBA_ROOT_PREFIX=${PWD}/env
eval "$(./env/micromamba shell hook -s posix)"
micromamba activate env

# Set pytorch to use just one cpu (for some reason, it's quicker)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

