#!/bin/sh

# batch setup
sed -i "s,BASE_DIR,'${PWD}/batch',g" batch_config.py
mkdir -p batch

# environment install
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
rm -r env
mv bin env
export MAMBA_ROOT_PREFIX=${PWD}/env
eval "$(./env/micromamba shell hook -s posix)"

micromamba activate
micromamba create -f env.yml

