#!/bin/bash
#SBATCH --job-name=p04-extract-__SHORT__
#SBATCH --output=/scratch/USER/results/04-activations/extract___SHORT__.log
#SBATCH --error=/scratch/USER/results/04-activations/extract___SHORT__.err
#SBATCH --partition=sporc
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=48G

# Env-fix stack (verbatim from notes/cluster-ops/README.md)
export PYTHONNOUSERSITE=1
export HF_HOME=/scratch/USER/hf
export TOKENIZERS_PARALLELISM=false
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDAHOSTCXX=/usr/bin/g++
export CUDA_HOME=/home/USER/miniforge3/envs/jrp/lib/python3.11/site-packages/nvidia/cu13
export PATH=/home/USER/miniforge3/envs/jrp/bin:$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=/home/USER/miniforge3/envs/jrp/lib
export VLLM_USE_FLASHINFER_SAMPLER=0
export FLASHINFER_WORKSPACE_BASE=/scratch/USER/flashinfer_cache
export TRITON_CACHE_DIR=/scratch/USER/triton_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p /scratch/USER/results/04-activations

cd /home/USER/jollen-rapid-project/projects/04-deception-probes

echo "Extracting activations for __MODEL__ at $(date)"

PYTHONPATH=. /home/USER/miniforge3/envs/jrp/bin/python -m p04.extract_activations \
    --model __MODEL__ \
    --out-dir /scratch/USER/results/04-activations

echo "Completed at $(date)"
