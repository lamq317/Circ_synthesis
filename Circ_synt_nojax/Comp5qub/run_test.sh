#!/bin/bash
#SBATCH --image=ghcr.io/nvidia/jax:jax
#SBATCH --nodes=1
#SBATCH --qos=debug
#SBATCH --constraint=gpu
#SBATCH -t 00:30:00
#SBATCH -A m4370

module load python/3.11
conda activate jax-demo
module load cudnn/8.9.3_cuda12
module load nccl/2.21.5


XLA_PYTHON_CLIENT_PREALLOCATE=False
XLA_PYTHON_CLIENT_ALLOCATOR=platform

nvidia-cuda-mps-control -d


srun shifter --module=gpu --image=ghcr.io/nvidia/jax:jax python3 samp_qf_synt_test.py 3

echo quit | nvidia-cuda-mps-control
