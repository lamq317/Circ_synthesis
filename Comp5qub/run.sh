#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 4
#SBATCH -q regular
#SBATCH -J comp5qub10worker_relaxed
#SBATCH --output=comp5qub1worker.out
#SBATCH -t 23:50:00
#SBATCH -A m4370


module load python/3.11
module load cudnn/8.9.3_cuda12
#module load nccl/2.18.3-cu12
#module load cudnn/9.1.0
module load nccl/2.21.5

XLA_PYTHON_CLIENT_PREALLOCATE=False 
XLA_PYTHON_CLIENT_ALLOCATOR=platform

nvidia-cuda-mps-control -d


#python compare_qfactor_sample_to_qfactor.py --input_qasm adder63_10q_block_28.qasm 
#python samp_qf_synt_test.py qsearch 4
#shifter --module=gpu --image=ghcr.io/nvidia/jax:jax python3 samp_qf_synt_test.py qsearch 10

python3 samp_qf_synt_test.py qsearch 10

echo quit | nvidia-cuda-mps-control
