#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J comp5qub10worker_relaxed
#SBATCH --output=comp5qub1worker.out
#SBATCH -t 23:50:00
#SBATCH -A m4370

#OpenMP settings:
#export OMP_NUM_THREADS=1
#export OMP_PLACES=threads
#export OMP_PROC_BIND=spread

#run the application:
#applications may perform better with --gpu-bind=none instead of --gpu-bind=single:1 
#srun -n 10 -c 12 --cpu_bind=cores -G 4 --gpu-bind=single:1  

module load python/3.11
#module load cudnn/8.9.3_cuda12
#module load nccl/2.18.3-cu12
#module load cudnn/9.1.0
#module load nccl/2.21.5

#XLA_PYTHON_CLIENT_PREALLOCATE=False 
#XLA_PYTHON_CLIENT_ALLOCATOR=platform

#nvidia-cuda-mps-control -d


#python compare_qfactor_sample_to_qfactor.py --input_qasm adder63_10q_block_28.qasm 
python samp_qf_synt_test.py qsearch 10
#shifter --module=gpu --image=ghcr.io/nvidia/jax:jax python3 samp_qf_synt_test.py qsearch 10

#echo quit | nvidia-cuda-mps-control
