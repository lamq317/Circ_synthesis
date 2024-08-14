#!/bin/bash
#SBATCH --job-name=<job_name>
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t <time_to_run>
#SBATCH -n <number_of_nodes>
#SBATCH --gpus=<Number of GPUs, not nodes>
#SBATCH --output=<full_path_to_log_file>


date
uname -a

module load cudnn/8.9.3_cuda12
module load nccl/2.18.3-cu12
module load python
conda activate jax-demo



echo "starting BQSKit managers on all nodes"
srun run_workers_and_managers.sh <number_of_gpus_per_node> <number_of_workers_per_gpu> &
managers_pid=$!

managers_started_file=$SCRATCH/managers_${SLURM_JOB_ID}_started
n=<number_of_nodes>

while [[ ! -f "$managers_started_file" ]]
do
        sleep 0.5
done

while [ "$(cat "$managers_started_file" | wc -l)" -lt "$n" ]; do
    sleep 1
done

echo "starting BQSKit server on main node"
bqskit-server $(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ') &> $SCRATCH/bqskit_logs/server_${SLURM_JOB_ID}.log &
server_pid=$!

uname -a >> $SCRATCH/server_${SLURM_JOB_ID}_started

echo "will run  python  your command "

 python  <Your command>


date

echo "Killing the server"
kill -2 $server_pid

sleep 2

