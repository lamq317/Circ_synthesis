#!/bin/bash

node_id=$(uname -n)
amount_of_gpus=$1
amount_of_workers_per_gpu=$2
total_amount_of_workers=$(($amount_of_gpus * $amount_of_workers_per_gpu))
manager_log_file="$SCRATCH/bqskit_logs/manager_${SLURM_JOB_ID}_${node_id}.log"
server_started_file="$SCRATCH/server_${SLURM_JOB_ID}_started"
managers_started_file="$SCRATCH/managers_${SLURM_JOB_ID}_started"

touch $managers_started_file

wait_for_outgoing_thread_in_manager_log() {
    while ! grep -q "Started outgoing thread." $manager_log_file; do
        sleep 1
    done
    uname -a >> $managers_started_file
}

start_mps_servers() {
    echo "Starting MPS servers on node $node_id with CUDA $CUDA_VISIBLE_DEVICES"
    nvidia-cuda-mps-control -d
}

wait_for_bqskit_server() {
    i=0
    while [[ ! -f $server_started_file && $i -lt 10 ]]; do
        sleep 1
        i=$((i+1))
    done
}

start_workers() {
    echo "Starting $total_amount_of_workers workers on $amount_of_gpus gpus"
    for (( gpu_id=0; gpu_id<$amount_of_gpus; gpu_id++ )); do
        XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$gpu_id bqskit-worker $amount_of_workers_per_gpu &> $SCRATCH/bqskit_logs/workers_${SLURM_JOB_ID}_${node_id}_${gpu_id}.log &
    done
    wait
}

stop_mps_servers() {
    echo "Stop MPS servers on node $node_id"
    echo quit | nvidia-cuda-mps-control
}

if [ $amount_of_gpus -eq 0 ]; then
    echo "Will run manager on node $node_id with n args of $amount_of_workers_per_gpu"
	bqskit-manager -n $amount_of_workers_per_gpu -v &> $manager_log_file
	echo "Manager finished on node $node_id"
else
    echo "Will run manager on node $node_id"
    bqskit-manager -x -n$total_amount_of_workers -vvv &> $manager_log_file &
    wait_for_outgoing_thread_in_manager_log
    start_mps_servers
    wait_for_bqskit_server
    start_workers
    echo "Manager and workers finished on node $node_id" >> $manager_log_file
    stop_mps_servers
fi
