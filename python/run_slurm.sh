#!/bin/bash
#SBATCH --job-name=test
#SBATCH --cpus-per-task=128
#SBATCH --nodes=2
#SBATCH --tasks-per-node=1
#SBATCH --partition=all
module load cuda/cuda-11.0
source ~/venv/bin/activate
let "worker_num=(${SLURM_NTASKS} - 1)"
let "total_cores=${SLURM_NTASKS} * ${SLURM_CPUS_PER_TASK}"
suffix='6379'
ip_head=$1:$suffix
export ip_head # Exporting for latter access by trainer.py
ulimit -n 65536
srun --nodes=1 --ntasks=1 --cpus-per-task=${SLURM_CPUS_PER_TASK} --nodelist=$1 --export=ALL,NCCL_SOCKET_IFNAME=ib0 ray start --head --block --dashboard-host 0.0.0.0 --port=6379 --num-cpus ${SLURM_CPUS_PER_TASK} &

sleep 3

srun --nodes=${worker_num} --ntasks=${worker_num} --cpus-per-task=${SLURM_CPUS_PER_TASK} --exclude=$1 --export=ALL,NCCL_SOCKET_IFNAME=ib0 ray start --address $ip_head --block --num-cpus ${SLURM_CPUS_PER_TASK} &

sleep 3
python3 -u ray_train.py --config=$2 --name=$3 --cluster $4 $5

