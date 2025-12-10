#!/bin/bash 
#SBATCH --job-name=multinode-training
#SBATCH --nodes=4
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:8
#SBATCH -o training.out
#SBATCH -e training.err
#SBATCH --ntasks=4
nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
node_id=${SLURM_NODEID}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | cut -d" " -f2)
echo Master Node IP: $head_node_ip
export LOGLEVEL=INFO
export NNODES=4
export NPROC_PER_NODE=8
export MASTER_ADDR=$head_node_ip
export MASTER_PORT=29500
export NODE_RANK=$node_id
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=^lo,docker0
export NCCL_TIMEOUT=180000000
export NCCL_DEBUG=INFO
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export FORCE_TORCHRUN=1   
source /mnt/jail/root/miniconda3/bin/activate
cd /mnt/jail/root/LLaMA-Factory
srun llamafactory-cli train examples/train.yaml