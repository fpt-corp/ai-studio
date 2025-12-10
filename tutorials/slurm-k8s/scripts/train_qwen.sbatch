#!/bin/bash 
#SBATCH --job-name=multinode-training
#SBATCH --nodes=4
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:8
#SBATCH -o training_%j.out
#SBATCH -e training_%j.err
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST))
nodes_array=($nodes)
head_node=${nodes_array[0]}

head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | cut -d" " -f2)
echo "Master Node: $head_node ($head_node_ip)"

export LOGLEVEL=INFO
export NNODES=$SLURM_NNODES # 4
export NPROC_PER_NODE=8
export MASTER_ADDR=$head_node_ip
export MASTER_PORT=29500

export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=^lo,docker0
export NCCL_DEBUG=INFO
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1

export FORCE_TORCHRUN=1   

srun bash -c '
    source /mnt/jail/root/miniconda3/bin/activate
    conda activate training
    
    echo "Node ID: $SLURM_NODEID - Hostname: $(hostname) - Master: $MASTER_ADDR"
    
    export NODE_RANK=$SLURM_NODEID
    
    cd /mnt/jail/root/LLaMA-Factory
    llamafactory-cli train examples/train.yaml
'