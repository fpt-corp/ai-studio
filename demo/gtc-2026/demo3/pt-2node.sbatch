#!/bin/bash
#SBATCH --job-name=pt-2node
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
MASTER_PORT=29500
NNODES="${SLURM_NNODES}"
NPROC_PER_NODE=2

echo "MASTER_ADDR=${MASTER_ADDR} NNODES=${NNODES} NPROC_PER_NODE=${NPROC_PER_NODE} SCRIPT_DIR=${SCRIPT_DIR}"

srun --ntasks="${NNODES}" --ntasks-per-node=1 bash -c '
  cd "'"${SCRIPT_DIR}"'"
  ./.venv/bin/torchrun \
    --nnodes='"${NNODES}"' \
    --nproc-per-node='"${NPROC_PER_NODE}"' \
    --node-rank=$SLURM_NODEID \
    --master_addr='"${MASTER_ADDR}"' \
    --master_port='"${MASTER_PORT}"' \
    ddp_demo.py \
    --epochs 1 \
    --batch-size 64 \
    --max-train-samples 8000 \
    --max-test-samples 2000
'
