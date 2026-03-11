#!/bin/bash
#SBATCH --job-name=pt-2node
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%j.out

set -euo pipefail

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)

./.venv/bin/torchrun \
  --nnodes=2 \
  --nproc-per-node=1 \
  --node-rank="$SLURM_NODEID" \
  --master_addr="$MASTER_ADDR" \
  --master_port=29500 \
  ddp_demo.py
