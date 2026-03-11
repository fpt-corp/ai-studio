#!/bin/bash
#SBATCH --job-name=pt-2node
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MASTER_NODE="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
MASTER_ADDR="$(
  srun --nodes=1 --ntasks=1 -w "${MASTER_NODE}" hostname --ip-address \
    | awk '{for(i=1;i<=NF;i++) if ($i !~ /^127\\./) {print $i; exit}}'
)"
if [[ -z "${MASTER_ADDR}" ]]; then
  MASTER_ADDR="${MASTER_NODE}"
fi
MASTER_PORT=29500
NNODES="${SLURM_NNODES}"
NPROC_PER_NODE=2

echo "MASTER_NODE=${MASTER_NODE} MASTER_ADDR=${MASTER_ADDR} NNODES=${NNODES} NPROC_PER_NODE=${NPROC_PER_NODE} SCRIPT_DIR=${SCRIPT_DIR}"

srun bash -c '
  echo "task_procid=$SLURM_PROCID nodeid=$SLURM_NODEID host=$(hostname)"
  /root/ai-studio/demo/gtc-2026/demo3/.venv/bin/torchrun \
    --nnodes='"${NNODES}"' \
    --nproc-per-node='"${NPROC_PER_NODE}"' \
    --node-rank=$SLURM_PROCID \
    --master_addr='"${MASTER_ADDR}"' \
    --master_port='"${MASTER_PORT}"' \
    ddp_demo.py \
    --epochs 5 \
    --batch-size 128 \
    --max-train-samples 8000 \
    --max-test-samples 2000
'