#!/usr/bin/env bash
set -euo pipefail

run_setup() {
  if ! command -v uv >/dev/null 2>&1; then
    echo "uv is not installed. Install uv first: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
  fi

  VENV_DIR=".venv"
  PYTHON_BIN="${VENV_DIR}/bin/python"

  uv venv "${VENV_DIR}"

  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"

  uv pip install --upgrade pip setuptools wheel

  # Optional override for CUDA-specific wheels, example:
  # export TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
  if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
    uv pip install --index-url "${TORCH_INDEX_URL}" torch
  else
    uv pip install torch
  fi

  "${PYTHON_BIN}" -c 'import torch; print("torch", torch.__version__); print("cuda", torch.cuda.is_available())'

  echo "Environment ready at ${VENV_DIR} on host $(hostname)"
}

if [[ "${1:-}" == "--on-worker" ]]; then
  run_setup
  exit 0
fi

if ! command -v srun >/dev/null 2>&1; then
  echo "srun is not available. Run this script from a Slurm login node." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Allocating worker node and setting up environment with uv..."
srun --pty -N1 --ntasks=1 --gres=gpu:1 bash -lc "cd '${SCRIPT_DIR}' && bash setup_env_uv.sh --on-worker"
exit 0
