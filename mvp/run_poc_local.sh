#!/usr/bin/env bash
# Run POC on a login node (CPU). For full clip / GPU, use: sbatch run_poc.slurm
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$(dirname "$0")"

module load python/3.10 ffmpeg/4.3.1 2>/dev/null || true

export YOLO_CONFIG_DIR="${ROOT}/mvp/.cache/ultralytics"
export TORCH_HOME="${ROOT}/mvp/.cache/torch"
export XDG_CACHE_HOME="${ROOT}/mvp/.cache/xdg"
export PIP_CACHE_DIR="${ROOT}/mvp/.cache/pip"
mkdir -p "$YOLO_CONFIG_DIR" "$TORCH_HOME" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  source "${ROOT}/.venv/bin/activate"
fi

python run_poc.py --config config.yaml "$@"
