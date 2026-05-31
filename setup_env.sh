#!/usr/bin/env bash
# Source from project root on HiPerGator (login or compute):
#   cd depotData/d-park-social-performance-evaluation
#   source setup_env.sh
#
# First-time install:
#   source setup_env.sh && install_mvp_deps

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE="${ROOT}/mvp/.cache"

module load python/3.10 ffmpeg/4.3.1 2>/dev/null || true

export YOLO_CONFIG_DIR="${CACHE}/ultralytics"
export TORCH_HOME="${CACHE}/torch"
export XDG_CACHE_HOME="${CACHE}/xdg"
export XDG_CONFIG_HOME="${CACHE}/xdg-config"
export PIP_CACHE_DIR="${CACHE}/pip"
mkdir -p "$YOLO_CONFIG_DIR" "$TORCH_HOME" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$PIP_CACHE_DIR"

if [[ ! -d "${ROOT}/.venv" ]]; then
  echo "Creating .venv at ${ROOT}/.venv ..."
  python3 -m venv "${ROOT}/.venv"
fi

# shellcheck source=/dev/null
source "${ROOT}/.venv/bin/activate"

install_mvp_deps() {
  pip install --upgrade pip
  pip install -r "${ROOT}/requirements-mvp.txt"
  python -c "
import cv2, pandas, numpy, yaml, lap, matplotlib
from ultralytics import YOLO
import torch
print('MVP env OK')
print('  python:', __import__('sys').version.split()[0])
print('  opencv:', cv2.__version__)
print('  ultralytics:', __import__('ultralytics').__version__)
print('  torch:', torch.__version__, '| cuda available:', torch.cuda.is_available())
"
}

echo "Activated: ${ROOT}/.venv ($(python -V 2>&1))"
echo "Cache/config under: ${CACHE}"
