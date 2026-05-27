#!/bin/bash
#SBATCH --job-name=park-mvp-poc
#SBATCH --partition=hpg-turin
#SBATCH --output=../outputs/mvp/slurm-%j.out
#SBATCH --error=../outputs/mvp/slurm-%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

set -euo pipefail
MVP_DIR="${SLURM_SUBMIT_DIR:?submit from mvp/}"
cd "$MVP_DIR"
mkdir -p "${MVP_DIR}/../outputs/mvp"

module load ffmpeg/4.3.1 cuda/12.8.1 pytorch/2.8.0

export YOLO_CONFIG_DIR="${MVP_DIR}/.cache/ultralytics"
export TORCH_HOME="${MVP_DIR}/.cache/torch"
export XDG_CACHE_HOME="${MVP_DIR}/.cache/xdg"
export PIP_CACHE_DIR="${MVP_DIR}/.cache/pip"
mkdir -p "$YOLO_CONFIG_DIR" "$TORCH_HOME" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR"

# Ultralytics only — cluster module provides CUDA torch (prepend would shadow it).
JOB_PKGS="${MVP_DIR}/.job_pkgs"
if [[ ! -f "${JOB_PKGS}/ultralytics/__init__.py" ]]; then
  pip install --cache-dir "$PIP_CACHE_DIR" -t "$JOB_PKGS" --no-deps ultralytics 1>&2
  pip install --cache-dir "$PIP_CACHE_DIR" -t "$JOB_PKGS" \
    opencv-python-headless pandas pyyaml lap scipy pillow psutil requests tqdm matplotlib 1>&2
fi
export PYTHONPATH="${PYTHONPATH:-}:${JOB_PKGS}"

python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"

python run_poc.py --config config.yaml "$@"
