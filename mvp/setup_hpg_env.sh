#!/bin/bash
# Source from SLURM scripts after: module load ffmpeg cuda pytorch
# Uses cluster CUDA torch/torchvision; job_pkgs only for ultralytics + small deps.

setup_hpg_env() {
  local mvp_dir="${1:?mvp dir}"
  export YOLO_CONFIG_DIR="${mvp_dir}/.cache/ultralytics"
  export TORCH_HOME="${mvp_dir}/.cache/torch"
  export XDG_CACHE_HOME="${mvp_dir}/.cache/xdg"
  export PIP_CACHE_DIR="${mvp_dir}/.cache/pip"
  mkdir -p "$YOLO_CONFIG_DIR" "$TORCH_HOME" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR"

  local job_pkgs="${mvp_dir}/.job_pkgs"
  if [[ ! -f "${job_pkgs}/ultralytics/__init__.py" ]]; then
    pip install --cache-dir "$PIP_CACHE_DIR" -t "$job_pkgs" --no-deps ultralytics
    pip install --cache-dir "$PIP_CACHE_DIR" -t "$job_pkgs" \
      opencv-python-headless pandas pyyaml lap scipy pillow psutil requests tqdm matplotlib
  fi
  # Never shadow cluster torch/torchvision (causes GLIBCXX / CUDA errors).
  rm -rf "${job_pkgs}/torch" "${job_pkgs}/torchvision" "${job_pkgs}/torchgen"
  # Cluster pytorch/2.8.0 (torch + torchvision) must stay first in sys.path.
  export PYTHONPATH="${PYTHONPATH:-}:${mvp_dir}:${job_pkgs}"
}
