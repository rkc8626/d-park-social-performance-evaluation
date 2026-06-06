#!/bin/bash
#SBATCH --job-name=park-train-cls
#SBATCH --partition=hpg-turin
#SBATCH --output=../outputs/mvp/slurm-train-cls-%j.out
#SBATCH --error=../outputs/mvp/slurm-train-cls-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1

set -euo pipefail
MVP_DIR="${SLURM_SUBMIT_DIR:?submit from mvp/}"
cd "$MVP_DIR"
ROOT="$(cd .. && pwd)"

module load ffmpeg/4.3.1 cuda/12.8.1 pytorch/2.8.0

# shellcheck source=setup_hpg_env.sh
source "${MVP_DIR}/setup_hpg_env.sh"
setup_hpg_env "$MVP_DIR"
mkdir -p "${MVP_DIR}/models"
pip install --cache-dir "${MVP_DIR}/.cache/pip" -t "${MVP_DIR}/.job_pkgs" \
  pandas pyyaml pillow opencv-python-headless 2>/dev/null || true

python -c "import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

python build_clip_dataset_index.py --config classifier_config.yaml
python train_clip_classifier.py --config classifier_config.yaml --device cuda

echo "Models in ${MVP_DIR}/models/"
