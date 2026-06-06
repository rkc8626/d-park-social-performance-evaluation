#!/bin/bash
#SBATCH --job-name=park-infer-cls
#SBATCH --partition=hpg-turin
#SBATCH --output=../outputs/mvp/slurm-infer-cls-%j.out
#SBATCH --error=../outputs/mvp/slurm-infer-cls-%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1

set -euo pipefail
MVP_DIR="${SLURM_SUBMIT_DIR:?submit from mvp/}"
cd "$MVP_DIR"

module load ffmpeg/4.3.1 cuda/12.8.1 pytorch/2.8.0
# shellcheck source=setup_hpg_env.sh
source "${MVP_DIR}/setup_hpg_env.sh"
setup_hpg_env "$MVP_DIR"

python run_video_pipeline.py --device cuda --skip-poc "$@"
