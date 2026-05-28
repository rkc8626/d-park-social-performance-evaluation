#!/bin/bash
#SBATCH --job-name=park-preview
#SBATCH --partition=hpg-default
#SBATCH --output=../outputs/mvp/slurm-preview-%j.out
#SBATCH --error=../outputs/mvp/slurm-preview-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
MVP_DIR="${SLURM_SUBMIT_DIR:?submit from mvp/}"
cd "$MVP_DIR"
mkdir -p "${MVP_DIR}/../outputs/mvp"

module load python/3.10 ffmpeg/4.3.1 2>/dev/null || true
if [[ -f ../.venv/bin/activate ]]; then
  source ../.venv/bin/activate
fi

# Usage: sbatch render_preview.slurm VIDEO TRACKS [ROI_JSON]
VIDEO="${1:?video path}"
TRACKS="${2:?tracks.csv path}"
ROI="${3:-}"

ARGS=(--video "$VIDEO" --tracks "$TRACKS" --output-fps 2 --scale 0.5)
[[ -n "$ROI" ]] && ARGS+=(--roi "$ROI")
python render_preview.py "${ARGS[@]}"
