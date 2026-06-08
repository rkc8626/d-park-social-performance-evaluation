#!/bin/bash
#SBATCH --job-name=park-batch
#SBATCH --account=ufdatastudios
#SBATCH --partition=hpg-b200
#SBATCH --output=../outputs/mvp/slurm-batch-%j.out
#SBATCH --error=../outputs/mvp/slurm-batch-%j.err
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:b200:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=chenz1@ufl.edu

# All depotData/east + west videos (regenerate list first).
# Submit: cd mvp && sbatch run_batch.slurm
# One video: sbatch run_batch.slurm GX020308
# Force redo one: python run_video_pipeline.py --video-id GX020308 --device cuda

set -euo pipefail
MVP_DIR="${SLURM_SUBMIT_DIR:?submit from mvp/}"
cd "$MVP_DIR"

module load ffmpeg/4.3.1 cuda/12.8.1 pytorch/2.8.0
# shellcheck source=setup_hpg_env.sh
source "${MVP_DIR}/setup_hpg_env.sh"
setup_hpg_env "$MVP_DIR"

if [[ ! -f "${MVP_DIR}/models/activity_resnet18.pt" ]]; then
  echo "ERROR: Train first: sbatch train_classifier.slurm" >&2
  exit 1
fi

python generate_videos_batch.py --sync-classifier

python -c "
import torch, torchvision
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('torchvision', torchvision.__version__)
"

VID="${1:-}"
if [[ -n "$VID" ]]; then
  python run_video_pipeline.py --video-id "$VID" --device cuda
else
  python run_video_pipeline.py --device cuda --skip-existing --continue-on-error
fi

echo "Batch complete."
