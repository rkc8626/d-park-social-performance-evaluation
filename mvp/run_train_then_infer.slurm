#!/bin/bash
#SBATCH --job-name=park-cls-all
#SBATCH --partition=hpg-turin
#SBATCH --output=../outputs/mvp/slurm-cls-all-%j.out
#SBATCH --error=../outputs/mvp/slurm-cls-all-%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1

# Train on 210 clips, then infer east+west and regenerate tables.
# Submit: cd mvp && sbatch run_train_then_infer.slurm

set -euo pipefail
MVP_DIR="${SLURM_SUBMIT_DIR:?submit from mvp/}"
JOB_TRAIN="${MVP_DIR}/train_classifier.slurm"
JOB_INFER="${MVP_DIR}/infer_classifier.slurm"

cd "$MVP_DIR"
bash "$JOB_TRAIN"
bash "$JOB_INFER" both