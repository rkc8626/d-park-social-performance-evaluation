#!/bin/bash
#SBATCH --job-name=park-sync
#SBATCH --partition=hpg-default
#SBATCH --output=../outputs/mvp/slurm-sync-%j.out
#SBATCH --error=../outputs/mvp/slurm-sync-%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from mvp/}"
bash sync_all.sh
