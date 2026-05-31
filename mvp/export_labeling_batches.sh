#!/usr/bin/env bash
# Export labeling clips for both MVP videos.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source setup_env.sh 2>/dev/null || true
module load ffmpeg/4.3.1 2>/dev/null || true

PY="${ROOT}/.venv/bin/python"
N="${1:-150}"

"$PY" mvp/export_track_clips.py --tracks outputs/mvp/GX020308/tracks.csv -n "$N"
"$PY" mvp/export_track_clips.py --tracks outputs/mvp/GX020055/tracks.csv -n "$N" --batch-id GX020055_batch01

echo "Done. Edit manifests under annotations/labeling/clips/"
