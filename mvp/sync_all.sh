#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(cd .. && pwd)"
PY="${ROOT}/.venv/bin/python"
[[ -x "$PY" ]] || PY=python3

module load ffmpeg/4.3.1 2>/dev/null || true

echo "=== Regenerate tables ==="
"$PY" regenerate_tables.py \
  --output-dir "${ROOT}/outputs/mvp/GX020055" \
  --roi "${ROOT}/annotations/roi/east.json" \
  --video-id GX020055

"$PY" regenerate_tables.py \
  --output-dir "${ROOT}/outputs/mvp/GX020308" \
  --roi "${ROOT}/annotations/roi/west.json" \
  --video-id GX020308

echo "=== Render previews (2 fps, 50% scale, libx264) ==="
"$PY" render_preview.py \
  --video "${ROOT}/../../west/GX020308.MP4" \
  --tracks "${ROOT}/outputs/mvp/GX020308/tracks.csv" \
  --roi "${ROOT}/annotations/roi/west.json" \
  --output-fps 2 --scale 0.5

"$PY" render_preview.py \
  --video "${ROOT}/../../east/GX020055.MP4" \
  --tracks "${ROOT}/outputs/mvp/GX020055/tracks.csv" \
  --roi "${ROOT}/annotations/roi/east.json" \
  --output-fps 2 --scale 0.5

echo "Done."
