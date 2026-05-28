#!/usr/bin/env bash
# Regenerate tables + preview for both completed MVP videos.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(cd .. && pwd)"
PY="${ROOT}/.venv/bin/python"
[[ -x "$PY" ]] || PY=python3

module load ffmpeg/4.3.1 2>/dev/null || true

echo "=== Regenerate tables (ROI + events + hourly) ==="
"$PY" regenerate_tables.py \
  --output-dir "${ROOT}/outputs/mvp/GX020055" \
  --roi "${ROOT}/annotations/roi/east.json" \
  --video-id GX020055

"$PY" regenerate_tables.py \
  --output-dir "${ROOT}/outputs/mvp/GX020308" \
  --roi "${ROOT}/annotations/roi/west.json" \
  --video-id GX020308

echo "=== Render previews (2 fps, half res, libx264) ==="
"$PY" render_preview.py \
  --video "${ROOT%/d-park*}/../west/GX020308.MP4" \
  --tracks "${ROOT}/outputs/mvp/GX020308/tracks.csv" \
  --roi "${ROOT}/annotations/roi/west.json" \
  --output-fps 2 --scale 0.5

# fix west path
WEST_VIDEO="$(cd "${ROOT}/../../west" 2>/dev/null && pwd)/GX020308.MP4" || true
EAST_VIDEO="$(cd "${ROOT}/../../east" 2>/dev/null && pwd)/GX020055.MP4" || true

if [[ -f "${ROOT}/../../west/GX020308.MP4" ]]; then
  "$PY" render_preview.py \
    --video "${ROOT}/../../west/GX020308.MP4" \
    --tracks "${ROOT}/outputs/mvp/GX020308/tracks.csv" \
    --roi "${ROOT}/annotations/roi/west.json" \
    --output-fps 2 --scale 0.5
fi

if [[ -f "${ROOT}/../../east/GX020055.MP4" ]]; then
  "$PY" render_preview.py \
    --video "${ROOT}/../../east/GX020055.MP4" \
    --tracks "${ROOT}/outputs/mvp/GX020055/tracks.csv" \
    --roi "${ROOT}/annotations/roi/east.json" \
    --output-fps 2 --scale 0.5
fi

echo "Done."
