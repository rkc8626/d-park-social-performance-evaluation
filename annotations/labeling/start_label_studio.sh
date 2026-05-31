#!/usr/bin/env bash
# Start Label Studio for one batch.
# Usage: ./start_label_studio.sh GX020055_batch01 [port]
set -euo pipefail

BATCH="${1:?batch folder name, e.g. GX020055_batch01}"
PORT="${2:-8080}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BATCH_DIR="$(cd "$(dirname "$0")/clips/${BATCH}" && pwd)"

if [[ ! -f "${BATCH_DIR}/label_studio_import.json" ]]; then
  echo "Run first: python mvp/label_studio_prepare.py --manifest ${BATCH_DIR}/manifest.csv"
  exit 1
fi

# shellcheck source=/dev/null
source "${ROOT}/setup_env.sh"
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="${BATCH_DIR}"

echo "Batch: ${BATCH_DIR}"
echo "Open http://localhost:${PORT} (SSH -L ${PORT}:localhost:${PORT} login9)"
echo "Import: ${BATCH_DIR}/label_studio_import.json"
echo "Template: ${ROOT}/annotations/labeling/label_studio.xml"

label-studio start --port "${PORT}" --no-browser
