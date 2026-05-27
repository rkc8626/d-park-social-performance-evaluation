# MVP / proof-of-concept pipeline

Off-the-shelf **YOLOv8n + ByteTrack** on a short clip window to answer: *can we detect and track people and bikes in these park videos before building the full pipeline?*

This implements a subset of the parent [README](../README.md) (Phases 5–6 style outputs only).

## Quick start

```bash
cd depotData/d-park-social-performance-evaluation/mvp
./run_poc_local.sh
```

Override video or window:

```bash
./run_poc_local.sh --video ../../west/GX020308.MP4 --max-seconds 60
```

GPU (recommended for 4K):

```bash
sbatch run_poc.slurm
```

## Outputs

Written to `../outputs/mvp/<video_id>/`:

| File | Purpose |
|------|---------|
| `video_quality_report.json` | Resolution, detection rate, usability flag |
| `tracks.csv` | Per-frame detections + rule-based activity |
| `hourly_metrics.csv` | Aggregates for the processed window |
| `poc_summary.md` | Human-readable one-page summary |
| `preview_annotated.mp4` | Short annotated clip for visual QA |

## What is intentionally skipped

- ROI polygons (lake, trees, park boundary)
- Nature exposure and dwell time inside park zones
- Child-age proxy classifier
- Manual validation / mAP

## Environment note (UF HPC)

`/home/chenz1` may be full. Caches are directed to `mvp/.cache/` on Orange storage via `run_poc_local.sh` and `run_poc.py`.
