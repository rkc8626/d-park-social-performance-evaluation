# Per-video ROI (when the frame shifts)

One `east.json` / `west.json` is a **default**. If lake/trees do not line up on a clip, adjust ROI for that video only.

## 1. Interactive editor (local laptop)

Needs a display (OpenCV window). Sync one MP4 to your machine, or work on a machine with GUI.

```bash
cd depotData/d-park-social-performance-evaluation

# Example: west clip where lake polygon is shifted
.venv/bin/python mvp/roi_editor.py \
  --video ../depotData/west/GX020307.MP4 \
  --camera west \
  --timestamp 300
```

**Controls**

| Key | Action |
|-----|--------|
| Drag | Move nearest vertex on current layer |
| Arrow keys | Nudge current layer (lake or one tree) |
| `a` `d` `w` `x` | Nudge **all** lake+tree layers together (camera drift) |
| `n` / `p` | Next / previous layer |
| `r` | Reset from camera default |
| `s` | Save to `annotations/roi/per_video/<video_id>.json` |
| `q` | Quit |

Pick `--timestamp` where lake/trees are easy to see (seconds into the clip).

## 2. Sync ROI back to the server

```bash
rsync -avz annotations/roi/per_video/ \
  you@login:/orange/ufdatastudios/chenz1/depotData/d-park-social-performance-evaluation/annotations/roi/per_video/
```

## 3. Recalculate metrics (no GPU, fast)

On login or locally:

```bash
cd depotData/d-park-social-performance-evaluation/mvp

# One video
python reapply_roi_batch.py --video-id GX020307

# All videos (uses per_video JSON when present)
python reapply_roi_batch.py
```

Updates `tracks.csv` zone columns and `hourly_metrics.csv` (`pct_near_tree`, `pct_near_lake`, distances).

## 4. Optional: refresh demo overlay

```bash
python render_preview.py \
  --video ../../west/GX020307.MP4 \
  --tracks ../outputs/mvp/GX020307/tracks.csv \
  --roi ../annotations/roi/per_video/GX020307.json \
  --max-duration 10 --scale 0.222 \
  --output ../outputs/mvp/GX020307/demo_10s_480p.mp4
```

(`--scale 0.222` ≈ 480p height from 4K.)

## Resolution

Per-video files store polygons in **that frame's pixel space** (`resolution: 3840x2160`). The pipeline scales automatically via `roi_utils.load_roi_polygons`.
