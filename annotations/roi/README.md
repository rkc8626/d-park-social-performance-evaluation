# Manual ROI polygons (lake / trees)

Annotate **once per camera** (`east`, `west`). Lake and trees are static in the frame.

## Reference frames (3840×2160)

| Camera | Image | Source video | Timestamp |
|--------|-------|--------------|-----------|
| east | `reference_frames/GX020055_east_ref.jpg` | `depotData/east/GX020055.MP4` | mid-frame (~14 min) |
| west | `reference_frames/GX020308_west_ref.jpg` | `depotData/west/GX020308.MP4` | mid-frame (~4.8 min) |

Open these in CVAT, Label Studio, Roboflow, or any image polygon tool.  
Coordinates are **pixel [x, y]** on the full 4K frame (origin top-left).

## JSON format (multiple trees)

Use one polygon per tree. Add or remove `tree_N` keys as needed.

```json
{
  "camera_id": "east",
  "video_id": "GX020055",
  "lake": [[x, y], [x, y], ...],
  "tree_1": [[x, y], ...],
  "tree_2": [[x, y], ...],
  "tree_3": [[x, y], ...]
}
```

Save completed files as:

- `annotations/roi/east.json`
- `annotations/roi/west.json`

(Your labels in `*.template.json` have been copied to `east.json` / `west.json`.)

**Label resolution:** If you annotated on a downscaled image (e.g. `2048x1152`), set `"resolution"` in JSON — the pipeline scales polygons to **3840×2160** video automatically (scale ×1.875).

**Label format** (CVAT-style wrapper is OK):

```json
"tree_1": [{ "label": "tree_1", "points": [[x,y], ...] }]
```

Apply to existing `tracks.csv`:

```bash
cd mvp
python apply_roi_to_outputs.py --output-dir ../outputs/mvp/GX020055 --roi ../annotations/roi/east.json
```

## Preview with ROIs drawn

```bash
cd depotData/d-park-social-performance-evaluation/mvp
python render_preview.py \
  --video ../../east/GX020055.MP4 \
  --tracks ../outputs/mvp/GX020055/tracks.csv \
  --roi ../annotations/roi/east.json
```

## Metrics (later)

`pct_near_tree` / `pct_near_lake` in `hourly_metrics.csv` will use these polygons (person centroid inside any `tree_*` or `lake` polygon).
