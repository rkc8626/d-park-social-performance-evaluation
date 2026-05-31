# Manual activity & age labeling

## Quick start

```bash
cd depotData/d-park-social-performance-evaluation
source setup_env.sh
module load ffmpeg/4.3.1

# Export ~150 clips per video (JPEG crops + manifest)
python mvp/export_track_clips.py --tracks outputs/mvp/GX020308/tracks.csv
python mvp/export_track_clips.py --tracks outputs/mvp/GX020055/tracks.csv

# Optional short MP4 per clip (easier in CVAT)
python mvp/export_track_clips.py --tracks outputs/mvp/GX020308/tracks.csv --write-mp4
```

Output: `annotations/labeling/clips/<video_id>_batch01/`

| File | Purpose |
|------|---------|
| `manifest.csv` | **Fill `label_status` first**, then activity/age if `valid` |
| `label_queue.csv` | Same as manifest (for annotators) |
| `clips/<clip_id>/frame_01.jpg …` | 5 crops per track segment |
| `clips/<clip_id>.mp4` | Only if `--write-mp4` |

## How to label

1. Open `LABEL_GUIDE.md`.
2. For each row, open `clips/<clip_id>/` (5 JPEG crops).
3. Set **`label_status`**:
   - `not_a_person` — false YOLO box (common)
   - `skip` — cannot tell
   - `valid` — real person → also fill activity + age
4. Save `manifest.csv`.

**Not a person?** Use `not_a_person`, leave `activity_label` and `apparent_age_group` empty.

Allowed values:

- **label_status:** `valid`, `not_a_person`, `skip`
- **activity_label** (if valid): `walking`, `running`, `biking`, `sitting`, `standing`, `talking_socializing`, `playing`, `dog_walking`, `exercising`, `picnic_resting`
- **apparent_age_group** (if valid): `child`, `adult`

## Apply labels to tracks (before training models)

```bash
python mvp/merge_manual_labels.py \
  --tracks outputs/mvp/GX020308/tracks.csv \
  --manifest annotations/labeling/clips/GX020308_batch01/manifest.csv

cd mvp
python regenerate_tables.py \
  --output-dir ../outputs/mvp/GX020308 \
  --roi ../annotations/roi/west.json \
  --video-id GX020308
```

This updates `tracks.csv` / `hourly_metrics` for **labeled clips only**. It does not auto-label the full video.

## Auto-labeling later (trained models)

1. Use the same clips + filled `manifest.csv` as **training data**.
2. Train activity + age classifiers (see `annotations/labeling/TRAINING.md`).
3. Run inference on all tracks → then `regenerate_tables.py`.

Manual labels are the ground truth; models learn from them and then predict on new frames.

## Labeling without cluster localhost

**No SSH port forward?** → [`LOCAL_LABELING.md`](LOCAL_LABELING.md)  
Copy batch to laptop → `python mvp/local_label_ui.py --batch-dir .` → http://127.0.0.1:8765

## Label Studio (on cluster or laptop)

**Full guide:** [`LABEL_STUDIO.md`](LABEL_STUDIO.md)

```bash
python mvp/label_studio_prepare.py \
  --manifest annotations/labeling/clips/GX020055_batch01/manifest.csv

./annotations/labeling/start_label_studio.sh GX020055_batch01 8080
# On laptop: ssh -L 8080:localhost:8080 chenz1@login9.rc.ufl.edu
# Browser: http://localhost:8080
```

## CVAT

Use `--write-mp4` and upload `clips/*.mp4` as a task list, or upload full video and use `manifest.csv` track_id / times as guide.
