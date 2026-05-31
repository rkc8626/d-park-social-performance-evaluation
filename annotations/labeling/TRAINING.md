# From manual labels to auto-labeling

## Flow

```
export_track_clips.py  →  you fill manifest.csv  →  merge_manual_labels.py (optional audit)
                              ↓
                    train on crops/clips (your ML step)
                              ↓
                    label_tracks_infer.py (future)  →  full tracks.csv
                              ↓
                    regenerate_tables.py
```

## Training data format

Each `manifest.csv` row with non-empty labels + `clips/<clip_id>/frame_*.jpg` is one example:

- **Activity model:** input = 5 crops (or MP4), target = `activity_label`
- **Age model:** same crops, target = `apparent_age_group`

Suggested split: set `split` column to `train` / `val` / `test` in manifest before training.

## Baseline models (not bundled in MVP)

| Task | Simple baseline | Stronger |
|------|-----------------|----------|
| Activity | sklearn on motion stats + CNN embedding | MMAction2 / VideoMAE fine-tune |
| Age | EfficientNet-B0 on person crops | Same, 2-class head |

Minimum useful set: **~200 labeled clips** (mix east/west, near/far, group/solo).

## After training

A future `label_tracks_infer.py` will:

1. Load `tracks.csv` and video
2. Run the trained models on every person track (sliding windows)
3. Write `activity_label` and `apparent_age_group` on all frames
4. You run `regenerate_tables.py` for `apparent_child_pct` and activity diversity

Until that script exists, use `merge_manual_labels.py` for labeled rows only.
