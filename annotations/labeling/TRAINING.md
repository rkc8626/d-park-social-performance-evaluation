# Train & infer clip classifiers (HiPerGator)

Uses **~188 valid** labeled clips (east + west batches) to train ResNet18 heads for **activity** (10 classes) and **age** (child/adult), then labels all person tracks.

## One-shot (train + infer + tables)

```bash
cd depotData/d-park-social-performance-evaluation/mvp
sbatch run_train_then_infer.slurm
```

Logs: `outputs/mvp/slurm-cls-all-<jobid>.out`

## Step by step

```bash
cd mvp

# 1) Train (GPU, ~30–60 min)
sbatch train_classifier.slurm

# 2) After train finishes — infer + regenerate tables (~1–3 h both videos)
sbatch infer_classifier.slurm        # both videos
# sbatch infer_classifier.slurm GX020055
# sbatch infer_classifier.slurm GX020308
```

## What gets created

| Path | Content |
|------|---------|
| `annotations/labeling/dataset_index.csv` | train/val split |
| `mvp/models/activity_resnet18.pt` | activity classifier |
| `mvp/models/age_resnet18.pt` | age classifier |
| `mvp/models/classifier_labels.json` | class lists + val acc |
| `outputs/mvp/*/tracks.csv` | auto labels on non-manual tracks |
| `outputs/mvp/*/hourly_metrics.csv` | updated metrics |

## Behavior

- **Training:** 5 JPEG crops per clip, average logits across frames.
- **Inference:** sample 5 timestamps per person track from video; skip track IDs you already labeled manually (`valid` in manifest).
- **Low confidence** (&lt; 0.35): leave existing rule-based label unchanged.
- Tune `mvp/classifier_config.yaml` (`epochs`, `infer_confidence_min`, etc.).

## Check job

```bash
squeue -u $USER
tail -f ../outputs/mvp/slurm-train-cls-*.out
```

## After infer

Review val accuracy in `mvp/models/classifier_labels.json`. Spot-check `tracks.csv` and optional preview render.

## Batch: 5 east + 5 west videos

Training weights + same ROI per camera (`east.json` / `west.json`):

```bash
cd mvp
# Edit video list: videos_batch.yaml
sbatch run_batch.slurm              # all 10
sbatch run_batch.slurm GX020308     # one video
```

Per video: YOLO+ByteTrack → ROI → classifier → `tracks.csv` / `hourly_metrics.csv` under `outputs/mvp/<video_id>/`.
