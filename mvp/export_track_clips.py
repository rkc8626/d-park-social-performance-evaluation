#!/usr/bin/env python3
"""Export person track clips for manual activity + age labeling.

Writes:
  <export_root>/<batch_id>/
    manifest.csv          — one row per clip (fill activity_label, apparent_age_group)
    clips/<clip_id>/      — frame_01.jpg … frame_N.jpg (person crops)
    clips/<clip_id>.mp4   — optional (--write-mp4)
    label_queue.csv       — copy for annotators (same as manifest)

After labeling, merge into tracks:
  python merge_manual_labels.py --batch <batch_id>
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

ACTIVITIES = [
    "walking",
    "running",
    "biking",
    "sitting",
    "standing",
    "talking_socializing",
    "playing",
    "dog_walking",
    "exercising",
    "picnic_resting",
]

AGE_GROUPS = ["child", "adult"]

# Not used for activity/age training or visitor metrics
LABEL_STATUS_SKIP = "skip"  # too blurry / occluded — leave tracks unchanged
LABEL_STATUS_NOT_PERSON = "not_a_person"  # YOLO false positive — exclude whole track
LABEL_STATUS_VALID = "valid"  # real person — fill activity + age


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(path: Path | None) -> dict:
    cfg_path = path or Path(__file__).with_name("labeling_config.yaml")
    with cfg_path.open() as f:
        return yaml.safe_load(f)


def resolve_video_path(video_id: str, tracks_path: Path, cfg: dict) -> Path:
    root = _project_root()
    rel = cfg.get("videos", {}).get(video_id)
    if not rel:
        raise SystemExit(f"No video path for {video_id} in labeling_config.yaml")
    p = Path(rel)
    if not p.is_absolute():
        p = (root / p).resolve()
    if not p.is_file():
        depot = root.parent  # depotData/
        for sub in ("west", "east"):
            alt = depot / sub / f"{video_id}.MP4"
            if alt.is_file():
                return alt
        raise SystemExit(f"Video not found: {p}")
    return p


def track_summaries(tracks: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    persons = tracks[tracks["class"] == "person"].copy()
    if persons.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for tid, g in persons.groupby("track_id"):
        g = g.sort_values("timestamp")
        t0, t1 = float(g["timestamp"].min()), float(g["timestamp"].max())
        dwell = t1 - t0
        if dwell < cfg["min_dwell_sec"] or len(g) < cfg["min_track_frames"]:
            continue
        spd = pd.to_numeric(g["speed"], errors="coerce").fillna(0.0)
        conf = pd.to_numeric(g["confidence"], errors="coerce").fillna(0.0)
        h = (g["bbox_y2"] - g["bbox_y1"]).astype(float)
        wbox = (g["bbox_x2"] - g["bbox_x1"]).astype(float)
        area = h * wbox
        mean_conf = float(conf.mean())
        mean_h = float(h.mean())
        mean_area = float(area.mean())
        if mean_conf < cfg.get("min_confidence", 0.0):
            continue
        if mean_h < cfg.get("min_bbox_height_px", 0):
            continue
        if mean_area < cfg.get("min_bbox_area_px", 0):
            continue
        zone = g["zone"].mode().iloc[0] if len(g) else ""
        rows.append(
            {
                "track_id": int(tid),
                "start_time": t0,
                "end_time": t1,
                "duration_s": dwell,
                "n_frames": len(g),
                "mean_confidence": mean_conf,
                "mean_bbox_height_px": mean_h,
                "mean_bbox_area_px": mean_area,
                "mean_speed": float(spd.mean()),
                "max_speed": float(spd.max()),
                "zone": str(zone),
                "near_lake": bool(g["near_lake"].any()) if "near_lake" in g.columns else False,
                "near_tree": bool(g["near_tree"].any()) if "near_tree" in g.columns else False,
                "in_group": bool((pd.to_numeric(g["group_id"], errors="coerce") > 0).any())
                if "group_id" in g.columns
                else False,
                "video_id": str(g["video_id"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def _strata(row: pd.Series, cfg: dict) -> list[str]:
    tags = []
    if row["near_lake"]:
        tags.append("near_lake")
    if row["near_tree"]:
        tags.append("near_tree")
    if row["in_group"]:
        tags.append("in_group")
    if row["max_speed"] >= cfg["speed_running_min"]:
        tags.append("fast")
    elif row["mean_speed"] >= cfg["speed_walking_min"]:
        tags.append("moving")
    else:
        tags.append("slow")
    return tags


def sample_tracks(summary: pd.DataFrame, n: int, seed: int, cfg: dict) -> pd.DataFrame:
    if len(summary) <= n:
        return summary.copy()

    rng = random.Random(seed)
    pools: dict[str, list[int]] = {
        "near_lake": [],
        "near_tree": [],
        "in_group": [],
        "fast": [],
        "slow": [],
        "moving": [],
    }
    for i, row in summary.iterrows():
        for tag in _strata(row, cfg):
            if tag in pools:
                pools[tag].append(i)

    picked: set[int] = set()
    order = ["near_lake", "near_tree", "in_group", "fast", "moving", "slow"]
    per_pool = max(1, n // len(order))
    for key in order:
        cand = [i for i in pools.get(key, []) if i not in picked]
        rng.shuffle(cand)
        for i in cand[:per_pool]:
            picked.add(i)
        if len(picked) >= n:
            break

    rest = [i for i in summary.index if i not in picked]
    rng.shuffle(rest)
    for i in rest:
        if len(picked) >= n:
            break
        picked.add(i)

    return summary.loc[sorted(picked)].copy()


def _crop(frame: np.ndarray, bbox: tuple[float, float, float, float], margin: float) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    pad_x, pad_y = bw * margin, bh * margin
    x1 = int(max(0, x1 - pad_x))
    y1 = int(max(0, y1 - pad_y))
    x2 = int(min(w, x2 + pad_x))
    y2 = int(min(h, y2 + pad_y))
    if x2 <= x1 or y2 <= y1:
        return frame
    return frame[y1:y2, x1:x2]


def export_clip_frames(
    video: Path,
    track_rows: pd.DataFrame,
    out_dir: Path,
    cfg: dict,
    cap: cv2.VideoCapture | None,
) -> cv2.VideoCapture:
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")

    t0 = float(track_rows["timestamp"].min()) - cfg["clip_pad_sec"]
    t1 = float(track_rows["timestamp"].max()) + cfg["clip_pad_sec"]
    n = int(cfg["frames_per_clip"])
    times = np.linspace(max(0, t0), t1, n)

    out_dir.mkdir(parents=True, exist_ok=True)
    for k, t in enumerate(times):
        # nearest row in track
        idx = (track_rows["timestamp"] - t).abs().idxmin()
        row = track_rows.loc[idx]
        bbox = (
            float(row["bbox_x1"]),
            float(row["bbox_y1"]),
            float(row["bbox_x2"]),
            float(row["bbox_y2"]),
        )
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        crop = _crop(frame, bbox, cfg["crop_margin"])
        cv2.imwrite(str(out_dir / f"frame_{k+1:02d}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])

    return cap


def write_mp4_from_frames(clip_dir: Path, mp4_path: Path, fps: float = 2.0) -> None:
    frames = sorted(clip_dir.glob("frame_*.jpg"))
    if not frames:
        return
    im0 = cv2.imread(str(frames[0]))
    h, w = im0.shape[:2]
    tmp = mp4_path.with_suffix(".avi")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    for f in frames:
        im = cv2.imread(str(f))
        if im is None:
            continue
        if im.shape[:2] != (h, w):
            im = cv2.resize(im, (w, h))
        writer.write(im)
    writer.release()
    if shutil.which("ffmpeg"):
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(tmp),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(mp4_path),
            ],
            check=False,
        )
        tmp.unlink(missing_ok=True)
    else:
        tmp.rename(mp4_path)


def run_export(
    tracks_path: Path,
    output_dir: Path,
    batch_id: str,
    n_clips: int,
    seed: int,
    write_mp4: bool,
    cfg: dict,
) -> Path:
    tracks = pd.read_csv(tracks_path)
    video_id = str(tracks["video_id"].iloc[0])
    video_path = resolve_video_path(video_id, tracks_path, cfg)

    summary = track_summaries(tracks, cfg)
    if summary.empty:
        raise SystemExit("No tracks passed min_dwell / min_frames filters.")

    chosen = sample_tracks(summary, n_clips, seed, cfg)
    batch_dir = output_dir / batch_id
    clips_root = batch_dir / "clips"
    clips_root.mkdir(parents=True, exist_ok=True)

    persons = tracks[tracks["class"] == "person"]
    cap: cv2.VideoCapture | None = None
    manifest_rows: list[dict] = []

    for _, row in chosen.iterrows():
        tid = int(row["track_id"])
        g = persons[persons["track_id"] == tid].sort_values("timestamp")
        clip_id = f"{video_id}_tr{tid:04d}"
        clip_dir = clips_root / clip_id
        cap = export_clip_frames(video_path, g, clip_dir, cfg, cap)
        if write_mp4:
            write_mp4_from_frames(clip_dir, clips_root / f"{clip_id}.mp4")

        tags = ",".join(_strata(row, cfg))
        manifest_rows.append(
            {
                "clip_id": clip_id,
                "video_id": video_id,
                "track_id": tid,
                "start_time": round(float(row["start_time"]), 3),
                "end_time": round(float(row["end_time"]), 3),
                "duration_s": round(float(row["duration_s"]), 3),
                "n_frames": int(row["n_frames"]),
                "zone": row["zone"],
                "strata": tags,
                "near_lake": row["near_lake"],
                "near_tree": row["near_tree"],
                "in_group": row["in_group"],
                "mean_speed": round(float(row["mean_speed"]), 2),
                "mean_confidence": round(float(row["mean_confidence"]), 3),
                "mean_bbox_height_px": round(float(row["mean_bbox_height_px"]), 1),
                "clip_dir": str(clip_dir.relative_to(batch_dir)),
                "video_path": str(video_path),
                "label_status": "",
                "activity_label": "",
                "apparent_age_group": "",
                "annotator": "",
                "notes": "",
                "split": "",
            }
        )

    if cap is not None:
        cap.release()

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = batch_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    manifest.to_csv(batch_dir / "label_queue.csv", index=False)

    meta = {
        "batch_id": batch_id,
        "video_id": video_id,
        "n_clips": len(manifest),
        "n_candidates": len(summary),
        "activities": ACTIVITIES,
        "age_groups": AGE_GROUPS,
        "label_status_values": [
            LABEL_STATUS_VALID,
            LABEL_STATUS_SKIP,
            LABEL_STATUS_NOT_PERSON,
        ],
    }
    (batch_dir / "batch_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"Wrote {len(manifest)} clips -> {batch_dir}")
    print(f"  manifest: {manifest_path}")
    return batch_dir


def main() -> None:
    p = argparse.ArgumentParser(description="Export track clips for manual labeling.")
    p.add_argument("--tracks", type=Path, required=True, help="outputs/mvp/<video_id>/tracks.csv")
    p.add_argument("--batch-id", default=None, help="default: <video_id>_batch01")
    p.add_argument("--export-root", type=Path, default=None)
    p.add_argument("-n", "--num-clips", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--write-mp4", action="store_true")
    p.add_argument("--config", type=Path, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    tracks = pd.read_csv(args.tracks)
    vid = str(tracks["video_id"].iloc[0])
    batch_id = args.batch_id or f"{vid}_batch01"
    export_root = args.export_root or (_project_root() / cfg["export_root"]).resolve()
    if not export_root.is_absolute():
        export_root = (_project_root() / export_root).resolve()

    n = args.num_clips if args.num_clips is not None else int(cfg["default_n_clips"])
    run_export(args.tracks, export_root, batch_id, n, args.seed, args.write_mp4, cfg)


if __name__ == "__main__":
    main()
