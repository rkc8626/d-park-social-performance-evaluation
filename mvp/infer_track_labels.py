#!/usr/bin/env python3
"""Run trained classifiers on all person tracks; update tracks.csv."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from torchvision import models, transforms

from export_track_clips import ACTIVITIES, AGE_GROUPS, LABEL_STATUS_NOT_PERSON
from train_clip_classifier import _make_model


def _resolve_video(root: Path, video_id: str, cfg: dict) -> Path:
    rel = cfg.get("videos", {}).get(video_id)
    if not rel:
        raise SystemExit(f"No video path for {video_id}")
    p = (root / rel).resolve()
    if p.is_file():
        return p
    depot = root.parent
    for sub in ("west", "east"):
        alt = depot / sub / f"{video_id}.MP4"
        if alt.is_file():
            return alt
    raise SystemExit(f"Video not found: {p}")


def _crop(frame: np.ndarray, bbox: tuple[float, float, float, float], margin: float = 0.15) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    pad_x, pad_y = bw * margin, bh * margin
    x1 = int(max(0, x1 - pad_x))
    y1 = int(max(0, y1 - pad_y))
    x2 = int(min(w, x2 + pad_x))
    y2 = int(min(h, y2 + pad_y))
    return frame[y1:y2, x1:x2]


def _load_classifier(path: Path, device: torch.device) -> tuple[torch.nn.Module, list[str], int]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    classes = list(ckpt["classes"])
    image_size = int(ckpt.get("image_size", 224))
    model = _make_model(len(classes))
    model.load_state_dict(ckpt["model"])
    model.eval()
    model.to(device)
    return model, classes, image_size


def _predict_frames(
    model: torch.nn.Module,
    classes: list[str],
    crops: list[np.ndarray],
    image_size: int,
    device: torch.device,
) -> tuple[str, float]:
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            norm,
        ]
    )
    logits_sum = None
    with torch.no_grad():
        for crop in crops:
            if crop.size == 0:
                continue
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            t = tf(Image.fromarray(rgb)).unsqueeze(0).to(device)
            out = model(t)
            logits_sum = out if logits_sum is None else logits_sum + out
    if logits_sum is None:
        return "", 0.0
    logits = logits_sum / len(crops)
    prob = torch.softmax(logits, dim=1)[0]
    idx = int(prob.argmax().item())
    return classes[idx], float(prob[idx].item())


def _manual_track_keys(root: Path, cfg: dict) -> set[tuple[str, int]]:
    """(video_id, track_id) pairs with human labels — do not overwrite on that video."""
    keys: set[tuple[str, int]] = set()
    for batch_rel in cfg.get("batches", []):
        batch_dir = Path(batch_rel)
        if not batch_dir.is_absolute():
            batch_dir = (root / batch_dir).resolve()
        manifest = batch_dir / "manifest.csv"
        if not manifest.is_file():
            continue
        m = pd.read_csv(manifest)
        m = m[m["label_status"].astype(str).str.strip() == "valid"]
        for _, row in m.iterrows():
            keys.add((str(row["video_id"]), int(row["track_id"])))
    return keys


def infer_video(
    tracks_path: Path,
    output_dir: Path,
    cfg: dict,
    device: torch.device,
    overwrite_manual: bool,
) -> None:
    root = Path(__file__).resolve().parents[1]
    models_dir = (Path(__file__).parent / cfg.get("models_dir", "models")).resolve()
    act_path = models_dir / cfg.get("activity_model", "activity_resnet18.pt")
    age_path = models_dir / cfg.get("age_model", "age_resnet18.pt")
    if not act_path.is_file() or not age_path.is_file():
        raise SystemExit(f"Train models first. Missing {act_path} or {age_path}")

    act_model, act_classes, image_size = _load_classifier(act_path, device)
    age_model, age_classes, _ = _load_classifier(age_path, device)

    tracks = pd.read_csv(tracks_path)
    video_id = str(tracks["video_id"].iloc[0])
    video_path = _resolve_video(root, video_id, cfg)
    persons = tracks[tracks["class"] == "person"].copy()
    manual_keys = _manual_track_keys(root, cfg) if not overwrite_manual else set()

    n_samples = int(cfg.get("infer_samples_per_track", 5))
    conf_min = float(cfg.get("infer_confidence_min", 0.35))
    margin = 0.15

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {video_path}")

    track_ids = sorted(persons["track_id"].unique())
    n_done = 0
    for tid in track_ids:
        tid = int(tid)
        if (video_id, tid) in manual_keys:
            continue
        g = persons[persons["track_id"] == tid].sort_values("timestamp")
        if len(g) < 2:
            continue
        idxs = np.linspace(0, len(g) - 1, min(n_samples, len(g))).astype(int)
        crops: list[np.ndarray] = []
        for i in idxs:
            row = g.iloc[int(i)]
            t = float(row["timestamp"])
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            bbox = (
                float(row["bbox_x1"]),
                float(row["bbox_y1"]),
                float(row["bbox_x2"]),
                float(row["bbox_y2"]),
            )
            c = _crop(frame, bbox, margin)
            if c.size > 0:
                crops.append(c)

        act, act_p = _predict_frames(act_model, act_classes, crops, image_size, device)
        age, age_p = _predict_frames(age_model, age_classes, crops, image_size, device)
        if act_p < conf_min and age_p < conf_min:
            continue

        mask = tracks["track_id"] == tid
        if act and act_p >= conf_min:
            tracks.loc[mask, "activity_label"] = act
        if age and age_p >= conf_min:
            tracks.loc[mask, "apparent_age_group"] = age
        n_done += 1

    cap.release()
    tracks.to_csv(tracks_path, index=False)
    skipped = sum(1 for t in track_ids if (video_id, int(t)) in manual_keys)
    print(f"{video_id}: inferred {n_done} tracks (skipped {skipped} manual), saved {tracks_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tracks", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path(__file__).with_name("classifier_config.yaml"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--overwrite-manual", action="store_true")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load(args.config.read_text())
    if args.overwrite_manual:
        cfg["overwrite_manual"] = True
    device = torch.device(args.device)
    infer_video(args.tracks, args.tracks.parent, cfg, device, bool(cfg.get("overwrite_manual", False)))


if __name__ == "__main__":
    main()
