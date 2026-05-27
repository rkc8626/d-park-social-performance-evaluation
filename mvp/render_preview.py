#!/usr/bin/env python3
"""Render preview_annotated.mp4 from an existing tracks.csv (no re-inference)."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def load_roi_polygons(roi_path: Path | None) -> list[tuple[str, np.ndarray]]:
    """Load optional ROI JSON: { \"lake\": [[x,y],...], \"tree_area\": [...] }."""
    if roi_path is None or not roi_path.is_file():
        return []
    import json

    data = json.loads(roi_path.read_text())
    skip = {"camera_id", "video_id", "reference_frame", "resolution", "notes"}
    out: list[tuple[str, np.ndarray]] = []
    for name, pts in data.items():
        if name.startswith("_") or name in skip or not isinstance(pts, list) or len(pts) < 3:
            continue
        arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        out.append((name, arr))
    return out


def draw_rois(frame: np.ndarray, rois: list[tuple[str, np.ndarray]]) -> None:
    palette = [(0, 200, 80), (0, 180, 120), (80, 220, 80), (0, 160, 60), (120, 255, 120)]
    for i, (name, poly) in enumerate(rois):
        if name == "lake":
            c = (255, 120, 0)
        elif name.startswith("tree"):
            c = palette[i % len(palette)]
        else:
            c = (200, 200, 0)
        cv2.polylines(frame, [poly], True, c, 2)
        if len(poly):
            cx, cy = poly[:, 0, :].mean(axis=0).astype(int)
            cv2.putText(frame, name, (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2)


def render(
    video_path: Path,
    tracks_path: Path,
    output_path: Path,
    roi_path: Path | None = None,
) -> None:
    tracks = pd.read_csv(tracks_path)
    if tracks.empty:
        raise SystemExit(f"No rows in {tracks_path}")

    rois = load_roi_polygons(roi_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    timestamps = sorted(tracks["timestamp"].unique())
    # Infer stride from median delta between sampled timestamps
    if len(timestamps) > 1:
        deltas = np.diff(timestamps)
        dt = float(np.median(deltas[deltas > 0])) or (1.0 / fps)
        out_fps = max(1.0, 1.0 / dt)
    else:
        out_fps = fps

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, out_fps, (w, h))

    class_colors = {
        "person": (0, 255, 0),
        "bicycle": (255, 180, 0),
        "cyclist": (255, 180, 0),
    }

    for t in timestamps:
        frame_idx = int(round(float(t) * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        draw_rois(frame, rois)
        subset = tracks[tracks["timestamp"] == t]
        for _, row in subset.iterrows():
            x1, y1, x2, y2 = int(row.bbox_x1), int(row.bbox_y1), int(row.bbox_x2), int(row.bbox_y2)
            cls = str(row["class"])
            color = class_colors.get(cls, (0, 200, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{cls} #{int(row.track_id)}"
            cv2.putText(frame, label, (x1, max(y1 - 6, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        writer.write(frame)

    cap.release()
    writer.release()
    print(f"Wrote {output_path} ({len(timestamps)} frames @ {out_fps:.2f} fps)")


def main() -> None:
    p = argparse.ArgumentParser(description="Render annotated preview from tracks.csv")
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--tracks", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--roi", type=Path, default=None, help="Optional ROI polygons JSON")
    args = p.parse_args()
    out = args.output or args.tracks.parent / "preview_annotated.mp4"
    render(args.video, args.tracks, out, args.roi)


if __name__ == "__main__":
    main()
