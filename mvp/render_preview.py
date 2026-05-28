#!/usr/bin/env python3
"""Render preview_annotated.mp4 from tracks.csv (no re-inference).

Fixes green-screen issues from HEVC frame seeking by using sequential decode
or ffmpeg frame extraction, and encodes with libx264 (yuv420p).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def load_roi_polygons(
    roi_path: Path | None, target_size: tuple[int, int] | None = None
) -> list[tuple[str, np.ndarray]]:
    if roi_path is None or not roi_path.is_file():
        return []
    from roi_utils import load_roi_polygons as _load

    polys, _ = _load(roi_path, target_size=target_size)
    return polys


def draw_rois(frame: np.ndarray, rois: list[tuple[str, np.ndarray]]) -> None:
    palette = [(0, 200, 80), (0, 180, 120), (80, 220, 80), (0, 160, 60), (120, 255, 120)]
    for i, (name, poly) in enumerate(rois):
        if name == "park":
            continue  # full-frame park: skip overlay to reduce clutter
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


def subsample_timestamps(timestamps: list[float], output_fps: float) -> list[float]:
    if output_fps <= 0 or len(timestamps) < 2:
        return timestamps
    ts = np.array(timestamps, dtype=float)
    t_end = float(ts[-1])
    targets = np.arange(0.0, t_end + 1e-6, 1.0 / output_fps)
    kept: list[float] = []
    for tt in targets:
        i = int(np.argmin(np.abs(ts - tt)))
        t = float(ts[i])
        if not kept or t != kept[-1]:
            kept.append(t)
    return kept


def frame_ok(frame: np.ndarray | None) -> bool:
    if frame is None or frame.size == 0:
        return False
    return float(frame.mean()) > 8.0 and float(frame.std()) > 5.0


def grab_frame_ffmpeg(video_path: Path, t_sec: float, w: int, h: int) -> np.ndarray | None:
    if not shutil.which("ffmpeg"):
        return None
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{t_sec:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{w}x{h}",
        "pipe:1",
    ]
    try:
        raw = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=120)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    need = w * h * 3
    if len(raw) < need:
        return None
    return np.frombuffer(raw[:need], dtype=np.uint8).reshape((h, w, 3))


def read_frame_sequential(
    cap: cv2.VideoCapture,
    target_idx: int,
    state: dict,
) -> np.ndarray | None:
    """Decode forward from last position (reliable for HEVC)."""
    cur = state.get("idx", -1)
    frame = state.get("frame")
    if target_idx < cur:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        cur = -1
        frame = None
    while cur < target_idx:
        ok, frame = cap.read()
        if not ok:
            return frame
        cur += 1
    state["idx"] = cur
    state["frame"] = frame
    return frame


def encode_mp4_ffmpeg(frame_paths: list[Path], output_path: Path, fps: float) -> None:
    if not frame_paths:
        raise SystemExit("No frames to encode")
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found — module load ffmpeg/4.3.1")
    pattern = str(frame_paths[0].parent / "%06d.jpg")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            pattern,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        check=True,
    )


def render(
    video_path: Path,
    tracks_path: Path,
    output_path: Path,
    roi_path: Path | None = None,
    output_fps: float = 2.0,
    scale: float = 0.5,
    max_duration: float | None = None,
) -> None:
    tracks = pd.read_csv(tracks_path)
    if tracks.empty:
        raise SystemExit(f"No rows in {tracks_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rois = load_roi_polygons(roi_path, target_size=(w, h))

    ts_all = sorted(tracks["timestamp"].unique(), key=float)
    if max_duration is not None:
        ts_all = [t for t in ts_all if float(t) <= max_duration]
    timestamps = subsample_timestamps([float(t) for t in ts_all], output_fps)
    n_source = len(tracks["timestamp"].unique())

    out_w = max(2, int(w * scale) // 2 * 2)
    out_h = max(2, int(h * scale) // 2 * 2)

    class_colors = {
        "person": (255, 200, 0),
        "bicycle": (0, 165, 255),
        "cyclist": (0, 165, 255),
    }

    decode_state: dict = {}
    written = 0
    skipped = 0

    with tempfile.TemporaryDirectory(prefix="preview_frames_") as tmp:
        tmp_path = Path(tmp)
        frame_paths: list[Path] = []

        for i, t in enumerate(timestamps):
            frame_idx = int(round(float(t) * fps))
            frame = read_frame_sequential(cap, frame_idx, decode_state)
            if not frame_ok(frame):
                frame = grab_frame_ffmpeg(video_path, float(t), w, h)
            if not frame_ok(frame):
                skipped += 1
                continue

            draw_rois(frame, rois)
            subset = tracks[np.isclose(tracks["timestamp"].astype(float), float(t))]
            for _, row in subset.iterrows():
                x1, y1, x2, y2 = (
                    int(row.bbox_x1),
                    int(row.bbox_y1),
                    int(row.bbox_x2),
                    int(row.bbox_y2),
                )
                cls = str(row["class"])
                color = class_colors.get(cls, (0, 200, 255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{cls} #{int(row.track_id)}"
                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 6, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

            if scale != 1.0:
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

            out_file = tmp_path / f"{written:06d}.jpg"
            cv2.imwrite(str(out_file), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            frame_paths.append(out_file)
            written += 1

        cap.release()

        if written == 0:
            raise SystemExit("No valid frames decoded — check video path and ffmpeg")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        encode_mp4_ffmpeg(frame_paths, output_path, output_fps)

    print(
        f"Wrote {output_path} ({written} frames @ {output_fps:.2f} fps, {out_w}x{out_h}, "
        f"skipped {skipped} bad seeks; source track samples={n_source})"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Render annotated preview from tracks.csv")
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--tracks", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--roi", type=Path, default=None)
    p.add_argument("--output-fps", type=float, default=2.0)
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument(
        "--max-duration",
        type=float,
        default=None,
        help="Only render timestamps <= this many seconds (for testing)",
    )
    args = p.parse_args()
    out = args.output or args.tracks.parent / "preview_annotated.mp4"
    render(
        args.video,
        args.tracks,
        out,
        args.roi,
        output_fps=args.output_fps,
        scale=args.scale,
        max_duration=args.max_duration,
    )


if __name__ == "__main__":
    main()
