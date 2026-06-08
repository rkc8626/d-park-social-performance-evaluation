#!/usr/bin/env python3
"""Run YOLO+track, classifier inference, and ROI tables for batch videos."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
import yaml

import cv2

from regenerate_tables import regenerate
from render_preview import render as render_preview
from roi_utils import resolve_roi_for_video
from run_poc import load_config, run_poc

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEMO_SECONDS = 10.0
DEMO_HEIGHT = 480
DEMO_FPS = 2.0
DEMO_NAME = "demo_10s_480p.mp4"
PIPELINE_DONE = ".pipeline_complete"


def _resolve(base: Path, rel: str) -> Path:
    p = Path(rel)
    return p.resolve() if p.is_absolute() else (base / p).resolve()


def load_batch(path: Path) -> dict:
    base = path.parent.resolve()
    cfg = yaml.safe_load(path.read_text())
    out: dict = {"output_dir": str(_resolve(base, cfg["output_dir"]))}
    for side in ("east", "west"):
        sc = cfg[side]
        videos = []
        for v in sc["videos"]:
            videos.append(
                {
                    "id": v["id"],
                    "path": str(_resolve(base, v["path"])),
                    "camera_id": sc["camera_id"],
                    "roi": str(_resolve(base, sc["roi"])),
                }
            )
        out[side] = videos
    for key in ("max_seconds", "frame_stride", "preview_seconds"):
        if key in cfg:
            out[key] = cfg[key]
    return out


def render_demo(entry: dict, out_root: Path, skip_if_exists: bool = False) -> None:
    demo_path = out_root / DEMO_NAME
    tracks_path = out_root / "tracks.csv"
    if not tracks_path.is_file():
        print(f"=== Demo {entry['id']}: skip (no tracks.csv) ===")
        return
    if skip_if_exists and demo_path.is_file():
        print(f"=== Demo {entry['id']}: skip (exists) ===")
        return

    cap = cv2.VideoCapture(str(entry["path"]))
    if not cap.isOpened():
        print(f"=== Demo {entry['id']}: skip (cannot open video) ===")
        return
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 2160
    cap.release()
    scale = DEMO_HEIGHT / h

    roi_path = resolve_roi_for_video(PROJECT_ROOT, entry["id"], entry["camera_id"])
    print(f"=== Demo {entry['id']} ({DEMO_SECONDS}s @ {DEMO_HEIGHT}p) ===")
    try:
        render_preview(
            Path(entry["path"]),
            tracks_path,
            demo_path,
            roi_path,
            output_fps=DEMO_FPS,
            scale=scale,
            max_duration=DEMO_SECONDS,
        )
    except SystemExit as exc:
        print(f"=== Demo {entry['id']}: WARN ({exc}) — retry from t=0 without track filter ===")
        try:
            render_preview(
                Path(entry["path"]),
                tracks_path,
                demo_path,
                roi_path,
                output_fps=DEMO_FPS,
                scale=scale,
                max_duration=DEMO_SECONDS,
                time_window=(0.0, DEMO_SECONDS),
            )
        except SystemExit as exc2:
            print(f"=== Demo {entry['id']}: FAILED ({exc2}) — continuing batch ===")


def iter_videos(batch: dict, video_id: str | None = None):
    for side in ("east", "west"):
        for v in batch[side]:
            if video_id and v["id"] != video_id:
                continue
            yield v


def process_video(
    entry: dict,
    poc_base: dict,
    batch: dict,
    cls_cfg: dict,
    device: "torch.device",
    skip_poc: bool,
) -> None:
    from infer_track_labels import infer_video
    vid = entry["id"]
    out_root = Path(batch["output_dir"]) / vid
    tracks_path = out_root / "tracks.csv"

    if not skip_poc or not tracks_path.is_file():
        cfg = copy.deepcopy(poc_base)
        cfg["video_path"] = entry["path"]
        cfg["video_id"] = vid
        cfg["camera_id"] = entry["camera_id"]
        cfg["output_dir"] = batch["output_dir"]
        if "max_seconds" in batch:
            cfg["max_seconds"] = batch["max_seconds"]
        if "frame_stride" in batch:
            cfg["frame_stride"] = batch["frame_stride"]
        if "preview_seconds" in batch:
            cfg["preview_seconds"] = batch["preview_seconds"]
        print(f"\n=== POC {vid} ({entry['camera_id']}) ===")
        run_poc(cfg)
    else:
        print(f"\n=== POC {vid}: skipped (tracks exist) ===")

    roi_path = resolve_roi_for_video(PROJECT_ROOT, vid, entry["camera_id"])
    print(f"=== ROI zones {vid} ({roi_path.name}) ===")
    regenerate(out_root, roi_path, vid)

    print(f"=== Infer {vid} ===")
    infer_video(
        tracks_path,
        out_root,
        cls_cfg,
        device,
        bool(cls_cfg.get("overwrite_manual", False)),
        video_path=Path(entry["path"]),
    )

    print(f"=== Tables {vid} ===")
    regenerate(out_root, roi_path, vid)

    render_demo(entry, out_root)
    (out_root / PIPELINE_DONE).write_text("ok\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Batch video pipeline with trained classifiers")
    p.add_argument("--batch", type=Path, default=Path(__file__).with_name("videos_batch.yaml"))
    p.add_argument("--poc-config", type=Path, default=Path(__file__).with_name("config.yaml"))
    p.add_argument("--classifier-config", type=Path, default=Path(__file__).with_name("classifier_config.yaml"))
    p.add_argument("--video-id", default=None, help="Process one video only")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--skip-poc", action="store_true", help="Reuse existing tracks.csv")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip videos that already have hourly_metrics.csv + tracks.csv",
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Log failures and continue with remaining videos (batch runs)",
    )
    args = p.parse_args()

    batch = load_batch(args.batch)
    poc_base = load_config(args.poc_config)
    cls_cfg = yaml.safe_load(args.classifier_config.read_text())
    device = torch.device(args.device)

    entries = list(iter_videos(batch, args.video_id))
    if not entries:
        raise SystemExit(f"No videos matched --video-id {args.video_id!r}")

    print(f"Processing {len(entries)} video(s) on {device}")
    for entry in entries:
        if not Path(entry["path"]).is_file():
            raise SystemExit(f"Video not found: {entry['path']}")
        out_root = Path(batch["output_dir"]) / entry["id"]
        if args.skip_existing and (out_root / PIPELINE_DONE).is_file():
            print(f"\n=== Skip {entry['id']} (pipeline complete) ===")
            render_demo(entry, out_root, skip_if_exists=True)
            continue
        if args.skip_existing and (out_root / "tracks.csv").is_file():
            print(f"\n=== Resume {entry['id']} (tracks exist, skip POC) ===")
            process_video(entry, poc_base, batch, cls_cfg, device, skip_poc=True)
            continue
        try:
            process_video(entry, poc_base, batch, cls_cfg, device, args.skip_poc)
        except Exception as exc:
            if not args.continue_on_error:
                raise
            print(f"\n=== ERROR {entry['id']}: {exc} ===", flush=True)

    print("\nAll done.")


if __name__ == "__main__":
    main()
