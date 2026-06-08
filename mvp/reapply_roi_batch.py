#!/usr/bin/env python3
"""Re-apply ROI (per-video or camera default) and refresh metrics tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from regenerate_tables import regenerate
from roi_utils import resolve_roi_for_video


def _load_batch(batch_path: Path) -> list[dict]:
    base = batch_path.parent.resolve()
    cfg = yaml.safe_load(batch_path.read_text())
    out: list[dict] = []
    for side in ("east", "west"):
        sc = cfg[side]
        cam = sc["camera_id"]
        for v in sc["videos"]:
            vid = v["id"]
            out.append(
                {
                    "video_id": vid,
                    "camera_id": cam,
                    "output_dir": (base / cfg["output_dir"]).resolve() / vid,
                }
            )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Recompute ROI zones + hourly metrics for batch outputs")
    p.add_argument("--batch", type=Path, default=Path(__file__).with_name("videos_batch.yaml"))
    p.add_argument("--video-id", default=None, help="Only one video")
    p.add_argument("--output-root", type=Path, default=None, help="Override outputs/mvp root")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    entries = _load_batch(args.batch)
    if args.video_id:
        entries = [e for e in entries if e["video_id"] == args.video_id]
    if not entries:
        raise SystemExit("No videos matched")

    for e in entries:
        out = args.output_root / e["video_id"] if args.output_root else e["output_dir"]
        tracks = out / "tracks.csv"
        if not tracks.is_file() or tracks.stat().st_size < 50:
            print(f"Skip {e['video_id']}: no tracks")
            continue
        roi = resolve_roi_for_video(root, e["video_id"], e["camera_id"])
        print(f"=== {e['video_id']} roi={roi.name} ===")
        regenerate(out, roi, e["video_id"])

    print("Done.")


if __name__ == "__main__":
    main()
