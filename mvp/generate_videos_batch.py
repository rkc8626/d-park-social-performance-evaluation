#!/usr/bin/env python3
"""Write videos_batch.yaml from all MP4 files under depotData/east and west."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEPOT = ROOT.parent  # depotData/


def _videos(side: str) -> list[dict]:
    d = DEPOT / side
    if not d.is_dir():
        raise SystemExit(f"Missing {d}")
    items = []
    for p in sorted(d.glob("*.MP4")):
        items.append({"id": p.stem, "path": f"../../{side}/{p.name}"})
    return items


def _classifier_video_paths() -> dict[str, str]:
    """Paths relative to project root (same convention as classifier_config.yaml)."""
    out: dict[str, str] = {}
    for side in ("east", "west"):
        for item in _videos(side):
            out[item["id"]] = f"../{side}/{item['id']}.MP4"
    return out


def _sync_classifier_config(classifier_path: Path) -> None:
    cfg = yaml.safe_load(classifier_path.read_text())
    cfg["videos"] = _classifier_video_paths()
    classifier_path.write_text(yaml.dump(cfg, sort_keys=False, allow_unicode=True))
    print(f"Synced {len(cfg['videos'])} videos into {classifier_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path(__file__).with_name("videos_batch.yaml"))
    p.add_argument(
        "--sync-classifier",
        action="store_true",
        help="Also update classifier_config.yaml videos map (all east+west MP4s)",
    )
    args = p.parse_args()

    cfg = {
        "output_dir": "../outputs/mvp",
        "east": {
            "camera_id": "east",
            "roi": "../annotations/roi/east.json",
            "videos": _videos("east"),
        },
        "west": {
            "camera_id": "west",
            "roi": "../annotations/roi/west.json",
            "videos": _videos("west"),
        },
        "max_seconds": 0,
        "frame_stride": 4,
        "preview_seconds": 0,
    }
    header = (
        "# Auto-generated — all depotData/east + west videos.\n"
        "# Regenerate: python generate_videos_batch.py\n"
        "# Same lake/tree ROI per camera (east.json / west.json).\n"
    )
    args.out.write_text(header + yaml.dump(cfg, sort_keys=False, allow_unicode=True))
    ne, nw = len(cfg["east"]["videos"]), len(cfg["west"]["videos"])
    print(f"Wrote {args.out} — east={ne}, west={nw}, total={ne + nw}")

    if args.sync_classifier:
        _sync_classifier_config(Path(__file__).with_name("classifier_config.yaml"))


if __name__ == "__main__":
    main()
