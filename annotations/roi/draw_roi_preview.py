#!/usr/bin/env python3
"""Draw ROI polygon(s) on a reference frame and save a PNG.

Example (west lake only):
  cd depotData/d-park-social-performance-evaluation
  .venv/bin/python annotations/roi/draw_roi_preview.py \\
    --roi annotations/roi/west.json --layers lake
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mvp"))
from roi_utils import load_roi_polygons  # noqa: E402

_COLORS: dict[str, tuple[int, int, int]] = {
    "lake": (0, 140, 255),
    "park": (200, 180, 255),
}
_TREE_PALETTE = [(0, 200, 80), (0, 180, 120), (80, 220, 80), (0, 160, 60), (120, 255, 120)]


def _color(name: str, index: int) -> tuple[int, int, int]:
    if name in _COLORS:
        return _COLORS[name]
    if name.startswith("tree"):
        return _TREE_PALETTE[index % len(_TREE_PALETTE)]
    return (200, 200, 0)


def main() -> None:
    p = argparse.ArgumentParser(description="Overlay ROI polygons on reference frame.")
    p.add_argument("--roi", type=Path, required=True, help="east.json / west.json")
    p.add_argument(
        "--layers",
        nargs="+",
        default=["lake"],
        help="Polygon keys to draw, e.g. lake tree_1 (default: lake)",
    )
    p.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Background image (default: reference_frame from JSON)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG (default: reference_frames/<video_id>_<layers>.png)",
    )
    p.add_argument("--alpha", type=float, default=0.35, help="Overlay blend 0–1")
    args = p.parse_args()

    roi_path = args.roi.resolve()
    roi_dir = roi_path.parent

    import json

    meta = json.loads(roi_path.read_text())
    ref_rel = meta.get("reference_frame")
    image_path = args.image
    if image_path is None:
        if not ref_rel:
            p.error("No reference_frame in JSON; pass --image")
        image_path = roi_dir / ref_rel
    image_path = image_path.resolve()

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise SystemExit(f"Cannot read image: {image_path}")
    h, w = frame.shape[:2]

    polys, info = load_roi_polygons(roi_path, target_size=(w, h))
    want = {x.lower() for x in args.layers}
    selected = [(n, poly) for n, poly in polys if n.lower() in want]
    if not selected:
        names = [n for n, _ in polys]
        raise SystemExit(f"No layers matched {args.layers}. Available: {names}")

    overlay = frame.copy()
    for i, (name, poly) in enumerate(selected):
        c = _color(name, i)
        cv2.fillPoly(overlay, [poly], c)
        cv2.polylines(overlay, [poly], True, c, 4)
        cx, cy = poly[:, 0, :].mean(axis=0).astype(int)
        cv2.putText(
            overlay,
            name,
            (int(cx) - 20, int(cy)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            3,
        )
        cv2.putText(
            overlay,
            name,
            (int(cx) - 20, int(cy)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            c,
            2,
        )
        for j, (x, y) in enumerate(poly.reshape(-1, 2)):
            cv2.circle(overlay, (int(x), int(y)), 8, (0, 255, 255), -1)
            cv2.putText(
                overlay,
                str(j),
                (int(x) + 10, int(y) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

    out = args.output
    if out is None:
        vid = meta.get("video_id", roi_path.stem)
        layer_tag = "_".join(args.layers)
        out = roi_dir / "reference_frames" / f"{vid}_{layer_tag}_preview.png"
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    blend = cv2.addWeighted(frame, 1.0 - args.alpha, overlay, args.alpha, 0)
    cv2.imwrite(str(out), blend)
    print(f"Wrote {out}")
    print(f"Image {w}x{h}, scale {info['scale']}, layers: {[n for n, _ in selected]}")


if __name__ == "__main__":
    main()
