"""Load ROI polygons and test point-in-polygon (with optional resolution scaling)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_META = {"camera_id", "video_id", "reference_frame", "resolution", "notes"}


def _parse_resolution(s: str) -> tuple[int, int] | None:
    if not s or "x" not in str(s).lower():
        return None
    w, h = str(s).lower().split("x", 1)
    return int(w), int(h)


def _extract_points(obj: object) -> list[list[float]] | None:
    if isinstance(obj, dict) and "points" in obj:
        return _extract_points(obj["points"])
    if isinstance(obj, list):
        if len(obj) == 1 and isinstance(obj[0], dict):
            return _extract_points(obj[0])
        if len(obj) >= 3 and all(
            isinstance(p, (list, tuple)) and len(p) >= 2 for p in obj
        ):
            return [[float(p[0]), float(p[1])] for p in obj]
    return None


def load_roi_polygons(
    roi_path: Path,
    target_size: tuple[int, int] | None = None,
) -> tuple[list[tuple[str, np.ndarray]], dict[str, object]]:
    """
    Returns (polygons, meta). Polygons are OpenCV-ready Nx1x2 int32 arrays in target_size space.
    """
    data = json.loads(roi_path.read_text())
    ann_size = _parse_resolution(data.get("resolution", ""))
    scale_x = scale_y = 1.0
    if ann_size and target_size:
        scale_x = target_size[0] / ann_size[0]
        scale_y = target_size[1] / ann_size[1]

    polys: list[tuple[str, np.ndarray]] = []
    for name, raw in data.items():
        if name.startswith("_") or name in _META:
            continue
        pts = _extract_points(raw)
        if not pts or len(pts) < 3:
            continue
        scaled = np.array(
            [[int(round(p[0] * scale_x)), int(round(p[1] * scale_y))] for p in pts],
            dtype=np.int32,
        ).reshape(-1, 1, 2)
        polys.append((name, scaled))

    meta = {
        "annotation_resolution": ann_size,
        "target_resolution": target_size,
        "scale": (scale_x, scale_y),
    }
    return polys, meta


def centroid_in_poly(cx: float, cy: float, poly: np.ndarray) -> bool:
    import cv2

    return cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0


def classify_centroid(
    cx: float,
    cy: float,
    polys: list[tuple[str, np.ndarray]],
) -> tuple[bool, bool, str]:
    """Returns (near_lake, near_any_tree, zone_label)."""
    near_lake = False
    near_tree = False
    zone = "other"
    for name, poly in polys:
        if not centroid_in_poly(cx, cy, poly):
            continue
        if name == "lake":
            near_lake = True
            zone = "lake"
        elif name.startswith("tree"):
            near_tree = True
            if zone in ("other", "park"):
                zone = name
        elif name == "park":
            if zone == "other":
                zone = "park"
    return near_lake, near_tree, zone


def min_distance_to_poly(cx: float, cy: float, poly: np.ndarray) -> float:
    import cv2

    d = cv2.pointPolygonTest(poly, (float(cx), float(cy)), True)
    return 0.0 if d >= 0 else abs(float(d))
