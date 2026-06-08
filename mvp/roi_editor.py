#!/usr/bin/env python3
"""Interactive local editor for lake / tree ROI polygons on a video frame.

Use when the camera drifts so east.json / west.json no longer line up.

Example (on laptop with synced video):
  cd depotData/d-park-social-performance-evaluation
  .venv/bin/python mvp/roi_editor.py \\
    --video ../depotData/west/GX020307.MP4 --camera west --timestamp 300

Controls:
  Left-drag     move nearest vertex (current layer)
  Arrow keys    nudge current layer (lake or one tree)
  Shift+arrows  nudge ALL lake+tree layers together (camera shift)
  n / p         next / previous layer
  r             reset from camera default ROI
  s             save to annotations/roi/per_video/<video_id>.json
  q             quit (prompts if unsaved)
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np

from roi_utils import polys_to_frame_points, resolve_roi_for_video, write_roi_json

ROOT = Path(__file__).resolve().parents[1]
_COLORS = {
    "lake": (0, 140, 255),
}
_TREE_PALETTE = [(0, 200, 80), (0, 180, 120), (80, 220, 80), (0, 160, 60), (120, 255, 120)]


def _color(name: str, index: int) -> tuple[int, int, int]:
    if name in _COLORS:
        return _COLORS[name]
    if name.startswith("tree"):
        return _TREE_PALETTE[index % len(_TREE_PALETTE)]
    return (200, 200, 0)


def grab_frame(video_path: Path, timestamp_sec: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp_sec) * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise SystemExit(f"Cannot read frame at t={timestamp_sec}s from {video_path}")
    return frame


class ROIEditor:
    def __init__(
        self,
        frame: np.ndarray,
        polygons: dict[str, list[list[int]]],
        *,
        camera_id: str,
        video_id: str,
        out_path: Path,
        display_max_w: int = 1280,
    ) -> None:
        self.frame = frame
        self.h, self.w = frame.shape[:2]
        self.polygons = deepcopy(polygons)
        self._base = deepcopy(polygons)
        self.camera_id = camera_id
        self.video_id = video_id
        self.out_path = out_path
        self.layers = sorted(
            self.polygons.keys(), key=lambda n: (0 if n == "lake" else 1, n)
        )
        if not self.layers:
            raise SystemExit("No lake/tree polygons to edit in base ROI")
        self.layer_i = 0
        self.dirty = False
        self.drag_vertex: int | None = None
        self.scale = min(1.0, display_max_w / self.w)
        self.disp_w = int(self.w * self.scale)
        self.disp_h = int(self.h * self.scale)
        self.win = f"ROI editor — {video_id} (q=quit s=save)"

    def _disp(self, pt: list[int]) -> tuple[int, int]:
        return int(pt[0] * self.scale), int(pt[1] * self.scale)

    def _frame_pt(self, x: int, y: int) -> list[int]:
        return [
            int(np.clip(round(x / self.scale), 0, self.w - 1)),
            int(np.clip(round(y / self.scale), 0, self.h - 1)),
        ]

    def _current_layer(self) -> str:
        return self.layers[self.layer_i]

    def _nudge(self, dx: int, dy: int, all_layers: bool) -> None:
        names = self.layers if all_layers else [self._current_layer()]
        for name in names:
            self.polygons[name] = [[p[0] + dx, p[1] + dy] for p in self.polygons[name]]
        self.dirty = True

    def _nearest_vertex(self, x: int, y: int, radius: int = 12) -> int | None:
        layer = self._current_layer()
        best, best_d = None, radius * radius
        for i, pt in enumerate(self.polygons[layer]):
            dx, dy = self._disp(pt)
            d = (dx - x) ** 2 + (dy - y) ** 2
            if d <= best_d:
                best, best_d = i, d
        return best

    def _draw(self) -> np.ndarray:
        canvas = cv2.resize(self.frame, (self.disp_w, self.disp_h))
        overlay = canvas.copy()
        cur = self._current_layer()
        for i, name in enumerate(self.layers):
            pts = np.array([self._disp(p) for p in self.polygons[name]], dtype=np.int32)
            color = _color(name, i)
            active = name == cur
            thick = 3 if active else 2
            cv2.polylines(overlay, [pts], True, color, thick)
            cv2.fillPoly(overlay, [pts], color)
            for j, (x, y) in enumerate(pts):
                r = 7 if active and j == self.drag_vertex else 5
                cv2.circle(overlay, (int(x), int(y)), r, (0, 255, 255), -1)
            cx, cy = pts.mean(axis=0).astype(int)
            cv2.putText(
                overlay, name, (int(cx), int(cy)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            )
        blend = cv2.addWeighted(canvas, 0.55, overlay, 0.45, 0)
        help1 = f"Layer [{self.layer_i + 1}/{len(self.layers)}]: {cur}  |  dirty={self.dirty}"
        help2 = "drag=vertex  arrows=layer  Shift+arrows=all  n/p=layer  r=reset  s=save  q=quit"
        cv2.putText(blend, help1, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
        cv2.putText(blend, help2, (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        return blend

    def _save(self) -> None:
        write_roi_json(
            self.out_path,
            camera_id=self.camera_id,
            video_id=self.video_id,
            frame_size=(self.w, self.h),
            polygons=self.polygons,
            notes=f"Interactive alignment for {self.video_id}",
        )
        self.dirty = False
        print(f"Saved {self.out_path}")

    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        layer = self._current_layer()
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_vertex = self._nearest_vertex(x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_vertex is not None:
            self.polygons[layer][self.drag_vertex] = self._frame_pt(x, y)
            self.dirty = True
        elif event == cv2.EVENT_LBUTTONUP:
            self.drag_vertex = None

    def run(self) -> None:
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win, self.disp_w, self.disp_h)
        cv2.setMouseCallback(self.win, self._on_mouse)
        step = max(1, int(round(4 / self.scale)))
        while True:
            cv2.imshow(self.win, self._draw())
            key = cv2.waitKey(20) & 0xFF
            if key == ord("q"):
                if self.dirty:
                    print("Unsaved changes — press s to save or q again to discard.")
                    key2 = cv2.waitKey(0) & 0xFF
                    if key2 == ord("s"):
                        self._save()
                break
            if key == ord("s"):
                self._save()
            elif key == ord("r"):
                self.polygons = deepcopy(self._base)
                self.dirty = True
            elif key == ord("n"):
                self.layer_i = (self.layer_i + 1) % len(self.layers)
            elif key == ord("p"):
                self.layer_i = (self.layer_i - 1) % len(self.layers)
            elif key in (81, 2424832):  # left
                self._nudge(-step, 0, False)
            elif key in (83, 2555904):  # right
                self._nudge(step, 0, False)
            elif key in (82, 2490368):  # up
                self._nudge(0, -step, False)
            elif key in (84, 2621440):  # down
                self._nudge(0, step, False)
            elif key == ord("a"):
                self._nudge(-step, 0, True)
            elif key == ord("d"):
                self._nudge(step, 0, True)
            elif key == ord("w"):
                self._nudge(0, -step, True)
            elif key == ord("x"):
                self._nudge(0, step, True)
        cv2.destroyAllWindows()


def main() -> None:
    p = argparse.ArgumentParser(description="Interactive lake/tree ROI editor (local GUI)")
    p.add_argument("--video", type=Path, required=True, help="Source MP4 for this clip")
    p.add_argument("--camera", choices=["east", "west"], required=True)
    p.add_argument("--timestamp", type=float, default=300.0, help="Frame time in seconds")
    p.add_argument(
        "--base-roi",
        type=Path,
        default=None,
        help="Starting ROI (default: camera east.json / west.json)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save path (default: annotations/roi/per_video/<video_id>.json)",
    )
    p.add_argument("--display-width", type=int, default=1280)
    args = p.parse_args()

    video_path = args.video.resolve()
    video_id = video_path.stem
    base_roi = args.base_roi or (ROOT / "annotations" / "roi" / f"{args.camera}.json")
    out_path = args.output or (ROOT / "annotations" / "roi" / "per_video" / f"{video_id}.json")

    frame = grab_frame(video_path, args.timestamp)
    h, w = frame.shape[:2]
    polygons = polys_to_frame_points(base_roi.resolve(), (w, h))
    if not polygons:
        raise SystemExit(f"No editable polygons in {base_roi}")

    print(f"Video: {video_path} ({w}x{h}) @ t={args.timestamp}s")
    print(f"Base ROI: {base_roi}")
    print(f"Layers: {', '.join(sorted(polygons))}")
    print(f"Save to: {out_path}")

    ROIEditor(
        frame,
        polygons,
        camera_id=args.camera,
        video_id=video_id,
        out_path=out_path.resolve(),
        display_max_w=args.display_width,
    ).run()


if __name__ == "__main__":
    main()
