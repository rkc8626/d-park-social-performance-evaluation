#!/usr/bin/env python3
"""
Minimum viable park-video analytics POC.

Uses off-the-shelf YOLOv8 + ByteTrack to validate that a fixed camera clip is
usable before investing in full ROI annotation and custom models.

Outputs (under output_dir / video_id):
  - videos.csv, tracks.csv, events.csv, hourly_metrics.csv
  - video_quality_report.json, poc_summary.md
  - preview_annotated.mp4  (optional short clip)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Keep Ultralytics/torch caches off $HOME when home quota is tight (HPC).
_CACHE_ROOT = Path(__file__).resolve().parent / ".cache"
if "YOLO_CONFIG_DIR" not in os.environ:
    os.environ["YOLO_CONFIG_DIR"] = str(_CACHE_ROOT / "ultralytics")
if "TORCH_HOME" not in os.environ:
    os.environ["TORCH_HOME"] = str(_CACHE_ROOT / "torch")
if "XDG_CACHE_HOME" not in os.environ:
    os.environ["XDG_CACHE_HOME"] = str(_CACHE_ROOT / "xdg")

import cv2
import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO


@dataclass
class TrackState:
    class_name: str
    first_t: float = math.inf
    last_t: float = 0.0
    speeds: list[float] = field(default_factory=list)
    activities: list[str] = field(default_factory=list)
    centroids: list[tuple[float, float, float]] = field(default_factory=list)  # t, x, y


def resolve_device(cfg: dict[str, Any]) -> str | int:
    """Use GPU when available unless config forces 'cpu'."""
    forced = cfg.get("device")
    if forced is not None:
        return forced
    try:
        import torch

        if torch.cuda.is_available():
            return 0
    except ImportError:
        pass
    return "cpu"


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        cfg = yaml.safe_load(f)
    base = path.parent.resolve()
    for key in ("video_path", "output_dir"):
        if key in cfg and not Path(cfg[key]).is_absolute():
            cfg[key] = str((base / cfg[key]).resolve())
    return cfg


def bbox_centroid(box: np.ndarray) -> tuple[float, float]:
    x1, y1, x2, y2 = box[:4]
    return float((x1 + x2) / 2), float((y1 + y2) / 2)


def speed_px_per_s(
    prev: tuple[float, float, float], cur: tuple[float, float, float]
) -> float:
    t0, x0, y0 = prev
    t1, x1, y1 = cur
    dt = t1 - t0
    if dt <= 0:
        return 0.0
    return math.hypot(x1 - x0, y1 - y0) / dt


def classify_activity(
    class_name: str,
    speed: float,
    standing_max: float,
    walking_min: float,
    running_min: float,
) -> str:
    if class_name in ("bicycle", "cyclist"):
        return "biking"
    if speed < standing_max:
        return "standing"
    if speed >= running_min:
        return "running"
    if speed >= walking_min:
        return "walking"
    return "standing"


def assign_groups(
    people: list[tuple[int, float, float]],
    max_dist: float,
) -> dict[int, int]:
    """Greedy spatial clustering; returns track_id -> group_id (-1 if solo)."""
    if len(people) < 2:
        return {tid: -1 for tid, _, _ in people}

    parent = {tid: tid for tid, _, _ in people}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    coords = {tid: (x, y) for tid, x, y in people}
    ids = list(coords)
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            ax, ay = coords[a]
            bx, by = coords[b]
            if math.hypot(ax - bx, ay - by) <= max_dist:
                union(a, b)

    groups: dict[int, list[int]] = defaultdict(list)
    for tid in ids:
        groups[find(tid)].append(tid)

    out: dict[int, int] = {}
    gid = 0
    for members in groups.values():
        if len(members) < 2:
            for tid in members:
                out[tid] = -1
        else:
            for tid in members:
                out[tid] = gid
            gid += 1
    return out


def resolve_max_seconds(cfg: dict[str, Any], duration_sec: float) -> float:
    val = cfg.get("max_seconds")
    if val is None or val == 0:
        return duration_sec
    return float(val)


def build_videos_row(cfg: dict[str, Any], probe: dict[str, Any]) -> dict[str, Any]:
    vid = cfg.get("video_id", "unknown")
    return {
        "video_id": vid,
        "camera_id": cfg.get("camera_id", ""),
        "start_time": cfg.get("start_time", ""),
        "end_time": cfg.get("end_time", ""),
        "fps": round(probe["fps"], 4),
        "resolution": f"{probe['width']}x{probe['height']}",
        "location": cfg.get("location", cfg.get("camera_id", "")),
        "notes": cfg.get("notes", "MVP: YOLOv8n + ByteTrack; no ROI polygons"),
    }


def build_events_df(detections: pd.DataFrame, video_id: str, gap_sec: float = 2.0) -> pd.DataFrame:
    cols = [
        "video_id",
        "track_id",
        "start_time",
        "end_time",
        "activity_label",
        "zone",
        "group_id",
    ]
    if detections.empty:
        return pd.DataFrame(columns=cols)

    rows: list[dict[str, Any]] = []
    for tid, g in detections.groupby("track_id", sort=False):
        g = g.sort_values("timestamp")
        seg_start = float(g.iloc[0]["timestamp"])
        seg_end = seg_start
        act = g.iloc[0]["activity_label"]
        zone = g.iloc[0]["zone"]
        gid = g.iloc[0]["group_id"]
        for i in range(1, len(g)):
            row = g.iloc[i]
            t = float(row["timestamp"])
            gap = t - float(g.iloc[i - 1]["timestamp"])
            if row["activity_label"] != act or gap > gap_sec:
                rows.append(
                    {
                        "video_id": video_id,
                        "track_id": int(tid),
                        "start_time": round(seg_start, 3),
                        "end_time": round(seg_end, 3),
                        "activity_label": act,
                        "zone": zone,
                        "group_id": gid,
                    }
                )
                seg_start = t
                act = row["activity_label"]
                zone = row["zone"]
                gid = row["group_id"]
            seg_end = t
        rows.append(
            {
                "video_id": video_id,
                "track_id": int(tid),
                "start_time": round(seg_start, 3),
                "end_time": round(seg_end, 3),
                "activity_label": act,
                "zone": zone,
                "group_id": gid,
            }
        )
    return pd.DataFrame(rows)


def build_hourly_metrics_df(
    detections: pd.DataFrame,
    track_states: dict[int, TrackState],
    events: pd.DataFrame,
    video_id: str,
) -> pd.DataFrame:
    cols = [
        "hour",
        "pedestrian_count",
        "biker_count",
        "visitor_count",
        "median_dwell_time",
        "total_visitor_minutes",
        "pct_near_tree",
        "pct_near_lake",
        "avg_distance_to_tree",
        "avg_distance_to_lake",
        "activity_diversity",
        "group_visitor_pct",
        "mean_group_size",
        "apparent_child_pct",
    ]
    if detections.empty:
        return pd.DataFrame(columns=cols)

    det = detections.copy()
    det["hour"] = (det["timestamp"] // 3600).astype(int)
    rows: list[dict[str, Any]] = []

    for hour, hdf in det.groupby("hour", sort=True):
        persons = hdf[hdf["class"] == "person"]
        bikes = hdf[hdf["class"].isin(["bicycle", "cyclist"])]
        person_ids = set(persons["track_id"].unique())
        bike_ids = set(bikes["track_id"].unique())

        dwell = [
            track_states[tid].last_t - track_states[tid].first_t
            for tid in person_ids
            if tid in track_states and math.isfinite(track_states[tid].first_t)
        ]
        grouped = set()
        for tid in person_ids:
            sub = persons[persons["track_id"] == tid]
            if (sub["group_id"] != "").any():
                grouped.add(tid)

        hour_events = events[events["video_id"] == video_id] if not events.empty else events
        if not hour_events.empty:
            in_hour = hour_events[
                (hour_events["start_time"] // 3600 <= hour)
                & (hour_events["end_time"] // 3600 >= hour)
            ]
            act_div = in_hour["activity_label"].nunique() if len(in_hour) else 0
        else:
            act_div = hdf["activity_label"].nunique()

        sizes = (
            hdf[hdf["group_id"] != ""].groupby(["timestamp", "group_id"])["track_id"].nunique()
        )
        mean_grp = float(sizes.mean()) if len(sizes) else 0.0

        rows.append(
            {
                "hour": int(hour),
                "pedestrian_count": len(person_ids),
                "biker_count": len(bike_ids),
                "visitor_count": len(person_ids),
                "median_dwell_time": float(np.median(dwell)) if dwell else 0.0,
                "total_visitor_minutes": float(sum(dwell) / 60.0),
                "pct_near_tree": "",
                "pct_near_lake": "",
                "avg_distance_to_tree": "",
                "avg_distance_to_lake": "",
                "activity_diversity": int(act_div),
                "group_visitor_pct": (
                    round(100.0 * len(grouped) / len(person_ids), 2) if person_ids else 0.0
                ),
                "mean_group_size": round(mean_grp, 2),
                "apparent_child_pct": "",
            }
        )
    return pd.DataFrame(rows)


def video_probe(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return {
        "path": str(path),
        "fps": fps,
        "width": w,
        "height": h,
        "frame_count": n,
        "duration_sec": n / fps if fps else 0,
    }


def run_poc(cfg: dict[str, Any]) -> Path:
    video_path = Path(cfg["video_path"])
    out_root = Path(cfg["output_dir"]) / cfg.get("video_id", video_path.stem)
    out_root.mkdir(parents=True, exist_ok=True)

    probe = video_probe(video_path)
    max_seconds = resolve_max_seconds(cfg, probe["duration_sec"])
    stride = int(cfg.get("frame_stride", 4))
    imgsz = int(cfg.get("imgsz", 1280))
    conf = float(cfg.get("conf", 0.25))
    preview_seconds = int(cfg.get("preview_seconds", 30))

    class_map = cfg.get("classes", {"person": 0, "bicycle": 1})
    id_to_name = {v: k for k, v in class_map.items()}
    target_ids = set(class_map.values())

    standing_max = float(cfg.get("speed_standing_max", 8))
    walking_min = float(cfg.get("speed_walking_min", 8))
    running_min = float(cfg.get("speed_running_min", 45))
    group_dist = float(cfg.get("group_distance_px", 120))

    device = resolve_device(cfg)
    model = YOLO(cfg.get("model", "yolov8n.pt"))
    tracker = cfg.get("tracker", "bytetrack.yaml")
    print(f"Inference device: {device}")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or probe["fps"]
    max_frame = int(min(probe["frame_count"], max_seconds * fps))

    writer = None
    # preview_seconds: 0=off, >0=first N seconds, -1=full processed window
    if preview_seconds < 0:
        preview_max_frame = max_frame
    elif preview_seconds > 0:
        preview_max_frame = int(preview_seconds * fps)
    else:
        preview_max_frame = 0

    track_rows: list[dict[str, Any]] = []
    track_states: dict[int, TrackState] = {}
    frames_processed = 0
    frames_with_detection = 0
    total_detections = 0
    frame_idx = 0
    while frame_idx < max_frame:
        ok, frame = cap.read()
        if not ok:
            break
        t_sec = frame_idx / fps

        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        results = model.track(
            frame,
            persist=True,
            tracker=tracker,
            conf=conf,
            imgsz=imgsz,
            classes=list(target_ids),
            device=device,
            verbose=False,
        )[0]

        frames_processed += 1
        dets_this_frame = 0
        people_for_groups: list[tuple[int, float, float]] = []

        boxes = results.boxes
        if boxes is not None and len(boxes):
            frames_with_detection += 1
            for box in boxes:
                cls_id = int(box.cls.item())
                if cls_id not in target_ids:
                    continue
                conf_v = float(box.conf.item())
                xyxy = box.xyxy.cpu().numpy().reshape(-1)
                class_name = id_to_name.get(cls_id, str(cls_id))
                if box.id is None:
                    continue
                tid = int(box.id.item())
                cx, cy = bbox_centroid(xyxy)
                dets_this_frame += 1
                total_detections += 1

                if class_name == "person":
                    people_for_groups.append((tid, cx, cy))

                st = track_states.get(tid)
                if st is None:
                    st = TrackState(class_name=class_name)
                    track_states[tid] = st
                st.last_t = t_sec
                st.first_t = min(st.first_t, t_sec)

                centroid = (t_sec, cx, cy)
                speed = 0.0
                if st.centroids:
                    speed = speed_px_per_s(st.centroids[-1], centroid)
                st.centroids.append(centroid)
                st.speeds.append(speed)
                act = classify_activity(
                    class_name, speed, standing_max, walking_min, running_min
                )
                st.activities.append(act)

        group_map = assign_groups(people_for_groups, group_dist)

        if boxes is not None and len(boxes):
            for box in boxes:
                if box.id is None:
                    continue
                tid = int(box.id.item())
                cls_id = int(box.cls.item())
                if cls_id not in target_ids:
                    continue
                xyxy = box.xyxy.cpu().numpy().reshape(-1)
                class_name = id_to_name.get(cls_id, str(cls_id))
                gid = group_map.get(tid, -1)
                speed = track_states[tid].speeds[-1] if track_states[tid].speeds else 0.0
                act = track_states[tid].activities[-1] if track_states[tid].activities else ""
                track_rows.append(
                    {
                        "video_id": cfg.get("video_id", video_path.stem),
                        "timestamp": round(t_sec, 3),
                        "track_id": tid,
                        "class": class_name,
                        "bbox_x1": float(xyxy[0]),
                        "bbox_y1": float(xyxy[1]),
                        "bbox_x2": float(xyxy[2]),
                        "bbox_y2": float(xyxy[3]),
                        "zone": "full_frame_poc",
                        "speed": round(speed, 2),
                        "activity_label": act,
                        "group_id": gid if gid >= 0 else "",
                        "confidence": round(float(box.conf.item()), 4),
                    }
                )

        # Annotated preview (full window when preview_seconds == -1)
        if preview_max_frame > 0 and frame_idx < preview_max_frame:
            ann = results.plot()
            if writer is None:
                h, w = ann.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    str(out_root / "preview_annotated.mp4"),
                    fourcc,
                    fps / stride,
                    (w, h),
                )
            writer.write(ann)

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    video_id = cfg.get("video_id", video_path.stem)
    detections_df = pd.DataFrame(track_rows)

    track_cols = [
        "video_id",
        "timestamp",
        "track_id",
        "class",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "zone",
        "speed",
        "apparent_age_group",
        "group_id",
        "confidence",
    ]
    if not detections_df.empty:
        tracks_df = detections_df.assign(apparent_age_group="uncertain")[track_cols]
    else:
        tracks_df = pd.DataFrame(columns=track_cols)

    events_df = build_events_df(detections_df, video_id)
    hourly_df = build_hourly_metrics_df(detections_df, track_states, events_df, video_id)
    videos_df = pd.DataFrame([build_videos_row(cfg, probe)])

    videos_path = out_root / "videos.csv"
    tracks_path = out_root / "tracks.csv"
    events_path = out_root / "events.csv"
    metrics_path = out_root / "hourly_metrics.csv"
    videos_df.to_csv(videos_path, index=False)
    tracks_df.to_csv(tracks_path, index=False)
    events_df.to_csv(events_path, index=False)
    hourly_df.to_csv(metrics_path, index=False)

    unique_person = {
        tid for tid, st in track_states.items() if st.class_name == "person"
    }
    unique_bike = {
        tid
        for tid, st in track_states.items()
        if st.class_name in ("bicycle", "cyclist")
    }
    dwell = [
        st.last_t - st.first_t
        for st in track_states.values()
        if st.class_name == "person" and math.isfinite(st.first_t)
    ]
    activity_counts = (
        detections_df["activity_label"].value_counts().to_dict()
        if not detections_df.empty
        else {}
    )

    det_rate = (
        frames_with_detection / frames_processed if frames_processed else 0.0
    )
    quality = {
        **probe,
        "processed_seconds": min(max_seconds, probe["duration_sec"]),
        "frame_stride": stride,
        "frames_processed": frames_processed,
        "frames_with_detection": frames_with_detection,
        "detection_frame_rate": round(det_rate, 4),
        "total_detections": total_detections,
        "unique_person_tracks": len(unique_person),
        "unique_bicycle_tracks": len(unique_bike),
        "model": cfg.get("model"),
        "tracker": tracker,
        "device": str(device),
        "imgsz": imgsz,
        "usable_for_full_pipeline": det_rate >= 0.15 and len(unique_person) >= 1,
        "warnings": [],
    }
    if probe["width"] >= 3840:
        quality["warnings"].append("4K source — consider fixed imgsz and per-camera calibration")
    if det_rate < 0.15:
        quality["warnings"].append("Low detection rate; check angle, lighting, or confidence threshold")
    if len(unique_person) == 0 and len(unique_bike) == 0:
        quality["warnings"].append("No person/bicycle tracks — video may be unsuitable or scene is empty")

    quality_path = out_root / "video_quality_report.json"
    with quality_path.open("w") as f:
        json.dump(quality, f, indent=2)

    summary_lines = [
        f"# POC summary — {cfg.get('video_id', video_path.stem)}",
        "",
        f"- **Source:** `{video_path}`",
        f"- **Resolution / duration:** {probe['width']}x{probe['height']}, {probe['duration_sec']:.1f}s",
        f"- **Processed window:** first {max_seconds}s, stride={stride}",
        f"- **Detection frame rate:** {det_rate:.1%} ({frames_with_detection}/{frames_processed} sampled frames)",
        f"- **Unique person tracks:** {len(unique_person)}",
        f"- **Unique bicycle tracks:** {len(unique_bike)}",
        f"- **Usable for full pipeline (heuristic):** {quality['usable_for_full_pipeline']}",
        "",
        "## Activity distribution (rule-based proxy)",
        "",
    ]
    for act, cnt in sorted(activity_counts.items(), key=lambda x: -x[1]):
        summary_lines.append(f"- {act}: {cnt}")
    if quality["warnings"]:
        summary_lines.extend(["", "## Warnings", ""])
        for w in quality["warnings"]:
            summary_lines.append(f"- {w}")

    summary_path = out_root / "poc_summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n")

    print(f"Wrote {videos_path}")
    print(f"Wrote {tracks_path} ({len(tracks_df)} rows)")
    print(f"Wrote {events_path} ({len(events_df)} rows)")
    print(f"Wrote {metrics_path} ({len(hourly_df)} hour buckets)")
    print(f"Wrote {quality_path}")
    print(f"Usable for full pipeline: {quality['usable_for_full_pipeline']}")
    return out_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Run park video MVP POC")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.yaml"),
        help="YAML config path",
    )
    parser.add_argument("--video", type=Path, default=None, help="Override video path")
    parser.add_argument("--video-id", default=None, help="Override video_id (default: video stem)")
    parser.add_argument("--camera-id", default=None, help="Override camera_id")
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.video:
        cfg["video_path"] = str(args.video.resolve())
        cfg["video_id"] = args.video_id or Path(cfg["video_path"]).stem
    if args.video_id:
        cfg["video_id"] = args.video_id
    if args.camera_id:
        cfg["camera_id"] = args.camera_id
    if args.max_seconds is not None:
        cfg["max_seconds"] = args.max_seconds
    if args.output_dir:
        cfg["output_dir"] = str(args.output_dir.resolve())

    if not Path(cfg["video_path"]).is_file():
        print(f"Video not found: {cfg['video_path']}", file=sys.stderr)
        sys.exit(1)

    run_poc(cfg)


if __name__ == "__main__":
    main()
