#!/usr/bin/env python3
"""Regenerate events.csv and hourly_metrics.csv from tracks.csv (after ROI / zone updates)."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from apply_roi_to_outputs import apply as apply_roi
from run_poc import build_events_df, classify_activity


def _mean_dist(hdf: pd.DataFrame, col: str) -> float | str:
    if col not in hdf.columns:
        return ""
    vals = pd.to_numeric(hdf[col], errors="coerce").dropna()
    if vals.empty:
        return ""
    return round(float(vals.mean()), 1)


def rebuild_hourly(
    tracks: pd.DataFrame,
    events: pd.DataFrame,
    video_id: str,
) -> pd.DataFrame:
    persons = tracks[tracks["class"] == "person"].copy()
    if persons.empty:
        return pd.DataFrame()

    if "in_park" in persons.columns:
        persons = persons[persons["in_park"] == True]  # noqa: E712

    dwell_by_tid = persons.groupby("track_id")["timestamp"].agg(["min", "max"])
    dwell_by_tid["dwell_s"] = dwell_by_tid["max"] - dwell_by_tid["min"]

    lake_visitors = set(persons.loc[persons["near_lake"], "track_id"].astype(int).unique())
    tree_visitors = set(persons.loc[persons["near_tree"], "track_id"].astype(int).unique())

    tracks = tracks.copy()
    tracks["hour"] = (tracks["timestamp"] // 3600).astype(int)
    rows: list[dict] = []
    for hour, hdf in tracks.groupby("hour", sort=True):
        hdf_p = hdf[hdf["class"] == "person"]
        if "in_park" in hdf_p.columns:
            hdf_p = hdf_p[hdf_p["in_park"] == True]  # noqa: E712
        hdf_b = hdf[hdf["class"].isin(["bicycle", "cyclist"])]
        if "in_park" in hdf_b.columns:
            hdf_b = hdf_b[hdf_b["in_park"] == True]  # noqa: E712
        pids = set(hdf_p["track_id"].astype(int))
        bids = set(hdf_b["track_id"].astype(int))

        dwell = [
            float(dwell_by_tid.loc[tid, "dwell_s"])
            for tid in pids
            if tid in dwell_by_tid.index
        ]
        grouped = {
            tid
            for tid in pids
            if (hdf_p[hdf_p["track_id"] == tid]["group_id"].astype(str) != "").any()
            and (hdf_p[hdf_p["track_id"] == tid]["group_id"].astype(str) != "nan").any()
        }

        hour_events = events[
            (events["start_time"] // 3600 <= hour) & (events["end_time"] // 3600 >= hour)
        ]
        act_div = int(hour_events["activity_label"].nunique()) if len(hour_events) else 0

        gdf = hdf_p[hdf_p["group_id"].astype(str).replace("nan", "") != ""]
        sizes = (
            gdf.groupby(["timestamp", "group_id"])["track_id"].nunique()
            if len(gdf)
            else pd.Series(dtype=float)
        )

        lake_p = lake_visitors & pids
        tree_p = tree_visitors & pids

        rows.append(
            {
                "hour": int(hour),
                "pedestrian_count": len(pids),
                "biker_count": len(bids),
                "visitor_count": len(pids),
                "median_dwell_time": float(np.median(dwell)) if dwell else 0.0,
                "total_visitor_minutes": float(sum(dwell) / 60.0),
                "pct_near_tree": round(100.0 * len(tree_p) / len(pids), 2) if pids else 0.0,
                "pct_near_lake": round(100.0 * len(lake_p) / len(pids), 2) if pids else 0.0,
                "avg_distance_to_tree": _mean_dist(hdf_p, "dist_tree"),
                "avg_distance_to_lake": _mean_dist(hdf_p, "dist_lake"),
                "activity_diversity": act_div,
                "group_visitor_pct": round(100.0 * len(grouped) / len(pids), 2) if pids else 0.0,
                "mean_group_size": round(float(sizes.mean()), 2) if len(sizes) else 0.0,
                "apparent_child_pct": "",
            }
        )
    return pd.DataFrame(rows)


def regenerate(
    output_dir: Path,
    roi_path: Path,
    video_id: str,
    video_size: tuple[int, int] = (3840, 2160),
) -> None:
    apply_roi(output_dir, roi_path, video_size)

    tracks = pd.read_csv(output_dir / "tracks.csv")
    tracks["activity_label"] = tracks.apply(
        lambda r: classify_activity(str(r["class"]), float(r["speed"]), 8, 8, 45),
        axis=1,
    )
    tracks.to_csv(output_dir / "tracks.csv", index=False)

    events = build_events_df(tracks, video_id)
    events.to_csv(output_dir / "events.csv", index=False)

    hourly = rebuild_hourly(tracks, events, video_id)
    hourly.to_csv(output_dir / "hourly_metrics.csv", index=False)

    print(f"{output_dir}: events={len(events)}, hourly rows={len(hourly)}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--roi", type=Path, required=True)
    p.add_argument("--video-id", required=True)
    args = p.parse_args()
    regenerate(args.output_dir, args.roi, args.video_id)


if __name__ == "__main__":
    main()
