#!/usr/bin/env python3
"""Apply ROI labels to existing tracks.csv and refresh hourly_metrics.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from roi_utils import classify_centroid, load_roi_polygons, min_distance_to_poly


def apply(output_dir: Path, roi_path: Path, video_size: tuple[int, int] = (3840, 2160)) -> None:
    tracks_path = output_dir / "tracks.csv"
    if not tracks_path.is_file():
        raise SystemExit(f"Missing {tracks_path}")

    polys, meta = load_roi_polygons(roi_path, target_size=video_size)
    if not polys:
        raise SystemExit(f"No polygons in {roi_path}")
    print(f"Loaded {len(polys)} polygons from {roi_path} (scale {meta['scale']})")

    df = pd.read_csv(tracks_path)
    lake_polys = [p for n, p in polys if n == "lake"]
    tree_polys = [p for n, p in polys if n.startswith("tree")]

    zones, near_lake, near_tree, d_lake, d_tree = [], [], [], [], []
    for _, row in df.iterrows():
        cx = (row.bbox_x1 + row.bbox_x2) / 2
        cy = (row.bbox_y1 + row.bbox_y2) / 2
        nl, nt, zone = classify_centroid(cx, cy, polys)
        zones.append(zone)
        near_lake.append(nl)
        near_tree.append(nt)
        d_lake.append(
            min(min_distance_to_poly(cx, cy, p) for p in lake_polys) if lake_polys else np.nan
        )
        d_tree.append(
            min(min_distance_to_poly(cx, cy, p) for p in tree_polys) if tree_polys else np.nan
        )

    df["zone"] = zones
    df["near_lake"] = near_lake
    df["near_tree"] = near_tree
    df.to_csv(tracks_path, index=False)

    persons = df[df["class"] == "person"].copy()
    person_ids = persons["track_id"].unique()

    lake_visitors = set(
        persons.loc[persons["near_lake"], "track_id"].astype(int).unique()
    )
    tree_visitors = set(
        persons.loc[persons["near_tree"], "track_id"].astype(int).unique()
    )

    persons["hour"] = (persons["timestamp"] // 3600).astype(int)
    hm_path = output_dir / "hourly_metrics.csv"
    old_hm = pd.read_csv(hm_path) if hm_path.is_file() else pd.DataFrame()

    roi_by_hour = []
    for hour, hdf in persons.groupby("hour", sort=True):
        pids = set(hdf["track_id"].astype(int))
        lake_p = lake_visitors & pids
        tree_p = tree_visitors & pids
        idx = hdf.index.tolist()
        tree_dists = [d_tree[i] for i in idx if np.isfinite(d_tree[i])]
        lake_dists = [d_lake[i] for i in idx if np.isfinite(d_lake[i])]
        roi_by_hour.append(
            {
                "hour": int(hour),
                "pct_near_tree": round(100.0 * len(tree_p) / len(pids), 2) if pids else 0.0,
                "pct_near_lake": round(100.0 * len(lake_p) / len(pids), 2) if pids else 0.0,
                "avg_distance_to_tree": round(float(np.mean(tree_dists)), 1) if tree_dists else 0.0,
                "avg_distance_to_lake": round(float(np.mean(lake_dists)), 1) if lake_dists else 0.0,
            }
        )

    roi_df = pd.DataFrame(roi_by_hour)
    if not old_hm.empty:
        for c in ["pct_near_tree", "pct_near_lake", "avg_distance_to_tree", "avg_distance_to_lake"]:
            if c in old_hm.columns:
                old_hm.drop(columns=[c], inplace=True)
        merged = old_hm.merge(roi_df, on="hour", how="left")
        merged.to_csv(hm_path, index=False)
    else:
        roi_df.to_csv(hm_path, index=False)

    print(f"Updated {tracks_path} (zone from ROI)")
    print(f"Updated {hm_path}")
    n = max(len(person_ids), 1)
    print(f"Visitors with any lake detection: {len(lake_visitors)}/{len(person_ids)} ({100*len(lake_visitors)/n:.1f}%)")
    print(f"Visitors with any tree detection: {len(tree_visitors)}/{len(person_ids)} ({100*len(tree_visitors)/n:.1f}%)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--roi", type=Path, required=True)
    p.add_argument("--width", type=int, default=3840)
    p.add_argument("--height", type=int, default=2160)
    args = p.parse_args()
    apply(args.output_dir, args.roi, (args.width, args.height))


if __name__ == "__main__":
    main()
