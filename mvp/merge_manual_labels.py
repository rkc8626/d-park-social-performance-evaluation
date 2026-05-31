#!/usr/bin/env python3
"""Apply filled manifest.csv labels to tracks.csv (per-frame in time range).

label_status (recommended):
  valid         — real person: set activity_label + apparent_age_group
  skip          — unusable clip: no change to tracks (omit from training)
  not_a_person  — false detection: mark whole track_id, exclude from metrics

After merge, run regenerate_tables.py for updated events / hourly_metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from export_track_clips import (
    ACTIVITIES,
    AGE_GROUPS,
    LABEL_STATUS_NOT_PERSON,
    LABEL_STATUS_SKIP,
    LABEL_STATUS_VALID,
)


def _norm(s: object) -> str:
    return str(s).strip().lower().replace(" ", "_")


def apply_manifest(
    tracks_path: Path,
    manifest_path: Path,
    dry_run: bool = False,
) -> pd.DataFrame:
    tracks = pd.read_csv(tracks_path)
    manifest = pd.read_csv(manifest_path)

    if "label_status" not in manifest.columns:
        manifest["label_status"] = ""

    has_status = manifest["label_status"].astype(str).str.strip().ne("")
    has_labels = manifest["activity_label"].astype(str).str.strip().ne("") | manifest[
        "apparent_age_group"
    ].astype(str).str.strip().ne("")
    todo = manifest[has_status | has_labels]
    if todo.empty:
        raise SystemExit(
            "No labeled rows. Fill label_status and/or activity_label / apparent_age_group."
        )

    n_act = n_age = n_fp = n_skip = 0
    for _, row in todo.iterrows():
        tid = int(row["track_id"])
        status = _norm(row.get("label_status", ""))
        act = str(row.get("activity_label", "")).strip()
        age = str(row.get("apparent_age_group", "")).strip()

        if status in ("", "valid") and act in ("", "nan") and age in ("", "nan"):
            if status != LABEL_STATUS_VALID:
                continue
            print(f"Skip {row['clip_id']}: label_status=valid but no activity/age filled")
            continue

        if status == LABEL_STATUS_NOT_PERSON or _norm(act) in (
            "not_a_person",
            "not_person",
            "false_positive",
            "fp",
        ):
            mask = tracks["track_id"] == tid
            tracks.loc[mask, "activity_label"] = LABEL_STATUS_NOT_PERSON
            tracks.loc[mask, "apparent_age_group"] = ""
            n_fp += int(mask.sum())
            continue

        if status == LABEL_STATUS_SKIP or _norm(act) in ("skip", "unclear", "unknown"):
            n_skip += 1
            continue

        t0, t1 = float(row["start_time"]), float(row["end_time"])
        mask = (
            (tracks["track_id"] == tid)
            & (tracks["timestamp"] >= t0 - 1e-6)
            & (tracks["timestamp"] <= t1 + 1e-6)
        )
        if act and _norm(act) not in ("skip", "not_a_person", "not_person"):
            label = act.replace("/", "_")
            if label not in ACTIVITIES:
                print(f"Warning: unknown activity '{act}' on {row['clip_id']}")
            tracks.loc[mask, "activity_label"] = label
            n_act += int(mask.sum())
        if age:
            if age not in AGE_GROUPS:
                print(f"Warning: unknown age '{age}' on {row['clip_id']} (use child|adult)")
            tracks.loc[mask, "apparent_age_group"] = age
            n_age += int(mask.sum())

    print(
        f"Applied: activity_rows={n_act}, age_rows={n_age}, "
        f"false_positive_tracks={n_fp}, skipped_clips={n_skip}"
    )
    if not dry_run:
        tracks.to_csv(tracks_path, index=False)
        print(f"Updated {tracks_path}")
    return tracks


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tracks", type=Path, required=True)
    p.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Filled manifest.csv from a labeling batch",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    apply_manifest(args.tracks, args.manifest, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
