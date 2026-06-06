#!/usr/bin/env python3
"""Build combined dataset_index.csv from labeled batch manifests (valid clips only)."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd
import yaml

from export_track_clips import ACTIVITIES, AGE_GROUPS


def _split(clip_id: str, val_ratio: float) -> str:
    h = int(hashlib.md5(clip_id.encode()).hexdigest(), 16)
    return "val" if (h % 1000) < int(val_ratio * 1000) else "train"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path(__file__).with_name("classifier_config.yaml"))
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load(args.config.read_text())
    val_ratio = float(cfg.get("val_ratio", 0.15))

    rows: list[dict] = []
    for batch_rel in cfg["batches"]:
        batch_dir = Path(batch_rel)
        if not batch_dir.is_absolute():
            batch_dir = (root / batch_dir).resolve()
        manifest = pd.read_csv(batch_dir / "manifest.csv")
        valid = manifest[manifest["label_status"].astype(str).str.strip() == "valid"]
        for _, r in valid.iterrows():
            act = str(r.get("activity_label", "")).strip()
            age = str(r.get("apparent_age_group", "")).strip()
            if act not in ACTIVITIES or age not in AGE_GROUPS:
                continue
            cid = str(r["clip_id"])
            rows.append(
                {
                    "clip_id": cid,
                    "batch_dir": str(batch_dir),
                    "clip_subdir": str(r["clip_dir"]),
                    "video_id": str(r["video_id"]),
                    "track_id": int(r["track_id"]),
                    "activity_label": act,
                    "apparent_age_group": age,
                    "split": _split(cid, val_ratio),
                }
            )

    out = (root / cfg["dataset_index"]).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} clips -> {out}")
    print("split:", df["split"].value_counts().to_dict())
    print("activity:", df["activity_label"].nunique(), "classes")
    print("age:", df["apparent_age_group"].value_counts().to_dict())


if __name__ == "__main__":
    main()
