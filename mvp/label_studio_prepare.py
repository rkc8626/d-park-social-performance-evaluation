#!/usr/bin/env python3
"""Build Label Studio task import JSON from a labeling batch manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from export_track_clips import ACTIVITIES, AGE_GROUPS


def _local_url(batch_dir: Path, rel_path: Path) -> str:
    """Path relative to LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT (= batch_dir)."""
    rel = rel_path.relative_to(batch_dir).as_posix()
    return f"/data/local-files/?d={rel}"


def build_tasks(manifest: pd.DataFrame, batch_dir: Path) -> list[dict]:
    tasks: list[dict] = []
    for _, row in manifest.iterrows():
        clip_dir = batch_dir / row["clip_dir"]
        frames = sorted(clip_dir.glob("frame_*.jpg"))
        if len(frames) < 1:
            continue
        while len(frames) < 5:
            frames.append(frames[-1])

        data = {
            "clip_id": str(row["clip_id"]),
            "video_id": str(row["video_id"]),
            "track_id": str(int(row["track_id"])),
            "start_time": str(row["start_time"]),
            "end_time": str(row["end_time"]),
        }
        for i, fp in enumerate(frames[:5], start=1):
            data[f"f{i}"] = _local_url(batch_dir, fp)

        tasks.append({"data": data})
    return tasks


def main() -> None:
    p = argparse.ArgumentParser(description="Create label_studio_import.json for a batch.")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    manifest_path = args.manifest.resolve()
    batch_dir = manifest_path.parent
    manifest = pd.read_csv(manifest_path)
    tasks = build_tasks(manifest, batch_dir)

    out = args.out or batch_dir / "label_studio_import.json"
    out.write_text(json.dumps(tasks, indent=2))

    env_file = batch_dir / "label_studio.env"
    env_file.write_text(
        f"""# Source before starting Label Studio (from this batch directory):
#   source label_studio.env
#   label-studio start --port 8080

export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT={batch_dir}
"""
    )

    print(f"Wrote {len(tasks)} tasks -> {out}")
    print(f"Wrote env file -> {env_file}")
    print(f"Set DOCUMENT_ROOT to: {batch_dir}")
    print("Activities:", ", ".join(ACTIVITIES))
    print("Age groups:", ", ".join(AGE_GROUPS))


if __name__ == "__main__":
    main()
