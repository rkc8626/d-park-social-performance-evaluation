#!/usr/bin/env python3
"""Merge Label Studio JSON export back into manifest.csv."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _choice_value(result_item: dict) -> str | None:
    v = result_item.get("value") or {}
    choices = v.get("choices")
    if choices:
        return str(choices[0])
    return None


def _text_value(result_item: dict) -> str | None:
    v = result_item.get("value") or {}
    t = v.get("text")
    if isinstance(t, list) and t:
        return str(t[0])
    if isinstance(t, str):
        return t
    return None


def parse_export(export_path: Path) -> dict[str, dict]:
    """Return clip_id -> {label_status, activity_label, apparent_age_group, notes}."""
    raw = json.loads(export_path.read_text())
    if isinstance(raw, dict) and "tasks" in raw:
        tasks = raw["tasks"]
    elif isinstance(raw, list):
        tasks = raw
    else:
        tasks = [raw]

    by_clip: dict[str, dict] = {}
    for task in tasks:
        data = task.get("data") or {}
        clip_id = str(data.get("clip_id", ""))
        if not clip_id:
            continue

        anns = task.get("annotations") or []
        if not anns:
            continue
        # use latest completed annotation
        ann = anns[-1]
        if ann.get("was_cancelled"):
            continue
        result = ann.get("result") or []

        row = {
            "label_status": "",
            "activity_label": "",
            "apparent_age_group": "",
            "notes": "",
        }
        for item in result:
            name = item.get("from_name", "")
            if name == "label_status":
                row["label_status"] = _choice_value(item) or ""
            elif name == "activity":
                row["activity_label"] = _choice_value(item) or ""
            elif name == "age":
                row["apparent_age_group"] = _choice_value(item) or ""
            elif name == "notes":
                row["notes"] = _text_value(item) or ""

        by_clip[clip_id] = row
    return by_clip


def merge_into_manifest(manifest_path: Path, export_path: Path, out_path: Path | None) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    labels = parse_export(export_path)
    if not labels:
        raise SystemExit("No annotations found in Label Studio export.")

    for col in ("label_status", "activity_label", "apparent_age_group", "notes"):
        if col not in manifest.columns:
            manifest[col] = ""

    n = 0
    for i, row in manifest.iterrows():
        cid = str(row["clip_id"])
        if cid not in labels:
            continue
        for col in ("label_status", "activity_label", "apparent_age_group", "notes"):
            val = labels[cid].get(col, "")
            if val:
                manifest.at[i, col] = val
        n += 1

    out = out_path or manifest_path
    manifest.to_csv(out, index=False)
    manifest.to_csv(manifest_path.parent / "label_queue.csv", index=False)
    print(f"Updated {n} rows in {out}")
    return manifest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--export", type=Path, required=True, help="Label Studio JSON export")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    merge_into_manifest(args.manifest, args.export, args.out)


if __name__ == "__main__":
    main()
