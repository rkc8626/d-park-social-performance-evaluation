#!/usr/bin/env python3
"""Minimal labeling UI — run on YOUR laptop (no HiPerGator port forward).

Copy a batch folder from Orange to your computer, then:

  cd GX020055_batch01
  python /path/to/local_label_ui.py --batch-dir .

Open http://127.0.0.1:8765 in your browser. Progress saves to manifest.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ACTIVITIES = [
    "walking",
    "running",
    "biking",
    "sitting",
    "standing",
    "talking_socializing",
    "playing",
    "dog_walking",
    "exercising",
    "picnic_resting",
]
STATUSES = ["valid", "not_a_person", "skip"]
AGES = ["child", "adult"]


def load_manifest(batch_dir: Path) -> tuple[list[dict], list[str]]:
    path = batch_dir / "manifest.csv"
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    fields = list(rows[0].keys()) if rows else []
    for col in ("label_status", "activity_label", "apparent_age_group", "notes"):
        if col not in fields:
            fields.append(col)
    return rows, fields


def save_manifest(batch_dir: Path, rows: list[dict], fields: list[str]) -> None:
    path = batch_dir / "manifest.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    queue = batch_dir / "label_queue.csv"
    queue.write_text(path.read_text())


def page_html(batch_dir: Path, rows: list[dict], index: int) -> str:
    row = rows[index]
    clip_dir = batch_dir / row["clip_dir"]
    frames = sorted(clip_dir.glob("frame_*.jpg"))
    imgs = "".join(
        f'<img src="/img/{index}/{i}" alt="frame {i+1}"/>' for i in range(len(frames))
    )
    status = row.get("label_status", "")
    act = row.get("activity_label", "")
    age = row.get("apparent_age_group", "")
    notes = row.get("notes", "")

    opt = lambda name, vals, cur: "".join(
        f'<option value="{v}"{" selected" if v == cur else ""}>{v}</option>' for v in vals
    )

    done = sum(1 for r in rows if str(r.get("label_status", "")).strip())
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<title>Label {row.get("clip_id","")}</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 1rem 2rem; max-width: 1200px; }}
  .frames {{ display: flex; flex-wrap: wrap; gap: 8px; }}
  .frames img {{ max-height: 220px; border: 1px solid #ccc; }}
  form {{ margin-top: 1rem; display: grid; gap: 0.5rem; max-width: 480px; }}
  label {{ font-weight: 600; }}
  nav {{ margin: 1rem 0; }}
  .meta {{ color: #444; font-size: 0.95rem; }}
</style></head>
<body>
  <p><b>Clip {index + 1} / {len(rows)}</b> — labeled: {done}</p>
  <p class="meta">clip_id={row.get("clip_id")} track_id={row.get("track_id")}
     t={row.get("start_time")}–{row.get("end_time")}s
     conf={row.get("mean_confidence","")} h={row.get("mean_bbox_height_px","")}</p>
  <div class="frames">{imgs}</div>
  <form method="POST" action="/save/{index}">
    <label>label_status</label>
    <select name="label_status" required>{opt("s", STATUSES, status)}</select>
    <label>activity_label (if valid)</label>
    <select name="activity_label"><option value=""></option>{opt("a", ACTIVITIES, act)}</select>
    <label>apparent_age_group (if valid)</label>
    <select name="apparent_age_group"><option value=""></option>{opt("g", AGES, age)}</select>
    <label>notes</label>
    <input name="notes" value="{notes}" style="width:100%"/>
    <button type="submit">Save &amp; next</button>
  </form>
  <nav>
    {"<a href='/label/" + str(index - 1) + "'>← Prev</a> | " if index > 0 else ""}
    <a href="/">Index</a>
    {" | <a href='/label/" + str(index + 1) + "'>Next →</a>" if index + 1 < len(rows) else ""}
  </nav>
</body></html>"""


def index_html(rows: list[dict]) -> str:
    lines = ["<html><body><h1>Labeling index</h1><ul>"]
    for i, r in enumerate(rows):
        st = r.get("label_status", "") or "—"
        lines.append(
            f"<li><a href='/label/{i}'>{r.get('clip_id')}</a> — {st}</li>"
        )
    lines.append("</ul></body></html>")
    return "\n".join(lines)


class Handler(BaseHTTPRequestHandler):
    batch_dir: Path
    rows: list[dict]
    fields: list[str]

    def log_message(self, fmt: str, *args: object) -> None:
        print(fmt % args)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send_html(index_html(self.rows))
            return
        if path.startswith("/label/"):
            idx = int(path.split("/")[-1])
            self._send_html(page_html(self.batch_dir, self.rows, idx))
            return
        if path.startswith("/img/"):
            parts = path.strip("/").split("/")
            idx, frame_i = int(parts[1]), int(parts[2])
            row = self.rows[idx]
            clip_dir = self.batch_dir / row["clip_dir"]
            frames = sorted(clip_dir.glob("frame_*.jpg"))
            if frame_i >= len(frames):
                self.send_error(404)
                return
            data = frames[frame_i].read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if not path.startswith("/save/"):
            self.send_error(404)
            return
        idx = int(path.split("/")[-1])
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode()
        form = {k: v[0] for k, v in parse_qs(body).items()}
        for k in ("label_status", "activity_label", "apparent_age_group", "notes"):
            self.rows[idx][k] = form.get(k, "")
        save_manifest(self.batch_dir, self.rows, self.fields)
        nxt = min(idx + 1, len(self.rows) - 1)
        self.send_response(303)
        self.send_header("Location", f"/label/{nxt}")
        self.end_headers()

    def _send_html(self, html: str) -> None:
        data = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    p = argparse.ArgumentParser(description="Local browser labeling (no SSH tunnel).")
    p.add_argument("--batch-dir", type=Path, default=".", help="GX020055_batch01 folder")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()

    batch_dir = args.batch_dir.resolve()
    if not (batch_dir / "manifest.csv").is_file():
        raise SystemExit(f"No manifest.csv in {batch_dir}")

    rows, fields = load_manifest(batch_dir)
    Handler.batch_dir = batch_dir
    Handler.rows = rows
    Handler.fields = fields

    server = HTTPServer((args.host, args.port), Handler)
    print(f"Labeling UI: http://{args.host}:{args.port}/")
    print(f"Batch: {batch_dir}")
    print("Press Ctrl+C to stop. manifest.csv updates on each Save.")
    server.serve_forever()


if __name__ == "__main__":
    main()
