# Label Studio — park clip labeling

> **No SSH / localhost access to login9?** Use **[LOCAL_LABELING.md](LOCAL_LABELING.md)** — copy clips to your laptop and run `mvp/local_label_ui.py` or Label Studio locally.

Label **one clip = one task** (5 frames shown). Export back to `manifest.csv`, then run `merge_manual_labels.py`.

## 1. Install (once)

On HiPerGator login node (use project venv on Orange, not home):

```bash
cd depotData/d-park-social-performance-evaluation
source setup_env.sh
pip install label-studio
```

## 2. Prepare tasks (per batch)

Example for east batch:

```bash
python mvp/label_studio_prepare.py \
  --manifest annotations/labeling/clips/GX020055_batch01/manifest.csv
```

Creates in that batch folder:

- `label_studio_import.json` — import this in Label Studio
- `label_studio.env` — local image paths

Repeat for west:

```bash
python mvp/label_studio_prepare.py \
  --manifest annotations/labeling/clips/GX020308_batch01/manifest.csv
```

## 3. Start Label Studio

```bash
cd annotations/labeling/clips/GX020055_batch01
source ../../../../setup_env.sh   # or absolute path to setup_env.sh
source label_studio.env
label-studio start --port 8080 --no-browser
```

**SSH port forward** from your laptop:

```bash
ssh -L 8080:localhost:8080 chenz1@login9.rc.ufl.edu
```

Open in browser: **http://localhost:8080**

## 4. Create project (first time only)

1. **Create Project** → name e.g. `park-east-batch01`
2. **Settings → Labeling Interface → Custom template**
3. Paste contents of `annotations/labeling/label_studio.xml` → **Save**
4. **Settings → Cloud Storage → Add Source Storage**
   - Storage Type: **Local files**
   - Path: same as `LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT`  
     (the batch folder, e.g. `.../GX020055_batch01`)
   - Treat as: **Files**
   - **Add** and enable storage

   If images do not load, skip cloud storage — local serving via `label_studio.env` is enough when URLs use `/data/local-files/?d=clips/...`.

5. **Import** → upload `label_studio_import.json`

## 5. Label tasks

For each task:

1. Check all **5 frames**
2. **label_status**
   - `not_a_person` — false detection (common)
   - `skip` — cannot tell
   - `valid` — real person → then pick **activity** and **age**
3. Optional **notes**
4. Submit → next task

Shortcuts: use Label Studio **Settings → Machine Learning** off; use **Filters** to show unlabeled only.

## 6. Export from Label Studio

1. **Export** → format **JSON**
2. Save as e.g. `label_studio_export.json` in the batch folder

## 7. Merge into manifest.csv

```bash
cd depotData/d-park-social-performance-evaluation

python mvp/label_studio_export_to_manifest.py \
  --manifest annotations/labeling/clips/GX020055_batch01/manifest.csv \
  --export annotations/labeling/clips/GX020055_batch01/label_studio_export.json
```

## 8. Apply to tracks + metrics

```bash
python mvp/merge_manual_labels.py \
  --tracks outputs/mvp/GX020055/tracks.csv \
  --manifest annotations/labeling/clips/GX020055_batch01/manifest.csv

cd mvp
python regenerate_tables.py \
  --output-dir ../outputs/mvp/GX020055 \
  --roi ../annotations/roi/east.json \
  --video-id GX020055
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Images broken (icon only) | `source label_studio.env` before `label-studio start`; DOCUMENT_ROOT must be the **batch** folder (contains `clips/`) |
| Activity/age hidden | Select `valid` first — fields show when valid is selected |
| Wrong batch mixed | One Label Studio project per batch; re-run `label_studio_prepare.py` |
| Resume labeling | Export JSON periodically; re-import updates same `clip_id` rows |

## File map

| File | Role |
|------|------|
| `label_studio.xml` | Labeling UI template |
| `label_studio_import.json` | Tasks to import |
| `label_studio_export.json` | Your export from LS |
| `manifest.csv` | Ground truth table for pipeline |
