# Labeling without HiPerGator localhost / SSH tunnel

If you **cannot** open `http://localhost:8080` to the login node (no SSH `-L`, no IDE port forward), use one of these:

---

## Option A — Local web UI (recommended, no Label Studio)

### 1. Copy batch to your laptop

From your laptop:

```bash
rsync -avz --progress \
  chenz1@login9.rc.ufl.edu:/orange/ufdatastudios/chenz1/depotData/d-park-social-performance-evaluation/annotations/labeling/clips/GX020055_batch01/ \
  ~/park_labeling/GX020055_batch01/
```

(~150 clips × 5 JPEGs — allow a few minutes.)

Copy `mvp/local_label_ui.py` too, or clone the repo locally.

### 2. Run UI on your machine

```bash
cd ~/park_labeling/GX020055_batch01
python3 /path/to/mvp/local_label_ui.py --batch-dir .
```

Open **http://127.0.0.1:8765** in your browser (this is **your** laptop’s localhost, not HiPerGator).

- Pick `label_status` → if `valid`, set activity + age  
- **Save & next** writes `manifest.csv` immediately  

### 3. Upload manifest back to Orange

```bash
rsync -avz ~/park_labeling/GX020055_batch01/manifest.csv \
  chenz1@login9.rc.ufl.edu:/orange/ufdatastudios/chenz1/depotData/d-park-social-performance-evaluation/annotations/labeling/clips/GX020055_batch01/
```

### 4. Merge on cluster

```bash
cd depotData/d-park-social-performance-evaluation
source setup_env.sh
python mvp/merge_manual_labels.py \
  --tracks outputs/mvp/GX020055/tracks.csv \
  --manifest annotations/labeling/clips/GX020055_batch01/manifest.csv
cd mvp && python regenerate_tables.py \
  --output-dir ../outputs/mvp/GX020055 \
  --roi ../annotations/roi/east.json --video-id GX020055
```

---

## Option B — Label Studio on your laptop

Same `rsync` as above, then on **your Mac/Windows/Linux**:

```bash
pip install label-studio
cd ~/park_labeling/GX020055_batch01
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=$PWD

# Regenerate import JSON with local paths (on laptop, from repo):
python mvp/label_studio_prepare.py --manifest manifest.csv

label-studio start --port 8080
```

Open **http://127.0.0.1:8080** locally → import `label_studio_import.json` + paste `label_studio.xml`.

Export JSON → `label_studio_export_to_manifest.py` (on laptop or cluster).

---

## Option C — Excel / CSV only (no server)

1. `rsync` the batch folder.
2. Open `manifest.csv` in Excel / LibreOffice.
3. View images in `clips/<clip_id>/` in Finder/Explorer.
4. Fill `label_status`, `activity_label`, `apparent_age_group`.
5. `rsync` manifest back → `merge_manual_labels.py`.

---

## Option D — UF Open OnDemand (if you have access)

Some HiPerGator accounts can start a **desktop or Jupyter** session in a browser via [OOD](https://www.rc.ufl.edu/documentation/web-apps/open-ondemand/).  
Then run Label Studio or `local_label_ui.py` **inside that session** and use the URL OOD gives you (not login-node SSH tunnel).

---

## Why cluster `localhost` fails

`label-studio start` on **login9** binds to that node’s port 8080. Your laptop only reaches it via **SSH port forwarding** (`ssh -L`) or a **web portal**.  
If your network blocks that, cluster localhost is not usable from your browser — **run the UI on the machine where your browser runs** (Option A or B).
