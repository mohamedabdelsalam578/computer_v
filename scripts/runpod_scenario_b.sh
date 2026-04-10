#!/usr/bin/env bash
# Scenario B: HF dataset (raw images + manifests + weights) → GPU RetinaFace → train.
# Prerequisites: set HF_REPO_ID and REPO_URL (or pass as args).
#
# Usage:
#   export HF_REPO_ID="YOUR_USERNAME/ferretnet-data"
#   export REPO_URL="https://github.com/YOUR_USERNAME/YOUR_REPO.git"
#   bash scripts/runpod_scenario_b.sh
#
# Optional:
#   PROJECT_ROOT=/workspace/AI_COMPUTER_VISION  (default if unset)
#   MAC_PREFIX=/Users/MAC/Desktop/Projects/AI_COMPUTER_VISION  (path prefix in CSVs to replace)
set -euo pipefail

HF_REPO_ID="${HF_REPO_ID:-}"
REPO_URL="${REPO_URL:-}"
PROJECT_ROOT="${PROJECT_ROOT:-/workspace/AI_COMPUTER_VISION}"
MAC_PREFIX="${MAC_PREFIX:-/Users/MAC/Desktop/Projects/AI_COMPUTER_VISION}"

if [[ -z "$HF_REPO_ID" || -z "$REPO_URL" ]]; then
  echo "Set HF_REPO_ID and REPO_URL, e.g.:"
  echo "  export HF_REPO_ID=YOUR_USERNAME/ferretnet-data"
  echo "  export REPO_URL=https://github.com/YOUR_USERNAME/YOUR_REPO.git"
  exit 1
fi

echo "=== Clone repo ==="
if [[ ! -d "$PROJECT_ROOT/.git" ]]; then
  git clone "$REPO_URL" "$PROJECT_ROOT"
fi
cd "$PROJECT_ROOT"
export PROJECT_ROOT="$PROJECT_ROOT"

echo "=== Dependencies ==="
pip install -q -r requirements.txt
pip install -q huggingface_hub tensorboard

echo "=== GPU check ==="
python3 -c "import torch; print('CUDA:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"

echo "=== Download Hugging Face dataset repo → project layout ==="
python3 << PY
from pathlib import Path
import shutil
import os
from huggingface_hub import snapshot_download

repo_id = os.environ["HF_REPO_ID"]
root = Path(os.environ["PROJECT_ROOT"]).resolve()
staging = root / "hf_staging"
snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=str(staging))

data_raw = root / "data_raw"
weights = root / "weights"
data_raw.mkdir(parents=True, exist_ok=True)
weights.mkdir(parents=True, exist_ok=True)

shutil.move(str(staging / "gravex_200k"), str(data_raw / "gravex_200k"))
shutil.move(str(staging / "manifests"), str(data_raw / "manifests"))
w = staging / "weights"
if w.is_dir():
    for p in w.iterdir():
        shutil.move(str(p), str(weights / p.name))
shutil.rmtree(staging, ignore_errors=True)
print("Staged dataset under data_raw/ and weights/.")
PY

echo "=== Remap manifest paths (Mac → pod) ==="
python3 << PY
from pathlib import Path
import os

mac = os.environ.get("MAC_PREFIX", "/Users/MAC/Desktop/Projects/AI_COMPUTER_VISION")
pod = os.environ["PROJECT_ROOT"]
root = Path(pod)
for name in ("train.csv", "val.csv", "test.csv"):
    f = root / "data_raw" / "manifests" / name
    if not f.is_file():
        print(f"skip missing {f}")
        continue
    t = f.read_text().replace(mac, pod)
    f.write_text(t)
    print(f"remapped {f}")
PY

echo "=== RetinaFace (GPU if available) ==="
python3 scripts/precompute_face_crops.py

echo "=== Training ==="
python3 training/train.py

echo "=== Done ==="
