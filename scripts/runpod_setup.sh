#!/bin/bash
# ============================================================
# RunPod Setup — FerretNet Training
# Dataset downloads directly from Kaggle (no upload needed).
#
# Before running:
#   1. Push code to GitHub
#   2. Set KAGGLE_USERNAME and KAGGLE_KEY env vars in RunPod
#      (RunPod UI → Pod → Environment Variables)
#
# Usage inside pod terminal:
#   bash scripts/runpod_setup.sh
# ============================================================
set -e

REPO_URL="https://github.com/YOUR_USERNAME/YOUR_REPO.git"   # ← change this
export PROJECT_ROOT="/workspace/AI_COMPUTER_VISION"

echo "=== [1/6] Repo check ==="
# If already cloned (running from inside the repo), just stay here
if [ -f "training/train.py" ]; then
    echo "Already inside repo — skipping clone."
    PROJECT_ROOT="$(pwd)"
else
    git clone "$REPO_URL" "$PROJECT_ROOT"
    cd "$PROJECT_ROOT"
fi
echo "export PROJECT_ROOT=$PROJECT_ROOT" >> ~/.bashrc
export PROJECT_ROOT="$PROJECT_ROOT"

echo "=== [2/6] Install dependencies ==="
pip install -q -r requirements.txt
pip install -q kagglehub tensorboard

echo "=== [3/6] Download RetinaFace weights from Google Drive ==="
mkdir -p weights/
pip install -q gdown
# gdown --folder downloads into a subfolder named after the folder — move files up
gdown --folder "https://drive.google.com/drive/folders/1gAeC7Vqaq8QlRDGAfcLa_Oho0uNzTMcH" \
      --output /tmp/gdrive_weights/ --remaining-ok
# Move files from nested weights/weights/ to weights/
find /tmp/gdrive_weights/ -name "*.pth" -exec mv {} weights/ \;
echo "Weights downloaded:"
ls -lh weights/

echo "=== [3b/6] Verify GPU ==="
python3 -c "
import torch
print('CUDA:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
"

echo "=== [4/6] Download dataset from Kaggle ==="
# Kaggle credentials come from env vars set in RunPod UI:
#   KAGGLE_USERNAME = your kaggle username
#   KAGGLE_KEY      = your kaggle API key
mkdir -p ~/.kaggle
echo "{\"username\":\"$KAGGLE_USERNAME\",\"key\":\"$KAGGLE_KEY\"}" > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

python3 -c "
import kagglehub, shutil, os
from pathlib import Path

print('Downloading GRAVEX-200K from Kaggle...')
cache_path = kagglehub.dataset_download('muhammadbilal6305/200k-real-vs-ai-visuals-by-mbilal')
dest = Path('data_raw/gravex_200k')
dest.mkdir(parents=True, exist_ok=True)
for item in Path(cache_path).iterdir():
    d = dest / item.name
    if not d.exists():
        if item.is_dir(): shutil.copytree(str(item), str(d))
        else: shutil.copy2(str(item), str(d))
print('Dataset ready at', dest)
"

echo "=== [5/6] Generate manifests ==="
python3 scripts/download_data.py --skip-download

echo "=== [6/6] Precompute face crops (GPU-accelerated) ==="
# On RTX 4090: ~500 img/s → 200k images done in ~7 minutes
python3 scripts/precompute_face_crops.py

echo ""
echo "============================================"
echo "  Setup complete! Start training:"
echo "  python3 training/train.py"
echo ""
echo "  Monitor with TensorBoard:"
echo "  tensorboard --logdir lightning_logs --host 0.0.0.0 --port 6006"
echo "============================================"
