#!/bin/bash
# ============================================================
# RunPod Setup Script — FerretNet Training
# Run this once inside the pod terminal after connecting.
#
# Usage:
#   bash scripts/runpod_setup.sh
# ============================================================
set -e

REPO_URL="https://github.com/YOUR_USERNAME/YOUR_REPO.git"   # ← change this
PROJECT_ROOT="/workspace/AI_COMPUTER_VISION"
MAC_PREFIX="/Users/MAC/Desktop/Projects/AI_COMPUTER_VISION"

echo "=== Step 1: Clone repo ==="
git clone "$REPO_URL" "$PROJECT_ROOT"
cd "$PROJECT_ROOT"

echo "=== Step 2: Install dependencies ==="
pip install -q -r requirements.txt
pip install -q tensorboard

echo "=== Step 3: Verify GPU ==="
python3 -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"

echo "=== Step 4: Download dataset from HuggingFace ==="
# Upload your data_raw/ folder to HuggingFace first:
#   huggingface-cli upload YOUR_USERNAME/gravex-200k data_raw/ data_raw/
# Then pull it here:
pip install -q huggingface_hub
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='YOUR_USERNAME/gravex-200k',   # ← change this
    repo_type='dataset',
    local_dir='data_raw/'
)
print('Dataset downloaded.')
"

echo "=== Step 5: Download weights ==="
# ferretnet-b-median-3.pth — either push to git or download here
# If it's in the repo already: nothing to do
# Otherwise: gdown / wget from wherever you stored it
mkdir -p weights/
# Example: gdown "https://drive.google.com/uc?id=YOUR_ID" -O weights/ferretnet-b-median-3.pth

echo "=== Step 6: Remap manifest paths ==="
# The CSVs have Mac absolute paths — remap them to pod paths
python3 -c "
import os
mac_prefix = '$MAC_PREFIX'
pod_prefix = '$PROJECT_ROOT'
manifests = [
    'data_raw/manifests/train_with_faces.csv',
    'data_raw/manifests/val_with_faces.csv',
    'data_raw/manifests/test_with_faces.csv',
]
for path in manifests:
    if not os.path.exists(path):
        print(f'Skipping {path} — not found')
        continue
    with open(path, 'r') as f:
        content = f.read()
    content = content.replace(mac_prefix, pod_prefix)
    with open(path, 'w') as f:
        f.write(content)
    print(f'Remapped: {path}')
"

echo "=== Step 7: Set PROJECT_ROOT env var ==="
export PROJECT_ROOT="$PROJECT_ROOT"
echo "export PROJECT_ROOT=$PROJECT_ROOT" >> ~/.bashrc

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Start training:"
echo "  cd $PROJECT_ROOT"
echo "  python3 training/train.py"
echo ""
echo "Monitor with TensorBoard:"
echo "  tensorboard --logdir $PROJECT_ROOT/lightning_logs --host 0.0.0.0 --port 6006"
