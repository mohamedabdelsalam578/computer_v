# Scenario B: Raw data on Hugging Face, RetinaFace + training on RunPod

Upload only what the pod needs (no precomputed face crops). RetinaFace on a GPU pod is much faster than CPU face detection on your Mac, so total wall time beats uploading precomputed crops from a home connection.

## Time budget (typical)

| Step | Time |
|------|------|
| Upload ~2.4 GB from Mac (~10 MB/s) | ~4 min |
| Download on RunPod | ~5 sec |
| RetinaFace on pod (GPU, ~500 img/s vs ~65 img/s Mac CPU) | ~7 min |
| Training on RTX 4090 (~20 min early stop) | ~20 min |
| **Total** | **~31 min** |

Roughly **~16 minutes faster** than uploading large precomputed artifacts from a typical home uplink.

## What to upload (~2.4 GB + FerretNet checkpoint)

| Component | Size (order of) | Remote path in dataset repo |
|-----------|------------------|----------------------------|
| Raw GRAVEX images | ~2.2 GB | `gravex_200k/` |
| CSV manifests (`train.csv`, `val.csv`, `test.csv`) | ~75 MB | `manifests/` |
| RetinaFace weights | ~108 MB | `weights/Resnet50_Final.pth` |
| FerretNet backbone (training) | (your file) | `weights/ferretnet-b-median-3.pth` |

Replace `YOUR_USERNAME` and `YOUR_REPO` everywhere below.

---

## On your Mac — install CLI and log in

```bash
pip install huggingface_hub
huggingface-cli login
```

## On your Mac — upload only what is needed

```bash
# Raw images
huggingface-cli upload YOUR_USERNAME/ferretnet-data \
    data_raw/gravex_200k   gravex_200k   --repo-type dataset

# Manifests (paths will be rewritten on the pod)
huggingface-cli upload YOUR_USERNAME/ferretnet-data \
    data_raw/manifests     manifests     --repo-type dataset

# RetinaFace ResNet50 weights (used by scripts/precompute_face_crops.py)
huggingface-cli upload YOUR_USERNAME/ferretnet-data \
    weights/Resnet50_Final.pth  weights/Resnet50_Final.pth  --repo-type dataset

# FerretNet pretrained checkpoint (used by training/train.py)
huggingface-cli upload YOUR_USERNAME/ferretnet-data \
    weights/ferretnet-b-median-3.pth  weights/ferretnet-b-median-3.pth  --repo-type dataset
```

---

## On RunPod — one script (recommended)

From the project root after cloning:

```bash
export HF_REPO_ID="YOUR_USERNAME/ferretnet-data"
export REPO_URL="https://github.com/YOUR_USERNAME/YOUR_REPO.git"
bash scripts/runpod_scenario_b.sh
```

The script installs dependencies, pulls the dataset repo into a staging folder, moves `gravex_200k/` and `manifests/` under `data_raw/` and all `weights/*` to project `weights/`, remaps absolute paths in the CSVs, runs RetinaFace on GPU, then starts training.

---

## On RunPod — same steps by hand

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git /workspace/AI_COMPUTER_VISION
cd /workspace/AI_COMPUTER_VISION
pip install -r requirements.txt
pip install huggingface_hub
export PROJECT_ROOT=/workspace/AI_COMPUTER_VISION

# Download dataset repo; re-layout so weights/ sit at project root (not under data_raw/)
python3 << 'PY'
from pathlib import Path
import shutil
from huggingface_hub import snapshot_download

repo_id = "YOUR_USERNAME/ferretnet-data"
staging = Path("hf_staging")
snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=str(staging))

root = Path(".").resolve()
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
print("Download and layout complete.")
PY

# Remap Mac paths → pod paths in base manifests (before face precompute)
python3 << 'PY'
from pathlib import Path
mac = "/Users/MAC/Desktop/Projects/AI_COMPUTER_VISION"
pod = "/workspace/AI_COMPUTER_VISION"
for name in ("train.csv", "val.csv", "test.csv"):
    f = Path("data_raw/manifests") / name
    if not f.is_file():
        continue
    t = f.read_text().replace(mac, pod)
    f.write_text(t)
print("Paths remapped.")
PY

# RetinaFace on GPU (~7 min for ~200k images on a strong GPU)
python3 scripts/precompute_face_crops.py

# Training (~20 min early stop on RTX 4090-class GPU)
python3 training/train.py
```

---

## Notes

- Set `PROJECT_ROOT` on the pod if your checkout is not `/workspace/AI_COMPUTER_VISION` (see `configs/base_config.py`).
- Face precompute prefers **CUDA**, then MPS, then CPU (`scripts/precompute_face_crops.py`).
