#!/usr/bin/env bash
# Run DeepFake Detective (Streamlit) on RunPod with sane defaults.
#
# Usage (from repo root on the pod):
#   bash scripts/streamlit_runpod.sh
#
# Optional env:
#   PROJECT_ROOT   default: /workspace/AI_COMPUTER_VISION
#   STREAMLIT_PORT default: 8501
#   AI_CV_CLIP_USE_LAST — if set to 1, same as checking “CLIP: prefer last.ckpt” in the UI
#     (omit after training if you want best clip-*.ckpt by val metric in the filename)
set -euo pipefail

ROOT="${PROJECT_ROOT:-/workspace/AI_COMPUTER_VISION}"
PORT="${STREAMLIT_PORT:-8501}"
cd "$ROOT"
export PROJECT_ROOT="$ROOT"
export AI_CV_CLIP_USE_LAST="${AI_CV_CLIP_USE_LAST:-1}"

exec python3 -m streamlit run app/streamlit_app.py \
  --server.address 0.0.0.0 \
  --server.port "$PORT"
