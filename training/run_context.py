"""Print training runtime context (matches typical Lightning / RunPod log headers)."""

from __future__ import annotations

import os


def print_distributed_env() -> None:
    """LOCAL_RANK, CUDA_VISIBLE_DEVICES, and active torch device (if torch is importable)."""
    rank = os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0"))
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    cvd_disp = cvd if cvd.strip() else "(all visible)"
    print(f"LOCAL_RANK: {rank} - CUDA_VISIBLE_DEVICES: [{cvd_disp}]")
    try:
        import torch
    except ImportError:
        print("(torch not installed — skipping device line)")
        return
    if torch.cuda.is_available():
        i = torch.cuda.current_device()
        print(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")
    elif torch.backends.mps.is_available():
        print("Device: Apple MPS")
    else:
        print("Device: CPU")
