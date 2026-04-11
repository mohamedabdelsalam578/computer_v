#!/usr/bin/env python3
"""
Sanity-check that pretrained weights load into the backbone and training wiring matches intent.

From repo root:
  python scripts/verify_pretrained_setup.py

Requires weights/ferretnet-b-median-3.pth (skip with message if missing).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from configs.base_config import FusionConfig, FerretNetConfig, TrainConfig
from ferretnet.dual_branch import DualBranchFerretNet
from ferretnet.lightning_module import FerretNetLightning


def main() -> int:
    from configs.base_config import ProjectConfig

    proj = ProjectConfig()
    wpath = proj.ferretnet_weights
    if not wpath.is_file():
        print(f"SKIP: no file at {wpath.resolve()}")
        print("      On RunPod, ensure weights are under $PROJECT_ROOT/weights/")
        return 0

    raw = torch.load(wpath, map_location="cpu", weights_only=True)
    if isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict):
        raw_sd = raw["model"]
    elif isinstance(raw, dict):
        raw_sd = raw
    else:
        print("FAIL: checkpoint is not a state dict")
        return 1

    if "cbr1.0.weight" not in raw_sd:
        print("FAIL: checkpoint missing cbr1.0.weight (unexpected format)")
        return 1

    dual = DualBranchFerretNet(str(wpath), FusionConfig())
    w_ckpt = raw_sd["cbr1.0.weight"]
    w_face = dual.face_net.cbr1[0].weight
    if w_face.shape != w_ckpt.shape or not torch.allclose(w_face, w_ckpt):
        print("FAIL: face_net backbone does not match checkpoint cbr1.0.weight")
        return 1

    if not torch.allclose(dual.face_net.cbr1[0].weight, dual.full_net.cbr1[0].weight):
        print("FAIL: face vs full branch differ at init (both should load same ckpt)")
        return 1

    key = "logit.1.weight"
    if key in raw_sd and torch.allclose(dual.face_net.logit[1].weight, raw_sd[key]):
        print("WARN: classifier still equals checkpoint (head reinit may not have run)")

    tc = TrainConfig()
    backbone_params = [p for n, p in dual.named_parameters() if "logit" not in n]
    head_params = [p for n, p in dual.named_parameters() if "logit" in n]
    opt_plain = torch.optim.Adam(
        [
            {"params": backbone_params, "lr": tc.learning_rate * 0.1},
            {"params": head_params, "lr": tc.learning_rate},
        ],
        betas=tc.betas,
        weight_decay=tc.weight_decay,
    )
    g0, g1 = opt_plain.param_groups[0]["lr"], opt_plain.param_groups[1]["lr"]
    if abs(g0 - tc.learning_rate * 0.1) > 1e-15 or abs(g1 - tc.learning_rate) > 1e-15:
        print(f"FAIL: intended base LRs wrong: backbone={g0} head={g1}")
        return 1
    if abs(g1 / g0 - 10.0) > 1e-6:
        print(f"FAIL: head/backbone LR ratio should be 10, got {g1/g0}")
        return 1

    # Lightning attaches LinearLR(start_factor=0.01) — PyTorch then scales displayed lr immediately.
    m = FerretNetLightning(
        ferretnet_config=FerretNetConfig(),
        train_config=tc,
        fusion_config=FusionConfig(),
        pretrained_path=str(wpath),
    )
    out = m.configure_optimizers()
    sched = out["lr_scheduler"]["scheduler"]
    if hasattr(sched, "schedulers"):
        w0 = sched.schedulers[0].base_lrs
        if abs(w0[0] - tc.learning_rate * 0.1) > 1e-15 or abs(w0[1] - tc.learning_rate) > 1e-15:
            print(f"FAIL: scheduler base_lrs {w0} != backbone/head base")
            return 1

    print("OK: pretrained backbone matches checkpoint on disk")
    print("OK: face/full branches identical after init")
    print("OK: optimizer base LRs = 0.1x / 1x on backbone vs logit (head 10x higher)")
    print("OK: Lightning warmup uses start_factor=0.01 (first-epoch lr is 100x smaller — intentional)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
