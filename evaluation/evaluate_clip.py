"""
Evaluate CLIP dual-branch checkpoint on a *_with_faces.csv manifest.

Usage:
    python evaluation/evaluate_clip.py --checkpoint lightning_logs_clip/checkpoints/last.ckpt
    python evaluation/evaluate_clip.py --checkpoint ... --max_samples 2000 --batch_size 64
"""
import os

for _k, _v in (
    ("OMP_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
    ("NUMEXPR_NUM_THREADS", "1"),
    ("VECLIB_MAX_THREADS", "1"),
):
    os.environ.setdefault(_k, _v)

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.base_config import ProjectConfig
from data.transforms import get_val_transforms
from data.dual_branch_dataset import DualBranchDataset
from clip_detector.lightning_module import CLIPDetectorLightning
from evaluation.metrics import compute_metrics

CLIP_SIZE = 224


def main():
    parser = argparse.ArgumentParser(description="Evaluate CLIP detector on a manifest.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to CLIP Lightning .ckpt")
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Default: data_raw/manifests/test_with_faces.csv under PROJECT_ROOT",
    )
    parser.add_argument("--remap_from", type=str, default=None)
    parser.add_argument("--remap_to", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None, help="Cap dataset size for quick runs")
    parser.add_argument("--threshold", type=float, default=0.5, help="For precision/recall/F1")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--time_trial", type=int, default=0, help="Run N extra forward-only batches for timing")
    args = parser.parse_args()

    if (args.remap_from is None) ^ (args.remap_to is None):
        parser.error("Use both --remap_from and --remap_to, or neither.")

    ck = Path(args.checkpoint).expanduser()
    if not ck.is_file():
        parser.error(f"Checkpoint not found: {ck}")

    cfg = ProjectConfig()
    output_dir = args.output_dir or str(cfg.results_dir / "clip")
    os.makedirs(output_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Loading CLIP checkpoint: {ck}")
    model = CLIPDetectorLightning.load_from_checkpoint(str(ck), weights_only=False)
    model.eval()
    model.to(device)

    manifest = args.manifest or str(cfg.data_raw / "manifests" / "test_with_faces.csv")
    manifest = str(Path(manifest).expanduser().resolve())
    if not Path(manifest).is_file():
        parser.error(f"Manifest not found: {manifest}")

    transform = get_val_transforms(CLIP_SIZE)
    test_dataset = DualBranchDataset(
        manifest_csv=manifest,
        face_transform=transform,
        full_transform=transform,
        face_size=CLIP_SIZE,
        random_face_select=False,
        path_prefix_override=args.remap_to,
        original_prefix=args.remap_from,
    )
    n = len(test_dataset)
    if args.max_samples is not None and args.max_samples < n:
        test_dataset.samples = test_dataset.samples[: args.max_samples]
        n = len(test_dataset)

    loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    all_labels = []
    face_scores_all = []
    full_scores_all = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for face_crops, full_images, labels in tqdm(loader, desc="CLIP eval"):
            face_crops = face_crops.to(device)
            full_images = full_images.to(device)
            face_logits, full_logits = model.model(face_crops, full_images)
            face_probs = torch.sigmoid(face_logits).squeeze(1).cpu().numpy()
            full_probs = torch.sigmoid(full_logits).squeeze(1).cpu().numpy()
            all_labels.extend(labels.numpy())
            face_scores_all.extend(face_probs)
            full_scores_all.extend(full_probs)
    elapsed = time.perf_counter() - t0

    all_labels = np.array(all_labels)
    face_scores_all = np.array(face_scores_all)
    full_scores_all = np.array(full_scores_all)

    face_m = compute_metrics(all_labels, face_scores_all, threshold=args.threshold)
    m = compute_metrics(all_labels, full_scores_all, threshold=args.threshold)

    print("\n" + "=" * 60)
    print(f"CLIP Face Branch  (n={n}, threshold={args.threshold})")
    print("=" * 60)
    print(f"  Accuracy:  {face_m['accuracy']:.4f}")
    print(f"  Precision: {face_m['precision']:.4f}")
    print(f"  Recall:    {face_m['recall']:.4f}")
    print(f"  F1:        {face_m['f1']:.4f}")
    print(f"  AUC-ROC:   {face_m['auc_roc']:.4f}")

    print("\n" + "=" * 60)
    print(f"CLIP Full Image Branch  (n={n}, threshold={args.threshold})")
    print("=" * 60)
    print(f"  Accuracy:  {m['accuracy']:.4f}")
    print(f"  Precision: {m['precision']:.4f}")
    print(f"  Recall:    {m['recall']:.4f}")
    print(f"  F1:        {m['f1']:.4f}")
    print(f"  AUC-ROC:   {m['auc_roc']:.4f}")
    print(f"  AP (AUC-PR): {m['average_precision']:.4f}")
    print(f"  EER:       {m['eer']:.4f}")
    print(f"\n  Inference: {elapsed:.2f}s total, {n / elapsed:.1f} img/s (incl. data loading)")
    print(f"  Confusion matrix:\n{m['confusion_matrix']}")
    print(f"\n{m['classification_report']}")

    # Optional micro-benchmark: model-only-ish
    if args.time_trial > 0 and len(test_dataset) > 0:
        batch = next(iter(DataLoader(test_dataset, batch_size=min(args.batch_size, 8), shuffle=False)))
        fc, fi, _ = batch
        fc, fi = fc.to(device), fi.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        for _ in range(args.time_trial):
            with torch.no_grad():
                _ = model.model(fc, fi)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t1
        print(f"\n  Forward-only ({args.time_trial} reps, batch={fc.shape[0]}): {dt/args.time_trial*1000:.2f} ms/rep")

    import json

    out = {k: v for k, v in m.items() if k not in ("fpr", "tpr", "roc_thresholds", "classification_report")}
    out["n_samples"] = int(n)
    out["wall_seconds"] = float(elapsed)
    out["images_per_sec"] = float(n / elapsed) if elapsed > 0 else 0.0
    out["checkpoint"] = str(ck)
    with open(Path(output_dir) / "clip_eval_results.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved {output_dir}/clip_eval_results.json")


if __name__ == "__main__":
    main()
