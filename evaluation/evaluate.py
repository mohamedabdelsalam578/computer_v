"""
Evaluate on a split manifest (face + full + fused metrics).

Usage:
    python evaluation/evaluate.py --checkpoint path/to/best.ckpt
    python evaluation/evaluate.py --pretrained_pth weights/ferretnet-b-median-3.pth \\
        --manifest data_raw/manifests/test_with_faces.csv
"""
import os

for _k, _v in (
    ("OMP_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
    ("NUMEXPR_NUM_THREADS", "1"),
    ("VECLIB_MAXIMUM_THREADS", "1"),
):
    os.environ.setdefault(_k, _v)

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from configs.base_config import (
    ProjectConfig,
    DataConfig,
    FusionConfig,
    FerretNetConfig,
    TrainConfig,
)
from data.transforms import get_val_transforms
from data.dual_branch_dataset import DualBranchDataset
from ferretnet.lightning_module import FerretNetLightning
from evaluation.metrics import compute_metrics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate dual-branch FerretNet on a *_with_faces.csv manifest.",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--checkpoint",
        type=str,
        help="Lightning .ckpt from training (fine-tuned weights).",
    )
    src.add_argument(
        "--pretrained_pth",
        type=str,
        help="Raw ferretnet-b-median-3.pth only: backbone + reinit head (same as train init, not fine-tuned).",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to test_with_faces.csv (default: data_raw/manifests/test_with_faces.csv under PROJECT_ROOT).",
    )
    parser.add_argument(
        "--remap_from",
        type=str,
        default=None,
        help="If CSV paths point elsewhere (e.g. RunPod), prefix to replace (e.g. /workspace/AI_COMPUTER_VISION).",
    )
    parser.add_argument(
        "--remap_to",
        type=str,
        default=None,
        help="Local prefix (e.g. /Users/you/.../AI_COMPUTER_VISION); use with --remap_from.",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Eval batch size (lower if MPS OOM).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="DataLoader workers on Mac; use 0 if multiprocessing issues.",
    )
    args = parser.parse_args()

    if (args.remap_from is None) ^ (args.remap_to is None):
        parser.error("Use both --remap_from and --remap_to, or neither.")

    if args.checkpoint:
        _ck = Path(args.checkpoint).expanduser()
        if not _ck.is_file():
            parser.error(
                f"Checkpoint not found: {_ck.resolve()}\n"
                "  Pass the real .ckpt path (not a placeholder). Examples:\n"
                "    --checkpoint lightning_logs/checkpoints/last.ckpt\n"
                "    --checkpoint ~/Downloads/ferretnet-epoch=10-val_fused_acc=0.6100.ckpt"
            )
    else:
        _pw = Path(args.pretrained_pth).expanduser()
        if not _pw.is_file():
            parser.error(f"Pretrained file not found: {_pw.resolve()}")

    cfg = ProjectConfig()
    data_cfg = DataConfig()
    output_dir = args.output_dir or str(cfg.results_dir / "evaluation")
    os.makedirs(output_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        model = FerretNetLightning.load_from_checkpoint(
            args.checkpoint,
            weights_only=False,
        )
    else:
        pth = Path(args.pretrained_pth).expanduser().resolve()
        print(f"Loading pretrained weights (no fine-tune ckpt): {pth}")
        model = FerretNetLightning(
            ferretnet_config=FerretNetConfig(),
            train_config=TrainConfig(),
            fusion_config=FusionConfig(),
            pretrained_path=str(pth),
        )
    model.eval()
    model.to(device)

    manifest = args.manifest or str(cfg.data_raw / "manifests" / "test_with_faces.csv")
    manifest = str(Path(manifest).expanduser().resolve())
    if not Path(manifest).is_file():
        parser.error(f"Manifest not found: {manifest}")
    transform = get_val_transforms(data_cfg.full_image_size)
    test_dataset = DualBranchDataset(
        manifest_csv=manifest,
        face_transform=transform,
        full_transform=transform,
        face_size=data_cfg.face_crop_size,
        random_face_select=False,
        path_prefix_override=args.remap_to,
        original_prefix=args.remap_from,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Run evaluation
    all_labels = []
    face_scores = []
    full_scores = []
    fused_scores = []

    fusion = model.model.fusion

    with torch.no_grad():
        for face_crops, full_images, labels in tqdm(test_loader, desc="Evaluating"):
            face_crops = face_crops.to(device)
            full_images = full_images.to(device)

            face_logits, full_logits = model.model(face_crops, full_images)
            face_probs = torch.sigmoid(face_logits).squeeze(1).cpu().numpy()
            full_probs = torch.sigmoid(full_logits).squeeze(1).cpu().numpy()
            fused_probs = fusion.face_weight * face_probs + fusion.full_weight * full_probs

            all_labels.extend(labels.numpy())
            face_scores.extend(face_probs)
            full_scores.extend(full_probs)
            fused_scores.extend(fused_probs)

    all_labels = np.array(all_labels)
    face_scores = np.array(face_scores)
    full_scores = np.array(full_scores)
    fused_scores = np.array(fused_scores)

    # Compute metrics for each branch
    print("\n" + "=" * 60)
    print("FACE BRANCH ONLY")
    print("=" * 60)
    face_metrics = compute_metrics(all_labels, face_scores)
    print_metrics(face_metrics)

    print("\n" + "=" * 60)
    print("FULL IMAGE BRANCH ONLY")
    print("=" * 60)
    full_metrics = compute_metrics(all_labels, full_scores)
    print_metrics(full_metrics)

    print("\n" + "=" * 60)
    print("FUSED (0.6 * face + 0.4 * full)")
    print("=" * 60)
    fused_metrics = compute_metrics(all_labels, fused_scores)
    print_metrics(fused_metrics)

    # Plot ROC curves
    plot_roc(all_labels, face_scores, full_scores, fused_scores, output_dir)

    # Save results
    save_results(face_metrics, full_metrics, fused_metrics, output_dir)
    print(f"\nResults saved to {output_dir}")


def print_metrics(m):
    print(f"  Accuracy:  {m['accuracy']:.4f}")
    print(f"  AUC-ROC:   {m['auc_roc']:.4f}")
    print(f"  AP:        {m['average_precision']:.4f}")
    print(f"  EER:       {m['eer']:.4f} (threshold={m['eer_threshold']:.4f})")
    print(f"  Optimal:   threshold={m['optimal_threshold']:.4f}")
    print(f"  Confusion Matrix:\n{m['confusion_matrix']}")
    print(f"\n{m['classification_report']}")


def plot_roc(y_true, face_scores, full_scores, fused_scores, output_dir):
    from sklearn.metrics import roc_curve
    plt.figure(figsize=(8, 6))
    for scores, label in [(face_scores, 'Face Branch'), (full_scores, 'Full Branch'), (fused_scores, 'Fused')]:
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)
        plt.plot(fpr, tpr, label=f'{label} (AUC={auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/roc_curves.png", dpi=150)
    plt.close()


def save_results(face_m, full_m, fused_m, output_dir):
    import json
    results = {
        'face_branch': {k: v for k, v in face_m.items() if not isinstance(v, np.ndarray)},
        'full_branch': {k: v for k, v in full_m.items() if not isinstance(v, np.ndarray)},
        'fused': {k: v for k, v in fused_m.items() if not isinstance(v, np.ndarray)},
    }
    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(f"{output_dir}/results.json", 'w') as f:
        json.dump(results, f, default=convert, indent=2)


if __name__ == '__main__':
    from sklearn.metrics import roc_auc_score
    main()
