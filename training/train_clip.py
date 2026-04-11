"""
Fine-tune CLIPDualBranchDetector on GRAVEX-200K.

CLIP ViT-L/14 dual-branch:
  - Face branch processes face crops (224x224)
  - Full branch processes full images (224x224)
  - Both branches start from CLIP ViT-L/14 pretrained weights
  - Differential lr: encoder=lr×0.01, head=lr×1.0

Usage:
    python training/train_clip.py
    python training/train_clip.py --resume lightning_logs_clip/checkpoints/last.ckpt
    python training/train_clip.py --model ViT-B/32   # smaller/faster
    python training/train_clip.py --epochs 20 --lr 3e-4
"""
import os

# Cap BLAS/OMP threads before torch/scipy import
for _k, _v in (
    ("OMP_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
    ("NUMEXPR_NUM_THREADS", "1"),
):
    os.environ.setdefault(_k, _v)

import sys
import argparse
import torch
import torch.multiprocessing as mp
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Hardware detection ─────────────────────────────────────────────────────────
if torch.cuda.is_available():
    ACCELERATOR = 'gpu'
    DEVICE_NAME = torch.cuda.get_device_name(0)
    PIN_MEMORY  = True
    NUM_WORKERS = min(4, os.cpu_count() or 4)   # keep low — CLIP workers can deadlock
    # CLIP ViT-L/14 is ~307M params per branch → 614M total dual-branch
    # RTX 4090 (24GB): 606M dual-branch CLIP needs gradient checkpointing.
    # bs=16 + accum=8 = effective 128. Checkpointing saves ~60% activation memory.
    BATCH_SIZE  = 16
    ACCUM       = 8                # effective bs = 128
    torch.set_float32_matmul_precision('high')
elif torch.backends.mps.is_available():
    ACCELERATOR = 'mps'
    DEVICE_NAME = 'Apple MPS'
    PIN_MEMORY  = False
    NUM_WORKERS = 4
    BATCH_SIZE  = 8               # CLIP is large, MPS memory limited
    ACCUM       = 16              # effective bs = 128
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    torch.set_float32_matmul_precision('high')
else:
    ACCELERATOR = 'cpu'
    DEVICE_NAME = 'CPU'
    PIN_MEMORY  = False
    NUM_WORKERS = 4
    BATCH_SIZE  = 4
    ACCUM       = 32
# ──────────────────────────────────────────────────────────────────────────────

MODEL_MAP = {
    "ViT-L/14": "openai/clip-vit-large-patch14",
    "ViT-B/32":  "openai/clip-vit-base-patch32",
}

from configs.base_config import (
    ProjectConfig, TrainConfig, DataConfig, FusionConfig
)
from data.transforms import get_train_transforms, get_val_transforms
from data.dual_branch_dataset import DualBranchDataset
from clip_detector.lightning_module import CLIPDetectorLightning
from training.callbacks import ValMetricsCSV

CLIP_IMAGE_SIZE = 224   # CLIP ViT-L/14 and ViT-B/32 both use 224×224


def main():
    parser = argparse.ArgumentParser(description="Fine-tune CLIP dual-branch AI detector")
    parser.add_argument('--resume',     type=str,   default=None,
                        help='Resume from Lightning checkpoint (full state).')
    parser.add_argument('--model',      type=str,   default='ViT-L/14',
                        choices=list(MODEL_MAP.keys()))
    parser.add_argument('--batch_size', type=int,   default=None)
    parser.add_argument('--epochs',     type=int,   default=None)
    parser.add_argument('--lr',         type=float, default=None)
    args = parser.parse_args()

    mp.set_start_method('spawn', force=True)

    proj_cfg  = ProjectConfig()
    train_cfg = TrainConfig()
    data_cfg  = DataConfig()
    fusion_cfg = FusionConfig()

    # Hardware defaults
    train_cfg.batch_size  = BATCH_SIZE
    train_cfg.num_workers = NUM_WORKERS

    # CLI overrides
    if args.batch_size: train_cfg.batch_size  = args.batch_size
    if args.epochs:     train_cfg.max_epochs  = args.epochs
    if args.lr:         train_cfg.learning_rate = args.lr

    clip_model_name = MODEL_MAP[args.model]
    manifests_dir   = proj_cfg.data_raw / "manifests"

    # CLIP uses 224×224 — pass size to the shared transform builders
    train_transform = get_train_transforms(CLIP_IMAGE_SIZE)
    val_transform   = get_val_transforms(CLIP_IMAGE_SIZE)

    train_dataset = DualBranchDataset(
        manifest_csv=str(manifests_dir / "train_with_faces.csv"),
        face_transform=train_transform,
        full_transform=train_transform,
        face_size=CLIP_IMAGE_SIZE,
    )
    val_dataset = DualBranchDataset(
        manifest_csv=str(manifests_dir / "val_with_faces.csv"),
        face_transform=val_transform,
        full_transform=val_transform,
        face_size=CLIP_IMAGE_SIZE,
        random_face_select=False,
    )

    _persist = train_cfg.num_workers > 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        persistent_workers=_persist,
        prefetch_factor=2 if _persist else None,
        drop_last=True,
        pin_memory=PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size * 2,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        persistent_workers=_persist,
        prefetch_factor=2 if _persist else None,
        pin_memory=PIN_MEMORY,
    )

    model = CLIPDetectorLightning(
        model_name=clip_model_name,
        train_config=train_cfg,
        fusion_config=fusion_cfg,
    )

    # Separate checkpoint dir for CLIP (keeps FerretNet checkpoints intact)
    proj_cfg.results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = proj_cfg.project_root / "lightning_logs_clip" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=str(ckpt_dir),
            monitor='val_fused_acc',
            mode='max',
            save_top_k=3,
            filename='clip-{epoch:02d}-{val_fused_acc:.4f}',
            save_last=True,
        ),
        pl.callbacks.EarlyStopping(
            monitor='val_fused_acc',
            patience=train_cfg.early_stopping_patience,
            mode='max',
            verbose=True,
        ),
        pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
        ValMetricsCSV(csv_path=str(proj_cfg.results_dir / "val_metrics_clip.csv")),
    ]

    trainer = pl.Trainer(
        accelerator=ACCELERATOR,
        devices=1,
        max_epochs=train_cfg.max_epochs,
        gradient_clip_val=train_cfg.gradient_clip_val,
        precision=train_cfg.precision,
        accumulate_grad_batches=ACCUM,
        callbacks=callbacks,
        default_root_dir=str(proj_cfg.project_root),
        log_every_n_steps=50,
        benchmark=(ACCELERATOR == 'gpu'),
        num_sanity_val_steps=0,   # skip sanity check — gradient checkpointing + eval deadlocks
    )

    eff_bs = train_cfg.batch_size * ACCUM
    steps  = len(train_dataset) // eff_bs
    print(f"\n{'='*60}")
    print(f"  Model:           CLIP {args.model}  ({clip_model_name})")
    print(f"  Device:          {DEVICE_NAME}")
    print(f"  Accelerator:     {ACCELERATOR.upper()}")
    print(f"  Batch size:      {train_cfg.batch_size} × accum {ACCUM} = {eff_bs} effective")
    print(f"  Workers:         {train_cfg.num_workers}")
    print(f"  Precision:       {train_cfg.precision}")
    print(f"  Steps/epoch:     {steps:,}")
    print(f"  Train samples:   {len(train_dataset):,}")
    print(f"  Val samples:     {len(val_dataset):,}")
    print(f"  Max epochs:      {train_cfg.max_epochs}  (patience={train_cfg.early_stopping_patience})")
    print(f"  Checkpoints:     {ckpt_dir.resolve()}")
    print(f"  Val metrics CSV: {(proj_cfg.results_dir / 'val_metrics_clip.csv').resolve()}")
    print(f"{'='*60}\n")

    trainer.fit(
        model,
        train_loader,
        val_loader,
        ckpt_path=args.resume,
        weights_only=False if args.resume else None,
    )


if __name__ == '__main__':
    main()
