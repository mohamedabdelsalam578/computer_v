"""
Fine-tune VGG16DualBranchDetector on GRAVEX-200K.

VGG16 dual-branch (ImageNet pretrained, 14.7M encoder params):
  - Face branch processes face crops (224×224)
  - Full branch processes full images (224×224)
  - Shared encoder, differential lr: encoder=lr×0.1, heads=lr×1.0

Usage:
    python training/train_vgg16.py
    python training/train_vgg16.py --resume lightning_logs_vgg16/checkpoints/last.ckpt
    python training/train_vgg16.py --epochs 25 --lr 1e-4
    python training/train_vgg16.py --batch_size 256 --num_workers 16
"""
import os

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
    # VGG16 encoder is 14.7M params — much lighter than CLIP (87M).
    # Push bs=512 so GPU compute stays saturated; VGG16 fits easily in 24GB.
    NUM_WORKERS = 16
    BATCH_SIZE  = 512
    ACCUM       = 1              # effective bs = 512 (no accumulation needed)
    torch.set_float32_matmul_precision('high')
elif torch.backends.mps.is_available():
    ACCELERATOR = 'mps'
    DEVICE_NAME = 'Apple MPS'
    PIN_MEMORY  = False
    NUM_WORKERS = 4
    BATCH_SIZE  = 32
    ACCUM       = 4              # effective bs = 128
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    torch.set_float32_matmul_precision('high')
else:
    ACCELERATOR = 'cpu'
    DEVICE_NAME = 'CPU'
    PIN_MEMORY  = False
    NUM_WORKERS = 4
    BATCH_SIZE  = 32
    ACCUM       = 4
# ──────────────────────────────────────────────────────────────────────────────

from configs.base_config import (
    ProjectConfig, TrainConfig, DataConfig, FusionConfig
)
from data.transforms import get_train_transforms, get_val_transforms
from data.dual_branch_dataset import DualBranchDataset
from vgg16_detector.lightning_module import VGG16DetectorLightning
from training.callbacks import ValMetricsCSV

VGG_IMAGE_SIZE = 224   # VGG16 standard input size


def main():
    parser = argparse.ArgumentParser(description="Fine-tune VGG16 dual-branch AI detector")
    parser.add_argument('--resume',      type=str,   default=None,
                        help='Resume from Lightning checkpoint.')
    parser.add_argument('--batch_size',  type=int,   default=None)
    parser.add_argument('--accum_grad_batches', type=int, default=None)
    parser.add_argument('--num_workers', type=int,   default=None)
    parser.add_argument('--epochs',      type=int,   default=None)
    parser.add_argument('--lr',          type=float, default=None)
    args = parser.parse_args()

    mp.set_start_method('spawn', force=True)

    proj_cfg   = ProjectConfig()
    train_cfg  = TrainConfig()
    fusion_cfg = FusionConfig()

    # Hardware defaults
    train_cfg.batch_size  = BATCH_SIZE
    train_cfg.num_workers = NUM_WORKERS

    # CLI overrides
    if args.batch_size:           train_cfg.batch_size    = args.batch_size
    if args.num_workers is not None: train_cfg.num_workers = args.num_workers
    if args.epochs:               train_cfg.max_epochs    = args.epochs
    if args.lr:                   train_cfg.learning_rate = args.lr

    accum_grad = ACCUM if args.accum_grad_batches is None else args.accum_grad_batches

    manifests_dir = proj_cfg.data_raw / "manifests"

    train_transform = get_train_transforms(VGG_IMAGE_SIZE)
    val_transform   = get_val_transforms(VGG_IMAGE_SIZE)

    train_dataset = DualBranchDataset(
        manifest_csv=str(manifests_dir / "train_with_faces.csv"),
        face_transform=train_transform,
        full_transform=train_transform,
        face_size=VGG_IMAGE_SIZE,
    )
    val_dataset = DualBranchDataset(
        manifest_csv=str(manifests_dir / "val_with_faces.csv"),
        face_transform=val_transform,
        full_transform=val_transform,
        face_size=VGG_IMAGE_SIZE,
        random_face_select=False,
    )

    _persist = train_cfg.num_workers > 0
    _resuming = bool(args.resume)
    # VGG16 has no gradient checkpointing — persistent workers safe even on resume
    _persistent_ok = _persist
    _prefetch = 8 if ACCELERATOR == "gpu" else (2 if _persist else None)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        persistent_workers=_persistent_ok,
        prefetch_factor=_prefetch,
        drop_last=True,
        pin_memory=PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size * 2,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        persistent_workers=_persistent_ok,
        prefetch_factor=_prefetch,
        pin_memory=PIN_MEMORY,
    )

    model = VGG16DetectorLightning(
        pretrained=True,
        train_config=train_cfg,
        fusion_config=fusion_cfg,
    )

    # torch.compile gives ~20-40% speedup on small models like VGG16 (PyTorch ≥ 2.0)
    if ACCELERATOR == 'gpu' and hasattr(torch, 'compile'):
        print("  Compiling model with torch.compile (mode=reduce-overhead)...")
        model.model = torch.compile(model.model, mode='reduce-overhead')

    proj_cfg.results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = proj_cfg.project_root / "lightning_logs_vgg16" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=str(ckpt_dir),
            monitor='val_fused_acc',
            mode='max',
            save_top_k=3,
            filename='vgg16-{epoch:02d}-{val_fused_acc:.4f}',
            save_last=True,
        ),
        pl.callbacks.EarlyStopping(
            monitor='val_fused_acc',
            patience=train_cfg.early_stopping_patience,
            mode='max',
            verbose=True,
        ),
        pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
        ValMetricsCSV(csv_path=str(proj_cfg.results_dir / "val_metrics_vgg16.csv")),
    ]

    trainer = pl.Trainer(
        accelerator=ACCELERATOR,
        devices=1,
        max_epochs=train_cfg.max_epochs,
        gradient_clip_val=train_cfg.gradient_clip_val,
        precision=train_cfg.precision,
        accumulate_grad_batches=accum_grad,
        callbacks=callbacks,
        default_root_dir=str(proj_cfg.project_root),
        log_every_n_steps=50,
        benchmark=(ACCELERATOR == 'gpu'),
        num_sanity_val_steps=0,
    )

    eff_bs = train_cfg.batch_size * accum_grad
    steps  = len(train_dataset) // eff_bs
    print(f"\n{'='*60}")
    compiled = ACCELERATOR == 'gpu' and hasattr(torch, 'compile')
    print(f"  Model:           VGG16 dual-branch (ImageNet pretrained){' + torch.compile' if compiled else ''}")
    print(f"  Device:          {DEVICE_NAME}")
    print(f"  Accelerator:     {ACCELERATOR.upper()}")
    print(f"  Batch size:      {train_cfg.batch_size} × accum {accum_grad} = {eff_bs} effective")
    print(f"  Workers:         {train_cfg.num_workers}")
    print(f"  Precision:       {train_cfg.precision}")
    print(f"  Steps/epoch:     {steps:,}")
    print(f"  Train samples:   {len(train_dataset):,}")
    print(f"  Val samples:     {len(val_dataset):,}")
    print(f"  Max epochs:      {train_cfg.max_epochs}  (patience={train_cfg.early_stopping_patience})")
    print(f"  Checkpoints:     {ckpt_dir.resolve()}")
    print(f"  Val metrics CSV: {(proj_cfg.results_dir / 'val_metrics_vgg16.csv').resolve()}")
    if _resuming:
        print("  Resume:          persistent_workers=OFF (avoids post-restore dataloader freezes)")
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
