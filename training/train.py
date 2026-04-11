"""
Fine-tune DualBranchFerretNet on GRAVEX-200K.

Auto-detects hardware: CUDA (RunPod) → MPS (Mac) → CPU fallback.

Usage:
    python training/train.py
    python training/train.py --resume lightning_logs/version_0/checkpoints/last.ckpt
    python training/train.py --batch_size 128 --epochs 15
"""
import os
import sys
import argparse
import torch
import torch.multiprocessing as mp
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Detect hardware and configure accordingly ──────────────────────────────────
if torch.cuda.is_available():
    ACCELERATOR   = 'gpu'
    DEVICE_NAME   = torch.cuda.get_device_name(0)
    PIN_MEMORY    = True       # CUDA benefits from pinned memory
    NUM_WORKERS   = min(16, os.cpu_count())
    BATCH_SIZE    = 64         # RTX 4090 max for dual-branch (12.5GB/23GB)
    ACCUM         = 2          # effective bs=128 via gradient accumulation
    torch.set_float32_matmul_precision('high')  # TF32 on CUDA
elif torch.backends.mps.is_available():
    ACCELERATOR   = 'mps'
    DEVICE_NAME   = 'Apple MPS'
    PIN_MEMORY    = False      # unified memory — pinning is a no-op
    NUM_WORKERS   = 8          # 8 of 10 logical cores
    BATCH_SIZE    = 64         # fits in 24 GB with accum
    ACCUM         = 2          # effective batch = 128
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    os.environ['PYTORCH_MPS_ALLOCATOR_POLICY'] = 'garbage_collection'
    torch.set_float32_matmul_precision('high')  # TF32 on M-series
else:
    ACCELERATOR   = 'cpu'
    DEVICE_NAME   = 'CPU'
    PIN_MEMORY    = False
    NUM_WORKERS   = 4
    BATCH_SIZE    = 16
    ACCUM         = 4
# ──────────────────────────────────────────────────────────────────────────────

from configs.base_config import (
    ProjectConfig, FerretNetConfig, TrainConfig, DataConfig, FusionConfig
)
from data.transforms import get_train_transforms, get_val_transforms
from data.dual_branch_dataset import DualBranchDataset
from ferretnet.lightning_module import FerretNetLightning
from training.callbacks import ValMetricsCSV


def main():
    parser = argparse.ArgumentParser(description="Fine-tune DualBranchFerretNet")
    parser.add_argument('--resume',     type=str,   default=None)
    parser.add_argument('--batch_size', type=int,   default=None)
    parser.add_argument('--epochs',     type=int,   default=None)
    parser.add_argument('--lr',         type=float, default=None)
    args = parser.parse_args()

    # spawn required for MPS; also safe for CUDA
    mp.set_start_method('spawn', force=True)

    proj_cfg   = ProjectConfig()
    model_cfg  = FerretNetConfig()
    train_cfg  = TrainConfig()
    data_cfg   = DataConfig()
    fusion_cfg = FusionConfig()

    # Override defaults with hardware-optimal values
    train_cfg.batch_size  = BATCH_SIZE
    train_cfg.num_workers = NUM_WORKERS

    # CLI overrides
    if args.batch_size: train_cfg.batch_size = args.batch_size
    if args.epochs:     train_cfg.max_epochs = args.epochs
    if args.lr:         train_cfg.learning_rate = args.lr

    manifests_dir   = proj_cfg.data_raw / "manifests"
    train_transform = get_train_transforms(data_cfg.full_image_size)
    val_transform   = get_val_transforms(data_cfg.full_image_size)

    train_dataset = DualBranchDataset(
        manifest_csv=str(manifests_dir / "train_with_faces.csv"),
        face_transform=train_transform,
        full_transform=train_transform,
        face_size=data_cfg.face_crop_size,
    )
    val_dataset = DualBranchDataset(
        manifest_csv=str(manifests_dir / "val_with_faces.csv"),
        face_transform=val_transform,
        full_transform=val_transform,
        face_size=data_cfg.face_crop_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=True,
        pin_memory=PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size * 2,   # no backward → safe to double
        shuffle=False,
        num_workers=train_cfg.num_workers,
        persistent_workers=True,
        prefetch_factor=4,
        pin_memory=PIN_MEMORY,
    )

    model = FerretNetLightning(
        ferretnet_config=model_cfg,
        train_config=train_cfg,
        fusion_config=fusion_cfg,
        pretrained_path=str(proj_cfg.ferretnet_weights),
    )

    proj_cfg.results_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            monitor='val_fused_acc',
            mode='max',
            save_top_k=3,
            filename='ferretnet-{epoch:02d}-{val_fused_acc:.4f}',
            save_last=True,
        ),
        pl.callbacks.EarlyStopping(
            monitor='val_fused_acc',
            patience=train_cfg.early_stopping_patience,
            mode='max',
            verbose=True,
        ),
        pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
        ValMetricsCSV(csv_path=str(proj_cfg.results_dir / "val_metrics.csv")),
    ]

    trainer = pl.Trainer(
        accelerator=ACCELERATOR,
        devices=1,
        max_epochs=train_cfg.max_epochs,
        gradient_clip_val=train_cfg.gradient_clip_val,
        precision=train_cfg.precision,       # '16-mixed' on all hardware
        accumulate_grad_batches=ACCUM,
        callbacks=callbacks,
        default_root_dir=str(proj_cfg.lightning_logs),
        log_every_n_steps=50,
        benchmark=(ACCELERATOR == 'gpu'),    # cuDNN benchmark only on CUDA
    )

    eff_bs = train_cfg.batch_size * ACCUM
    steps  = len(train_dataset) // eff_bs
    print(f"\n{'='*60}")
    print(f"  Device:          {DEVICE_NAME}")
    print(f"  Accelerator:     {ACCELERATOR.upper()}")
    print(f"  Batch size:      {train_cfg.batch_size} × accum {ACCUM} = {eff_bs} effective")
    print(f"  Workers:         {train_cfg.num_workers}")
    print(f"  Precision:       {train_cfg.precision}")
    print(f"  Steps/epoch:     {steps:,}")
    print(f"  Train samples:   {len(train_dataset):,}")
    print(f"  Val samples:     {len(val_dataset):,}")
    print(f"  Max epochs:      {train_cfg.max_epochs}  (early stop patience={train_cfg.early_stopping_patience})")
    print(f"{'='*60}\n")

    trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume)


if __name__ == '__main__':
    main()
