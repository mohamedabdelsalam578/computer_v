import os
import csv
from datetime import datetime

import pytorch_lightning as pl


class ValMetricsCSV(pl.Callback):
    """Append validation metrics to CSV after each validation epoch.
    Persists across resume — always has complete training history."""

    def __init__(self, csv_path: str = "results/val_metrics.csv"):
        self.csv_path = csv_path
        self.fieldnames = [
            'epoch', 'train_loss', 'val_loss',
            'val_face_acc', 'val_full_acc', 'val_fused_acc',
            'lr', 'timestamp',
        ]
        # Write header if file doesn't exist
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        if not os.path.exists(csv_path):
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        lr = trainer.optimizers[0].param_groups[0]['lr'] if trainer.optimizers else 0.0

        # Read accuracies directly from module attributes (self.log may
        # not populate callback_metrics before callbacks run)
        face_acc = getattr(pl_module, '_last_face_acc', None)
        full_acc = getattr(pl_module, '_last_full_acc', None)
        fused_acc = getattr(pl_module, '_last_fused_acc', None)

        row = {
            'epoch': trainer.current_epoch,
            'train_loss': _item(metrics.get('train_loss')),
            'val_loss': _item(metrics.get('val_loss')),
            'val_face_acc': face_acc if face_acc is not None else _item(metrics.get('val_face_acc')),
            'val_full_acc': full_acc if full_acc is not None else _item(metrics.get('val_full_acc')),
            'val_fused_acc': fused_acc if fused_acc is not None else _item(metrics.get('val_fused_acc')),
            'lr': lr,
            'timestamp': datetime.now().isoformat(),
        }
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


def _item(val):
    if val is None:
        return ''
    if hasattr(val, 'item'):
        return val.item()
    return val
