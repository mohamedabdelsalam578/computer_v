import torch
import torch.nn as nn
import pytorch_lightning as pl

from vgg16_detector.vgg16_dual_branch import VGG16DualBranchDetector
from configs.base_config import TrainConfig, FusionConfig


class VGG16DetectorLightning(pl.LightningModule):
    """PyTorch Lightning module for fine-tuning VGG16DualBranchDetector."""

    def __init__(
        self,
        pretrained: bool = True,
        train_config: TrainConfig = None,
        fusion_config: FusionConfig = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.train_cfg = train_config or TrainConfig()
        self.criterion = nn.BCEWithLogitsLoss()
        self.model     = VGG16DualBranchDetector(pretrained=pretrained,
                                                  fusion_config=fusion_config)

        self._val_face_correct  = []
        self._val_full_correct  = []
        self._val_fused_correct = []
        self._val_total         = []

    def forward(self, face_crop, full_image):
        return self.model(face_crop, full_image)

    def training_step(self, batch, batch_idx):
        face_crops, full_images, labels = batch
        labels = labels.float().unsqueeze(1)

        face_logits, full_logits = self.model(face_crops, full_images)
        loss = (self.train_cfg.face_loss_weight * self.criterion(face_logits, labels)
                + self.train_cfg.full_loss_weight * self.criterion(full_logits, labels))

        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        face_crops, full_images, labels = batch
        labels = labels.float().unsqueeze(1)

        face_logits, full_logits = self.model(face_crops, full_images)
        loss = (self.train_cfg.face_loss_weight * self.criterion(face_logits, labels)
                + self.train_cfg.full_loss_weight * self.criterion(full_logits, labels))

        labels_long = labels.long().squeeze(1)
        face_preds  = (torch.sigmoid(face_logits) > 0.5).long().squeeze(1)
        full_preds  = (torch.sigmoid(full_logits) > 0.5).long().squeeze(1)
        face_probs  = torch.sigmoid(face_logits).squeeze(1)
        full_probs  = torch.sigmoid(full_logits).squeeze(1)
        fused_probs = (self.model.fusion.face_weight * face_probs
                       + self.model.fusion.full_weight * full_probs)
        fused_preds = (fused_probs > 0.5).long()

        self._val_face_correct.append((face_preds  == labels_long).sum().item())
        self._val_full_correct.append((full_preds  == labels_long).sum().item())
        self._val_fused_correct.append((fused_preds == labels_long).sum().item())
        self._val_total.append(labels_long.numel())

        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        total = sum(self._val_total)
        if total == 0:
            return

        face_acc  = sum(self._val_face_correct)  / total
        full_acc  = sum(self._val_full_correct)  / total
        fused_acc = sum(self._val_fused_correct) / total

        # Store as attributes so ValMetricsCSV callback can read them reliably
        self._last_face_acc  = face_acc
        self._last_full_acc  = full_acc
        self._last_fused_acc = fused_acc

        self.log('val_face_acc',  face_acc,  prog_bar=False)
        self.log('val_full_acc',  full_acc,  prog_bar=False)
        self.log('val_fused_acc', fused_acc, prog_bar=True)

        self._val_face_correct.clear()
        self._val_full_correct.clear()
        self._val_fused_correct.clear()
        self._val_total.clear()

    def configure_optimizers(self):
        # VGG16 encoder: low lr (pretrained features, don't destroy them)
        # Two heads: full lr (random init, need fast convergence)
        encoder_params = list(self.model.encoder.parameters())
        head_params    = (list(self.model.face_head.parameters())
                          + list(self.model.full_head.parameters()))

        optimizer = torch.optim.AdamW(
            [
                {'params': encoder_params, 'lr': self.train_cfg.learning_rate * 0.1},
                {'params': head_params,    'lr': self.train_cfg.learning_rate},
            ],
            weight_decay=self.train_cfg.weight_decay,
        )
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01,
            total_iters=self.train_cfg.warmup_epochs,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.train_cfg.max_epochs - self.train_cfg.warmup_epochs,
            eta_min=1e-7,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine],
            milestones=[self.train_cfg.warmup_epochs],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
