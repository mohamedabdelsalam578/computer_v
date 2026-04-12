import torch
import torch.nn as nn
import torchvision.models as models
from configs.base_config import FusionConfig


class VGG16DualBranchDetector(nn.Module):
    """
    VGG16 with shared convolutional encoder and two classification heads.

    Architecture:
      - Shared VGG16 features (conv layers, pretrained ImageNet)
      - AdaptiveAvgPool → 512-dim feature vector
      - face_head: Linear(512→256) → ReLU → Dropout(0.5) → Linear(256→1)
      - full_head: same structure, separate weights

    Single batched encoder pass (face+full stacked) keeps VRAM ~2× lower
    than two separate encoders.

    Input: 224×224, ImageNet normalization.
    Output: raw logits (B, 1) — apply sigmoid for probability.
    """

    def __init__(self, pretrained: bool = True, fusion_config: FusionConfig = None):
        super().__init__()
        self.fusion = fusion_config or FusionConfig()

        weights = models.VGG16_Weights.DEFAULT if pretrained else None
        vgg = models.vgg16(weights=weights)

        # Shared convolutional backbone (14.7M params, outputs 512×7×7 for 224×224 in)
        self.encoder = vgg.features
        self.pool    = nn.AdaptiveAvgPool2d((1, 1))   # → (B, 512, 1, 1)

        # Two independent classification heads
        self.face_head = self._build_head(512)
        self.full_head = self._build_head(512)

        total = sum(p.numel() for p in self.parameters()) / 1e6
        print(f"  VGG16DualBranchDetector | encoder=14.7M | total={total:.1f}M params")

    @staticmethod
    def _build_head(in_features: int) -> nn.Sequential:
        head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )
        nn.init.xavier_uniform_(head[0].weight)
        nn.init.zeros_(head[0].bias)
        nn.init.xavier_uniform_(head[3].weight)
        nn.init.zeros_(head[3].bias)
        return head

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """VGG16 features → flattened 512-dim vector."""
        feats = self.encoder(x)          # (B, 512, 7, 7)
        feats = self.pool(feats)         # (B, 512, 1, 1)
        return feats.flatten(1)          # (B, 512)

    def forward(self, face_crop: torch.Tensor, full_image: torch.Tensor):
        """
        Single encoder pass for both branches.

        Args:
            face_crop:  (B, 3, 224, 224)
            full_image: (B, 3, 224, 224)
        Returns:
            face_logit: (B, 1)
            full_logit: (B, 1)
        """
        B = face_crop.shape[0]
        features   = self._encode(torch.cat([face_crop, full_image], dim=0))  # (2B, 512)
        face_logit = self.face_head(features[:B])
        full_logit = self.full_head(features[B:])
        return face_logit, full_logit

    @torch.no_grad()
    def predict(self, face_crops_list: list, full_image: torch.Tensor) -> float:
        """
        Inference with weighted fusion.

        Args:
            face_crops_list: list of (1, 3, 224, 224) tensors
            full_image:      (1, 3, 224, 224)
        Returns:
            float: AI probability [0, 1]
        """
        full_feat  = self._encode(full_image)
        full_prob  = torch.sigmoid(self.full_head(full_feat))[0, 0].item()

        if not face_crops_list:
            return full_prob

        face_scores = [
            torch.sigmoid(self.face_head(self._encode(c)))[0, 0].item()
            for c in face_crops_list
        ]
        return self.fusion.face_weight * max(face_scores) + self.fusion.full_weight * full_prob
