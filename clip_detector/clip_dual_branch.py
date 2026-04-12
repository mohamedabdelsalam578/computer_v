import torch
import torch.nn as nn
from transformers import CLIPVisionModel
from configs.base_config import FusionConfig


class CLIPDualBranchDetector(nn.Module):
    """
    CLIP ViT-L/14 with shared encoder and two classification heads.

    Key design: ONE shared encoder processes face+full in a single batched
    forward pass. This halves VRAM vs two separate encoders (307M vs 606M),
    eliminates OOM, and allows bs=64 without gradient checkpointing.

    Face branch and full branch produce separate logits via independent heads.
    The shared encoder learns universal AI-vs-real features; each head adapts
    those features to its input domain (face crop vs full image).

    Each face and the full image are classified independently — no fusion.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        fusion_config: FusionConfig = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.fusion = fusion_config or FusionConfig()

        print(f"  Loading shared CLIP encoder: {model_name}")
        # use_safetensors=True avoids torch.load CVE-2025-32434 check
        # (safetensors format is safe regardless of torch version)
        self.encoder = CLIPVisionModel.from_pretrained(model_name, use_safetensors=True)
        self.encoder.train()   # HuggingFace models can start with submodules in eval

        hidden_size = self.encoder.config.hidden_size  # ViT-L/14 → 1024

        # Two independent heads — shared features, separate decisions
        self.face_head = nn.Linear(hidden_size, 1)
        self.full_head = nn.Linear(hidden_size, 1)
        nn.init.xavier_uniform_(self.face_head.weight)
        nn.init.zeros_(self.face_head.bias)
        nn.init.xavier_uniform_(self.full_head.weight)
        nn.init.zeros_(self.full_head.bias)

        total_params = sum(p.numel() for p in self.parameters()) / 1e6
        print(f"  hidden_size={hidden_size}  |  params={total_params:.1f}M  "
              f"(shared encoder — half the VRAM of dual-encoder)")

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.train(mode)
        return self

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """L2-pool the CLS token from the CLIP vision encoder."""
        return self.encoder(pixel_values=x).pooler_output  # (B, hidden_size)

    def forward(self, face_crop: torch.Tensor, full_image: torch.Tensor):
        """
        Single encoder pass for both branches — efficient and VRAM-friendly.

        Args:
            face_crop:  (B, 3, 224, 224)
            full_image: (B, 3, 224, 224)
        Returns:
            face_logits: (B, 1)
            full_logits: (B, 1)
        """
        B = face_crop.shape[0]
        # One pass: (2B, 3, 224, 224) → (2B, hidden_size)
        features = self._encode(torch.cat([face_crop, full_image], dim=0))
        face_logit = self.face_head(features[:B])
        full_logit = self.full_head(features[B:])
        return face_logit, full_logit

    @torch.no_grad()
    def predict(self, face_crops_list: list, full_image: torch.Tensor):
        """
        Independent inference — no fusion.

        Args:
            face_crops_list: list of (1, 3, 224, 224) tensors
            full_image:      (1, 3, 224, 224)
        Returns:
            (full_prob, face_scores): full image P(AI) and list of per-face P(AI)
        """
        full_feat = self._encode(full_image)
        full_prob = torch.sigmoid(self.full_head(full_feat))[0, 0].item()

        face_scores = []
        for crop in face_crops_list:
            feat = self._encode(crop)
            score = torch.sigmoid(self.face_head(feat))[0, 0].item()
            face_scores.append(score)

        return full_prob, face_scores
