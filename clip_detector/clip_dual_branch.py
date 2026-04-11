import torch
import torch.nn as nn
from transformers import CLIPVisionModel
from configs.base_config import FusionConfig


class CLIPDualBranchDetector(nn.Module):
    """
    Dual-branch AI image detector using CLIP ViT vision encoder.

    Face branch: CLIP ViT on face crops (224x224)
    Full branch: CLIP ViT on full images (224x224)
    Fusion: 0.6 * max(face_scores) + 0.4 * full_score

    Fine-tuning strategy:
        - CLIP encoders: lr × 0.01  (preserve rich pretrained features)
        - Classification heads: lr × 1.0  (fast adaptation, fresh weights)
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        fusion_config: FusionConfig = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.fusion = fusion_config or FusionConfig()

        print(f"  Loading CLIP vision encoder: {model_name}")
        self.face_encoder = CLIPVisionModel.from_pretrained(model_name)
        self.full_encoder = CLIPVisionModel.from_pretrained(model_name)

        # hidden_size: ViT-L/14 → 1024, ViT-B/32 → 768
        hidden_size = self.face_encoder.config.hidden_size

        # Classification heads — initialized fresh (not pretrained)
        self.face_head = nn.Linear(hidden_size, 1)
        self.full_head = nn.Linear(hidden_size, 1)
        nn.init.xavier_uniform_(self.face_head.weight)
        nn.init.zeros_(self.face_head.bias)
        nn.init.xavier_uniform_(self.full_head.weight)
        nn.init.zeros_(self.full_head.bias)

        # Gradient checkpointing: trades compute for memory (~60% less VRAM)
        self.face_encoder.gradient_checkpointing_enable()
        self.full_encoder.gradient_checkpointing_enable()

        # CLIPVisionModel.from_pretrained() leaves internal modules in eval mode.
        # Explicitly set train mode so dropout/checkpointing work correctly.
        self.face_encoder.train()
        self.full_encoder.train()

        print(f"  CLIP hidden_size: {hidden_size}  |  Total params: "
              f"{sum(p.numel() for p in self.parameters()) / 1e6:.1f}M  |  "
              f"gradient checkpointing: ON")

    def train(self, mode: bool = True):
        """Override to ensure CLIP encoder submodules follow train/eval mode."""
        super().train(mode)
        # Propagate explicitly — CLIPVisionModel internals can get stuck in eval
        self.face_encoder.train(mode)
        self.full_encoder.train(mode)
        return self

    def _encode(self, encoder: CLIPVisionModel, x: torch.Tensor) -> torch.Tensor:
        """Returns pooled CLS token features: (B, hidden_size)."""
        return encoder(pixel_values=x).pooler_output

    def forward(self, face_crop: torch.Tensor, full_image: torch.Tensor):
        """
        Args:
            face_crop:  (B, 3, 224, 224)
            full_image: (B, 3, 224, 224)
        Returns:
            face_logits: (B, 1)
            full_logits: (B, 1)
        """
        face_logit = self.face_head(self._encode(self.face_encoder, face_crop))
        full_logit = self.full_head(self._encode(self.full_encoder, full_image))
        return face_logit, full_logit

    def forward_face_branch(self, x: torch.Tensor) -> torch.Tensor:
        return self.face_head(self._encode(self.face_encoder, x))

    def forward_full_branch(self, x: torch.Tensor) -> torch.Tensor:
        return self.full_head(self._encode(self.full_encoder, x))

    @torch.no_grad()
    def predict(self, face_crops_list: list, full_image: torch.Tensor) -> float:
        """
        Inference with fusion.

        Args:
            face_crops_list: list of (1, 3, 224, 224) tensors (one per detected face)
            full_image:      (1, 3, 224, 224)
        Returns:
            float: fused probability of being AI-generated [0, 1]
        """
        full_prob = torch.sigmoid(self.forward_full_branch(full_image))[0, 0].item()

        if not face_crops_list:
            return full_prob

        face_scores = [
            torch.sigmoid(self.forward_face_branch(crop))[0, 0].item()
            for crop in face_crops_list
        ]
        max_face_score = max(face_scores)
        return self.fusion.face_weight * max_face_score + self.fusion.full_weight * full_prob
