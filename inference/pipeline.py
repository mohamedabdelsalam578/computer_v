"""End-to-end inference pipeline: RetinaFace + DualBranchFerretNet."""

import torch
from PIL import Image
from ferretnet.dual_branch import DualBranchFerretNet
from retinaface.detector import RetinaFaceDetector
from data.transforms import get_val_transforms
from configs.base_config import FusionConfig


class FerretNetPipeline:
    """Full inference pipeline for AI vs Real image detection."""

    def __init__(
        self,
        checkpoint_path: str,
        retinaface_weights: str,
        ferretnet_pretrained: str = None,
        device: str = "mps",
    ):
        self.device = torch.device(device if (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()) or device == "cpu" else "cpu")

        # Load RetinaFace
        self.face_detector = RetinaFaceDetector(
            weights_path=retinaface_weights,
            device=str(self.device),
        )

        # Load FerretNet from Lightning checkpoint
        from ferretnet.lightning_module import FerretNetLightning
        lightning_model = FerretNetLightning.load_from_checkpoint(
            checkpoint_path,
            weights_only=False,
        )
        self.model = lightning_model.model
        self.model.to(self.device)
        self.model.eval()

        self.transform = get_val_transforms()

    @torch.no_grad()
    def predict(self, image_path: str) -> dict:
        """
        Full inference on a single image.

        Returns dict with:
            label: "AI" or "Real"
            confidence: float
            faces_detected: int
            face_scores: list[float]
            full_score: float
            fused_score: float
        """
        image = Image.open(image_path).convert('RGB')

        # Detect faces
        face_crops = self.face_detector.detect_faces(image, crop_size=256, margin=0.20)

        # Prepare full image tensor
        full_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Full branch
        full_logit = self.model.forward_full_branch(full_tensor)
        full_prob = torch.sigmoid(full_logit)[0, 0].item()

        # Face branch
        face_probs = []
        face_tensors = []
        for crop in face_crops:
            crop_tensor = self.transform(crop).unsqueeze(0).to(self.device)
            face_tensors.append(crop_tensor)
            logit = self.model.forward_face_branch(crop_tensor)
            prob = torch.sigmoid(logit)[0, 0].item()
            face_probs.append(prob)

        # Fuse
        if face_probs:
            max_face = max(face_probs)
            fused = self.model.fusion.face_weight * max_face + self.model.fusion.full_weight * full_prob
        else:
            fused = full_prob

        label = "AI" if fused > self.model.fusion.threshold else "Real"

        return {
            'label': label,
            'confidence': abs(fused - 0.5) * 2,  # 0-1 scale, higher = more confident
            'faces_detected': len(face_crops),
            'face_scores': face_probs,
            'full_score': full_prob,
            'fused_score': fused,
        }
