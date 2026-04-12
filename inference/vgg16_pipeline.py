import time
import glob
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image
import torch
import torchvision.transforms as T

from inference.base_pipeline import BasePipeline, PipelineResult
from inference.face_detector import detect_faces
from configs.base_config import (
    ProjectConfig, FusionConfig,
    VGG_NORMALIZE_MEAN, VGG_NORMALIZE_STD,
)

IMAGE_SIZE = 224   # VGG16 standard input


def _find_weights(proj: ProjectConfig) -> Tuple[Optional[Path], str]:
    """
    Returns (path, kind) where kind is 'pt' (exported) or 'ckpt' (Lightning).
    Priority: exported weights/vgg16_finetuned.pt → best .ckpt → last.ckpt
    """
    pt_path = proj.weights_dir / "vgg16_finetuned.pt"
    if pt_path.exists():
        return pt_path, "pt"

    ckpt_dir = proj.project_root / "lightning_logs_vgg16" / "checkpoints"
    if ckpt_dir.exists():
        candidates = [p for p in glob.glob(str(ckpt_dir / "vgg16-*.ckpt")) if 'last' not in p]
        if candidates:
            best = Path(sorted(candidates, key=lambda p: p.split('=')[-1], reverse=True)[0])
            return best, "ckpt"
        last = ckpt_dir / "last.ckpt"
        if last.exists():
            return last, "ckpt"

    return None, "none"


class VGG16Pipeline(BasePipeline):
    name = "VGG16"
    description = "VGG16 ImageNet pretrained, dual classification heads (14.7M encoder params)"

    def __init__(self):
        self._model = None
        self._transform = T.Compose([
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=VGG_NORMALIZE_MEAN, std=VGG_NORMALIZE_STD),
        ])
        self._device = ("cuda" if torch.cuda.is_available() else
                        "mps"  if torch.backends.mps.is_available() else "cpu")
        proj = ProjectConfig()
        self._ckpt_path, self._ckpt_kind = _find_weights(proj)

    def is_available(self) -> bool:
        return self._ckpt_path is not None and self._ckpt_path.exists()

    def _load_model(self):
        if self._model is not None:
            return
        from vgg16_detector.lightning_module import VGG16DetectorLightning

        if self._ckpt_kind == "pt":
            checkpoint = torch.load(str(self._ckpt_path), map_location=self._device,
                                    weights_only=False)
            lightning = VGG16DetectorLightning(pretrained=False)  # weights come from ckpt
            lightning.load_state_dict(checkpoint["state_dict"])
            self._model = lightning.eval().to(self._device)
        else:
            self._model = VGG16DetectorLightning.load_from_checkpoint(
                str(self._ckpt_path),
                map_location=self._device,
                weights_only=False,
            )
            self._model.eval().to(self._device)

    def classify(self, image: Image.Image, threshold: float = 0.5) -> PipelineResult:
        if not self.is_available():
            return PipelineResult(
                pipeline_name=self.name, label="Unavailable",
                probability=0, face_score=-1, full_score=0,
                fused_score=0, num_faces=0,
                error="No checkpoint found. Train VGG16 pipeline first.",
            )
        try:
            self._load_model()
            t0 = time.time()
            face_crops, face_boxes = detect_faces(image, crop_size=IMAGE_SIZE)

            full_tensor = self._transform(image.resize((IMAGE_SIZE, IMAGE_SIZE))).unsqueeze(0).to(self._device)

            with torch.no_grad():
                full_logit = self._model.model.full_head(
                    self._model.model._encode(full_tensor)
                )
                full_score = torch.sigmoid(full_logit)[0, 0].item()

                face_score = -1.0
                if face_crops:
                    scores = []
                    for crop in face_crops:
                        t     = self._transform(crop).unsqueeze(0).to(self._device)
                        feat  = self._model.model._encode(t)
                        logit = self._model.model.face_head(feat)
                        scores.append(torch.sigmoid(logit)[0, 0].item())
                    face_score = max(scores)

            fusion = self._model.model.fusion
            fused  = (fusion.face_weight * face_score + fusion.full_weight * full_score
                      if face_score >= 0 else full_score)

            label = "AI Generated" if fused > threshold else "Real"
            return PipelineResult(
                pipeline_name=self.name,
                label=label,
                probability=fused,
                face_score=face_score,
                full_score=full_score,
                fused_score=fused,
                num_faces=len(face_crops),
                face_crops=face_crops,
                face_boxes=face_boxes,
                processing_time=time.time() - t0,
            )
        except Exception as e:
            return PipelineResult(
                pipeline_name=self.name, label="Error",
                probability=0, face_score=-1, full_score=0,
                fused_score=0, num_faces=0, error=str(e),
            )
