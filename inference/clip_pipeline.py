import time
import glob
from pathlib import Path
from typing import Optional
from PIL import Image
import torch
import torchvision.transforms as T

from inference.base_pipeline import BasePipeline, PipelineResult
from inference.face_detector import detect_faces
from configs.base_config import (
    ProjectConfig, FusionConfig,
    FERRETNET_NORMALIZE_MEAN, FERRETNET_NORMALIZE_STD,
)

IMAGE_SIZE = 224   # CLIP ViT-B/32 expects 224×224


def _get_best_ckpt(ckpt_dir: Path) -> Optional[Path]:
    if not ckpt_dir.exists():
        return None
    candidates = [p for p in glob.glob(str(ckpt_dir / "clip-*.ckpt")) if 'last' not in p]
    if candidates:
        return Path(sorted(candidates, key=lambda p: p.split('=')[-1], reverse=True)[0])
    last = ckpt_dir / "last.ckpt"
    return last if last.exists() else None


class CLIPPipeline(BasePipeline):
    name = "CLIP ViT-B/32"
    description = "Shared CLIP ViT-B/32 encoder, dual classification heads (87M params)"

    def __init__(self):
        self._model = None
        self._transform = T.Compose([
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=FERRETNET_NORMALIZE_MEAN, std=FERRETNET_NORMALIZE_STD),
        ])
        self._device = ("cuda" if torch.cuda.is_available() else
                        "mps"  if torch.backends.mps.is_available() else "cpu")
        self._ckpt_path = _get_best_ckpt(
            ProjectConfig().project_root / "lightning_logs_clip" / "checkpoints"
        )

    def is_available(self) -> bool:
        return self._ckpt_path is not None and self._ckpt_path.exists()

    def _load_model(self):
        if self._model is not None:
            return
        from clip_detector.lightning_module import CLIPDetectorLightning
        self._model = CLIPDetectorLightning.load_from_checkpoint(
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
                error="No checkpoint found. Train CLIP pipeline first.",
            )
        try:
            self._load_model()
            t0 = time.time()
            face_crops, face_boxes = detect_faces(image, crop_size=IMAGE_SIZE)

            full_tensor = self._transform(image.resize((IMAGE_SIZE, IMAGE_SIZE))).unsqueeze(0).to(self._device)

            with torch.no_grad():
                # Use shared encoder — encode face and full in one pass if possible
                full_logit = self._model.model.full_head(
                    self._model.model._encode(full_tensor)
                )
                full_score = torch.sigmoid(full_logit)[0, 0].item()

                face_score = -1.0
                if face_crops:
                    scores = []
                    for crop in face_crops:
                        t = self._transform(crop).unsqueeze(0).to(self._device)
                        feat  = self._model.model._encode(t)
                        logit = self._model.model.face_head(feat)
                        scores.append(torch.sigmoid(logit)[0, 0].item())
                    face_score = max(scores)

            fusion = self._model.model.fusion
            if face_score >= 0:
                fused = fusion.face_weight * face_score + fusion.full_weight * full_score
            else:
                fused = full_score

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
