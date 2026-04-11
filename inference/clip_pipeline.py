import glob
import os
import time
from pathlib import Path
from typing import Optional, Tuple
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


def _find_weights(proj: ProjectConfig) -> Tuple[Optional[Path], str]:
    """
    Returns (path, kind) where kind is 'pt' (exported) or 'ckpt' (Lightning).

    Resolution order:
      1. AI_CV_CLIP_CHECKPOINT — explicit .ckpt or .pt path (highest priority)
      2. weights/clip_finetuned.pt — exported bundle
      3. If AI_CV_CLIP_USE_LAST=1: lightning_logs_clip/checkpoints/last.ckpt (in-flight training)
      4. Best lightning_logs_clip/checkpoints/clip-*.ckpt by val_fused_acc in filename
      5. last.ckpt if nothing else matched
    """
    override = os.environ.get("AI_CV_CLIP_CHECKPOINT", "").strip()
    if override:
        p = Path(override).expanduser()
        if p.is_file():
            kind = "pt" if p.suffix.lower() == ".pt" else "ckpt"
            return p, kind

    pt_path = proj.weights_dir / "clip_finetuned.pt"
    if pt_path.exists():
        return pt_path, "pt"

    ckpt_dir = proj.project_root / "lightning_logs_clip" / "checkpoints"
    if not ckpt_dir.exists():
        return None, "none"

    last = ckpt_dir / "last.ckpt"
    use_last = os.environ.get("AI_CV_CLIP_USE_LAST", "").strip().lower() in (
        "1", "true", "yes", "on",
    )
    if use_last and last.is_file():
        return last, "ckpt"

    candidates = [
        p for p in glob.glob(str(ckpt_dir / "clip-*.ckpt"))
        if "last" not in Path(p).name.lower()
    ]
    if candidates:
        best = Path(sorted(candidates, key=lambda p: p.split("=")[-1], reverse=True)[0])
        return best, "ckpt"

    if last.is_file():
        return last, "ckpt"

    return None, "none"


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

    def _resolved_weights(self) -> Tuple[Optional[Path], str]:
        """Resolve on every call (env + disk) so Streamlit cache never pins stale paths."""
        return _find_weights(ProjectConfig())

    def is_available(self) -> bool:
        path, _ = self._resolved_weights()
        return path is not None and path.exists()

    def _load_model(self):
        if self._model is not None:
            return
        ckpt_path, ckpt_kind = self._resolved_weights()
        if ckpt_path is None or not ckpt_path.exists():
            return
        from clip_detector.lightning_module import CLIPDetectorLightning

        if ckpt_kind == "pt":
            # Exported weights — load state_dict directly (no optimizer state)
            checkpoint = torch.load(str(ckpt_path), map_location=self._device,
                                    weights_only=False)
            hparams = checkpoint.get("hyper_parameters", {})
            model_name = hparams.get("model_name", "openai/clip-vit-base-patch32")
            lightning = CLIPDetectorLightning(model_name=model_name)
            lightning.load_state_dict(checkpoint["state_dict"])
            self._model = lightning.eval().to(self._device)
        else:
            # Full Lightning checkpoint (RunPod)
            self._model = CLIPDetectorLightning.load_from_checkpoint(
                str(ckpt_path),
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
            face_crops, face_boxes, face_det_scores = detect_faces(image, crop_size=IMAGE_SIZE)

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
                face_detection_scores=face_det_scores,
                processing_time=time.time() - t0,
            )
        except Exception as e:
            return PipelineResult(
                pipeline_name=self.name, label="Error",
                probability=0, face_score=-1, full_score=0,
                fused_score=0, num_faces=0, error=str(e),
            )
