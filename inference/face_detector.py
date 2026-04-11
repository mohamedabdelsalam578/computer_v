"""Shared face detector — loaded once, reused by all pipelines."""
import time
from pathlib import Path
from typing import Optional
from PIL import Image

_detector = None   # singleton


def get_detector(weights_path: str = None, device: str = None):
    """Return cached RetinaFaceDetector, or None if weights not found."""
    global _detector
    if _detector is not None:
        return _detector

    import torch
    from configs.base_config import ProjectConfig

    if weights_path is None:
        weights_path = str(ProjectConfig().retinaface_weights)

    if not Path(weights_path).exists():
        return None

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        from retinaface.detector import RetinaFaceDetector
        _detector = RetinaFaceDetector(weights_path=weights_path, device=device)
        return _detector
    except Exception:
        return None


def detect_faces(image: Image.Image, crop_size: int = 224):
    """
    Returns (face_crops, face_boxes, face_detection_scores).
    If detector unavailable, returns three empty lists — pipelines fall back to full-image only.
    """
    detector = get_detector()
    if detector is None:
        return [], [], []

    try:
        detections = detector.detect(image)
        boxes = [d["bbox"] for d in detections]
        scores = [float(d["confidence"]) for d in detections]
        crops = detector.detect_faces(image, crop_size=crop_size)
        return crops, boxes, scores
    except Exception:
        return [], [], []
