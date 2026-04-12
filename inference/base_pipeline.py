from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
from PIL import Image


@dataclass
class PipelineResult:
    pipeline_name: str
    label: str              # overall label (based on full_score)
    full_score: float       # full-image branch P(AI) in [0, 1]
    face_scores: list = field(default_factory=list)   # per-face P(AI), one per crop
    face_labels: list = field(default_factory=list)    # per-face "AI Generated"/"Real"
    num_faces: int = 0
    face_crops: list = field(default_factory=list)     # list of PIL Images
    face_boxes: list = field(default_factory=list)     # list of (x1,y1,x2,y2)
    processing_time: float = 0.0
    error: Optional[str] = None


class BasePipeline(ABC):
    """
    Abstract base for all AI-vs-Real detection pipelines.

    To add a new pipeline:
        1. Subclass BasePipeline
        2. Set `name` and `description`
        3. Implement `is_available()` and `classify()`
        4. Register it in inference/registry.py
    """
    name: str = "unnamed"
    description: str = ""

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if model weights/checkpoint exist and model can be loaded."""
        ...

    @abstractmethod
    def classify(self, image: Image.Image, threshold: float = 0.5) -> PipelineResult:
        """
        Detect faces, classify each face independently + full image independently.
        No fusion — each face and the full image get their own AI/Real prediction.

        Args:
            image:     PIL RGB image (any size)
            threshold: decision threshold — P(AI) > threshold → "AI Generated"
        """
        ...
