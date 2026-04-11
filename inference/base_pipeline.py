from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
from PIL import Image


@dataclass
class PipelineResult:
    pipeline_name: str
    label: str              # "AI Generated" or "Real"
    probability: float      # P(AI) in [0, 1]
    face_score: float       # max face-branch score (-1 if no faces detected)
    full_score: float       # full-image branch score
    fused_score: float      # final fused score used for decision
    num_faces: int
    face_crops: list = field(default_factory=list)   # list of PIL Images
    face_boxes: list = field(default_factory=list)   # list of (x1,y1,x2,y2)
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
        Detect faces, run both branches, fuse scores, return PipelineResult.

        Args:
            image:     PIL RGB image (any size)
            threshold: decision threshold — P(AI) > threshold → "AI Generated"
        """
        ...
