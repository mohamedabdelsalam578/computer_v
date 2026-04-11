import os
from dataclasses import dataclass, field
from pathlib import Path

# Override with PROJECT_ROOT on RunPod, Colab, etc. Otherwise infer repo root from this file
# (…/configs/base_config.py → parent.parent), so Streamlit is not tied to a single machine path.
def _default_project_root() -> Path:
    env = os.environ.get("PROJECT_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parent.parent


_ROOT = _default_project_root()


@dataclass
class ProjectConfig:
    project_root: Path = field(default_factory=lambda: _ROOT)
    data_raw:     Path = field(default_factory=lambda: _ROOT / "data_raw")
    weights_dir:  Path = field(default_factory=lambda: _ROOT / "weights")
    results_dir:  Path = field(default_factory=lambda: _ROOT / "results")
    lightning_logs: Path = field(default_factory=lambda: _ROOT / "lightning_logs")

    @property
    def ferretnet_weights(self) -> Path:
        return self.weights_dir / "ferretnet-b-median-3.pth"

    @property
    def retinaface_weights(self) -> Path:
        return self.weights_dir / "Resnet50_Final.pth"


@dataclass
class FerretNetConfig:
    in_channels: int = 3
    num_classes: int = 1
    dim: int = 96
    depths: tuple = (2, 2)
    lpd_func: str = "median"
    window_size: int = 3


@dataclass
class DataConfig:
    dataset_name: str = "muhammadbilal6305/200k-real-vs-ai-visuals-by-mbilal"
    face_crop_size: int = 256
    full_image_size: int = 256
    face_margin: float = 0.20
    jpeg_quality_range: tuple = (65, 100)
    jpeg_probability: float = 0.5
    seed: int = 42


@dataclass
class TrainConfig:
    batch_size: int = 64
    max_epochs: int = 30
    learning_rate: float = 1e-4   # back to standard fine-tuning lr
    betas: tuple = (0.9, 0.999)
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    gradient_clip_val: float = 1.0
    precision: str = "16-mixed"
    num_workers: int = 16
    early_stopping_patience: int = 7
    face_loss_weight: float = 0.5
    full_loss_weight: float = 0.5


@dataclass
class FusionConfig:
    face_weight: float = 0.6
    full_weight: float = 0.4
    threshold: float = 0.5


@dataclass
class RetinaFaceConfig:
    image_size: int = 840
    confidence_threshold: float = 0.02
    nms_threshold: float = 0.4
    top_k: int = 5000
    keep_top_k: int = 750
    vis_threshold: float = 0.6
    min_face_size: int = 30


FERRETNET_NORMALIZE_MEAN = [0.48145466, 0.4578275, 0.40821073]
FERRETNET_NORMALIZE_STD = [0.26862954, 0.26130258, 0.27577711]
