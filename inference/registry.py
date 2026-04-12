"""
Pipeline registry — add new pipelines here.

To register a new pipeline:
    1. Create inference/my_pipeline.py with class MyPipeline(BasePipeline)
    2. Import and add it to ALL_PIPELINES below
"""
from inference.ferretnet_pipeline import FerretNetPipeline
from inference.clip_pipeline import CLIPPipeline
from inference.vgg16_pipeline import VGG16Pipeline

# Ordered list — shown in this order in the UI
ALL_PIPELINES = [
    FerretNetPipeline(),
    CLIPPipeline(),
    VGG16Pipeline(),
]


def get_available_pipelines():
    """Return pipelines that have a trained checkpoint available."""
    return [p for p in ALL_PIPELINES if p.is_available()]


def get_all_pipelines():
    """Return all pipelines (available and unavailable)."""
    return ALL_PIPELINES
