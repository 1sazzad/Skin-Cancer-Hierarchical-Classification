"""Model construction utilities for project experiments."""

from src.models.classification_backbone import (
    SUPPORTED_CLASSIFICATION_ARCHITECTURES,
    build_classification_model,
)
from src.models.densenet_baseline import build_densenet121
from src.models.efficientnet_baseline import build_efficientnet_b0
from src.models.shared_three_task import (
    SUPPORTED_SHARED_ARCHITECTURES,
    build_shared_three_task_efficientnet_b0,
    build_shared_three_task_model,
)


__all__ = [
    "SUPPORTED_CLASSIFICATION_ARCHITECTURES",
    "SUPPORTED_SHARED_ARCHITECTURES",
    "build_classification_model",
    "build_densenet121",
    "build_efficientnet_b0",
    "build_shared_three_task_efficientnet_b0",
    "build_shared_three_task_model",
]
