"""Tiny-ViT: a Vision Transformer image classifier written from scratch in PyTorch."""

from tiny_vit.config import DataConfig, ModelConfig, TrainConfig
from tiny_vit.model import VisionTransformer

__all__ = ["DataConfig", "ModelConfig", "TrainConfig", "VisionTransformer"]
__version__ = "0.2.0"
