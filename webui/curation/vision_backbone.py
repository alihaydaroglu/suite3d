from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import torch
import torch.nn as nn


class VisionBackbone(ABC):
    """Abstract base for vision feature extractors."""

    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: nn.Module | None = None
        self.embedding_dim: int = 0

    @abstractmethod
    def _build_model(self) -> nn.Module: ...

    def load(self) -> VisionBackbone:
        self.model = self._build_model().to(self.device).eval()
        return self

    def _preprocess(self, images_uint8: np.ndarray) -> torch.Tensor:
        """ImageNet normalisation, NHWC uint8 → NCHW float32 tensor."""
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = images_uint8.astype(np.float32) / 255.0
        x = (x - mean) / std
        x = np.transpose(x, (0, 3, 1, 2))  # NHWC → NCHW
        return torch.from_numpy(x)

    @torch.no_grad()
    def extract(self, images: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """
        Extract embeddings from RGB images.

        Args:
            images: (N, 224, 224, 3) uint8
            batch_size: inference batch size

        Returns:
            (N, embedding_dim) float32
        """
        all_embeddings = []
        for i in range(0, len(images), batch_size):
            batch = self._preprocess(images[i : i + batch_size]).to(self.device)
            features = self.model(batch)
            if features.ndim > 2:
                features = features.mean(dim=(-2, -1))  # global avg pool fallback
            all_embeddings.append(features.cpu().numpy())
        return np.concatenate(all_embeddings, axis=0).astype(np.float32)


class DINOv2Backbone(VisionBackbone):
    """DINOv2-small (ViT-S/14). CLS token, 384-D."""

    def __init__(self, device: str | None = None):
        super().__init__(device)
        self.embedding_dim = 384

    def _build_model(self) -> nn.Module:
        return torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")


class ResNetBackbone(VisionBackbone):
    """ResNet-18 or ResNet-50 with classification head removed."""

    def __init__(
        self,
        variant: Literal["resnet18", "resnet50"] = "resnet18",
        device: str | None = None,
    ):
        super().__init__(device)
        self.variant = variant
        self.embedding_dim = 512 if variant == "resnet18" else 2048

    def _build_model(self) -> nn.Module:
        import torchvision.models as models

        weights_map = {
            "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
            "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
        }
        factory, weights = weights_map[self.variant]
        model = factory(weights=weights)
        model.fc = nn.Identity()
        return model


def get_backbone(name: str = "dinov2", **kwargs) -> VisionBackbone:
    """Factory. name ∈ {"dinov2", "resnet18", "resnet50"}."""
    backends = {
        "dinov2": DINOv2Backbone,
        "resnet18": lambda **kw: ResNetBackbone("resnet18", **kw),
        "resnet50": lambda **kw: ResNetBackbone("resnet50", **kw),
    }
    if name not in backends:
        raise ValueError(f"Unknown backbone {name!r}. Choose from {list(backends)}")
    return backends[name](**kwargs).load()
