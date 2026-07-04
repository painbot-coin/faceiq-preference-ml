"""Siamese preference comparator: shared CNN backbone -> scalar score per photo.

P(A wins) = sigmoid(s(A) - s(B)); trained with binary cross-entropy on the matchup
winner. Higher score = more attractive per the GT preference structure.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


BACKBONES = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT, 512),
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT, 2048),
}


class PreferenceScorer(nn.Module):
    """Scores a single face photo. Used siamese-style on both sides of a pair."""

    def __init__(self, backbone: str = "resnet18", pretrained: bool = True):
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f"unknown backbone {backbone!r}; options: {list(BACKBONES)}")
        ctor, weights, feat_dim = BACKBONES[backbone]
        net = ctor(weights=weights if pretrained else None)
        net.fc = nn.Identity()
        self.backbone = net
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) -> (B,) scalar scores."""
        return self.head(self.backbone(x)).squeeze(-1)


class PairwiseModel(nn.Module):
    """Wraps the scorer for pairwise training: logit = s(A) - s(B)."""

    def __init__(self, backbone: str = "resnet18", pretrained: bool = True):
        super().__init__()
        self.scorer = PreferenceScorer(backbone, pretrained)

    def forward(self, photo_a: torch.Tensor, photo_b: torch.Tensor) -> torch.Tensor:
        """Returns logits for P(A wins). Shape (B,)."""
        return self.scorer(photo_a) - self.scorer(photo_b)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
