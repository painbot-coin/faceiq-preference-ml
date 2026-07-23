"""Siamese preference comparator: shared CNN backbone -> scalar score per photo.

P(A wins) = sigmoid(s(A) - s(B)); trained with binary cross-entropy on the matchup
winner. Higher score = more attractive per the GT preference structure.

With `variance_head=True` (UOL-style uncertainty, research log §5), each face is
scored as a Gaussian N(mu, sigma^2) instead of a point: the pairwise logit becomes
(mu_A - mu_B) / sqrt(1 + var_A + var_B), which reduces to plain RankNet as both
variances -> 0. High-variance faces flatten their pair logits toward 0, so the
model can express "genuinely ambiguous" instead of being forced to pick a side.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from .backbones.arcface import ARCface_BACKBONES, ArcFaceBackbone, is_arcface_backbone


BACKBONES = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT, 512),
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT, 2048),
}

# DINOv2 ViTs loaded via torch.hub (no extra pip deps); embedding dim per variant.
DINOV2_BACKBONES = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
}


# log(sigma^2) range: sigma in ~[0.05, 7.4]. The floor keeps gradients finite;
# the ceiling stops the model from zeroing hard pairs' loss by inflating variance.
LOG_VAR_MIN, LOG_VAR_MAX = -6.0, 4.0


class PreferenceScorer(nn.Module):
    """Scores a single face photo. Used siamese-style on both sides of a pair."""

    def __init__(
        self, backbone: str = "resnet18", pretrained: bool = True, variance_head: bool = False
    ):
        super().__init__()
        if backbone in DINOV2_BACKBONES:
            net = torch.hub.load("facebookresearch/dinov2", backbone, pretrained=pretrained)
            feat_dim = DINOV2_BACKBONES[backbone]
        elif is_arcface_backbone(backbone):
            if backbone != "arcface_r50":
                raise ValueError(f"unsupported ArcFace variant {backbone!r}")
            net = ArcFaceBackbone()
            feat_dim = ARCface_BACKBONES[backbone]
        elif backbone in BACKBONES:
            ctor, weights, feat_dim = BACKBONES[backbone]
            net = ctor(weights=weights if pretrained else None)
            net.fc = nn.Identity()
        else:
            options = list(BACKBONES) + list(DINOV2_BACKBONES) + list(ARCface_BACKBONES)
            raise ValueError(f"unknown backbone {backbone!r}; options: {options}")
        self.backbone = net
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )
        self.variance_head = variance_head
        if variance_head:
            self.log_var_head = nn.Sequential(
                nn.Linear(feat_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(256, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) -> (B,) scalar scores (the Gaussian mean when variance_head)."""
        return self.head(self.backbone(x)).squeeze(-1)

    def forward_dist(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, 3, H, W) -> (mu, log_var), each (B,). Requires variance_head=True."""
        feats = self.backbone(x)
        mu = self.head(feats).squeeze(-1)
        log_var = self.log_var_head(feats).squeeze(-1).clamp(LOG_VAR_MIN, LOG_VAR_MAX)
        return mu, log_var


class PairwiseModel(nn.Module):
    """Wraps the scorer for pairwise training: logit = s(A) - s(B).

    With variance_head, logit = (mu_A - mu_B) / sqrt(1 + var_A + var_B); the constant
    1 is the standard-RankNet logistic noise floor, so the model only ever *adds*
    uncertainty on top of the point-score baseline.
    """

    def __init__(
        self, backbone: str = "resnet18", pretrained: bool = True, variance_head: bool = False
    ):
        super().__init__()
        self.scorer = PreferenceScorer(backbone, pretrained, variance_head)

    def forward(self, photo_a: torch.Tensor, photo_b: torch.Tensor) -> torch.Tensor:
        """Returns logits for P(A wins). Shape (B,)."""
        if self.scorer.variance_head:
            mu_a, log_var_a = self.scorer.forward_dist(photo_a)
            mu_b, log_var_b = self.scorer.forward_dist(photo_b)
            scale = torch.sqrt(1.0 + log_var_a.exp() + log_var_b.exp())
            return (mu_a - mu_b) / scale
        return self.scorer(photo_a) - self.scorer(photo_b)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
