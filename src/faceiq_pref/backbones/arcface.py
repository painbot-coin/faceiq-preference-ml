"""ArcFace R50 (InsightFace buffalo_l w600k) as a PyTorch backbone.

Weights come from ~/.insightface/models/buffalo_l/w600k_r50.onnx (auto-downloaded
on first use). Converted once to PyTorch via onnx2torch and cached under
artifacts/.cache/arcface_r50_torch.pt.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

ARCface_BACKBONES = {
    "arcface_r50": 512,
}

_CACHE = Path("artifacts/.cache/arcface_r50_torch.pt")
_ONNX = Path.home() / ".insightface/models/buffalo_l/w600k_r50.onnx"


def is_arcface_backbone(name: str) -> bool:
    return name in ARCface_BACKBONES


def _ensure_arcface_onnx() -> Path:
    if _ONNX.exists():
        return _ONNX
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1)
    if not _ONNX.exists():
        raise FileNotFoundError(f"ArcFace ONNX not found after download: {_ONNX}")
    return _ONNX


def load_arcface_r50() -> nn.Module:
    """Return ArcFace R50 embedding net (B, 3, 112, 112) -> (B, 512). Input: BGR, (x-0.5)/0.5."""
    if _CACHE.exists():
        return torch.load(_CACHE, map_location="cpu", weights_only=False)

    from onnx2torch import convert

    net = convert(str(_ensure_arcface_onnx()))
    net.eval()
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net, _CACHE)
    return net


class ArcFaceBackbone(nn.Module):
    """Wraps cached ArcFace R50; converts RGB dataloader tensors to BGR."""

    def __init__(self):
        super().__init__()
        self.net = load_arcface_r50()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, [2, 1, 0], :, :]
        return self.net(x)
