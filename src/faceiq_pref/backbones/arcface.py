"""InsightFace ArcFace recognition nets as PyTorch backbones.

Weights come from the InsightFace model packs (auto-downloaded on first use),
converted once via onnx2torch and cached under artifacts/.cache/.

Supported:
  arcface_r50  — buffalo_l / w600k_r50.onnx   (WebFace600K, 512-d)
  arcface_r100 — antelopev2 / glintr100.onnx  (Glint360K, 512-d)
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

ARCface_BACKBONES = {
    "arcface_r50": 512,
    "arcface_r100": 512,
}

# name -> (pack, onnx filename, cache path)
_SPEC: dict[str, tuple[str, str, Path]] = {
    "arcface_r50": (
        "buffalo_l",
        "w600k_r50.onnx",
        Path("artifacts/.cache/arcface_r50_torch.pt"),
    ),
    "arcface_r100": (
        "antelopev2",
        "glintr100.onnx",
        Path("artifacts/.cache/arcface_r100_torch.pt"),
    ),
}


def is_arcface_backbone(name: str) -> bool:
    return name in ARCface_BACKBONES


def _onnx_candidates(pack: str, onnx_name: str) -> list[Path]:
    root = Path.home() / ".insightface" / "models" / pack
    # Some packs unzip as models/<pack>/<pack>/*.onnx (nested).
    return [root / onnx_name, root / pack / onnx_name]


def _find_onnx(pack: str, onnx_name: str) -> Path | None:
    for path in _onnx_candidates(pack, onnx_name):
        if path.exists():
            return path
    return None


def _ensure_onnx(pack: str, onnx_name: str) -> Path:
    found = _find_onnx(pack, onnx_name)
    if found is not None:
        return found

    import zipfile

    models_root = Path.home() / ".insightface" / "models"
    models_root.mkdir(parents=True, exist_ok=True)
    zip_path = models_root / f"{pack}.zip"
    if not zip_path.exists():
        # FaceAnalysis downloads the zip; nested layout can then break its assert,
        # so we download via the same URL convention and unzip ourselves.
        import urllib.request

        url = (
            f"https://github.com/deepinsight/insightface/releases/download/v0.7/{pack}.zip"
        )
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, zip_path)

    pack_dir = models_root / pack
    pack_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(pack_dir)

    # Flatten models/<pack>/<pack>/* -> models/<pack>/* when nested.
    nested = pack_dir / pack
    if nested.is_dir():
        for child in nested.iterdir():
            dest = pack_dir / child.name
            if not dest.exists():
                child.rename(dest)
        try:
            nested.rmdir()
        except OSError:
            pass

    found = _find_onnx(pack, onnx_name)
    if found is None:
        raise FileNotFoundError(
            f"ArcFace ONNX not found after downloading pack {pack!r}: "
            f"looked in {[str(p) for p in _onnx_candidates(pack, onnx_name)]}"
        )
    return found


def load_arcface(name: str) -> nn.Module:
    """Return embedding net (B, 3, 112, 112) -> (B, 512). Input: BGR, (x-0.5)/0.5."""
    if name not in _SPEC:
        raise ValueError(f"unknown ArcFace backbone {name!r}; options: {list(_SPEC)}")
    pack, onnx_name, cache = _SPEC[name]
    if cache.exists():
        return torch.load(cache, map_location="cpu", weights_only=False)

    from onnx2torch import convert

    net = convert(str(_ensure_onnx(pack, onnx_name)))
    net.eval()
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net, cache)
    return net


def load_arcface_r50() -> nn.Module:
    """Backward-compatible alias."""
    return load_arcface("arcface_r50")


class ArcFaceBackbone(nn.Module):
    """Wraps a cached ArcFace net; converts RGB dataloader tensors to BGR."""

    def __init__(self, name: str = "arcface_r50"):
        super().__init__()
        if name not in ARCface_BACKBONES:
            raise ValueError(f"unsupported ArcFace variant {name!r}")
        self.name = name
        self.net = load_arcface(name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, [2, 1, 0], :, :]
        return self.net(x)
