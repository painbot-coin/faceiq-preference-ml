"""Training loop for the pairwise preference comparator.

Data flow: export dir -> face-id split -> PairDataset (loads photos lazily) ->
BCE-with-logits on winner -> checkpoints/<run>/best.pt + artifacts/<run>/metrics.json.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from .backbones.arcface import is_arcface_backbone
from .data import Export, Face, Matchup, split_by_face_id
from .model import PairwiseModel, pick_device

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
# ArcFace w600k_r50: (pixel/255 - 0.5) / 0.5; backbone forward swaps RGB -> BGR.
ARCface_MEAN = [0.5, 0.5, 0.5]
ARCface_STD = [0.5, 0.5, 0.5]


def build_transforms(backbone: str, image_size: int, augment: bool) -> transforms.Compose:
    ops: list = [transforms.Resize((image_size, image_size))]
    if augment:
        ops.append(transforms.RandomHorizontalFlip())
    ops.append(transforms.ToTensor())
    if is_arcface_backbone(backbone):
        ops.append(transforms.Normalize(ARCface_MEAN, ARCface_STD))
    else:
        ops.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return transforms.Compose(ops)


@dataclass
class TrainConfig:
    export_dir: str
    run_name: str = "train-v1"
    backbone: str = "resnet18"
    image_size: int = 224
    batch_size: int = 64
    epochs: int = 10
    lr: float = 1e-4
    weight_decay: float = 1e-4
    val_fraction: float = 0.2
    split_seed: int = 42
    num_workers: int = 4
    skip_ties: bool = True
    augment: bool = True
    max_pairs: int | None = None  # subsample for smoke tests
    freeze_backbone: bool = False  # embedding probe: train only the head
    confidence_filter: str | None = None  # train only: high | medium | low (human labels kept)
    variance_head: bool = False  # UOL-style per-face Gaussian; probabilistic RankNet logit

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        import yaml

        raw = yaml.safe_load(Path(path).read_text())
        return cls(**raw)


class PairDataset(Dataset):
    def __init__(
        self,
        matchups: list[Matchup],
        faces: dict[str, Face],
        export: Export,
        backbone: str,
        image_size: int,
        augment: bool,
    ):
        self.rows = matchups
        self.faces = faces
        self.export = export
        self.tf = build_transforms(backbone, image_size, augment)

    def __len__(self) -> int:
        return len(self.rows)

    def _load(self, face_id: str) -> torch.Tensor:
        path = self.export.image_path(self.faces[face_id])
        with Image.open(path) as img:
            return self.tf(img.convert("RGB"))

    def __getitem__(self, idx: int):
        m = self.rows[idx]
        label = 1.0 if m.final_outcome == "A" else 0.0 if m.final_outcome == "B" else 0.5
        return self._load(m.face_a_id), self._load(m.face_b_id), torch.tensor(label)


def _apply_confidence_filter(rows: list[Matchup], floor: str) -> list[Matchup]:
    """Keep human-audited pairs; for VLM pairs require confidence >= floor."""
    order = {"low": 0, "medium": 1, "high": 2}
    min_level = order[floor]
    return [
        m
        for m in rows
        if m.human_labeled_at is not None
        or (m.confidence is not None and order.get(m.confidence, 0) >= min_level)
    ]


def _filter_rows(rows: list[Matchup], faces: dict[str, Face], export: Export, cfg: TrainConfig):
    kept = []
    for m in rows:
        if cfg.skip_ties and m.is_tie:
            continue
        pa = export.image_path(faces[m.face_a_id])
        pb = export.image_path(faces[m.face_b_id])
        if pa.exists() and pb.exists():
            kept.append(m)
    return kept


@torch.no_grad()
def evaluate(model: PairwiseModel, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss, correct, seen = 0.0, 0, 0
    for a, b, y in loader:
        a, b, y = a.to(device), b.to(device), y.to(device)
        logits = model(a, b)
        total_loss += criterion(logits, y).item() * len(y)
        correct += ((logits > 0) == (y > 0.5)).sum().item()
        seen += len(y)
    return {"loss": total_loss / max(seen, 1), "accuracy": correct / max(seen, 1)}


def train(export: Export, cfg: TrainConfig) -> dict:
    device = pick_device()
    faces = export.faces()
    matchups = export.all_matchups()
    if cfg.max_pairs:
        matchups = matchups[: cfg.max_pairs]

    train_rows, val_rows = split_by_face_id(matchups, cfg.val_fraction, cfg.split_seed)
    if cfg.confidence_filter:
        before = len(train_rows)
        train_rows = _apply_confidence_filter(train_rows, cfg.confidence_filter)
        print(
            f"confidence filter >= {cfg.confidence_filter} (train only): "
            f"{before} -> {len(train_rows)} pairs"
        )
    train_rows = _filter_rows(train_rows, faces, export, cfg)
    val_rows = _filter_rows(val_rows, faces, export, cfg)

    train_ds = PairDataset(train_rows, faces, export, cfg.backbone, cfg.image_size, cfg.augment)
    val_ds = PairDataset(val_rows, faces, export, cfg.backbone, cfg.image_size, augment=False)
    train_dl = DataLoader(
        train_ds, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True
    )
    val_dl = DataLoader(val_ds, cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True)

    model = PairwiseModel(cfg.backbone, variance_head=cfg.variance_head).to(device)
    if cfg.freeze_backbone:
        for p in model.scorer.backbone.parameters():
            p.requires_grad = False
        model.scorer.backbone.eval()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    ckpt_dir = Path("checkpoints") / cfg.run_name
    art_dir = Path("artifacts") / cfg.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)

    history, best_acc = [], 0.0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        if cfg.freeze_backbone:
            model.scorer.backbone.eval()
        running, seen = 0.0, 0
        for a, b, y in tqdm(train_dl, desc=f"epoch {epoch}/{cfg.epochs}"):
            a, b, y = a.to(device), b.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(a, b), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(y)
            seen += len(y)

        val = evaluate(model, val_dl, device)
        row = {
            "epoch": epoch,
            "train_loss": running / max(seen, 1),
            "val_loss": val["loss"],
            "val_accuracy": val["accuracy"],
            "time": time.time(),
        }
        history.append(row)
        print(
            f"epoch {epoch}: train_loss={row['train_loss']:.4f} "
            f"val_loss={val['loss']:.4f} val_acc={val['accuracy']:.4f}"
        )
        if val["accuracy"] > best_acc:
            best_acc = val["accuracy"]
            torch.save(
                {"model": model.state_dict(), "config": asdict(cfg), "epoch": epoch},
                ckpt_dir / "best.pt",
            )

        metrics = {
            "config": asdict(cfg),
            "device": str(device),
            "train_pairs": len(train_rows),
            "val_pairs": len(val_rows),
            "best_val_accuracy": best_acc,
            "history": history,
        }
        (art_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    return metrics
