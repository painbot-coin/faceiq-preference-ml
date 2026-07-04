"""Evaluation: held-out pairwise accuracy and rank agreement vs Bradley-Terry."""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from PIL import Image
from scipy.stats import kendalltau, spearmanr
from torchvision import transforms
from tqdm import tqdm

from .data import Export, Face
from .model import PreferenceScorer, pick_device
from .train import IMAGENET_MEAN, IMAGENET_STD


@torch.no_grad()
def score_all_faces(
    checkpoint_path: str | Path,
    export: Export,
    image_size: int = 224,
    batch_size: int = 128,
) -> dict[str, float]:
    """Run the trained scorer over every face photo. Returns faceId -> model score."""
    device = pick_device()
    ckpt = torch.load(checkpoint_path, map_location=device)
    backbone = ckpt["config"]["backbone"]
    scorer = PreferenceScorer(backbone, pretrained=False).to(device)
    # PairwiseModel stores weights under scorer.*
    state = {k.removeprefix("scorer."): v for k, v in ckpt["model"].items()}
    scorer.load_state_dict(state)
    scorer.eval()

    tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    faces = [f for f in export.faces().values() if export.image_path(f).exists()]
    scores: dict[str, float] = {}
    for i in tqdm(range(0, len(faces), batch_size), desc="scoring faces"):
        batch = faces[i : i + batch_size]
        tensors = []
        for f in batch:
            with Image.open(export.image_path(f)) as img:
                tensors.append(tf(img.convert("RGB")))
        out = scorer(torch.stack(tensors).to(device))
        for f, s in zip(batch, out.cpu().tolist()):
            scores[f.face_id] = s
    return scores


def rank_agreement_vs_bt(
    model_scores: dict[str, float], ratings_csv: str | Path
) -> dict[str, float]:
    """Kendall tau + Spearman rho between model ranking and BT theta ranking."""
    bt_theta: dict[str, float] = {}
    with open(ratings_csv) as f:
        for row in csv.DictReader(f):
            bt_theta[row["faceId"]] = float(row["theta"])

    common = sorted(set(model_scores) & set(bt_theta))
    ms = [model_scores[f] for f in common]
    bt = [bt_theta[f] for f in common]
    tau, _ = kendalltau(ms, bt)
    rho, _ = spearmanr(ms, bt)
    return {"kendall_tau": float(tau), "spearman_rho": float(rho), "n_faces": len(common)}
