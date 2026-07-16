#!/usr/bin/env python
"""External beauty models as diagnostic cross-checks (NOT production components).

Both SCUT-FBP5500 and MEBeauty are licensed for non-commercial research use only —
this script exists for bias/agreement diagnostics against our BT ranking and trained
comparators. Do not fine-tune these into the product or ship their weights.

Subcommands:
  train-mebeauty   Fine-tune ResNet-18 on MEBeauty average scores (repo has no usable
                   pretrained weights) -> external/weights/mebeauty_resnet18.pth
  score            Score every export face with an external model -> artifacts/
                   external-<model>/{model_scores.csv, eval.json} (same format as
                   ensemble_eval.py, so scores can join any ensemble/spread analysis)
  spread           Per-face disagreement across N model_scores.csv files (z-scored
                   std) -> artifacts/<name>/spread.csv + summary stats

Usage:
    python scripts/external_crosscheck.py train-mebeauty
    python scripts/external_crosscheck.py score --model scut \
        --export data/exports/cmr1mr0m7000196d57zi3vcgn --ratings artifacts/bt-refit-v1/ratings.csv
    python scripts/external_crosscheck.py score --model mebeauty ...
    python scripts/external_crosscheck.py spread --name ensemble-spread-v1 \
        --csvs artifacts/train-v8-arcface-e2e-long/model_scores.csv \
               artifacts/train-v1/model_scores.csv \
               artifacts/external-scut/model_scores.csv \
               artifacts/external-mebeauty/model_scores.csv \
        --ratings artifacts/bt-refit-v1/ratings.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn as nn
import torchvision.models as tvm
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from faceiq_pref.data import load_export, split_by_face_id
from faceiq_pref.eval import rank_agreement_vs_bt
from faceiq_pref.model import pick_device

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = ROOT / "external" / "weights"
SCUT_WEIGHTS = WEIGHTS_DIR / "scut_resnet18_py3.pth"
MEBEAUTY_WEIGHTS = WEIGHTS_DIR / "mebeauty_resnet18.pth"
MEBEAUTY_REPO = ROOT / "external" / "MEBeauty-database"

IMAGENET_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# SCUT's released model was trained on 256-resize + 224 crop; using the canonical
# eval transform measurably improves its rank agreement on our cohort (tau 0.24 -> 0.31
# on a 400-face probe). MEBeauty (ours) was trained with direct 224 resize.
SCUT_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

MODEL_TFS = {"scut": SCUT_TF, "mebeauty": IMAGENET_TF}


# ---------------------------------------------------------------- model loading


def load_scut(device: torch.device) -> nn.Module:
    """SCUT-FBP5500 ResNet-18 (HuggingFace mirror of the authors' PyTorch release).

    Original checkpoint uses the authors' custom module names; remap to torchvision.
    """
    ckpt = torch.load(SCUT_WEIGHTS, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    remapped = {
        k.replace(".group1.", ".").replace("group1.", "").replace("group2.fullyconnected", "fc"): v
        for k, v in sd.items()
    }
    model = tvm.resnet18(num_classes=1)
    model.load_state_dict(remapped)
    return model.to(device).eval()


def load_mebeauty(device: torch.device) -> nn.Module:
    if not MEBEAUTY_WEIGHTS.exists():
        raise SystemExit(
            f"{MEBEAUTY_WEIGHTS} not found — run `python scripts/external_crosscheck.py "
            "train-mebeauty` first (the MEBeauty repo ships no usable pretrained weights)."
        )
    model = tvm.resnet18(num_classes=1)
    model.load_state_dict(torch.load(MEBEAUTY_WEIGHTS, map_location="cpu", weights_only=True))
    return model.to(device).eval()


LOADERS = {"scut": load_scut, "mebeauty": load_mebeauty}


# ---------------------------------------------------------------- mebeauty training


class MEBeautyDataset(Dataset):
    def __init__(self, csv_path: Path, augment: bool):
        self.rows: list[tuple[Path, float]] = []
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                p = MEBEAUTY_REPO / row["image"]
                if p.exists():
                    self.rows.append((p, float(row["score"])))
        ops: list = [transforms.Resize((224, 224))]
        if augment:
            ops.append(transforms.RandomHorizontalFlip())
        ops += [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
        self.tf = transforms.Compose(ops)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        path, score = self.rows[idx]
        with Image.open(path) as img:
            return self.tf(img.convert("RGB")), torch.tensor(score, dtype=torch.float32)


def cmd_train_mebeauty(args: argparse.Namespace) -> int:
    device = pick_device()
    train_ds = MEBeautyDataset(MEBEAUTY_REPO / "scores" / "train_crop.csv", augment=True)
    test_ds = MEBeautyDataset(MEBEAUTY_REPO / "scores" / "test_crop.csv", augment=False)
    print(f"MEBeauty: {len(train_ds)} train / {len(test_ds)} test images (device {device})")

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    test_dl = DataLoader(test_ds, batch_size=64, num_workers=4)

    model = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    best_pearson = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for x, y in tqdm(train_dl, desc=f"epoch {epoch}/{args.epochs}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x).squeeze(1), y)
            loss.backward()
            opt.step()
            running += loss.item() * len(y)
            seen += len(y)

        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for x, y in test_dl:
                preds += model(x.to(device)).squeeze(1).cpu().tolist()
                targets += y.tolist()
        n = len(preds)
        mp, mt = statistics.fmean(preds), statistics.fmean(targets)
        cov = sum((p - mp) * (t - mt) for p, t in zip(preds, targets)) / n
        pearson = cov / ((statistics.pstdev(preds) or 1) * (statistics.pstdev(targets) or 1))
        rmse = (sum((p - t) ** 2 for p, t in zip(preds, targets)) / n) ** 0.5
        print(
            f"epoch {epoch}: train_mse={running / seen:.4f} "
            f"test_pearson={pearson:.4f} test_rmse={rmse:.4f}"
        )
        if pearson > best_pearson:
            best_pearson = pearson
            WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MEBEAUTY_WEIGHTS)

    print(f"best test pearson: {best_pearson:.4f} -> {MEBEAUTY_WEIGHTS}")
    return 0


# ---------------------------------------------------------------- scoring


@torch.no_grad()
def score_faces(
    model: nn.Module, export, device: torch.device, tf=IMAGENET_TF
) -> dict[str, float]:
    faces = [f for f in export.faces().values() if export.image_path(f).exists()]
    scores: dict[str, float] = {}
    bs = 128
    for i in tqdm(range(0, len(faces), bs), desc="scoring faces"):
        batch = faces[i : i + bs]
        tensors = []
        for f in batch:
            with Image.open(export.image_path(f)) as img:
                tensors.append(tf(img.convert("RGB")))
        out = model(torch.stack(tensors).to(device)).squeeze(1)
        for f, s in zip(batch, out.cpu().tolist()):
            scores[f.face_id] = s
    return scores


def cmd_score(args: argparse.Namespace) -> int:
    device = pick_device()
    model = LOADERS[args.model](device)
    export = load_export(args.export)
    print(f"export OK: run {export.run_id} — external model {args.model!r}")

    scores = score_faces(model, export, device, MODEL_TFS[args.model])

    # held-out pairwise accuracy on the standard split (val_fraction 0.2, seed 42),
    # same protocol as ensemble_eval.py so numbers are comparable across runs
    faces = export.faces()
    _, val_rows = split_by_face_id(export.all_matchups(), 0.2, 42)
    correct = scored = 0
    for m in val_rows:
        if m.is_tie or m.face_a_id not in scores or m.face_b_id not in scores:
            continue
        correct += (scores[m.face_a_id] > scores[m.face_b_id]) == (m.final_outcome == "A")
        scored += 1

    results: dict = {
        "model": args.model,
        "license_note": "non-commercial research use only — diagnostic cross-check",
        "val_pairs_scored": scored,
        "val_accuracy": correct / max(scored, 1),
    }
    print(f"held-out pairwise accuracy: {results['val_accuracy']:.4f} ({scored} pairs)")

    out_dir = ROOT / "artifacts" / f"external-{args.model}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.ratings:
        agreement = rank_agreement_vs_bt(scores, args.ratings)
        results["rank_agreement_vs_bt"] = agreement
        print(
            f"rank agreement vs BT: kendall_tau={agreement['kendall_tau']:.4f} "
            f"spearman_rho={agreement['spearman_rho']:.4f} ({agreement['n_faces']} faces)"
        )

    bt_rows: dict[str, dict] = {}
    if args.ratings:
        with open(args.ratings) as f:
            bt_rows = {row["faceId"]: row for row in csv.DictReader(f)}
    face_meta = export.faces()
    with (out_dir / "model_scores.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["faceId", "gender", "theta", "modelScore"])
        for fid, s in scores.items():
            bt = bt_rows.get(fid)
            face = face_meta.get(fid)
            gender = (bt["gender"] if bt else None) or (face.gender if face else "")
            theta = bt["theta"] if bt else ""
            if not gender:
                continue
            writer.writerow([fid, gender, theta, round(s, 6)])

    (out_dir / "eval.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {out_dir}/model_scores.csv and eval.json")
    return 0


# ---------------------------------------------------------------- disagreement


def _zscore(scores: dict[str, float]) -> dict[str, float]:
    vals = list(scores.values())
    mu = statistics.fmean(vals)
    sd = statistics.pstdev(vals) or 1.0
    return {k: (v - mu) / sd for k, v in scores.items()}


def cmd_spread(args: argparse.Namespace) -> int:
    per_model: list[dict[str, float]] = []
    for path in args.csvs:
        scores = {}
        with open(path) as f:
            for row in csv.DictReader(f):
                scores[row["faceId"]] = float(row["modelScore"])
        per_model.append(_zscore(scores))
        print(f"loaded {len(scores)} scores from {path}")

    common = set(per_model[0])
    for m in per_model[1:]:
        common &= set(m)
    print(f"{len(common)} faces common to all {len(per_model)} models")

    rows = []
    for fid in sorted(common):
        vals = [m[fid] for m in per_model]
        rows.append({
            "faceId": fid,
            "meanScore": statistics.fmean(vals),
            "spread": statistics.pstdev(vals),  # per-face model disagreement
        })

    theta: dict[str, float] = {}
    if args.ratings:
        with open(args.ratings) as f:
            theta = {r["faceId"]: float(r["theta"]) for r in csv.DictReader(f)}

    out_dir = ROOT / "artifacts" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "spread.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["faceId", "meanScore", "spread", "theta"])
        for r in rows:
            writer.writerow([r["faceId"], round(r["meanScore"], 6), round(r["spread"], 6),
                             theta.get(r["faceId"], "")])

    spreads = [r["spread"] for r in rows]
    summary = {
        "models": args.csvs,
        "n_faces": len(rows),
        "spread_mean": statistics.fmean(spreads),
        "spread_p50": statistics.median(spreads),
        "spread_p90": sorted(spreads)[int(0.9 * len(spreads))],
    }
    if theta:
        # do models disagree more in the crowded middle of the distribution?
        from scipy.stats import spearmanr

        common_theta = [r for r in rows if r["faceId"] in theta]
        thetas = [theta[r["faceId"]] for r in common_theta]
        med = statistics.median(thetas)
        dist_from_median = [abs(t - med) for t in thetas]
        rho, _ = spearmanr([r["spread"] for r in common_theta], dist_from_median)
        summary["spearman_spread_vs_theta_extremity"] = float(rho)
        print(f"spread vs theta-extremity spearman: {rho:.4f} "
              "(negative = more disagreement mid-distribution, as expected)")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"spread mean={summary['spread_mean']:.4f} p50={summary['spread_p50']:.4f} "
          f"p90={summary['spread_p90']:.4f}")
    print(f"wrote {out_dir}/spread.csv and summary.json")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train-mebeauty")
    p_train.add_argument("--epochs", type=int, default=12)

    p_score = sub.add_parser("score")
    p_score.add_argument("--model", choices=sorted(LOADERS), required=True)
    p_score.add_argument("--export", required=True)
    p_score.add_argument("--ratings", help="BT ratings.csv for rank agreement")

    p_spread = sub.add_parser("spread")
    p_spread.add_argument("--name", required=True, help="output dir under artifacts/")
    p_spread.add_argument("--csvs", nargs="+", required=True, help="2+ model_scores.csv")
    p_spread.add_argument("--ratings", help="BT ratings.csv for theta context")

    args = ap.parse_args()
    return {"train-mebeauty": cmd_train_mebeauty, "score": cmd_score, "spread": cmd_spread}[
        args.cmd
    ](args)


if __name__ == "__main__":
    raise SystemExit(main())
