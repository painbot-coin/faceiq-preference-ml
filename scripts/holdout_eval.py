#!/usr/bin/env python
"""Final-check holdout evaluation.

All training runs share the seed-42 face split, so the ~600 val faces were never
trained on — but repeated model selection against the same 2,068 val pairs slowly
overfits our *choices* to it. This script carves the val faces in half with a
fixed independent seed (777): pairs where BOTH faces fall in the holdout half are
the holdout set (~500 pairs). Use it only for final checks of a chosen champion,
never for picking between runs.

Usage:
    python scripts/holdout_eval.py --checkpoint checkpoints/train-v8-arcface-e2e-long/best.pt
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch.utils.data import DataLoader

from faceiq_pref.data import Matchup, load_export, split_by_face_id
from faceiq_pref.model import PairwiseModel, pick_device
from faceiq_pref.train import PairDataset, TrainConfig, _filter_rows, evaluate

HOLDOUT_SEED = 777
HOLDOUT_FRACTION = 0.5


def carve_holdout(val_rows: list[Matchup]) -> tuple[list[Matchup], list[Matchup]]:
    """Split val pairs into (selection, holdout) by face id. Deterministic."""
    val_faces = sorted({fid for m in val_rows for fid in (m.face_a_id, m.face_b_id)})
    rng = random.Random(HOLDOUT_SEED)
    rng.shuffle(val_faces)
    holdout_faces = set(val_faces[: int(len(val_faces) * HOLDOUT_FRACTION)])

    selection, holdout = [], []
    for m in val_rows:
        a, b = m.face_a_id in holdout_faces, m.face_b_id in holdout_faces
        if a and b:
            holdout.append(m)
        elif not a and not b:
            selection.append(m)
        # mixed pairs dropped from both, same rule as the train/val split
    return selection, holdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="path to best.pt")
    ap.add_argument("--export", help="export dir; defaults to checkpoint config export_dir")
    ap.add_argument("--out", help="defaults to artifacts/<run_name>/holdout_eval.json")
    args = ap.parse_args()

    device = pick_device()
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = TrainConfig.from_saved(ckpt["config"])
    export = load_export(args.export or cfg.export_dir)
    print(f"export OK: run {export.run_id} — holdout eval for {cfg.run_name!r}")

    model = PairwiseModel(cfg.backbone, pretrained=False, variance_head=cfg.variance_head).to(
        device
    )
    model.load_state_dict(ckpt["model"])
    model.eval()

    faces = export.faces()
    matchups = export.all_matchups()
    if cfg.max_pairs:
        matchups = matchups[: cfg.max_pairs]
    _, val_rows = split_by_face_id(matchups, cfg.val_fraction, cfg.split_seed)
    selection_rows, holdout_rows = carve_holdout(val_rows)

    results: dict = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": ckpt.get("epoch"),
        "holdout_seed": HOLDOUT_SEED,
        "holdout_fraction": HOLDOUT_FRACTION,
    }
    for name, rows in [("selection_val", selection_rows), ("holdout", holdout_rows)]:
        n_raw = len(rows)
        rows = _filter_rows(rows, faces, export, cfg)
        ds = PairDataset(rows, faces, export, cfg.backbone, cfg.image_size, augment=False)
        dl = DataLoader(ds, cfg.batch_size, num_workers=cfg.num_workers)
        m = evaluate(model, dl, device)
        results[name] = {
            "pairs_in_split": n_raw,
            "pairs_scored": len(rows),
            "loss": m["loss"],
            "accuracy": m["accuracy"],
        }
        print(f"{name}: accuracy={m['accuracy']:.4f} ({len(rows)} pairs)")

    out = Path(args.out) if args.out else Path("artifacts") / cfg.run_name / "holdout_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
