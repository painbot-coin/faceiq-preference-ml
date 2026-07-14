#!/usr/bin/env python
"""Evaluate an ensemble of trained comparators (score averaging, no training).

Each checkpoint scores every face; per-model scores are z-normalized (different
backbones produce different score scales) and averaged. Because the comparator
is siamese, pairwise logit = s(A) - s(B), so per-face scores reproduce pairwise
predictions exactly.

Val split is rebuilt from the first checkpoint's config (all runs share
val_fraction 0.2 / seed 42, so the 2,068-pair val set matches single-run evals).

Usage:
    python scripts/ensemble_eval.py \
        --checkpoints checkpoints/train-v7-arcface-e2e/best.pt checkpoints/train-v1/best.pt \
        --name ensemble-v7-v1 \
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

from faceiq_pref.data import load_export, split_by_face_id
from faceiq_pref.eval import rank_agreement_vs_bt, score_all_faces
from faceiq_pref.train import TrainConfig, _filter_rows


def zscore(scores: dict[str, float]) -> dict[str, float]:
    vals = list(scores.values())
    mu = statistics.fmean(vals)
    sd = statistics.pstdev(vals) or 1.0
    return {k: (v - mu) / sd for k, v in scores.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True, help="2+ best.pt paths")
    ap.add_argument("--name", required=True, help="run name, e.g. ensemble-v7-v1")
    ap.add_argument("--export", help="export dir; defaults to first checkpoint config")
    ap.add_argument("--ratings", help="BT ratings.csv for rank agreement (optional)")
    args = ap.parse_args()

    first_cfg = TrainConfig(**torch.load(args.checkpoints[0], map_location="cpu")["config"])
    export = load_export(args.export or first_cfg.export_dir)
    print(f"export OK: run {export.run_id} — ensemble {args.name!r} ({len(args.checkpoints)} models)")

    per_model: list[dict[str, float]] = []
    for ckpt in args.checkpoints:
        print(f"scoring with {ckpt} ...")
        per_model.append(zscore(score_all_faces(ckpt, export)))

    common = set(per_model[0])
    for m in per_model[1:]:
        common &= set(m)
    ensemble = {f: sum(m[f] for m in per_model) / len(per_model) for f in common}

    # held-out pairwise accuracy on the shared val split
    faces = export.faces()
    matchups = export.all_matchups()
    _, val_rows = split_by_face_id(matchups, first_cfg.val_fraction, first_cfg.split_seed)
    n_raw = len(val_rows)
    val_rows = _filter_rows(val_rows, faces, export, first_cfg)

    correct = scored = 0
    for m in val_rows:
        if m.face_a_id not in ensemble or m.face_b_id not in ensemble:
            continue
        pred_a_wins = ensemble[m.face_a_id] > ensemble[m.face_b_id]
        correct += pred_a_wins == (m.final_outcome == "A")
        scored += 1

    results: dict = {
        "checkpoints": args.checkpoints,
        "val_pairs_in_split": n_raw,
        "val_pairs_scored": scored,
        "val_accuracy": correct / max(scored, 1),
    }
    print(f"held-out pairwise accuracy: {results['val_accuracy']:.4f} ({scored} pairs)")

    out_dir = Path("artifacts") / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.ratings:
        agreement = rank_agreement_vs_bt(ensemble, args.ratings)
        results["rank_agreement_vs_bt"] = agreement
        print(
            f"rank agreement vs BT: kendall_tau={agreement['kendall_tau']:.4f} "
            f"spearman_rho={agreement['spearman_rho']:.4f} ({agreement['n_faces']} faces)"
        )
        bt_rows: dict[str, dict] = {}
        with open(args.ratings) as f:
            for row in csv.DictReader(f):
                bt_rows[row["faceId"]] = row
        with (out_dir / "model_scores.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["faceId", "gender", "theta", "modelScore"])
            for fid, s in ensemble.items():
                bt = bt_rows.get(fid)
                if bt:
                    writer.writerow([fid, bt["gender"], bt["theta"], round(s, 6)])
        print(f"wrote {out_dir / 'model_scores.csv'}")

    (out_dir / "eval.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {out_dir / 'eval.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
