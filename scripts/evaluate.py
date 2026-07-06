#!/usr/bin/env python
"""Evaluate a trained comparator checkpoint: held-out pairwise accuracy + rank
agreement vs the Bradley-Terry ranking.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/train-v1/best.pt \
        [--export data/exports/<runId>] \
        [--ratings artifacts/bt-refit-v1/ratings.csv] \
        [--out artifacts/train-v1/eval.json]

--export defaults to the export_dir stored in the checkpoint config.
--out defaults to artifacts/<run_name>/eval.json (run_name from the checkpoint config),
so the dashboard's training tab picks results up automatically. With --ratings, a
per-face model_scores.csv (faceId, gender, theta, modelScore) is written alongside
for the model-vs-BT charts.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from faceiq_pref.data import load_export
from faceiq_pref.eval import heldout_pairwise_accuracy, rank_agreement_vs_bt, score_all_faces


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="path to best.pt")
    ap.add_argument("--export", help="export dir; defaults to checkpoint config export_dir")
    ap.add_argument("--ratings", help="BT ratings.csv for rank agreement (optional)")
    ap.add_argument("--out", help="eval JSON path; defaults to artifacts/<run_name>/eval.json")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    export_dir = args.export or ckpt["config"]["export_dir"]
    run_name = ckpt["config"].get("run_name", "eval")
    out = Path(args.out) if args.out else Path("artifacts") / run_name / "eval.json"
    export = load_export(export_dir)
    print(f"export OK: run {export.run_id}")

    results: dict = heldout_pairwise_accuracy(args.checkpoint, export)
    print(
        f"held-out pairwise accuracy: {results['val_accuracy']:.4f} "
        f"({results['val_pairs_scored']} pairs, "
        f"{results['skipped_ties_or_missing_images']} skipped)"
    )

    if args.ratings:
        scores = score_all_faces(args.checkpoint, export)
        agreement = rank_agreement_vs_bt(scores, args.ratings)
        results["rank_agreement_vs_bt"] = agreement
        print(
            f"rank agreement vs BT: kendall_tau={agreement['kendall_tau']:.4f} "
            f"spearman_rho={agreement['spearman_rho']:.4f} "
            f"({agreement['n_faces']} faces)"
        )

        # per-face scores joined with BT theta, for the dashboard scatter
        bt_rows: dict[str, dict] = {}
        with open(args.ratings) as f:
            for row in csv.DictReader(f):
                bt_rows[row["faceId"]] = row
        scores_path = out.parent / "model_scores.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with scores_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["faceId", "gender", "theta", "modelScore"])
            for fid, s in scores.items():
                bt = bt_rows.get(fid)
                if bt:
                    writer.writerow([fid, bt["gender"], bt["theta"], round(s, 6)])
        print(f"wrote {scores_path}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
