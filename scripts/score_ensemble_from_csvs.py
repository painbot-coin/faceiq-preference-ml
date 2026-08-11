#!/usr/bin/env python
"""Z-score average existing model_scores.csv files into an ensemble artifact.

No forward passes. Writes artifacts/<name>/{model_scores.csv, eval.json} so
labs_composite_eval.py --leakage-split can score the blend.

Usage:
    python scripts/score_ensemble_from_csvs.py \
        --name ensemble-v25-v1 \
        --runs train-v25-panel-ft-v23b train-v1
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_scores(run: str) -> tuple[dict[str, float], dict[str, tuple[str, str]]]:
    path = ROOT / "artifacts" / run / "model_scores.csv"
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"empty or missing scores: {path}")
    scores = {r["faceId"]: float(r["modelScore"]) for r in rows if r.get("modelScore")}
    meta = {r["faceId"]: (r["gender"], r["theta"]) for r in rows}
    return scores, meta


def zscore(scores: dict[str, float]) -> dict[str, float]:
    vals = list(scores.values())
    mu = statistics.fmean(vals)
    sd = statistics.pstdev(vals) or 1.0
    return {k: (v - mu) / sd for k, v in scores.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="artifact run dirs with model_scores.csv")
    args = ap.parse_args()

    z_models: list[dict[str, float]] = []
    meta: dict[str, tuple[str, str]] = {}
    for run in args.runs:
        scores, m = load_scores(run)
        meta.update(m)
        z_models.append(zscore(scores))
        print(f"loaded {run}: {len(scores)} faces")

    common = set(z_models[0])
    for m in z_models[1:]:
        common &= set(m)
    ensemble = {f: sum(m[f] for m in z_models) / len(z_models) for f in sorted(common)}

    out = ROOT / "artifacts" / args.name
    out.mkdir(parents=True, exist_ok=True)
    with (out / "model_scores.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["faceId", "gender", "theta", "modelScore"])
        for fid, s in ensemble.items():
            g, th = meta[fid]
            w.writerow([fid, g, th, round(s, 6)])

    eval_json = {
        "checkpoints": [f"checkpoints/{r}/best.pt" for r in args.runs],
        "method": "zscore_average_from_model_scores",
        "members": args.runs,
        "n_faces": len(ensemble),
    }
    (out / "eval.json").write_text(json.dumps(eval_json, indent=2))
    print(f"wrote {out}/model_scores.csv ({len(ensemble)} faces)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
