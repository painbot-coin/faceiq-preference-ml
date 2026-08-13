#!/usr/bin/env python
"""List bothHeldOut uniform run-4 pairs where the comparator misses panel majority.

Yardstick matches labs_composite_eval --leakage-split: panel majority on
labels/panel-run-4-random, val faces from the run's face-id split, ranking control =
vote-blind bt-refit-v2-qc theta (raw score comparison, same as leakage_split control).

Comparator picks use modelScore from artifacts/<run>/model_scores.csv (same ordering as
placement for pairwise sign when scores differ).

Usage:
    python scripts/bothheldout_error_audit.py --run train-v25-panel-ft-v23b
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from accuracy_matrix import load_votes  # noqa: E402
from faceiq_pref.data import load_export, split_by_face_id  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="train-v25-panel-ft-v23b")
    ap.add_argument("--control-ranking", default="artifacts/bt-refit-v2-qc")
    ap.add_argument("--panel", default="labels/panel-run-4-random")
    ap.add_argument("--rejects", default="artifacts/panel-run-v4/reject-pids.txt")
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    art = ROOT / "artifacts" / a.run
    scores_path = art / "model_scores.csv"
    if not scores_path.exists() or scores_path.stat().st_size < 100:
        raise SystemExit(
            f"missing/empty {scores_path} — pull from GPU or run evaluate.py first"
        )

    export = next(
        p for p in sorted((ROOT / "data" / "exports").glob("*"))
        if (p / "faces.jsonl").exists()
    )
    scores = pd.read_csv(scores_path)
    model = dict(zip(scores["faceId"], scores["modelScore"]))
    ctrl = pd.read_csv(ROOT / a.control_ranking / "ratings.csv")
    control = dict(zip(ctrl["faceId"], ctrl["theta"]))

    _, val_rows = split_by_face_id(
        load_export(export).all_matchups(), a.val_fraction, a.split_seed
    )
    held_out = {f for m in val_rows for f in (m.face_a_id, m.face_b_id)}

    rejects = {
        ln.strip()
        for ln in (ROOT / a.rejects).read_text().splitlines()
        if ln.strip()
    }
    votes, meta = load_votes(
        ROOT / a.panel / "results",
        ROOT / a.panel / "sample-meta.json",
        rejects,
        True,
    )

    errors = []
    both_held = right = wrong_model_right_ctrl = both_wrong = wrong_model_wrong_ctrl_tie = 0
    # also count control-only misses among model-correct for completeness
    model_right_ctrl_wrong = 0

    for pid, (va, vb) in votes.items():
        m = meta.get(pid)
        if m is None or va == vb:
            continue
        fa, fb = m["faceAId"], m["faceBId"]
        if fa not in held_out or fb not in held_out:
            continue
        if fa not in model or fb not in model or fa not in control or fb not in control:
            continue
        sa, sb = model[fa], model[fb]
        ca, cb = control[fa], control[fb]
        if sa == sb:
            continue
        both_held += 1
        maj_a = va > vb
        model_a = sa > sb
        ctrl_a = ca > cb
        model_ok = model_a == maj_a
        ctrl_ok = ctrl_a == maj_a
        if model_ok:
            right += 1
            if not ctrl_ok:
                model_right_ctrl_wrong += 1
            continue
        # model wrong
        if ctrl_ok:
            wrong_model_right_ctrl += 1
            bucket = "modelWrong_controlRight"
        else:
            both_wrong += 1
            bucket = "bothWrong"
            if ca == cb:
                wrong_model_wrong_ctrl_tie += 1
                bucket = "modelWrong_controlTie"

        gap = abs(ca - cb)
        errors.append({
            "pairId": pid,
            "pairIndex": m.get("pairIndex"),
            "faceAId": fa,
            "faceBId": fb,
            "votesA": va,
            "votesB": vb,
            "panelMajority": "A" if maj_a else "B",
            "modelPick": "A" if model_a else "B",
            "controlPick": "A" if ctrl_a else ("B" if ca != cb else "tie"),
            "modelScoreA": float(sa),
            "modelScoreB": float(sb),
            "controlThetaA": float(ca),
            "controlThetaB": float(cb),
            "btGap": float(gap),
            "bucket": bucket,
        })

    summary = {
        "run": a.run,
        "panel": a.panel,
        "pairDistribution": "uniform run-4",
        "metric": "panel majority",
        "leakageStratum": "bothHeldOut",
        "controlRanking": a.control_ranking,
        "bothHeldOutPairs": both_held,
        "modelCorrect": right,
        "modelAccuracy": right / both_held if both_held else float("nan"),
        "modelWrong": len(errors),
        "modelWrong_controlRight": wrong_model_right_ctrl,
        "bothWrong": both_wrong,
        "modelWrong_controlTie": wrong_model_wrong_ctrl_tie,
        "modelRight_controlWrong": model_right_ctrl_wrong,
        "note": (
            "Comparator picks from model_scores.csv; control from vote-blind "
            "bt-refit-v2-qc theta. Same bothHeldOut face-id split as labs_composite_eval."
        ),
    }

    out_json = Path(a.out) if a.out else art / "bothHeldOut-error-audit.json"
    out_csv = out_json.with_suffix(".csv")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "errors": errors}
    out_json.write_text(json.dumps(payload, indent=2))
    if errors:
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(errors[0].keys()))
            w.writeheader()
            w.writerows(errors)

    print(f"{a.run} bothHeldOut error audit (uniform run-4, panel majority)")
    print(f"  pairs: {both_held}  model acc: {summary['modelAccuracy']:.1%}  "
          f"errors: {len(errors)}")
    print(f"  model wrong & control right: {wrong_model_right_ctrl}")
    print(f"  both wrong:                  {both_wrong}")
    print(f"  model wrong & control tie:   {wrong_model_wrong_ctrl_tie}")
    print(f"  model right & control wrong: {model_right_ctrl_wrong}")
    print(f"-> {out_json.relative_to(ROOT)}")
    if errors:
        print(f"-> {out_csv.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
