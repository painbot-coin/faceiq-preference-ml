#!/usr/bin/env python
"""Eval-only placement / readout ablations on bothHeldOut panel majority.

Yardstick matches labs_composite_eval --leakage-split: uniform run-4 pairs,
panel majority, bothHeldOut faces from the run's face-id split, control =
vote-blind bt-refit-v2-qc. No training.

Usage:
    python scripts/placement_readout_ablation.py --run train-v25-panel-ft-v23b
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from accuracy_matrix import load_votes  # noqa: E402
from faceiq_pref.data import load_export, split_by_face_id  # noqa: E402
from faceiq_pref.placement import (  # noqa: E402
    ANCHORS_TOP10,
    UNBOUNDED_MARGIN,
    place,
    stratified_reference_ids,
)
from faceiq_pref.calibrate import make_scorer  # noqa: E402


def fit_theta_soft(
    outcomes: list[float], ref_thetas: list[float], max_iter: int = 60
) -> tuple[float, float, bool]:
    """Conditional MLE with soft outcomes in [0, 1] (hard bools are 0/1 special case)."""
    if not ref_thetas:
        raise ValueError("reference set is empty")
    if all(y >= 1.0 - 1e-12 for y in outcomes):
        return max(ref_thetas) + UNBOUNDED_MARGIN, float("inf"), False
    if all(y <= 1e-12 for y in outcomes):
        return min(ref_thetas) - UNBOUNDED_MARGIN, float("inf"), False

    theta = sum(ref_thetas) / len(ref_thetas)
    info = 0.0
    for _ in range(max_iter):
        grad = info = 0.0
        for y, tj in zip(outcomes, ref_thetas):
            p = 1.0 / (1.0 + math.exp(-(theta - tj)))
            grad += y - p
            info += p * (1 - p)
        if info < 1e-12:
            break
        step = grad / info
        theta += max(-2.0, min(2.0, step))
        if abs(step) < 1e-9:
            break
    return theta, (1.0 / math.sqrt(info) if info > 0 else float("inf")), True


def place_soft(
    score: float,
    reference_scores: list[float],
    reference_thetas: list[float],
    population_thetas: list[float],
    temperature: float,
) -> float:
    """Soft-margin placement: y_j = sigmoid((score - rs_j) / T); return score_ten."""
    to_ten = make_scorer(ANCHORS_TOP10)
    population = sorted(population_thetas)
    outcomes = [
        1.0 / (1.0 + math.exp(-(score - rs) / temperature)) for rs in reference_scores
    ]
    theta, _, _ = fit_theta_soft(outcomes, reference_thetas)

    def pct_of(t: float) -> float:
        lo, hi = 0, len(population)
        while lo < hi:
            mid = (lo + hi) // 2
            if population[mid] < t:
                lo = mid + 1
            else:
                hi = mid
        return lo / len(population)

    return to_ten(pct_of(theta))


def zscore_by_gender(scores: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for _, g in scores.groupby("gender"):
        mu = float(g["modelScore"].mean())
        sd = float(g["modelScore"].std(ddof=0))
        if sd < 1e-12:
            sd = 1.0
        for fid, s in zip(g["faceId"], g["modelScore"]):
            out[fid] = (float(s) - mu) / sd
    return out


def place_all(
    score_of: dict[str, float],
    rank: pd.DataFrame,
    n_refs: int,
    mode: str,
    soft_t: float | None = None,
) -> dict[str, float]:
    placed: dict[str, float] = {}
    for gender in ("female", "male"):
        rg = rank[rank["gender"] == gender]
        theta_of = dict(zip(rg["faceId"], rg["theta"]))
        pct_of = dict(zip(rg["faceId"], rg["percentile"]))
        usable = {f: p for f, p in pct_of.items() if f in score_of}
        ref_ids = stratified_reference_ids(usable, n=n_refs)
        ref_thetas = [theta_of[f] for f in ref_ids]
        ref_scores = [score_of[f] for f in ref_ids]
        population = list(theta_of.values())
        for f in usable:
            if mode == "hard":
                placed[f] = place(
                    score_of[f], ref_scores, ref_thetas, population_thetas=population
                ).score_ten
            elif mode == "soft":
                assert soft_t is not None
                placed[f] = place_soft(
                    score_of[f], ref_scores, ref_thetas, population, soft_t
                )
            elif mode == "raw_score":
                placed[f] = score_of[f]
            else:
                raise ValueError(mode)
    return placed


def bothheldout_stats(
    scores_for_pick: dict[str, float],
    votes,
    meta,
    held_out: set[str],
    control: dict[str, float],
) -> dict:
    rows = []
    for pid, (va, vb) in votes.items():
        m = meta.get(pid)
        if m is None or va == vb:
            continue
        fa, fb = m["faceAId"], m["faceBId"]
        if fa not in held_out or fb not in held_out:
            continue
        if fa not in scores_for_pick or fb not in scores_for_pick:
            continue
        if fa not in control or fb not in control:
            continue
        sa, sb = scores_for_pick[fa], scores_for_pick[fb]
        if sa == sb:
            continue
        maj_a = va > vb
        model_ok = int((sa > sb) == maj_a)
        ctrl_ok = int((control[fa] > control[fb]) == maj_a) if control[fa] != control[fb] else 0
        rows.append((model_ok, ctrl_ok, pid))

    n = len(rows)
    if not n:
        return {"pairs": 0, "placed": float("nan"), "control": float("nan"), "margin": float("nan")}
    placed_acc = float(np.mean([a for a, _, _ in rows]))
    ctrl_acc = float(np.mean([b for _, b, _ in rows]))
    d = np.array([a - b for a, b, _ in rows])
    rng = np.random.default_rng(0)
    boot = d[rng.integers(0, len(d), (4000, len(d)))].mean(1)
    return {
        "pairs": n,
        "placed": placed_acc,
        "control": ctrl_acc,
        "margin": float(d.mean()),
        "marginCi95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "correct": int(sum(a for a, _, _ in rows)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="train-v25-panel-ft-v23b")
    ap.add_argument("--ranking", default="artifacts/bt-refit-v5-panel")
    ap.add_argument("--control-ranking", default="artifacts/bt-refit-v2-qc")
    ap.add_argument("--panel", default="labels/panel-run-4-random")
    ap.add_argument("--rejects", default="artifacts/panel-run-v4/reject-pids.txt")
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument(
        "--out",
        default=None,
        help="default: artifacts/<run>/placement-readout-ablation.json",
    )
    a = ap.parse_args()

    art = ROOT / "artifacts" / a.run
    scores = pd.read_csv(art / "model_scores.csv")
    raw_scores = dict(zip(scores["faceId"], scores["modelScore"].astype(float)))
    z_scores = zscore_by_gender(scores)

    rank = pd.read_csv(ROOT / a.ranking / "ratings.csv")
    ctrl = pd.read_csv(ROOT / a.control_ranking / "ratings.csv")
    control = dict(zip(ctrl["faceId"], ctrl["theta"]))

    export = next(
        p for p in sorted((ROOT / "data" / "exports").glob("*")) if (p / "faces.jsonl").exists()
    )
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

    ablations: list[tuple[str, dict]] = []

    # 1) Raw comparator sign (no placement)
    ablations.append(
        (
            "raw_modelScore",
            {
                "mode": "raw_score",
                "n_refs": None,
                "score_transform": "none",
                "soft_T": None,
            },
        )
    )
    ablations.append(
        (
            "raw_modelScore_zscore_gender",
            {
                "mode": "raw_score",
                "n_refs": None,
                "score_transform": "zscore_gender",
                "soft_T": None,
            },
        )
    )

    # 2) Hard placement, reference count sweep (tier-stratified)
    for n in (50, 100, 200, 400):
        ablations.append(
            (
                f"place_hard_refs{n}",
                {
                    "mode": "hard",
                    "n_refs": n,
                    "score_transform": "none",
                    "soft_T": None,
                },
            )
        )

    # 3) Soft-margin placement at default 200 refs
    for t in (0.5, 1.0, 2.0):
        ablations.append(
            (
                f"place_soft_T{t}_refs200",
                {
                    "mode": "soft",
                    "n_refs": 200,
                    "score_transform": "none",
                    "soft_T": t,
                },
            )
        )

    # 4) Soft + z-scored scores (magnitude scale identified within gender)
    ablations.append(
        (
            "place_soft_T1_zscore_refs200",
            {
                "mode": "soft",
                "n_refs": 200,
                "score_transform": "zscore_gender",
                "soft_T": 1.0,
            },
        )
    )

    # Ship baseline: hard place 200 is the production path in labs_composite_eval
    baseline_name = "place_hard_refs200"
    results = []
    for name, cfg in ablations:
        score_src = z_scores if cfg["score_transform"] == "zscore_gender" else raw_scores
        if cfg["mode"] == "raw_score":
            picks = score_src
        else:
            picks = place_all(
                score_src,
                rank,
                int(cfg["n_refs"]),
                cfg["mode"],
                soft_t=cfg["soft_T"],
            )
        stats = bothheldout_stats(picks, votes, meta, held_out, control)
        row = {"name": name, **cfg, **stats}
        results.append(row)
        print(
            f"{name:36s}  bothHeldOut={stats['placed']:.1%}  "
            f"n={stats['pairs']}  control={stats['control']:.1%}  "
            f"margin={stats['margin']:+.1%}"
        )

    baseline = next(r for r in results if r["name"] == baseline_name)
    base_acc = baseline["placed"]
    for r in results:
        r["deltaVsShipPlace"] = (
            None if math.isnan(r["placed"]) else float(r["placed"] - base_acc)
        )

    clear_wins = [
        r
        for r in results
        if r["name"] != baseline_name
        and not math.isnan(r["placed"])
        and r["placed"] >= base_acc + 0.02  # ≥ +2 pts on n≈110
    ]
    # Also note any that beat raw ceiling meaningfully
    verdict = (
        "SHIP candidate — at least one readout ablation ≥ +2 pts vs place_hard_refs200"
        if clear_wins
        else "KILL placement/readout as accuracy lever — no ablation clears +2 pts on bothHeldOut"
    )

    out = {
        "run": a.run,
        "panel": a.panel,
        "pairDistribution": "uniform run-4",
        "metric": "panel majority",
        "leakageStratum": "bothHeldOut",
        "ranking": a.ranking,
        "controlRanking": a.control_ranking,
        "shipBaseline": baseline_name,
        "shipBaselineAcc": base_acc,
        "yardstickNote": (
            "labs_composite_eval --leakage-split picks winners by placed score_ten "
            "(hard sign vs tier-stratified refs, default n=200). Raw modelScore "
            "pairwise is a readout ablation, not the production path."
        ),
        "verdict": verdict,
        "clearWins": [r["name"] for r in clear_wins],
        "ablations": results,
    }
    out_path = Path(a.out) if a.out else art / "placement-readout-ablation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nverdict: {verdict}")
    print(f"-> {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
