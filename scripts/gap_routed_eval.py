#!/usr/bin/env python
"""Gap-routed ensemble: comparator on near pairs, ranking/placement on far ones.

Programme hypothesis (README priority 1c, programme-direction-review §): the comparator
beats the ranking under a ~10-percentile-point gap and loses 10–45. Routing is not a
global blend (§5.6 ruled those out). Measure against **panel majority** on held-out pairs.

Two routers:

* **Oracle** — route on the true BT percentile gap. Upper bound; not available at upload
  time for two strangers, but tells us whether the hypothesis has headroom at all.
* **Predicted** — route on |placed_a − placed_b|. Available in production for any two
  scored faces. If this fails while oracle works, the gap signal is the bottleneck.

Far-pair arm for oracle uses `bt-refit-v2-qc` (predates panel votes) so majority numbers
are not circular. Predicted far-pair arm uses placement (the production score).

Usage:
    python scripts/gap_routed_eval.py
    python scripts/gap_routed_eval.py --run train-v22-ship-0.2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from accuracy_matrix import ceilings, load_votes  # noqa: E402
from faceiq_pref.calibrate import make_scorer  # noqa: E402
from faceiq_pref.data import load_export, split_by_face_id  # noqa: E402
from faceiq_pref.placement import ANCHORS_TOP10, place, stratified_reference_ids  # noqa: E402

MIN_VAL_FRACTION = 0.2
# Percentile-point thresholds to sweep for oracle routing (programme default: 10).
PCT_THRESHOLDS = (5.0, 10.0, 15.0, 20.0, 30.0, 45.0)
# /10 thresholds for predicted-gap routing (T*ln2 ≈ 1.33 is one resolvable band).
TEN_THRESHOLDS = (0.25, 0.5, 1.0, 1.33, 2.0, 3.0)


def place_all(run: str, ranking: Path, n_refs: int) -> tuple[dict, dict, dict, dict]:
    """-> placed /10, modelScore, BT percentile, BT /10 for every scored face."""
    export = next(p for p in sorted((ROOT / "data" / "exports").glob("*"))
                  if (p / "faces.jsonl").exists())
    rank = pd.read_csv(ranking / "ratings.csv")
    scores = pd.read_csv(ROOT / "artifacts" / run / "model_scores.csv")
    to_ten = make_scorer(ANCHORS_TOP10)

    placed: dict[str, float] = {}
    model: dict[str, float] = {}
    pct: dict[str, float] = {}
    bt_ten: dict[str, float] = {}
    for gender in ("female", "male"):
        rg = rank[rank["gender"] == gender]
        sg = scores[scores["gender"] == gender]
        theta_of = dict(zip(rg["faceId"], rg["theta"]))
        pct_of = dict(zip(rg["faceId"], rg["percentile"]))
        score_of = dict(zip(sg["faceId"], sg["modelScore"]))
        usable = {f: p for f, p in pct_of.items() if f in score_of}
        ref_ids = stratified_reference_ids(usable, n=n_refs)
        ref_thetas = [theta_of[f] for f in ref_ids]
        ref_scores = [score_of[f] for f in ref_ids]
        population = list(theta_of.values())
        for f in usable:
            placed[f] = place(score_of[f], ref_scores, ref_thetas,
                              population_thetas=population).score_ten
            model[f] = score_of[f]
            pct[f] = pct_of[f]
            bt_ten[f] = to_ten(pct_of[f])
    return placed, model, pct, bt_ten


def score_picks(votes, meta, picks_a: dict[str, bool]) -> dict:
    maj_hit = maj_n = 0
    vote_hit = vote_n = 0.0
    for pid, (va, vb) in votes.items():
        if pid not in picks_a:
            continue
        a_wins = picks_a[pid]
        vote_hit += va if a_wins else vb
        vote_n += va + vb
        if va != vb:
            maj_hit += int(a_wins == (va > vb))
            maj_n += 1
    return {
        "vsMajority": maj_hit / maj_n if maj_n else float("nan"),
        "vsVotes": vote_hit / vote_n if vote_n else float("nan"),
        "pairs": maj_n,
    }


def bootstrap_delta(hit_a: list[int], hit_b: list[int], n_boot: int = 4000) -> dict:
    """Paired bootstrap of accuracy_a − accuracy_b."""
    a = np.asarray(hit_a, dtype=float)
    b = np.asarray(hit_b, dtype=float)
    d = a - b
    rng = np.random.default_rng(0)
    boot = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    return {
        "delta": float(d.mean()),
        "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "pairs": int(len(d)),
    }


def build_rows(votes, meta, placed, model, pct, far_score, held_out):
    """One row per majority-labelled pair both scorers can judge."""
    rows = []
    for pid, (va, vb) in votes.items():
        m = meta.get(pid)
        if m is None or va == vb:
            continue
        fa, fb = m["faceAId"], m["faceBId"]
        if fa not in placed or fb not in placed or fa not in model or fb not in model:
            continue
        if fa not in pct or fb not in pct or fa not in far_score or fb not in far_score:
            continue
        if placed[fa] == placed[fb] and model[fa] == model[fb]:
            continue
        maj_a = va > vb
        rows.append({
            "pairId": pid,
            "majA": maj_a,
            "gapPct": abs(pct[fa] - pct[fb]) * 100.0,  # percentile points 0–100
            "gapTen": abs(placed[fa] - placed[fb]),
            "cmpA": model[fa] > model[fb],
            "placedA": placed[fa] > placed[fb],
            "farA": far_score[fa] > far_score[fb],
            "nHeldOut": (fa in held_out) + (fb in held_out),
        })
    return rows


def route_picks(rows, gap_key: str, threshold: float, near_key: str, far_key: str):
    """-> picks_a dict and per-pair hit lists vs placement-only for delta."""
    picks = {}
    hit_route, hit_placed, hit_cmp, hit_far = [], [], [], []
    n_near = 0
    for r in rows:
        use_near = r[gap_key] < threshold
        pick = r[near_key] if use_near else r[far_key]
        picks[r["pairId"]] = pick
        n_near += int(use_near)
        hit_route.append(int(pick == r["majA"]))
        hit_placed.append(int(r["placedA"] == r["majA"]))
        hit_cmp.append(int(r["cmpA"] == r["majA"]))
        hit_far.append(int(r["farA"] == r["majA"]))
    return picks, {
        "nNear": n_near,
        "nFar": len(rows) - n_near,
        "fracNear": n_near / len(rows) if rows else float("nan"),
        "vsPlaced": bootstrap_delta(hit_route, hit_placed),
        "vsComparator": bootstrap_delta(hit_route, hit_cmp),
        "vsFarArm": bootstrap_delta(hit_route, hit_far),
    }


def stratum(rows, n_held: int | None):
    if n_held is None:
        return rows
    return [r for r in rows if r["nHeldOut"] == n_held]


def eval_router(rows, gap_key, thresholds, near_key, far_key, votes, meta):
    out = []
    for t in thresholds:
        picks, stats = route_picks(rows, gap_key, t, near_key, far_key)
        scored = score_picks(votes, meta, picks)
        out.append({"threshold": t, **scored, **stats})
    return out


def pure_arms(rows, votes, meta):
    return {
        "comparator": score_picks(votes, meta, {r["pairId"]: r["cmpA"] for r in rows}),
        "placement": score_picks(votes, meta, {r["pairId"]: r["placedA"] for r in rows}),
        "farArm": score_picks(votes, meta, {r["pairId"]: r["farA"] for r in rows}),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="train-v14-panel-ship")
    ap.add_argument("--ranking", default="artifacts/bt-refit-v5-panel",
                    help="ranking used to place faces (production yardstick)")
    ap.add_argument("--far-ranking", default="artifacts/bt-refit-v2-qc",
                    help="oracle far-arm ranking; must predate panel votes")
    ap.add_argument("--panel", default="labels/panel-run-4-random")
    ap.add_argument("--rejects", default="artifacts/panel-run-v4/reject-pids.txt")
    ap.add_argument("--references", type=int, default=200)
    ap.add_argument("--val-fraction", type=float, default=MIN_VAL_FRACTION)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out) if a.out else ROOT / "artifacts" / a.run / "gap-routed.json"

    placed, model, pct, _bt_ten = place_all(a.run, ROOT / a.ranking, a.references)
    far_rank = pd.read_csv(ROOT / a.far_ranking / "ratings.csv")
    far_score = dict(zip(far_rank["faceId"], far_rank["theta"]))

    export = next(p for p in sorted((ROOT / "data" / "exports").glob("*"))
                  if (p / "faces.jsonl").exists())
    _, val_rows = split_by_face_id(load_export(export).all_matchups(),
                                   a.val_fraction, a.split_seed)
    held_out = {f for m in val_rows for f in (m.face_a_id, m.face_b_id)}

    rejects = {ln.strip() for ln in (ROOT / a.rejects).read_text().splitlines() if ln.strip()}
    votes, meta = load_votes(ROOT / a.panel / "results",
                             ROOT / a.panel / "sample-meta.json", rejects, True)
    rows = build_rows(votes, meta, placed, model, pct, far_score, held_out)
    ceil = ceilings(votes)

    result: dict = {
        "run": a.run,
        "ranking": a.ranking,
        "farRanking": a.far_ranking,
        "panel": a.panel,
        "successMetric": "panel majority on held-out pairs",
        "ceiling": {k: v for k, v in ceil.items() if isinstance(v, float)},
        "strata": {},
    }

    print(f"\n{a.run} - gap-routed ensemble vs panel majority")
    print(f"ceiling (one rater vs majority): {ceil['vsMajority']:.1%}")
    print(f"oracle far arm: {a.far_ranking} (vote-blind)")
    print(f"pairs scorAble: {len(rows)}\n")

    for label, n_held in (("all", None), ("bothTrained", 0), ("oneHeldOut", 1),
                          ("bothHeldOut", 2)):
        sub = stratum(rows, n_held)
        if len(sub) < 30:
            continue
        arms = pure_arms(sub, votes, meta)
        oracle = eval_router(sub, "gapPct", PCT_THRESHOLDS, "cmpA", "farA", votes, meta)
        predicted = eval_router(sub, "gapTen", TEN_THRESHOLDS, "cmpA", "placedA", votes, meta)
        # Also: near=comparator, far=placement, routed on true pct gap (oracle gap, prod arms)
        hybrid = eval_router(sub, "gapPct", PCT_THRESHOLDS, "cmpA", "placedA", votes, meta)

        best_oracle = max(oracle, key=lambda r: r["vsMajority"])
        best_pred = max(predicted, key=lambda r: r["vsMajority"])
        best_hybrid = max(hybrid, key=lambda r: r["vsMajority"])

        result["strata"][label] = {
            "pairs": len(sub),
            "arms": arms,
            "oraclePctGap": oracle,
            "predictedTenGap": predicted,
            "oracleGapPlacementFar": hybrid,
            "bestOracle": best_oracle,
            "bestPredicted": best_pred,
            "bestHybrid": best_hybrid,
        }

        print(f"=== {label}  (n={len(sub)}) ===")
        print(f"  pure  comparator {arms['comparator']['vsMajority']:.1%}   "
              f"placement {arms['placement']['vsMajority']:.1%}   "
              f"ranking(v2) {arms['farArm']['vsMajority']:.1%}")
        print(f"  best ORACLE   (pct gap<{best_oracle['threshold']:.0f} -> cmp, else rank): "
              f"{best_oracle['vsMajority']:.1%}  "
              f"d vs placement {best_oracle['vsPlaced']['delta']:+.1%} "
              f"[{best_oracle['vsPlaced']['ci95'][0]:+.1%}, "
              f"{best_oracle['vsPlaced']['ci95'][1]:+.1%}]  "
              f"near {best_oracle['fracNear']:.0%}")
        print(f"  best PREDICTED (|d/10|<{best_pred['threshold']} -> cmp, else placed): "
              f"{best_pred['vsMajority']:.1%}  "
              f"d vs placement {best_pred['vsPlaced']['delta']:+.1%} "
              f"[{best_pred['vsPlaced']['ci95'][0]:+.1%}, "
              f"{best_pred['vsPlaced']['ci95'][1]:+.1%}]  "
              f"near {best_pred['fracNear']:.0%}")
        print(f"  best HYBRID   (true pct gap, far=placement): "
              f"{best_hybrid['vsMajority']:.1%}  "
              f"d vs placement {best_hybrid['vsPlaced']['delta']:+.1%} "
              f"[{best_hybrid['vsPlaced']['ci95'][0]:+.1%}, "
              f"{best_hybrid['vsPlaced']['ci95'][1]:+.1%}]")
        print()

    # Programme default: oracle at 10 pct points on both-held-out
    focus = result["strata"].get("bothHeldOut") or result["strata"].get("all")
    if focus:
        o10 = next(r for r in focus["oraclePctGap"] if r["threshold"] == 10.0)
        result["verdict"] = {
            "oracleAt10": o10,
            "bestOracle": focus["bestOracle"],
            "bestPredicted": focus["bestPredicted"],
            "note": (
                "Ship gap-routing only if bestPredicted (or oracleAt10) beats placement "
                "on bothHeldOut with CI clear of zero. Oracle headroom with CI on zero "
                "means the hypothesis has no recoverable gain even with perfect gap knowledge."
            ),
        }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
