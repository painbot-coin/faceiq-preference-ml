"""Does blending the Labs deterministic score into the placed /10 help? Two targets, two answers.

The interesting part of this script is that it scores the same blend against **two different
targets** and they disagree, which is the whole reason it exists:

* **against BT /10** — the blend looks like a clear, statistically significant win.
* **against real human votes** — the blend is flat to slightly worse, at every weight.

Both cannot be improvements. BT theta is a *proxy* for what people think, fitted on Gemini pairwise
labels; the panel votes are the thing itself. When a change moves the proxy and not the target, it is
fitting the proxy's error. That is the confirmation-bias failure mode this programme has worried
about in the abstract, caught in the concrete: run only the first measurement and you ship a "+0.05
significant improvement" that does nothing for a single user.

So the verdict is **do not blend**, and the script stays because the negative result is worth being
able to re-run when the EBM arrives — the EBM is the same architecture with a better second opinion,
and this is the harness that will judge it.

Labs scores need no computation: `labsOverallScore` is already on all 3,000 faces in `faces.jsonl`.
It is rank-normalised onto our /10 through the same anchor curve before blending, because the two
scales are not otherwise comparable.

Usage:
    python scripts/labs_composite_eval.py
    python scripts/labs_composite_eval.py --run train-v12-panel-soft
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from accuracy_matrix import ceilings, load_votes  # noqa: E402

from faceiq_pref.calibrate import make_scorer  # noqa: E402
from faceiq_pref.placement import (  # noqa: E402
    ANCHORS_TOP10,
    place,
    stratified_reference_ids,
    tier_of,
)

WEIGHTS = (1.0, 0.9, 0.85, 0.8, 0.7, 0.5, 0.0)
MIN_VAL_FRACTION = 0.2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="train-v14-panel-ship")
    ap.add_argument("--ranking", default="artifacts/bt-refit-v5-panel")
    ap.add_argument("--panel", default="labels/panel-run-4-random")
    ap.add_argument("--rejects", default="artifacts/panel-run-v4/reject-pids.txt")
    ap.add_argument("--references", type=int, default=200)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out) if a.out else ROOT / "artifacts" / a.run / "labs-composite.json"

    export = next(p for p in sorted((ROOT / "data" / "exports").glob("*"))
                  if (p / "faces.jsonl").exists())
    faces = pd.DataFrame(json.loads(ln)
                         for ln in (export / "faces.jsonl").read_text().splitlines())
    labs = dict(zip(faces["faceId"], faces["labsOverallScore"]))
    rank = pd.read_csv(ROOT / a.ranking / "ratings.csv")
    to_ten = make_scorer(ANCHORS_TOP10)
    scores = pd.read_csv(ROOT / "artifacts" / a.run / "model_scores.csv")

    # Place every cohort face once, and put Labs on the same /10 ladder by rank.
    placed: dict[str, float] = {}
    labs_ten: dict[str, float] = {}
    truth: dict[str, float] = {}
    for gender in ("female", "male"):
        rg = rank[rank["gender"] == gender]
        sg = scores[scores["gender"] == gender]
        theta_of = dict(zip(rg["faceId"], rg["theta"]))
        pct_of = dict(zip(rg["faceId"], rg["percentile"]))
        score_of = dict(zip(sg["faceId"], sg["modelScore"]))
        usable = {f: p for f, p in pct_of.items() if f in score_of}
        ref_ids = stratified_reference_ids(usable, n=a.references)
        ref_thetas = [theta_of[f] for f in ref_ids]
        ref_scores = [score_of[f] for f in ref_ids]
        population = list(theta_of.values())
        ladder = sorted(labs[f] for f in usable if f in labs)
        for f in usable:
            placed[f] = place(score_of[f], ref_scores, ref_thetas,
                              population_thetas=population).score_ten
            labs_ten[f] = to_ten(float(np.searchsorted(ladder, labs[f]) / len(ladder)))
            truth[f] = to_ten(pct_of[f])

    ids = sorted(scores["faceId"])
    random.Random(42).shuffle(ids)
    held_out = set(ids[: int(len(ids) * MIN_VAL_FRACTION)])
    ref_all = set()
    for gender in ("female", "male"):
        rg = rank[rank["gender"] == gender]
        sg = scores[scores["gender"] == gender]
        usable = {f: p for f, p in zip(rg["faceId"], rg["percentile"])
                  if f in set(sg["faceId"])}
        ref_all |= set(stratified_reference_ids(usable, n=a.references))
    subjects = [f for f in held_out if f in placed and f not in ref_all]

    # ---- target 1: BT /10 ------------------------------------------------------
    p = np.array([placed[f] for f in subjects])
    lt = np.array([labs_ten[f] for f in subjects])
    t = np.array([truth[f] for f in subjects])
    tiers = np.array([tier_of(v)[0] for v in t])
    rng = np.random.default_rng(0)
    vs_bt = []
    for w in WEIGHTS:
        blend = w * p + (1 - w) * lt
        err = np.abs(blend - t)
        base = np.abs(p - t)
        idx = rng.integers(0, len(p), (2000, len(p)))
        gain = base[idx].mean(1) - err[idx].mean(1)
        lo, hi = np.percentile(gain, [2.5, 97.5])
        vs_bt.append({
            "w": w, "median": float(np.median(err)), "mean": float(err.mean()),
            "p90": float(np.percentile(err, 90)),
            "tierExact": float((np.array([tier_of(v)[0] for v in blend]) == tiers).mean()),
            "gainCI": [float(lo), float(hi)],
        })

    # ---- target 2: real human votes --------------------------------------------
    rejects = {ln.strip() for ln in (ROOT / a.rejects).read_text().splitlines() if ln.strip()}
    votes, meta = load_votes(ROOT / a.panel / "results",
                             ROOT / a.panel / "sample-meta.json", rejects, True)
    vs_panel = []
    for w in WEIGHTS:
        maj_hit = maj_n = 0
        vote_hit = vote_n = 0.0
        for pid, (va, vb) in votes.items():
            m = meta.get(pid)
            if m is None:
                continue
            fa, fb = m["faceAId"], m["faceBId"]
            if fa not in placed or fb not in placed:
                continue
            sa = w * placed[fa] + (1 - w) * labs_ten[fa]
            sb = w * placed[fb] + (1 - w) * labs_ten[fb]
            if sa == sb:
                continue
            picks_a = sa > sb
            vote_hit += va if picks_a else vb
            vote_n += va + vb
            if va != vb:
                maj_hit += int(picks_a == (va > vb))
                maj_n += 1
        vs_panel.append({"w": w, "vsMajority": maj_hit / maj_n, "vsVotes": vote_hit / vote_n,
                         "pairs": maj_n})

    err_corr = float(np.corrcoef(p - t, lt - t)[0, 1])
    ceil = ceilings(votes)
    result = {
        "run": a.run, "ranking": a.ranking, "panel": a.panel,
        "subjects": len(subjects), "errorCorrelation": err_corr,
        "spearmanLabsVsBT": float(spearmanr(lt, t).statistic),
        "spearmanPlacedVsBT": float(spearmanr(p, t).statistic),
        "ceiling": {k: v for k, v in ceil.items() if isinstance(v, float)},
        "vsBT": vs_bt, "vsPanel": vs_panel,
        "verdict": "do not blend — helps against BT, flat-to-worse against humans",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print(f"\n{a.run}, {len(subjects)} held-out subjects, {vs_panel[0]['pairs']} panel pairs")
    print(f"error correlation placed vs Labs: {err_corr:+.3f}  "
          f"(near zero, which is why the BT-target result looks so good)")
    print(f"\n{'w on comparator':>16} | {'BT med':>7} {'BT p90':>7} {'BT tier':>8} | "
          f"{'vs majority':>12} {'vs votes':>9}")
    print(f"{'':>16} | {'':>7} {'':>7} {'':>8} | "
          f"{ceil['vsMajority']:>11.1%}* {ceil['vsVotes']:>8.1%}*  <- human ceiling")
    for b, q in zip(vs_bt, vs_panel):
        tag = "  <- placement only" if b["w"] == 1.0 else ("  <- Labs only" if b["w"] == 0 else "")
        print(f"{b['w']:>16.0%} | {b['median']:>7.3f} {b['p90']:>7.2f} {b['tierExact']:>8.1%} | "
              f"{q['vsMajority']:>12.1%} {q['vsVotes']:>9.1%}{tag}")
    best_bt = min(vs_bt, key=lambda r: r["median"])
    best_panel = max(vs_panel, key=lambda r: r["vsMajority"])
    print(f"\nbest against BT:     w={best_bt['w']:.2f}  (median {best_bt['median']:.3f})")
    print(f"best against humans: w={best_panel['w']:.2f}  ({best_panel['vsMajority']:.1%})")
    print("\nThe two targets disagree. Humans are the target and BT is the proxy, so the answer "
          "is\nDO NOT BLEND. A change that moves the proxy and not the target is fitting the "
          "proxy's error.")
    print(f"-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
