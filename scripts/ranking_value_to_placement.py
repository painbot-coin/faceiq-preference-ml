"""Does a ranking built on more human votes make the production score agree with humans more?

This is the direct evidence for or against the $7,463 buy zone, and it is not the same question the
playbook's headroom table answers. That table asks "would a purchased vote beat the free Gemini label
on this pair" — a question about *label quality*. This asks the downstream one: **does better label
quality reach the number a user sees?**

The reference set's theta comes from a BT refit, and each successive refit folded in another panel
run, so the refits form a natural dose ladder: v2-qc has no human votes, v5-panel has 65,894. Hold the
comparator fixed, swap only the ranking that supplies reference theta, and score every variant against
the same held-out human votes.

Note the bias runs *toward* finding an effect: run 4's votes are inside `bt-refit-v5-panel`, so
evaluating v5 on run 4 pairs is partly in-sample and should flatter it. If it still comes out flat,
the null is real.

Usage:
    python scripts/ranking_value_to_placement.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from accuracy_matrix import ceilings, load_votes  # noqa: E402

from faceiq_pref.placement import place, stratified_reference_ids  # noqa: E402

LADDER = [
    ("bt-refit-v2-qc", "none (VLM only)"),
    ("bt-refit-v3-panel", "run 1"),
    ("bt-refit-v4-panel", "runs 1+3"),
    ("bt-refit-v5-panel", "runs 1+3+4"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="train-v14-panel-ship")
    ap.add_argument("--panel", default="labels/panel-run-4-random")
    ap.add_argument("--rejects", default="artifacts/panel-run-v4/reject-pids.txt")
    ap.add_argument("--references", type=int, default=200)
    ap.add_argument("--out", default="artifacts/ranking-value-to-placement.json")
    a = ap.parse_args()

    rejects = {ln.strip() for ln in (ROOT / a.rejects).read_text().splitlines() if ln.strip()}
    votes, meta = load_votes(ROOT / a.panel / "results",
                             ROOT / a.panel / "sample-meta.json", rejects, True)
    ceil = ceilings(votes)
    scores = pd.read_csv(ROOT / "artifacts" / a.run / "model_scores.csv")

    rows = []
    for name, note in LADDER:
        path = ROOT / "artifacts" / name / "ratings.csv"
        if not path.exists():
            continue
        rank = pd.read_csv(path)
        placed: dict[str, float] = {}
        for gender in ("female", "male"):
            rg = rank[rank["gender"] == gender]
            sg = scores[scores["gender"] == gender]
            theta_of = dict(zip(rg["faceId"], rg["theta"]))
            pct_of = dict(zip(rg["faceId"], rg["percentile"]))
            score_of = dict(zip(sg["faceId"], sg["modelScore"]))
            usable = {f: p for f, p in pct_of.items() if f in score_of}
            if len(usable) < 50:
                continue
            ref_ids = stratified_reference_ids(usable, n=a.references)
            ref_thetas = [theta_of[f] for f in ref_ids]
            ref_scores = [score_of[f] for f in ref_ids]
            population = list(theta_of.values())
            for f in usable:
                placed[f] = place(score_of[f], ref_scores, ref_thetas,
                                  population_thetas=population).score_ten

        maj_hit = maj_n = 0
        vote_hit = vote_n = 0.0
        for pid, (va, vb) in votes.items():
            m = meta.get(pid)
            if m is None:
                continue
            fa, fb = m["faceAId"], m["faceBId"]
            if fa not in placed or fb not in placed or placed[fa] == placed[fb]:
                continue
            picks_a = placed[fa] > placed[fb]
            vote_hit += va if picks_a else vb
            vote_n += va + vb
            if va != vb:
                maj_hit += int(picks_a == (va > vb))
                maj_n += 1
        rows.append({"ranking": name, "panelRuns": note, "vsMajority": maj_hit / maj_n,
                     "vsVotes": vote_hit / vote_n, "pairs": maj_n})

    spread = max(r["vsMajority"] for r in rows) - min(r["vsMajority"] for r in rows)
    result = {"comparator": a.run, "panel": a.panel, "references": a.references,
              "ceiling": {k: v for k, v in ceil.items() if isinstance(v, float)},
              "ladder": rows, "spreadVsMajority": spread,
              "verdict": ("flat — more votes in the ranking do not reach the production score"
                          if spread < 0.01 else "a dose effect is present, investigate")}
    (ROOT / a.out).write_text(json.dumps(result, indent=2))

    print(f"\ncomparator held fixed at {a.run}; only the reference theta changes")
    print(f"human ceiling: vsMajority {ceil['vsMajority']:.1%}  vsVotes {ceil['vsVotes']:.1%}")
    print(f"\n{'ranking supplying reference theta':34} {'panel runs in it':>17} "
          f"{'vsMajority':>11} {'vsVotes':>9}")
    for r in rows:
        print(f"{r['ranking']:34} {r['panelRuns']:>17} {r['vsMajority']:>11.1%} "
              f"{r['vsVotes']:>9.1%}")
    print(f"\nspread across the whole ladder: {spread:.1%} -> {result['verdict']}")
    if spread < 0.01:
        print("65,894 human votes moved the production score by less than a point of accuracy.\n"
              "Reference thetas enter the fit as fixed constants with weight at most 0.25 each,\n"
              "spread over 200 references, so improvements to them average out. The placement is\n"
              "dominated by the COMPARATOR's ordering. Money spent on labels has to reach the user\n"
              "through the comparator's TRAINING SET, not through the reference set.")


if __name__ == "__main__":
    main()
