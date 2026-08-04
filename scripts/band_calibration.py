#!/usr/bin/env python
"""Turn a point /10 score into a defensible band, using measured crowd agreement.

Two different things get called a "band" and conflating them produces nonsense, so this script
computes the one that can be measured today and prints what the other one needs.

**Disagreement band (this script).** People genuinely differ. Fit `P(the higher score wins) =
sigmoid(/10 gap / T)` against real votes and invert it: the band around a score is the width
you must move before a stated fraction of the crowd would agree with the new ordering. This
band does **not** shrink with more data or more photos — it is a property of the population,
and it is the honest thing to put in front of a user.

**Estimation band (not this script).** How well we know *this face's* score given the
comparisons we have. That is `scripts/bt_uncertainty.py` (median 95% interval 0.91 /10 points),
and it does shrink, as 1/sqrt(comparisons).

**Photo band (unmeasured).** How much a score moves between two photos of the same person.
Nothing in this export can measure it — all 3,000 rows are one front view of 3,000 distinct
people. Any "upload more photos and the band narrows" feature is entirely gated on it, because
that is the only component averaging over photos can reduce.

Fit `T` on pairs the ranking was **not** fitted on, or the curve is circular. Run 4's 2,500
uniform pairs are the honest set (they are inside `bt-refit-v5-panel` now, so pass
`bt-refit-v2-qc`, which predates every panel vote).

Usage:
    python scripts/band_calibration.py \
        --ratings artifacts/bt-refit-v2-qc/ratings.csv \
        --panel labels/panel-run-4-random/results labels/panel-run-4-random/sample-meta.json \
        --rejects artifacts/panel-run-v4/reject-pids.txt \
        --out artifacts/panel-run-v4/band-calibration.json
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

AGREEMENT_LEVELS = [0.55, 0.60, 2 / 3, 0.75, 0.80, 0.90, 0.95]
GAPS = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]


def load_pairs(ratings: Path, panels, rejects: set[str]):
    """-> [(|/10 gap|, votes for the higher-scoring face, total votes)] and the score table."""
    score: dict[str, float] = {}
    gender: dict[str, str] = {}
    with open(ratings, newline="") as fh:
        for r in csv.DictReader(fh):
            score[r["faceId"]] = float(r["scoreOutOf10"])
            gender[r["faceId"]] = r["gender"]

    votes: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0])
    for results_dir, meta_path in panels:
        meta = json.loads(Path(meta_path).read_text())["pairs"]
        by_id = {m["pairId"]: m for m in meta}
        for line in (Path(results_dir) / "judgments.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            j = json.loads(line)
            if j["isGold"] or j["studyId"] == "test" or j["prolificPid"] in rejects:
                continue
            m = by_id.get(j["pairId"])
            if m is None:
                continue
            cell = votes[(str(meta_path), j["pairId"])]
            a_side = "left" if m["faceAOnLeft"] else "right"
            if j["choice"] == "tie":
                cell[0] += 0.5
                cell[1] += 0.5
            elif j["choice"] == a_side:
                cell[0] += 1.0
            else:
                cell[1] += 1.0

    meta_lookup = {}
    for _, meta_path in panels:
        for m in json.loads(Path(meta_path).read_text())["pairs"]:
            meta_lookup[(str(meta_path), m["pairId"])] = m

    out = []
    for key, (wa, wb) in votes.items():
        m = meta_lookup[key]
        a, b = m["faceAId"], m["faceBId"]
        if a not in score or b not in score:
            continue
        higher = wa if score[a] > score[b] else wb
        out.append((abs(score[a] - score[b]), higher, wa + wb))
    return out, score, gender


def fit_temperature(data) -> float:
    """Grid-search T in `p = sigmoid(gap / T)` on the vote-level Bernoulli log-likelihood."""
    def ll(t: float) -> float:
        s = 0.0
        for gap, higher, n in data:
            p = min(max(1 / (1 + math.exp(-gap / t)), 1e-9), 1 - 1e-9)
            s += higher * math.log(p) + (n - higher) * math.log(1 - p)
        return s
    return max(((ll(t / 200), t / 200) for t in range(20, 1600)))[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", default="artifacts/bt-refit-v2-qc/ratings.csv")
    ap.add_argument("--panel", nargs=2, action="append", required=True,
                    metavar=("RESULTS", "META"))
    ap.add_argument("--rejects", nargs="*", default=[])
    ap.add_argument("--out")
    args = ap.parse_args()

    metrics = Path(args.ratings).parent / "metrics.json"
    if metrics.exists() and "panelVotes" in json.loads(metrics.read_text()):
        raise SystemExit(
            f"refusing to calibrate on {args.ratings}: this refit absorbed panel votes, so the "
            f"curve would be measured on its own training data. Pass a ranking fitted before "
            f"the votes, e.g. artifacts/bt-refit-v2-qc/ratings.csv.")

    rejects: set[str] = set()
    for p in args.rejects:
        if Path(p).exists():
            rejects |= set(Path(p).read_text().split())

    data, score, gender = load_pairs(Path(args.ratings), [tuple(p) for p in args.panel], rejects)
    print(f"{len(data):,} pairs, {sum(n for _, _, n in data):,.0f} votes, "
          f"ranking {Path(args.ratings).parent.name}")

    t = fit_temperature(data)
    agree = lambda g: 1 / (1 + math.exp(-g / t))          # noqa: E731
    gap_for = lambda p: t * math.log(p / (1 - p))          # noqa: E731
    print(f"\nP(higher /10 score wins) = sigmoid(gap / T),  T = {t:.3f} on the /10 scale")

    print("\nwhat a /10 gap buys:")
    for g in GAPS:
        print(f"  {g:4.2f} points -> {agree(g):5.1%} of votes agree")

    print("\nband half-width for a stated agreement level — the honest resolution statement:")
    by_g: dict[str, list[float]] = defaultdict(list)
    for fid, s in score.items():
        by_g[gender[fid]].append(s)
    for ss in by_g.values():
        ss.sort()

    def crowdedness(w: float) -> str:
        """Share of each gender's cohort sitting within +/- w of its own median face."""
        parts = []
        for g in sorted(by_g):
            ss = by_g[g]
            med = ss[len(ss) // 2]
            n = bisect.bisect_right(ss, med + w) - bisect.bisect_left(ss, med - w)
            parts.append(f"{g[:1].upper()} {n / len(ss):.0%}")
        return " / ".join(parts)

    levels = []
    for p in AGREEMENT_LEVELS:
        g = gap_for(p)
        levels.append({"agreement": p, "gapPoints": g,
                       "shareOfCohortWithinBandOfMedian": crowdedness(g / 2)})
        print(f"  {p:5.1%} agree -> {g:4.2f} points apart; a +/-{g / 2:.2f} band around the "
              f"median contains {crowdedness(g / 2)} of the cohort")

    print("\nReading: a point score with no band asserts an ordering that fewer than 6 in 10 people\n"
          "would endorse unless the two faces are ~0.8 /10 apart. Publish the band, not the decimal.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "ranking": Path(args.ratings).parent.name,
            "pairs": len(data),
            "votes": sum(n for _, _, n in data),
            "temperatureOnTenScale": t,
            "agreementByGap": [{"gapPoints": g, "agreement": agree(g)} for g in GAPS],
            "bandForAgreement": levels,
        }, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
