#!/usr/bin/env python
"""Compare two comparator runs on the same human votes, with the pairing kept intact.

`eval_vs_panel.py` reports each run's agreement with human votes separately. Subtracting two
of those numbers and eyeballing the difference overstates the uncertainty badly, because the
two runs are scored on the *same* 1,928 pairs and agree with each other on most of them. The
honest question is not "could each accuracy have come out this way by chance" but "on the
pairs where the two runs disagree, does one of them win often enough to believe".

So this bootstraps over PAIRS, not votes. Votes are clustered inside a pair — twelve raters
looking at one pair are not twelve independent observations of the model's skill on that
pair — and resampling votes directly would shrink the interval by roughly sqrt(12) and
manufacture significance. Each resample draws pairs with replacement and recomputes the
vote-weighted accuracy of both runs on the drawn pairs.

Usage:
    python scripts/compare_panel_evals.py \
        --dump soft artifacts/train-v12-panel-soft/panel-pairs.csv \
        --dump hard artifacts/train-v13-panel-hard/panel-pairs.csv \
        --out artifacts/panel-arms-compare.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

BANDS = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 45), (45, 101)]
N_BOOT = 10_000
PREDICTOR = "neural comparator"


def load(path: str) -> dict[int, dict]:
    with open(path, newline="") as fh:
        return {int(r["pairIndex"]): r for r in csv.DictReader(fh)}


def acc(rows: list[dict], col: str) -> float:
    """Vote-weighted agreement: share of individual human votes matching the pick."""
    hit = tot = 0.0
    for r in rows:
        if not r.get(col):
            continue
        wa, wb = float(r["votesA"]), float(r["votesB"])
        hit += wa if int(r[col]) else wb
        tot += wa + wb
    return hit / tot if tot else float("nan")


def boot_diff(rows_a: list[dict], rows_b: list[dict], col: str, seed: int = 0):
    """Paired bootstrap over pairs -> (mean diff, lo, hi, share of resamples where a > b)."""
    rng = random.Random(seed)
    n = len(rows_a)
    diffs = []
    wins = 0
    for _ in range(N_BOOT):
        idx = [rng.randrange(n) for _ in range(n)]
        d = acc([rows_a[i] for i in idx], col) - acc([rows_b[i] for i in idx], col)
        diffs.append(d)
        wins += d > 0
    diffs.sort()
    return (
        sum(diffs) / len(diffs),
        diffs[int(0.025 * len(diffs))],
        diffs[int(0.975 * len(diffs))],
        wins / N_BOOT,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", nargs=2, action="append", required=True, metavar=("NAME", "CSV"))
    ap.add_argument("--baseline", default="export label (VLM)",
                    help="column present in every dump to compare each run against")
    ap.add_argument("--out")
    args = ap.parse_args()

    dumps = {name: load(path) for name, path in args.dump}
    names = list(dumps)
    shared = set.intersection(*(set(d) for d in dumps.values()))
    print(f"{len(shared):,} pairs scored by all {len(names)} runs")

    # One aligned row list per run, same pair order, so index i is the same pair everywhere.
    order = sorted(shared)
    aligned = {n: [dumps[n][i] for i in order] for n in names}
    ref = aligned[names[0]]
    votes = sum(float(r["votesA"]) + float(r["votesB"]) for r in ref)

    print(f"\n{'run':16} {'vs humans':>10} {'vs ' + args.baseline[:14]:>20}")
    result = {"pairs": len(order), "votes": votes, "runs": {}, "pairwise": [], "byGapBand": []}
    for n in names:
        a = acc(aligned[n], PREDICTOR)
        b = acc(aligned[n], args.baseline)
        print(f"{n:16} {a:9.2%} {a - b:+19.2f} pts")
        result["runs"][n] = {"accuracy": a, "baseline": b, "delta": a - b}

    # Each run against the shared baseline, paired.
    print(f"\npaired bootstrap vs {args.baseline} ({N_BOOT:,} resamples over pairs):")
    for n in names:
        rows = aligned[n]
        base = [dict(r, **{PREDICTOR: r[args.baseline]}) for r in rows]
        m, lo, hi, p = boot_diff(rows, base, PREDICTOR, seed=1)
        verdict = "significant" if lo > 0 else "not significant"
        print(f"  {n:14} {m * 100:+.2f} pts  95% CI [{lo * 100:+.2f}, {hi * 100:+.2f}]  "
              f"beats baseline in {p:.1%} of resamples  {verdict}")
        result["runs"][n]["vsBaseline"] = {"mean": m, "lo": lo, "hi": hi, "pWin": p}

    # Head-to-head between runs.
    if len(names) > 1:
        print("\nhead-to-head (paired, same pairs):")
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                m, lo, hi, p = boot_diff(aligned[a], aligned[b], PREDICTOR, seed=2)
                verdict = "separable" if lo > 0 or hi < 0 else "NOT separable — a tie"
                print(f"  {a} - {b}: {m * 100:+.2f} pts  95% CI "
                      f"[{lo * 100:+.2f}, {hi * 100:+.2f}]  {verdict}")
                result["pairwise"].append({"a": a, "b": b, "mean": m, "lo": lo, "hi": hi,
                                           "pWin": p, "separable": bool(lo > 0 or hi < 0)})

    # Where the money was made: agreement by how far apart the ranking puts the two faces.
    banded = [r for r in ref if r["gapPct"]]
    if banded:
        print(f"\nby percentile gap ({args.baseline} in brackets):")
        print(f"{'gap':>8} {'pairs':>6} " + " ".join(f"{n[:12]:>13}" for n in names)
              + f" {'baseline':>10}")
        for lo_b, hi_b in BANDS:
            sel = [k for k, i in enumerate(order)
                   if ref[k]["gapPct"] and lo_b <= float(ref[k]["gapPct"]) < hi_b]
            if not sel:
                continue
            row = {"band": f"{lo_b}-{min(hi_b, 100)}", "pairs": len(sel)}
            line = f"{row['band']:>8} {len(sel):6} "
            for n in names:
                v = acc([aligned[n][k] for k in sel], PREDICTOR)
                row[n] = v
                line += f"{v:12.1%} "
            bl = acc([ref[k] for k in sel], args.baseline)
            row["baseline"] = bl
            result["byGapBand"].append(row)
            print(line + f"{bl:9.1%}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
