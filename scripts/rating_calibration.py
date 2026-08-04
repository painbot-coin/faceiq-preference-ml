#!/usr/bin/env python
"""Is the /10 score calibrated? Does a 7.0 beat a 5.0 as often as we imply?

Pairwise accuracy is a diagnostic; the shipped product is a NUMBER on a face, and a
number makes a promise. If we tell one person 7.0 and another 5.0, we are implying the
first is preferred by a clear majority. This measures whether that holds, using the only
data that can settle it — the panel's votes.

Two readings, both from the same pairs:

1. **By /10 gap** — the product-facing one. For pairs whose scores differ by 0-0.5, 0.5-1,
   1-2, 2+ points, what share of human votes went to the higher-scored face? This is what
   sets the honest BAND WIDTH: the smallest gap at which the ordering is reliable.
2. **Reliability diagram** — bin pairs by the win probability BT implies,
   `sigmoid(beta * (theta_i - theta_j))`, and compare to the observed vote share. A curve
   below the diagonal means the ranking is overconfident.

Circularity matters here. `bt-refit-v4-panel` absorbed these votes while fitting, so its
implied probabilities are scored on their own training data; pass a VLM-only refit such as
`bt-refit-v2-qc` for the honest number. Both are worth printing, and the gap between them
is the size of the correction the panel bought.

Usage:
    python scripts/rating_calibration.py \
        --export data/exports/cmr1mr0m7000196d57zi3vcgn \
        --panel labels/panel-pilot/results labels/panel-pilot/sample-meta.json \
        --panel labels/panel-run-3/results labels/panel-run-3/sample-meta.json \
        --rejects artifacts/panel-run-v1/reject-pids.txt \
                  artifacts/panel-run-v3/reject-pids.txt \
        --ratings artifacts/bt-refit-v2-qc/ratings.csv \
        --ratings artifacts/bt-refit-v4-panel/ratings.csv \
        --out artifacts/bt-refit-v4-panel/calibration.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from faceiq_pref.data import load_export  # noqa: E402
from faceiq_pref.panel import load_panel_votes  # noqa: E402

SCORE_GAPS = [(0.0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 99.0)]
PROB_BINS = [(0.5, 0.55), (0.55, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]


def wilson(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    """Wilson interval — behaves sensibly near 0 and 1, unlike the normal approximation."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (centre - half, centre + half)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--panel", nargs=2, action="append", required=True,
                    metavar=("RESULTS", "META"))
    ap.add_argument("--rejects", nargs="*", default=[])
    ap.add_argument("--ratings", action="append", required=True)
    ap.add_argument("--extra-beta", type=float,
                    help="also score each NON-circular ranking with this discrimination. "
                         "Passing the panel's measured beta_human gives an honest "
                         "calibration test: one scalar fit on 65k votes leaks far less "
                         "than per-face theta refit on the same votes.")
    ap.add_argument("--exclude-faces", default="artifacts/face-qc-v1/exclude-faces.csv")
    ap.add_argument("--exclude-genders", default="artifacts/face-qc-v1/gender-fixes.csv")
    ap.add_argument("--out")
    args = ap.parse_args()

    export = load_export(args.export)
    matchups = export.all_matchups()
    dropped: set[str] = set()
    for p in (args.exclude_faces, args.exclude_genders):
        if p and Path(p).exists():
            with open(p, newline="") as fh:
                dropped.update(r["faceId"] for r in csv.DictReader(fh))
    by_index = {m.pair_index: m for m in matchups
                if m.face_a_id not in dropped and m.face_b_id not in dropped}

    rejects: set[str] = set()
    for p in args.rejects:
        if Path(p).exists():
            rejects |= set(Path(p).read_text().split())
    votes = load_panel_votes([list(p) for p in args.panel], rejects)

    report: dict = {"panelPairs": len(votes), "rankings": {}}

    for rp in args.ratings:
        name = Path(rp).parent.name
        with open(rp, newline="") as fh:
            rows = list(csv.DictReader(fh))
        theta = {r["faceId"]: float(r["theta"]) for r in rows}
        score = {r["faceId"]: float(r["scoreOutOf10"]) for r in rows}
        mp = Path(rp).parent / "metrics.json"
        meta = json.loads(mp.read_text()) if mp.exists() else {}
        circular = "panelVotes" in meta
        beta_h = None
        for g in (meta.get("genders") or {}).values():
            if g.get("betaHuman"):
                beta_h = float(g["betaHuman"])
                break

        keys = [i for i in votes if i in by_index
                and by_index[i].face_a_id in theta and by_index[i].face_b_id in theta]
        print("\n" + "=" * 78)
        print(f"{name}  ({len(keys):,} panel pairs scored"
              + (", CIRCULAR — fit on these votes" if circular else "") + ")")
        if beta_h:
            print(f"  beta_human = {beta_h:.4f}: humans are {1 / beta_h:.1f}x flatter than "
                  f"the VLM scale theta is expressed in")

        # ---- 1. by /10 gap: the product-facing band width -------------------
        print(f"\n  {'/10 gap':>10} {'pairs':>6} {'votes':>7} "
              f"{'higher score wins':>18}  95% CI")
        gap_rows = []
        for lo, hi in SCORE_GAPS:
            hits = tot = 0.0
            n_pairs = 0
            for i in keys:
                m = by_index[i]
                sa, sb = score[m.face_a_id], score[m.face_b_id]
                if not (lo <= abs(sa - sb) < hi):
                    continue
                wa, wb = votes[i]
                hits += wa if sa > sb else wb
                tot += wa + wb
                n_pairs += 1
            if not tot:
                continue
            lo_ci, hi_ci = wilson(hits, tot)
            label = f"{lo:g}-{hi:g}" if hi < 90 else f"{lo:g}+"
            gap_rows.append({"gap": label, "pairs": n_pairs, "votes": tot,
                             "higherScoreWinShare": hits / tot,
                             "ci": [lo_ci, hi_ci]})
            print(f"  {label:>10} {n_pairs:6,} {tot:7,.0f} "
                  f"{hits / tot:17.1%}  [{lo_ci:.1%}, {hi_ci:.1%}]")

        # ---- 2. reliability diagram ------------------------------------------
        def reliability(beta: float) -> list[dict]:
            print(f"\n  reliability (implied by sigmoid({beta:.3f} * theta gap)):")
            print(f"  {'implied':>12} {'pairs':>6} {'observed':>9} {'gap':>8}")
            out = []
            for lo, hi in PROB_BINS:
                hits = tot = implied_sum = 0.0
                n_pairs = 0
                for i in keys:
                    m = by_index[i]
                    d = theta[m.face_a_id] - theta[m.face_b_id]
                    p_hi = 1.0 / (1.0 + np.exp(-beta * abs(d)))
                    if not (lo <= p_hi < hi):
                        continue
                    wa, wb = votes[i]
                    hits += wa if d > 0 else wb
                    tot += wa + wb
                    implied_sum += p_hi * (wa + wb)
                    n_pairs += 1
                if not tot:
                    continue
                implied, observed = implied_sum / tot, hits / tot
                out.append({"bin": f"{lo:g}-{hi:g}", "pairs": n_pairs, "votes": tot,
                            "implied": implied, "observed": observed,
                            "gap": observed - implied})
                print(f"  {f'{lo:g}-{hi:g}':>12} {n_pairs:6,} {observed:9.1%} "
                      f"{observed - implied:+8.1%}")
            if out:
                worst = min(out, key=lambda r: r["gap"])
                if worst["gap"] < -0.05:
                    print(f"  OVERCONFIDENT: the {worst['bin']} bin implies "
                          f"{worst['implied']:.1%} but people vote "
                          f"{worst['observed']:.1%} ({worst['gap']:+.1%}).")
            return out

        rel_rows = reliability(beta_h if beta_h else 1.0)
        entry = {"circular": circular, "betaHuman": beta_h, "pairsScored": len(keys),
                 "byScoreGap": gap_rows, "reliability": rel_rows}

        # An honest calibration test for a VLM-only ranking: keep its theta, borrow only
        # the panel's temperature. Fitting one scalar on 65k votes is a far smaller loan
        # from the test set than refitting 2,866 per-face thetas on them.
        if args.extra_beta and not circular:
            print(f"\n  --- same ranking, temperature-corrected to "
                  f"beta={args.extra_beta:.4f} ---")
            entry["reliabilityAtExtraBeta"] = reliability(args.extra_beta)
            entry["extraBeta"] = args.extra_beta

        report["rankings"][name] = entry

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
