#!/usr/bin/env python
"""How far apart must two faces be in the ranking before the ranking really orders them?

Written to answer a specific and reasonable complaint: *"a face sits next to faces that are
obviously in a different tier — is that a bad label or a bad algorithm?"*

Neither, usually. It is a **precision** question, and the data can answer it directly. Two
independent bounds:

1. **Estimation.** `bt_uncertainty.py` gives each face a resampled rank interval. If two faces'
   intervals overlap, the data does not order them — the point ranking still prints one above the
   other because a sort has to output *something*, and that printed order is the part that is
   over-precise, not the model.
2. **Human agreement.** `band_calibration.py` fits `P(agree) = sigmoid(/10 gap / T)`. Below some
   gap the crowd itself is a coin flip, so no amount of extra data would order those faces
   either — the ordering is not there to be found.

The output is one number for the product: **the rank separation at which a difference is real**.
Everything closer than that should be displayed as a band, not as a rank.

Usage:
    python scripts/rank_resolution.py \
        --uncertainty artifacts/bt-refit-v4-panel/uncertainty.csv \
        --band-calibration artifacts/panel-run-v4/band-calibration.json \
        --out artifacts/bt-refit-v4-panel/rank-resolution.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uncertainty", default="artifacts/bt-refit-v4-panel/uncertainty.csv")
    ap.add_argument("--band-calibration",
                    default="artifacts/panel-run-v4/band-calibration.json")
    ap.add_argument("--out")
    args = ap.parse_args()

    with open(args.uncertainty, newline="") as fh:
        rows = list(csv.DictReader(fh))
    by_gender: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_gender[r["gender"]].append(r)

    t = None
    if Path(args.band_calibration).exists():
        t = json.loads(Path(args.band_calibration).read_text())["temperatureOnTenScale"]

    out: dict = {"uncertainty": args.uncertainty, "temperatureOnTenScale": t, "genders": {}}
    for gender, faces in sorted(by_gender.items()):
        faces.sort(key=lambda r: -float(r["theta"]))
        n = len(faces)
        lo = [float(r["resampleRankLo"]) for r in faces]
        hi = [float(r["resampleRankHi"]) for r in faces]
        widths = sorted(hi[i] - lo[i] for i in range(n))
        eff = sorted(float(r["effectiveComparisons"]) for r in faces)

        # Rank 1 is the BEST rank, so face i sits above face j when i's worst plausible rank is
        # still better (numerically smaller) than j's best plausible rank: hi[i] < lo[j].
        # Getting this inequality backwards silently reports "0% distinguishable", which looks
        # like a dramatic finding rather than a sign error.
        seps = []
        for i in range(n):
            j = i + 1
            while j < n and hi[i] >= lo[j]:
                j += 1
            seps.append(j - i if j < n else float("nan"))
        clean = sorted(s for s in seps if not math.isnan(s))
        med_sep = clean[len(clean) // 2] if clean else float("nan")

        # Adjacent pairs whose intervals do NOT overlap — i.e. rank steps the data supports.
        adjacent_real = sum(1 for i in range(n - 1) if hi[i] < lo[i + 1]) / max(n - 1, 1)

        g = {
            "faces": n,
            "medianEffectiveComparisons": eff[len(eff) // 2],
            "medianRankIntervalWidth": widths[len(widths) // 2],
            "medianRankIntervalWidthPct": widths[len(widths) // 2] / n,
            "adjacentPairsDistinguishable": adjacent_real,
            "medianRanksApartToBeDistinguishable": med_sep,
        }
        out["genders"][gender] = g

        print(f"\n=== {gender} — {n:,} faces ===")
        print(f"  median effective comparisons per face      {g['medianEffectiveComparisons']:.1f}")
        print(f"  median 95% rank interval                   {g['medianRankIntervalWidth']:.0f} "
              f"places ({g['medianRankIntervalWidthPct']:.0%} of the list)")
        print(f"  adjacent pairs the data can tell apart     "
              f"{g['adjacentPairsDistinguishable']:.2%}")
        print(f"  ranks apart before a difference is real    "
              f"~{g['medianRanksApartToBeDistinguishable']:.0f} places")

    if t:
        # The other bound: the crowd's own resolution, independent of our sample size.
        print(f"\n=== the crowd's own limit (T = {t:.3f} on the /10 scale) ===")
        for p in (0.55, 0.60, 2 / 3, 0.75):
            gap = t * math.log(p / (1 - p))
            print(f"  {p:5.1%} of people agree only once two faces differ by {gap:4.2f} /10 points")
        out["crowdLimit"] = {f"{p:.3f}": t * math.log(p / (1 - p))
                             for p in (0.55, 0.60, 2 / 3, 0.75)}
        print("\nSo two things cap rank resolution, and they are independent: how well we measured\n"
              "each face (fixable with more comparisons, as 1/sqrt(k)) and how much people agree\n"
              "(not fixable at all). Neighbouring faces looking mis-ordered is the expected\n"
              "appearance of both — the sort prints a total order the evidence does not support.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
