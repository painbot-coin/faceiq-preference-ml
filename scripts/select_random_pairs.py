#!/usr/bin/env python
"""Draw pairs **uniformly at random** — the study that measures the product, not the blind spot.

Every panel study so far deliberately selected near-ties, because that is where the VLM label
is worthless and a human vote buys the most (log §5.7). That was the right call for *improving*
the ranking and it is the wrong sample for *describing* it. Three consequences we cannot fix
with the data we own:

1. **No population accuracy.** Every number in §5.5 is measured on hand-picked hard pairs, so
   the programme's headline "56.9% vs a 59.1% ceiling" is worst-case by construction and says
   nothing about the pair a user actually generates.
2. **No honest calibration.** `bt-refit-v4-panel` absorbed all 7,780 pairs we own, so testing
   its /10 curve against them is testing a model on its training data (§5.6).
3. **Two unresolved spend bands.** Headroom above a 20-point percentile gap is +0.1 [−2.5,
   +2.5] and −1.1 [−2.9, +0.6] on only 389 and 262 pairs, and those pairs are a *biased*
   subsample of their bands. That uncertainty gates ~$15.3k of further labelling.

A uniform draw fixes all three with one study, because ~66% of it lands in the two bands we
know least about while the rest spreads across the range.

**Uniformity is the product here, so this script deliberately refuses to stratify.** No gap
filter, and no ethnicity floor by default — a floor swaps pairs to hit per-group counts, which
is exactly the selection effect that would make the resulting accuracy unquotable. Gender is
balanced only because the eligible pool is already 49.1/50.9, so forcing 50/50 changes nothing
measurable and keeps the two per-gender rankings symmetric.

Note the draw is uniform over **unbought** pairs, which is required for purpose 2. Since past
draws took mostly close pairs, the unbought pool is wider than the export as a whole (0–2 is
2.6% of it vs 5.6% of the export). `percentileGap` and `band` are recorded per pair in
sample-meta.json so a population estimate can be reweighted to true export shares at analysis
time; the printed table shows both.

Output schema is identical to `select_panel_pairs.py` / `select_topup_pairs.py`, so the rating
app, `analyze_panel_run.py` and `refit_bt_panel.py` consume it unchanged. `stratum` is set to
the headroom band, which makes `analyze_panel_run.py` report per-band accuracy and human
ceiling for free — that is the headroom table this study exists to produce.

Usage:
    python scripts/select_random_pairs.py --export data/exports/<runId> \
        --n 2500 \
        --exclude-pairs labels/panel-pilot/pairs.json labels/panel-run-3/pairs.json \
        --out labels/panel-run-4-random
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from faceiq_pref.data import load_export  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from select_panel_pairs import (  # noqa: E402
    load_face_ids,
    load_primary_ethnicity,
    sample_gender_balanced,
)
from select_topup_pairs import (  # noqa: E402
    already_bought,
    carry_golds,
    load_ratings,
)

BANDS = [(0.0, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, 45.0), (45.0, 1e9)]
# Measured headroom per band: what a human vote adds over the free VLM label (log §5.7).
HEADROOM = {"0-2": 6.2, "2-5": 5.9, "5-10": 3.5, "10-20": 3.8, "20-45": 0.1, "45-100": -1.1}
PROLIFIC_MIN_REPRESENTATIVE = 300  # representative samples will not run below this


def band_of(gap: float) -> str:
    for lo, hi in BANDS:
        if lo <= gap < hi:
            return f"{lo:g}-{min(hi, 100.0):g}"
    return "45-100"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--ratings", default="artifacts/bt-refit-v2-qc/ratings.csv",
                    help="defines the percentile bands. Must be a refit that never saw panel "
                         "votes, or the bands encode the crowd's own corrections (log §5.7)")
    ap.add_argument("--allow-panel-ranking", action="store_true")
    ap.add_argument("--n", type=int, default=2500,
                    help="pairs to draw. 2,500 at 12 votes each fits one 300-rater "
                         "representative study, Prolific's minimum")
    ap.add_argument("--exclude-pairs", nargs="*",
                    default=["labels/panel-pilot/pairs.json", "labels/panel-run-3/pairs.json"],
                    help="previous draws — excluded so the calibration test is out-of-sample")
    ap.add_argument("--exclude-faces", default="artifacts/face-qc-v1/exclude-faces.csv")
    ap.add_argument("--exclude-genders", default="artifacts/face-qc-v1/gender-fixes.csv")
    ap.add_argument("--attributes", default="labels/face-attributes.json")
    ap.add_argument("--example-pairs", default="labels/panel-pilot/example-pairs.json")
    ap.add_argument("--golds", default="labels/panel-pilot/golds.json")
    ap.add_argument("--gold-pairs", default="labels/panel-pilot/pairs.json")
    ap.add_argument("--no-gender-balance", action="store_true",
                    help="draw purely at random instead of forcing a 50/50 gender split")
    ap.add_argument("--out", default="labels/panel-run-4-random")
    ap.add_argument("--seed", type=int, default=20260802)
    ap.add_argument("--pairs-per-session", type=int, default=100)
    ap.add_argument("--reward", type=float, default=3.00)
    ap.add_argument("--fee", type=float, default=0.4286)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    export = load_export(args.export)
    faces = export.faces()
    matchups = export.all_matchups()
    thetas, pcts = load_ratings(Path(args.ratings))
    print(f"export OK: run {export.run_id}, {len(matchups)} matchups, {len(thetas)} ranked faces")

    metrics = Path(args.ratings).parent / "metrics.json"
    if (metrics.exists() and "panelVotes" in json.loads(metrics.read_text())
            and not args.allow_panel_ranking):
        raise SystemExit(
            f"refusing to band on {args.ratings}: this refit was fitted on panel votes, so its "
            f"percentile gaps already encode the crowd's corrections and every band boundary is "
            f"circular (log §5.7). Pass artifacts/bt-refit-v2-qc/ratings.csv."
        )

    bad = load_face_ids(Path(args.exclude_faces)) | load_face_ids(Path(args.exclude_genders))
    ethnicity = load_primary_ethnicity(Path(args.attributes))
    prior = already_bought(args.exclude_pairs)
    examples: set[str] = set()
    if Path(args.example_pairs).exists():
        examples = set(json.loads(Path(args.example_pairs).read_text())["pairIds"])
    print(f"excluding {len(prior)} already-bought pairs, {len(examples)} worked examples, "
          f"{len(bad)} QC faces")

    pool, gap_of, skipped = [], {}, Counter()
    export_bands = Counter()  # every ranked pair, for the reweighting reference
    for m in matchups:
        pa, pb = pcts.get(m.face_a_id), pcts.get(m.face_b_id)
        if pa is not None and pb is not None:
            export_bands[band_of(abs(pa - pb) * 100)] += 1
        if m.comparison_id in prior:
            skipped["already bought"] += 1
            continue
        if m.comparison_id in examples:
            skipped["worked example"] += 1
            continue
        if m.face_a_id in bad or m.face_b_id in bad:
            skipped["QC-excluded face"] += 1
            continue
        if pa is None or pb is None:
            skipped["face not ranked"] += 1
            continue
        gap_of[m.comparison_id] = abs(pa - pb) * 100
        pool.append(m)

    print(f"skipped: {dict(skipped)}")
    print(f"eligible pool (NO gap filter — this draw is uniform): {len(pool):,} pairs")
    if len(pool) < args.n:
        print(f"  WARNING: only {len(pool)} available for --n {args.n}; taking all")
    n = min(args.n, len(pool))

    if args.no_gender_balance:
        selected = rng.sample(pool, n)
    else:
        selected = sample_gender_balanced(pool, n, rng)

    # Uniformity check: the draw's band mix should track the pool's, and both are reported
    # against the whole export so a population estimate can be reweighted later.
    pool_bands = Counter(band_of(gap_of[m.comparison_id]) for m in pool)
    draw_bands = Counter(band_of(gap_of[m.comparison_id]) for m in selected)
    tot_export = sum(export_bands.values())
    print(f"\nband mix — draw vs the pool it came from vs the whole export ({n} pairs):")
    print(f"{'band':>8} {'drawn':>6} {'draw %':>8} {'pool %':>8} {'export %':>9} "
          f"{'headroom':>9}  purpose")
    for lo, hi in BANDS:
        b = f"{lo:g}-{min(hi, 100.0):g}"
        if not pool_bands[b]:
            continue
        why = ("resolves the open spend question" if HEADROOM[b] <= 1.0
               else "confirms the known buy zone")
        print(f"{b:>8} {draw_bands[b]:6,} {draw_bands[b] / n:8.1%} "
              f"{pool_bands[b] / len(pool):8.1%} {export_bands[b] / tot_export:9.1%} "
              f"{HEADROOM[b]:+8.1f}pt  {why}")
    open_bands = sum(draw_bands[b] for b in draw_bands if HEADROOM[b] <= 1.0)
    print(f"\n{open_bands:,} of {n:,} pairs ({open_bands / n:.0%}) land in the two bands whose "
          f"headroom is unresolved —\nthe cells that gate ~$15.3k of further labelling.")

    rng.shuffle(selected)
    pairs_out, meta_out = [], []
    for m in selected:
        a_left = rng.random() < 0.5
        left_id, right_id = (m.face_a_id, m.face_b_id) if a_left else (m.face_b_id, m.face_a_id)
        pairs_out.append({
            "pairId": m.comparison_id,
            "gender": m.gender,
            "left": {"faceId": left_id, "photoUrl": faces[left_id].photo_url},
            "right": {"faceId": right_id, "photoUrl": faces[right_id].photo_url},
        })
        ta, tb = thetas[m.face_a_id], thetas[m.face_b_id]
        gap = gap_of[m.comparison_id]
        b = band_of(gap)
        meta_out.append({
            "pairId": m.comparison_id,
            "pairIndex": m.pair_index,
            # stratum = band, so analyze_panel_run.py emits the per-band accuracy and human
            # ceiling this study exists to measure, with no extra code.
            "stratum": f"random-{b}",
            "bucket": b,
            "band": b,
            "winProb": round(1.0 / (1.0 + math.exp(-abs(ta - tb))), 6),
            "percentileGap": round(gap, 4),
            "faceAId": m.face_a_id,
            "faceBId": m.face_b_id,
            "thetaA": ta,
            "thetaB": tb,
            "vlmOutcome": m.vlm_outcome,
            "vlmConfidence": m.confidence,
            "humanLabeled": m.human_labeled_at is not None,
            "inPilot500": False,
            "isGoldCandidate": False,
            "faceAOnLeft": a_left,
            "ethnicityA": ethnicity.get(m.face_a_id),
            "ethnicityB": ethnicity.get(m.face_b_id),
        })

    golds = carry_golds(Path(args.golds), Path(args.gold_pairs))
    by_cid = {m.comparison_id: m for m in matchups}
    carried = 0
    for gid, (left_id, right_id) in golds.items():
        m = by_cid.get(gid)
        if m is None:
            print(f"  warning: gold {gid} not in this export, skipped")
            continue
        pairs_out.append({
            "pairId": gid,
            "gender": m.gender,
            "left": {"faceId": left_id, "photoUrl": faces[left_id].photo_url},
            "right": {"faceId": right_id, "photoUrl": faces[right_id].photo_url},
        })
        ta, tb = thetas.get(m.face_a_id), thetas.get(m.face_b_id)
        meta_out.append({
            "pairId": gid,
            "pairIndex": m.pair_index,
            "stratum": "gold",
            "bucket": "very_clear",
            "band": None,
            "winProb": round(1.0 / (1.0 + math.exp(-abs(ta - tb))), 6)
            if ta is not None and tb is not None else None,
            "faceAId": m.face_a_id,
            "faceBId": m.face_b_id,
            "thetaA": ta,
            "thetaB": tb,
            "vlmOutcome": m.vlm_outcome,
            "vlmConfidence": m.confidence,
            "humanLabeled": m.human_labeled_at is not None,
            "inPilot500": False,
            "isGoldCandidate": True,
            "faceAOnLeft": left_id == m.face_a_id,
            "ethnicityA": ethnicity.get(m.face_a_id),
            "ethnicityB": ethnicity.get(m.face_b_id),
        })
        carried += 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat()
    (out_dir / "pairs.json").write_text(json.dumps({
        "version": 1, "runId": export.run_id, "seed": args.seed, "createdAt": stamp,
        "pairCount": len(pairs_out), "pairs": pairs_out}, indent=1) + "\n")
    (out_dir / "sample-meta.json").write_text(json.dumps({
        "version": 1, "runId": export.run_id, "seed": args.seed, "ratings": args.ratings,
        "design": "uniform random over unbought pairs; no gap filter, no ethnicity floor",
        "maxPercentileGap": None, "createdAt": stamp,
        "excludedPriorPairs": len(prior),
        "exportBandShares": {k: v / tot_export for k, v in sorted(export_bands.items())},
        "drawBandCounts": dict(sorted(draw_bands.items())),
        "ethnicityCoverage": {}, "pairs": meta_out}, indent=1) + "\n")

    print(f"\nwrote {out_dir / 'pairs.json'} ({len(pairs_out)} pairs = "
          f"{len(selected)} new + {carried} carried golds, blinded)")
    print(f"wrote {out_dir / 'sample-meta.json'} (answer key — gitignored)")
    print(f"gender: {dict(Counter(m.gender for m in selected))}")
    gaps = sorted(gap_of[m.comparison_id] for m in selected)
    print(f"percentile gap: median {gaps[len(gaps) // 2]:.1f}, "
          f"min {gaps[0]:.1f}, max {gaps[-1]:.1f}  (unfiltered, as intended)")
    faces_touched = {f for m in selected for f in (m.face_a_id, m.face_b_id)}
    print(f"faces touched: {len(faces_touched)} "
          f"({len(selected) * 2 / len(faces_touched):.1f} comparisons per face)")

    # Study sizing. Representative samples cannot run under 300 participants, which floors the
    # vote budget: at 100 pairs each that is 30,000 votes whether or not we need them, so the
    # lever is how many pairs to spread them over, not how much to spend.
    raters = max(PROLIFIC_MIN_REPRESENTATIVE,
                 math.ceil(n * 12 / args.pairs_per_session))
    votes = raters * args.pairs_per_session
    reward = raters * args.reward
    total = reward * (1 + args.fee)
    print("\n=== Prolific study parameters ===")
    print(f"  participants          {raters}  (representative-sample minimum is "
          f"{PROLIFIC_MIN_REPRESENTATIVE})")
    print(f"  pairs per session     {args.pairs_per_session}   (PAIRS_PER_SESSION)")
    print(f"  total votes           {votes:,}")
    print(f"  votes per pair        {votes / n:.1f}")
    print(f"  reward per submission ${args.reward:.2f}")
    print(f"  total cost            ${total:,.2f}  (${reward:,.2f} + {args.fee:.2%} fee) "
          f"= ${total / votes:.4f}/vote")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
