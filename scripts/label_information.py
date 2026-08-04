#!/usr/bin/env python
"""How much information does each label source actually carry, and can we buy it cheaper?

Three questions this repo kept answering by assertion, answered here with numbers.

1. **Is a single score "faulty by nature" when the crowd only agrees 59%?**
   Binary accuracy asks "did you name the winner", which is close to unanswerable on a
   near-tie and so bottoms out near the crowd's own self-agreement. But a score that says
   6.00 vs 6.05 is making a *quantitative* claim — that the crowd will split roughly evenly
   — and that claim is checkable. So we correlate the model's score gap against the crowd's
   observed vote share, and compare it to the ceiling implied by how noisy an 8-vote share
   is in the first place (split-half, Spearman-Brown corrected). A model near that ceiling
   is recovering essentially all the margin signal the data can express, whatever its
   binary accuracy says.

2. **Can Gemini's own confidence tell us which pairs to send to humans?**
   If high-confidence pairs are the ones the crowd finds easy, we only need to buy panel
   votes on the medium/low tail, which is 15% of the export instead of all of it. If
   confidence is flat against crowd decisiveness, it is not a router and the selection has
   to come from the ranking's percentile gap instead.

3. **What would each labelling strategy cost?** Priced from the observed panel rate, per
   confidence x decile-gap cell, so "panel-label everything" stops being a vibe.

Usage:
    python scripts/label_information.py --export data/exports/<runId> \
        --scores artifacts/train-v15-panel-weighted/model_scores.csv \
        --panel labels/panel-pilot/results labels/panel-pilot/sample-meta.json \
        --panel labels/panel-run-3/results labels/panel-run-3/sample-meta.json \
        --rejects artifacts/panel-run-v1/reject-pids.txt \
                  artifacts/panel-run-v3/reject-pids.txt \
        --out artifacts/label-information-v1/report.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from faceiq_pref.data import load_export  # noqa: E402
from faceiq_pref.panel import load_panel_votes, load_rejects  # noqa: E402

# Observed cost of the two runs actually purchased: $3,186 for 65,894 usable votes.
COST_PER_VOTE = 3186.0 / 65894.0


def split_half_reliability(
    raw: dict[int, tuple[float, float]], keys: list[int], reps: int = 200, seed: int = 0
) -> tuple[float, float]:
    """Reliability of the observed vote share, and the correlation ceiling it implies.

    An 8-vote share is a noisy estimate of the crowd's true split, so even a perfect
    predictor cannot correlate 1.0 with it. Splitting each pair's votes into halves and
    correlating the two halves measures how much of the observed variance is real; the
    Spearman-Brown step scales that from half-samples up to the full vote count, and the
    square root converts a reliability into the maximum correlation an external predictor
    can reach against the noisy observation.
    """
    rng = random.Random(seed)
    rs = []
    for _ in range(reps):
        left, right = [], []
        for idx in keys:
            wa, wb = raw[idx]
            # Rebuild individual votes so halves are drawn at vote level, not share level.
            ballots = [1.0] * int(wa) + [0.0] * int(wb)
            if wa % 1:  # a tie contributed 0.5 to each side
                ballots.append(0.5)
            if wb % 1:
                ballots.append(0.5)
            if len(ballots) < 4:
                continue
            rng.shuffle(ballots)
            mid = len(ballots) // 2
            a, b = ballots[:mid], ballots[mid:]
            left.append(sum(a) / len(a))
            right.append(sum(b) / len(b))
        if len(left) > 10:
            r = spearmanr(left, right).statistic
            if r == r:
                rs.append(r)
    if not rs:
        return float("nan"), float("nan")
    r_half = sum(rs) / len(rs)
    r_full = 2 * r_half / (1 + r_half) if r_half > -1 else float("nan")  # Spearman-Brown
    return r_full, (r_full**0.5 if r_full > 0 else float("nan"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--scores", action="append", default=[],
                    help="repeatable: model_scores.csv from a run (faceId,modelScore,theta)")
    ap.add_argument("--panel", nargs=2, action="append", required=True,
                    metavar=("RESULTS", "META"))
    ap.add_argument("--rejects", nargs="*", default=[])
    ap.add_argument("--ratings", default="artifacts/bt-refit-v4-panel/ratings.csv",
                    help="ranking that defines percentile-gap bands and Labs scores")
    ap.add_argument("--extra-ratings", action="append", default=[],
                    help="repeatable: other ratings.csv to score for margin recovery")
    ap.add_argument("--exclude-faces", default="artifacts/face-qc-v1/exclude-faces.csv")
    ap.add_argument("--exclude-genders", default="artifacts/face-qc-v1/gender-fixes.csv")
    ap.add_argument("--min-votes", type=int, default=4)
    ap.add_argument("--band-reps", type=int, default=2000,
                    help="bootstrap replicates for the per-band headroom interval")
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

    raw = load_panel_votes([tuple(p) for p in args.panel], load_rejects(args.rejects))
    keys = [i for i in raw if i in by_index and sum(raw[i]) >= args.min_votes]
    print(f"export OK: run {export.run_id} — {len(matchups):,} matchups")
    print(f"panel pairs with >={args.min_votes} votes: {len(keys):,} "
          f"({sum(sum(raw[i]) for i in keys):,.0f} votes)")

    report: dict = {"panelPairs": len(keys),
                    "panelVotes": float(sum(sum(raw[i]) for i in keys))}

    # Panel vote shares were applied as TRAIN targets in v12+, so a model scored on a pair it
    # trained on is reciting a target, not predicting one. Every predictor is therefore scored
    # on the intersection of the runs' validation splits — rebuilt from each run's stored
    # val_fraction/split_seed exactly as split_by_face_id did it — so the table is comparable
    # and no row is quoting its own training data.
    # Sections 2 and 3 study the VLM and the ranking, neither of which trained on these
    # votes, so they keep all `keys`. Only section 1 narrows.
    face_ids = sorted({f for m in matchups for f in (m.face_a_id, m.face_b_id)})
    lf_keys = list(keys)
    for sp in args.scores:
        mp = Path(sp).parent / "metrics.json"
        if not mp.exists():
            print(f"  ! {sp}: no metrics.json, cannot verify its split — skipping")
            continue
        rc = json.loads(mp.read_text()).get("config", {})
        shuffled = face_ids[:]
        random.Random(rc.get("split_seed", 42)).shuffle(shuffled)
        val = set(shuffled[: int(len(shuffled) * rc.get("val_fraction", 0.2))])
        before = len(lf_keys)
        lf_keys = [i for i in lf_keys
                   if by_index[i].face_a_id in val and by_index[i].face_b_id in val]
        print(f"  leak-free filter from {Path(sp).parent.name} "
              f"(val_fraction {rc.get('val_fraction')}): {before:,} -> {len(lf_keys):,} pairs")
    if not lf_keys:
        raise SystemExit("No leak-free panel pairs across the requested runs. Score runs "
                         "trained at a larger val_fraction, or drop --scores.")
    report["leakFreePairs"] = len(lf_keys)
    report["leakFreeVotes"] = float(sum(sum(raw[i]) for i in lf_keys))

    # ---------------------------------------------------------------- 1. margin recovery
    rel, ceiling = split_half_reliability(raw, lf_keys)
    print("\n=== 1. Does a single score capture the MARGIN, not just the winner? ===")
    print(f"vote-share reliability (Spearman-Brown, full sample): {rel:.3f}")
    print(f"  => max correlation any predictor can reach vs the observed share: {ceiling:.3f}")
    print(f"\n{'predictor':34} {'r(score gap, vote share)':>24} {'% of ceiling':>13} {'pairs':>7}")

    shares = {i: raw[i][0] / (raw[i][0] + raw[i][1]) for i in keys}
    margin_rows = []

    def margin_of(name: str, gap: dict[int, float]) -> None:
        sel = [i for i in lf_keys if i in gap]
        if len(sel) < 30:
            return
        r = spearmanr([gap[i] for i in sel], [shares[i] for i in sel]).statistic
        frac = r / ceiling if ceiling == ceiling and ceiling else float("nan")
        margin_rows.append({"predictor": name, "spearman": r, "fracOfCeiling": frac,
                            "pairs": len(sel)})
        print(f"{name:34} {r:24.3f} {frac:12.1%} {len(sel):7,}")

    for sp in args.scores:
        if not Path(sp).exists():
            continue
        with open(sp, newline="") as fh:
            rows = list(csv.DictReader(fh))
        ms = {r["faceId"]: float(r["modelScore"]) for r in rows if r.get("modelScore")}
        margin_of(f"NN {Path(sp).parent.name}", {
            i: ms[by_index[i].face_a_id] - ms[by_index[i].face_b_id]
            for i in lf_keys if by_index[i].face_a_id in ms and by_index[i].face_b_id in ms})

    # A refit that absorbed these votes is scoring its own training data, so it is an upper
    # reference, not a result. The VLM-only refit is the honest ranking baseline.
    for rp in args.extra_ratings + [args.ratings]:
        if not Path(rp).exists():
            continue
        with open(rp, newline="") as fh:
            rows_r = list(csv.DictReader(fh))
        name = Path(rp).parent.name
        mp = Path(rp).parent / "metrics.json"
        circ = mp.exists() and "panelVotes" in json.loads(mp.read_text())
        th = {r["faceId"]: float(r["theta"]) for r in rows_r}
        margin_of(f"BT {name}{' (CIRCULAR)' if circ else ''}", {
            i: th[by_index[i].face_a_id] - th[by_index[i].face_b_id]
            for i in lf_keys if by_index[i].face_a_id in th and by_index[i].face_b_id in th})

    if Path(args.ratings).exists():
        with open(args.ratings, newline="") as fh:
            rr = list(csv.DictReader(fh))
        labs = {r["faceId"]: float(r["labsOverallScore"]) for r in rr
                if r.get("labsOverallScore")}
        margin_of("Labs overall_score", {
            i: labs[by_index[i].face_a_id] - labs[by_index[i].face_b_id]
            for i in lf_keys if by_index[i].face_a_id in labs and by_index[i].face_b_id in labs})

    # The VLM has only a winner and a coarse tier, so its best possible "margin" is
    # +/-1 scaled by confidence. Included to show what a binary label can and cannot do.
    tier = {"high": 1.0, "medium": 0.5, "low": 0.25}
    margin_of("VLM label x confidence tier", {
        i: (1 if by_index[i].final_outcome == "A" else -1)
           * tier.get(by_index[i].confidence or "", 0.5)
        for i in lf_keys if by_index[i].final_outcome in ("A", "B")})
    report["marginRecovery"] = {"voteShareReliability": rel, "correlationCeiling": ceiling,
                               "predictors": margin_rows}

    # -------------------------------------------------- 2. is VLM confidence a router?
    print("\n=== 2. Can Gemini's confidence route pairs to humans? ===")
    print("If confidence tracked crowd difficulty, 'high' pairs would be decisive and the")
    print("VLM would be right on them. decisiveness = mean |vote share - 0.5| (0 = dead")
    print("split, 0.5 = unanimous).")
    print(f"\n{'confidence':10} {'pairs':>6} {'decisiveness':>13} {'VLM agrees w/ votes':>20}")
    conf_rows = []
    for tname in ("high", "medium", "low"):
        sel = [i for i in keys if (by_index[i].confidence or "") == tname]
        if not sel:
            continue
        dec = sum(abs(shares[i] - 0.5) for i in sel) / len(sel)
        hit = tot = 0.0
        for i in sel:
            wa, wb = raw[i]
            if by_index[i].final_outcome in ("A", "B"):
                hit += wa if by_index[i].final_outcome == "A" else wb
                tot += wa + wb
        acc = hit / tot if tot else float("nan")
        conf_rows.append({"confidence": tname, "pairs": len(sel), "decisiveness": dec,
                          "vlmAgreement": acc})
        print(f"{tname:10} {len(sel):6,} {dec:13.3f} {acc:19.1%}")
    report["byConfidence"] = conf_rows

    # The percentile gap is the alternative router. Same table, banded by gap, plus the one
    # column that decides whether buying a human vote in a band is worth anything: the crowd
    # ceiling. Where the VLM already matches what one more rater would score, a purchased vote
    # buys a duplicate of a label we get free.
    if Path(args.ratings).exists():
        pct = {r["faceId"]: float(r["percentile"]) for r in rr}

        def band_stats(sel: list[int]) -> tuple[float, float]:
            """-> (VLM agreement, leave-one-out crowd ceiling) over a set of pairs."""
            hit = tot = ceil_hit = ceil_n = 0.0
            for i in sel:
                wa, wb = raw[i]
                if by_index[i].final_outcome in ("A", "B"):
                    hit += wa if by_index[i].final_outcome == "A" else wb
                    tot += wa + wb
                # Leave-one-out: how often one rater matches the majority of the others.
                for w, other in ((wa, wb), (wb, wa)):
                    if w:
                        ceil_hit += w if (w - 1) > other else 0.0
                        ceil_n += w
            return (hit / tot if tot else float("nan"),
                    ceil_hit / ceil_n if ceil_n else float("nan"))

        # Headroom drives a five-figure spend decision, so it needs an interval. Resampling is
        # over *pairs*, which is the unit we would buy more of; the wide bands hold very few
        # purchased pairs and that has to be visible rather than hidden behind a point estimate.
        rng = random.Random(17)
        print(f"\n{'pct gap':>10} {'pairs':>6} {'decisive':>9} {'VLM vs votes':>13}"
              f" {'1 rater (ceiling)':>18} {'headroom (95% CI)':>26} {'hi-conf':>8}")
        gap_rows = []
        for lo, hi in [(0, 2), (2, 5), (5, 10), (10, 20), (20, 45), (45, 101)]:
            sel = [i for i in keys
                   if by_index[i].face_a_id in pct and by_index[i].face_b_id in pct
                   and lo <= abs(pct[by_index[i].face_a_id]
                                 - pct[by_index[i].face_b_id]) * 100 < hi]
            if not sel:
                continue
            dec = sum(abs(shares[i] - 0.5) for i in sel) / len(sel)
            acc, ceil = band_stats(sel)
            boots = []
            for _ in range(args.band_reps):
                bs = [rng.choice(sel) for _ in sel]
                a2, c2 = band_stats(bs)
                if a2 == a2 and c2 == c2:
                    boots.append(c2 - a2)
            boots.sort()
            b_lo = boots[int(0.025 * len(boots))] if boots else float("nan")
            b_hi = boots[int(0.975 * len(boots))] if boots else float("nan")
            hc = sum(1 for i in sel if (by_index[i].confidence or "") == "high") / len(sel)
            verdict = "BUY" if b_lo > 0 else ("skip" if b_hi < 0 else "unresolved")
            gap_rows.append({"band": f"{lo}-{min(hi, 100)}", "pairs": len(sel),
                             "decisiveness": dec, "vlmAgreement": acc,
                             "crowdCeiling": ceil, "headroom": ceil - acc,
                             "headroomLo": b_lo, "headroomHi": b_hi, "verdict": verdict,
                             "highConfShare": hc})
            print(f"{lo:4}-{min(hi, 100):<4} {len(sel):6,} {dec:9.3f} {acc:12.1%} "
                  f"{ceil:17.1%}   {(ceil - acc) * 100:+5.1f} [{b_lo * 100:+5.1f},"
                  f"{b_hi * 100:+5.1f}] {verdict:10} {hc:7.1%}")
        print("\nheadroom = what a human vote could add over the free VLM label in that band.")
        print("BUY = interval entirely above 0. unresolved = we cannot yet tell, do not spend.")
        report["byPercentileGap"] = gap_rows

    # ------------------------------------------------------------------ 3. what it costs
    print("\n=== 3. Cost of buying panel labels, by cell (12 votes/pair) ===")
    cells = Counter()
    for m in matchups:
        same = m.face_a_decile == m.face_b_decile
        cells[((m.confidence or "?"), "same-decile" if same else "cross-decile")] += 1
    have = defaultdict(int)
    for i in keys:
        m = by_index[i]
        same = m.face_a_decile == m.face_b_decile
        have[((m.confidence or "?"), "same-decile" if same else "cross-decile")] += 1
    print(f"{'cell':28} {'pairs':>7} {'bought':>7} {'remaining':>10} {'cost @12 votes':>15}")
    cost_rows = []
    for cell, n in sorted(cells.items(), key=lambda kv: -kv[1]):
        rem = n - have[cell]
        cost = rem * 12 * COST_PER_VOTE
        cost_rows.append({"cell": " ".join(cell), "pairs": n, "bought": have[cell],
                          "remaining": rem, "cost": cost})
        print(f"{cell[0] + ' ' + cell[1]:28} {n:7,} {have[cell]:7,} {rem:10,} ${cost:14,.0f}")
    total = sum(r["cost"] for r in cost_rows)
    print(f"{'TOTAL (panel-label everything)':28} {sum(cells.values()):7,} "
          f"{sum(have.values()):7,} {'':10} ${total:14,.0f}")
    report["costPerVote"] = COST_PER_VOTE
    report["cells"] = cost_rows
    report["costLabelEverything"] = total

    # Priced against the router that works. Confidence cannot separate close pairs from
    # far ones, but the percentile gap can, so this is the table a spend decision reads:
    # the low bands are where the VLM label is worth nothing and a human vote is worth most.
    if Path(args.ratings).exists():
        print("\n=== 3b. Same cost, banded by percentile gap (the usable router) ===")
        allpct = {r["faceId"]: float(r["percentile"]) for r in rr}
        tot_by_band = Counter()
        for m in matchups:
            if m.face_a_id in allpct and m.face_b_id in allpct:
                g = abs(allpct[m.face_a_id] - allpct[m.face_b_id]) * 100
                for lo, hi in [(0, 2), (2, 5), (5, 10), (10, 20), (20, 45), (45, 101)]:
                    if lo <= g < hi:
                        tot_by_band[f"{lo}-{min(hi, 100)}"] += 1
                        break
        have_band = Counter(r["band"] for r in gap_rows for _ in range(r["pairs"]))
        print(f"{'pct gap':>10} {'pairs':>7} {'bought':>7} {'remaining':>10} "
              f"{'cost @12':>11} {'VLM acc':>8}")
        band_cost = []
        for row in gap_rows:
            b = row["band"]
            n = tot_by_band[b]
            rem = max(n - have_band[b], 0)
            cost = rem * 12 * COST_PER_VOTE
            band_cost.append({"band": b, "pairs": n, "bought": have_band[b],
                              "remaining": rem, "cost": cost,
                              "vlmAgreement": row["vlmAgreement"]})
            print(f"{b:>10} {n:7,} {have_band[b]:7,} {rem:10,} ${cost:10,.0f} "
                  f"{row['vlmAgreement']:7.1%}")
        chance = [r for r in band_cost if r["vlmAgreement"] < 0.53]
        print(f"\nbands where the VLM is at chance (<53%): {sum(r['pairs'] for r in chance):,} "
              f"pairs, {sum(r['remaining'] for r in chance):,} unlabelled, "
              f"${sum(r['cost'] for r in chance):,.0f} to cover at 12 votes each")
        report["byBandCost"] = band_cost
        report["costCoverChanceBands"] = sum(r["cost"] for r in chance)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
