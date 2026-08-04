#!/usr/bin/env python
"""Draw a targeted top-up of *hard* pairs — the run-3+ selection rule.

`select_panel_pairs.py` draws run 2's designed experiment: fixed stratum shares plus an
overlap stratum for cross-checking prior labels. That was the right sample for *asking*
whether the VLM labels hold up. It is the wrong sample for what we learned to do next.

Run 2's answer (see `panel-pilot-findings.md`): the VLM-derived ranking is at chance on
pairs it scores as near-ties, and fine everywhere else. So a top-up draw is one band, not
five — pairs whose two faces sit within `--max-gap` percentile points of each other — and
it must **exclude pairs we have already bought**, because breadth is what pays:

    2,250 pairs x  6 votes -> 53.8% on unseen hard pairs
    1,125 pairs x 12 votes -> 52.0%   (same money)

Output schema matches `select_panel_pairs.py` exactly, so the rating app, `analyze_panel_run.py`
and `refit_bt_panel.py` all consume it unchanged. `pairIndex` is what later joins each
study's votes onto the export.

**`--ratings` must be a ranking that never saw panel votes** (i.e. `bt-refit-v2-qc`, not any
`-panel` refit). A panel-fitted ranking has already moved the faces whose pairs the crowd
overruled, so pairs where the VLM was wrong get pushed *out* of the near-tie band and the draw
systematically misses the very pairs worth buying. Log §5.7 measures the damage: the same
headroom table cut on `bt-refit-v4-panel` inverts its recommendation entirely.

**`--max-gap 20` is the measured buy zone**, not a guess (log §5.7, and the table reproduced in
`panel-study-playbook.md` §2a). Human votes beat the free VLM label by +3.5 to +6.2 points at a
0–20 percentile gap, every one of those four bands significant. Above 20 the VLM already scores
at single-rater level and the interval on any gain straddles zero, so a wider draw spends real
money on duplicate labels.

Usage:
    python scripts/select_topup_pairs.py --export data/exports/<runId> \
        --ratings artifacts/bt-refit-v2-qc/ratings.csv \
        --max-gap 20 --n 3200 \
        --exclude-pairs labels/panel-pilot/pairs.json \
        --out labels/panel-run-3
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from faceiq_pref.data import Matchup, load_export  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from select_panel_pairs import (  # noqa: E402
    FLOOR_GROUPS,
    groups_in,
    load_face_ids,
    load_primary_ethnicity,
    sample_gender_balanced,
)


def load_ratings(path: Path) -> tuple[dict[str, float], dict[str, float]]:
    thetas: dict[str, float] = {}
    pcts: dict[str, float] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            thetas[row["faceId"]] = float(row["theta"])
            pcts[row["faceId"]] = float(row["percentile"])
    return thetas, pcts


def already_bought(paths: list[str]) -> set[str]:
    """pairIds from any previous draw's pairs.json (or sample-meta.json)."""
    seen: set[str] = set()
    for p in paths:
        path = Path(p)
        if not path.exists():
            print(f"  warning: {p} not found, not excluding anything from it")
            continue
        doc = json.loads(path.read_text())
        seen.update(x["pairId"] for x in doc.get("pairs", []))
    return seen


def carry_golds(golds_path: Path, source_pairs: Path) -> dict[str, tuple[str, str]]:
    """Gold pairId -> (left faceId, right faceId), **orientation preserved**.

    Two hard constraints force this. `lib/pairs.ts` throws at boot if a gold is absent
    from pairs.json, so every gold must ride along in each draw. And `golds.json` records
    the expected answer as a *side* ("left"/"right"), so re-randomising which side a face
    lands on would silently mark half the golds backwards. Carry the original sides.
    """
    if not golds_path.exists() or not source_pairs.exists():
        print(f"  warning: {golds_path} or {source_pairs} missing — no golds carried; "
              "the rating app will refuse to boot")
        return {}
    gold_ids = set(json.loads(golds_path.read_text())["expected"])
    sides = {p["pairId"]: (p["left"]["faceId"], p["right"]["faceId"])
             for p in json.loads(source_pairs.read_text())["pairs"]}
    out = {}
    for gid in sorted(gold_ids):
        if gid not in sides:
            print(f"  warning: gold {gid} not in {source_pairs}, cannot carry orientation")
            continue
        out[gid] = sides[gid]
    return out


def top_up_floor(selected: list[Matchup], pool: list[Matchup],
                 ethnicity: dict[str, str], floor: int,
                 rng: random.Random) -> tuple[list[Matchup], dict[str, int]]:
    """Swap to meet a per-group ethnicity floor, preserving size and gender mix."""
    chosen = {m.comparison_id for m in selected}
    swaps = 0
    for group in FLOOR_GROUPS:
        for _ in range(floor * 3):
            have = sum(1 for m in selected if group in groups_in(m, ethnicity))
            if have >= floor:
                break
            incoming = [m for m in pool
                        if m.comparison_id not in chosen and group in groups_in(m, ethnicity)]
            if not incoming:
                break
            new = rng.choice(incoming)
            deficient = {g for g in FLOOR_GROUPS
                         if sum(1 for m in selected if g in groups_in(m, ethnicity)) < floor}
            droppable = [m for m in selected
                         if m.gender == new.gender
                         and not (groups_in(m, ethnicity) & deficient)]
            if not droppable:
                break
            out = rng.choice(droppable)
            selected.remove(out)
            chosen.discard(out.comparison_id)
            selected.append(new)
            chosen.add(new.comparison_id)
            swaps += 1
    coverage = {g: sum(1 for m in selected if g in groups_in(m, ethnicity))
                for g in FLOOR_GROUPS}
    coverage["_swaps"] = swaps
    return selected, coverage


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--ratings", default="artifacts/bt-refit-v2-qc/ratings.csv",
                    help="must be a refit that never saw panel votes; see module docstring")
    ap.add_argument("--max-gap", type=float, default=20.0,
                    help="keep pairs within this many percentile points. Default 20 is the "
                         "measured buy zone (log §5.7); above it the VLM already matches a "
                         "single rater and votes buy duplicates")
    ap.add_argument("--allow-panel-ranking", action="store_true",
                    help="override the refusal to draw from a panel-fitted refit")
    ap.add_argument("--n", type=int, default=3200)
    ap.add_argument("--exclude-pairs", nargs="*", default=["labels/panel-pilot/pairs.json"],
                    help="previous draws' pairs.json — these pairs are not re-bought")
    ap.add_argument("--exclude-faces", default="artifacts/face-qc-v1/exclude-faces.csv")
    ap.add_argument("--exclude-genders", default="artifacts/face-qc-v1/gender-fixes.csv")
    ap.add_argument("--attributes", default="labels/face-attributes.json")
    ap.add_argument("--example-pairs", default="labels/panel-pilot/example-pairs.json")
    ap.add_argument("--golds", default="labels/panel-pilot/golds.json",
                    help="golds ride along in every draw; the app won't boot without them")
    ap.add_argument("--gold-pairs", default="labels/panel-pilot/pairs.json",
                    help="draw the golds' original left/right from here, unchanged")
    ap.add_argument("--floor", type=int, default=0,
                    help="minimum pairs containing a face of each major group (0 = off)")
    ap.add_argument("--out", default="labels/panel-run-3")
    ap.add_argument("--seed", type=int, default=20260731)
    ap.add_argument("--votes", type=float, default=6.0,
                    help="target votes per pair; 6-8 is the measured sweet spot")
    ap.add_argument("--reward", type=float, default=3.00, help="Prolific reward per submission")
    ap.add_argument("--fee", type=float, default=0.4286, help="Prolific platform fee")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    export = load_export(args.export)
    faces = export.faces()
    matchups = export.all_matchups()
    thetas, pcts = load_ratings(Path(args.ratings))
    print(f"export OK: run {export.run_id}, {len(matchups)} matchups, {len(thetas)} ranked faces")

    # Selecting on a ranking that absorbed panel votes pushes the pairs the crowd overruled out
    # of the near-tie band, so the draw misses exactly what it is meant to buy (log §5.7).
    metrics = Path(args.ratings).parent / "metrics.json"
    if (metrics.exists() and "panelVotes" in json.loads(metrics.read_text())
            and not args.allow_panel_ranking):
        raise SystemExit(
            f"refusing to draw from {args.ratings}: this refit was fitted on panel votes, so "
            f"the percentile gaps already encode the crowd's corrections and the near-tie band "
            f"will systematically exclude the pairs worth buying (log §5.7). Pass a VLM-only "
            f"refit such as artifacts/bt-refit-v2-qc/ratings.csv, or --allow-panel-ranking if "
            f"you have a reason and are recording it."
        )

    bad = load_face_ids(Path(args.exclude_faces)) | load_face_ids(Path(args.exclude_genders))
    ethnicity = load_primary_ethnicity(Path(args.attributes))
    prior = already_bought(args.exclude_pairs)
    examples: set[str] = set()
    if Path(args.example_pairs).exists():
        examples = set(json.loads(Path(args.example_pairs).read_text())["pairIds"])
    print(f"excluding {len(prior)} already-bought pairs, {len(examples)} worked examples, "
          f"{len(bad)} QC faces")

    pool: list[Matchup] = []
    gap_of: dict[str, float] = {}
    skipped = Counter()
    for m in matchups:
        if m.comparison_id in prior:
            skipped["already bought"] += 1
            continue
        if m.comparison_id in examples:
            skipped["worked example"] += 1
            continue
        if m.face_a_id in bad or m.face_b_id in bad:
            skipped["QC-excluded face"] += 1
            continue
        pa, pb = pcts.get(m.face_a_id), pcts.get(m.face_b_id)
        if pa is None or pb is None:
            skipped["face not ranked"] += 1
            continue
        gap = abs(pa - pb) * 100
        if gap > args.max_gap:
            skipped["gap too wide"] += 1
            continue
        gap_of[m.comparison_id] = gap
        pool.append(m)

    print(f"skipped: {dict(skipped)}")
    print(f"eligible within {args.max_gap:g} percentile points: {len(pool)} pairs")
    if len(pool) < args.n:
        print(f"  WARNING: only {len(pool)} available for --n {args.n}; taking all")

    selected = sample_gender_balanced(pool, min(args.n, len(pool)), rng)
    coverage: dict[str, int] = {}
    if args.floor:
        selected, coverage = top_up_floor(selected, pool, ethnicity, args.floor, rng)

    # Report the draw against the measured headroom bands, so a study's expected value is
    # visible before it is paid for rather than reconstructed afterwards (log §5.7).
    HEADROOM = {"0-2": 6.2, "2-5": 5.9, "5-10": 3.5, "10-20": 3.8, "20-45": 0.1, "45-100": -1.1}
    band_counts = Counter()
    for m in selected:
        g = gap_of[m.comparison_id]
        for lo, hi, name in ((0, 2, "0-2"), (2, 5, "2-5"), (5, 10, "5-10"),
                             (10, 20, "10-20"), (20, 45, "20-45"), (45, 1e9, "45-100")):
            if lo <= g < hi:
                band_counts[name] += 1
                break
    print(f"\ndraw by headroom band ({len(selected)} pairs):")
    for name in ("0-2", "2-5", "5-10", "10-20", "20-45", "45-100"):
        if band_counts[name]:
            flag = "" if HEADROOM[name] > 1.0 else "   <- outside the measured buy zone"
            print(f"  {name:>7} {band_counts[name]:5} pairs   headroom {HEADROOM[name]:+.1f}pt{flag}")
    outside = sum(n for b, n in band_counts.items() if HEADROOM[b] <= 1.0)
    if outside:
        print(f"  WARNING: {outside} of {len(selected)} pairs ({outside / len(selected):.0%}) sit "
              f"in bands where a human vote is not measurably better than the free VLM label")

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
        meta_out.append({
            "pairId": m.comparison_id,
            "pairIndex": m.pair_index,
            "stratum": f"topup-0-{args.max_gap:g}",
            "bucket": "close",
            "winProb": round(1.0 / (1.0 + math.exp(-abs(ta - tb))), 6),
            "percentileGap": round(gap_of[m.comparison_id], 4),
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

    # Golds are appended after the draw, bypassing the gap filter (they are deliberately
    # very-clear pairs) and the already-bought filter. lib/pairs.ts keeps them out of the
    # real-pair pool by pairId, so they add no cost beyond the ~5% the app injects.
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
        "maxPercentileGap": args.max_gap, "createdAt": stamp,
        "excludedPriorPairs": len(prior),
        "ethnicityCoverage": coverage, "pairs": meta_out}, indent=1) + "\n")

    gaps = [gap_of[m.comparison_id] for m in selected]
    print(f"\nwrote {out_dir/'pairs.json'} ({len(pairs_out)} pairs = "
          f"{len(selected)} new + {carried} carried golds, blinded)")
    print(f"wrote {out_dir/'sample-meta.json'} (answer key — gitignored)")
    print(f"gender: {dict(Counter(m.gender for m in selected))}")
    print(f"percentile gap: median {sorted(gaps)[len(gaps)//2]:.1f}, max {max(gaps):.1f}")
    faces_touched = {f for m in selected for f in (m.face_a_id, m.face_b_id)}
    per_face = Counter(f for m in selected for f in (m.face_a_id, m.face_b_id))
    print(f"faces touched: {len(faces_touched)} "
          f"({len(selected)*2/len(faces_touched):.1f} new comparisons per face)")
    print(f"  faces appearing once: {sum(1 for c in per_face.values() if c == 1)}")
    if coverage:
        swaps = coverage.pop("_swaps")
        print(f"ethnicity coverage (floor {args.floor}, {swaps} swaps):")
        for g, n in sorted(coverage.items(), key=lambda kv: -kv[1]):
            print(f"  {g:<16} {n}{'' if n >= args.floor else '  << UNDER FLOOR'}")
    # Prolific's representative sample will not run below 300 participants, so the
    # rater count is floored there even when fewer would hit the vote target.
    raters = max(math.ceil(len(selected) * args.votes / 100), 300)
    votes = raters * 100
    reward = raters * args.reward
    print(f"\nnext: {raters} raters x 100 pairs = {votes:,} votes over {len(selected)} pairs "
          f"= {votes/len(selected):.2f} votes/pair")
    print(f"      ${reward:,.2f} reward + {args.fee:.2%} fee = "
          f"${reward*(1+args.fee):,.2f}  (${reward*(1+args.fee)/votes:.4f}/vote)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
