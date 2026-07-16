#!/usr/bin/env python
"""Sample the pilot-500 pairs for two-rater human labeling.

Stratifies matchups by Bradley-Terry win probability p = sigmoid(|thetaA - thetaB|):
close (p < 0.60), mid (0.60-0.80), clear (0.80-0.95), very clear (>= 0.95), plus a
random stratum drawn from all eligible pairs. Excludes pairs already human-audited
during the original GT run (the raters have seen those).

Outputs (to --out, default labels/pilot-500/):
  pairs.json        blinded pair list, committed to git. Contains ONLY pair id,
                    gender, and left/right face ids + photo URLs — no bucket, no
                    Gemini label, no theta.
  sample-meta.json  the answer key (bucket, p, thetas, Gemini outcome/confidence).
                    Gitignored; regenerable by rerunning with the same seed.

Usage:
    python scripts/select_pilot_pairs.py --export data/exports/<runId>
        [--ratings artifacts/bt-refit-v1/ratings.csv] [--out labels/pilot-500]
        [--seed 20260715]
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

from faceiq_pref.data import Matchup, load_export

BUCKETS = [
    ("close", 0.0, 0.60),
    ("mid", 0.60, 0.80),
    ("clear", 0.80, 0.95),
    ("very_clear", 0.95, 1.01),
]

# stratum -> target pair count (the "random" stratum is drawn across all buckets)
STRATA = {"close": 175, "mid": 125, "clear": 100, "random": 100}


def bucket_of(p: float) -> str:
    for name, lo, hi in BUCKETS:
        if lo <= p < hi:
            return name
    raise ValueError(f"win probability {p} out of range")


def load_thetas(ratings_csv: Path) -> dict[str, float]:
    thetas: dict[str, float] = {}
    with ratings_csv.open() as f:
        for row in csv.DictReader(f):
            thetas[row["faceId"]] = float(row["theta"])
    return thetas


def sample_gender_balanced(
    candidates: list[Matchup], n: int, rng: random.Random
) -> list[Matchup]:
    """Draw n pairs, half from each gender (matchups are same-gender pairs)."""
    by_gender: dict[str, list[Matchup]] = {}
    for m in candidates:
        by_gender.setdefault(m.gender, []).append(m)
    genders = sorted(by_gender)
    picked: list[Matchup] = []
    # Even split across genders; remainder + shortfall spill into the other gender.
    quota = {g: n // len(genders) for g in genders}
    for g in genders[: n % len(genders)]:
        quota[g] += 1
    shortfall = 0
    for g in genders:
        pool = by_gender[g]
        take = min(quota[g], len(pool))
        shortfall += quota[g] - take
        picked.extend(rng.sample(pool, take))
    if shortfall:
        remaining = [m for m in candidates if m not in set(picked)]
        picked.extend(rng.sample(remaining, min(shortfall, len(remaining))))
    return picked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True, help="export directory (contains manifest.json)")
    ap.add_argument("--ratings", default="artifacts/bt-refit-v1/ratings.csv")
    ap.add_argument("--out", default="labels/pilot-500")
    ap.add_argument("--seed", type=int, default=20260715)
    args = ap.parse_args()

    export = load_export(args.export)
    print(f"export OK: run {export.run_id}")
    faces = export.faces()
    matchups = export.all_matchups()
    thetas = load_thetas(Path(args.ratings))
    rng = random.Random(args.seed)

    # Eligible = never human-audited, both faces have a BT theta.
    eligible: list[tuple[Matchup, float]] = []
    n_audited = n_missing_theta = 0
    for m in matchups:
        if m.human_labeled_at is not None:
            n_audited += 1
            continue
        ta, tb = thetas.get(m.face_a_id), thetas.get(m.face_b_id)
        if ta is None or tb is None:
            n_missing_theta += 1
            continue
        p = 1.0 / (1.0 + math.exp(-abs(ta - tb)))
        eligible.append((m, p))
    print(
        f"{len(eligible)} eligible pairs "
        f"(excluded {n_audited} human-audited, {n_missing_theta} missing theta)"
    )

    by_bucket: dict[str, list[Matchup]] = {name: [] for name, _, _ in BUCKETS}
    p_of: dict[str, float] = {}
    for m, p in eligible:
        by_bucket[bucket_of(p)].append(m)
        p_of[m.comparison_id] = p
    for name, pool in by_bucket.items():
        print(f"  {name}: {len(pool)} available")

    selected: list[tuple[Matchup, str]] = []  # (matchup, stratum)
    taken: set[str] = set()
    for stratum in ("close", "mid", "clear"):
        picks = sample_gender_balanced(by_bucket[stratum], STRATA[stratum], rng)
        selected.extend((m, stratum) for m in picks)
        taken.update(m.comparison_id for m in picks)

    random_pool = [m for m, _ in eligible if m.comparison_id not in taken]
    picks = sample_gender_balanced(random_pool, STRATA["random"], rng)
    selected.extend((m, "random") for m in picks)

    total = sum(STRATA.values())
    if len(selected) != total:
        raise RuntimeError(f"selected {len(selected)} pairs, expected {total}")

    # Shuffle presentation order so raters can't infer stratum from position,
    # and randomize which face shows on the left.
    rng.shuffle(selected)

    pairs_out = []
    meta_out = []
    for m, stratum in selected:
        a_left = rng.random() < 0.5
        left_id, right_id = (m.face_a_id, m.face_b_id) if a_left else (m.face_b_id, m.face_a_id)
        pairs_out.append(
            {
                "pairId": m.comparison_id,
                "gender": m.gender,
                "left": {"faceId": left_id, "photoUrl": faces[left_id].photo_url},
                "right": {"faceId": right_id, "photoUrl": faces[right_id].photo_url},
            }
        )
        meta_out.append(
            {
                "pairId": m.comparison_id,
                "pairIndex": m.pair_index,
                "stratum": stratum,
                "bucket": bucket_of(p_of[m.comparison_id]),
                "winProb": round(p_of[m.comparison_id], 6),
                "thetaA": thetas[m.face_a_id],
                "thetaB": thetas[m.face_b_id],
                "vlmOutcome": m.vlm_outcome,
                "vlmConfidence": m.confidence,
                "faceAOnLeft": a_left,
            }
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_doc = {
        "version": 1,
        "runId": export.run_id,
        "seed": args.seed,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "pairCount": len(pairs_out),
        "pairs": pairs_out,
    }
    (out_dir / "pairs.json").write_text(json.dumps(pairs_doc, indent=1) + "\n")
    meta_doc = {
        "version": 1,
        "runId": export.run_id,
        "seed": args.seed,
        "strata": STRATA,
        "pairs": meta_out,
    }
    (out_dir / "sample-meta.json").write_text(json.dumps(meta_doc, indent=1) + "\n")

    strata_counts = Counter(s for _, s in selected)
    bucket_counts = Counter(bucket_of(p_of[m.comparison_id]) for m, _ in selected)
    gender_counts = Counter(m.gender for m, _ in selected)
    print(f"\nwrote {out_dir / 'pairs.json'} ({len(pairs_out)} pairs, blinded)")
    print(f"wrote {out_dir / 'sample-meta.json'} (answer key — gitignored)")
    print(f"strata: {dict(strata_counts)}")
    print(f"buckets (incl. random draws): {dict(bucket_counts)}")
    print(f"gender: {dict(gender_counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
