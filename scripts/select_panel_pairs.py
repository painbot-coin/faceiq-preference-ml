#!/usr/bin/env python
"""Draw the ~3,000-pair panel sample (execution plan Step 2).

Differs from select_pilot_pairs.py in four ways:
  * panel strata, not the pilot-500 strata (close 35 / mid 25 / clear 20 /
    low-confidence 10 / overlap 10),
  * an **overlap** stratum that deliberately *includes* previously labelled
    pairs (750-pair GT audit + pilot-500) so prior labels get a free cross-check,
  * an ethnicity coverage floor, using the self-declared values joined in
    Stage 0.5,
  * QC-excluded and gender-relabelled faces are dropped (Stage 0/1.5).

Gold candidates — pilot-500 pairs where Dit, Alex and Gemini all agreed — are
force-included in the overlap stratum, because golds must exist in pairs.json or
the app refuses to start.

Outputs (to --out, default labels/panel-pilot/):
  pairs.json        blinded: pairId, gender, left/right faceId + photoUrl. Ships.
  sample-meta.json  answer key: stratum, bucket, winProb, thetas, VLM label,
                    which side face A landed on. Stays here, gitignored.

Usage:
    python scripts/select_panel_pairs.py --export data/exports/<runId> \
        --ratings artifacts/bt-refit-v2-qc/ratings.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from faceiq_pref.data import Matchup, load_export  # noqa: E402

BUCKETS = [
    ("close", 0.0, 0.60),
    ("mid", 0.60, 0.80),
    ("clear", 0.80, 0.95),
    ("very_clear", 0.95, 1.01),
]

# Execution plan Step 2 shares, as fractions of --n.
STRATUM_SHARE = {
    "close": 0.35,
    "mid": 0.25,
    "clear": 0.20,  # clear + very_clear
    "lowconf": 0.10,  # medium/low VLM confidence, any bucket
    "overlap": 0.10,  # human-audited or pilot-500 pairs
}

# Groups big enough to carry a floor (Stage 0.5 crosstab); the
# native_american/pacific_islander/other tail is pooled and reported instead.
FLOOR_GROUPS = ["white", "hispanic", "east_asian", "black", "middle_eastern", "south_asian"]
FLOOR_PER_GROUP = 150


def bucket_of(p: float) -> str:
    for name, lo, hi in BUCKETS:
        if lo <= p < hi:
            return name
    raise ValueError(f"win probability {p} out of range")


def load_thetas(path: Path) -> dict[str, float]:
    with path.open() as f:
        return {row["faceId"]: float(row["theta"]) for row in csv.DictReader(f)}


def load_face_ids(path: Path | None) -> set[str]:
    if not path:
        return set()
    with path.open() as f:
        return {row["faceId"] for row in csv.DictReader(f)}


def load_primary_ethnicity(path: Path) -> dict[str, str]:
    doc = json.loads(path.read_text())
    out: dict[str, str] = {}
    for face_id, attrs in doc["faces"].items():
        eths = attrs.get("ethnicities") or []
        if eths and eths[0] in FLOOR_GROUPS:
            out[face_id] = eths[0]
        elif eths:
            out[face_id] = "other"
    return out


def unanimous_gold_candidates(pilot_dir: Path, strict: bool = True) -> set[str]:
    return set(unanimous_gold_winners(pilot_dir, strict))


def unanimous_gold_winners(pilot_dir: Path, strict: bool = True) -> dict[str, str]:
    """pilot-500 pairs where both humans and Gemini picked the same face.

    Returns pairId -> the winning **faceId** (side-independent, so it can be
    re-oriented for whichever side the panel draw puts that face on).

    `strict` additionally requires the pair to be very_clear (winProb >= 0.95)
    at high VLM confidence. Unanimity alone is not enough for a gold: 63 of the
    220 unanimous pairs are *close* pairs where three raters happened to land on
    the same side, and failing a rater on one of those would be unjust.
    """
    meta_path = pilot_dir / "sample-meta.json"
    if not meta_path.exists():
        return {}
    meta = {p["pairId"]: p for p in json.loads(meta_path.read_text())["pairs"]}
    pairs_doc = json.loads((pilot_dir / "pairs.json").read_text())
    sides = {
        p["pairId"]: (p["left"]["faceId"], p["right"]["faceId"]) for p in pairs_doc["pairs"]
    }

    def winner_from_vote(pair_id: str, choice: str) -> str | None:
        if choice not in ("left", "right") or pair_id not in sides:
            return None
        left, right = sides[pair_id]
        return left if choice == "left" else right

    def winner_from_vlm(pair_id: str) -> str | None:
        m = meta.get(pair_id)
        if not m or m.get("vlmOutcome") not in ("A", "B"):
            return None
        left, right = sides[pair_id]
        # faceAOnLeft tells us which physical side face A occupied.
        a_face = left if m["faceAOnLeft"] else right
        b_face = right if m["faceAOnLeft"] else left
        return a_face if m["vlmOutcome"] == "A" else b_face

    human_votes: dict[str, dict[str, str]] = defaultdict(dict)
    for votes_file in sorted(pilot_dir.glob("votes-*.jsonl")):
        rater = votes_file.stem.removeprefix("votes-")
        for line in votes_file.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            w = winner_from_vote(row["pairId"], row["choice"])
            if w:  # a later line for the same pair overwrites an earlier one
                human_votes[row["pairId"]][rater] = w

    golds: dict[str, str] = {}
    for pair_id, per_rater in human_votes.items():
        if len(per_rater) < 2:
            continue
        picks = set(per_rater.values())
        vlm = winner_from_vlm(pair_id)
        if not vlm or len(picks) != 1 or vlm not in picks:
            continue
        if strict:
            m = meta.get(pair_id, {})
            if m.get("bucket") != "very_clear" or m.get("vlmConfidence") != "high":
                continue
        golds[pair_id] = picks.pop()
    return golds


def sample_gender_balanced(
    candidates: list[Matchup], n: int, rng: random.Random
) -> list[Matchup]:
    by_gender: dict[str, list[Matchup]] = defaultdict(list)
    for m in candidates:
        by_gender[m.gender].append(m)
    genders = sorted(by_gender)
    if not genders:
        return []
    quota = {g: n // len(genders) for g in genders}
    for g in genders[: n % len(genders)]:
        quota[g] += 1
    picked: list[Matchup] = []
    shortfall = 0
    for g in genders:
        pool = by_gender[g]
        take = min(quota[g], len(pool))
        shortfall += quota[g] - take
        picked.extend(rng.sample(pool, take))
    if shortfall:
        chosen = {m.comparison_id for m in picked}
        remaining = [m for m in candidates if m.comparison_id not in chosen]
        picked.extend(rng.sample(remaining, min(shortfall, len(remaining))))
    return picked


def groups_in(m: Matchup, ethnicity: dict[str, str]) -> set[str]:
    return {
        g
        for g in (ethnicity.get(m.face_a_id), ethnicity.get(m.face_b_id))
        if g in FLOOR_GROUPS
    }


def enforce_ethnicity_floor(
    selected: list[tuple[Matchup, str]],
    pool_by_stratum: dict[str, list[Matchup]],
    ethnicity: dict[str, str],
    rng: random.Random,
) -> tuple[list[tuple[Matchup, str]], dict[str, int]]:
    """Swap pairs in, preserving each stratum's size and gender mix.

    A deficient group's incoming pair replaces an outgoing pair of the same
    stratum and gender that carries no deficient group — so the floor is met
    without disturbing the strata the experiment is built on. `overlap` is never
    swapped out: those pairs are there for the cross-check (and hold the golds).
    """
    chosen_ids = {m.comparison_id for m, _ in selected}
    swaps = 0
    for group in FLOOR_GROUPS:
        for _ in range(FLOOR_PER_GROUP * 3):  # bounded; breaks out when satisfied
            have = sum(1 for m, _ in selected if group in groups_in(m, ethnicity))
            if have >= FLOOR_PER_GROUP:
                break
            incoming = [
                m
                for stratum in ("close", "mid", "clear", "lowconf")
                for m in pool_by_stratum[stratum]
                if m.comparison_id not in chosen_ids and group in groups_in(m, ethnicity)
            ]
            if not incoming:
                break
            new = rng.choice(incoming)
            new_stratum = next(
                s for s in ("close", "mid", "clear", "lowconf")
                if any(x.comparison_id == new.comparison_id for x in pool_by_stratum[s])
            )
            # Drop something we can afford to lose: same stratum, same gender,
            # and only carrying groups that are already comfortably above floor.
            deficient = {
                g
                for g in FLOOR_GROUPS
                if sum(1 for m, _ in selected if g in groups_in(m, ethnicity)) < FLOOR_PER_GROUP
            }
            droppable = [
                (m, s)
                for m, s in selected
                if s == new_stratum
                and m.gender == new.gender
                and not (groups_in(m, ethnicity) & deficient)
            ]
            if not droppable:
                break
            out = rng.choice(droppable)
            selected.remove(out)
            chosen_ids.discard(out[0].comparison_id)
            selected.append((new, new_stratum))
            chosen_ids.add(new.comparison_id)
            swaps += 1

    coverage = {
        g: sum(1 for m, _ in selected if g in groups_in(m, ethnicity)) for g in FLOOR_GROUPS
    }
    coverage["_swaps"] = swaps
    return selected, coverage


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--ratings", default="artifacts/bt-refit-v2-qc/ratings.csv")
    ap.add_argument("--exclude-faces", default="artifacts/face-qc-v1/exclude-faces.csv")
    ap.add_argument("--exclude-genders", default="artifacts/face-qc-v1/gender-fixes.csv")
    ap.add_argument("--attributes", default="labels/face-attributes.json")
    ap.add_argument("--pilot-dir", default="labels/pilot-500")
    ap.add_argument("--example-pairs", default="labels/panel-pilot/example-pairs.json")
    ap.add_argument("--out", default="labels/panel-pilot")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20260727)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    export = load_export(args.export)
    faces = export.faces()
    matchups = export.all_matchups()
    thetas = load_thetas(Path(args.ratings))
    print(f"export OK: run {export.run_id}, {len(matchups)} matchups, {len(thetas)} ranked faces")

    bad_faces = load_face_ids(Path(args.exclude_faces)) | load_face_ids(
        Path(args.exclude_genders)
    )
    ethnicity = load_primary_ethnicity(Path(args.attributes))
    example_pairs = set()
    if Path(args.example_pairs).exists():
        example_pairs = set(json.loads(Path(args.example_pairs).read_text())["pairIds"])

    pilot_pairs = set()
    pilot_pairs_path = Path(args.pilot_dir) / "pairs.json"
    if pilot_pairs_path.exists():
        pilot_pairs = {
            p["pairId"] for p in json.loads(pilot_pairs_path.read_text())["pairs"]
        }
    gold_candidates = unanimous_gold_candidates(Path(args.pilot_dir))
    print(f"pilot-500: {len(pilot_pairs)} pairs, {len(gold_candidates)} unanimous gold candidates")

    # Eligibility and stratum assignment.
    p_of: dict[str, float] = {}
    pool: dict[str, list[Matchup]] = defaultdict(list)
    skipped = Counter()
    for m in matchups:
        if m.comparison_id in example_pairs:
            skipped["example pair"] += 1
            continue
        if m.face_a_id in bad_faces or m.face_b_id in bad_faces:
            skipped["QC-excluded face"] += 1
            continue
        ta, tb = thetas.get(m.face_a_id), thetas.get(m.face_b_id)
        if ta is None or tb is None:
            skipped["face not ranked"] += 1
            continue
        p = 1.0 / (1.0 + math.exp(-abs(ta - tb)))
        p_of[m.comparison_id] = p

        previously_labelled = m.human_labeled_at is not None or m.comparison_id in pilot_pairs
        if previously_labelled:
            pool["overlap"].append(m)
            continue
        if m.confidence in ("medium", "low"):
            pool["lowconf"].append(m)
            continue
        b = bucket_of(p)
        pool["clear" if b in ("clear", "very_clear") else b].append(m)

    print(f"skipped: {dict(skipped)}")
    for name in ("close", "mid", "clear", "lowconf", "overlap"):
        print(f"  {name}: {len(pool[name])} available")

    targets = {s: round(args.n * share) for s, share in STRATUM_SHARE.items()}
    targets["close"] += args.n - sum(targets.values())  # absorb rounding

    selected: list[tuple[Matchup, str]] = []

    # Golds first — they must survive into pairs.json.
    forced = [m for m in pool["overlap"] if m.comparison_id in gold_candidates]
    forced = forced[: targets["overlap"]]
    selected.extend((m, "overlap") for m in forced)
    forced_ids = {m.comparison_id for m in forced}
    rest_overlap = [m for m in pool["overlap"] if m.comparison_id not in forced_ids]
    picks = sample_gender_balanced(rest_overlap, targets["overlap"] - len(forced), rng)
    selected.extend((m, "overlap") for m in picks)
    print(f"overlap: {len(forced)} forced golds + {len(picks)} sampled")

    for stratum in ("close", "mid", "clear", "lowconf"):
        picks = sample_gender_balanced(pool[stratum], targets[stratum], rng)
        if len(picks) < targets[stratum]:
            print(f"  WARNING {stratum}: only {len(picks)} of {targets[stratum]} available")
        selected.extend((m, stratum) for m in picks)

    selected, coverage = enforce_ethnicity_floor(selected, pool, ethnicity, rng)

    rng.shuffle(selected)
    pairs_out, meta_out = [], []
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
                "faceAId": m.face_a_id,
                "faceBId": m.face_b_id,
                "thetaA": thetas[m.face_a_id],
                "thetaB": thetas[m.face_b_id],
                "vlmOutcome": m.vlm_outcome,
                "vlmConfidence": m.confidence,
                "humanLabeled": m.human_labeled_at is not None,
                "inPilot500": m.comparison_id in pilot_pairs,
                "isGoldCandidate": m.comparison_id in gold_candidates,
                "faceAOnLeft": a_left,
                "ethnicityA": ethnicity.get(m.face_a_id),
                "ethnicityB": ethnicity.get(m.face_b_id),
            }
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pairs.json").write_text(
        json.dumps(
            {
                "version": 1,
                "runId": export.run_id,
                "seed": args.seed,
                "createdAt": datetime.now(timezone.utc).isoformat(),
                "pairCount": len(pairs_out),
                "pairs": pairs_out,
            },
            indent=1,
        )
        + "\n"
    )
    (out_dir / "sample-meta.json").write_text(
        json.dumps(
            {
                "version": 1,
                "runId": export.run_id,
                "seed": args.seed,
                "ratings": args.ratings,
                "strata": targets,
                "ethnicityCoverage": coverage,
                "pairs": meta_out,
            },
            indent=1,
        )
        + "\n"
    )

    print(f"\nwrote {out_dir / 'pairs.json'} ({len(pairs_out)} pairs, blinded)")
    print(f"wrote {out_dir / 'sample-meta.json'} (answer key — gitignored)")
    print(f"strata: {dict(Counter(s for _, s in selected))}")
    print(f"buckets: {dict(Counter(bucket_of(p_of[m.comparison_id]) for m, _ in selected))}")
    print(f"gender: {dict(Counter(m.gender for m, _ in selected))}")
    print(f"gold candidates in draw: {sum(1 for m, _ in selected if m.comparison_id in gold_candidates)}")
    swaps = coverage.pop("_swaps")
    print(f"ethnicity coverage (pairs containing a face of that group, floor {FLOOR_PER_GROUP}):")
    for g, n in sorted(coverage.items(), key=lambda kv: -kv[1]):
        flag = "" if n >= FLOOR_PER_GROUP else "  << UNDER FLOOR"
        print(f"  {g:<16} {n}{flag}")
    print(f"({swaps} swaps to satisfy the floor)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
