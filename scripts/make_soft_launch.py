#!/usr/bin/env python
"""Carve a small soft-launch subset out of the panel draw.

The app hands each rater the 100 *least-judged* pairs. Ship all 3,000 to a
12-rater soft launch and every rater gets a disjoint slice — 12 sessions, 1 vote
per pair, no inter-rater agreement to measure and nothing to sanity-check the
run with. Shipping ~100 real pairs instead makes all 12 raters converge on the
same pairs, which is exactly the overlap the Step 6 gates need.

Golds are carried over whole: the app throws if golds.json names a pair that
isn't in pairs.json, and 33 golds is already thin.

The subset is stratum-proportional and gender-balanced so the soft launch reads
like a miniature of the real thing.

Outputs to labels/panel-pilot/soft-launch/: pairs.json, golds.json, examples.json
— the three files that get copied into faceiq-rating/data/.

Usage:
    python scripts/make_soft_launch.py [--n 100]
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel-dir", default="labels/panel-pilot")
    ap.add_argument("--out", default="labels/panel-pilot/soft-launch")
    ap.add_argument("--n", type=int, default=100, help="non-gold pairs to include")
    ap.add_argument("--seed", type=int, default=20260727)
    args = ap.parse_args()

    panel_dir = Path(args.panel_dir)
    pairs_doc = json.loads((panel_dir / "pairs.json").read_text())
    golds_doc = json.loads((panel_dir / "golds.json").read_text())
    meta = {m["pairId"]: m for m in json.loads((panel_dir / "sample-meta.json").read_text())["pairs"]}
    golds = set(golds_doc["expected"])
    rng = random.Random(args.seed)

    real = [p for p in pairs_doc["pairs"] if p["pairId"] not in golds]
    by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for p in real:
        by_key[(meta[p["pairId"]]["stratum"], p["gender"])].append(p)

    # Proportional allocation, largest-remainder so the counts land on exactly n.
    shares = {k: len(v) / len(real) * args.n for k, v in by_key.items()}
    quota = {k: int(v) for k, v in shares.items()}
    for k in sorted(shares, key=lambda k: -(shares[k] - quota[k]))[: args.n - sum(quota.values())]:
        quota[k] += 1

    picked: list[dict] = []
    for key, pool in by_key.items():
        picked.extend(rng.sample(pool, min(quota[key], len(pool))))
    rng.shuffle(picked)

    gold_pairs = [p for p in pairs_doc["pairs"] if p["pairId"] in golds]
    if len(gold_pairs) != len(golds):
        orphans = golds - {p["pairId"] for p in pairs_doc["pairs"]}
        raise SystemExit(
            f"golds.json names {len(orphans)} pair(s) missing from the draw: "
            f"{sorted(orphans)}. The app throws at boot on these — rerun "
            f"scripts/make_golds.py --decisions artifacts/panel-golds-v1/decisions.jsonl"
        )
    out_pairs = picked + gold_pairs
    rng.shuffle(out_pairs)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pairs.json").write_text(
        json.dumps(
            {
                **{k: v for k, v in pairs_doc.items() if k != "pairs"},
                "createdAt": datetime.now(timezone.utc).isoformat(),
                "softLaunchSubset": {
                    "realPairs": len(picked),
                    "goldPairs": len(gold_pairs),
                    "drawnFrom": pairs_doc["pairCount"],
                    "seed": args.seed,
                },
                "pairCount": len(out_pairs),
                "pairs": out_pairs,
            },
            indent=1,
        )
        + "\n"
    )
    shutil.copy(panel_dir / "golds.json", out_dir / "golds.json")
    shutil.copy(panel_dir / "examples.json", out_dir / "examples.json")

    print(f"wrote {out_dir}/pairs.json — {len(picked)} real + {len(gold_pairs)} gold")
    print(f"copied golds.json ({len(golds)} golds) and examples.json unchanged")
    print(f"strata: {dict(Counter(meta[p['pairId']]['stratum'] for p in picked))}")
    print(f"gender: {dict(Counter(p['gender'] for p in picked))}")
    print(
        f"\nship with:  cp {out_dir}/{{pairs,golds,examples}}.json ../faceiq-rating/data/\n"
        f"then in faceiq-rating: git commit -am 'soft-launch data' && git push"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
