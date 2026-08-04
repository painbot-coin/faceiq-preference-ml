#!/usr/bin/env python
"""Build the gold-standard attention checks for the panel run.

A gold is a pilot-500 pair that was *unanimous* (Dit, Alex and Gemini all picked
the same face) **and** very_clear at high VLM confidence. Both halves matter: the
unanimity says humans don't disagree about it, the theta gap says the pair isn't
merely a lucky coincidence. A rater who misses several of these wasn't looking.

The panel draw shuffles left/right independently of pilot-500, so the stored
answer is re-oriented onto whichever side the winning face landed on here.

Outputs (to --out, default labels/panel-pilot/):
  golds.json         {expected: {pairId: "left"|"right"}} — ships to the app.
  golds-review.json  photo URLs, theta gap, rater comments. Feeds app/gold_review.py.

Run after select_panel_pairs.py. If app/gold_review.py has written decisions,
pass --decisions to drop the pairs a human rejected.

Usage:
    python scripts/make_golds.py --export data/exports/<runId>
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from select_panel_pairs import unanimous_gold_winners  # noqa: E402


def load_comments(pilot_dir: Path) -> dict[str, dict[str, str]]:
    """pairId -> {rater: comment} so the reviewer can see the stated reasoning."""
    out: dict[str, dict[str, str]] = defaultdict(dict)
    for votes_file in sorted(pilot_dir.glob("votes-*.jsonl")):
        rater = votes_file.stem.removeprefix("votes-")
        for line in votes_file.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("comment"):
                out[row["pairId"]][rater] = row["comment"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-dir", default="labels/pilot-500")
    ap.add_argument("--panel-dir", default="labels/panel-pilot")
    ap.add_argument(
        "--decisions",
        help="decisions.jsonl from app/gold_review.py; rejected pairs are dropped",
    )
    ap.add_argument("--out", default="labels/panel-pilot")
    ap.add_argument(
        "--loose",
        action="store_true",
        help="accept any unanimous pair, not just very_clear/high-confidence ones",
    )
    args = ap.parse_args()

    panel_dir = Path(args.panel_dir)
    pairs_doc = json.loads((panel_dir / "pairs.json").read_text())
    panel_pairs = {p["pairId"]: p for p in pairs_doc["pairs"]}
    panel_meta = {
        p["pairId"]: p for p in json.loads((panel_dir / "sample-meta.json").read_text())["pairs"]
    }

    winners = unanimous_gold_winners(Path(args.pilot_dir), strict=not args.loose)
    comments = load_comments(Path(args.pilot_dir))
    print(f"{len(winners)} unanimous gold candidates in pilot-500")

    rejected: set[str] = set()
    if args.decisions and Path(args.decisions).exists():
        for line in Path(args.decisions).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("decision") == "reject":
                rejected.add(row["pairId"])
            else:
                rejected.discard(row["pairId"])
        print(f"{len(rejected)} rejected in review")

    expected: dict[str, str] = {}
    review: list[dict] = []
    missing = 0
    for pair_id, winner_face in sorted(winners.items()):
        pair = panel_pairs.get(pair_id)
        if pair is None:
            missing += 1
            continue
        if pair_id in rejected:
            continue
        if winner_face == pair["left"]["faceId"]:
            side = "left"
        elif winner_face == pair["right"]["faceId"]:
            side = "right"
        else:
            raise RuntimeError(f"gold {pair_id}: winner {winner_face} not in the panel pair")
        expected[pair_id] = side
        m = panel_meta[pair_id]
        review.append(
            {
                "pairId": pair_id,
                "gender": pair["gender"],
                "expected": side,
                "winnerFaceId": winner_face,
                "left": pair["left"],
                "right": pair["right"],
                "winProb": m["winProb"],
                "thetaGap": round(abs(m["thetaA"] - m["thetaB"]), 4),
                "vlmConfidence": m["vlmConfidence"],
                "comments": comments.get(pair_id, {}),
            }
        )

    if missing:
        print(f"{missing} candidates are not in the panel draw (QC-excluded or example pairs)")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "golds.json").write_text(
        json.dumps(
            {
                "version": 1,
                "runId": pairs_doc["runId"],
                "createdAt": datetime.now(timezone.utc).isoformat(),
                "source": "pilot-500 unanimous (Dit + Alex + Gemini), very_clear/high"
                if not args.loose
                else "pilot-500 unanimous (Dit + Alex + Gemini)",
                "goldCount": len(expected),
                "expected": expected,
            },
            indent=1,
        )
        + "\n"
    )
    (out_dir / "golds-review.json").write_text(
        json.dumps({"version": 1, "golds": review}, indent=1) + "\n"
    )

    print(f"\nwrote {out_dir / 'golds.json'} ({len(expected)} golds)")
    print(f"wrote {out_dir / 'golds-review.json'} (for app/gold_review.py)")
    sides = {"left": 0, "right": 0}
    for s in expected.values():
        sides[s] += 1
    print(f"expected side balance: {sides}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
