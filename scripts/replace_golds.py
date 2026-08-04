#!/usr/bin/env python
"""Retire golds the panel run proved ambiguous and replace them from panel data.

The original 20 golds were pilot-500 pairs where three judges (Dit, Alex, Gemini)
agreed unanimously and the VLM called it very_clear at high confidence. Run 2 put
~90 real raters on each of them and three turned out to be genuinely contested
(59-60% pass) — they were failing honest raters rather than catching inattentive
ones.

Run 2 also handed us a much better source of golds than three-judge unanimity: 22
non-gold pairs where 13-28 independent raters agreed **unanimously**. A pair that
~20 strangers call the same way is unmistakable by construction, which is exactly
what an attention check needs to be.

Replacements are chosen to preserve the retired pairs' left/right balance, so a
rater who simply presses one key cannot pass the screen.

Usage:
    python scripts/replace_golds.py            # dry run, prints the plan
    python scripts/replace_golds.py --write     # rewrite golds.json
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Below this run-2 pass rate a gold is not an attention check any more.
RETIRE_BELOW = 0.70
# A replacement must be unanimous across at least this many real raters.
MIN_VOTES = 12
# Allow at most this many "too close to call" votes on a replacement.
MAX_TIES = 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel-dir", default="labels/panel-pilot")
    ap.add_argument("--results", default="labels/panel-pilot/results")
    ap.add_argument("--labels", default="artifacts/panel-run-v1/majority-labels.csv")
    ap.add_argument("--write", action="store_true", help="rewrite golds.json in place")
    args = ap.parse_args()

    panel = Path(args.panel_dir)
    golds_path = panel / "golds.json"
    doc = json.loads(golds_path.read_text())
    expected: dict[str, str] = doc["expected"]

    judgments = [json.loads(line) for line
                 in (Path(args.results) / "judgments.jsonl").read_text().splitlines()
                 if line.strip()]
    rows = list(csv.DictReader(open(args.labels)))
    meta = {m["pairId"]: m for m in
            json.loads((panel / "sample-meta.json").read_text())["pairs"]}

    # --- how did each shipped gold actually perform? ------------------------
    seen: dict[str, list[str]] = defaultdict(list)
    for j in judgments:
        if j["isGold"] and j["studyId"] != "test":
            seen[j["pairId"]].append(j["choice"])

    retire, keep = [], []
    for pair_id, want in expected.items():
        chs = seen.get(pair_id, [])
        rate = sum(c == want for c in chs) / len(chs) if chs else None
        (retire if rate is not None and rate < RETIRE_BELOW else keep).append(
            (pair_id, want, rate, len(chs)))

    print(f"== shipped golds: {len(expected)} ==")
    for pair_id, want, rate, n in sorted(retire + keep, key=lambda x: x[2] or 0):
        tag = "RETIRE" if (pair_id, want, rate, n) in retire else ""
        print(f"  {pair_id}  want {want:5} n={n:3} pass "
              f"{'n/a' if rate is None else f'{rate:.0%}':>4}  {tag}")

    if not retire:
        print("\nnothing to retire")
        return 0

    need = Counter(want for _, want, _, _ in retire)
    print(f"\nretiring {len(retire)}: {dict(need)} — replacements must match that "
          f"left/right split")

    # --- candidate replacements: unanimous across many real raters ----------
    gold_ids = set(expected)
    cands = []
    for r in rows:
        v, left, right, ties = (int(r["votes"]), int(r["left"]),
                                int(r["right"]), int(r["tie"]))
        if r["pairId"] in gold_ids or v < MIN_VOTES or ties > MAX_TIES:
            continue
        if max(left, right) != v - ties:      # unanimous among decided votes
            continue
        side = "left" if left > right else "right"
        m = meta[r["pairId"]]
        bt = "left" if (m["thetaA"] > m["thetaB"]) == m["faceAOnLeft"] else "right"
        # Prefer pairs the BT fit and the VLM also call the same way: three
        # independent sources agreeing is as safe as this gets.
        cands.append({
            "pairId": r["pairId"], "side": side, "votes": v, "ties": ties,
            "stratum": r["stratum"],
            "corroborated": (r["vlmChoice"] == side) and (bt == side),
        })

    cands.sort(key=lambda c: (not c["corroborated"], c["stratum"] != "clear", -c["votes"]))
    print(f"\n{len(cands)} unanimous candidates (>={MIN_VOTES} votes, <={MAX_TIES} tie)")

    picked = []
    for side, n in need.items():
        pool = [c for c in cands if c["side"] == side and c not in picked][:n]
        if len(pool) < n:
            raise SystemExit(f"only {len(pool)} '{side}' candidates for {n} slots")
        picked.extend(pool)

    print("\n== replacements ==")
    print(f"{'pairId':28} {'side':>5} {'votes':>5} {'stratum':9} corroborated")
    for c in picked:
        print(f"{c['pairId']:28} {c['side']:>5} {c['votes']:5} {c['stratum']:9} "
              f"{'yes' if c['corroborated'] else 'NO'}")

    new_expected = {k: v for k, v in expected.items()
                    if k not in {p for p, _, _, _ in retire}}
    for c in picked:
        new_expected[c["pairId"]] = c["side"]

    sides = Counter(new_expected.values())
    print(f"\nnew gold set: {len(new_expected)} golds, {dict(sides)}")

    if not args.write:
        print("\ndry run — rerun with --write to apply")
        return 0

    shutil.copy(golds_path, golds_path.with_suffix(".json.bak"))
    doc.update({
        "goldCount": len(new_expected),
        "expected": new_expected,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "source": (f"{doc.get('source', 'pilot-500 unanimous')}; "
                   f"{len(retire)} retired after panel run v1 (<{RETIRE_BELOW:.0%} "
                   f"pass), replaced with panel-unanimous pairs"),
        "revision": doc.get("revision", 1) + 1,
        "retired": {p: {"want": w, "passRate": round(r, 3), "votes": n}
                    for p, w, r, n in retire},
    })
    golds_path.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"\nwrote {golds_path} (backup at {golds_path.with_suffix('.json.bak')})")
    print(f"ship with:  cp {golds_path} ../faceiq-rating/data/golds.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
