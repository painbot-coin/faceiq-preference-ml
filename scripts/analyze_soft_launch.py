#!/usr/bin/env python
"""Read the soft-launch export and answer the three questions it was run to answer.

1. **How long does a session actually take?** The reward for run 2 is
   hourly-rate x estimated minutes, so the observed median is worth real money.
   Note the session row is created when the rater clicks "I consent — start
   rating", *after* consent + instructions are on screen, so DB duration is
   rating time only — add reading time before quoting Prolific a number.
2. **Which raters failed?** Gold pass rate is the screen; fast-click rate and
   tie rate are context, not grounds on their own.
3. **Which golds are secretly ambiguous?** A gold that good raters fail is a bad
   gold, not a bad rater. Retire it before run 2 rather than using it to reject
   360 people's work.

Test rows (studyId == "test") are dropped throughout — they are Dit clicking
through, and in the soft launch two of them deliberately failed golds.

Usage:
    python scripts/analyze_soft_launch.py
    python scripts/analyze_soft_launch.py --results labels/panel-pilot/soft-launch/results
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TEST_STUDY_ID = "test"
FAST_CLICK_MS = 2000
# Below this gold pass rate a rater is a reject candidate. 3 of 5 golds wrong is
# not "had an off moment", it is not doing the task.
GOLD_PASS_FLOOR = 0.6
# Minutes of consent + instructions + worked examples a first-timer spends before
# the session row exists. Measured against Prolific's own median, which starts at
# study acceptance; keep this honest, underpayment gets studies flagged.
PREAMBLE_MIN = 4.0


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def pct(numerator: int, denominator: int) -> str:
    return f"{100 * numerator / denominator:.0f}%" if denominator else "n/a"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="labels/panel-pilot/soft-launch/results")
    ap.add_argument("--golds", default="labels/panel-pilot/soft-launch/golds.json")
    ap.add_argument("--hourly", type=float, default=20.0, help="target $/hr for run 2")
    args = ap.parse_args()

    results = Path(args.results)
    judgments = read_jsonl(results / "judgments.jsonl")
    sessions = read_jsonl(results / "sessions.jsonl")
    expected: dict[str, str] = json.loads(Path(args.golds).read_text())["expected"]

    tests = [s for s in sessions if s["studyId"] == TEST_STUDY_ID]
    sessions = [s for s in sessions if s["studyId"] != TEST_STUDY_ID]
    judgments = [j for j in judgments if j["studyId"] != TEST_STUDY_ID]
    print(f"{len(sessions)} real sessions, {len(judgments)} judgments "
          f"({len(tests)} test sessions dropped)")

    # --- 1. Session duration ---------------------------------------------------
    done = [s for s in sessions if s["completedAt"]]
    minutes = sorted(
        (parse_ts(s["completedAt"]) - parse_ts(s["startedAt"])).total_seconds() / 60
        for s in done
    )
    med = statistics.median(minutes)
    print(f"\n== session length ({len(done)} completed, rating time only) ==")
    print(f"median {med:.1f} min   range {minutes[0]:.1f}-{minutes[-1]:.1f} min")
    print(f"p25 {statistics.quantiles(minutes, n=4)[0]:.1f}   "
          f"p75 {statistics.quantiles(minutes, n=4)[2]:.1f}")
    est = med + PREAMBLE_MIN
    print(f"+{PREAMBLE_MIN:.0f} min consent/instructions -> quote ~{est:.0f} min per submission")
    print(f"at ${args.hourly:.0f}/hr that is ${args.hourly * est / 60:.2f} per submission")

    per_item = [j["responseTimeMs"] for j in judgments]
    print(f"median per judgment {statistics.median(per_item) / 1000:.1f}s   "
          f"fast (<{FAST_CLICK_MS}ms) {pct(sum(t < FAST_CLICK_MS for t in per_item), len(per_item))}")

    # --- 2. Per-rater quality -------------------------------------------------
    by_rater: dict[str, list[dict]] = defaultdict(list)
    for j in judgments:
        by_rater[j["prolificPid"]].append(j)

    print(f"\n== raters ({len(by_rater)}) ==")
    print(f"{'prolificPid':26} {'n':>4} {'gold':>7} {'med s':>6} {'fast':>5} {'tie':>5}  flag")
    flagged: list[str] = []
    for pid, rows in sorted(by_rater.items()):
        golds = [j for j in rows if j["isGold"]]
        # A tie on a gold is a miss: golds were picked as unanimous very-clear
        # calls, so "too close to call" is the wrong answer by construction.
        hits = sum(j["choice"] == expected.get(j["pairId"]) for j in golds)
        rate = hits / len(golds) if golds else 1.0
        times = [j["responseTimeMs"] for j in rows]
        ties = sum(j["choice"] == "tie" for j in rows)
        fast = sum(t < FAST_CLICK_MS for t in times)
        flag = "REJECT?" if rate < GOLD_PASS_FLOOR else ""
        if flag:
            flagged.append(pid)
        print(f"{pid:26} {len(rows):4} {hits:3}/{len(golds):<3} "
              f"{statistics.median(times) / 1000:6.1f} "
              f"{pct(fast, len(times)):>5} {pct(ties, len(rows)):>5}  {flag}")

    print(f"\n{len(flagged)} rater(s) under the {GOLD_PASS_FLOOR:.0%} gold floor"
          + (f": {', '.join(flagged)}" if flagged else ""))

    # --- 3. Per-gold quality --------------------------------------------------
    print(f"\n== golds ({len(expected)} shipped) ==")
    by_gold: dict[str, list[str]] = defaultdict(list)
    for j in judgments:
        if j["isGold"]:
            by_gold[j["pairId"]].append(j["choice"])

    unseen = [g for g in expected if g not in by_gold]
    suspect: list[str] = []
    for pair_id, choices in sorted(by_gold.items(), key=lambda kv: len(kv[1])):
        want = expected[pair_id]
        hits = sum(c == want for c in choices)
        marker = ""
        if len(choices) >= 2 and hits / len(choices) < 0.75:
            marker = "AMBIGUOUS?"
            suspect.append(pair_id)
        print(f"{pair_id}  want {want:5} seen {len(choices):2}  "
              f"pass {pct(hits, len(choices)):>4}  {dict(Counter(choices))}  {marker}")

    if unseen:
        print(f"\n{len(unseen)} gold(s) never shown: {', '.join(unseen)}")
    print(f"{len(suspect)} gold(s) look ambiguous — review the photos before run 2"
          + (f": {', '.join(suspect)}" if suspect else ""))

    # --- Inter-rater agreement on the real pairs ------------------------------
    real: dict[str, list[str]] = defaultdict(list)
    for j in judgments:
        if not j["isGold"]:
            real[j["pairId"]].append(j["choice"])

    votes = [len(v) for v in real.values()]
    unanimous = sum(len(set(v)) == 1 for v in real.values())
    majorities = []
    for choices in real.values():
        top = Counter(choices).most_common(1)[0][1]
        majorities.append(top / len(choices))
    print("\n== real pairs ==")
    print(f"{len(real)} pairs, {min(votes)}-{max(votes)} votes each "
          f"(median {statistics.median(votes):.0f})")
    print(f"unanimous {pct(unanimous, len(real))}   "
          f"mean majority share {statistics.mean(majorities):.0%}")
    print(f"tie votes {pct(sum(c == 'tie' for v in real.values() for c in v), sum(votes))}")

    comments = [j for j in judgments if j["comment"]]
    print(f"\n{len(comments)} comments on {len(judgments)} judgments "
          f"({pct(len(comments), len(judgments))})")
    feedback = [s for s in sessions if s.get("feedback")]
    for s in feedback:
        print(f"  feedback [{s['prolificPid']}]: {s['feedback']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
