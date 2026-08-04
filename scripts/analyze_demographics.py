#!/usr/bin/env python
"""Is there a per-audience effect worth building for, and can our panel even measure one?

Alex's proposal is to label each face once per viewer cohort so the product can say "your jaw
is worth +0.9 to white American viewers but +0.3 to East Asian viewers". That is a 12x data
bill, so before designing around it we should ask the cheap version of the question on votes
we have already paid for: **do raters from different cohorts disagree with each other more
than raters from the same cohort do?**

The trap is that any two groups of raters disagree, because raters are noisy. Comparing
cohorts directly would find a "cohort effect" made entirely of sampling noise. So this uses a
matched design:

    in-group   train on half of cohort T, predict the other half of cohort T
    out-group  train on an equally sized set of raters NOT in T, predict the same half of T

Both predictors face the identical test votes and are built from the identical number of
raters, so noise cancels. If preferences really are cohort-specific, the in-group predictor
wins. If the two are level, cohort membership carries no signal our data can see, and the
per-audience feature would be shipping noise.

Also reports cell sizes, because a cohort with 40 raters cannot support its own ranking
however real the effect turns out to be.

Usage:
    python scripts/analyze_demographics.py \
        --panel labels/panel-pilot/results labels/panel-pilot/sample-meta.json \
        --panel labels/panel-run-3/results labels/panel-run-3/sample-meta.json \
        --rejects artifacts/panel-run-v1/reject-pids.txt \
                  artifacts/panel-run-v3/reject-pids.txt \
        --out artifacts/panel-demographics/summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from faceiq_pref.panel import load_rejects  # noqa: E402

ATTRS = ["Sex", "Ethnicity simplified", "age_band"]
MISSING = {"DATA_EXPIRED", "CONSENT_REVOKED", "", "Not Applicable", None}


def age_band(raw: str) -> str:
    try:
        a = int(float(raw))
    except (TypeError, ValueError):
        return "unknown"
    for lo, hi in ((18, 25), (25, 35), (35, 45), (45, 55), (55, 200)):
        if lo <= a < hi:
            return f"{lo}-{hi - 1}" if hi < 200 else "55+"
    return "unknown"


def load_demographics(paths: list[str]) -> dict[str, dict]:
    """Prolific participant id -> demographic row, pooled across study exports."""
    people: dict[str, dict] = {}
    for p in paths:
        with open(p, newline="") as fh:
            for row in csv.DictReader(fh):
                pid = row.get("Participant id")
                if not pid:
                    continue
                row["age_band"] = age_band(row.get("Age", ""))
                # A participant who took both studies keeps one record; demographics are a
                # property of the person, not the submission.
                people.setdefault(pid, row)
    return people


def load_judgments(panels: list[tuple[str, str]], rejects: set[str]):
    """-> {pair_index: {pid: 1.0 chose A | 0.0 chose B | 0.5 tie}}"""
    per_pair: dict[int, dict[str, float]] = defaultdict(dict)
    for results, meta_path in panels:
        by_id = {m["pairId"]: m for m in json.loads(Path(meta_path).read_text())["pairs"]}
        for line in (Path(results) / "judgments.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            j = json.loads(line)
            if j["studyId"] == "test" or j["isGold"] or j["prolificPid"] in rejects:
                continue
            m = by_id.get(j["pairId"])
            if m is None:
                continue
            a_side = "left" if m["faceAOnLeft"] else "right"
            v = 0.5 if j["choice"] == "tie" else (1.0 if j["choice"] == a_side else 0.0)
            per_pair[m["pairIndex"]][j["prolificPid"]] = v
    return per_pair


def predict_from(per_pair, raters: set[str]) -> dict[int, float]:
    """A rater group's per-pair vote share — the group's collective opinion."""
    out = {}
    for idx, votes in per_pair.items():
        vals = [v for pid, v in votes.items() if pid in raters]
        if vals:
            out[idx] = sum(vals) / len(vals)
    return out


def score(pred: dict[int, float], per_pair, test: set[str]) -> tuple[float, int]:
    """Share of the test raters' individual votes matching the predictor's preferred face."""
    hit = tot = 0.0
    for idx, votes in per_pair.items():
        p = pred.get(idx)
        if p is None or p == 0.5:
            continue
        for pid, v in votes.items():
            if pid not in test or v == 0.5:
                continue
            tot += 1
            hit += 1.0 if (v > 0.5) == (p > 0.5) else 0.0
    return (hit / tot if tot else float("nan")), int(tot)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", nargs=2, action="append", required=True,
                    metavar=("RESULTS", "META"))
    ap.add_argument("--demographics", nargs="*", default=None,
                    help="defaults to <results>/demographics.csv for each --panel")
    ap.add_argument("--rejects", nargs="*", default=[])
    ap.add_argument("--seeds", type=int, default=25)
    ap.add_argument("--min-cohort", type=int, default=40,
                    help="skip cohorts smaller than this; they cannot be split and tested")
    ap.add_argument("--out")
    args = ap.parse_args()

    panels = [tuple(p) for p in args.panel]
    demo_paths = args.demographics or [str(Path(r) / "demographics.csv") for r, _ in panels]
    missing = [p for p in demo_paths if not Path(p).exists()]
    if missing:
        raise SystemExit(f"missing demographic export(s): {missing}")

    rejects = load_rejects(args.rejects)
    people = load_demographics(demo_paths)
    per_pair = load_judgments(panels, rejects)
    voters = {pid for votes in per_pair.values() for pid in votes}
    known = voters & set(people)
    print(f"raters with votes in the fit: {len(voters):,}")
    print(f"  matched to a demographic record: {len(known):,} "
          f"({len(known) / len(voters):.1%})")
    print(f"pairs judged: {len(per_pair):,}")

    report: dict = {"raters": len(voters), "matched": len(known), "attributes": {}}

    for attr in ATTRS:
        groups: dict[str, set[str]] = defaultdict(set)
        for pid in known:
            val = people[pid].get(attr)
            if val not in MISSING and val != "unknown":
                groups[val].add(pid)
        if not groups:
            continue
        total = sum(len(v) for v in groups.values())
        print(f"\n=== {attr} ===")
        print(f"{'cohort':28} {'raters':>7} {'share':>7} {'votes':>9}")
        sizes = {}
        for val, pids in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            nv = sum(1 for votes in per_pair.values() for pid in votes if pid in pids)
            sizes[val] = {"raters": len(pids), "share": len(pids) / total, "votes": nv}
            print(f"{val[:27]:28} {len(pids):7,} {len(pids) / total:6.1%} {nv:9,}")

        # Matched in-group vs out-group test, per cohort large enough to split.
        rows = []
        for val, pids in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            if len(pids) < args.min_cohort:
                continue
            others = set().union(*[p for k, p in groups.items() if k != val]) - pids
            ins, outs = [], []
            for seed in range(args.seeds):
                rng = random.Random(1000 + seed)
                members = sorted(pids)
                rng.shuffle(members)
                half = len(members) // 2
                fit, test = set(members[:half]), set(members[half:])
                if not fit or not test or len(others) < len(fit):
                    continue
                out_fit = set(rng.sample(sorted(others), len(fit)))
                a, na = score(predict_from(per_pair, fit), per_pair, test)
                b, nb = score(predict_from(per_pair, out_fit), per_pair, test)
                if na and nb:
                    ins.append(a)
                    outs.append(b)
            if not ins:
                continue
            gap = st.mean(ins) - st.mean(outs)
            sd = st.pstdev([i - o for i, o in zip(ins, outs)]) or 1e-9
            rows.append({"cohort": val, "raters": len(pids), "inGroup": st.mean(ins),
                         "outGroup": st.mean(outs), "gap": gap, "gapSd": sd})
        if rows:
            print("\n  matched test — does a cohort predict its own members better?")
            print(f"  {'cohort':22} {'in-group':>9} {'out-group':>10} {'gap':>8} {'+/-':>7}")
            for r in rows:
                print(f"  {r['cohort'][:21]:22} {r['inGroup']:8.2%} {r['outGroup']:9.2%} "
                      f"{r['gap'] * 100:+7.2f} {r['gapSd'] * 100:6.2f}")
            best = max(rows, key=lambda r: r["gap"])
            print(f"  largest in-group advantage: {best['cohort']} at "
                  f"{best['gap'] * 100:+.2f} points "
                  f"({best['gap'] / best['gapSd']:+.1f} sd)")
        report["attributes"][attr] = {"cohorts": sizes, "matchedTest": rows}

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
