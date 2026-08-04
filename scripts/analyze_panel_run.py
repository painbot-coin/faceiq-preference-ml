#!/usr/bin/env python
"""Analyse the full panel pilot run: rater QC, gold health, and the Step 6 gates.

Two jobs, deliberately separated because they pull in opposite directions:

**Rater QC** decides who to pay. Gold pass rate alone is *not* sufficient — with
only 5 golds per rater, an honest rater at the task's natural ~85% accuracy fails
3+ of 5 about 2.7% of the time, so on 360 raters you expect ~10 people to look
guilty by luck. Rejecting on golds alone would punish them. So we cross-check
against agreement-with-the-other-raters (leave-one-out majority over that rater's
own 100 pairs — 20x more evidence than 5 golds) plus straight-lining and speed.
A rater is only a reject candidate when *independent* signals agree.

**Gate analysis** decides whether the VLM labels survive. Majority vote per pair,
split-half stability, Gemini-vs-majority by bucket, vote entropy.

Usage:
    python scripts/analyze_panel_run.py
    python scripts/analyze_panel_run.py --results labels/panel-pilot/results
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TEST_STUDY_ID = "test"
FAST_CLICK_MS = 2000
# Gold screen. 2 of 5 is the documented reject line, but see the module docstring:
# it is evidence, not a verdict.
GOLD_FAIL_AT = 0.5
# Straight-lining: one key pressed for essentially the whole session.
STRAIGHT_LINE_AT = 0.90
# Below this median response time a human is not looking at two photographs.
IMPOSSIBLE_MS = 700
# An abandoned session was never submitted, so there is no payment to reject — only
# votes to drop. Below this, treat it as a dropout rather than misconduct.
MIN_JUDGMENTS_TO_REJECT = 50


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def pct(num: float, den: float) -> str:
    return f"{100 * num / den:.0f}%" if den else "n/a"


def forced_majority(choices: list[str]) -> str | None:
    """Majority over left/right, ignoring ties. None when undetermined."""
    left = choices.count("left")
    right = choices.count("right")
    if left > right:
        return "left"
    if right > left:
        return "right"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="labels/panel-pilot/results")
    ap.add_argument("--panel-dir", default="labels/panel-pilot")
    ap.add_argument("--out", default="artifacts/panel-run-v1")
    ap.add_argument("--seed", type=int, default=20260730)
    args = ap.parse_args()

    results = Path(args.results)
    panel = Path(args.panel_dir)
    judgments = read_jsonl(results / "judgments.jsonl")
    sessions = read_jsonl(results / "sessions.jsonl")
    expected: dict[str, str] = json.loads((panel / "golds.json").read_text())["expected"]
    meta = {m["pairId"]: m for m in json.loads((panel / "sample-meta.json").read_text())["pairs"]}
    rng = random.Random(args.seed)

    judgments = [j for j in judgments if j["studyId"] != TEST_STUDY_ID]
    sessions = [s for s in sessions if s["studyId"] != TEST_STUDY_ID]

    # --- integrity -----------------------------------------------------------
    print("== integrity ==")
    print(f"{len(sessions)} sessions, {len(judgments)} judgments")
    ids = {j["judgmentId"] for j in judgments}
    print(f"judgment ids unique: {len(ids) == len(judgments)}")
    by_session = Counter(j["sessionId"] for j in judgments)
    print(f"sum of per-session counts matches: "
          f"{sum(by_session.values()) == len(judgments)}")
    pids = Counter(s["prolificPid"] for s in sessions)
    repeats = {p: n for p, n in pids.items() if n > 1}
    print(f"{len(pids)} distinct raters; {len(repeats)} with >1 session")
    complete = [s for s in sessions if s["completedAt"]]
    print(f"{len(complete)} completed, {len(sessions) - len(complete)} abandoned "
          f"(their votes are still real and are kept)")
    unknown = {j["pairId"] for j in judgments} - set(meta)
    print(f"judgments on pairs absent from sample-meta: {len(unknown)}")

    # --- duration ------------------------------------------------------------
    mins = sorted((parse_ts(s["completedAt"]) - parse_ts(s["startedAt"])).total_seconds() / 60
                  for s in complete)
    q = statistics.quantiles(mins, n=4)
    print("\n== session length (rating only, excludes instructions) ==")
    print(f"median {statistics.median(mins):.1f} min   p25 {q[0]:.1f}   p75 {q[2]:.1f}   "
          f"range {mins[0]:.1f}-{mins[-1]:.1f}")
    rts = [j["responseTimeMs"] for j in judgments]
    print(f"per-judgment median {statistics.median(rts) / 1000:.1f}s   "
          f"under {FAST_CLICK_MS}ms: {pct(sum(t < FAST_CLICK_MS for t in rts), len(rts))}")

    # --- votes per pair ------------------------------------------------------
    votes: dict[str, list[dict]] = defaultdict(list)
    for j in judgments:
        if not j["isGold"]:
            votes[j["pairId"]].append(j)
    hist = Counter(len(v) for v in votes.values())
    print("\n== coverage ==")
    print(f"{len(votes)} real pairs judged; votes-per-pair: {dict(sorted(hist.items()))}")

    # --- leave-one-out agreement, per rater ---------------------------------
    # The workhorse quality signal: ~100 comparisons per rater instead of 5.
    choices_by_pair = {p: [j["choice"] for j in v] for p, v in votes.items()}
    agree: dict[str, list[int]] = defaultdict(list)
    for pair_id, rows in votes.items():
        all_choices = choices_by_pair[pair_id]
        for j in rows:
            others = list(all_choices)
            others.remove(j["choice"])
            maj = forced_majority(others)
            if maj is None or j["choice"] == "tie":
                continue
            agree[j["prolificPid"]].append(1 if j["choice"] == maj else 0)

    by_rater: dict[str, list[dict]] = defaultdict(list)
    for j in judgments:
        by_rater[j["prolificPid"]].append(j)

    rows = []
    for pid, rs in by_rater.items():
        golds = [j for j in rs if j["isGold"]]
        hit = sum(j["choice"] == expected.get(j["pairId"]) for j in golds)
        a = agree.get(pid, [])
        ch = Counter(j["choice"] for j in rs)
        t = [j["responseTimeMs"] for j in rs]
        rows.append({
            "prolificPid": pid,
            "n": len(rs),
            "goldsSeen": len(golds),
            "goldsHit": hit,
            "goldRate": hit / len(golds) if golds else None,
            "looAgree": sum(a) / len(a) if a else None,
            "looN": len(a),
            "medianMs": int(statistics.median(t)),
            "fastRate": sum(x < FAST_CLICK_MS for x in t) / len(t),
            "tieRate": ch["tie"] / len(rs),
            "topChoiceShare": max(ch.values()) / len(rs),
        })

    pop = [r["looAgree"] for r in rows if r["looAgree"] is not None]
    mu, sd = statistics.mean(pop), statistics.pstdev(pop)
    floor = mu - 2 * sd
    print("\n== rater agreement with the other 11 ==")
    print(f"mean {mu:.1%}  sd {sd:.1%}  -> flag below mu-2sd = {floor:.1%}")
    print(f"deciles: {[f'{statistics.quantiles(pop, n=10)[i]:.0%}' for i in range(9)]}")

    # Reject only where independent signals agree. Any single one of these is
    # noise on its own; two together is a pattern.
    for r in rows:
        why = []
        if r["goldRate"] is not None and r["goldRate"] <= GOLD_FAIL_AT:
            why.append("golds")
        if r["looAgree"] is not None and r["looAgree"] < floor:
            why.append("agreement")
        # Needs a real session behind it: on n=1 a single click is trivially 100%.
        if r["n"] >= 20 and r["topChoiceShare"] >= STRAIGHT_LINE_AT:
            why.append("straight-lining")
        if r["n"] >= 20 and r["medianMs"] < IMPOSSIBLE_MS:
            why.append("impossible-speed")
        r["signals"] = why
        r["verdict"] = ("REJECT" if len(why) >= 2 else "watch" if why else "ok")

    rejects = sorted([r for r in rows if r["verdict"] == "REJECT"],
                     key=lambda r: r["looAgree"] or 0)
    watch = [r for r in rows if r["verdict"] == "watch"]

    print(f"\n== verdicts ==  {len(rejects)} reject, {len(watch)} watch, "
          f"{len(rows) - len(rejects) - len(watch)} clean")
    if rejects:
        print(f"\n{'prolificPid':26} {'n':>4} {'gold':>6} {'agree':>6} {'medMs':>6} "
              f"{'fast':>5} {'tie':>5} {'top1':>5}  signals")
        for r in rejects:
            print(f"{r['prolificPid']:26} {r['n']:4} {r['goldsHit']:2}/{r['goldsSeen']:<3} "
                  f"{r['looAgree']:6.0%} {r['medianMs']:6} {r['fastRate']:5.0%} "
                  f"{r['tieRate']:5.0%} {r['topChoiceShare']:5.0%}  {','.join(r['signals'])}")

    print("\n-- 'watch' (one signal only — approve, do not reject) --")
    print(f"{'prolificPid':26} {'n':>4} {'gold':>6} {'agree':>6} {'medMs':>6}  signal")
    for r in sorted(watch, key=lambda r: r["looAgree"] if r["looAgree"] is not None else 0):
        a = f"{r['looAgree']:6.0%}" if r["looAgree"] is not None else "   n/a"
        print(f"{r['prolificPid']:26} {r['n']:4} {r['goldsHit']:2}/{r['goldsSeen']:<3} "
              f"{a} {r['medianMs']:6}  {','.join(r['signals'])}")

    # Does failing golds actually predict bad judgment? If the two signals are
    # unrelated, the gold screen is measuring luck, not quality.
    gf = [r["looAgree"] for r in rows
          if r["goldRate"] is not None and r["goldRate"] <= GOLD_FAIL_AT and r["looAgree"]]
    gp = [r["looAgree"] for r in rows
          if r["goldRate"] is not None and r["goldRate"] > GOLD_FAIL_AT and r["looAgree"]]
    print("\n== does the gold screen track real quality? ==")
    print(f"raters failing golds (<={GOLD_FAIL_AT:.0%}): n={len(gf)}, "
          f"mean agreement {statistics.mean(gf):.1%}" if gf else "no gold failures")
    print(f"raters passing golds:            n={len(gp)}, "
          f"mean agreement {statistics.mean(gp):.1%}")

    # --- gold health ---------------------------------------------------------
    gold_votes: dict[str, list[str]] = defaultdict(list)
    for j in judgments:
        if j["isGold"]:
            gold_votes[j["pairId"]].append(j["choice"])
    print(f"\n== golds ({len(expected)} shipped, ~{len(judgments) // 20 // 20 * 20}+ views each) ==")
    print(f"{'pairId':28} {'want':>5} {'seen':>5} {'pass':>5}  distribution")
    bad_golds = []
    for pair_id, chs in sorted(gold_votes.items(), key=lambda kv: sum(
            c == expected[kv[0]] for c in kv[1]) / len(kv[1])):
        want = expected[pair_id]
        hit = sum(c == want for c in chs)
        rate = hit / len(chs)
        mark = "RETIRE" if rate < 0.7 else ""
        if mark:
            bad_golds.append(pair_id)
        print(f"{pair_id:28} {want:>5} {len(chs):5} {rate:5.0%}  "
              f"{dict(Counter(chs))} {mark}")

    # --- Step 6: majority labels, stability, Gemini agreement ---------------
    labels = {}
    for pair_id, chs in choices_by_pair.items():
        maj = forced_majority(chs)
        c = Counter(chs)
        labels[pair_id] = {
            "pairId": pair_id,
            "votes": len(chs),
            "left": c["left"], "right": c["right"], "tie": c["tie"],
            "majority": maj,
            "majorityShare": (max(c["left"], c["right"]) / len(chs)) if maj else None,
            "stratum": meta[pair_id]["stratum"],
            "bucket": meta[pair_id]["bucket"],
            "winProb": meta[pair_id]["winProb"],
            "vlmChoice": (
                ("left" if meta[pair_id]["faceAOnLeft"] else "right")
                if meta[pair_id]["vlmOutcome"] == "A"
                else ("right" if meta[pair_id]["faceAOnLeft"] else "left")
            ),
            "vlmConfidence": meta[pair_id]["vlmConfidence"],
            "btChoice": (
                ("left" if meta[pair_id]["faceAOnLeft"] else "right")
                if meta[pair_id]["thetaA"] > meta[pair_id]["thetaB"]
                else ("right" if meta[pair_id]["faceAOnLeft"] else "left")
            ),
        }

    determined = [v for v in labels.values() if v["majority"]]
    print("\n== majority labels ==")
    print(f"{len(determined)} of {len(labels)} pairs have a determined majority "
          f"({len(labels) - len(determined)} dead splits)")
    print(f"mean majority share {statistics.mean(v['majorityShare'] for v in determined):.1%}")
    print(f"unanimous {pct(sum(v['majorityShare'] == 1 for v in determined), len(determined))}")
    print(f"tie votes {pct(sum(v['tie'] for v in labels.values()), sum(v['votes'] for v in labels.values()))}")

    def split_half(chs: list[str], trials: int = 25) -> float:
        ok = tot = 0
        for _ in range(trials):
            s = rng.sample(chs, len(chs))
            a, b = forced_majority(s[: len(s) // 2]), forced_majority(s[len(s) // 2:])
            if a and b:
                tot += 1
                ok += a == b
        return ok / tot if tot else 0.0

    # Human ceiling per stratum: the fair yardstick for Gemini. "Is the VLM worse
    # than a person?" only means something against how well people do here.
    h_by_stratum: dict[str, list[int]] = defaultdict(list)
    for pair_id, rows_ in votes.items():
        st = meta[pair_id]["stratum"]
        all_choices = choices_by_pair[pair_id]
        for j in rows_:
            others = list(all_choices)
            others.remove(j["choice"])
            maj = forced_majority(others)
            if maj is None or j["choice"] == "tie":
                continue
            h_by_stratum[st].append(1 if j["choice"] == maj else 0)

    print("\n== Step 6 gates, by stratum ==")
    print(f"{'stratum':10} {'pairs':>6} {'split-half':>11} {'human H':>9} "
          f"{'Gemini G':>9} {'G-H':>7} {'BT theta':>9} {'maj share':>10} {'splits':>7}")
    preferred = ["close", "mid", "clear", "lowconf", "overlap"]
    present = {v["stratum"] for v in labels.values()}
    for stratum in preferred + sorted(present - set(preferred)):
        grp = [v for v in labels.values() if v["stratum"] == stratum]
        if not grp:
            continue
        det = [v for v in grp if v["majority"]]
        sh = statistics.mean(split_half(choices_by_pair[v["pairId"]]) for v in grp)
        g = sum(v["vlmChoice"] == v["majority"] for v in det) / len(det)
        h = statistics.mean(h_by_stratum[stratum])
        # Does the existing BT fit (built on VLM labels) predict the human majority?
        # Direction comes from theta, not winProb: winProb is stored orientation-free
        # (always >= 0.5, it is the gap magnitude used to stratify the draw).
        bt = sum(v["btChoice"] == v["majority"] for v in det) / len(det)
        print(f"{stratum:10} {len(grp):6} {sh:11.1%} {h:9.1%} {g:9.1%} "
              f"{g - h:+7.1%} {bt:11.1%} "
              f"{statistics.mean(v['majorityShare'] for v in det):10.1%} "
              f"{pct(len(grp) - len(det), len(grp)):>7}")
    allsh = statistics.mean(split_half(choices_by_pair[v["pairId"]]) for v in labels.values())
    allg = sum(v["vlmChoice"] == v["majority"] for v in determined) / len(determined)
    allh = statistics.mean(x for v in h_by_stratum.values() for x in v)
    allbt = sum(v["btChoice"] == v["majority"] for v in determined) / len(determined)
    print(f"{'ALL':10} {len(labels):6} {allsh:11.1%} {allh:9.1%} {allg:9.1%} "
          f"{allg - allh:+7.1%} {allbt:11.1%}")

    # Retiring the three worst golds: does the reject list change? If the screen
    # is manufacturing failures, this is where it shows.
    bad = set(bad_golds)
    if bad:
        still = 0
        rescued = 0
        for pid, rs in by_rater.items():
            g2 = [j for j in rs if j["isGold"] and j["pairId"] not in bad]
            if not g2:
                continue
            rate = sum(j["choice"] == expected[j["pairId"]] for j in g2) / len(g2)
            was = next(r for r in rows if r["prolificPid"] == pid)
            if was["goldRate"] is not None and was["goldRate"] <= GOLD_FAIL_AT:
                if rate <= GOLD_FAIL_AT:
                    still += 1
                else:
                    rescued += 1
        print(f"\n== if the {len(bad)} worst golds are retired ==")
        print(f"of the {still + rescued} raters who failed the gold screen, "
              f"{rescued} pass without them and {still} still fail")

    print("\n== Gemini agreement by its own confidence ==")
    for conf in ["high", "medium", "low"]:
        grp = [v for v in determined if v["vlmConfidence"] == conf]
        if grp:
            g = sum(v["vlmChoice"] == v["majority"] for v in grp) / len(grp)
            print(f"{conf:8} n={len(grp):5}  agrees {g:.1%}")

    # Human ceiling H: how often does one human match the other 11?
    print("\n== human ceiling H (leave-one-out, all raters) ==")
    flat = [x for a in agree.values() for x in a]
    print(f"{sum(flat) / len(flat):.1%} over {len(flat)} comparisons")

    # --- outputs -------------------------------------------------------------
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "majority-labels.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(next(iter(labels.values()))))
        w.writeheader()
        w.writerows(labels.values())
    with (out / "raters.csv").open("w", newline="") as f:
        fields = [k for k in rows[0] if k != "signals"] + ["signals"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({**r, "signals": ";".join(r["signals"])})
    (out / "reject-pids.txt").write_text(
        "".join(f"{r['prolificPid']}\n" for r in rejects))
    # Two different decisions, so two lists. reject-pids.txt drops votes from the fit
    # and costs the rater nothing. A Prolific rejection is permanent and appealable, so
    # it needs behavioural proof: golds and low agreement can both be bad luck or
    # unusual taste, but a median under IMPOSSIBLE_MS or one button all session cannot.
    payment = [r for r in rejects
               if {"impossible-speed", "straight-lining"} & set(r["signals"])
               and r["n"] >= MIN_JUDGMENTS_TO_REJECT]
    (out / "prolific-rejections.txt").write_text(
        "".join(f"{r['prolificPid']}\n" for r in payment))
    print(f"\nwrote {out}/majority-labels.csv, raters.csv, reject-pids.txt "
          f"({len(rejects)} dropped from the fit), prolific-rejections.txt "
          f"({len(payment)} defensible on appeal)")
    for r in rejects:
        if r in payment:
            note = "reject on Prolific"
        elif r["n"] < MIN_JUDGMENTS_TO_REJECT:
            note = "dropout — no submission to action, votes dropped"
        else:
            note = ("PAY — exclude the votes, but the evidence is statistical, "
                    "not behavioural")
        print(f"  {r['prolificPid']}  {','.join(r['signals']):45} -> {note}")

    fb = [s for s in sessions if s.get("feedback")]
    print(f"\n== {len(fb)} feedback messages ==")
    for s in fb:
        print(f"  [{s['prolificPid']}] {s['feedback']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
