#!/usr/bin/env python
"""Does human agreement rise with the ranking's percentile gap, on an unbiased pair sample?

This is the load-bearing assumption behind every spend decision in the programme, and until
run 4 it had only ever been tested on pairs *selected* to be near-ties. That selection makes
the test circular in a specific way: the strata were cut on a ranking derived from VLM labels,
so "close pairs are contested" could have been an artefact of how the draw was built rather
than a fact about faces.

Run 4 removes the selection. Pairs were drawn uniformly at random over the unbought export,
bands were recorded and never imposed, and the human votes are the first data the ranking has
never seen. So the three curves below are a genuine out-of-sample test of the ordering:

* `decisiveness`  mean |vote share - 0.5|  — how far the crowd is from a coin flip
* `maj share`     mean share held by the winning side
* `splitHalf`     do two disjoint halves of raters agree on the winner
* `btAgrees`      does the higher-theta face win the popular vote

A monotone rise means the ranking's *distance* is real and the /10 gap can be quoted as a
confidence statement. A flat curve would mean the percentile ladder is decorative.

Usage:
    python scripts/gap_agreement_curve.py \
        --panel labels/panel-run-4-random/results labels/panel-run-4-random/sample-meta.json \
        --rejects artifacts/panel-run-v4/reject-pids.txt \
        --out artifacts/panel-run-v4/gap-curve.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

BANDS = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 45), (45, 101)]


def spearman(xs: list[float], ys: list[float]) -> float:
    def rank(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        out = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def load(panels, rejects):
    """-> [{gap, votesA, votesB, ballots, btPicksA}] one row per pair with >=4 votes."""
    votes: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    ballots: dict[str, list[tuple[str, bool]]] = defaultdict(list)
    meta_all: dict[str, dict] = {}
    for results_dir, meta_path in panels:
        meta = json.loads(Path(meta_path).read_text())["pairs"]
        by_id = {m["pairId"]: m for m in meta}
        for line in (Path(results_dir) / "judgments.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            j = json.loads(line)
            if j["studyId"] == "test" or j["isGold"] or j["prolificPid"] in rejects:
                continue
            m = by_id.get(j["pairId"])
            if m is None or m.get("percentileGap") is None:
                continue
            meta_all[j["pairId"]] = m
            a_side = "left" if m["faceAOnLeft"] else "right"
            cell = votes[j["pairId"]]
            if j["choice"] == "tie":
                cell[0] += 0.5
                cell[1] += 0.5
            elif j["choice"] == a_side:
                cell[0] += 1.0
                ballots[j["pairId"]].append((j["prolificPid"], True))
            else:
                cell[1] += 1.0
                ballots[j["pairId"]].append((j["prolificPid"], False))
    rows = []
    for pid, (wa, wb) in votes.items():
        if wa + wb < 4:
            continue
        m = meta_all[pid]
        rows.append({
            "pairId": pid,
            "gap": float(m["percentileGap"]),
            "votesA": wa, "votesB": wb,
            "share": wa / (wa + wb),
            "btPicksA": m["thetaA"] > m["thetaB"],
            "ballots": ballots[pid],
        })
    return rows


def split_half(rows, seed: int, draws: int = 200) -> float:
    """Share of pairs where two disjoint rater halves name the same winner."""
    rng = random.Random(seed)
    hits = tot = 0
    for _ in range(draws):
        for r in rows:
            b = r["ballots"]
            if len(b) < 4:
                continue
            idx = list(range(len(b)))
            rng.shuffle(idx)
            half = len(idx) // 2
            s1 = sum(1 if b[i][1] else -1 for i in idx[:half])
            s2 = sum(1 if b[i][1] else -1 for i in idx[half:2 * half])
            if s1 == 0 or s2 == 0:
                continue
            hits += (s1 > 0) == (s2 > 0)
            tot += 1
    return hits / tot if tot else float("nan")


def coin_flip_null(rows, seed: int = 3, draws: int = 40):
    """What decisiveness and majority share look like if every pair were a true 50/50.

    Load-bearing for reading the table: with 12 ballots, sampling noise alone produces a
    majority share near 61%, so a band sitting at 68% is *not* "barely better than a coin
    flip" — it is 7 points of real signal. Re-simulated at each pair's own ballot count so
    the null matches the observed coverage rather than an idealised 12.
    """
    rng = random.Random(seed)
    dec = maj = n = 0.0
    for _ in range(draws):
        for r in rows:
            k = int(round(r["votesA"] + r["votesB"]))
            if k < 4:
                continue
            wa = sum(rng.random() < 0.5 for _ in range(k))
            share = wa / k
            dec += abs(share - 0.5)
            maj += max(share, 1 - share)
            n += 1
    return (dec / n, maj / n) if n else (float("nan"), float("nan"))


def boot_ci(vals: list[float], draws: int = 2000, seed: int = 7):
    if not vals:
        return [float("nan")] * 2
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(vals) for _ in vals) / len(vals) for _ in range(draws))
    return [means[int(0.025 * draws)], means[int(0.975 * draws)]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", nargs=2, action="append", required=True,
                    metavar=("RESULTS", "META"))
    ap.add_argument("--rejects", nargs="*", default=[])
    ap.add_argument("--out")
    args = ap.parse_args()

    rejects: set[str] = set()
    for p in args.rejects:
        if Path(p).exists():
            rejects |= set(Path(p).read_text().split())

    rows = load([tuple(p) for p in args.panel], rejects)
    print(f"{len(rows):,} pairs with >=4 votes, "
          f"{sum(r['votesA'] + r['votesB'] for r in rows):,.0f} votes")

    gaps = [r["gap"] for r in rows]
    dec = [abs(r["share"] - 0.5) for r in rows]
    maj = [max(r["share"], 1 - r["share"]) for r in rows]
    null_dec, null_maj = coin_flip_null(rows)
    print("\nover all pairs, no binning:")
    print(f"  spearman(percentile gap, decisiveness)  = {spearman(gaps, dec):+.3f}")
    print(f"  spearman(percentile gap, winner share)  = {spearman(gaps, maj):+.3f}")
    print(f"  coin-flip null at these ballot counts   = decisiveness {null_dec:.3f}, "
          f"majority share {null_maj:.1%}")

    print(f"\n{'band':>9} {'pairs':>6} {'decisive':>9} {'maj share':>19} "
          f"{'split-half':>11} {'BT agrees':>10}")
    out_bands = []
    for lo, hi in BANDS:
        sel = [r for r in rows if lo <= r["gap"] < hi]
        if not sel:
            continue
        d = sum(abs(r["share"] - 0.5) for r in sel) / len(sel)
        mvals = [max(r["share"], 1 - r["share"]) for r in sel]
        m = sum(mvals) / len(mvals)
        ci = boot_ci(mvals)
        sh = split_half(sel, seed=lo + 1)
        bt_hit = sum((r["votesA"] if r["btPicksA"] else r["votesB"]) for r in sel)
        bt_tot = sum(r["votesA"] + r["votesB"] for r in sel)
        out_bands.append({"band": f"{lo}-{min(hi, 100)}", "pairs": len(sel),
                          "decisiveness": d, "majorityShare": m, "majorityShareCI": ci,
                          "splitHalf": sh, "btAgrees": bt_hit / bt_tot})
        print(f"{lo}-{min(hi, 100):<7} {len(sel):6} {d:9.3f} "
              f"{m:8.1%} [{ci[0]:.1%},{ci[1]:.1%}] {sh:10.1%} {bt_hit / bt_tot:9.1%}")

    monotone = all(a["majorityShare"] <= b["majorityShare"]
                   for a, b in zip(out_bands, out_bands[1:]))
    bt_monotone = all(a["btAgrees"] <= b["btAgrees"]
                      for a, b in zip(out_bands, out_bands[1:]))
    print(f"\nmajority share monotone across bands: {monotone}")
    print(f"BT agreement monotone across bands:   {bt_monotone}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "pairs": len(rows),
            "spearmanGapDecisiveness": spearman(gaps, dec),
            "spearmanGapWinnerShare": spearman(gaps, maj),
            "coinFlipNull": {"decisiveness": null_dec, "majorityShare": null_maj},
            "bands": out_bands,
            "majorityShareMonotone": monotone,
            "btAgreementMonotone": bt_monotone,
        }, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
