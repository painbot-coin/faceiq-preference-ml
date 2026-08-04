#!/usr/bin/env python
"""Does more panel spend keep paying? Measures the two curves separately.

The $715 decision and the $15,300 decision are not the same question:

  * **depth**  — more votes on pairs we already cover (12 -> 25 votes/pair)
  * **breadth** — votes on pairs we do not cover at all (1,050 -> 11,712 pairs)

Both are measured against a set of panel pairs held out *entirely* — none of their
votes enter any fit — so improvement can only come from better theta on the faces
involved, which is precisely the mechanism that would have to carry a bigger buy.

A flat curve means the next dollar buys nothing. Run this before spending.

Usage:
    python scripts/panel_value_curve.py --export data/exports/<runId>
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from faceiq_pref.bt_panel import build_cells, fit_joint_bt  # noqa: E402
from faceiq_pref.data import load_export  # noqa: E402

GENDERS = ("female", "male")


def load_judgments(results: Path, rejects: Path):
    rej = set(rejects.read_text().split()) if rejects.exists() else set()
    out = []
    for line in (results / "judgments.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        j = json.loads(line)
        if j["studyId"] == "test" or j["isGold"] or j["prolificPid"] in rej:
            continue
        out.append(j)
    return out


def tally(judgments, meta_by_id, pair_filter=None, cap=None, seed=0):
    """-> {pair_index: (wins A, wins B)}; `cap` keeps at most N votes per pair."""
    grouped: dict[int, list] = defaultdict(list)
    for j in judgments:
        m = meta_by_id.get(j["pairId"])
        if m is None:
            continue
        if pair_filter is not None and m["pairIndex"] not in pair_filter:
            continue
        grouped[m["pairIndex"]].append((j, m))
    rng = random.Random(seed)
    votes: dict[int, tuple[float, float]] = {}
    for idx, items in grouped.items():
        if cap is not None and len(items) > cap:
            items = rng.sample(items, cap)
        wa = wb = 0.0
        for j, m in items:
            a_side = "left" if m["faceAOnLeft"] else "right"
            if j["choice"] == "tie":
                wa += 0.5
                wb += 0.5
            elif j["choice"] == a_side:
                wa += 1.0
            else:
                wb += 1.0
        votes[idx] = (wa, wb)
    return votes


def fit(matchups, votes):
    theta = {}
    for g in GENDERS:
        fi, ce, co, ob, nv, nh, df, dm = build_cells(matchups, votes, g)
        theta.update(fit_joint_bt(fi, ce, co, ob, g, nv, nh, df, dm).theta)
    return theta


def score(theta, eval_votes, by_index):
    correct = total = 0.0
    for idx, (wa, wb) in eval_votes.items():
        m = by_index.get(idx)
        if m is None or m.face_a_id not in theta or m.face_b_id not in theta:
            continue
        d = theta[m.face_a_id] - theta[m.face_b_id]
        if d == 0:
            continue
        correct += wa if d > 0 else wb
        total += wa + wb
    return correct / total if total else float("nan"), total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--panel-results", default="labels/panel-pilot/results")
    ap.add_argument("--panel-meta", default="labels/panel-pilot/sample-meta.json")
    ap.add_argument("--rejects", default="artifacts/panel-run-v1/reject-pids.txt")
    ap.add_argument("--exclude-faces", default="artifacts/face-qc-v1/exclude-faces.csv")
    ap.add_argument("--exclude-genders", default="artifacts/face-qc-v1/gender-fixes.csv")
    ap.add_argument("--holdout-pairs", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--seeds", type=int, default=5, help="repeats for the equal-budget test")
    ap.add_argument("--out", default="artifacts/panel-value-v1")
    args = ap.parse_args()

    export = load_export(args.export)
    matchups = export.all_matchups()
    dropped: set[str] = set()
    for p in (args.exclude_faces, args.exclude_genders):
        if p:
            dropped.update(r["faceId"] for r in csv.DictReader(open(p)))
    matchups = [m for m in matchups
                if m.face_a_id not in dropped and m.face_b_id not in dropped]
    by_index = {m.pair_index: m for m in matchups}

    meta = json.loads(Path(args.panel_meta).read_text())["pairs"]
    meta_by_id = {m["pairId"]: m for m in meta}
    strat = {m["pairIndex"]: m["stratum"] for m in meta}
    judgments = load_judgments(Path(args.panel_results), Path(args.rejects))
    print(f"{len(judgments):,} usable panel judgments")

    all_idx = sorted({m["pairIndex"] for m in meta if m["pairIndex"] in by_index})
    rng = random.Random(args.seed)
    shuffled = all_idx[:]
    rng.shuffle(shuffled)
    n_hold = int(len(shuffled) * args.holdout_pairs)
    hold = set(shuffled[:n_hold])
    pool = shuffled[n_hold:]

    eval_votes = tally(judgments, meta_by_id, pair_filter=hold)
    eval_close = {k: v for k, v in eval_votes.items() if strat.get(k) == "close"}
    n_eval = sum(a + b for a, b in eval_votes.values())
    print(f"held-out pairs: {len(hold)} ({n_eval:,.0f} votes, none used in any fit)")
    print(f"  of which close-stratum: {len(eval_close)}")
    print(f"training pool: {len(pool)} pairs\n")

    results: list[dict] = []

    def run(label, votes, n_pairs, votes_per_pair):
        theta = fit(matchups, votes)
        acc, tot = score(theta, eval_votes, by_index)
        acc_c, tot_c = score(theta, eval_close, by_index)
        spend = sum(a + b for a, b in votes.values()) * 1900 / 36290
        results.append({"label": label, "pairsWithVotes": n_pairs,
                        "votesPerPair": votes_per_pair,
                        "totalVotes": sum(a + b for a, b in votes.values()),
                        "spendEquivalent": spend,
                        "heldOutAccuracy": acc, "heldOutClose": acc_c})
        print(f"  {label:34} {n_pairs:5} pairs  {acc:7.2%}  {acc_c:7.2%}  "
              f"${spend:7,.0f}")

    print("BREADTH — more pairs covered, all votes on each")
    print(f"  {'config':34} {'pairs':>5}  {'all':>7}  {'close':>7}  {'spend':>8}")
    run("no panel votes (VLM only)", {}, 0, 0)
    for frac in (0.25, 0.5, 0.75, 1.0):
        take = set(pool[: int(len(pool) * frac)])
        v = tally(judgments, meta_by_id, pair_filter=take, seed=args.seed)
        vpp = np.mean([a + b for a, b in v.values()]) if v else 0
        run(f"{frac:.0%} of pool ({len(take)} pairs)", v, len(take), float(vpp))

    print("\nDEPTH — every pool pair, capped votes per pair")
    print(f"  {'config':34} {'pairs':>5}  {'all':>7}  {'close':>7}  {'spend':>8}")
    full = set(pool)
    for cap in (2, 4, 6, 8, 10, 12, None):
        v = tally(judgments, meta_by_id, pair_filter=full, cap=cap, seed=args.seed)
        vpp = np.mean([a + b for a, b in v.values()])
        run(f"cap {cap if cap else 'none'} votes/pair", v, len(full), float(vpp))

    # ---- the decisive one: same money, spent wide vs spent deep -------------
    # A budget buys a fixed number of votes. Spend it on many pairs shallowly or
    # few pairs deeply? Repeated over seeds because 750 held-out pairs are noisy.
    print(f"\nEQUAL BUDGET — ~13,500 votes (~$700) split three ways, {args.seeds} seeds")
    print(f"  {'config':34} {'all':>15}  {'close':>15}")
    budget_configs = [(2250, 6), (1687, 8), (1125, 12)]
    equal: list[dict] = []
    for n_pairs, cap in budget_configs:
        accs, closes, spends = [], [], []
        for s in range(args.seeds):
            r = random.Random(1000 + s)
            take = set(r.sample(pool, min(n_pairs, len(pool))))
            v = tally(judgments, meta_by_id, pair_filter=take, cap=cap, seed=1000 + s)
            theta = fit(matchups, v)
            a, _ = score(theta, eval_votes, by_index)
            c, _ = score(theta, eval_close, by_index)
            accs.append(a)
            closes.append(c)
            spends.append(sum(x + y for x, y in v.values()) * 1900 / 36290)
        equal.append({"pairs": n_pairs, "votesPerPair": cap,
                      "spendEquivalent": float(np.mean(spends)),
                      "heldOutAccuracyMean": float(np.mean(accs)),
                      "heldOutAccuracySd": float(np.std(accs)),
                      "heldOutCloseMean": float(np.mean(closes)),
                      "heldOutCloseSd": float(np.std(closes))})
        print(f"  {f'{n_pairs} pairs x {cap} votes (${np.mean(spends):,.0f})':34} "
              f"{np.mean(accs):8.2%} +/-{np.std(accs):5.2%}  "
              f"{np.mean(closes):8.2%} +/-{np.std(closes):5.2%}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "equal-budget.json").write_text(json.dumps(
        {"seeds": args.seeds, "configs": equal}, indent=2))
    (out / "value-curve.json").write_text(json.dumps({
        "heldOutPairs": len(hold), "heldOutVotes": n_eval,
        "heldOutClosePairs": len(eval_close),
        "costPerVote": 1900 / 36290, "curve": results}, indent=2))
    with (out / "value-curve.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\nwrote {out}/value-curve.json and .csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
