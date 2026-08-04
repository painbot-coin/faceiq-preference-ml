#!/usr/bin/env python
"""Did the newest run's money buy anything that generalises? Run after every study.

`refit_bt_panel.py` withholds *raters*, so a pair keeps some votes inside the fit. That
answers "can we predict the rest of the crowd on a pair we bought", which is the right
question for a pair we own but flatters new spend: some of the gain is the pair's own
votes rather than better face scores.

This withholds whole *pairs* from the newest run instead. None of their votes enter any
fit, so a model can only do better on them by having learned better theta for the faces
involved. That is the only mechanism by which the next study could help pairs we have
not bought — which is to say, the only mechanism that justifies buying more.

Ladder, all scored on the same withheld pairs:

    VLM only            what BT saw before any human money
    + prior runs        what we shipped before this study
    + this run          what this study bought

If the last step is flat, the run bought coverage and nothing else, and the next study
should change target rather than scale.

Usage:
    python scripts/panel_run_delta.py --export data/exports/<runId> \
        --prior labels/panel-pilot/results labels/panel-pilot/sample-meta.json \
        --new   labels/panel-run-3/results labels/panel-run-3/sample-meta.json \
        --rejects artifacts/panel-run-v1/reject-pids.txt \
                  artifacts/panel-run-v3/reject-pids.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from faceiq_pref.bt_panel import build_cells, fit_joint_bt  # noqa: E402
from faceiq_pref.data import load_export  # noqa: E402

GENDERS = ("female", "male")


def load_run(results: Path, meta_path: Path, rejects: set[str]):
    meta = json.loads(meta_path.read_text())["pairs"]
    by_id = {m["pairId"]: m for m in meta}
    rows = []
    for line in (results / "judgments.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        j = json.loads(line)
        if j["studyId"] == "test" or j["isGold"] or j["prolificPid"] in rejects:
            continue
        m = by_id.get(j["pairId"])
        if m is None:
            continue
        rows.append((j, m))
    return rows, meta


def tally(rows, keep=None):
    """-> {pair_index: (wins A, wins B)}; ties count half to each side."""
    votes: dict[int, list[float]] = {}
    for j, m in rows:
        idx = m["pairIndex"]
        if keep is not None and idx not in keep:
            continue
        cell = votes.setdefault(idx, [0.0, 0.0])
        a_side = "left" if m["faceAOnLeft"] else "right"
        if j["choice"] == "tie":
            cell[0] += 0.5
            cell[1] += 0.5
        elif j["choice"] == a_side:
            cell[0] += 1.0
        else:
            cell[1] += 1.0
    return {k: (v[0], v[1]) for k, v in votes.items()}


def merge(*vote_maps):
    out: dict[int, list[float]] = {}
    for vm in vote_maps:
        for idx, (a, b) in vm.items():
            cell = out.setdefault(idx, [0.0, 0.0])
            cell[0] += a
            cell[1] += b
    return {k: (v[0], v[1]) for k, v in out.items()}


def fit(matchups, votes):
    theta = {}
    for g in GENDERS:
        fi, ce, co, ob, nv, nh, df, dm = build_cells(matchups, votes, g)
        theta.update(fit_joint_bt(fi, ce, co, ob, g, nv, nh, df, dm).theta)
    return theta


def score(theta, eval_votes, by_index):
    """Share of individual withheld votes that agree with theta's ordering."""
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
    return (correct / total if total else float("nan")), total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--prior", nargs=2, action="append", default=[],
                    metavar=("RESULTS", "META"), help="repeatable: earlier studies")
    ap.add_argument("--new", nargs=2, required=True, metavar=("RESULTS", "META"))
    ap.add_argument("--rejects", nargs="*", default=[])
    ap.add_argument("--exclude-faces", default="artifacts/face-qc-v1/exclude-faces.csv")
    ap.add_argument("--exclude-genders", default="artifacts/face-qc-v1/gender-fixes.csv")
    ap.add_argument("--holdout", type=float, default=0.25)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--spend", type=float, required=True,
                    help="what the newest run cost, for $/point")
    ap.add_argument("--out", default="artifacts/panel-run-delta-v3")
    args = ap.parse_args()

    rejects: set[str] = set()
    for p in args.rejects:
        if Path(p).exists():
            rejects |= set(Path(p).read_text().split())

    export = load_export(args.export)
    matchups = export.all_matchups()
    dropped: set[str] = set()
    for p in (args.exclude_faces, args.exclude_genders):
        if p and Path(p).exists():
            with open(p, newline="") as fh:
                dropped.update(r["faceId"] for r in csv.DictReader(fh))
    matchups = [m for m in matchups
                if m.face_a_id not in dropped and m.face_b_id not in dropped]
    by_index = {m.pair_index: m for m in matchups}
    print(f"export {export.run_id}: {len(matchups):,} matchups after QC "
          f"({len(dropped)} faces dropped)")

    prior_rows = []
    for results, meta in args.prior:
        rows, _ = load_run(Path(results), Path(meta), rejects)
        prior_rows += rows
        print(f"prior  {results}: {len(rows):,} votes")
    new_rows, new_meta = load_run(Path(args.new[0]), Path(args.new[1]), rejects)
    print(f"new    {args.new[0]}: {len(new_rows):,} votes")

    prior_votes = tally(prior_rows)
    new_idx = sorted({m["pairIndex"] for _, m in new_rows if m["pairIndex"] in by_index})
    print(f"{len(new_idx):,} new pairs live in the post-QC export\n")

    ladder = ["VLM only", "+ prior runs", "+ this run"]
    acc: dict[str, list[float]] = {k: [] for k in ladder}
    n_hold_pairs, n_hold_votes = [], []

    for s in range(args.seeds):
        shuffled = new_idx[:]
        random.Random(500 + s).shuffle(shuffled)
        cut = int(len(shuffled) * args.holdout)
        hold, pool = set(shuffled[:cut]), set(shuffled[cut:])
        eval_votes = tally(new_rows, keep=hold)
        pool_votes = tally(new_rows, keep=pool)
        n_hold_pairs.append(len(hold))
        n_hold_votes.append(sum(a + b for a, b in eval_votes.values()))

        for label, votes in (("VLM only", {}),
                             ("+ prior runs", prior_votes),
                             ("+ this run", merge(prior_votes, pool_votes))):
            a, _ = score(fit(matchups, votes), eval_votes, by_index)
            acc[label].append(a)
        print(f"seed {s}: " + "  ".join(f"{k} {acc[k][-1]:.2%}" for k in ladder))

    print(f"\n{args.seeds} seeds, {int(np.mean(n_hold_pairs)):,} withheld pairs "
          f"(~{int(np.mean(n_hold_votes)):,} votes), none in any fit")
    print(f"  {'model':14} {'accuracy':>16} {'vs previous step':>18}")
    prev = None
    summary = []
    for k in ladder:
        mu, sd = float(np.mean(acc[k])), float(np.std(acc[k]))
        delta = "" if prev is None else f"{mu - prev:+.2%}"
        print(f"  {k:14} {mu:8.2%} +/-{sd:5.2%} {delta:>18}")
        summary.append({"model": k, "accuracyMean": mu, "accuracySd": sd,
                        "deltaVsPrevious": None if prev is None else mu - prev})
        prev = mu

    # ---- dose curve: is this run's own contribution still climbing? ----------
    # The ladder shows one step and cannot show curvature. Feeding a growing share of
    # the run's pairs into the fit does: a curve still rising at 100% means the next
    # study buys about what this one did, a flattening one means stop.
    print(f"\nDOSE — prior runs plus a growing share of this run's pool, "
          f"{args.seeds} seeds")
    print(f"  {'share of pool':14} {'pairs':>6} {'votes':>8} {'accuracy':>16}")
    dose: list[dict] = []
    for frac in (0.25, 0.5, 0.75, 1.0):
        accs, npairs, nvotes = [], [], []
        for s in range(args.seeds):
            shuffled = new_idx[:]
            random.Random(500 + s).shuffle(shuffled)
            cut = int(len(shuffled) * args.holdout)
            hold, pool = set(shuffled[:cut]), shuffled[cut:]
            take = set(pool[: int(len(pool) * frac)])
            ev = tally(new_rows, keep=hold)
            pv = tally(new_rows, keep=take)
            a, _ = score(fit(matchups, merge(prior_votes, pv)), ev, by_index)
            accs.append(a)
            npairs.append(len(take))
            nvotes.append(sum(x + y for x, y in pv.values()))
        dose.append({"fraction": frac, "pairs": int(np.mean(npairs)),
                     "votes": int(np.mean(nvotes)),
                     "accuracyMean": float(np.mean(accs)),
                     "accuracySd": float(np.std(accs))})
        print(f"  {f'{frac:.0%}':14} {int(np.mean(npairs)):6,} "
              f"{int(np.mean(nvotes)):8,} {np.mean(accs):8.2%} "
              f"+/-{np.std(accs):5.2%}")
    for a, b in zip(dose, dose[1:]):
        print(f"  {a['fraction']:.0%} -> {b['fraction']:.0%}: "
              f"{(b['accuracyMean'] - a['accuracyMean']) * 100:+.2f} pts for "
              f"{b['votes'] - a['votes']:,} more votes")

    gain = summary[-1]["deltaVsPrevious"]
    # Paired across seeds: the same holdout scores both models, so the seed-to-seed
    # spread cancels and this is the interval that matters for the spend decision.
    paired = [n - p for n, p in zip(acc["+ this run"], acc["+ prior runs"])]
    print(f"\nthis run's marginal gain: {gain:+.2%} "
          f"(paired across seeds: {np.mean(paired):+.2%} +/-{np.std(paired):.2%}, "
          f"{sum(1 for d in paired if d > 0)}/{len(paired)} seeds positive)")
    if gain and gain > 0:
        print(f"cost of the gain: ${args.spend:,.0f} for {gain * 100:.2f} points "
              f"= ${args.spend / (gain * 100):,.0f} per point")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run-delta.json").write_text(json.dumps({
        "export": export.run_id,
        "newRun": args.new[0],
        "priorRuns": [p[0] for p in args.prior],
        "protocol": "hold out whole pairs from the newest run; no held-out pair's "
                    "votes enter any fit, so gains must travel through face theta",
        "holdoutFraction": args.holdout,
        "seeds": args.seeds,
        "heldOutPairs": int(np.mean(n_hold_pairs)),
        "heldOutVotes": int(np.mean(n_hold_votes)),
        "spend": args.spend,
        "ladder": summary,
        "doseCurve": dose,
        "marginalGain": gain,
        "marginalGainPairedSd": float(np.std(paired)),
        "seedsPositive": sum(1 for d in paired if d > 0),
        "dollarsPerPoint": (args.spend / (gain * 100)) if gain and gain > 0 else None,
        "perSeed": acc,
    }, indent=2))
    print(f"\nwrote {out}/run-delta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
