#!/usr/bin/env python
"""Is the bottleneck the cohort, the matchup graph, or human perception?

Before rebuilding from scratch — a bigger cohort, a cleaner cohort, or 80-100k matchups
instead of 52k — measure which of those is actually limiting us. Three diagnostics:

  1. **composition** — is the cohort skewed toward attractive faces? If it is, percentile
     calibration is biased: a mid-population face gets a low /10 because it is below the
     *cohort's* median rather than the population's.
  2. **graph density** — subsample the VLM matchups, refit, and score against the panel's
     human votes. Human votes are never in these fits, so this is a clean read on whether
     more machine comparisons per face would buy anything. If the curve is flat from ~26
     comparisons/face, doubling to 100k matchups is wasted.
  3. **resolution** — translate percentile gaps into /10 points and human agreement, so
     "the bands" have a plain-English meaning.

Usage:
    python scripts/cohort_capacity.py --export data/exports/<runId>
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from refit_bt_panel import load_panel_votes  # noqa: E402

from faceiq_pref.bt_panel import build_cells, fit_joint_bt  # noqa: E402
from faceiq_pref.data import load_export  # noqa: E402

GENDERS = ("female", "male")


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def composition(export, ratings: Path, bad: set[str]) -> None:
    section("1. COHORT COMPOSITION — is the sample skewed toward attractive faces?")
    faces = export.faces()
    kept = {fid: f for fid, f in faces.items() if fid not in bad}
    print(f"{len(faces):,} faces exported, {len(kept):,} after QC exclusions")

    deciles = Counter()
    labs: list[float] = []
    for f in kept.values():
        if f.decile_bin is not None:
            deciles[int(f.decile_bin)] += 1
        if f.labs_overall_score is not None:
            labs.append(float(f.labs_overall_score))

    if deciles:
        total = sum(deciles.values())
        print("\nLabs decile bin (1 = least attractive by the legacy formula):")
        for d in sorted(deciles):
            n = deciles[d]
            bar = "#" * round(40 * n / max(deciles.values()))
            print(f"  D{d:<2} {n:5,} {n/total:6.1%} {bar}")
        flat = total / len(deciles)
        worst = max(abs(deciles[d] - flat) / flat for d in deciles)
        print(f"  max deviation from a flat design: {worst:.0%}")

    if labs:
        a = np.array(labs)
        print(f"\nLabs overall_score: mean {a.mean():.2f}, median {np.median(a):.2f}, "
              f"p10 {np.percentile(a,10):.2f}, p90 {np.percentile(a,90):.2f}")

    if ratings.exists():
        with ratings.open() as f:
            rows = [r for r in csv.DictReader(f)]
        ten = np.array([float(r["scoreOutOf10"]) for r in rows])
        print(f"\nOur /10 after calibration: mean {ten.mean():.2f}, "
              f"median {np.median(ten):.2f}, p10 {np.percentile(ten,10):.2f}, "
              f"p90 {np.percentile(ten,90):.2f}")
        print("  (this distribution is imposed by the pre-registered anchor curve, so it "
              "cannot reveal cohort skew — the decile design above is the check that can)")


def density(export, matchups, votes, bad: set[str], seeds: int) -> list[dict]:
    section("2. GRAPH DENSITY — would more VLM matchups per face help?")
    print("VLM-only fits at varying density, scored on the panel's 36k human votes.")
    print("Human votes never enter these fits, so the comparison is clean.\n")

    by_index = {m.pair_index: m for m in matchups}
    fractions = [0.25, 0.5, 0.75, 1.0]

    # Score only on faces every config retains, so the eval set never changes.
    thetas: dict[float, dict[int, dict[str, float]]] = {}
    for frac in fractions:
        thetas[frac] = {}
        for s in range(seeds):
            rng = random.Random(500 + s)
            sub = [m for m in matchups if rng.random() < frac]
            th: dict[str, float] = {}
            for g in GENDERS:
                fi, ce, co, ob, nv, nh, df, dm = build_cells(
                    sub, {}, g, drop_below_min=False)
                if fi:
                    th.update(fit_joint_bt(fi, ce, co, ob, g, nv, nh, df, dm).theta)
            thetas[frac][s] = th

    common = set.intersection(*[set(t.keys())
                                for per in thetas.values() for t in per.values()])
    print(f"scoring on {len(common):,} faces present in every fit")

    out = []
    for frac in fractions:
        accs, dens = [], []
        for s in range(seeds):
            th = thetas[frac][s]
            rng = random.Random(500 + s)
            sub = [m for m in matchups if rng.random() < frac]
            cnt: dict[str, int] = defaultdict(int)
            for m in sub:
                cnt[m.face_a_id] += 1
                cnt[m.face_b_id] += 1
            dens.append(np.mean([cnt[f] for f in common if f in cnt]))
            correct = total = 0.0
            for idx, (wa, wb) in votes.items():
                m = by_index.get(idx)
                if m is None or m.face_a_id not in common or m.face_b_id not in common:
                    continue
                d = th[m.face_a_id] - th[m.face_b_id]
                if d == 0:
                    continue
                correct += wa if d > 0 else wb
                total += wa + wb
            accs.append(correct / total if total else float("nan"))
        out.append({"fraction": frac, "matchups": round(len(matchups) * frac),
                    "comparisonsPerFace": float(np.mean(dens)),
                    "accuracy": float(np.mean(accs)), "sd": float(np.std(accs))})
        r = out[-1]
        print(f"  {r['matchups']:6,} matchups  {r['comparisonsPerFace']:5.1f}/face  "
              f"{r['accuracy']:7.2%} +/-{r['sd']:.2%}")

    gain = out[-1]["accuracy"] - out[-2]["accuracy"]
    step = out[-1]["comparisonsPerFace"] - out[-2]["comparisonsPerFace"]
    print(f"\nlast segment: {gain:+.2%} for {step:+.1f} comparisons/face")
    print(f"extrapolated to ~70/face (100k matchups): {gain * (70-out[-1]['comparisonsPerFace'])/step:+.2%} "
          "— straight-line, so an upper bound; real curves bend down")
    return out


def headroom(matchups, votes, ratings: Path, strata: dict[int, str]) -> dict:
    """The best score *any* ranking could get, so we know how much is left to buy.

    A ranking predicts one side per pair. On a pair the panel split 7-5, always naming the
    7 side scores 7/12 — there is no ranking that does better, because the same pair is
    shown to people who disagree. Averaging that bound over the evaluation set gives a hard
    ceiling. Distance to it, not distance to 100%, is the remaining headroom, and it is
    what decides whether more data can still pay.
    """
    section("4. HEADROOM — how much is even left to win?")
    by_index = {m.pair_index: m for m in matchups}
    with ratings.open() as f:
        theta = {r["faceId"]: float(r["theta"]) for r in csv.DictReader(f)}

    tot = best = cur = 0.0
    per: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    for idx, (wa, wb) in votes.items():
        m = by_index.get(idx)
        if m is None or m.face_a_id not in theta or m.face_b_id not in theta:
            continue
        n = wa + wb
        if not n:
            continue
        d = theta[m.face_a_id] - theta[m.face_b_id]
        hit = wa if d > 0 else wb
        tot += n
        best += max(wa, wb)
        cur += hit
        cell = per[strata.get(idx, "?")]
        cell[0] += n
        cell[1] += max(wa, wb)
        cell[2] += hit

    print(f"{'stratum':10} {'ceiling':>9} {'v3 now':>9} {'headroom':>9} {'votes':>9}")
    out = {}
    for s in ("close", "mid", "lowconf", "overlap", "clear"):
        if s not in per:
            continue
        n, b, c = per[s]
        print(f"{s:10} {b/n:9.1%} {c/n:9.1%} {(b-c)/n:9.1%} {n:9,.0f}")
        out[s] = {"ceiling": b / n, "current": c / n, "headroom": (b - c) / n}
    print(f"{'ALL':10} {best/tot:9.1%} {cur/tot:9.1%} {(best-cur)/tot:9.1%} {tot:9,.0f}")
    out["all"] = {"ceiling": best / tot, "current": cur / tot,
                  "headroom": (best - cur) / tot}
    print("\nThe ceiling is below 100% because the same pair is shown to people who")
    print("disagree; no ranking can satisfy both. 'headroom' is what any amount of")
    print("additional data could still buy, in total.")
    return out


def extrapolate(curve: list[dict], ceiling: float) -> None:
    """Fit accuracy = a - b/sqrt(k) and read off where more matchups stop paying.

    A Bradley-Terry ability estimate has standard error proportional to 1/sqrt(k) in the
    number of comparisons k, so accuracy approaches its asymptote in 1/sqrt(k) rather than
    linearly. Fitting that shape is the honest extrapolation; a straight line through the
    last two points is not, and overstates the return.
    """
    section("5. EXTRAPOLATION — what a denser graph would actually buy")
    k = np.array([r["comparisonsPerFace"] for r in curve])
    a = np.array([r["accuracy"] for r in curve])
    x = 1.0 / np.sqrt(k)
    slope, intercept = np.polyfit(x, a, 1)
    pred = intercept + slope * x
    ss_res = float(np.sum((a - pred) ** 2))
    ss_tot = float(np.sum((a - a.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")

    print(f"fit: accuracy = {intercept:.4f} - {-slope:.4f}/sqrt(k)   (R^2 = {r2:.3f})")
    print(f"asymptote at infinite matchups: {intercept:.2%}")
    print(f"measured now at k={k[-1]:.1f}: {a[-1]:.2%}\n")
    print(f"{'comparisons/face':>18} {'matchups':>10} {'predicted':>10} {'gain vs now':>12}")
    for target in (33.4, 50, 70, 100, 200):
        p = intercept + slope / np.sqrt(target)
        mu = round(target * 2867 / 2)
        star = "  <- 100k matchups" if target == 70 else ""
        print(f"{target:18.0f} {mu:10,} {p:10.2%} {p-a[-1]:+12.2%}{star}")
    print(f"\nEven infinite VLM matchups top out at {intercept:.1%}, which is "
          f"{ceiling - intercept:.1%} short of the {ceiling:.1%} ceiling.")
    print("That residual is the part only a different *source* of labels can reach —")
    print("more draws from a labeller that is at chance on a pair add precision, not signal.")


def resolution(export, matchups, votes, ratings: Path) -> list[dict]:
    section("3. RESOLUTION — what a 'percentile gap band' actually means")
    with ratings.open() as f:
        rows = list(csv.DictReader(f))
    pct = {r["faceId"]: float(r["percentile"]) for r in rows}
    ten = {r["faceId"]: float(r["scoreOutOf10"]) for r in rows}
    by_index = {m.pair_index: m for m in matchups}

    recs = []
    for idx, (wa, wb) in votes.items():
        m = by_index.get(idx)
        if m is None or m.face_a_id not in pct or m.face_b_id not in pct:
            continue
        n = wa + wb
        if not n:
            continue
        gap = abs(pct[m.face_a_id] - pct[m.face_b_id]) * 100
        hi = m.face_a_id if pct[m.face_a_id] > pct[m.face_b_id] else m.face_b_id
        agree = (wa if hi == m.face_a_id else wb) / n
        recs.append({"gap": gap,
                     "tenGap": abs(ten[m.face_a_id] - ten[m.face_b_id]),
                     "agree": agree})
    df = recs
    bands = [(0, 5), (5, 10), (10, 20), (20, 30), (30, 45), (45, 60), (60, 80), (80, 101)]
    print(f"{'band (pctile)':>14} {'/10 gap':>9} {'pairs':>7} {'humans agree':>13}")
    out = []
    for lo, hi in bands:
        sel = [r for r in df if lo <= r["gap"] < hi]
        if not sel:
            continue
        out.append({"band": f"{lo}-{min(hi,100)}",
                    "tenGap": float(np.mean([r["tenGap"] for r in sel])),
                    "pairs": len(sel),
                    "humansAgree": float(np.mean([r["agree"] for r in sel]))})
        r = out[-1]
        print(f"{r['band']:>14} {r['tenGap']:9.2f} {r['pairs']:7,} {r['humansAgree']:13.1%}")
    print("\nRead it as: two faces N percentile points apart differ by this many /10 points,")
    print("and this share of raters picked the one we rate higher.")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--ratings", default="artifacts/bt-refit-v3-panel/ratings.csv")
    ap.add_argument("--panel-results", default="labels/panel-pilot/results")
    ap.add_argument("--panel-meta", default="labels/panel-pilot/sample-meta.json")
    ap.add_argument("--rejects", default="artifacts/panel-run-v1/reject-pids.txt")
    ap.add_argument("--exclude-faces", default="artifacts/face-qc-v1/exclude-faces.csv")
    ap.add_argument("--exclude-genders", default="artifacts/face-qc-v1/gender-fixes.csv")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="artifacts/cohort-capacity-v1")
    args = ap.parse_args()

    export = load_export(args.export)
    bad: set[str] = set()
    for p in (args.exclude_faces, args.exclude_genders):
        if p and Path(p).exists():
            bad.update(r["faceId"] for r in csv.DictReader(open(p)))
    matchups = [m for m in export.all_matchups()
                if m.face_a_id not in bad and m.face_b_id not in bad]
    votes, _ = load_panel_votes(Path(args.panel_results), Path(args.panel_meta),
                                Path(args.rejects))
    print(f"export {export.run_id}: {len(matchups):,} matchups after QC, "
          f"{len(votes):,} panel-voted pairs")

    strata = {m["pairIndex"]: m["stratum"]
              for m in json.loads(Path(args.panel_meta).read_text())["pairs"]}

    composition(export, Path(args.ratings), bad)
    dens = density(export, matchups, votes, bad, args.seeds)
    head = headroom(matchups, votes, Path(args.ratings), strata)
    extrapolate(dens, head["all"]["ceiling"])
    res = resolution(export, matchups, votes, Path(args.ratings))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "capacity.json").write_text(json.dumps(
        {"densityCurve": dens, "headroom": head, "resolutionBands": res}, indent=2))
    print(f"\nwrote {out}/capacity.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
