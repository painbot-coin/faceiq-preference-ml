"""Which checkpoint should the Inference tab use? Rank them on placement error, not val accuracy.

`bestValAccuracy` in `metrics.json` answers "how often does this model pick the winner of a pair from
the export", which is not the production question. Production asks: **placed on the cohort scale via
the reference set, how far is this face from where the ranking of record puts it?** Those are
different questions and they do not have to agree — a model can win pairs while placing badly if its
score ordering is locally right and globally compressed.

Cheap because no forward passes are needed: `artifacts/<run>/model_scores.csv` already holds every
cohort face's comparator score, and placement only uses the *sign* of each comparison.

**The honest caveat, and it is the whole reason for `--val-only`.** Cohort faces are training faces
for most runs, so placing them is optimistic. Worse for cross-run comparison, runs differ in
`val_fraction` (0.5 for v12/v17, 0.2 for v14), so a run that trained on more faces looks better for a
reason that has nothing to do with quality. `--val-only` reproduces each run's own val split from its
config (the split is deterministic: `random.Random(42)` over sorted face ids) and scores only faces
that run never saw. Use it for any comparison you intend to act on.

Usage:
    python scripts/placement_by_checkpoint.py --val-only
    python scripts/placement_by_checkpoint.py --ranking artifacts/bt-refit-v5-panel --references 200
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# The smallest val_fraction any run used. Its val faces are held out for every run.
MIN_VAL_FRACTION = 0.2

from faceiq_pref.calibrate import make_scorer  # noqa: E402
from faceiq_pref.placement import (  # noqa: E402
    ANCHORS_TOP10,
    place,
    stratified_reference_ids,
    tier_of,
)


def val_faces_for(run: str, all_face_ids: list[str]) -> set[str] | None:
    """Reproduce a run's val face set from its config. -> None if the config is missing."""
    cfg = ROOT / "configs" / f"{run.replace('train-', 'train-')}.yaml"
    matches = list((ROOT / "configs").glob(f"*{run.split('train-')[-1]}*.yaml"))
    if not cfg.exists() and matches:
        cfg = matches[0]
    if not cfg.exists():
        return None
    frac = 0.2
    for line in cfg.read_text().splitlines():
        if line.strip().startswith("val_fraction:"):
            frac = float(line.split(":", 1)[1].split("#")[0].strip())
    ids = sorted(all_face_ids)
    random.Random(42).shuffle(ids)
    return set(ids[: int(len(ids) * frac)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranking", default="artifacts/bt-refit-v5-panel")
    ap.add_argument("--references", type=int, default=200)
    ap.add_argument("--val-only", action="store_true",
                    help="score only faces the run never trained on (do this before acting)")
    ap.add_argument("--common-val", action="store_true",
                    help="score the val faces of the STRICTEST split (val_fraction 0.2), which are "
                         "held out for every run because the split shares a seed and the 0.2 set is "
                         "a subset of the 0.5 set. The only fair cross-run comparison.")
    ap.add_argument("--out", default="artifacts/placement-by-checkpoint.json")
    a = ap.parse_args()

    rank = pd.read_csv(Path(a.ranking) / "ratings.csv")
    to_ten = make_scorer(ANCHORS_TOP10)
    runs = sorted(p.parent for p in (ROOT / "artifacts").glob("*/model_scores.csv"))
    out: list[dict] = []

    for run_dir in runs:
        run = run_dir.name
        if not (ROOT / "checkpoints" / run / "best.pt").exists():
            continue
        scores = pd.read_csv(run_dir / "model_scores.csv")
        row: dict = {"run": run}
        for gender in ("female", "male"):
            rg = rank[rank["gender"] == gender]
            sg = scores[scores["gender"] == gender]
            theta_of = dict(zip(rg["faceId"], rg["theta"]))
            pct_of = dict(zip(rg["faceId"], rg["percentile"]))
            score_of = dict(zip(sg["faceId"], sg["modelScore"]))
            usable = {f: p for f, p in pct_of.items() if f in score_of}
            if len(usable) < 50:
                continue

            ref_ids = stratified_reference_ids(usable, n=a.references)
            ref_set = set(ref_ids)
            ref_thetas = [theta_of[f] for f in ref_ids]
            ref_scores = [score_of[f] for f in ref_ids]
            population = list(theta_of.values())

            targets = [f for f in usable if f not in ref_set]
            if a.common_val:
                # ids[:20%] is a subset of ids[:50%] under a shared seed, so these faces are val
                # for every run regardless of its own val_fraction.
                ids = sorted(scores["faceId"])
                random.Random(42).shuffle(ids)
                common = set(ids[: int(len(ids) * MIN_VAL_FRACTION)])
                targets = [f for f in targets if f in common]
            elif a.val_only:
                vf = val_faces_for(run, list(scores["faceId"]))
                if vf is not None:
                    targets = [f for f in targets if f in vf]
            if len(targets) < 30:
                continue

            errs, placed, truth, unbounded, tier_hit = [], [], [], 0, 0
            for f in targets:
                pl = place(score_of[f], ref_scores, ref_thetas, population_thetas=population)
                true_ten = to_ten(pct_of[f])
                errs.append(abs(pl.score_ten - true_ten))
                placed.append(pl.score_ten)
                truth.append(true_ten)
                unbounded += int(not pl.bounded)
                tier_hit += int(pl.tier == tier_of(true_ten)[0])

            row[f"{gender}_n"] = len(targets)
            row[f"{gender}_median"] = float(np.median(errs))
            row[f"{gender}_p90"] = float(np.percentile(errs, 90))
            row[f"{gender}_spearman"] = float(spearmanr(placed, truth).statistic)
            row[f"{gender}_tierExact"] = tier_hit / len(targets)
            row[f"{gender}_unbounded"] = unbounded / len(targets)
        if any(k.endswith("_median") for k in row):
            meds = [v for k, v in row.items() if k.endswith("_median")]
            row["median"] = float(np.mean(meds))
            out.append(row)

    out.sort(key=lambda r: r["median"])
    (ROOT / a.out).write_text(json.dumps(
        {"ranking": a.ranking, "references": a.references, "valOnly": a.val_only, "runs": out},
        indent=2,
    ))

    scope = ("the common val split (val_fraction 0.2, held out for every run)" if a.common_val
             else "each run's own val split — NOT comparable across runs" if a.val_only
             else "ALL non-reference faces (optimistic)")
    print(f"\nplacement error on {scope}, {a.references} refs from {Path(a.ranking).name}")
    print(f"{'run':28} {'F med':>6} {'F p90':>6} {'M med':>6} {'M p90':>6} "
          f"{'tier ok':>8} {'rho':>6} {'unbnd':>6} {'n':>6}")
    for r in out:
        def g(k, d=float("nan")):
            return r.get(k, d)
        tier = np.mean([g("female_tierExact", 0), g("male_tierExact", 0)])
        rho = np.mean([g("female_spearman", 0), g("male_spearman", 0)])
        unb = np.mean([g("female_unbounded", 0), g("male_unbounded", 0)])
        n = g("female_n", 0) + g("male_n", 0)
        print(f"{r['run'][:28]:28} {g('female_median'):6.3f} {g('female_p90'):6.2f} "
              f"{g('male_median'):6.3f} {g('male_p90'):6.2f} {tier:8.1%} {rho:6.3f} "
              f"{unb:6.1%} {n:6.0f}")
    print(f"\n-> {a.out}")
    if not (a.common_val or a.val_only):
        print("re-run with --common-val before acting: runs differ in val_fraction, so this table "
              "rewards having trained on more faces.")
    elif a.val_only and not a.common_val:
        print("--val-only scores each run on ITS OWN val set, so the sets differ in size and "
              "membership. Use --common-val to compare runs.")


if __name__ == "__main__":
    main()
