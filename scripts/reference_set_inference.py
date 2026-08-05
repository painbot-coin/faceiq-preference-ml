#!/usr/bin/env python
"""Score a new face by comparing it against the cohort, instead of reading the raw scalar.

The problem this exists to fix. The comparator is *trained* to answer "is A better than B" and is
currently *read* as "what is A's absolute score". Those are not the same thing: a Siamese network's
output has no anchored zero and no anchored scale, because only *differences* ever appear in the
loss. Any shift in the head's bias, or any distribution shift between cohort photos and a user's
upload, moves every score together while changing not one pairwise comparison. That is the most
likely reason an uploaded face scores oddly while pairwise accuracy looks fine.

The fix is to use the model the way it was trained. Compare the new face against N reference faces
whose Bradley-Terry theta we already know, then ask: **what theta would explain those outcomes?**

    P(new beats ref_j) = sigmoid(theta_new - theta_j)

Maximise the likelihood over theta_new. That is one-dimensional and strictly concave, so a dozen
Newton steps solve it exactly, and the curvature at the optimum gives a standard error for free:

    se(theta_new) = 1 / sqrt( sum_j p_j (1 - p_j) )

Three things fall out of that formula that answer real design questions:

* **Why not 10 reference faces?** Information adds across references, so se falls as 1/sqrt(N).
  Ten references cannot place anyone precisely.
* **Why not obsess over curating them?** The weight of reference j is p_j(1-p_j), which is 0.25 for
  an evenly-matched opponent and collapses toward 0 for a mismatch. References far from the user
  contribute almost nothing, so the set self-selects and a hand-picked ladder is not needed. Using
  the whole ranking is simpler *and* strictly more informative than a curated subset.
* **Where the precision comes from.** Faces near the user's own level. A reference set with no mass
  near a given user places that user badly however large it is.

Honest limits, stated because they bound what this can fix:

* It cannot repair the *ordering* — if the comparator ranks two faces wrongly, comparing against a
  reference set ranks them wrongly too. It fixes *placement on the scale*, which is a different and
  more likely failure.
* The reference faces are cohort faces the model trained on. That is correct for production (we
  want references whose theta we know well) but it means this script must evaluate on **val-split
  faces only**, or the "new" face is one the model has seen.
* theta_new is on the BT theta scale, so it maps to a percentile and then to /10 through the same
  pre-registered anchor curve as everything else.

Usage:
    python scripts/reference_set_inference.py \
        --export data/exports/cmr1mr0m7000196d57zi3vcgn \
        --checkpoint checkpoints/train-v16-panel-run4/best.pt \
        --ratings artifacts/bt-refit-v5-panel/ratings.csv \
        --out artifacts/train-v16-panel-run4/reference-set.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from faceiq_pref.calibrate import score_out_of_10  # noqa: E402
from faceiq_pref.data import load_export  # noqa: E402
from faceiq_pref.eval import score_all_faces  # noqa: E402
from faceiq_pref.train import TrainConfig  # noqa: E402

SIZES = [10, 25, 50, 100, 200, 400, 800, 1600]

# Where the subject sits on the scale, for the per-band breakdown. Extremes are the interesting
# cells: a face better than every reference has no finite MLE, and no reference-set size fixes it.
SUBJECT_BANDS = [(0.0, 0.25, "bottom 25%"), (0.25, 0.75, "middle 50%"),
                 (0.75, 0.95, "top 25-5%"), (0.95, 1.01, "top 5%")]


def spearman(xs: list[float], ys: list[float]) -> float:
    def rank(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        out = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                out[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return out

    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def fit_theta(margins: list[float], ref_thetas: list[float]) -> tuple[float, float]:
    """MLE for theta_new given per-reference model margins, plus its standard error.

    `margins[j]` is the comparator's logit for "new beats reference j". We convert it to a hard
    outcome rather than trusting its magnitude, because the comparator's logit scale is exactly
    the thing we do not believe — its *ordering* is what we are willing to use. So each reference
    contributes a win or a loss, and theta_new is fitted to explain that record.

    Newton on a strictly concave 1-D log-likelihood: converges in a handful of steps. A record
    with no losses (or no wins) has no finite MLE, which is the truth for a face better than every
    reference; it is capped and flagged rather than silently returned as a large number.
    """
    wins = [1.0 if m > 0 else 0.0 for m in margins]
    if all(w == 1.0 for w in wins) or all(w == 0.0 for w in wins):
        # Unbounded above/below. Return the extreme reference plus a margin, flagged by se=inf.
        return (max(ref_thetas) + 1.0 if wins[0] == 1.0 else min(ref_thetas) - 1.0), float("inf")
    theta = sum(ref_thetas) / len(ref_thetas)
    for _ in range(60):
        grad = info = 0.0
        for w, tj in zip(wins, ref_thetas):
            p = 1.0 / (1.0 + math.exp(-(theta - tj)))
            grad += w - p
            info += p * (1 - p)
        if info < 1e-12:
            break
        step = grad / info
        theta += max(-2.0, min(2.0, step))
        if abs(step) < 1e-9:
            break
    return theta, (1.0 / math.sqrt(info) if info > 0 else float("inf"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--ratings", default="artifacts/bt-refit-v5-panel/ratings.csv")
    ap.add_argument("--sizes", type=int, nargs="*", default=SIZES)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out")
    args = ap.parse_args()

    export = load_export(args.export)
    matchups = export.all_matchups()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = TrainConfig.from_saved(ckpt["config"])

    with open(args.ratings, newline="") as fh:
        rows = list(csv.DictReader(fh))
    theta = {r["faceId"]: float(r["theta"]) for r in rows}
    pct = {r["faceId"]: float(r["percentile"]) for r in rows}
    gender = {r["faceId"]: r["gender"] for r in rows}

    # The model's own val split: faces it has never seen. These stand in for a new upload.
    face_ids = sorted({f for m in matchups for f in (m.face_a_id, m.face_b_id)})
    shuffled = face_ids[:]
    random.Random(cfg.split_seed).shuffle(shuffled)
    val = set(shuffled[: int(len(shuffled) * cfg.val_fraction)])

    scores = score_all_faces(args.checkpoint, export, image_size=cfg.image_size)
    print(f"checkpoint {args.checkpoint} ({cfg.backbone} @ {cfg.image_size}px, "
          f"val_fraction {cfg.val_fraction})")

    rng = random.Random(args.seed)
    out: dict = {"checkpoint": args.checkpoint, "ratings": args.ratings, "genders": {}}

    for g in sorted({gender[f] for f in theta}):
        pool = [f for f in theta if gender[f] == g and f in scores]
        subjects = [f for f in pool if f in val]
        refs_all = [f for f in pool if f not in val]           # references are seen faces
        if len(subjects) < 50 or len(refs_all) < 10:
            print(f"  {g}: too few faces (subjects {len(subjects)}, refs {len(refs_all)}), skipping")
            continue
        # The reference pool is capped by the split: at val_fraction 0.5 only half the cohort is
        # available as seen references. Clip rather than skip, and include the full pool as the
        # last point so the curve shows where it actually flattens.
        sizes = sorted({min(n, len(refs_all)) for n in args.sizes} | {len(refs_all)})

        truth = [pct[f] for f in subjects]
        # Baseline: the raw scalar, which is how the model is read today.
        raw = [scores[f] for f in subjects]
        base_rho = spearman(raw, truth)

        rows_out = []
        print(f"\n=== {g} — {len(subjects):,} unseen subjects, {len(refs_all):,} reference faces ===")
        print(f"{'refs':>6} {'spearman vs BT':>15} {'median |err| /10':>17} "
              f"{'p90 |err| /10':>14} {'median se(theta)':>17}")
        print(f"{'raw':>6} {base_rho:15.4f} {'—':>17} {'—':>14} {'—':>17}")

        for n in sizes:
            refs = rng.sample(refs_all, n)
            ref_thetas = [theta[r] for r in refs]
            est, ses, errs = [], [], []
            for f in subjects:
                margins = [scores[f] - scores[r] for r in refs]
                th, se = fit_theta(margins, ref_thetas)
                est.append(th)
                if math.isfinite(se):
                    ses.append(se)
                # Map the fitted theta onto the cohort percentile scale, then to /10, and compare
                # against the face's own /10. This is the number a user would see.
                below = sum(1 for t in ref_thetas if t < th) / len(ref_thetas)
                errs.append(abs(score_out_of_10(below) - score_out_of_10(pct[f])))
            rho = spearman(est, truth)
            errs.sort()
            ses.sort()
            row = {"refs": n, "spearman": rho,
                   "medianAbsErrorTen": errs[len(errs) // 2],
                   "p90AbsErrorTen": errs[int(0.9 * len(errs))],
                   "medianSeTheta": ses[len(ses) // 2] if ses else None,
                   "unboundedShare": 1 - len(ses) / len(subjects)}
            rows_out.append(row)
            print(f"{n:6} {rho:15.4f} {row['medianAbsErrorTen']:17.3f} "
                  f"{row['p90AbsErrorTen']:14.3f} "
                  f"{(row['medianSeTheta'] if row['medianSeTheta'] else float('nan')):17.3f}")

        out["genders"][g] = {"subjects": len(subjects), "references": len(refs_all),
                             "rawScalarSpearman": base_rho, "bySize": rows_out}

        # ---- what se(theta) is actually worth, and who gets placed badly ----------------
        # Both questions decide reference-set size, and both answers are "not what you'd guess".
        ths = sorted(theta[f] for f in pool)
        nn = len(ths)

        def pct_of(t: float) -> float:
            lo, hi = 0, nn
            while lo < hi:
                mid = (lo + hi) // 2
                if ths[mid] < t:
                    lo = mid + 1
                else:
                    hi = mid
            return lo / nn

        se200 = next((r["medianSeTheta"] for r in rows_out if r["refs"] == 200), None)
        if se200:
            print(f"\n  se(theta) = {se200:.3f} at 200 refs is worth this much /10:")
            for label, q in (("p25", .25), ("median", .50), ("p75", .75), ("p90", .90)):
                t0 = ths[min(int(q * nn), nn - 1)]
                w = score_out_of_10(pct_of(t0 + se200)) - score_out_of_10(pct_of(t0 - se200))
                print(f"    {label:6} {score_out_of_10(pct_of(t0)):4.2f}/10  "
                      f"estimation band +/-{w / 2:.2f}")
            print("  Compare the disagreement band at 2-in-3 agreement: +/-0.67 /10. Estimation is "
                  "the SMALL term,\n  so more references shrink something that is already ~5x "
                  "below the floor set by human disagreement.")

        refs = rng.sample(refs_all, min(200, len(refs_all)))
        ref_thetas = [theta[r] for r in refs]
        print("\n  placement quality by where the subject sits (200 refs):")
        print(f"    {'subject band':14} {'n':>4} {'median |err| /10':>17} {'p90':>7} "
              f"{'no finite MLE':>14}")
        band_rows = []
        for lo, hi, name in SUBJECT_BANDS:
            sel = [f for f in subjects if lo <= pct[f] < hi]
            if len(sel) < 15:
                continue
            errs, unbounded = [], 0
            for f in sel:
                th, se = fit_theta([scores[f] - scores[r] for r in refs], ref_thetas)
                if not math.isfinite(se):
                    unbounded += 1
                below = sum(1 for t in ref_thetas if t < th) / len(ref_thetas)
                errs.append(abs(score_out_of_10(below) - score_out_of_10(pct[f])))
            errs.sort()
            band_rows.append({"band": name, "n": len(sel),
                              "medianAbsErrorTen": errs[len(errs) // 2],
                              "p90AbsErrorTen": errs[int(0.9 * len(errs))],
                              "noFiniteMleShare": unbounded / len(sel)})
            print(f"    {name:14} {len(sel):4} {errs[len(errs) // 2]:17.3f} "
                  f"{errs[int(0.9 * len(errs))]:7.3f} {unbounded / len(sel):13.1%}")
        out["genders"][g]["bySubjectBand"] = band_rows

    print("\nReading this table:")
    print("  * `raw` is how the model is read today — the bare scalar, ranked against BT truth.")
    print("  * Reference-set rows use only the SIGN of each comparison, so they cannot benefit")
    print("    from the scalar's magnitude; if they match or beat `raw` on spearman, the")
    print("    magnitude was carrying no extra information and the anchoring is free.")
    print("  * median |err| is in /10 points and is the number a user would feel.")
    print("  * se(theta) shrinks as 1/sqrt(refs); watch where it stops paying for itself.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
