"""Validate reference-set placement on faces from outside the cohort.

Every accuracy number we have so far shares one weakness: the yardstick is Bradley-Terry theta
fitted on the same 3,000 faces the comparator trained on. Held-out *pairs* are honest about
generalisation to new comparisons; they say nothing about generalisation to new *faces from a
different source*, which is what production actually gets. This closes that gap with photos the
programme has never touched.

Three tests, deliberately separated because they fail for different reasons and have different
fixes:

1. **Ordering.** You judge pairs of unseen photos. Does the placed score reproduce your ordering?
   This needs no absolute labels and no /10 scale, so it cannot be argued with — it is the thing the
   comparator was trained to do, measured on faces it has never seen from a source it has never
   seen. A failure here is a model problem.

2. **Calibration.** Optional hand ranges per face (`ranges.json`). Splits into a *shift* (everything
   placed 1.2 low -> the anchor ladder is wrong, cheap fix) and a *spread* (each face wrong by a
   different amount -> the model is wrong, expensive fix). Reporting one number for both is how you
   spend a month on the model to fix an arithmetic problem.

3. **Band coverage.** Does the band contain your range? The band claims to be the width at which
   two thirds of people agree, so at the 2/3 setting roughly two thirds of your ranges should sit
   inside it. Much higher and the band is padding; much lower and it is lying.

**Your own repeat rate is the ceiling, not 100%.** `make-pairs` re-asks a fraction of pairs later in
the queue; if you only agree with yourself 80% of the time then 80% is the score to beat, and any
number quoted against 100% is a number quoted against an impossible target. This is the same
correction that moved the panel ceiling from 100% to 74.9% (log §5.8).

Usage:

    # 1. drop 50-100 photos (any source, one face each) into data/validation/set-1/photos/
    python scripts/validate_placement.py make-pairs --set set-1 --pairs 300 --repeat 0.1

    # 2. judge them in the dashboard: Inference tab -> "Validate on unseen faces"
    #    (or write judgments.jsonl yourself: {"a": ..., "b": ..., "winner": ...})

    # 3. place every photo with the production path
    python scripts/validate_placement.py score --set set-1 \
        --checkpoint checkpoints/train-v17-panel-select/best.pt \
        --context artifacts/train-v17-panel-select --ranking artifacts/bt-refit-v5-panel

    # 4. read the verdict
    python scripts/validate_placement.py report --set set-1
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from faceiq_pref.offcohort import gap_curve, pair_table, summarise  # noqa: E402
from faceiq_pref.placement import (  # noqa: E402
    DEFAULT_AGREEMENT,
    place,
    stratified_reference_ids,
    tier_of,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
VALIDATION_ROOT = Path("data/validation")


def set_dir(name: str) -> Path:
    return ROOT / VALIDATION_ROOT / name


def photo_paths(name: str) -> list[Path]:
    d = set_dir(name) / "photos"
    if not d.exists():
        raise SystemExit(f"missing {d} — create it and drop photos in.")
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


# ------------------------------------------------------------------ make-pairs


def make_pairs(name: str, n_pairs: int, repeat: float, seed: int,
               max_photos: int = 0) -> None:
    photos = photo_paths(name)
    if len(photos) < 4:
        raise SystemExit(f"only {len(photos)} photos — need at least 4 to form pairs.")
    ids = [p.name for p in photos]
    rng = random.Random(seed)

    # More faces is not better at a fixed judging budget: it is pairs *per face* that decides
    # whether a face's position is determined at all, so spreading 300 clicks over 600 faces
    # buys breadth we cannot use and loses the local BT fit. Sampled at random, not by score,
    # to keep the draw representative of what production sees.
    if max_photos and len(ids) > max_photos:
        ids = sorted(rng.sample(ids, max_photos))
        print(f"sampled {max_photos} of {len(photos)} photos to keep pairs-per-face usable")

    # Every unordered pair, sampled without replacement, so no pair is asked twice by accident and
    # the repeat set below is the only source of duplicates.
    allp = [(a, b) for i, a in enumerate(ids) for b in ids[i + 1:]]
    rng.shuffle(allp)
    chosen = allp[:n_pairs]

    n_rep = int(round(len(chosen) * repeat))
    repeats = [chosen[i] for i in rng.sample(range(len(chosen)), n_rep)] if n_rep else []
    # Flip the display order of repeats so the pair is not visually identical, which would measure
    # recall rather than judgement.
    repeats = [(b, a) for a, b in repeats]

    queue = [{"a": a, "b": b, "repeatOf": None} for a, b in chosen]
    queue += [{"a": a, "b": b, "repeatOf": f"{b}|{a}"} for a, b in repeats]

    out = set_dir(name) / "pairs.json"
    out.write_text(json.dumps(
        {"set": name, "photos": ids, "seed": seed, "pairs": queue}, indent=2
    ))
    print(f"{len(ids)} photos, {len(chosen)} pairs + {len(repeats)} repeats -> {out}")
    print(f"total to judge: {len(queue)}  (~{len(queue) * 4 / 60:.0f} min at 4s each)")
    if len(chosen) < 5 * len(ids):
        print(
            f"warning: {len(chosen) / len(ids):.1f} pairs per photo. Under ~5 the per-face "
            "ordering is too weakly determined to fit a reference BT, so only aggregate pairwise "
            "accuracy will be meaningful."
        )


# ---------------------------------------------------------------------- score


def score_set(
    name: str, checkpoint: str, context: str, ranking: str, gender: str,
    n_refs: int, agreement: float, normalize: bool,
) -> None:
    import torch
    from PIL import Image

    from faceiq_pref.model import PreferenceScorer
    from faceiq_pref.train import TrainConfig, build_transforms

    ckpt = torch.load(checkpoint, map_location="cpu")
    cfg = TrainConfig.from_saved(ckpt["config"])
    scorer = PreferenceScorer(cfg.backbone, pretrained=False)
    # Checkpoints hold the full siamese comparator; the scoring tower is the `scorer.` subtree.
    scorer.load_state_dict({k.removeprefix("scorer."): v for k, v in ckpt["model"].items()})
    scorer.eval()
    tf = build_transforms(cfg.backbone, cfg.image_size, augment=False)

    cohort = pd.read_csv(Path(context) / "model_scores.csv")
    cohort = cohort[cohort["gender"] == gender]
    rank = pd.read_csv(Path(ranking) / "ratings.csv")
    rank = rank[rank["gender"] == gender]

    theta_of = dict(zip(rank["faceId"], rank["theta"]))
    pct_of = dict(zip(rank["faceId"], rank["percentile"]))
    score_of = dict(zip(cohort["faceId"], cohort["modelScore"]))
    population_thetas = list(theta_of.values())

    usable = {f: p for f, p in pct_of.items() if f in score_of}
    ref_ids = stratified_reference_ids(usable, n=n_refs)
    ref_thetas = [theta_of[f] for f in ref_ids]
    ref_scores = [score_of[f] for f in ref_ids]

    normalizer = None
    if normalize:
        try:
            from faceiq_pref.preprocess import normalize_front_photo

            normalizer = normalize_front_photo
        except ImportError:
            print("mediapipe missing — scoring uncropped. Placements will not be comparable to "
                  "the cohort, whose photos were all cropped.", file=sys.stderr)

    rows = []
    for p in photo_paths(name):
        img = Image.open(p).convert("RGB")
        cropped = normalizer(img) if normalizer else None
        if normalizer and cropped is None:
            print(f"  no face detected in {p.name} — skipped", file=sys.stderr)
            continue
        with torch.no_grad():
            s = scorer(tf(cropped or img).unsqueeze(0)).item()
        pl = place(s, ref_scores, ref_thetas,
                   population_thetas=population_thetas, agreement=agreement)
        rows.append({
            "photo": p.name, "modelScore": s, "theta": pl.theta, "se": pl.se,
            "bounded": pl.bounded, "percentile": pl.percentile, "scoreOutOf10": pl.score_ten,
            "bandLow": pl.band_low, "bandHigh": pl.band_high, "tier": pl.tier,
        })

    df = pd.DataFrame(rows)
    out = set_dir(name) / "placements.csv"
    df.to_csv(out, index=False)
    meta = {
        "checkpoint": checkpoint, "context": context, "ranking": ranking, "gender": gender,
        "references": len(ref_ids), "agreement": agreement, "normalized": normalizer is not None,
    }
    (set_dir(name) / "placements-meta.json").write_text(json.dumps(meta, indent=2))
    print(f"placed {len(df)} photos -> {out}")
    if not df.empty:
        print(f"  /10 range {df['scoreOutOf10'].min():.2f}-{df['scoreOutOf10'].max():.2f}, "
              f"median {df['scoreOutOf10'].median():.2f}, "
              f"{(~df['bounded']).sum()} unbounded")


# --------------------------------------------------------------------- report


def fit_local_bt(judgments: list[dict], photos: list[str], iters: int = 500) -> dict[str, float]:
    """Bradley-Terry on the validation judgments alone — your ordering, on its own scale.

    Deliberately not the cohort fit: this is *your* ranking of these photos, so Kendall tau against
    it measures agreement on ordering without importing the cohort's scale, its anchor ladder, or
    any of the VLM labels.

    Pass only photos that were **judged**. `make-pairs --max-photos` draws a subset, so a set can
    hold placed photos with no comparisons at all; those get theta straight from the +1/+2 priors,
    i.e. one shared constant, and feeding that block of artificial ties to Kendall tau drags it down
    for no reason (measured: +0.587 with 4 unjudged photos in, +0.618 with them out).
    """
    idx = {p: i for i, p in enumerate(photos)}
    wins = np.zeros((len(photos), len(photos)))
    for j in judgments:
        a, b, w = j["a"], j["b"], j["winner"]
        if a not in idx or b not in idx or w not in (a, b):
            continue
        loser = b if w == a else a
        wins[idx[w], idx[loser]] += 1

    played = wins + wins.T
    theta = np.zeros(len(photos))
    for _ in range(iters):
        exp_t = np.exp(theta)
        # MM update (Hunter 2004): stable without a line search, and the +1 priors keep faces that
        # won or lost everything from running to +-inf on a set this small.
        expected = played * exp_t[:, None] / (exp_t[:, None] + exp_t[None, :] + 1e-12)
        num = wins.sum(1) + 1.0
        den = expected.sum(1) + 2.0 * exp_t / (exp_t + 1.0)
        theta = np.log(np.clip(num, 1e-9, None)) - np.log(np.clip(den, 1e-9, None)) + theta
        theta -= theta.mean()
    return dict(zip(photos, theta))


def report(name: str, agreement: float) -> None:
    d = set_dir(name)
    if not (d / "placements.csv").exists():
        raise SystemExit(f"no placements at {d / 'placements.csv'} — run `score` first.")
    placements = pd.read_csv(d / "placements.csv")
    if not (d / "judgments.jsonl").exists():
        raise SystemExit(f"no judgments at {d / 'judgments.jsonl'} — label some pairs first.")
    judgments = [json.loads(ln) for ln in (d / "judgments.jsonl").read_text().splitlines() if ln]
    ten = dict(zip(placements["photo"], placements["scoreOutOf10"]))

    # ---- 1+2. ceiling, ordering, the band — all from the shared pair table ------
    # Same join the dashboard's review view renders, so a pair marked green there is a pair counted
    # here. `pair_table` is where skips, placed-score ties and repeats are separated out.
    table = pair_table(judgments, ten)
    out: dict = {"set": name, "photos": len(placements)}
    out.update(summarise(table, agreement))
    table.to_csv(d / "pair-review.csv", index=False)
    curve = gap_curve(table)
    out["gapCurve"] = curve.to_dict("records") if not curve.empty else []

    # ---- 3. Kendall tau against a BT fit on your judgments ---------------------
    scored = [j for j in judgments if j.get("winner")]
    photos = sorted({p for j in scored for p in (j["a"], j["b"])} & set(ten))
    out["photosJudged"] = len(photos)
    if photos and len(scored) >= 3 * len(photos):
        from scipy.stats import kendalltau, spearmanr

        yours = fit_local_bt(scored, photos)
        a = [yours[p] for p in photos]
        b = [ten[p] for p in photos]
        out["kendallTauVsYourBt"] = float(kendalltau(a, b).statistic)
        out["spearmanVsYourBt"] = float(spearmanr(a, b).statistic)
    else:
        out["kendallTauVsYourBt"] = None
        out["note"] = (
            f"only {len(scored)} pairs for {len(photos)} photos — too few for a per-face BT fit, "
            "so rank correlation is omitted. Aggregate accuracy above is still valid."
        )

    # ---- 4. hand ranges: shift vs spread, and band coverage --------------------
    rng_file = d / "ranges.json"
    if rng_file.exists():
        ranges = json.loads(rng_file.read_text())
        rows = []
        for p, r in ranges.items():
            if p not in ten:
                continue
            lo, hi = float(r[0]), float(r[1])
            pl = placements[placements["photo"] == p].iloc[0]
            rows.append({
                "photo": p, "mid": (lo + hi) / 2, "placed": ten[p],
                "overlap": not (pl["bandHigh"] < lo or pl["bandLow"] > hi),
                "inRange": lo <= ten[p] <= hi,
                "yourWidth": hi - lo,
            })
        r = pd.DataFrame(rows)
        if not r.empty:
            err = r["placed"] - r["mid"]
            shift = float(err.median())
            out["handLabelled"] = len(r)
            out["scaleShiftPoints"] = shift
            out["spreadAfterShiftPoints"] = float((err - shift).abs().median())
            out["rawErrorPoints"] = float(err.abs().median())
            out["bandOverlapsYourRange"] = float(r["overlap"].mean())
            out["placedInsideYourRange"] = float(r["inRange"].mean())
            out["yourMedianRangeWidth"] = float(r["yourWidth"].median())
            out["tierAgreement"] = float(np.mean([
                tier_of(m)[0] == tier_of(p)[0] for m, p in zip(r["mid"], r["placed"])
            ]))

    (d / "report.json").write_text(json.dumps(out, indent=2))
    _print_report(out)


def _print_report(o: dict) -> None:
    def pct(v):
        return "  n/a" if v is None else f"{v:6.1%}"

    def ci(key):
        v = o.get(key)
        return "" if not v else f"  [{v[0]:.1%}, {v[1]:.1%}]"

    print(f"\n=== validation: {o['set']} ===")
    print(f"{o['photos']} unseen photos, {o['judgments']} judgments -> "
          f"{o.get('pairsUsed', 0)} scored pairs "
          f"({o.get('freshPairs', 0)} fresh, minus {o.get('skipped', 0)} skipped and "
          f"{o.get('tiedOnPlacedScore', 0)} tied on the placed score)\n")

    print("1. ORDERING — does the placed score reproduce your choices?")
    print(f"   your self-agreement on repeats  {pct(o.get('yourSelfAgreement'))}"
          f"{ci('yourSelfAgreementCi95')}   <- the ceiling "
          f"({o.get('yourRepeatPairs', 0)} repeat pairs)")
    print(f"   system vs you                   {pct(o.get('orderingAccuracy'))}"
          f"{ci('orderingAccuracyCi95')}")
    if o.get("shareOfCeiling") is not None:
        print(f"   share of the ceiling captured   {pct(o['shareOfCeiling'])}")
    print("   coin flip                        50.0%")
    if o.get("medianGapPoints") is not None:
        # The log's §5.8 rule: an accuracy without its pair distribution is not a number. This draw
        # is uniform over the set, so it is the easy end — near-tie draws score ~30 pts lower.
        print(f"   pair draw: uniform over the set, median placed gap "
              f"{o['medianGapPoints']:.2f} /10\n")

    g = o.get("resolvableGapPoints")
    print(f"2. THE BAND — pairs closer than {g:.2f} /10 are the ones we claim *the population* "
          f"does not resolve")
    print(f"   closer than that   {pct(o.get('accuracy_withinBand'))}"
          f"  ({o.get('pairs_withinBand', 0)} pairs)")
    print(f"   further apart      {pct(o.get('accuracy_beyondBand'))}"
          f"  ({o.get('pairs_beyondBand', 0)} pairs)  <- should be high")
    if o.get("predictedAgreement") is not None:
        print(f"   agreement the curve predicted {pct(o['predictedAgreement'])}, "
              f"observed {pct(o['observedAgreement'])}")
    print("   Read these against the curve, not against 50%: T was fitted on panel *ballots*, so it")
    print("   predicts a randomly drawn rater. One self-consistent rater beats that by construction,")
    print("   so within-band accuracy above chance is expected. The failure to look for is a FLAT")
    print("   curve below — no rise with the gap means the score cannot say when to trust it.")
    for r in o.get("gapCurve", []):
        print(f"     gap {r['gapLow']:5.2f}-{r['gapHigh']:5.2f}  n={r['pairs']:3d}   "
              f"you {r['accuracy']:6.1%}   curve predicted {r['predicted']:6.1%}")
    print()

    if o.get("kendallTauVsYourBt") is not None:
        print("3. RANKING")
        print(f"   Kendall tau vs a BT fit on your judgments   {o['kendallTauVsYourBt']:+.3f}")
        print(f"   Spearman                                    {o['spearmanVsYourBt']:+.3f}\n")
    elif o.get("note"):
        print(f"3. RANKING — {o['note']}\n")

    if "handLabelled" in o:
        print(f"4. CALIBRATION — {o['handLabelled']} hand-ranged faces")
        print(f"   scale shift (fixable by re-anchoring)  {o['scaleShiftPoints']:+.2f} /10")
        print(f"   spread after removing the shift         {o['spreadAfterShiftPoints']:.2f} /10"
              "   <- this is the model's error")
        print(f"   raw median error                        {o['rawErrorPoints']:.2f} /10")
        print(f"   band overlaps your range               {pct(o['bandOverlapsYourRange'])}"
              f"   <- ~{o.get('placedInsideYourRange', 0):.0%} point-inside")
        print(f"   same tier as your range's midpoint     {pct(o['tierAgreement'])}")
        print(f"   your own median range width             {o['yourMedianRangeWidth']:.2f} /10"
              "   <- compare to the band width\n")

    verdict = []
    share = o.get("shareOfCeiling")
    if share is not None:
        if share > 1.05:
            # The model agreeing with you more than you agree with yourself is not a triumph; it
            # means your repeat noise, not the model, is what the number is measuring.
            verdict.append(
                f"share of ceiling is {share:.0%} — above 100% means your own repeat noise "
                "dominates, so judge more carefully or add repeats before trusting this"
            )
        elif share > 0.9:
            verdict.append("ordering generalises off-cohort")
        else:
            # Not "it degrades": that reading needs a comparable cohort-internal number, and a
            # single rater's ceiling is a far harsher denominator than the panel's 74.9% crowd
            # ceiling, so this ratio is not comparable to any share-of-ceiling in the log.
            verdict.append(
                f"{o['orderingAccuracy']:.1%} against your own {o['yourSelfAgreement']:.1%} "
                f"ceiling leaves {(1 - share):.0%} of the headroom on the table — real model error "
                "on pairs you resolve consistently, but do not read it as 'degrades off-cohort' "
                "without a cohort-internal number on this same pair draw and label source"
            )
    if o.get("yourRepeatPairs", 0) < 40:
        verdict.append(
            f"only {o.get('yourRepeatPairs', 0)} repeat pairs, so the ceiling itself has a wide "
            "interval (±~8pp) and every ratio against it inherits that"
        )
    if o.get("scaleShiftPoints") is not None and abs(o["scaleShiftPoints"]) > 0.4:
        verdict.append(
            f"the scale is shifted by {o['scaleShiftPoints']:+.2f} — re-anchor before touching "
            "the model"
        )
    if verdict:
        print("VERDICT: " + "; ".join(verdict))


# ----------------------------------------------------------------------- cli


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    mp = sub.add_parser("make-pairs", help="build the judging queue")
    mp.add_argument("--set", required=True)
    mp.add_argument("--pairs", type=int, default=300)
    mp.add_argument("--max-photos", type=int, default=0,
                    help="cap the faces drawn from (0 = all); keeps pairs-per-face usable")
    mp.add_argument("--repeat", type=float, default=0.10,
                    help="fraction re-asked to measure your own consistency (the ceiling)")
    mp.add_argument("--seed", type=int, default=20260804)

    sp = sub.add_parser("score", help="place every photo through the production path")
    sp.add_argument("--set", required=True)
    sp.add_argument("--checkpoint", required=True)
    sp.add_argument("--context", required=True, help="artifacts/<run> with model_scores.csv")
    sp.add_argument("--ranking", default="artifacts/bt-refit-v5-panel")
    sp.add_argument("--gender", default="female", choices=["female", "male"])
    sp.add_argument("--references", type=int, default=200)
    sp.add_argument("--agreement", type=float, default=DEFAULT_AGREEMENT)
    sp.add_argument("--no-normalize", action="store_true")

    rp = sub.add_parser("report", help="print the verdict")
    rp.add_argument("--set", required=True)
    rp.add_argument("--agreement", type=float, default=DEFAULT_AGREEMENT)

    a = ap.parse_args()
    if a.cmd == "make-pairs":
        make_pairs(a.set, a.pairs, a.repeat, a.seed, a.max_photos)
    elif a.cmd == "score":
        score_set(a.set, a.checkpoint, a.context, a.ranking, a.gender,
                  a.references, a.agreement, not a.no_normalize)
    else:
        report(a.set, a.agreement)


if __name__ == "__main__":
    main()
