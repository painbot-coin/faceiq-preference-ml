#!/usr/bin/env python
"""Per-face confidence for the BT ranking — which faces the data can actually place.

`bt.stability_check` reports ONE number per gender (80% subsample Spearman, 0.9931 for
v4-panel). A global rho that high is fully compatible with individual faces being badly
misplaced, and nothing in the pipeline says which ones. That gap is why the ranking can
only be audited by eye: a face that looks "ranked too low" might be a real error or might
be a face with a 30-percentile-wide interval, and today those are indistinguishable.

Two questions, deliberately answered separately rather than blended into one interval,
because they have different causes and different fixes:

1. `resample` — **sampling variability.** Bootstrap the comparison rows with replacement
   and refit. Answers "if we had drawn a different, equally sized set of comparisons,
   where would this face land?" Each face has only ~35 comparisons, so this is the
   dominant term for most faces.

2. `relabel` — **label sensitivity.** Keep the comparison set, and flip VLM labels at the
   rate we have MEASURED them to disagree with the human crowd, per percentile-gap band
   (about 1 in 2 for same-decile pairs). 41% of all comparisons are same-decile, so a
   large share of the graph is near-coin-flip evidence that BT currently treats as
   certain. Read this as an UPPER BOUND on label-driven movement, not a confidence
   interval: flipping independently at rate e adds noise rather than redrawing the label,
   since we do not know the truth. Panel pairs are immune — `build_cells` drops the VLM
   label wherever we bought human votes — which is itself the point.

Neither bootstrap can express the third problem, so it is reported as a flag instead.
An undefeated face has no losses in ANY resample, so its interval comes out narrow and
high, implying confidence we do not have: with ~35 comparisons and alpha=0.01, theta for
an undefeated face is bounded only by regularisation (v4-panel's top face sits at theta
18.04 against a 99th percentile of 8.73). Those faces are `notIdentified`, and their
rank relative to each other is not evidence about faces.

Usage:
    python scripts/bt_uncertainty.py \
        --export data/exports/cmr1mr0m7000196d57zi3vcgn \
        --panel-results labels/panel-pilot/results \
        --panel-meta    labels/panel-pilot/sample-meta.json \
        --panel-results labels/panel-run-3/results \
        --panel-meta    labels/panel-run-3/sample-meta.json \
        --rejects artifacts/panel-run-v1/reject-pids.txt \
                  artifacts/panel-run-v3/reject-pids.txt \
        --exclude-faces   artifacts/face-qc-v1/exclude-faces.csv \
        --exclude-genders artifacts/face-qc-v1/gender-fixes.csv \
        --ratings artifacts/bt-refit-v4-panel/ratings.csv \
        --out     artifacts/bt-refit-v4-panel
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from faceiq_pref.bt_panel import build_cells, fit_joint_bt  # noqa: E402
from faceiq_pref.calibrate import percentile_ranks, score_out_of_10  # noqa: E402
from faceiq_pref.data import Matchup, load_export  # noqa: E402
from refit_bt_panel import load_pooled_votes  # noqa: E402

# Same bands as eval_vs_panel.py so uncertainty and accuracy can be read side by side.
BANDS = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 45), (45, 101)]
WIDE_CI_PCT = 20.0    # percentile points; wider than this and "too low" is unfalsifiable
MIN_EFFECTIVE = 5.0   # effective comparisons; below this a face is placed by ~nothing
HUMAN_VOTE_ERROR = 0.409  # 1 - 0.591 panel ceiling: a single human vote vs the crowd


def band_of(gap_pct: float) -> str:
    for lo, hi in BANDS:
        if lo <= gap_pct < hi:
            return f"{lo}-{min(hi, 100)}"
    return f"{BANDS[-1][0]}-100"


def vlm_error_rates(
    matchups: list[Matchup],
    votes: dict[int, tuple[float, float]],
    pct: dict[str, float],
) -> tuple[dict[str, float], float, dict[str, int]]:
    """Measured VLM-vs-crowd disagreement per percentile-gap band.

    Only pairs where the panel voted can answer this, and only those with a crowd
    majority — a pair the crowd split 6-6 has no "right" answer to disagree with.
    """
    hit: dict[str, list[int]] = defaultdict(list)
    for m in matchups:
        v = votes.get(m.pair_index)
        if not v or m.final_outcome not in ("A", "B"):
            continue
        wa, wb = v
        if wa == wb:
            continue
        if m.face_a_id not in pct or m.face_b_id not in pct:
            continue
        gap = abs(pct[m.face_a_id] - pct[m.face_b_id]) * 100.0
        crowd_a = wa > wb
        hit[band_of(gap)].append(int((m.final_outcome == "A") != crowd_a))
    rates = {b: float(np.mean(v)) for b, v in hit.items() if v}
    counts = {b: len(v) for b, v in hit.items()}
    allv = [x for v in hit.values() for x in v]
    return rates, (float(np.mean(allv)) if allv else 0.0), counts


def reliability(err: float) -> float:
    """Information a noisy binary label carries, relative to a perfect one.

    A label that reports sign(theta_i - theta_j) with flip probability e carries
    (1 - 2e)^2 of the Fisher information of a noiseless label, so e = 0.5 is worth
    literally nothing and e = 0.01 is worth ~0.96 of a perfect comparison.

    Read the resulting `effective*` columns as a LOWER bound on information. `e` is
    measured as disagreement with a ~12-vote crowd majority, and on close pairs that
    majority is itself near-arbitrary (a 7-5 split is a 58% crowd), so disagreeing with
    it is not the same as being wrong. Note this does not say close pairs are worthless:
    twelve human votes on a close pair still sum to real information about a small gap,
    which is why `beta_human` was estimable at all. It says a SINGLE near-chance label is
    worth almost nothing, and 41% of the VLM graph is exactly that.
    """
    return (1.0 - 2.0 * err) ** 2


def fit_once(rows: list[Matchup], votes, gender: str) -> dict[str, float]:
    fi, cells, counts, obs, nv, nh, df, dm = build_cells(rows, votes, gender)
    res = fit_joint_bt(fi, cells, counts, obs, gender, nv, nh, df, dm)
    return res.theta


def summarise(samples: list[float], lo: float = 2.5, hi: float = 97.5):
    if not samples:
        return None, None, None
    a = np.asarray(samples, dtype=float)
    return float(np.percentile(a, lo)), float(np.percentile(a, hi)), float(a.std())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--panel-results", action="append", default=[])
    ap.add_argument("--panel-meta", action="append", default=[])
    ap.add_argument("--rejects", nargs="*", default=[])
    ap.add_argument("--exclude-faces", default="artifacts/face-qc-v1/exclude-faces.csv")
    ap.add_argument("--exclude-genders", default="artifacts/face-qc-v1/gender-fixes.csv")
    ap.add_argument("--ratings", required=True,
                    help="the ranking of record; supplies the point estimate and gap bands")
    ap.add_argument("--replicates", type=int, default=200)
    ap.add_argument("--mode", choices=["resample", "relabel", "both"], default="both")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="artifacts/bt-refit-v4-panel")
    args = ap.parse_args()

    export = load_export(args.export)
    matchups = export.all_matchups()

    dropped: set[str] = set()
    for p in (args.exclude_faces, args.exclude_genders):
        if p and Path(p).exists():
            with open(p, newline="") as fh:
                dropped.update(r["faceId"] for r in csv.DictReader(fh))
    matchups = [m for m in matchups
                if m.face_a_id not in dropped and m.face_b_id not in dropped]

    if len(args.panel_results) != len(args.panel_meta):
        raise SystemExit("--panel-results and --panel-meta must be the same length")
    votes, _ = load_pooled_votes(
        [(Path(r), Path(m)) for r, m in zip(args.panel_results, args.panel_meta)],
        [Path(p) for p in args.rejects],
    )

    with open(args.ratings, newline="") as fh:
        rating_rows = list(csv.DictReader(fh))
    point = {r["faceId"]: r for r in rating_rows}
    pct0 = {r["faceId"]: float(r["percentile"]) for r in rating_rows}

    print(f"{len(matchups):,} matchups after QC ({len(dropped)} faces excluded), "
          f"{len(votes):,} panel pairs")

    err_by_band, err_all, band_n = vlm_error_rates(matchups, votes, pct0)
    print("\nmeasured VLM-vs-crowd disagreement, by percentile gap "
          "(the label noise BT currently treats as certain):")
    for lo, hi in BANDS:
        b = f"{lo}-{min(hi, 100)}"
        if b in err_by_band:
            print(f"  {b:>7}: {err_by_band[b]:6.1%} wrong   (n={band_n[b]:,})")
    print(f"  overall: {err_all:.1%} on panel-covered pairs")

    # ---- per-face record and opponent quality ---------------------------------
    wins: dict[str, int] = defaultdict(int)
    losses: dict[str, int] = defaultdict(int)
    ties: dict[str, int] = defaultdict(int)
    same_dec: dict[str, int] = defaultdict(int)
    n_comp: dict[str, int] = defaultdict(int)
    eff_vlm: dict[str, float] = defaultdict(float)
    eff_panel: dict[str, float] = defaultdict(float)
    faces = export.faces()
    for m in matchups:
        for fid in (m.face_a_id, m.face_b_id):
            n_comp[fid] += 1
        fa, fb = faces.get(m.face_a_id), faces.get(m.face_b_id)
        if fa and fb and fa.decile_bin == fb.decile_bin:
            same_dec[m.face_a_id] += 1
            same_dec[m.face_b_id] += 1

        v = votes.get(m.pair_index)
        if v and (v[0] + v[1]) > 0:
            # Panel pairs: the VLM label is discarded by build_cells, and each human
            # vote is one noisy observation, so information is n_votes * reliability.
            w = (v[0] + v[1]) * reliability(HUMAN_VOTE_ERROR)
            eff_panel[m.face_a_id] += w
            eff_panel[m.face_b_id] += w
        else:
            gap = (abs(pct0[m.face_a_id] - pct0[m.face_b_id]) * 100.0
                   if m.face_a_id in pct0 and m.face_b_id in pct0 else None)
            e = err_by_band.get(band_of(gap), err_all) if gap is not None else err_all
            w = reliability(e)
            eff_vlm[m.face_a_id] += w
            eff_vlm[m.face_b_id] += w

        if m.final_outcome == "A":
            wins[m.face_a_id] += 1
            losses[m.face_b_id] += 1
        elif m.final_outcome == "B":
            wins[m.face_b_id] += 1
            losses[m.face_a_id] += 1
        else:
            ties[m.face_a_id] += 1
            ties[m.face_b_id] += 1

    genders = sorted({m.gender for m in matchups})
    modes = ["resample", "relabel"] if args.mode == "both" else [args.mode]

    # faceId -> mode -> list of per-replicate values
    th_s: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    pc_s: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    rk_s: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for gender in genders:
        rows = [m for m in matchups if m.gender == gender]
        print(f"\n== {gender} == {len(rows):,} comparisons")
        for mode in modes:
            rng = random.Random(args.seed + hash(gender + mode) % 10_000)
            done = 0
            for rep in range(args.replicates):
                if mode == "resample":
                    sample = [rows[rng.randrange(len(rows))] for _ in range(len(rows))]
                else:
                    sample = []
                    for m in rows:
                        # Panel-covered pairs keep their outcome: build_cells discards the
                        # VLM label there anyway, so flipping it would change nothing.
                        if m.pair_index in votes or m.final_outcome not in ("A", "B"):
                            sample.append(m)
                            continue
                        gap = (abs(pct0[m.face_a_id] - pct0[m.face_b_id]) * 100.0
                               if m.face_a_id in pct0 and m.face_b_id in pct0 else None)
                        e = err_by_band.get(band_of(gap), err_all) if gap is not None else err_all
                        if rng.random() < e:
                            flip = "B" if m.final_outcome == "A" else "A"
                            sample.append(replace(m, final_outcome=flip))
                        else:
                            sample.append(m)
                try:
                    theta = fit_once(sample, votes, gender)
                except Exception as exc:  # a degenerate resample is data, not a crash
                    print(f"    replicate {rep} skipped: {exc}")
                    continue
                pcts = percentile_ranks(theta)
                order = sorted(theta, key=lambda f: -theta[f])
                rank = {f: k + 1 for k, f in enumerate(order)}
                for fid, t in theta.items():
                    th_s[fid][mode].append(t)
                    pc_s[fid][mode].append(pcts[fid] * 100.0)
                    rk_s[fid][mode].append(float(rank[fid]))
                done += 1
                if (rep + 1) % 50 == 0:
                    print(f"    {mode}: {rep + 1}/{args.replicates}")
            print(f"    {mode}: {done} usable replicates")

    # ---- write per-face table -------------------------------------------------
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["faceId", "gender", "theta", "percentile", "scoreOutOf10", "comparisonCount",
            "wins", "losses", "ties", "sameDecileShare", "effectiveComparisons",
            "effectiveVlm", "effectivePanel", "decileBin", "labsOverallScore",
            "imagePath", "replicates"]
    for mode in modes:
        cols += [f"{mode}PctLo", f"{mode}PctHi", f"{mode}PctWidth",
                 f"{mode}ScoreLo", f"{mode}ScoreHi",
                 f"{mode}RankLo", f"{mode}RankHi", f"{mode}ThetaSd"]
    cols += ["notIdentified", "wideInterval", "lowInfoOpponents", "flags"]

    wide_n = defaultdict(int)
    rows_out = []
    for fid, pr in point.items():
        if fid not in pc_s:
            continue
        n = n_comp.get(fid, 0)
        row = {
            "faceId": fid, "gender": pr["gender"], "theta": pr["theta"],
            "percentile": pr["percentile"], "scoreOutOf10": pr["scoreOutOf10"],
            "comparisonCount": pr["comparisonCount"],
            "wins": wins[fid], "losses": losses[fid], "ties": ties[fid],
            "sameDecileShare": round(same_dec[fid] / n, 4) if n else "",
            "effectiveComparisons": round(eff_vlm[fid] + eff_panel[fid], 3),
            "effectiveVlm": round(eff_vlm[fid], 3),
            "effectivePanel": round(eff_panel[fid], 3),
            "decileBin": pr["decileBin"], "labsOverallScore": pr["labsOverallScore"],
            "imagePath": pr["imagePath"],
            "replicates": len(pc_s[fid].get(modes[0], [])),
        }
        for mode in modes:
            plo, phi, _ = summarise(pc_s[fid].get(mode, []))
            _, _, tsd = summarise(th_s[fid].get(mode, []))
            rlo, rhi, _ = summarise(rk_s[fid].get(mode, []))
            row[f"{mode}PctLo"] = round(plo, 3) if plo is not None else ""
            row[f"{mode}PctHi"] = round(phi, 3) if phi is not None else ""
            row[f"{mode}PctWidth"] = round(phi - plo, 3) if plo is not None else ""
            row[f"{mode}ScoreLo"] = (round(score_out_of_10(plo / 100.0), 3)
                                     if plo is not None else "")
            row[f"{mode}ScoreHi"] = (round(score_out_of_10(phi / 100.0), 3)
                                     if phi is not None else "")
            row[f"{mode}RankLo"] = int(rlo) if rlo is not None else ""
            row[f"{mode}RankHi"] = int(rhi) if rhi is not None else ""
            row[f"{mode}ThetaSd"] = round(tsd, 4) if tsd is not None else ""
            if plo is not None and phi - plo > WIDE_CI_PCT:
                wide_n[mode] += 1

        # An undefeated (or once-beaten) face cannot be placed from above: no resample
        # gives it a loss, so its interval understates the truth. Say so explicitly.
        not_ident = losses[fid] <= 1 and wins[fid] > 0
        widest = max((row[f"{m}PctWidth"] for m in modes
                      if row[f"{m}PctWidth"] != ""), default=0.0)
        wide = widest > WIDE_CI_PCT
        low_info = (eff_vlm[fid] + eff_panel[fid]) < MIN_EFFECTIVE
        flags = [name for name, on in
                 (("notIdentified", not_ident), ("wideInterval", wide),
                  ("lowInfoOpponents", low_info)) if on]
        row["notIdentified"] = int(not_ident)
        row["wideInterval"] = int(wide)
        row["lowInfoOpponents"] = int(low_info)
        row["flags"] = "|".join(flags)
        rows_out.append(row)

    rows_out.sort(key=lambda r: (r["gender"], -float(r["theta"])))
    csv_path = out_dir / "uncertainty.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows_out)

    # ---- summary --------------------------------------------------------------
    eff = np.array([float(r["effectiveComparisons"]) for r in rows_out])
    raw = np.array([float(r["comparisonCount"]) for r in rows_out])
    summary = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "ranking": args.ratings,
        "replicates": args.replicates,
        "modes": modes,
        "faces": len(rows_out),
        "wideIntervalThresholdPct": WIDE_CI_PCT,
        "minEffectiveComparisons": MIN_EFFECTIVE,
        "vlmErrorByBand": err_by_band,
        "vlmErrorOverall": err_all,
        "vlmErrorBandN": band_n,
        "medianComparisons": float(np.median(raw)),
        "medianEffectiveComparisons": float(np.median(eff)),
        "medianEffectiveVlm": float(np.median([float(r["effectiveVlm"]) for r in rows_out])),
        "medianEffectivePanel": float(np.median([float(r["effectivePanel"]) for r in rows_out])),
        "notIdentified": sum(r["notIdentified"] for r in rows_out),
        "lowInfoOpponents": sum(r["lowInfoOpponents"] for r in rows_out),
        "byMode": {},
    }
    print("\n" + "=" * 72)
    for mode in modes:
        widths = [float(r[f"{mode}PctWidth"]) for r in rows_out
                  if r[f"{mode}PctWidth"] != ""]
        ni = [float(r[f"{mode}PctWidth"]) for r in rows_out
              if r[f"{mode}PctWidth"] != "" and r["notIdentified"]]
        ok = [float(r[f"{mode}PctWidth"]) for r in rows_out
              if r[f"{mode}PctWidth"] != "" and not r["notIdentified"]]
        summary["byMode"][mode] = {
            "medianPctWidth": float(np.median(widths)) if widths else None,
            "p90PctWidth": float(np.percentile(widths, 90)) if widths else None,
            "wideIntervalFaces": wide_n[mode],
            "medianPctWidthNotIdentified": float(np.median(ni)) if ni else None,
            "medianPctWidthIdentified": float(np.median(ok)) if ok else None,
        }
        if widths:
            print(f"{mode:9}: median 95% interval spans "
                  f"{np.median(widths):.1f} percentile points, "
                  f"p90 {np.percentile(widths, 90):.1f}; "
                  f"{wide_n[mode]:,} faces wider than {WIDE_CI_PCT:.0f}")

    print(f"\ninformation: median {np.median(raw):.0f} comparisons per face, but only "
          f"{np.median(eff):.1f} effective "
          f"({summary['medianEffectiveVlm']:.1f} VLM + "
          f"{summary['medianEffectivePanel']:.1f} panel)")
    print(f"{'flags':12}: {summary['notIdentified']} faces not identified from above "
          f"(<=1 loss), {summary['lowInfoOpponents']:,} under "
          f"{MIN_EFFECTIVE:.0f} effective comparisons")
    ni_w = summary["byMode"][modes[0]]["medianPctWidthNotIdentified"]
    ok_w = summary["byMode"][modes[0]]["medianPctWidthIdentified"]
    if ni_w is not None and ok_w is not None and ni_w < ok_w:
        print(f"\nNOTE: the {summary['notIdentified']} not-identified faces show a "
              f"median interval of {ni_w:.1f} points vs {ok_w:.1f} for the rest. "
              f"That is an artifact, not confidence:\n      no resample can hand an "
              f"undefeated face a loss, so trust the flag over the interval for these.")

    (out_dir / "uncertainty.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {csv_path}")
    print(f"wrote {out_dir / 'uncertainty.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
