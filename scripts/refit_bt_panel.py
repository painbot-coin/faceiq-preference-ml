#!/usr/bin/env python
"""Refit BT absorbing the panel's graded human votes alongside the VLM labels.

The panel put ~12 votes on each of 2,980 pairs. Those vote *shares* carry information
plain BT never saw: which face wins, and by how much. This refit takes each vote as one
observation under its own discrimination (see `bt_panel.py`), so a 7-5 split pulls two
faces together while a 12-0 pushes them apart.

Held-out check, because fitting on the votes and then scoring against them would be
circular: half the panel's raters are withheld, the model is fit on the other half, and
both the VLM-only and joint fits are scored on the withheld half.

Usage:
    python scripts/refit_bt_panel.py --export data/exports/<runId> \
        --exclude-faces   artifacts/face-qc-v1/exclude-faces.csv \
        --exclude-genders artifacts/face-qc-v1/gender-fixes.csv \
        --out artifacts/bt-refit-v3-panel
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from faceiq_pref.bt import MIN_COMPARISONS, STABILITY_RHO_GATE  # noqa: E402
from faceiq_pref.bt_panel import build_cells, fit_joint_bt  # noqa: E402
from faceiq_pref.calibrate import calibrate  # noqa: E402
from faceiq_pref.data import load_export  # noqa: E402
from faceiq_pref.validate import spearman_vs_labs  # noqa: E402

TEST_STUDY = "test"


def load_panel_votes(results: Path, meta_path: Path, reject_path: Path | None,
                     rater_subset: set[str] | None = None,
                     reject_paths: list[Path] | None = None):
    """-> {pair_index: (wins for faceA, wins for faceB)}, ties split half each."""
    meta = json.loads(meta_path.read_text())["pairs"]
    by_pair_id = {m["pairId"]: m for m in meta}

    rejects: set[str] = set()
    for p in list(reject_paths or []) + ([reject_path] if reject_path else []):
        if p and p.exists():
            rejects |= set(p.read_text().split())

    tally: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])
    used_raters: set[str] = set()
    for line in (results / "judgments.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        j = json.loads(line)
        if j["studyId"] == TEST_STUDY or j["isGold"]:
            continue
        pid = j["prolificPid"]
        if pid in rejects:
            continue
        if rater_subset is not None and pid not in rater_subset:
            continue
        m = by_pair_id.get(j["pairId"])
        if m is None:
            continue
        used_raters.add(pid)
        cell = tally[m["pairIndex"]]
        a_side = "left" if m["faceAOnLeft"] else "right"
        if j["choice"] == "tie":
            cell[0] += 0.5
            cell[1] += 0.5
        elif j["choice"] == a_side:
            cell[0] += 1.0
        else:
            cell[1] += 1.0
    return {k: (v[0], v[1]) for k, v in tally.items()}, used_raters


def load_pooled_votes(runs: list[tuple[Path, Path]], reject_paths: list[Path],
                      rater_subset: set[str] | None = None):
    """Merge several studies' votes into one panel, keyed on the export's pairIndex.

    Each run ships its own pairs.json/sample-meta.json, but pairIndex is the export's
    stable id for a matchup, so pooling is a sum per pair rather than a join: disjoint
    draws union, and a pair bought twice for extra depth accumulates votes. That is why
    adding a run needs no schema change and no re-analysis of earlier runs.
    """
    pooled: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])
    raters: set[str] = set()
    for results, meta in runs:
        votes, used = load_panel_votes(results, meta, None, rater_subset,
                                       reject_paths=reject_paths)
        for idx, (wa, wb) in votes.items():
            cell = pooled[idx]
            cell[0] += wa
            cell[1] += wb
        raters |= used
    return {k: (v[0], v[1]) for k, v in pooled.items()}, raters


def all_raters(results_dirs: list[Path], reject_paths: list[Path]) -> list[str]:
    rejects: set[str] = set()
    for p in reject_paths:
        if p.exists():
            rejects |= set(p.read_text().split())
    pids = set()
    for results in results_dirs:
        for line in (results / "judgments.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            j = json.loads(line)
            if j["studyId"] != TEST_STUDY and j["prolificPid"] not in rejects:
                pids.add(j["prolificPid"])
    return sorted(pids)


def joint_stability(matchups, votes, gender: str, subsample: float = 0.8,
                    seed: int = 7) -> float:
    """Spearman between joint fits on two independent 80% subsamples of the pairs.

    Subsamples at pair level, like `bt.stability_check`, so the number stays comparable
    to earlier refits — a panel pair that survives keeps all of its votes.
    """
    rows = [m for m in matchups if m.gender == gender]
    rng = random.Random(seed)

    def sub(trial_seed: int) -> dict[str, float]:
        r = random.Random(trial_seed)
        sample = [m for m in rows if r.random() < subsample]
        fi, ce, co, ob, nv, nh, df, dm = build_cells(sample, votes, gender,
                                                     drop_below_min=False)
        return fit_joint_bt(fi, ce, co, ob, gender, nv, nh, df, dm).theta

    t1 = sub(rng.randint(0, 2**31))
    t2 = sub(rng.randint(0, 2**31))
    common = sorted(set(t1) & set(t2))
    rho, _ = spearmanr([t1[f] for f in common], [t2[f] for f in common])
    return float(rho)


def held_out_accuracy(theta: dict[str, float], votes, matchups, beta: float | None = None,
                      strata: dict[int, str] | None = None):
    """Weighted accuracy predicting withheld individual human votes, plus log-loss.

    Returns (accuracy, votes scored, log-loss, per-stratum accuracy).
    """
    by_index = {m.pair_index: m for m in matchups}
    correct = total = 0.0
    ll = 0.0
    n_ll = 0.0
    per: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for idx, (wa, wb) in votes.items():
        m = by_index.get(idx)
        if m is None or m.face_a_id not in theta or m.face_b_id not in theta:
            continue
        d = theta[m.face_a_id] - theta[m.face_b_id]
        if d == 0:
            continue
        hit = wa if d > 0 else wb
        correct += hit
        total += wa + wb
        if strata is not None:
            cell = per[strata.get(idx, "?")]
            cell[0] += hit
            cell[1] += wa + wb
        if beta is not None:
            p = 1.0 / (1.0 + np.exp(-np.clip(beta * d, -30, 30)))
            p = min(max(p, 1e-9), 1 - 1e-9)
            ll += wa * np.log(p) + wb * np.log(1 - p)
            n_ll += wa + wb
    by_stratum = {k: {"accuracy": v[0] / v[1], "votes": v[1]}
                  for k, v in per.items() if v[1] > 0}
    return (correct / total if total else float("nan"),
            total,
            -ll / n_ll if n_ll else float("nan"),
            by_stratum)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--panel-results", nargs="+",
                    default=["labels/panel-pilot/results"])
    ap.add_argument("--panel-meta", nargs="+",
                    default=["labels/panel-pilot/sample-meta.json"],
                    help="one sample-meta.json per --panel-results dir, same order")
    ap.add_argument("--rejects", nargs="*",
                    default=["artifacts/panel-run-v1/reject-pids.txt"])
    ap.add_argument("--exclude-faces")
    ap.add_argument("--exclude-genders")
    ap.add_argument("--baseline", default="artifacts/bt-refit-v2-qc/ratings.csv")
    ap.add_argument("--skip-stability", action="store_true", help="skip the 80%% subsample gate")
    ap.add_argument("--out", default="artifacts/bt-refit-v3-panel")
    args = ap.parse_args()

    export = load_export(args.export)
    matchups = export.all_matchups()
    print(f"export OK: run {export.run_id}, {len(matchups)} matchups")

    dropped: set[str] = set()
    for path in (args.exclude_faces, args.exclude_genders):
        if path:
            with open(path, newline="") as fh:
                dropped.update(r["faceId"] for r in csv.DictReader(fh))
    if dropped:
        before = len(matchups)
        matchups = [m for m in matchups
                    if m.face_a_id not in dropped and m.face_b_id not in dropped]
        print(f"QC exclusions: {len(dropped)} faces -> {before} -> {len(matchups)} matchups")

    if len(args.panel_results) != len(args.panel_meta):
        raise SystemExit("--panel-results and --panel-meta must be the same length")
    runs = [(Path(r), Path(m)) for r, m in zip(args.panel_results, args.panel_meta)]
    reject_paths = [Path(p) for p in args.rejects]
    votes, raters = load_pooled_votes(runs, reject_paths)
    tot = sum(a + b for a, b in votes.values())
    for r, m in runs:
        per_run, per_raters = load_pooled_votes([(r, m)], reject_paths)
        print(f"  {r}: {len(per_run)} pairs, "
              f"{sum(a + b for a, b in per_run.values()):,.0f} votes, "
              f"{len(per_raters)} raters")
    print(f"panel: {len(votes)} pairs, {tot:,.0f} votes from {len(raters)} raters "
          f"({tot/len(votes):.1f} per pair)")

    faces = export.faces()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics: dict = {
        "runId": export.run_id,
        "refitAt": datetime.now(timezone.utc).isoformat(),
        "basis": "VLM labels + panel human votes, per-source discrimination "
                 "(beta_vlm fixed at 1)",
        "panelPairs": len(votes),
        "panelVotes": round(tot),
        "panelRaters": len(raters),
        "vlmLabelsReplacedOnPanelPairs": True,
        "qcExcludedFaces": len(dropped),
        "genders": {},
    }

    rows: list[dict] = []
    gates_ok = True
    for gender in sorted({m.gender for m in matchups}):
        print(f"\n== {gender} ==")
        fi, cells, counts, obs, n_vlm, n_human, dropf, dropm = build_cells(
            matchups, votes, gender)
        res = fit_joint_bt(fi, cells, counts, obs, gender, n_vlm, n_human, dropf, dropm)
        if dropf:
            print(f"dropped {len(dropf)} face(s) under {MIN_COMPARISONS} distinct "
                  f"opponents (+{dropm} matchups): {dropf[:5]}")
        print(f"faces={len(res.theta)} cells={len(cells)} "
              f"vlm_obs={n_vlm:,.0f} human_obs={n_human:,.0f}")
        print(f"beta_human = {res.beta_human:.4f}  "
              f"(=> humans are {1/res.beta_human:.1f}x flatter than the VLM)")
        print(f"converged={res.converged} iters={res.n_iter} "
              f"components={res.n_components} below_min={len(res.faces_below_min)}")

        rho_stab = None
        if not args.skip_stability:
            rho_stab = joint_stability(matchups, votes, gender)
            print(f"stability rho = {rho_stab:.4f} (gate > {STABILITY_RHO_GATE})")

        gate = {"connected": res.connected,
                "noFaceBelowMin": len(res.faces_below_min) == 0,
                "converged": res.converged,
                "stabilityPassed": rho_stab is None or rho_stab > STABILITY_RHO_GATE}
        gates_ok = gates_ok and all(gate.values())

        cal = calibrate(res.theta)
        labs_rho, labs_n = spearman_vs_labs(res.theta, faces)
        th = np.array([res.theta[f] for f in res.theta])

        metrics["genders"][gender] = {
            "faces": len(res.theta),
            "betaHuman": res.beta_human,
            "humanVsVlmTemperature": 1 / res.beta_human,
            "vlmObservations": n_vlm,
            "humanObservations": n_human,
            "components": res.n_components,
            "droppedUnderConnectedFaces": res.dropped_faces,
            "droppedMatchups": res.dropped_matchups,
            "facesBelowMinComparisons": res.faces_below_min[:20],
            "minComparisons": MIN_COMPARISONS,
            "stabilityRho": rho_stab,
            "thetaSd": float(th.std()),
            "spearmanVsLabs": labs_rho,
            "labsScoredFaces": labs_n,
            "gates": gate,
        }

        for fid, c in cal.items():
            face = faces[fid]
            rows.append({"faceId": fid, "gender": gender,
                         "theta": round(c["theta"], 6),
                         "percentile": round(c["percentile"], 6),
                         "scoreOutOf10": round(c["scoreOutOf10"], 3),
                         "comparisonCount": counts.get(fid, 0),
                         "decileBin": face.decile_bin,
                         "labsOverallScore": face.labs_overall_score,
                         "imagePath": face.image_path})

    # ---- compare against the VLM-only baseline -------------------------------
    base = {r["faceId"]: float(r["theta"]) for r in csv.DictReader(open(args.baseline))}
    new = {r["faceId"]: r["theta"] for r in rows}
    common = sorted(set(base) & set(new))
    rho, _ = spearmanr([base[f] for f in common], [new[f] for f in common])
    base_pct = {r["faceId"]: float(r["percentile"])
                for r in csv.DictReader(open(args.baseline))}
    new_pct = {r["faceId"]: r["percentile"] for r in rows}
    moves = sorted(((abs(new_pct[f] - base_pct[f]) * 100, f) for f in common), reverse=True)
    base_score = {r["faceId"]: float(r["scoreOutOf10"])
                  for r in csv.DictReader(open(args.baseline))}
    new_score = {r["faceId"]: r["scoreOutOf10"] for r in rows}
    sdelta = [abs(new_score[f] - base_score[f]) for f in common]
    metrics["vsBaseline"] = {
        "baseline": args.baseline,
        "spearmanRho": float(rho),
        "facesCompared": len(common),
        "medianPercentileMove": float(np.median([m for m, _ in moves])),
        "p95PercentileMove": float(np.percentile([m for m, _ in moves], 95)),
        "maxPercentileMove": float(moves[0][0]),
        "medianScoreMove": float(np.median(sdelta)),
        "p95ScoreMove": float(np.percentile(sdelta, 95)),
        "maxScoreMove": float(max(sdelta)),
        "facesMovingOverHalfPoint": int(sum(1 for d in sdelta if d > 0.5)),
    }
    print(f"\nvs {args.baseline}: Spearman {rho:.4f} over {len(common)} faces")
    print(f"  percentile move: median {np.median([m for m, _ in moves]):.1f}, "
          f"p95 {np.percentile([m for m, _ in moves], 95):.1f}, max {moves[0][0]:.1f}")
    print(f"  /10 score move : median {np.median(sdelta):.2f}, "
          f"p95 {np.percentile(sdelta, 95):.2f}, max {max(sdelta):.2f}; "
          f"{sum(1 for d in sdelta if d > 0.5)} faces move > 0.5 pt")

    # ---- held-out: fit on half the raters, score on the other half -----------
    print("\n== held-out validation (disjoint rater halves) ==")
    pids = all_raters([r for r, _ in runs], reject_paths)
    random.Random(11).shuffle(pids)
    half = len(pids) // 2
    fit_pids, test_pids = set(pids[:half]), set(pids[half:])
    fit_votes, _ = load_pooled_votes(runs, reject_paths, fit_pids)
    test_votes, _ = load_pooled_votes(runs, reject_paths, test_pids)

    joint_theta: dict[str, float] = {}
    joint_beta: dict[str, float] = {}
    for gender in sorted({m.gender for m in matchups}):
        fi, ce, co, ob, nv, nh, df, dm = build_cells(matchups, fit_votes, gender)
        r = fit_joint_bt(fi, ce, co, ob, gender, nv, nh, df, dm)
        joint_theta.update(r.theta)
        joint_beta[gender] = r.beta_human

    strata = {m["pairIndex"]: m["stratum"]
              for _, mp in runs
              for m in json.loads(mp.read_text())["pairs"]}
    beta_mean = float(np.mean(list(joint_beta.values())))
    acc_new, n_ev, ll_new, by_new = held_out_accuracy(
        joint_theta, test_votes, matchups, beta_mean, strata)
    acc_old, _, _, by_old = held_out_accuracy(base, test_votes, matchups, None, strata)
    base_label = Path(args.baseline).parent.name or "baseline"
    print(f"predicting {n_ev:,.0f} withheld human votes:")
    print("  NOTE: the baseline is only an honest comparison on pairs it never saw. "
          "A panel baseline was fit on every rater, test half included.")
    print(f"  {'stratum':11} {base_label:>16} {'this fit':>9} {'gain':>7} {'votes':>8}")
    preferred = ["close", "mid", "clear", "lowconf", "overlap"]
    ordered = [s for s in preferred if s in by_new] + \
              sorted(s for s in by_new if s not in preferred)
    for st in ordered:
        o, n = by_old[st]["accuracy"], by_new[st]["accuracy"]
        print(f"  {st:11} {o:16.1%} {n:9.1%} {n-o:+7.1%} {by_new[st]['votes']:8,.0f}")
    print(f"  {'ALL':11} {acc_old:16.1%} {acc_new:9.1%} "
          f"{acc_new-acc_old:+7.1%} {n_ev:8,.0f}")
    metrics["heldOut"] = {
        "protocol": "fit on half the raters' votes, score on the disjoint half",
        "baselineCaveat": "byStratum is only honest where the baseline never saw the "
                          "pair's votes; a panel baseline was fit on all raters",
        "fitRaters": len(fit_pids), "testRaters": len(test_pids),
        "votesEvaluated": n_ev,
        "vlmOnlyAccuracy": acc_old,
        "jointAccuracy": acc_new,
        "delta": acc_new - acc_old,
        "jointLogLoss": ll_new,
        "byStratum": {st: {"vlmOnly": by_old[st]["accuracy"],
                           "joint": by_new[st]["accuracy"],
                           "votes": by_new[st]["votes"]}
                      for st in by_new if st in by_old},
    }

    rows.sort(key=lambda r: (r["gender"], -r["theta"]))
    with (out_dir / "ratings.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    metrics["allGatesPassed"] = gates_ok
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nwrote {out_dir/'ratings.csv'} ({len(rows)} faces) and metrics.json")
    print(f"GATES {'PASSED' if gates_ok else 'FAILED'}")
    return 0 if gates_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
