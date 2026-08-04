#!/usr/bin/env python
"""Score a blend of rating sources against real human votes.

Answers the question behind "should the product rating be an average of Labs, the network
and BT": does the average actually beat the best single ingredient on pairs no component
could have memorised? See `faceiq_pref.composite` for why a blend is not free lunch.

    # list what can be blended
    python scripts/composite_eval.py --export data/exports/<runId> --list

    # score a blend
    python scripts/composite_eval.py --export data/exports/<runId> \
        --panel labels/panel-pilot/results labels/panel-pilot/sample-meta.json \
        --panel labels/panel-run-3/results labels/panel-run-3/sample-meta.json \
        --rejects artifacts/panel-run-v1/reject-pids.txt \
                  artifacts/panel-run-v3/reject-pids.txt \
        --ratings artifacts/bt-refit-v4-panel/ratings.csv \
        --source train-v13-panel-hard --source train-v10-arcface-val50 \
        --out artifacts/composite-v1/report.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_vs_panel import load_votes  # noqa: E402
from faceiq_pref.calibrate import score_out_of_10  # noqa: E402
from faceiq_pref.composite import (  # noqa: E402
    METHODS,
    blend,
    discover_sources,
    evaluate_blend,
    normalise,
)
from faceiq_pref.data import load_export  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--panel", nargs=2, action="append", default=[],
                    metavar=("RESULTS", "META"))
    ap.add_argument("--rejects", nargs="*", default=[])
    ap.add_argument("--ratings", default="artifacts/bt-refit-v4-panel/ratings.csv",
                    help="ranking that defines the percentile-gap bands")
    ap.add_argument("--source", action="append", default=[],
                    help="repeatable source name, e.g. train-v13-panel-hard")
    ap.add_argument("--weight", action="append", type=float, default=[],
                    help="repeatable, parallel to --source (default: equal)")
    ap.add_argument("--method", choices=list(METHODS), default="percentile")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--exclude-faces", default="artifacts/face-qc-v1/exclude-faces.csv")
    ap.add_argument("--exclude-genders", default="artifacts/face-qc-v1/gender-fixes.csv")
    ap.add_argument("--list", action="store_true", help="print blendable sources and exit")
    ap.add_argument("--out")
    ap.add_argument("--dump-scores",
                    help="CSV of the blend's per-face /10 score alongside each component's. "
                         "The /10 comes from the SAME percentile ladder as BT "
                         "(calibrate.ANCHORS), so any blend has an identical /10 "
                         "distribution — only which face gets which score changes.")
    args = ap.parse_args()

    export = load_export(args.export)
    matchups = export.all_matchups()
    faces = export.faces()
    available = discover_sources(ROOT, faces, matchups)

    if args.list:
        print(f"{'source':30} {'kind':9} {'faces':>7} {'valSplit':>9}  notes")
        for s in available:
            val = "all" if s.val_faces is None else f"{len(s.val_faces):,}"
            note = ("CIRCULAR: " + s.caveat) if s.circular else s.caveat
            print(f"{s.name:30} {s.kind:9} {len(s.scores):7,} {val:>9}  {note[:70]}")
        return 0

    if not args.source:
        raise SystemExit("give at least one --source (see --list)")
    if not args.panel:
        raise SystemExit("--panel is required to score against human votes")

    by_name = {s.name: s for s in available}
    missing = [n for n in args.source if n not in by_name]
    if missing:
        raise SystemExit(f"unknown source(s): {missing}. Run with --list.")
    sources = [by_name[n] for n in args.source]
    weights = args.weight or [1.0] * len(sources)
    if len(weights) != len(sources):
        raise SystemExit("--weight must be given as many times as --source")

    rejects: set[str] = set()
    for p in args.rejects:
        if Path(p).exists():
            rejects |= set(Path(p).read_text().split())

    dropped: set[str] = set()
    for p in (args.exclude_faces, args.exclude_genders):
        if p and Path(p).exists():
            with open(p, newline="") as fh:
                dropped.update(r["faceId"] for r in csv.DictReader(fh))
    by_index = {m.pair_index: m for m in matchups
                if m.face_a_id not in dropped and m.face_b_id not in dropped}

    votes = load_votes([tuple(p) for p in args.panel], rejects)
    gender_of = {fid: f.gender for fid, f in faces.items()}
    with open(args.ratings, newline="") as fh:
        pct = {r["faceId"]: float(r["percentile"]) for r in csv.DictReader(fh)}

    rep = evaluate_blend(sources, weights, gender_of, votes, by_index, pct,
                         method=args.method, n_boot=args.bootstrap)
    if not rep.keys:
        raise SystemExit("no leak-free panel pairs for this combination of sources — "
                         "the val splits do not overlap the panel draw")

    print(f"blend: {args.method}-average of "
          + ", ".join(f"{s.name}x{w:g}" for s, w in zip(sources, weights)))
    print(f"leak-free panel pairs: {len(rep.keys):,} "
          f"({rep.votes_scored:,.0f} human votes)\n")
    print(f"{'predictor':34} {'agrees with human votes':>24} {'votes':>9}")
    for r in sorted(rep.rows, key=lambda r: -r["accuracy"]):
        mark = "  <- CIRCULAR" if r["circular"] else ""
        print(f"{r['predictor'][:34]:34} {r['accuracy']:23.2%} {r['votes']:9,.0f}{mark}")
    print(f"{'ceiling (another rater)':34} {rep.ceiling:23.2%}")

    if rep.versus_best:
        v = rep.versus_best
        verdict = ("beats it" if v["lo"] > 0 else
                   "loses to it" if v["hi"] < 0 else
                   "indistinguishable from it")
        print(f"\nblend minus best honest component ({rep.best_component}): "
              f"{v['delta']:+.2%} "
              f"[{v['lo']:+.2%}, {v['hi']:+.2%}] over {v['pairs']:,} paired pairs")
        print(f"  => the blend {verdict}")

    bands = rep.bands.get("BLEND") or []
    if bands:
        print(f"\nblend by percentile gap in {Path(args.ratings).parent.name}:")
        for b in bands:
            print(f"  {b['band']:>7} {b['pairs']:5} pairs  {b['accuracy']:6.1%}")

    if args.dump_scores:
        blended = blend(sources, weights, gender_of, args.method)
        normed = {s.name: normalise(s.scores, gender_of, args.method) for s in sources}
        bt10 = {}
        with open(args.ratings, newline="") as fh:
            for r in csv.DictReader(fh):
                bt10[r["faceId"]] = (float(r["scoreOutOf10"]), float(r["percentile"]) * 100)

        dp = Path(args.dump_scores)
        dp.parent.mkdir(parents=True, exist_ok=True)
        cols = ["faceId", "gender", "blendPct", "blendScoreOutOf10",
                "btPct", "btScoreOutOf10", "deltaVsBt", "decileBin", "imagePath"]
        for s in sources:
            cols += [f"pct__{s.name}", f"score10__{s.name}"]
        rows_out = []
        for fid, bp in blended.items():
            face = faces.get(fid)
            bt_s, bt_p = bt10.get(fid, (float("nan"), float("nan")))
            b10 = score_out_of_10(bp / 100.0)
            row = {"faceId": fid, "gender": gender_of.get(fid, ""),
                   "blendPct": round(bp, 3), "blendScoreOutOf10": round(b10, 3),
                   "btPct": round(bt_p, 3), "btScoreOutOf10": round(bt_s, 3),
                   "deltaVsBt": round(b10 - bt_s, 3),
                   "decileBin": getattr(face, "decile_bin", ""),
                   "imagePath": getattr(face, "image_path", "")}
            for s in sources:
                p = normed[s.name].get(fid)
                row[f"pct__{s.name}"] = round(p, 3) if p is not None else ""
                row[f"score10__{s.name}"] = (round(score_out_of_10(p / 100.0), 3)
                                             if p is not None else "")
            rows_out.append(row)
        rows_out.sort(key=lambda r: (r["gender"], -r["blendScoreOutOf10"]))
        with open(dp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows_out)
        print(f"\nwrote {dp} ({len(rows_out):,} faces)")

        sc = np.array([r["blendScoreOutOf10"] for r in rows_out])
        moved = np.array([abs(r["deltaVsBt"]) for r in rows_out
                          if not np.isnan(r["deltaVsBt"])])
        print(f"  blend /10: min {sc.min():.2f}  p50 {np.median(sc):.2f}  "
              f"p90 {np.percentile(sc, 90):.2f}  p99 {np.percentile(sc, 99):.2f}  "
              f"max {sc.max():.2f}")
        print(f"  vs BT /10: {np.median(moved):.2f} median move, "
              f"{(moved > 0.5).mean():.1%} of faces move more than 0.5")
        print("  NOTE: the /10 distribution is fixed by the percentile anchors "
              "(50th=5.0, 90th=7.0, 99th=8.0).")
        print("  A different model reshuffles WHO gets a 7, it cannot make the top "
              "score 9.5 — only new anchors can.")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "sources": args.source,
            "weights": weights,
            "method": args.method,
            "pairsScored": len(rep.keys),
            "humanVotes": rep.votes_scored,
            "ceiling": rep.ceiling,
            "predictors": rep.rows,
            "bestHonestComponent": rep.best_component,
            "blendVsBest": rep.versus_best,
            "bands": rep.bands,
        }, indent=2))
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
