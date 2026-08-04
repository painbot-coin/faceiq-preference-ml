#!/usr/bin/env python
"""BT refit CLI: fit per-gender Bradley-Terry, run gates, calibrate to /10.

Usage:
    python scripts/run_bt.py --export data/exports/<runId> [--out artifacts/bt-refit-v1]
                             [--confidence high] [--skip-stability]

Outputs <out>/ratings.csv and <out>/metrics.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from faceiq_pref.bt import MIN_COMPARISONS, STABILITY_RHO_GATE, fit_bt, stability_check
from faceiq_pref.calibrate import ANCHORS, calibrate
from faceiq_pref.data import load_export
from faceiq_pref.validate import spearman_vs_labs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True, help="export directory (contains manifest.json)")
    ap.add_argument("--out", default="artifacts/bt-refit-v1")
    ap.add_argument(
        "--confidence",
        choices=["high", "medium", "low"],
        help="only keep VLM-labeled pairs at this confidence or better (human-audited pairs always kept)",
    )
    ap.add_argument("--skip-stability", action="store_true", help="skip 80%% subsample check (slow)")
    ap.add_argument(
        "--exclude-faces",
        help="CSV with a faceId column (e.g. artifacts/face-qc-v1/exclude-faces.csv). "
        "Drops every matchup touching those faces, so a cleaned refit can be run "
        "before the labs-side exclusions are applied and re-exported.",
    )
    ap.add_argument(
        "--exclude-genders",
        help="CSV with a faceId column of gender relabels (gender-fixes.csv). Their "
        "existing matchups were generated within the wrong gender, so they are "
        "dropped too.",
    )
    args = ap.parse_args()

    export = load_export(args.export)
    print(f"export OK: run {export.run_id}")

    matchups = export.all_matchups()
    print(f"loaded {len(matchups)} matchups")

    if args.confidence:
        order = {"low": 0, "medium": 1, "high": 2}
        floor = order[args.confidence]
        before = len(matchups)
        matchups = [
            m
            for m in matchups
            if m.human_labeled_at is not None
            or (m.confidence is not None and order.get(m.confidence, 0) >= floor)
        ]
        print(f"confidence filter >= {args.confidence}: {before} -> {len(matchups)}")

    dropped_faces: set[str] = set()
    for path in (args.exclude_faces, args.exclude_genders):
        if path:
            with open(path, newline="") as fh:
                dropped_faces.update(row["faceId"] for row in csv.DictReader(fh))
    if dropped_faces:
        before = len(matchups)
        matchups = [
            m
            for m in matchups
            if m.face_a_id not in dropped_faces and m.face_b_id not in dropped_faces
        ]
        print(
            f"QC exclusions: {len(dropped_faces)} faces -> dropped "
            f"{before - len(matchups)} matchups ({before} -> {len(matchups)})"
        )

    faces = export.faces()
    genders = sorted({m.gender for m in matchups})
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    metrics: dict = {
        "runId": export.run_id,
        "refitAt": datetime.now(timezone.utc).isoformat(),
        "matchupsUsed": len(matchups),
        "confidenceFilter": args.confidence,
        "qcExcludedFaces": len(dropped_faces),
        "tieRule": "half-win each side (0.5)",
        "calibrationAnchors": ANCHORS,
        "genders": {},
    }
    gates_ok = True

    for gender in genders:
        print(f"\n== {gender} ==")
        result = fit_bt(matchups, gender)
        if result.dropped_faces:
            print(f"dropped {len(result.dropped_faces)} under-connected face(s) "
                  f"(+{result.dropped_matchups} matchups): {result.dropped_faces[:5]}")
        print(f"faces={len(result.theta)} components={result.n_components} "
              f"below_min={len(result.faces_below_min)}")

        rho = None
        if not args.skip_stability:
            rho = stability_check(matchups, gender)
            print(f"stability rho = {rho:.4f} (gate > {STABILITY_RHO_GATE})")

        gate = {
            "connected": result.connected,
            "noFaceBelowMin": len(result.faces_below_min) == 0,
            "stabilityPassed": rho is None or rho > STABILITY_RHO_GATE,
        }
        gates_ok = gates_ok and all(gate.values())

        calibrated = calibrate(result.theta)
        labs_rho, labs_n = spearman_vs_labs(result.theta, faces)

        metrics["genders"][gender] = {
            "faces": len(result.theta),
            "components": result.n_components,
            "droppedUnderConnectedFaces": result.dropped_faces,
            "droppedMatchups": result.dropped_matchups,
            "facesBelowMinComparisons": result.faces_below_min[:20],
            "minComparisons": MIN_COMPARISONS,
            "stabilityRho": rho,
            "spearmanVsLabs": labs_rho,
            "labsScoredFaces": labs_n,
            "gates": gate,
        }

        for fid, c in calibrated.items():
            face = faces[fid]
            all_rows.append(
                {
                    "faceId": fid,
                    "gender": gender,
                    "theta": round(c["theta"], 6),
                    "percentile": round(c["percentile"], 6),
                    "scoreOutOf10": round(c["scoreOutOf10"], 3),
                    "comparisonCount": result.comparison_counts[fid],
                    "decileBin": face.decile_bin,
                    "labsOverallScore": face.labs_overall_score,
                    "imagePath": face.image_path,
                }
            )

    all_rows.sort(key=lambda r: (r["gender"], -r["theta"]))
    ratings_path = out_dir / "ratings.csv"
    with ratings_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    metrics["allGatesPassed"] = gates_ok
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"\nwrote {ratings_path} ({len(all_rows)} faces)")
    print(f"wrote {out_dir / 'metrics.json'}")
    print(f"ALL GATES {'PASSED' if gates_ok else 'FAILED — inspect metrics.json'}")
    return 0 if gates_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
