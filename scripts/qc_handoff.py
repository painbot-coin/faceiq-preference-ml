#!/usr/bin/env python
"""Turn Stage 0 review decisions into the faceiq-labs handoff + an impact report.

Reads <qc>/decisions.jsonl (written by app/qc_review.py) and emits the two lists
faceiq-labs consumes, then measures what removing those faces does to the BT
comparison graph *before* anyone touches prod:

    exclude-faces.csv    faceId, labsReason, flagReason
    gender-fixes.csv     faceId, correctGender

Both exclusions and gender relabels lose all their existing edges: matchups were
generated within the labelled gender, so a mislabelled face's ~35 comparisons are
cross-gender in reality and cannot be salvaged by flipping the label.

Usage:
    python scripts/qc_handoff.py --qc artifacts/face-qc-v1 \
        --export data/exports/cmr1mr0m7000196d57zi3vcgn
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from faceiq_pref.data import load_export  # noqa: E402

# Flag reason -> faceiq-labs GT_EXPORT_EXCLUDE_REASONS
# (synthetic | quality | duplicate | other).
LABS_REASON = {
    "not_real_photo": "synthetic",
    "screenshot": "quality",
    "multiple_faces": "quality",
    "possible_minor": "other",
    "gender_mismatch": "other",
    "gender_ambiguous": "other",
}

MIN_COMPARISONS = 15  # research log §5.1 acceptance gate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qc", default="artifacts/face-qc-v1")
    ap.add_argument("--export", required=True)
    args = ap.parse_args()

    qc_dir = Path(args.qc)
    decisions: dict[str, dict] = {}
    for line in (qc_dir / "decisions.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            decisions[row["faceId"]] = row  # last write wins

    excluded = {f: d for f, d in decisions.items() if d["action"] == "exclude"}
    relabelled = {f: d for f, d in decisions.items() if d["action"].startswith("relabel:")}
    kept = {f: d for f, d in decisions.items() if d["action"] == "keep"}
    # Both groups lose their edges; only excluded faces leave the cohort.
    edge_loss = set(excluded) | set(relabelled)

    with (qc_dir / "exclude-faces.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["faceId", "labsReason", "flagReason"])
        for fid, d in sorted(excluded.items()):
            w.writerow([fid, LABS_REASON.get(d["flagReason"], "other"), d["flagReason"]])

    with (qc_dir / "gender-fixes.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["faceId", "correctGender"])
        for fid, d in sorted(relabelled.items()):
            w.writerow([fid, d["action"].split(":", 1)[1]])

    export = load_export(args.export, verify_hashes=False)
    faces = export.faces()
    matchups = export.all_matchups()

    per_face: Counter[str] = Counter()
    surviving_per_face: Counter[str] = Counter()
    dropped_pairs = 0
    for m in matchups:
        per_face[m.face_a_id] += 1
        per_face[m.face_b_id] += 1
        if m.face_a_id in edge_loss or m.face_b_id in edge_loss:
            dropped_pairs += 1
        else:
            surviving_per_face[m.face_a_id] += 1
            surviving_per_face[m.face_b_id] += 1

    survivors = [fid for fid in faces if fid not in excluded]
    thin = [f for f in survivors if surviving_per_face[f] < MIN_COMPARISONS]
    orphaned = [f for f in survivors if surviving_per_face[f] == 0]

    print(f"reviewed faces:      {len(decisions)}")
    print(f"  keep:              {len(kept)}")
    print(f"  exclude:           {len(excluded)}")
    print(f"  relabel gender:    {len(relabelled)}")
    print(f"\nfaces losing all edges (exclude + relabel): {len(edge_loss)}"
          f" ({len(edge_loss) / len(faces):.1%} of cohort)")
    print(f"cohort after exclusions: {len(survivors)} faces")
    print(f"\npairs before: {len(matchups)}")
    print(f"pairs dropped: {dropped_pairs} ({dropped_pairs / len(matchups):.1%})")
    print(f"pairs remaining: {len(matchups) - dropped_pairs}")
    print(f"\nsurviving faces under the {MIN_COMPARISONS}-comparison gate: {len(thin)}")
    print(f"  of those with zero comparisons left: {len(orphaned)}")
    if thin:
        print("  -> these drop out of BT; expected for relabelled faces, check the rest")
    print(f"\nwrote {qc_dir / 'exclude-faces.csv'} ({len(excluded)} rows)")
    print(f"wrote {qc_dir / 'gender-fixes.csv'} ({len(relabelled)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
