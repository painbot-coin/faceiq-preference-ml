#!/usr/bin/env python
"""Re-map an existing BT refit onto a different /10 calibration curve.

Does NOT refit Bradley-Terry: theta, percentile, and the ranking are copied verbatim;
only scoreOutOf10 is recomputed. Output is a sibling artifacts dir the dashboard
picks up automatically. Fully revertible — the source refit is never modified.

Usage:
    python scripts/recalibrate.py --refit artifacts/bt-refit-v1 \
        [--anchors cap95] [--out artifacts/bt-refit-v1-cap95]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from faceiq_pref.calibrate import ANCHORS, ANCHORS_CAP95, make_scorer

ANCHOR_SETS = {
    "v1": ANCHORS,
    "cap95": ANCHORS_CAP95,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refit", required=True, help="source refit dir (with ratings.csv)")
    ap.add_argument("--anchors", choices=sorted(ANCHOR_SETS), default="cap95")
    ap.add_argument("--out", help="output dir; defaults to <refit>-<anchors>")
    args = ap.parse_args()

    src = Path(args.refit)
    out = Path(args.out) if args.out else src.parent / f"{src.name}-{args.anchors}"
    anchors = ANCHOR_SETS[args.anchors]
    score = make_scorer(anchors)

    with (src / "ratings.csv").open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["scoreOutOf10"] = round(score(float(row["percentile"])), 3)

    out.mkdir(parents=True, exist_ok=True)
    with (out / "ratings.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metrics = json.loads((src / "metrics.json").read_text())
    metrics["calibrationAnchors"] = anchors
    metrics["calibrationVariant"] = args.anchors
    metrics["recalibratedFrom"] = src.name
    metrics["recalibratedAt"] = datetime.now(timezone.utc).isoformat()
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    top = max(rows, key=lambda r: float(r["scoreOutOf10"]))
    print(f"wrote {out}/ratings.csv ({len(rows)} faces) and metrics.json")
    print(f"anchors '{args.anchors}': max score now {top['scoreOutOf10']} "
          f"(was capped at {ANCHORS[-1][1]} in v1)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
