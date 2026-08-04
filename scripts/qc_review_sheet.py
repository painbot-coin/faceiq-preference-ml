#!/usr/bin/env python
"""Render the QC flag list as contact sheets so a human can clear them fast.

Step 2 of the panel-pilot runbook: a person eyeballs only the flagged faces.
One PNG per flag reason, faces labelled with their faceId so decisions can be
written straight back into faceiq-labs.

Usage:
    python scripts/qc_review_sheet.py --qc artifacts/face-qc-v1
        --export data/exports/<runId> [--cols 6] [--cell 220]
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qc", required=True, help="QC output dir containing flags.csv")
    ap.add_argument("--export", required=True, help="export dir with images/")
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--cell", type=int, default=220)
    args = ap.parse_args()

    qc_dir = Path(args.qc)
    images_dir = Path(args.export) / "images"
    with (qc_dir / "flags.csv").open() as f:
        flagged = list(csv.DictReader(f))
    if not flagged:
        print("no flags to review")
        return 0

    # A face can carry several reasons; show it under each so nothing is missed.
    by_reason: dict[str, list[dict]] = defaultdict(list)
    for row in flagged:
        for reason in row["reasons"].split(";"):
            if reason:
                by_reason[reason].append(row)

    out_dir = qc_dir / "review"
    out_dir.mkdir(parents=True, exist_ok=True)
    cell, pad, label_h = args.cell, 8, 26

    for reason, rows in sorted(by_reason.items()):
        cols = min(args.cols, len(rows))
        n_rows = (len(rows) + cols - 1) // cols
        img = Image.new(
            "RGB",
            (cols * (cell + pad) + pad, n_rows * (cell + label_h + pad) + pad + 20),
            "white",
        )
        draw = ImageDraw.Draw(img)
        draw.text((pad, 6), f"{reason} — {len(rows)} faces", fill="black")
        for i, row in enumerate(rows):
            x = pad + (i % cols) * (cell + pad)
            y = 26 + (i // cols) * (cell + label_h + pad)
            path = images_dir / f"{row['faceId']}.webp"
            if path.exists():
                img.paste(Image.open(path).convert("RGB").resize((cell, cell)), (x, y))
            caption = f"{row['faceId'][-8:]} {row['exportGender'][:1]}->{row['apparentGender'][:1]}"
            draw.text((x, y + cell + 4), caption, fill="black")
            draw.text((x, y + cell + 14), row.get("apparentAgeBand", ""), fill="black")
        path = out_dir / f"{reason}.png"
        img.save(path)
        print(f"wrote {path} ({len(rows)} faces)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
