#!/usr/bin/env python
"""Attach self-declared ethnicity to the GT cohort from the Labs face manifest.

The GT export carries only faceId/gender/decileBin/score, but the cohort was
drawn from a Labs manifest whose rows include `prod_race` (the user's own
onboarding answer, possibly multiple) and `prod_gender`. The two join on
export `sourceFaceId` == manifest `analysis_id`, so we get ethnicity locally —
no prod DB, no re-export, no VLM.

Ethnicity values already use the faceiq-labs `Race` vocabulary
(src/types/face.ts), so they line up with the app and with rater strata.

Outputs <out>/face-attributes.json: faceId -> {ethnicities, prodGender, ...},
plus a coverage report naming any cohort faces the manifest didn't cover
(those fall back to the VLM's apparent_ethnicity from scripts/qc_faces.py).

Usage:
    python scripts/join_face_attributes.py --export data/exports/<runId> \
        --manifests ~/Downloads/ab-data [--out labels]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def parse_race(raw: object) -> list[str]:
    """prod_race is a JSON array string ('["white"]'), a bare string, or empty."""
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(v) for v in raw if v]
    text = str(raw).strip()
    if text.startswith("["):
        try:
            return [str(v) for v in json.loads(text) if v]
        except json.JSONDecodeError:
            return []
    return [text] if text else []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True, help="export dir (contains faces.jsonl)")
    ap.add_argument(
        "--manifests",
        required=True,
        help="directory holding manifest.part-NN-of-NN.jsonl shards",
    )
    ap.add_argument("--out", default="labels")
    args = ap.parse_args()

    faces = [
        json.loads(line)
        for line in (Path(args.export) / "faces.jsonl").read_text().splitlines()
        if line.strip()
    ]
    by_source = {f["sourceFaceId"]: f for f in faces}
    print(f"cohort: {len(faces)} faces")

    shards = sorted(Path(args.manifests).glob("manifest.part-*.jsonl"))
    if not shards:
        raise SystemExit(f"no manifest.part-*.jsonl shards found in {args.manifests}")
    print(f"manifest shards: {len(shards)}")

    attributes: dict[str, dict] = {}
    gender_mismatches: list[str] = []
    rows_seen = 0
    for shard in shards:
        n_before = len(attributes)
        for line in shard.read_text().splitlines():
            if not line.strip():
                continue
            rows_seen += 1
            row = json.loads(line)
            face = by_source.get(row.get("analysis_id"))
            if face is None:
                continue
            ethnicities = parse_race(row.get("prod_race"))
            prod_gender = row.get("prod_gender")
            if prod_gender and prod_gender != face["gender"]:
                gender_mismatches.append(face["faceId"])
            attributes[face["faceId"]] = {
                "sourceFaceId": row["analysis_id"],
                "ethnicities": ethnicities,
                "prodGender": prod_gender,
            }
        print(f"  {shard.name}: +{len(attributes) - n_before} cohort faces")

    missing = [f["faceId"] for f in faces if f["faceId"] not in attributes]
    no_ethnicity = [fid for fid, a in attributes.items() if not a["ethnicities"]]

    counts = Counter()
    for a in attributes.values():
        for e in a["ethnicities"]:
            counts[e] += 1
    primary = Counter(a["ethnicities"][0] for a in attributes.values() if a["ethnicities"])

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "face-attributes.json"
    out_path.write_text(
        json.dumps(
            {
                "version": 1,
                "runId": json.loads((Path(args.export) / "manifest.json").read_text())["runId"],
                "source": "Labs face manifest (prod_race = user self-declared, may be multiple)",
                "createdAt": datetime.now(timezone.utc).isoformat(),
                "coverage": {
                    "cohortFaces": len(faces),
                    "matched": len(attributes),
                    "unmatched": len(missing),
                    "matchedButNoEthnicity": len(no_ethnicity),
                },
                "unmatchedFaceIds": missing,
                "faces": attributes,
            },
            indent=1,
        )
        + "\n"
    )

    print(f"\nscanned {rows_seen} manifest rows")
    print(f"matched {len(attributes)}/{len(faces)} cohort faces ({len(attributes)/len(faces):.1%})")
    print(f"  no ethnicity value: {len(no_ethnicity)}")
    print(f"  unmatched (need VLM fallback): {len(missing)}")
    print(f"prod_gender vs export gender mismatches: {len(gender_mismatches)}")
    print(f"\nprimary ethnicity: {dict(primary.most_common())}")
    print(f"any-mention counts: {dict(counts.most_common())}")
    if primary:
        top, n = primary.most_common(1)[0]
        print(f"largest group: {top} at {n/len(attributes):.1%} (cohort audit gate: <=80%)")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
