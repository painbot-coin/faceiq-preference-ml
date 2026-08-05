"""Download off-cohort front photos from a labs CSV export into a validation set.

Two things this does that a `wget -i` loop would not, both of which protect the validation:

**The blob path's `/faces/<id>/` segment is an account, not a face.** ~1.5 rows share one,
and it is tempting to dedupe on it — but 20 of those accounts carry both male and female
photos, i.e. people scoring their friends rather than re-uploading themselves. Deduping on
the account would throw away genuinely distinct faces, and the female set is too thin to
afford that. So this script downloads every row; removing repeat *identities* is
`scripts/audit_validation_identities.py`, which compares faces instead of guessing from ids.

**A refusal to download anything already in the cohort.** The whole point of this set is
faces the model has never seen, so the script hard-fails on any overlap with
`data/exports/<run>/faces.jsonl` rather than warning. Note the limit of that check: it
matches on face id, and the export carries no user id, so a *cohort user re-uploading
today* cannot be detected here. That has to be excluded upstream, on the labs side.

If the CSV carries a gender column the set is split into `<set>-female` / `<set>-male`,
because the two BT graphs never share an edge: a cross-gender pair has no θ difference to
be right or wrong about, so judging one is wasted effort and scoring one is meaningless.

Usage:
    python scripts/fetch_validation_photos.py --csv ~/Downloads/data-lab-front-photos-1000.csv \
        --set set-1 --limit 250
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
PERSON_RE = re.compile(r"/faces/([^/]+)/")


def cohort_ids() -> set[str]:
    exports = [p for p in sorted((ROOT / "data" / "exports").glob("*"))
               if (p / "faces.jsonl").exists()]
    ids: set[str] = set()
    for exp in exports:
        for line in (exp / "faces.jsonl").read_text().splitlines():
            row = json.loads(line)
            ids |= {row["sourceFaceId"], row["faceId"]}
    return ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--set", dest="set_name", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = all unique people")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gender-col", default="gender",
                    help="CSV column holding male/female; ignored if absent")
    a = ap.parse_args()

    rows = list(csv.DictReader(Path(a.csv).expanduser().open()))
    seen = cohort_ids()
    overlap = {r["face_id"] for r in rows} & seen
    if overlap:
        raise SystemExit(
            f"{len(overlap)} of {len(rows)} face ids are already in the cohort — this set is "
            f"contaminated and would inflate the result. Re-export with the exclusion list at "
            f"artifacts/cohort-exclusion/. First few: {sorted(overlap)[:5]}"
        )

    by_face: dict[str, dict] = {}
    for r in rows:
        by_face.setdefault(r["face_id"], r)
    people = [(PERSON_RE.search(r["front_photo_url"]).group(1)  # type: ignore[union-attr]
               if PERSON_RE.search(r["front_photo_url"]) else fid, r)
              for fid, r in by_face.items()]
    people.sort(key=lambda kv: kv[1]["face_created_at"])
    if a.limit:
        people = people[: a.limit]

    has_gender = a.gender_col in (rows[0] if rows else {})
    accounts = len({p for p, _ in people})
    print(f"{len(rows)} rows -> {len(people)} photos across {accounts} accounts "
          f"(identity dedupe happens later, in audit_validation_identities.py)")

    def set_for(row: dict) -> str:
        if not has_gender:
            return a.set_name
        g = (row.get(a.gender_col) or "").strip().lower()
        return f"{a.set_name}-{g}" if g in {"male", "female"} else f"{a.set_name}-unknown"

    targets = {}
    for _, row in people:
        d = ROOT / "data" / "validation" / set_for(row) / "photos"
        d.mkdir(parents=True, exist_ok=True)
        targets[row["face_id"]] = d
    print("downloading " + ", ".join(
        f"{sum(1 for _, r in people if set_for(r) == s)} -> {s}"
        for s in sorted({set_for(r) for _, r in people})))

    def get(item: tuple[str, dict]) -> tuple[str, str | None]:
        _, row = item
        dest = targets[row["face_id"]] / f"{row['face_id']}.webp"
        if dest.exists() and dest.stat().st_size > 0:
            return row["face_id"], None
        try:
            resp = requests.get(row["front_photo_url"], timeout=30)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
        except Exception as exc:  # noqa: BLE001 - report and continue, one bad url is not fatal
            return row["face_id"], str(exc)[:80]
        return row["face_id"], None

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        results = list(pool.map(get, people))
    failed = [(i, e) for i, e in results if e]

    per_set: dict[str, list] = {}
    for person, row in people:
        if (targets[row["face_id"]] / f"{row['face_id']}.webp").exists():
            per_set.setdefault(set_for(row), []).append((person, row))
    for s, members in sorted(per_set.items()):
        manifest = ROOT / "data" / "validation" / s / "source.csv"
        with manifest.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["photo", "faceId", "personId", "createdAt", "url"])
            for person, row in members:
                w.writerow([f"{row['face_id']}.webp", row["face_id"], person,
                            row["face_created_at"], row["front_photo_url"]])

    print(f"\n{sum(len(v) for v in per_set.values())} photos on disk, {len(failed)} failed")
    for i, e in failed[:5]:
        print(f"  {i}: {e}")
    if not has_gender:
        print(f"\n!! no '{a.gender_col}' column, so this set mixes genders. Do not make-pairs "
              f"yet: roughly half the queue would be cross-gender, and those pairs cannot be "
              f"scored — the two BT graphs share no edge. Re-export with a gender column.")
        return
    print("\nnext: python scripts/audit_validation_identities.py --set "
          + " --set ".join(sorted(per_set)))


if __name__ == "__main__":
    main()
