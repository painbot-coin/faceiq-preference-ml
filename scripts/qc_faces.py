#!/usr/bin/env python
"""Step 0 metadata QC over the GT cohort faces (panel-pilot execution plan §0).

One cheap Gemini call per face asking only *objective* questions — VLMs are
reliable here, unlike preference. Never asks anything about attractiveness.

Per face we collect:
  apparent_gender     male | female | ambiguous   (cross-checked vs export label)
  is_real_photo       false for AI-generated / render / anime / illustration
  apparent_age_band   under_18 | 18_24 | 25_34 | 35_44 | 45_plus
  usable              single face, fully visible, not a photo-of-a-screen
  issues[]            obstruction, screenshot, multiple_faces, heavy_filter, ...
  apparent_ethnicity  same vocabulary as faceiq-labs `Race` (src/types/face.ts) so
                      it cross-tabs directly against the self-declared value
  skin_tone           Fitzpatrick I-VI (stored nowhere else — genuinely new data)

Results are appended to <out>/results.jsonl and the run is resumable: rerunning
skips faces already present. Faces needing a human look land in <out>/flags.csv.

Fixes are applied in faceiq-labs (canonical DB), never here — this script only
produces the review list. See docs/research/panel-pilot-runbook.md.

Usage:
    export GEMINI_API_KEY=...            # or --env-file ../faceiq-labs/.env
    python scripts/qc_faces.py --export data/exports/<runId> [--limit 20]
        [--out artifacts/face-qc-v1] [--workers 8] [--model gemini-2.5-flash]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from faceiq_pref.data import load_export  # noqa: E402

PROMPT = """You are doing objective metadata quality control on a face photo for a
research dataset. Do NOT judge attractiveness, beauty, or desirability.

Answer only what you can see. Reply with JSON matching this schema exactly:

{
  "apparent_gender": "male" | "female" | "ambiguous",
  "gender_confidence": "high" | "medium" | "low",
  "is_real_photo": true | false,
  "is_real_photo_reason": "<short reason if false, else empty>",
  "apparent_age_band": "under_18" | "18_24" | "25_34" | "35_44" | "45_plus",
  "usable": true | false,
  "issues": ["obstruction" | "screenshot" | "multiple_faces" | "heavy_filter" |
             "extreme_angle" | "low_resolution" | "face_cropped" | "poor_lighting"],
  "apparent_ethnicity": "east_asian" | "south_asian" | "black" | "hispanic" |
                        "middle_eastern" | "native_american" | "pacific_islander" |
                        "white" | "mixed" | "other",
  "skin_tone_fitzpatrick": "I" | "II" | "III" | "IV" | "V" | "VI"
}

Important context: every image is an automatically produced tight crop of a
front-facing face. Tight framing is expected and is NOT a defect — do not report
"face_cropped" unless part of the eye, nose or mouth region is actually cut off.

Definitions:
- is_real_photo=false for AI-generated, 3D render, anime/cartoon, painting, or a
  heavily morphed composite. A normal phone selfie is real even if filtered.
- usable=false only if the facial features genuinely cannot be judged: a face
  substantially obstructed (hand, mask, sunglasses), more than one face in frame,
  a photo of a computer/phone screen (browser UI or window chrome visible), or
  resolution so low the features are indistinct.
- "screenshot" means the image is a capture of a screen showing another image,
  not merely that it is a crop.
- apparent_age_band is your best visual estimate; err younger when uncertain.

Return JSON only, no prose."""

FLAG_HEADER = [
    "faceId",
    "reasons",
    "exportGender",
    "apparentGender",
    "genderConfidence",
    "apparentAgeBand",
    "isRealPhoto",
    "usable",
    "issues",
    "photoUrl",
]


def load_env_file(path: Path) -> None:
    """Minimal .env reader: only sets keys that aren't already in the environment."""
    if not path.exists():
        raise SystemExit(f"--env-file {path} not found")
    for line in path.read_text().splitlines():
        m = re.match(r'\s*(?:export\s+)?([A-Z0-9_]+)\s*=\s*(.*)\s*$', line)
        if not m:
            continue
        key, value = m.group(1), m.group(2).strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def api_key() -> str:
    for name in ("GEMINI_API_KEY", "GOOGLE_GENERATIVE_AI_API_KEY", "GOOGLE_AI_API_KEY"):
        if os.environ.get(name):
            return os.environ[name]
    raise SystemExit(
        "no API key found — set GEMINI_API_KEY, or pass --env-file ../faceiq-labs/.env"
    )


def parse_response(text: str) -> dict:
    """Gemini sometimes wraps JSON in a code fence despite instructions."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    return json.loads(cleaned)


# Issues that mean the photo shouldn't be judged at all, vs. cosmetic notes we
# record but don't send to human review (raters are told to ignore photo quality).
HARD_ISSUES = {"screenshot", "multiple_faces"}


def flag_reasons(export_gender: str, r: dict) -> list[str]:
    """Only reasons worth a human's time — see the runbook's triage table."""
    reasons = []
    apparent = r.get("apparent_gender")
    if apparent == "ambiguous":
        reasons.append("gender_ambiguous")
    elif apparent and apparent != export_gender:
        reasons.append("gender_mismatch")
    if r.get("is_real_photo") is False:
        reasons.append("not_real_photo")
    if r.get("apparent_age_band") == "under_18":
        reasons.append("possible_minor")
    hard = HARD_ISSUES.intersection(r.get("issues") or [])
    if r.get("usable") is False and hard:
        reasons.extend(sorted(hard))
    return reasons


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True, help="export dir (contains manifest.json)")
    ap.add_argument("--out", default="artifacts/face-qc-v1")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, help="only process N faces (smoke test)")
    ap.add_argument("--env-file", help="read GEMINI_API_KEY from this .env file")
    args = ap.parse_args()

    if args.env_file:
        load_env_file(Path(args.env_file))

    from google import genai
    from google.genai import types

    export = load_export(args.export)
    faces = export.faces()
    print(f"export OK: run {export.run_id}, {len(faces)} faces")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"

    done: set[str] = set()
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["faceId"])
        print(f"resuming: {len(done)} faces already done")

    todo = [f for f in faces.values() if f.face_id not in done]
    if args.limit:
        todo = todo[: args.limit]
    if not todo:
        print("nothing to do")
        return 0
    print(f"processing {len(todo)} faces with {args.workers} workers, model {args.model}")

    client = genai.Client(api_key=api_key())
    export_root = Path(args.export)
    write_lock = threading.Lock()
    counts = {"ok": 0, "error": 0}

    def process(face) -> None:
        image_path = export_root / face.image_path
        try:
            image_bytes = image_path.read_bytes()
            response = client.models.generate_content(
                model=args.model,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/webp"),
                    PROMPT,
                ],
                config=types.GenerateContentConfig(
                    temperature=0, response_mime_type="application/json"
                ),
            )
            row = {"faceId": face.face_id, "exportGender": face.gender, **parse_response(response.text)}
            row["flagReasons"] = flag_reasons(face.gender, row)
        except Exception as e:  # keep going; reruns retry only failed faces
            row = {"faceId": face.face_id, "exportGender": face.gender, "error": str(e)[:300]}
        with write_lock:
            with results_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            counts["error" if "error" in row else "ok"] += 1
            n = counts["ok"] + counts["error"]
            if n % 50 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} (errors: {counts['error']})")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(process, todo))

    # ---- summary + flag list over everything gathered so far -----------------
    rows = [json.loads(line) for line in results_path.read_text().splitlines() if line.strip()]
    good = [r for r in rows if "error" not in r]
    flagged = [r for r in good if r.get("flagReasons")]

    reason_counts: dict[str, int] = {}
    for r in flagged:
        for reason in r["flagReasons"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    flags_path = out_dir / "flags.csv"
    with flags_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FLAG_HEADER)
        for r in sorted(flagged, key=lambda r: r["flagReasons"]):
            w.writerow(
                [
                    r["faceId"],
                    ";".join(r["flagReasons"]),
                    r.get("exportGender", ""),
                    r.get("apparent_gender", ""),
                    r.get("gender_confidence", ""),
                    r.get("apparent_age_band", ""),
                    r.get("is_real_photo", ""),
                    r.get("usable", ""),
                    ";".join(r.get("issues") or []),
                    faces[r["faceId"]].photo_url if r["faceId"] in faces else "",
                ]
            )

    dist: dict[str, dict[str, int]] = {
        "apparent_ethnicity": {},
        "skin_tone_fitzpatrick": {},
        "apparent_age_band": {},
    }
    for key in dist:
        for r in good:
            v = r.get(key)
            if v:
                dist[key][v] = dist[key].get(v, 0) + 1

    # Recorded for context only: raters are instructed to ignore photo quality.
    quality_counts: dict[str, int] = {}
    for r in good:
        for issue in r.get("issues") or []:
            quality_counts[issue] = quality_counts.get(issue, 0) + 1

    summary = {
        "runId": export.run_id,
        "model": args.model,
        "facesScored": len(good),
        "errors": len(rows) - len(good),
        "flagged": len(flagged),
        "flagReasons": reason_counts,
        "qualityIssues": quality_counts,
        "distributions": dist,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")

    print(f"\nscored {len(good)} faces ({len(rows) - len(good)} errors)")
    print(f"flagged for review: {len(flagged)} ({len(flagged) / max(len(good), 1):.1%}) {reason_counts}")
    print(f"quality notes (not flagged): {quality_counts}")
    print(f"ethnicity: {dist['apparent_ethnicity']}")
    print(f"skin tone: {dist['skin_tone_fitzpatrick']}")
    print(f"age bands: {dist['apparent_age_band']}")
    print(f"\nwrote {results_path}\nwrote {flags_path}\nwrote {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
