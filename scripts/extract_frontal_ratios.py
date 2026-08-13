#!/usr/bin/env python
"""Extract a small frontal-ratio table from cohort export photos via MediaPipe.

Uses the same Face Landmarker already wired in `faceiq_pref.preprocess` (478-pt mesh).
Writes scale-invariant ratios keyed by faceId for late fusion. This is a self-serve
prototype schema — not Labs `frontLandmarks` production parity.

Usage:
    python scripts/extract_frontal_ratios.py
    python scripts/extract_frontal_ratios.py --export data/exports/<runId> --out artifacts/fusion-landmarks-v1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from faceiq_pref.preprocess import _get_landmarker  # noqa: E402

# MediaPipe Face Mesh indices (478-pt). Documented join schema for this prototype.
IDX = {
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "left_eye_inner": 133,
    "right_eye_inner": 362,
    "nose_tip": 1,
    "nasion": 168,
    "forehead": 10,
    "chin": 152,
    "left_cheek": 234,
    "right_cheek": 454,
    "mouth_left": 61,
    "mouth_right": 291,
    "left_jaw": 172,
    "right_jaw": 397,
}

RATIO_NAMES = [
    "eye_span_over_face_width",
    "nose_length_over_face_height",
    "jaw_over_face_width",
    "mouth_over_face_width",
    "midface_over_face_height",
    "canthal_tilt_deg",
]


def _xy(lm, key: str, iw: float, ih: float) -> tuple[float, float]:
    p = lm[IDX[key]]
    return p.x * iw, p.y * ih


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def ratios_from_landmarks(lm, iw: int, ih: int) -> dict[str, float] | None:
    """Compute scale-invariant frontal ratios from one MediaPipe face."""
    le = _xy(lm, "left_eye_outer", iw, ih)
    re = _xy(lm, "right_eye_outer", iw, ih)
    lei = _xy(lm, "left_eye_inner", iw, ih)
    rei = _xy(lm, "right_eye_inner", iw, ih)
    tip = _xy(lm, "nose_tip", iw, ih)
    nas = _xy(lm, "nasion", iw, ih)
    fore = _xy(lm, "forehead", iw, ih)
    chin = _xy(lm, "chin", iw, ih)
    lc = _xy(lm, "left_cheek", iw, ih)
    rc = _xy(lm, "right_cheek", iw, ih)
    ml = _xy(lm, "mouth_left", iw, ih)
    mr = _xy(lm, "mouth_right", iw, ih)
    lj = _xy(lm, "left_jaw", iw, ih)
    rj = _xy(lm, "right_jaw", iw, ih)

    face_w = _dist(lc, rc)
    face_h = _dist(fore, chin)
    if face_w < 1e-6 or face_h < 1e-6:
        return None

    eye_span = _dist(le, re)
    # pupil proxy: midpoints of inner/outer corners
    left_pupil = ((le[0] + lei[0]) / 2.0, (le[1] + lei[1]) / 2.0)
    right_pupil = ((re[0] + rei[0]) / 2.0, (re[1] + rei[1]) / 2.0)
    eye_mid = (
        (left_pupil[0] + right_pupil[0]) / 2.0,
        (left_pupil[1] + right_pupil[1]) / 2.0,
    )
    mouth_mid = ((ml[0] + mr[0]) / 2.0, (ml[1] + mr[1]) / 2.0)

    tilt = math.degrees(math.atan2(re[1] - le[1], re[0] - le[0]))

    out = {
        "eye_span_over_face_width": eye_span / face_w,
        "nose_length_over_face_height": _dist(nas, tip) / face_h,
        "jaw_over_face_width": _dist(lj, rj) / face_w,
        "mouth_over_face_width": _dist(ml, mr) / face_w,
        "midface_over_face_height": _dist(eye_mid, mouth_mid) / face_h,
        "canthal_tilt_deg": tilt,
    }
    if not all(math.isfinite(v) for v in out.values()):
        return None
    return out


def extract_one(path: Path) -> dict[str, float] | None:
    import mediapipe as mp

    rgb = np.array(Image.open(path).convert("RGB"))
    ih, iw = rgb.shape[:2]
    result = _get_landmarker().detect(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    )
    if not result.face_landmarks:
        return None
    return ratios_from_landmarks(result.face_landmarks[0], iw, ih)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--export",
        default=None,
        help="export dir with images/; default = newest under data/exports",
    )
    ap.add_argument("--out", default="artifacts/fusion-landmarks-v1")
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N faces")
    a = ap.parse_args()

    if a.export:
        export = ROOT / a.export
    else:
        export = next(
            p
            for p in sorted((ROOT / "data" / "exports").glob("*"))
            if (p / "images").is_dir() and (p / "faces.jsonl").exists()
        )
    images = export / "images"
    if not images.is_dir():
        raise SystemExit(f"missing images dir: {images}")

    out_dir = ROOT / a.out
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "frontal-ratios.csv"
    meta_path = out_dir / "extract-meta.json"

    paths = sorted(images.glob("*.webp"))
    if a.limit:
        paths = paths[: a.limit]

    rows: list[dict] = []
    failed: list[str] = []
    t0 = time.time()
    for i, path in enumerate(paths, 1):
        face_id = path.stem
        try:
            ratios = extract_one(path)
        except Exception as exc:  # noqa: BLE001 — keep batch going
            failed.append(face_id)
            print(f"  fail {face_id}: {exc}", file=sys.stderr)
            ratios = None
        if ratios is None:
            if face_id not in failed:
                failed.append(face_id)
            continue
        row = {"faceId": face_id, **ratios}
        rows.append(row)
        if i % 100 == 0 or i == len(paths):
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed else 0
            print(f"  {i}/{len(paths)}  ok={len(rows)}  fail={len(failed)}  {rate:.1f}/s")

    fieldnames = ["faceId", *RATIO_NAMES]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    meta = {
        "export": str(export.relative_to(ROOT)),
        "source": "mediapipe_face_landmarker_478",
        "modelCache": "artifacts/.cache/face_landmarker.task",
        "indices": IDX,
        "ratioNames": RATIO_NAMES,
        "nImages": len(paths),
        "nOk": len(rows),
        "nFail": len(failed),
        "failedFaceIds": failed[:50],
        "failedFaceIdsTruncated": len(failed) > 50,
        "elapsedSec": round(time.time() - t0, 1),
        "note": (
            "Self-serve frontal ratios for late-fusion prototype. Not Labs "
            "frontLandmarks; still want exporter parity for production."
        ),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {csv_path.relative_to(ROOT)} ({len(rows)} faces)")
    print(f"wrote {meta_path.relative_to(ROOT)}  fails={len(failed)}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
