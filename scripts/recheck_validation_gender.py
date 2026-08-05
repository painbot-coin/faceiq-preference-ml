"""Re-check the gender label on a validation set, and quarantine anything doubtful.

Why this exists: labs' `prodGender` is self-declared and wrong often enough that a judging
queue built from it served a male/female pair. Gender is not cosmetic here — the male and
female Bradley-Terry graphs share no comparison, so their theta scales are only related
through the anchor curve. A cross-gender pair has no ground-truth answer to be right or
wrong about, and a face placed against the wrong reference set is scored on the wrong scale.

The check is a second opinion, not a replacement. InsightFace `buffalo_l` agrees with the
cohort's (human-corrected) labels on **94.2%** of 400 sampled faces once faces are detected
and aligned — good enough to *contradict* a label, not good enough to *be* the label. So we
keep only faces where the declared label and the model agree, and quarantine the rest.

Throwing away disagreements is close to free: the male set holds 601 faces and a study needs
70. Spending that surplus on certainty is the right trade.

Note the earlier trap: running `genderage.onnx` on a whole photo resized to 96x96 scores
60.4%, barely above chance, because the model expects a detector-aligned crop. Always go
through FaceAnalysis.

Usage:
    python scripts/recheck_validation_gender.py --set set-1-female --set set-1-male --apply
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]


def build_app():
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"],
                       allowed_modules=["detection", "genderage"])
    app.prepare(ctx_id=-1, det_size=(320, 320))
    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="sets", action="append", required=True)
    ap.add_argument("--apply", action="store_true",
                    help="move disagreements out of photos/ (default is a dry run)")
    a = ap.parse_args()

    app = build_app()
    for name in a.sets:
        photos = ROOT / "data" / "validation" / name / "photos"
        if not photos.exists():
            print(f"{name}: no photos/, skipping")
            continue
        declared = name.rsplit("-", 1)[-1]
        if declared not in {"male", "female"}:
            raise SystemExit(f"cannot read a gender off the set name {name!r}")

        paths = sorted(photos.glob("*.webp"))
        agree, disagree, undetected = [], [], []
        for i, p in enumerate(paths):
            img = cv2.imread(str(p))
            faces = app.get(img) if img is not None else []
            if not faces:
                undetected.append(p)
            else:
                f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                pred = "male" if f.sex == "M" else "female"
                (agree if pred == declared else disagree).append(p)
            print(f"\r  {name}: {i + 1}/{len(paths)}", end="", flush=True)
        print()

        print(f"  declared {declared}: {len(agree)} confirmed, {len(disagree)} contradicted, "
              f"{len(undetected)} with no detectable face")
        report = {
            "set": name, "declaredGender": declared, "confirmed": [p.name for p in agree],
            "contradicted": [p.name for p in disagree],
            "undetected": [p.name for p in undetected],
        }
        (ROOT / "data" / "validation" / name / "gender-recheck.json").write_text(
            json.dumps(report, indent=2))

        if a.apply:
            for reason, group in (("gender-disputed", disagree), ("no-face", undetected)):
                dest = ROOT / "data" / "validation" / name / "rejected" / reason
                dest.mkdir(parents=True, exist_ok=True)
                for p in group:
                    shutil.move(str(p), dest / p.name)
            print(f"  -> {len(list(photos.glob('*.webp')))} remain in {name}")
        else:
            print("  (dry run — rerun with --apply to quarantine)")


if __name__ == "__main__":
    main()
