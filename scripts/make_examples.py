#!/usr/bin/env python
"""Pick the 3 worked examples for the panel instruction sheet.

Selects one easy pair, one close pair, and one genuine coin-flip from the
pilot-500 set (where we already have Dit + Alex votes and Gemini labels), and
writes the app-facing examples.json plus the list of pairIds the panel draw must
exclude — a rater must never be measured on a pair whose answer we showed them.

Selection rules (override any of them with --clear/--close/--tie):
  clear  very_clear bucket, Dit + Alex + Gemini unanimous, highest win prob
  close  close bucket, Dit + Alex agree with each other, win prob nearest 0.5
  tie    close bucket where the two raters split or one called it a tie

All candidates are additionally gated on Stage 0 face QC: both faces must be
usable, issue-free and confidently the gender we labelled them. These three pairs
are the first thing every rater sees, so a screenshot or a mislabelled face here
would set the tone for the whole session.

Explanations are deliberately non-analytic: naming features ("better cheekbones")
would teach raters to score a checklist, which is the behaviour this whole
ground-truth effort is replacing.

Usage:
    python scripts/make_examples.py [--pilot labels/pilot-500]
        [--out labels/panel-pilot] [--preview] [--clear PAIRID] ...
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPLANATIONS = {
    "clear": (
        "An easy call — most people pick the same side immediately and agree. "
        "When the gap is this obvious, just click and move on."
    ),
    "close": (
        "Much closer. Both of our reviewers leaned the same way, but neither "
        "found it obvious. A slight lean still counts as an answer: pick the "
        "side you lean toward rather than calling it a tie."
    ),
    "tie": (
        "A genuine coin flip — our two reviewers split on this one. When you "
        "truly have no lean either way, use \u201ctoo close to call\u201d. Expect "
        "that to be rare."
    ),
}
LABELS = {
    "clear": "Example 1 \u2014 an easy call",
    "close": "Example 2 \u2014 a close call",
    "tie": "Example 3 \u2014 too close to call",
}


def qc_clean_faces(qc_dir: Path) -> set[str] | None:
    """Faces that came out of Stage 0 QC with nothing at all against them."""
    results = qc_dir / "results.jsonl"
    if not results.exists():
        return None
    clean: set[str] = set()
    for line in results.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "error" in r:  # two faces the VLM never returned a verdict for
            continue
        if (
            r["usable"]
            and r["is_real_photo"]
            and not r["issues"]
            and not r["flagReasons"]
            and r["gender_confidence"] == "high"
            and r["apparent_gender"] == r["exportGender"]
        ):
            clean.add(r["faceId"])
    for name in ("exclude-faces.csv", "gender-fixes.csv"):
        path = qc_dir / name
        if path.exists():
            with path.open() as fh:
                clean -= {row["faceId"] for row in csv.DictReader(fh)}
    return clean


def load_pilot(pilot_dir: Path) -> tuple[dict, dict, dict]:
    pairs = {p["pairId"]: p for p in json.loads((pilot_dir / "pairs.json").read_text())["pairs"]}
    meta = {
        m["pairId"]: m for m in json.loads((pilot_dir / "sample-meta.json").read_text())["pairs"]
    }
    votes: dict[str, dict[str, dict]] = {}
    for who in ("dit", "alex"):
        path = pilot_dir / f"votes-{who}.jsonl"
        votes[who] = {
            json.loads(line)["pairId"]: json.loads(line)
            for line in path.read_text().splitlines()
            if line.strip()
        }
    return pairs, meta, votes


def vlm_side(m: dict) -> str:
    """Gemini's winner expressed as the side it was shown on."""
    if m["vlmOutcome"] == "A":
        return "left" if m["faceAOnLeft"] else "right"
    if m["vlmOutcome"] == "B":
        return "right" if m["faceAOnLeft"] else "left"
    return "tie"


def candidates(
    pairs: dict, meta: dict, votes: dict, clean: set[str] | None = None
) -> dict[str, list[str]]:
    """kind -> ranked candidate pairIds, best first."""
    common = [pid for pid in pairs if pid in votes["dit"] and pid in votes["alex"]]
    if clean is not None:
        common = [
            pid
            for pid in common
            if pairs[pid]["left"]["faceId"] in clean and pairs[pid]["right"]["faceId"] in clean
        ]
    dit, alex = votes["dit"], votes["alex"]

    unanimous = [
        pid
        for pid in common
        if meta[pid]["bucket"] == "very_clear"
        and dit[pid]["choice"] == alex[pid]["choice"] != "tie"
        and vlm_side(meta[pid]) == dit[pid]["choice"]
    ]
    close_agree = [
        pid
        for pid in common
        if meta[pid]["bucket"] == "close" and dit[pid]["choice"] == alex[pid]["choice"] != "tie"
    ]
    split = [
        pid
        for pid in common
        if meta[pid]["bucket"] == "close" and dit[pid]["choice"] != alex[pid]["choice"]
    ]
    if not (unanimous and close_agree and split):
        raise SystemExit(
            f"no candidate for one of the kinds "
            f"(unanimous={len(unanimous)}, close={len(close_agree)}, split={len(split)})"
        )
    return {
        "clear": sorted(unanimous, key=lambda p: -meta[p]["winProb"]),
        "close": sorted(close_agree, key=lambda p: abs(meta[p]["winProb"] - 0.5)),
        "tie": sorted(split, key=lambda p: abs(meta[p]["winProb"] - 0.5)),
    }


def answer_text(kind: str, choice: str) -> str:
    if kind == "tie":
        return "Either \u2014 too close to call"
    return {"left": "Left", "right": "Right"}[choice] + (
        " (a slight lean)" if kind == "close" else ""
    )


def contact_sheet(rows: list[tuple[str, dict]], run_id: str, out: Path) -> Path | None:
    """Contact sheet of (caption, pair) rows so a human can eyeball them."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    cell, pad = 300, 12
    img = Image.new("RGB", (cell * 2 + pad * 3, (cell + 34) * len(rows) + pad), "white")
    draw = ImageDraw.Draw(img)
    images_dir = ROOT / "data" / "exports" / run_id / "images"
    for row, (caption, pair) in enumerate(rows):
        y = pad + row * (cell + 34)
        draw.text((pad, y), caption, fill="black")
        for col, side in enumerate(("left", "right")):
            path = images_dir / f"{pair[side]['faceId']}.webp"
            if not path.exists():
                continue
            face = Image.open(path).convert("RGB").resize((cell, cell))
            img.paste(face, (pad + col * (cell + pad), y + 22))
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", default="labels/pilot-500")
    ap.add_argument("--out", default="labels/panel-pilot")
    ap.add_argument("--qc-dir", default="artifacts/face-qc-v1")
    ap.add_argument(
        "--no-qc-filter",
        action="store_true",
        help="don't require both faces to be QC-clean (only for reproducing the pre-QC pick)",
    )
    ap.add_argument("--preview", action="store_true", help="write a contact sheet for review")
    ap.add_argument(
        "--list",
        type=int,
        metavar="N",
        help="write a contact sheet of the top N candidates per kind and exit "
        "(review, then pass the winners via --clear/--close/--tie)",
    )
    for kind in ("clear", "close", "tie"):
        ap.add_argument(f"--{kind}", help=f"override the {kind} example with this pairId")
    args = ap.parse_args()

    pilot_dir = Path(args.pilot)
    pairs, meta, votes = load_pilot(pilot_dir)
    doc = json.loads((pilot_dir / "pairs.json").read_text())
    clean = None if args.no_qc_filter else qc_clean_faces(Path(args.qc_dir))
    if clean is not None:
        print(f"QC-clean faces: {len(clean)}")
    ranked = candidates(pairs, meta, votes, clean)

    if args.list:
        rows = []
        for kind in ("clear", "close", "tie"):
            for pid in ranked[kind][: args.list]:
                rows.append((f"{kind}: {pid}  p={meta[pid]['winProb']:.3f}", pairs[pid]))
                print(
                    f"{kind:6s} {pid}  p={meta[pid]['winProb']:.3f}  "
                    f"dit={votes['dit'][pid]['choice']:5s} alex={votes['alex'][pid]['choice']}"
                )
        path = contact_sheet(
            rows, doc["runId"], ROOT / "artifacts" / "panel-pilot" / "example-candidates.png"
        )
        print(f"\nwrote {path}" if path else "\npreview skipped (Pillow not installed)")
        return 0

    chosen = {kind: ranked[kind][0] for kind in ("clear", "close", "tie")}
    for kind in ("clear", "close", "tie"):
        override = getattr(args, kind)
        if override:
            if override not in pairs:
                raise SystemExit(f"--{kind} {override} is not a pilot-500 pairId")
            chosen[kind] = override

    examples = []
    for kind in ("clear", "close", "tie"):
        pid = chosen[kind]
        pair, m = pairs[pid], meta[pid]
        choice = votes["dit"][pid]["choice"]
        examples.append(
            {
                "label": LABELS[kind],
                "left": {"photoUrl": pair["left"]["photoUrl"]},
                "right": {"photoUrl": pair["right"]["photoUrl"]},
                "answer": answer_text(kind, choice),
                "explanation": EXPLANATIONS[kind],
            }
        )
        print(
            f"{kind:6s} {pid}  bucket={m['bucket']:10s} p={m['winProb']:.3f}  "
            f"dit={votes['dit'][pid]['choice']:5s} alex={votes['alex'][pid]['choice']:5s} "
            f"gemini={vlm_side(m)}"
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "examples.json").write_text(
        json.dumps(
            {
                "version": 1,
                "note": (
                    "Worked examples for the rating app instruction sheet. Generated by "
                    "scripts/make_examples.py; explanations are intentionally non-analytic "
                    "so they don't teach raters a feature checklist."
                ),
                "createdAt": datetime.now(timezone.utc).isoformat(),
                "examples": examples,
            },
            indent=1,
        )
        + "\n"
    )
    (out_dir / "example-pairs.json").write_text(
        json.dumps(
            {
                "version": 1,
                "note": (
                    "pairIds shown as worked examples in the instruction sheet. The panel "
                    "draw MUST exclude these — we showed raters the answer."
                ),
                "runId": doc["runId"],
                "pairIds": [chosen[k] for k in ("clear", "close", "tie")],
            },
            indent=1,
        )
        + "\n"
    )
    print(f"\nwrote {out_dir / 'examples.json'}")
    print(f"wrote {out_dir / 'example-pairs.json'} (exclude these from the panel draw)")

    if args.preview:
        rows = [(f"{kind}: {chosen[kind]}", pairs[chosen[kind]]) for kind in ("clear", "close", "tie")]
        path = contact_sheet(
            rows, doc["runId"], ROOT / "artifacts" / "panel-pilot" / "examples-preview.png"
        )
        print(f"wrote {path}" if path else "preview skipped (Pillow not installed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
