#!/usr/bin/env python
"""Cross v25 bothHeldOut misses with attributes / QC / gender / BT gap.

Fusion data on disk today is thin (ethnicity + QC flags, no landmarks/ratios).
This script produces the measurable artifact for the Harsh data-lab brief:
which available signals concentrate on control-right / model-wrong pairs, and
what the lab should export first for a real late-fusion prototype.

Usage:
    python scripts/bothheldout_miss_feature_cross.py --run train-v25-panel-ft-v23b
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from accuracy_matrix import load_votes  # noqa: E402
from faceiq_pref.data import load_export, split_by_face_id  # noqa: E402


def _load_faces(export: Path) -> dict[str, dict]:
    out = {}
    for line in (export / "faces.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[row["faceId"]] = row
    return out


def _load_attrs(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("faces", payload)


def _load_qc(results_path: Path, flags_path: Path, exclude_path: Path) -> dict[str, dict]:
    by_id: dict[str, dict] = {}
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            fid = row.get("faceId") or row.get("id")
            if not fid:
                continue
            by_id[fid] = {
                "apparentEthnicity": row.get("apparent_ethnicity")
                or row.get("apparentEthnicity"),
                "skinTone": row.get("skin_tone_fitzpatrick")
                or row.get("skinTone"),
                "ageBand": row.get("apparent_age_band") or row.get("apparentAgeBand"),
                "issues": row.get("issues") or [],
                "usable": row.get("usable"),
                "isRealPhoto": row.get("is_real_photo", row.get("isRealPhoto")),
                "flagReasons": row.get("reasons") or row.get("flagReasons") or [],
            }
            if isinstance(by_id[fid]["issues"], str):
                by_id[fid]["issues"] = [
                    x for x in by_id[fid]["issues"].split(";") if x
                ]
            if isinstance(by_id[fid]["flagReasons"], str):
                by_id[fid]["flagReasons"] = [
                    x for x in by_id[fid]["flagReasons"].split(";") if x
                ]
    if flags_path.exists():
        with flags_path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                fid = row["faceId"]
                cur = by_id.setdefault(fid, {})
                reasons = [x for x in (row.get("reasons") or "").split(";") if x]
                issues = [x for x in (row.get("issues") or "").split(";") if x]
                cur["flagReasons"] = reasons or cur.get("flagReasons") or []
                cur["issues"] = issues or cur.get("issues") or []
                cur.setdefault("ageBand", row.get("apparentAgeBand"))
                cur.setdefault("usable", row.get("usable") == "True" if row.get("usable") else None)
    excluded = set()
    if exclude_path.exists():
        with exclude_path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                excluded.add(row["faceId"])
    for fid in by_id:
        by_id[fid]["qcExcluded"] = fid in excluded
    return by_id


def _primary_eth(attrs: dict | None) -> str | None:
    if not attrs:
        return None
    eth = attrs.get("ethnicities") or []
    return eth[0] if eth else None


def _face_bundle(fid: str, faces, attrs, qc) -> dict:
    f = faces.get(fid, {})
    a = attrs.get(fid, {})
    q = qc.get(fid, {})
    issues = list(q.get("issues") or [])
    flags = list(q.get("flagReasons") or [])
    return {
        "gender": f.get("gender"),
        "labsOverallScore": f.get("labsOverallScore"),
        "decileBin": f.get("decileBin"),
        "primaryEthnicity": _primary_eth(a),
        "prodGender": a.get("prodGender"),
        "qcApparentEthnicity": q.get("apparentEthnicity"),
        "qcSkinTone": q.get("skinTone"),
        "qcAgeBand": q.get("ageBand"),
        "qcIssues": issues,
        "qcFlagReasons": flags,
        "qcExcluded": bool(q.get("qcExcluded")),
        "hasAnyQcIssue": bool(issues) or bool(flags),
        "hasPhotoIssue": any(
            x in set(issues) | set(flags)
            for x in (
                "screenshot",
                "low_resolution",
                "heavy_filter",
                "poor_lighting",
                "obstruction",
                "extreme_angle",
                "face_cropped",
                "multiple_faces",
                "not_real_photo",
            )
        ),
    }


def _pair_row(err: dict, faces, attrs, qc) -> dict:
    fa, fb = err["faceAId"], err["faceBId"]
    a = _face_bundle(fa, faces, attrs, qc)
    b = _face_bundle(fb, faces, attrs, qc)
    eth_a = a["primaryEthnicity"] or a["qcApparentEthnicity"]
    eth_b = b["primaryEthnicity"] or b["qcApparentEthnicity"]
    return {
        **err,
        "genderA": a["gender"],
        "genderB": b["gender"],
        "sameGender": a["gender"] == b["gender"] and a["gender"] is not None,
        "ethnicityA": eth_a,
        "ethnicityB": eth_b,
        "ethnicityMatch": eth_a == eth_b and eth_a is not None,
        "skinToneA": a["qcSkinTone"],
        "skinToneB": b["qcSkinTone"],
        "ageBandA": a["qcAgeBand"],
        "ageBandB": b["qcAgeBand"],
        "qcExcludedEither": a["qcExcluded"] or b["qcExcluded"],
        "photoIssueEither": a["hasPhotoIssue"] or b["hasPhotoIssue"],
        "qcIssueEither": a["hasAnyQcIssue"] or b["hasAnyQcIssue"],
        "issuesA": ";".join(a["qcIssues"]),
        "issuesB": ";".join(b["qcIssues"]),
        "flagsA": ";".join(a["qcFlagReasons"]),
        "flagsB": ";".join(b["qcFlagReasons"]),
        "labsA": a["labsOverallScore"],
        "labsB": b["labsOverallScore"],
        "labsDelta": (
            None
            if a["labsOverallScore"] is None or b["labsOverallScore"] is None
            else abs(float(a["labsOverallScore"]) - float(b["labsOverallScore"]))
        ),
        "modelScoreDelta": abs(float(err["modelScoreA"]) - float(err["modelScoreB"])),
    }


def _rate(rows: list[dict], pred) -> dict:
    n = len(rows)
    k = sum(1 for r in rows if pred(r))
    return {"n": n, "k": k, "rate": (k / n) if n else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="train-v25-panel-ft-v23b")
    ap.add_argument("--audit", default=None)
    ap.add_argument("--attributes", default="labels/face-attributes.json")
    ap.add_argument("--qc-dir", default="artifacts/face-qc-v1")
    ap.add_argument("--control-ranking", default="artifacts/bt-refit-v2-qc")
    ap.add_argument("--panel", default="labels/panel-run-4-random")
    ap.add_argument("--rejects", default="artifacts/panel-run-v4/reject-pids.txt")
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    art = ROOT / "artifacts" / a.run
    audit_path = Path(a.audit) if a.audit else art / "bothHeldOut-error-audit.json"
    if not audit_path.exists():
        raise SystemExit(f"missing {audit_path}")

    export = next(
        p
        for p in sorted((ROOT / "data" / "exports").glob("*"))
        if (p / "faces.jsonl").exists()
    )
    faces = _load_faces(export)
    attrs = _load_attrs(ROOT / a.attributes)
    qc = _load_qc(
        ROOT / a.qc_dir / "results.jsonl",
        ROOT / a.qc_dir / "flags.csv",
        ROOT / a.qc_dir / "exclude-faces.csv",
    )

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    error_rows = [_pair_row(e, faces, attrs, qc) for e in audit["errors"]]
    ctrl_right = [r for r in error_rows if r["bucket"] == "modelWrong_controlRight"]
    both_wrong = [r for r in error_rows if r["bucket"] == "bothWrong"]

    # Rebuild full bothHeldOut set for baseline rates (same as error audit).
    scores = pd.read_csv(art / "model_scores.csv")
    model = dict(zip(scores["faceId"], scores["modelScore"]))
    ctrl = pd.read_csv(ROOT / a.control_ranking / "ratings.csv")
    control = dict(zip(ctrl["faceId"], ctrl["theta"]))
    _, val_rows = split_by_face_id(
        load_export(export).all_matchups(), a.val_fraction, a.split_seed
    )
    held_out = {f for m in val_rows for f in (m.face_a_id, m.face_b_id)}
    rejects = {
        ln.strip()
        for ln in (ROOT / a.rejects).read_text(encoding="utf-8").splitlines()
        if ln.strip()
    }
    votes, meta = load_votes(
        ROOT / a.panel / "results",
        ROOT / a.panel / "sample-meta.json",
        rejects,
        True,
    )

    all_both: list[dict] = []
    for pid, (va, vb) in votes.items():
        m = meta.get(pid)
        if m is None or va == vb:
            continue
        fa, fb = m["faceAId"], m["faceBId"]
        if fa not in held_out or fb not in held_out:
            continue
        if fa not in model or fb not in model or fa not in control or fb not in control:
            continue
        sa, sb = model[fa], model[fb]
        ca, cb = control[fa], control[fb]
        if sa == sb:
            continue
        maj_a = va > vb
        model_ok = (sa > sb) == maj_a
        ctrl_ok = (ca > cb) == maj_a
        if model_ok and ctrl_ok:
            bucket = "bothRight"
        elif model_ok and not ctrl_ok:
            bucket = "modelRight_controlWrong"
        elif (not model_ok) and ctrl_ok:
            bucket = "modelWrong_controlRight"
        else:
            bucket = "bothWrong"
        stub = {
            "pairId": pid,
            "faceAId": fa,
            "faceBId": fb,
            "votesA": va,
            "votesB": vb,
            "panelMajority": "A" if maj_a else "B",
            "modelPick": "A" if sa > sb else "B",
            "controlPick": "A" if ca > cb else ("B" if ca != cb else "tie"),
            "modelScoreA": float(sa),
            "modelScoreB": float(sb),
            "controlThetaA": float(ca),
            "controlThetaB": float(cb),
            "btGap": float(abs(ca - cb)),
            "bucket": bucket,
        }
        all_both.append(_pair_row(stub, faces, attrs, qc))

    def gap_lt1(r):
        return r["btGap"] < 1.0

    def gap_ge3(r):
        return r["btGap"] >= 3.0

    cohorts = {
        "allBothHeldOut": all_both,
        "modelCorrect": [r for r in all_both if r["bucket"] in ("bothRight", "modelRight_controlWrong")],
        "modelWrong_controlRight": ctrl_right,
        "bothWrong": both_wrong,
        "allModelWrong": error_rows,
    }

    signals = {
        "photoIssueEither": lambda r: r["photoIssueEither"],
        "qcIssueEither": lambda r: r["qcIssueEither"],
        "qcExcludedEither": lambda r: r["qcExcludedEither"],
        "ethnicityMismatch": lambda r: r["ethnicityMatch"] is False,
        "btGapLt1": gap_lt1,
        "btGapGe3": gap_ge3,
        "sameGender": lambda r: r["sameGender"],
    }

    rates = {
        name: {sig: _rate(rows, pred) for sig, pred in signals.items()}
        for name, rows in cohorts.items()
    }

    # Issue concentration on the 8 control-right misses
    issue_counter: Counter[str] = Counter()
    for r in ctrl_right:
        for side in ("issuesA", "issuesB", "flagsA", "flagsB"):
            for tok in (r.get(side) or "").split(";"):
                if tok:
                    issue_counter[tok] += 1

    eth_pairs = Counter(
        tuple(sorted([r["ethnicityA"] or "?", r["ethnicityB"] or "?"]))
        for r in ctrl_right
    )
    gender_pairs = Counter(
        (r["genderA"] or "?", r["genderB"] or "?") for r in ctrl_right
    )

    inventory = {
        "faceAttributes": {
            "path": str(ROOT / a.attributes),
            "exists": (ROOT / a.attributes).exists(),
            "fields": ["ethnicities", "prodGender", "sourceFaceId"],
            "nFaces": len(attrs),
            "note": "Demographic only — not geometry/ratios.",
        },
        "faceQcV1": {
            "path": str(ROOT / a.qc_dir),
            "exists": (ROOT / a.qc_dir).exists(),
            "fields": [
                "apparent_ethnicity",
                "skin_tone_fitzpatrick",
                "apparent_age_band",
                "issues",
                "flag reasons",
                "exclude-faces",
            ],
            "nFacesWithQc": len(qc),
            "note": "Photo QC + VLM demographics; no landmarks/ratios.",
        },
        "landmarksOrRatiosOnDisk": False,
        "blocker": (
            "No frontLandmarks / sideLandmarks / mediapipeLandmarks / ratio table in "
            "export or labels/. Labs DB already has them; exporter select omits them."
        ),
    }

    ask_harsh = [
        {
            "priority": 1,
            "signal": "frontLandmarks (+ schema: point count, order, coordinate space)",
            "why": (
                "Research log §5.6/§5.7: geometry EBM is the uncorrelated Plan B / "
                "fusion source. Already stored in Labs; one exporter select change."
            ),
            "joinKey": "faceId via images/<faceId>.webp filename",
            "minViable": "3,000 cohort faces with frozen landmark schema",
        },
        {
            "priority": 2,
            "signal": "derived frontal ratios (canthal tilt, facial thirds, jaw/width, eye spacing)",
            "why": (
                "Raw points are hard to fuse; ratio deltas [r_a - r_b] are the late-fusion "
                "features for logistic/MLP on top of (s_a - s_b)."
            ),
            "joinKey": "faceId",
            "minViable": "stable ratio dictionary documented once",
        },
        {
            "priority": 3,
            "signal": "pose / framing diagnostics (yaw, pitch, roll, face-fill, perspective flags)",
            "why": (
                "Existing QC issue flags are *not* enriched on control-right misses "
                "(62.5% vs 58.5% model-correct); perspective distortion is still the "
                "untested QC hole in production-scoring-pipeline.md."
            ),
            "joinKey": "faceId",
            "minViable": "per-face scalars + boolean perspectiveSuspect",
        },
        {
            "priority": 4,
            "signal": "skin_tone / age_band already in QC — keep; do not prioritize ethnicity for fusion",
            "why": (
                "Ethnicity mismatch is not enriched on the 8 control-right misses; "
                "demographic fusion is the wrong first bet and vote-risky."
            ),
            "joinKey": "faceId",
            "minViable": "already on disk via face-qc-v1",
        },
    ]

    summary = {
        "run": a.run,
        "yardstick": "uniform run-4, panel majority, bothHeldOut",
        "bothHeldOutPairs": len(all_both),
        "modelAccuracy": (
            sum(1 for r in all_both if r["bucket"] in ("bothRight", "modelRight_controlWrong"))
            / len(all_both)
            if all_both
            else None
        ),
        "modelWrong_controlRight": len(ctrl_right),
        "bothWrong": len(both_wrong),
        "ctrlRightBtGap": {
            "median": float(pd.Series([r["btGap"] for r in ctrl_right]).median())
            if ctrl_right
            else None,
            "lt1": sum(1 for r in ctrl_right if r["btGap"] < 1),
            "ge3": sum(1 for r in ctrl_right if r["btGap"] >= 3),
            "values": [round(r["btGap"], 3) for r in ctrl_right],
        },
        "verdict": (
            "Option 2 — fusion data too thin for Option 1 late-fusion prototype. "
            "On-disk signals are demographics + QC; landmarks/ratios missing. "
            "Do not ship ethnicity/QC late-fusion; ask Harsh for landmark+ratio export."
        ),
        "option1Attempted": False,
        "option1SkippedReason": (
            "labels/face-attributes.json is ethnicity/prodGender only; no landmark or "
            "ratio matrix to fuse with modelScore deltas without inventing geometry."
        ),
    }

    payload = {
        "summary": summary,
        "inventory": inventory,
        "ratesByCohort": rates,
        "controlRightMissIssueCounts": dict(issue_counter.most_common()),
        "controlRightEthnicityPairs": {
            f"{a}|{b}": n for (a, b), n in eth_pairs.most_common()
        },
        "controlRightGenderPairs": {
            f"{a}|{b}": n for (a, b), n in gender_pairs.most_common()
        },
        "askHarsh": ask_harsh,
        "controlRightMisses": ctrl_right,
        "note": (
            "Rates compare enrichment of available signals on control-right misses "
            "vs all bothHeldOut / model-correct. Vote-blind: uses bt-refit-v2-qc gaps "
            "and face-id val split; does not cut features on panel ranking."
        ),
    }

    out_json = Path(a.out) if a.out else art / "bothHeldOut-miss-feature-cross.json"
    out_csv = out_json.with_suffix(".csv")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if ctrl_right:
        # flatten lists for csv
        flat = []
        for r in ctrl_right:
            flat.append({k: v for k, v in r.items()})
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
            w.writeheader()
            w.writerows(flat)

    print(f"{a.run} bothHeldOut miss × feature cross")
    print(f"  bothHeldOut n={len(all_both)}  modelAcc={summary['modelAccuracy']:.1%}")
    print(f"  control-right misses: {len(ctrl_right)}  both-wrong: {len(both_wrong)}")
    print("  rates (photoIssueEither):")
    for name in ("allBothHeldOut", "modelCorrect", "modelWrong_controlRight"):
        r = rates[name]["photoIssueEither"]
        print(f"    {name}: {r['k']}/{r['n']} = {r['rate']:.1%}" if r["rate"] is not None else f"    {name}: n/a")
    print("  rates (ethnicityMismatch):")
    for name in ("allBothHeldOut", "modelCorrect", "modelWrong_controlRight"):
        r = rates[name]["ethnicityMismatch"]
        print(f"    {name}: {r['k']}/{r['n']} = {r['rate']:.1%}" if r["rate"] is not None else f"    {name}: n/a")
    print(f"-> {out_json.relative_to(ROOT)}")
    if ctrl_right:
        print(f"-> {out_csv.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
