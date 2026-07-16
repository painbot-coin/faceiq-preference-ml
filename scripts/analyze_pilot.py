#!/usr/bin/env python
"""Analyze pilot-500 votes: rater/Gemini/BT/model agreement by closeness bucket.

Joins labels/pilot-500/votes-*.jsonl with the export (Gemini label + confidence),
BT thetas, and optionally the neural comparator's predictions. All judges are
compared on the SAME outcome space (A/B/tie relative to the export's faceA/faceB).

Outputs to --out (default artifacts/pilot-500/):
  summary.md          headline numbers + tables (also printed to stdout)
  agreement.json      every agreement cell with counts, overall and per bucket
  per_pair.csv        full join, one row per pair — for ad-hoc digging
  disagreements.csv   pairs where raters disagree, with comments and photo URLs

Usage:
    python scripts/analyze_pilot.py --export data/exports/<runId>
        [--ratings artifacts/bt-refit-v1/ratings.csv] [--votes-dir labels/pilot-500]
        [--checkpoint checkpoints/train-v8-arcface-e2e-long/best.pt]
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from faceiq_pref.data import load_export

BUCKETS = [
    ("close", 0.0, 0.60),
    ("mid", 0.60, 0.80),
    ("clear", 0.80, 0.95),
    ("very_clear", 0.95, 1.01),
]
CALIBRATION_BINS = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.70), (0.70, 0.80),
                    (0.80, 0.90), (0.90, 0.95), (0.95, 1.01)]


def bucket_of(p: float) -> str:
    for name, lo, hi in BUCKETS:
        if lo <= p < hi:
            return name
    raise ValueError(f"win probability {p} out of range")


def load_thetas(ratings_csv: Path) -> dict[str, float]:
    with ratings_csv.open() as f:
        return {row["faceId"]: float(row["theta"]) for row in csv.DictReader(f)}


def load_votes_files(votes_dir: Path) -> dict[str, dict[str, dict]]:
    """rater -> pairId -> vote row (later lines win)."""
    raters: dict[str, dict[str, dict]] = {}
    for path in sorted(votes_dir.glob("votes-*.jsonl")):
        rater = path.stem.removeprefix("votes-")
        votes: dict[str, dict] = {}
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                votes[row["pairId"]] = row
        raters[rater] = votes
    return raters


def agreement(
    judge_a: dict[str, str], judge_b: dict[str, str], pair_ids: list[str]
) -> dict:
    """Exact-match agreement (ties count) + decisive-only (both picked a winner)."""
    common = [pid for pid in pair_ids if pid in judge_a and pid in judge_b]
    exact = sum(1 for pid in common if judge_a[pid] == judge_b[pid])
    decisive = [pid for pid in common if judge_a[pid] != "tie" and judge_b[pid] != "tie"]
    dec_match = sum(1 for pid in decisive if judge_a[pid] == judge_b[pid])
    return {
        "n": len(common),
        "exact": round(exact / len(common), 4) if common else None,
        "nDecisive": len(decisive),
        "decisive": round(dec_match / len(decisive), 4) if decisive else None,
    }


def score_faces_with_model(checkpoint: Path, export, face_ids: set[str]) -> dict[str, float]:
    """Comparator scores for just the faces used in the pilot pairs."""
    import torch
    from PIL import Image

    from faceiq_pref.model import PreferenceScorer, pick_device
    from faceiq_pref.train import TrainConfig, build_transforms

    device = pick_device()
    ckpt = torch.load(checkpoint, map_location=device)
    cfg = TrainConfig(**ckpt["config"])
    scorer = PreferenceScorer(cfg.backbone, pretrained=False).to(device)
    scorer.load_state_dict({k.removeprefix("scorer."): v for k, v in ckpt["model"].items()})
    scorer.eval()
    tf = build_transforms(cfg.backbone, cfg.image_size, augment=False)

    faces = [f for fid, f in export.faces().items() if fid in face_ids]
    missing = [f.face_id for f in faces if not export.image_path(f).exists()]
    if missing:
        raise RuntimeError(f"{len(missing)} pilot faces missing local images (need data/)")
    scores: dict[str, float] = {}
    with torch.no_grad():
        for i in range(0, len(faces), 64):
            batch = faces[i : i + 64]
            tensors = []
            for f in batch:
                with Image.open(export.image_path(f)) as img:
                    tensors.append(tf(img.convert("RGB")))
            out = scorer(torch.stack(tensors).to(device))
            scores.update(zip((f.face_id for f in batch), out.cpu().tolist()))
    return scores


def fmt_pct(v: float | None) -> str:
    return "—" if v is None else f"{v:.1%}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True, help="export directory (contains manifest.json)")
    ap.add_argument("--ratings", default="artifacts/bt-refit-v1/ratings.csv")
    ap.add_argument("--pairs", default="labels/pilot-500/pairs.json")
    ap.add_argument("--votes-dir", default="labels/pilot-500")
    ap.add_argument("--out", default="artifacts/pilot-500")
    ap.add_argument("--checkpoint", help="optional comparator checkpoint for model-vs-human")
    args = ap.parse_args()

    export = load_export(args.export)
    matchups = {m.comparison_id: m for m in export.all_matchups()}
    thetas = load_thetas(Path(args.ratings))
    pairs_doc = json.loads(Path(args.pairs).read_text())
    pairs = pairs_doc["pairs"]
    pair_ids = [p["pairId"] for p in pairs]
    votes_by_rater = load_votes_files(Path(args.votes_dir))
    if not votes_by_rater:
        print(f"no votes-*.jsonl files in {args.votes_dir}", file=sys.stderr)
        return 1
    for rater, votes in votes_by_rater.items():
        print(f"rater {rater}: {len(votes)} votes")

    # -- normalize every judge to A/B/tie relative to the export's faceA/faceB --

    left_face = {p["pairId"]: p["left"]["faceId"] for p in pairs}
    judges: dict[str, dict[str, str]] = {}
    comments: dict[str, dict[str, str]] = {}
    for rater, votes in votes_by_rater.items():
        outcomes: dict[str, str] = {}
        cms: dict[str, str] = {}
        for pid, row in votes.items():
            if pid not in matchups:
                continue
            if row["choice"] == "tie":
                outcomes[pid] = "tie"
            else:
                picked_left = row["choice"] == "left"
                winner = left_face[pid] if picked_left else next(
                    p["right"]["faceId"] for p in pairs if p["pairId"] == pid
                )
                outcomes[pid] = "A" if winner == matchups[pid].face_a_id else "B"
            cms[pid] = row.get("comment", "")
        judges[rater] = outcomes
        comments[rater] = cms

    judges["gemini"] = {
        pid: matchups[pid].vlm_outcome
        for pid in pair_ids
        if matchups[pid].vlm_outcome is not None
    }

    pinfo: dict[str, dict] = {}
    for pid in pair_ids:
        m = matchups[pid]
        ta, tb = thetas[m.face_a_id], thetas[m.face_b_id]
        p = 1.0 / (1.0 + math.exp(-abs(ta - tb)))
        pinfo[pid] = {"p": p, "bucket": bucket_of(p), "btFavorite": "A" if ta >= tb else "B"}
    judges["bt"] = {pid: pinfo[pid]["btFavorite"] for pid in pair_ids}

    if args.checkpoint:
        needed = {matchups[pid].face_a_id for pid in pair_ids}
        needed |= {matchups[pid].face_b_id for pid in pair_ids}
        print(f"scoring {len(needed)} faces with {args.checkpoint} ...")
        model_scores = score_faces_with_model(Path(args.checkpoint), export, needed)
        judges["model"] = {
            pid: "A"
            if model_scores[matchups[pid].face_a_id] >= model_scores[matchups[pid].face_b_id]
            else "B"
            for pid in pair_ids
        }

    # -- agreement matrices ---------------------------------------------------

    judge_names = list(judges)
    matrix: dict[str, dict] = {}
    for a, b in itertools.combinations(judge_names, 2):
        cell = {"overall": agreement(judges[a], judges[b], pair_ids)}
        for bname, _, _ in BUCKETS:
            bucket_pids = [pid for pid in pair_ids if pinfo[pid]["bucket"] == bname]
            cell[bname] = agreement(judges[a], judges[b], bucket_pids)
        matrix[f"{a} vs {b}"] = cell

    # -- BT calibration: how often does each judge pick the BT favorite -------

    calibration: dict[str, list[dict]] = {}
    for name in judge_names:
        if name == "bt":
            continue
        rows = []
        for lo, hi in CALIBRATION_BINS:
            in_bin = [
                pid
                for pid in pair_ids
                if lo <= pinfo[pid]["p"] < hi
                and judges[name].get(pid) not in (None, "tie")
            ]
            picked = sum(
                1 for pid in in_bin if judges[name][pid] == pinfo[pid]["btFavorite"]
            )
            rows.append(
                {
                    "bin": f"{lo:.2f}-{hi:.2f}",
                    "n": len(in_bin),
                    "pickedBtFavorite": round(picked / len(in_bin), 4) if in_bin else None,
                }
            )
        calibration[name] = rows

    # -- Gemini confidence vs human agreement ----------------------------------

    human_raters = [n for n in judge_names if n not in ("gemini", "bt", "model")]
    confidence_table: list[dict] = []
    for conf in ("high", "medium", "low"):
        for bname, _, _ in BUCKETS:
            pids = [
                pid
                for pid in pair_ids
                if matchups[pid].confidence == conf and pinfo[pid]["bucket"] == bname
            ]
            row: dict = {"confidence": conf, "bucket": bname, "n": len(pids)}
            for rater in human_raters:
                row[f"gemini_vs_{rater}"] = agreement(judges["gemini"], judges[rater], pids)[
                    "exact"
                ]
            confidence_table.append(row)

    # -- per-pair join + rater disagreements -----------------------------------

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_pair_rows = []
    disagreements = []
    for p in pairs:
        pid = p["pairId"]
        m = matchups[pid]
        row = {
            "pairId": pid,
            "gender": p["gender"],
            "bucket": pinfo[pid]["bucket"],
            "winProb": round(pinfo[pid]["p"], 4),
            "btFavorite": pinfo[pid]["btFavorite"],
            "gemini": judges["gemini"].get(pid),
            "geminiConfidence": m.confidence,
            "leftPhotoUrl": p["left"]["photoUrl"],
            "rightPhotoUrl": p["right"]["photoUrl"],
        }
        if "model" in judges:
            row["model"] = judges["model"].get(pid)
        for rater in human_raters:
            row[rater] = judges[rater].get(pid)
            row[f"{rater}_comment"] = comments[rater].get(pid, "")
        per_pair_rows.append(row)

        rater_votes = [judges[r].get(pid) for r in human_raters]
        voted = [v for v in rater_votes if v is not None]
        if len(voted) >= 2 and len(set(voted)) > 1:
            disagreements.append(row)

    for name, rows in (("per_pair.csv", per_pair_rows), ("disagreements.csv", disagreements)):
        if rows:
            with (out_dir / name).open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    (out_dir / "agreement.json").write_text(
        json.dumps(
            {
                "raters": {r: len(v) for r, v in votes_by_rater.items()},
                "matrix": matrix,
                "calibration": calibration,
                "geminiConfidence": confidence_table,
            },
            indent=1,
        )
        + "\n"
    )

    # -- markdown summary -------------------------------------------------------

    lines = ["# Pilot-500 agreement summary", ""]
    lines.append("Votes: " + ", ".join(f"{r} = {len(v)}" for r, v in votes_by_rater.items()))
    lines.append(f"Rater disagreements (both voted, different outcome): {len(disagreements)}")
    lines.append("")
    lines.append("## Agreement (exact / decisive-only)")
    lines.append("")
    header = "| pair | overall | " + " | ".join(b for b, _, _ in BUCKETS) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(BUCKETS) + 2))
    for key, cell in matrix.items():
        cols = [
            f"{fmt_pct(cell[c]['exact'])} / {fmt_pct(cell[c]['decisive'])} (n={cell[c]['n']})"
            for c in ["overall"] + [b for b, _, _ in BUCKETS]
        ]
        lines.append(f"| {key} | " + " | ".join(cols) + " |")
    lines.append("")
    lines.append("## BT calibration: fraction picking the BT favorite (decisive votes)")
    lines.append("")
    lines.append("| p bin | " + " | ".join(calibration) + " |")
    lines.append("|" + "---|" * (len(calibration) + 1))
    for i, (lo, hi) in enumerate(CALIBRATION_BINS):
        cols = [
            f"{fmt_pct(rows[i]['pickedBtFavorite'])} (n={rows[i]['n']})"
            for rows in calibration.values()
        ]
        lines.append(f"| {lo:.2f}-{hi:.2f} | " + " | ".join(cols) + " |")
    lines.append("")
    lines.append("## Gemini agreement with each rater, by Gemini confidence x bucket")
    lines.append("")
    lines.append(
        "| confidence | bucket | n | "
        + " | ".join(f"vs {r}" for r in human_raters)
        + " |"
    )
    lines.append("|" + "---|" * (3 + len(human_raters)))
    for row in confidence_table:
        if row["n"] == 0:
            continue
        cols = [fmt_pct(row[f"gemini_vs_{r}"]) for r in human_raters]
        lines.append(
            f"| {row['confidence']} | {row['bucket']} | {row['n']} | " + " | ".join(cols) + " |"
        )
    summary = "\n".join(lines) + "\n"
    (out_dir / "summary.md").write_text(summary)
    print("\n" + summary)
    print(f"wrote {out_dir}/summary.md, agreement.json, per_pair.csv, disagreements.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
