#!/usr/bin/env python
"""Score a trained comparator against real human votes instead of machine labels.

Every accuracy we quote for the neural model so far — 78.4% for `train-v8` — is measured
against the export's `finalOutcome`, which is a VLM label on ~98% of rows. The panel
proved the VLM is at chance on pairs where two faces are close. So that 78.4% is partly
"how well does the network reproduce Gemini", including Gemini's blind spot, and it cannot
tell us whether the network sees what people see.

This measures the thing we actually care about: given a pair, does the network pick the
face that more raters picked? Two guards make the number honest.

1. **No leakage.** Only panel pairs where *both* faces are in this checkpoint's own
   validation split are scored, rebuilt from the `val_fraction`/`split_seed` stored in the
   checkpoint. A pair whose faces the network trained on tells us nothing.
2. **Same metric as the BT work.** Accuracy is the share of *individual* human votes
   agreeing with the model's pick, identical to `panel_run_delta.py`, so the network and
   the ranking can be compared on one scale against the same ceiling.

Usage:
    python scripts/eval_vs_panel.py --export data/exports/<runId> \
        --checkpoint checkpoints/train-v10-arcface-val50/best.pt \
        --panel labels/panel-pilot/results labels/panel-pilot/sample-meta.json \
        --panel labels/panel-run-3/results labels/panel-run-3/sample-meta.json \
        --rejects artifacts/panel-run-v1/reject-pids.txt \
                  artifacts/panel-run-v3/reject-pids.txt \
        --ratings artifacts/bt-refit-v4-panel/ratings.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from faceiq_pref.data import load_export  # noqa: E402
from faceiq_pref.eval import score_all_faces  # noqa: E402
from faceiq_pref.train import TrainConfig  # noqa: E402

BANDS = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 45), (45, 101)]


def load_votes(panels: list[tuple[str, str]], rejects: set[str]):
    """-> {pair_index: (wins A, wins B)} pooled over studies, ties split half each."""
    votes: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for results, meta_path in panels:
        meta = json.loads(Path(meta_path).read_text())["pairs"]
        by_id = {m["pairId"]: m for m in meta}
        for line in (Path(results) / "judgments.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            j = json.loads(line)
            if j["studyId"] == "test" or j["isGold"] or j["prolificPid"] in rejects:
                continue
            m = by_id.get(j["pairId"])
            if m is None:
                continue
            cell = votes[m["pairIndex"]]
            a_side = "left" if m["faceAOnLeft"] else "right"
            if j["choice"] == "tie":
                cell[0] += 0.5
                cell[1] += 0.5
            elif j["choice"] == a_side:
                cell[0] += 1.0
            else:
                cell[1] += 1.0
    return {k: (v[0], v[1]) for k, v in votes.items()}


def val_faces_of(cfg: TrainConfig, matchups) -> set[str]:
    """Rebuild the checkpoint's own val face set — the same logic as split_by_face_id."""
    face_ids = sorted({f for m in matchups for f in (m.face_a_id, m.face_b_id)})
    shuffled = face_ids[:]
    random.Random(cfg.split_seed).shuffle(shuffled)
    return set(shuffled[: int(len(shuffled) * cfg.val_fraction)])


def agreement(pick_a: dict[int, bool], votes, keys) -> tuple[float, float]:
    """Share of individual human votes agreeing with a predictor's pick."""
    correct = total = 0.0
    for idx in keys:
        wa, wb = votes[idx]
        if idx not in pick_a:
            continue
        correct += wa if pick_a[idx] else wb
        total += wa + wb
    return (correct / total if total else float("nan")), total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--panel", nargs=2, action="append", required=True,
                    metavar=("RESULTS", "META"))
    ap.add_argument("--rejects", nargs="*", default=[])
    ap.add_argument("--ratings", action="append", default=[],
                    help="repeatable: BT ratings.csv for a head-to-head on these pairs")
    ap.add_argument("--gap-ranking",
                    help="which ratings.csv defines the percentile-gap bands "
                         "(default: the first --ratings)")
    ap.add_argument("--exclude-faces", default="artifacts/face-qc-v1/exclude-faces.csv")
    ap.add_argument("--exclude-genders", default="artifacts/face-qc-v1/gender-fixes.csv")
    ap.add_argument("--out")
    ap.add_argument("--dump-pairs",
                    help="per-pair picks + vote counts as CSV. Two runs' dumps cover the same "
                         "pairs, so scripts/compare_panel_evals.py can pair the comparison "
                         "instead of treating the two accuracies as independent samples.")
    args = ap.parse_args()

    rejects: set[str] = set()
    for p in args.rejects:
        if Path(p).exists():
            rejects |= set(Path(p).read_text().split())

    export = load_export(args.export)
    matchups = export.all_matchups()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = TrainConfig.from_saved(ckpt["config"])
    print(f"checkpoint {args.checkpoint}")
    print(f"  {cfg.backbone}, image_size {cfg.image_size}, "
          f"val_fraction {cfg.val_fraction}, split_seed {cfg.split_seed}, "
          f"epoch {ckpt.get('epoch')}")

    # The val split is rebuilt on the UNFILTERED export, exactly as training did it, so
    # the face set matches the run. QC exclusions are applied afterwards.
    val = val_faces_of(cfg, matchups)
    dropped: set[str] = set()
    for p in (args.exclude_faces, args.exclude_genders):
        if p and Path(p).exists():
            with open(p, newline="") as fh:
                dropped.update(r["faceId"] for r in csv.DictReader(fh))
    by_index = {m.pair_index: m for m in matchups
                if m.face_a_id not in dropped and m.face_b_id not in dropped}

    votes = load_votes([tuple(p) for p in args.panel], rejects)
    keys = [i for i, m in ((i, by_index[i]) for i in votes if i in by_index)
            if m.face_a_id in val and m.face_b_id in val]
    n_votes = sum(sum(votes[i]) for i in keys)
    print(f"\npanel pairs with votes: {len(votes):,}")
    print(f"  of those, fully inside this checkpoint's val split: {len(keys):,} "
          f"({n_votes:,.0f} human votes)")
    if not keys:
        raise SystemExit("No leak-free panel pairs. Train a run with a larger "
                         "val_fraction, or split so the panel pairs land in val.")

    scores = score_all_faces(args.checkpoint, export, image_size=cfg.image_size)
    nn_pick = {i: scores[by_index[i].face_a_id] > scores[by_index[i].face_b_id]
               for i in keys
               if by_index[i].face_a_id in scores and by_index[i].face_b_id in scores}
    print(f"  scored by the network: {len(nn_pick):,}")

    # The VLM label on the same pairs: the yardstick the network was actually trained
    # and validated against. The gap between these two columns is the whole point.
    vlm_pick = {i: by_index[i].final_outcome == "A" for i in keys
                if by_index[i].final_outcome in ("A", "B")}

    preds = {"neural comparator": nn_pick, "export label (VLM)": vlm_pick}
    bt_pct: dict[str, float] = {}
    gap_source = args.gap_ranking or (args.ratings[0] if args.ratings else None)
    circular: set[str] = set()
    for rp in args.ratings:
        if not Path(rp).exists():
            continue
        with open(rp, newline="") as fh:
            rows = list(csv.DictReader(fh))
        name = Path(rp).parent.name
        # A refit that absorbed the panel's votes is scoring its own training data here,
        # and can even beat the crowd ceiling by having memorised vote directions. Flag it
        # rather than silently reporting it next to honest numbers.
        mp = Path(rp).parent / "metrics.json"
        if mp.exists() and "panelVotes" in json.loads(mp.read_text()):
            circular.add(name)
        theta = {r["faceId"]: float(r["theta"]) for r in rows}
        if rp == gap_source:
            bt_pct = {r["faceId"]: float(r["percentile"]) for r in rows}
        preds[f"BT {name}"] = {
            i: theta[by_index[i].face_a_id] > theta[by_index[i].face_b_id]
            for i in keys
            if by_index[i].face_a_id in theta and by_index[i].face_b_id in theta
        }

    # Ceiling: how often the majority of the crowd predicts one of its own members.
    # Leave-one-out, so it is not inflated by the vote being counted twice.
    loo_hits = loo_n = 0.0
    for i in keys:
        wa, wb = votes[i]
        for w, other in ((wa, wb), (wb, wa)):
            if w:
                loo_hits += w if (w - 1) > other else 0.0
                loo_n += w
    ceiling = loo_hits / loo_n if loo_n else float("nan")

    print(f"\n{'predictor':30} {'agrees with human votes':>24} {'votes':>9}")
    summary = {}
    for label, pick in preds.items():
        acc, tot = agreement(pick, votes, keys)
        summary[label] = acc
        mark = "  <- CIRCULAR, fit on these votes" if any(
            c in label for c in circular) else ""
        print(f"{label:30} {acc:23.2%} {tot:9,.0f}{mark}")
    print(f"{'ceiling (another rater)':30} {ceiling:23.2%}")
    if circular:
        print("\nCIRCULAR rows absorbed these pairs' votes during fitting, so they are "
              "scoring their own\ntraining data and can exceed the ceiling. Compare the "
              "network against an honest row.")

    # By how far apart the ranking puts the two faces: where does the network add
    # something the ranking does not already have?
    if bt_pct:
        print(f"\nby percentile gap in {Path(gap_source).parent.name}:")
        head = f"{'gap':>9} {'pairs':>6}"
        for label in preds:
            head += f" {label[:18]:>19}"
        print(head)
        bands = []
        for lo, hi in BANDS:
            sel = [i for i in keys
                   if by_index[i].face_a_id in bt_pct and by_index[i].face_b_id in bt_pct
                   and lo <= abs(bt_pct[by_index[i].face_a_id]
                                 - bt_pct[by_index[i].face_b_id]) * 100 < hi]
            if not sel:
                continue
            row = {"band": f"{lo}-{min(hi, 100)}", "pairs": len(sel)}
            line = f"{row['band']:>9} {len(sel):6}"
            for label, pick in preds.items():
                a, _ = agreement(pick, votes, sel)
                row[label] = a
                line += f" {a:18.1%} "
            bands.append(row)
            print(line)

    if args.dump_pairs:
        dp = Path(args.dump_pairs)
        dp.parent.mkdir(parents=True, exist_ok=True)
        cols = sorted(preds)
        with open(dp, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["pairIndex", "votesA", "votesB", "gapPct", *cols])
            for i in sorted(keys):
                m = by_index[i]
                gap = ""
                if bt_pct and m.face_a_id in bt_pct and m.face_b_id in bt_pct:
                    gap = f"{abs(bt_pct[m.face_a_id] - bt_pct[m.face_b_id]) * 100:.4f}"
                wa, wb = votes[i]
                w.writerow([i, wa, wb, gap,
                            *("" if i not in preds[c] else int(preds[c][i]) for c in cols)])
        print(f"wrote {dp} ({len(keys):,} pairs)")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "checkpoint": args.checkpoint,
            "backbone": cfg.backbone,
            "valFraction": cfg.val_fraction,
            "splitSeed": cfg.split_seed,
            "protocol": "panel pairs with both faces in the checkpoint's val split; "
                        "accuracy is the share of individual human votes matching the "
                        "predictor's pick",
            "pairsScored": len(keys),
            "humanVotes": float(n_votes),
            "ceiling": ceiling,
            "accuracy": summary,
            "circularRankings": sorted(circular),
            "byGapBand": bands if bt_pct else None,
        }, indent=2))
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
