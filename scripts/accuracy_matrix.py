#!/usr/bin/env python
"""One table, every predictor, both accuracy metrics, both pair distributions.

Written because the programme's own write-ups mixed two different "accuracy" numbers, and the
mix flattered one predictor and penalised another:

* **vs the majority label** — did the predictor pick the face that more raters picked? A pair
  split 9-3 counts as one clean win, so this metric can reach 100%.
* **vs individual votes** — what share of the raw ballots agree with the pick? On that same 9-3
  pair a *perfect* predictor scores 75%, because a quarter of the crowd disagreed with itself.

The two are not comparable and each needs its own ceiling (one rater against the majority of
the others, computed on the same footing). `analyze_panel_run.py` reports the first,
`eval_vs_panel.py` and `panel_run_delta.py` report the second. Quoting a number from one next
to a number from the other, which earlier drafts of log §5.8 did, produces a 10-point artefact.

And both change again with the pair distribution: near-tie draws (runs 2-3) versus a uniform
draw (run 4). So there are four cells per predictor, and every claim has to say which one it is.

Usage:
    python scripts/accuracy_matrix.py \
        --ratings artifacts/bt-refit-v2-qc/ratings.csv \
        --panel labels/panel-run-4-random/results labels/panel-run-4-random/sample-meta.json \
        --label "uniform (run 4)" \
        --panel labels/panel-run-3/results labels/panel-run-3/sample-meta.json \
        --label "near-tie (run 3)" \
        --rejects artifacts/panel-run-v4/reject-pids.txt \
                  artifacts/panel-run-v3/reject-pids.txt \
        --out artifacts/panel-run-v4/accuracy-matrix.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def load_votes(results_dir: Path, meta_path: Path, rejects: set[str], drop_rejects: bool):
    """-> {pairId: (votes for face A, votes for face B)}, ties split half each."""
    meta = {m["pairId"]: m for m in json.loads(meta_path.read_text())["pairs"]}
    votes: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for line in (results_dir / "judgments.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        j = json.loads(line)
        if j["isGold"] or j["studyId"] == "test":
            continue
        if drop_rejects and j["prolificPid"] in rejects:
            continue
        m = meta.get(j["pairId"])
        if m is None:
            continue
        cell = votes[j["pairId"]]
        a_side = "left" if m["faceAOnLeft"] else "right"
        if j["choice"] == "tie":
            cell[0] += 0.5
            cell[1] += 0.5
        elif j["choice"] == a_side:
            cell[0] += 1.0
        else:
            cell[1] += 1.0
    return {k: tuple(v) for k, v in votes.items()}, meta


def score_predictor(votes, picks_a: dict[str, bool]) -> dict[str, float]:
    """Both metrics for one predictor, over the pairs it can score."""
    maj_hit = maj_n = vote_hit = vote_n = 0.0
    for pid, (wa, wb) in votes.items():
        if pid not in picks_a:
            continue
        a_wins = picks_a[pid]
        vote_hit += wa if a_wins else wb
        vote_n += wa + wb
        if wa != wb:                      # a dead split has no majority to be right about
            maj_hit += (wa > wb) == a_wins
            maj_n += 1
    return {"vsMajority": maj_hit / maj_n if maj_n else float("nan"),
            "vsVotes": vote_hit / vote_n if vote_n else float("nan"),
            "pairsWithMajority": int(maj_n), "votes": vote_n}


def ceilings(votes) -> dict[str, float]:
    """One rater against the majority of the others, on each metric's own footing.

    Leave-one-out both times, so the held-out vote is never counted on both sides. The two
    numbers differ a lot (~75% vs ~70% on a uniform draw) and each is the only honest
    comparator for its own metric.
    """
    maj_hit = maj_n = vote_hit = vote_n = 0.0
    for wa, wb in votes.values():
        for w, other in ((wa, wb), (wb, wa)):
            if not w:
                continue
            vote_hit += w if (w - 1) > other else 0.0
            vote_n += w
            # Same leave-one-out logic, but scored per rater against the *majority label* of the
            # remaining ballots rather than pooled across ballots.
            if (w - 1) != other:
                maj_hit += w if (w - 1) > other else 0.0
                maj_n += w
    return {"vsMajority": maj_hit / maj_n if maj_n else float("nan"),
            "vsVotes": vote_hit / vote_n if vote_n else float("nan")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", action="append", default=[],
                    help="repeatable: a BT ratings.csv to score")
    ap.add_argument("--panel", nargs=2, action="append", required=True,
                    metavar=("RESULTS", "META"))
    ap.add_argument("--label", action="append", default=[],
                    help="one per --panel, for the output table")
    ap.add_argument("--rejects", nargs="*", default=[])
    ap.add_argument("--keep-rejects", action="store_true",
                    help="do NOT drop excluded raters — reproduces analyze_panel_run.py")
    ap.add_argument("--out")
    args = ap.parse_args()

    rejects: set[str] = set()
    for p in args.rejects:
        if Path(p).exists():
            rejects |= set(Path(p).read_text().split())

    rankings = {}
    for rp in args.ratings or ["artifacts/bt-refit-v2-qc/ratings.csv"]:
        path = Path(rp)
        if not path.exists():
            continue
        mp = path.parent / "metrics.json"
        circular = mp.exists() and "panelVotes" in json.loads(mp.read_text())
        with open(path, newline="") as fh:
            rankings[path.parent.name + (" (CIRCULAR)" if circular else "")] = {
                r["faceId"]: float(r["theta"]) for r in csv.DictReader(fh)
            }

    out = {}
    for i, (results, meta_path) in enumerate(args.panel):
        label = args.label[i] if i < len(args.label) else Path(results).parts[-2]
        votes, meta = load_votes(Path(results), Path(meta_path), rejects,
                                 drop_rejects=not args.keep_rejects)
        rows = {}
        vlm = {p: meta[p]["vlmOutcome"] == "A" for p in votes
               if meta[p]["vlmOutcome"] in ("A", "B")}
        rows["Gemini label"] = score_predictor(votes, vlm)
        for name, theta in rankings.items():
            picks = {p: theta[meta[p]["faceAId"]] > theta[meta[p]["faceBId"]] for p in votes
                     if meta[p]["faceAId"] in theta and meta[p]["faceBId"] in theta}
            rows[f"BT {name}"] = score_predictor(votes, picks)
        rows["ceiling (one rater)"] = ceilings(votes)
        out[label] = {"pairs": len(votes),
                      "votes": sum(sum(v) for v in votes.values()), "predictors": rows}

        print(f"\n=== {label} — {len(votes):,} pairs, "
              f"{sum(sum(v) for v in votes.values()):,.0f} votes"
              f"{'' if args.keep_rejects else ', excluded raters dropped'} ===")
        print(f"{'predictor':34} {'vs majority label':>18} {'vs individual votes':>20}")
        for name, r in rows.items():
            print(f"{name:34} {r['vsMajority']:17.1%} {r['vsVotes']:19.1%}")

    print("\nThe two columns are NOT interchangeable. Each must be read against the ceiling in\n"
          "its own column: a predictor beating 'vs majority' does not imply it beats a person\n"
          "at predicting one person's vote, and vice versa.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
