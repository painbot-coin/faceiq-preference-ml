"""Turn archived Prolific panel judgments into soft training targets.

The export gives one hard winner per pair, almost always Gemini's. The panel gives us
something strictly richer on 7,780 of those pairs: a *vote share*. A 6-0 split and a 4-2
split are both "A wins" to the export, but they are very different claims about the world,
and the 4-2 case is the one the comparator currently gets wrong.

`load_panel_targets` collapses raw judgments into `{pair_index: target}` where the target is
the smoothed share of votes for face A, ready to hand to `BCEWithLogitsLoss` — which accepts
a float target natively.

Two details that matter more than they look:

* **Ties are votes, not missing data.** A rater who said "too close to call" (8.1% of all
  judgments) contributes half a vote to each side, matching how `bt_panel.py` treats them.
  A pair the panel split exactly evenly lands on 0.5 and teaches "these are equal" — the one
  thing the export can never express, because the VLM still named a winner on all 788 of them.
* **Raw shares are overconfident on small samples.** With six votes a 6-0 sweep would become
  a target of 1.0, asserting certainty from six clicks. A Laplace prior pulls it to 0.875
  while leaving 3-3 at exactly 0.5, so the strength of the target tracks the strength of the
  evidence.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def load_rejects(paths: list[str] | None) -> set[str]:
    """Prolific IDs whose votes are excluded from every fit (see analyze_panel_run.py)."""
    out: set[str] = set()
    for p in paths or []:
        if Path(p).exists():
            out |= set(Path(p).read_text().split())
    return out


def load_panel_votes(
    panels: list[tuple[str, str]],
    rejects: set[str] | None = None,
) -> dict[int, tuple[float, float]]:
    """Pool raw judgments across studies -> {pair_index: (votes for A, votes for B)}.

    `panels` is a list of (results_dir, sample_meta.json). Gold pairs, the `test` study and
    rejected raters are dropped. Pooling is keyed on `pairIndex`, which is stable across
    studies because every draw indexes into the same export.
    """
    rejects = rejects or set()
    votes: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for results_dir, meta_path in panels:
        meta = json.loads(Path(meta_path).read_text())["pairs"]
        by_pair_id = {m["pairId"]: m for m in meta}
        for line in (Path(results_dir) / "judgments.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            j = json.loads(line)
            if j["studyId"] == "test" or j["isGold"] or j["prolificPid"] in rejects:
                continue
            pair = by_pair_id.get(j["pairId"])
            if pair is None:
                continue
            cell = votes[pair["pairIndex"]]
            # The app randomises sides, so a "left" click only names a face via the draw.
            a_side = "left" if pair["faceAOnLeft"] else "right"
            if j["choice"] == "tie":
                cell[0] += 0.5
                cell[1] += 0.5
            elif j["choice"] == a_side:
                cell[0] += 1.0
            else:
                cell[1] += 1.0
    return {k: (v[0], v[1]) for k, v in votes.items()}


def load_panel_targets(
    panels: list[tuple[str, str]],
    rejects: set[str] | None = None,
    min_votes: int = 4,
    prior: float = 1.0,
    hard: bool = False,
) -> dict[int, float]:
    """{pair_index: P(A wins)} from panel votes, smoothed and thresholded on coverage.

    `min_votes` drops thinly covered pairs: run 2 predated the queue fix and left some pairs
    with one or two votes, which carry no more information than a coin toss but would
    otherwise produce a maximally confident target. `prior` is the Laplace count added to
    both sides.

    `hard` rounds each target back to 0 or 1, keeping only *which* face the crowd preferred
    and discarding *by how much*. It exists to isolate the two halves of the change: an arm
    trained this way tells us how much of any gain comes from correcting the winner versus
    from admitting uncertainty. Exactly-split pairs have no majority and are dropped.
    """
    out = {}
    for idx, (wa, wb) in load_panel_votes(panels, rejects).items():
        if wa + wb < min_votes:
            continue
        if hard:
            if wa != wb:
                out[idx] = 1.0 if wa > wb else 0.0
        else:
            out[idx] = (wa + prior) / (wa + wb + 2 * prior)
    return out
