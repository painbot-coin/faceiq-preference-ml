"""Blend several per-face scores into one rating, and measure the blend against people.

The tempting move is to average Labs, the network, and BT and call the result better
because it "uses more information". Averaging correlated estimators of the same quantity
only helps when their errors are independent, and there is no reason to assume that here:
the network was trained on VLM labels, BT was fit on the same VLM labels, and Labs is the
formula we are replacing. A blend can easily be worse than its best component.

So this module blends AND scores, on the only yardstick that settles it — the share of
individual human votes a predictor agrees with, on pairs it could not have memorised:

- **Leak-free pairs only.** Each neural component is restricted to its own val split,
  rebuilt from `val_fraction`/`split_seed`; a blend is restricted to the INTERSECTION.
  Since every run shares seed 42 and the split takes a prefix of one shuffle, a 0.2 split
  nests inside a 0.5 one, so mixing them costs pairs but stays valid.
- **Paired bootstrap**, resampling pairs. Blend and component are scored on the same
  pairs, so an unpaired comparison of two accuracies would overstate the uncertainty of
  their difference.
- **Circularity is refused, not annotated.** A BT refit that absorbed the panel's votes
  scores its own training data and can beat the crowd ceiling; `Source.circular` marks it.

Normalisation is always WITHIN GENDER. Male and female faces were never compared to each
other (the pair queue is same-gender), so the two theta scales share no origin and a
pooled z-score would silently encode "which gender scores higher".

Geometric mean is deliberately not offered: theta is signed, so the product is undefined,
and on an interval scale a geometric mean has no meaning even where it computes.
"""

from __future__ import annotations

import csv
import json
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

BANDS = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 45), (45, 101)]
METHODS = ("percentile", "zscore")


@dataclass
class Source:
    """One per-face score column, plus what makes it (un)trustworthy on panel pairs."""

    name: str
    kind: str                       # model | external | labs | bt
    scores: dict[str, float]
    val_faces: set[str] | None = None   # None = never trained on our faces
    circular: bool = False
    caveat: str = ""
    label: str = ""

    def __post_init__(self) -> None:
        if not self.label:
            self.label = self.name


def val_faces_of(val_fraction: float, split_seed: int, matchups) -> set[str]:
    """Rebuild a run's val face set — must mirror `train.split_by_face_id` exactly.

    Built from the UNFILTERED export, as training did it; QC exclusions apply afterwards.
    """
    face_ids = sorted({f for m in matchups for f in (m.face_a_id, m.face_b_id)})
    shuffled = face_ids[:]
    random.Random(split_seed).shuffle(shuffled)
    return set(shuffled[: int(len(shuffled) * val_fraction)])


def _val_split_for_run(run_dir: Path, matchups) -> set[str] | None:
    """A run's val faces, from its own config or — for an ensemble — from its members.

    An ensemble's `eval.json` records member checkpoints but no split of its own. Its
    leak-free set is the intersection of the members', since a face any member trained on
    is a face the ensemble effectively saw.
    """
    cfg_path = run_dir / "metrics.json"
    if cfg_path.exists():
        cfg = (json.loads(cfg_path.read_text()).get("config") or {})
        if cfg.get("val_fraction") is not None:
            return val_faces_of(float(cfg["val_fraction"]),
                                int(cfg.get("split_seed", 42)), matchups)

    eval_path = run_dir / "eval.json"
    if not eval_path.exists():
        return None
    members = (json.loads(eval_path.read_text()).get("checkpoints") or [])
    val: set[str] | None = None
    for ck in members:
        member_dir = run_dir.parent / Path(ck).parent.name
        cfg_path = member_dir / "metrics.json"
        if not cfg_path.exists():
            return None
        cfg = (json.loads(cfg_path.read_text()).get("config") or {})
        if cfg.get("val_fraction") is None:
            return None
        v = val_faces_of(float(cfg["val_fraction"]), int(cfg.get("split_seed", 42)),
                         matchups)
        val = v if val is None else (val & v)
    return val


def discover_sources(root: Path, faces: dict, matchups) -> list[Source]:
    """Every blendable score column on disk: training runs, externals, Labs, BT refits."""
    out: list[Source] = []

    for path in sorted((root / "artifacts").glob("*/model_scores.csv")):
        name = path.parent.name
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        scores = {r["faceId"]: float(r["modelScore"]) for r in rows
                  if r.get("modelScore") not in (None, "")}
        if not scores:
            continue
        external = name.startswith("external-")
        val: set[str] | None = None
        caveat = ""
        if external:
            caveat = ("trained on an outside dataset under a non-commercial research "
                      "licence — diagnostic only, must not ship")
        else:
            val = _val_split_for_run(path.parent, matchups)
            if val is None:
                # Fail closed. A model trained on our faces with an unknown split would
                # otherwise be scored on pairs it may have memorised, and land at the top
                # of the table for exactly that reason.
                val = set()
                caveat = ("no val_fraction on record and its members could not be "
                          "resolved, so no pair can be shown to be leak-free")
        out.append(Source(name=name, kind="external" if external else "model",
                          scores=scores, val_faces=val, caveat=caveat))

    labs = {fid: float(f.labs_overall_score) for fid, f in faces.items()
            if getattr(f, "labs_overall_score", None) is not None}
    if labs:
        out.append(Source(
            name="labsOverallScore", kind="labs", scores=labs, val_faces=None,
            label="Labs overall_score (legacy)",
            caveat=("the legacy formula this project exists to replace; a CLAUDE.md hard "
                    "rule bars it as a training label. Included as a baseline only"),
        ))

    for path in sorted((root / "artifacts").glob("*/ratings.csv")):
        name = path.parent.name
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        theta = {r["faceId"]: float(r["theta"]) for r in rows}
        if not theta:
            continue
        mp = path.parent / "metrics.json"
        circ = mp.exists() and "panelVotes" in json.loads(mp.read_text())
        out.append(Source(
            name=name, kind="bt", scores=theta, val_faces=None, circular=circ,
            label=f"BT {name}",
            caveat=("absorbed the panel's votes while fitting, so on these pairs it is "
                    "scoring its own training data and can exceed the crowd ceiling"
                    if circ else "fit on VLM labels only, so panel pairs are held out"),
        ))
    return out


# ---------------------------------------------------------------- blending


def normalise(scores: dict[str, float], gender_of: dict[str, str],
              method: str = "percentile") -> dict[str, float]:
    """Put one source on a common scale, within gender. Ties share the average rank."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    out: dict[str, float] = {}
    by_gender: dict[str, list[str]] = {}
    for fid in scores:
        by_gender.setdefault(gender_of.get(fid, "?"), []).append(fid)

    for fids in by_gender.values():
        vals = np.array([scores[f] for f in fids], dtype=float)
        if method == "zscore":
            sd = vals.std()
            z = (vals - vals.mean()) / sd if sd > 0 else np.zeros_like(vals)
            for f, v in zip(fids, z):
                out[f] = float(v)
        else:
            # average ranks within ties, so equal scores get an equal percentile
            ranks = rankdata(vals, method="average") - 1.0
            denom = max(len(vals) - 1, 1)
            for f, r in zip(fids, ranks):
                out[f] = float(r / denom * 100.0)
    return out


def blend(sources: Sequence[Source], weights: Sequence[float],
          gender_of: dict[str, str], method: str = "percentile") -> dict[str, float]:
    """Weighted average of normalised sources, over faces present in ALL of them.

    Restricting to the intersection keeps every face's blend built from the same
    ingredients; filling gaps with a per-source mean would quietly score some faces on
    fewer components than others and make the columns incomparable.
    """
    if not sources:
        return {}
    if len(weights) != len(sources):
        raise ValueError("weights and sources must be the same length")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("weights must sum to a positive number")

    common: set[str] | None = None
    for s in sources:
        common = set(s.scores) if common is None else (common & set(s.scores))
    normed = [normalise(s.scores, gender_of, method) for s in sources]
    return {
        fid: float(sum(w * n[fid] for w, n in zip(weights, normed)) / total)
        for fid in (common or set())
    }


# ---------------------------------------------------------------- scoring vs people


def leakfree_keys(sources: Sequence[Source], votes: dict[int, tuple[float, float]],
                  by_index: dict[int, object]) -> list[int]:
    """Panel pairs whose two faces are in the val split of every neural component."""
    val: set[str] | None = None
    for s in sources:
        if s.val_faces is None:
            continue
        val = set(s.val_faces) if val is None else (val & s.val_faces)
    keys = []
    for i in votes:
        m = by_index.get(i)
        if m is None:
            continue
        if val is not None and (m.face_a_id not in val or m.face_b_id not in val):
            continue
        keys.append(i)
    return sorted(keys)


def picks_from_scores(scores: dict[str, float], keys: Iterable[int],
                      by_index: dict[int, object]) -> dict[int, bool]:
    """-> {pair_index: model prefers face A}. Pairs it cannot score are omitted."""
    out = {}
    for i in keys:
        m = by_index[i]
        if m.face_a_id in scores and m.face_b_id in scores:
            out[i] = scores[m.face_a_id] > scores[m.face_b_id]
    return out


def agreement(pick_a: dict[int, bool], votes: dict[int, tuple[float, float]],
              keys: Iterable[int]) -> tuple[float, float]:
    """Share of INDIVIDUAL human votes agreeing with a pick — same metric as
    eval_vs_panel.py and panel_run_delta.py, so numbers stay comparable across scripts."""
    correct = total = 0.0
    for i in keys:
        if i not in pick_a:
            continue
        wa, wb = votes[i]
        correct += wa if pick_a[i] else wb
        total += wa + wb
    return (correct / total if total else float("nan")), total


def crowd_ceiling(votes: dict[int, tuple[float, float]], keys: Iterable[int]) -> float:
    """How often the crowd majority predicts one of its own members, leave-one-out."""
    hits = n = 0.0
    for i in keys:
        wa, wb = votes[i]
        for w, other in ((wa, wb), (wb, wa)):
            if w:
                hits += w if (w - 1) > other else 0.0
                n += w
    return hits / n if n else float("nan")


def band_breakdown(pick_a: dict[int, bool], votes, keys: Sequence[int],
                   by_index: dict[int, object], pct: dict[str, float]) -> list[dict]:
    """Accuracy split by how far apart the reference ranking puts the two faces."""
    rows = []
    for lo, hi in BANDS:
        sel = [i for i in keys
               if by_index[i].face_a_id in pct and by_index[i].face_b_id in pct
               and lo <= abs(pct[by_index[i].face_a_id]
                             - pct[by_index[i].face_b_id]) * 100 < hi]
        if not sel:
            continue
        acc, tot = agreement(pick_a, votes, sel)
        rows.append({"band": f"{lo}-{min(hi, 100)}", "pairs": len(sel),
                     "votes": tot, "accuracy": acc})
    return rows


def paired_bootstrap(pick_a: dict[int, bool], pick_b: dict[int, bool], votes,
                     keys: Sequence[int], n_boot: int = 2000,
                     seed: int = 5) -> dict[str, float]:
    """CI for (A - B) accuracy, resampling PAIRS so both are hit by the same resample.

    Two accuracies measured on the same pairs are strongly correlated; bootstrapping them
    independently would inflate the interval on their difference and hide a real gain.
    Pairs, not votes, are the resampling unit — the ~12 votes on one pair are not
    independent observations of anything.
    """
    shared = [i for i in keys if i in pick_a and i in pick_b]
    if not shared:
        return {"delta": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "pShareAbove": float("nan"), "pairs": 0}

    va = np.array([votes[i][0] for i in shared], dtype=float)
    vb = np.array([votes[i][1] for i in shared], dtype=float)
    tot = va + vb
    ca = np.where([pick_a[i] for i in shared], va, vb)
    cb = np.where([pick_b[i] for i in shared], va, vb)

    rng = np.random.default_rng(seed)
    n = len(shared)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        t = tot[idx].sum()
        deltas[b] = (ca[idx].sum() - cb[idx].sum()) / t if t else 0.0
    return {
        "delta": float(ca.sum() / tot.sum() - cb.sum() / tot.sum()),
        "lo": float(np.percentile(deltas, 2.5)),
        "hi": float(np.percentile(deltas, 97.5)),
        "pShareAbove": float(np.mean(deltas > 0)),
        "pairs": n,
    }


@dataclass
class BlendReport:
    keys: list[int] = field(default_factory=list)
    votes_scored: float = 0.0
    ceiling: float = float("nan")
    rows: list[dict] = field(default_factory=list)   # per predictor
    bands: dict[str, list[dict]] = field(default_factory=dict)
    best_component: str = ""
    versus_best: dict[str, float] = field(default_factory=dict)


def evaluate_blend(sources: Sequence[Source], weights: Sequence[float],
                   gender_of: dict[str, str], votes, by_index, pct: dict[str, float],
                   method: str = "percentile", n_boot: int = 2000) -> BlendReport:
    """Score the blend and every component on the same leak-free pairs."""
    keys = leakfree_keys(sources, votes, by_index)
    rep = BlendReport(keys=keys, ceiling=crowd_ceiling(votes, keys))
    if not keys:
        return rep

    blended = blend(sources, weights, gender_of, method)
    picks = {"BLEND": picks_from_scores(blended, keys, by_index)}
    for s in sources:
        picks[s.name] = picks_from_scores(s.scores, keys, by_index)

    accs: dict[str, float] = {}
    for name, pick in picks.items():
        acc, tot = agreement(pick, votes, keys)
        accs[name] = acc
        src = next((s for s in sources if s.name == name), None)
        rep.rows.append({
            "predictor": "BLEND" if name == "BLEND" else (src.label if src else name),
            "name": name,
            "accuracy": acc,
            "votes": tot,
            "pairs": len(pick),
            "circular": bool(src.circular) if src else False,
            "caveat": (src.caveat if src else ""),
        })
        rep.bands[name] = band_breakdown(pick, votes, keys, by_index, pct)
    rep.votes_scored = agreement(picks["BLEND"], votes, keys)[1]

    # "Better than the blend's best honest ingredient" is the only claim worth making;
    # beating a circular component proves nothing.
    honest = [s.name for s in sources if not s.circular]
    if honest:
        rep.best_component = max(honest, key=lambda n: accs.get(n, float("-inf")))
        rep.versus_best = paired_bootstrap(
            picks["BLEND"], picks[rep.best_component], votes, keys, n_boot=n_boot)
    return rep
