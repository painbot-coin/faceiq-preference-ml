"""Anchor reference panel: local storage + ladder placement (§5.4 Path B prototype).

Anchors are curated faces with committee-assigned product /10 scores, stored under
data/anchors/ (git-ignored) with a manifest.json. Photos are normalized with the
faceiq-labs front crop before storage so anchors and test photos share the exact
training-image geometry.

Placement: the comparator is a scalar scorer, so "user vs anchor" is
sigmoid(s(user) - s(anchor)). One user score, read against every anchor:
- point estimate: interpolate product score between the two anchors whose model
  scores straddle the user's
- confidence band: product scores of anchors in the ambiguous zone (win prob
  25-75%) — wide band = user sits near several rungs at once
- consistency: adjacent anchors whose model-score order contradicts their product
  order are panel defects; a user landing inside such an inverted interval is the
  "beats the 7 but loses to the 6" case.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
ANCHOR_DIR = ROOT / "data" / "anchors"
MANIFEST = ANCHOR_DIR / "manifest.json"

AMBIGUOUS_LO, AMBIGUOUS_HI = 0.25, 0.75


@dataclass(frozen=True)
class Anchor:
    anchor_id: str
    gender: str
    product_score: float
    image_path: str  # relative to ANCHOR_DIR
    label: str = ""


def load_anchors() -> list[Anchor]:
    if not MANIFEST.exists():
        return []
    return [Anchor(**row) for row in json.loads(MANIFEST.read_text())]


def _save_manifest(anchors: list[Anchor]) -> None:
    ANCHOR_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps([asdict(a) for a in anchors], indent=2))


def add_anchor(
    img: Image.Image, gender: str, product_score: float, label: str = ""
) -> Anchor | None:
    """Normalize with the faceiq-labs front crop and store. None if no face found."""
    from .preprocess import normalize_front_photo

    normalized = normalize_front_photo(img)
    if normalized is None:
        return None

    anchor_id = f"{gender}-{product_score:g}-{int(time.time() * 1000)}"
    rel = f"{gender}/{anchor_id}.webp"
    out_path = ANCHOR_DIR / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.save(out_path, "WEBP", quality=85)

    anchor = Anchor(anchor_id, gender, float(product_score), rel, label)
    _save_manifest(load_anchors() + [anchor])
    return anchor


def remove_anchor(anchor_id: str) -> None:
    anchors = load_anchors()
    keep = [a for a in anchors if a.anchor_id != anchor_id]
    for a in anchors:
        if a.anchor_id == anchor_id:
            (ANCHOR_DIR / a.image_path).unlink(missing_ok=True)
    _save_manifest(keep)


def anchor_image_path(anchor: Anchor) -> Path:
    return ANCHOR_DIR / anchor.image_path


# -- panel validation + ladder placement -------------------------------------


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def panel_violations(scored: list[tuple[Anchor, float]]) -> list[str]:
    """Adjacent product-score pairs where the model disagrees with the ordering."""
    by_label = sorted(scored, key=lambda t: t[0].product_score)
    problems = []
    for (a, sa), (b, sb) in zip(by_label, by_label[1:]):
        if sa >= sb and a.product_score < b.product_score:
            problems.append(
                f"model scores the {a.product_score:g} anchor ({sa:.2f}) at or above "
                f"the {b.product_score:g} anchor ({sb:.2f})"
            )
    return problems


def place_on_ladder(user_score: float, scored: list[tuple[Anchor, float]]) -> dict:
    """Place a user's model score on the anchor ladder.

    scored: [(anchor, model_score)] for one gender. Returns per-anchor rows,
    implied product score, confidence band, and consistency flags.
    """
    rows = [
        {
            "anchor": a,
            "model_score": s,
            "p_win": _sigmoid(user_score - s),
            "beats": user_score > s,
        }
        for a, s in sorted(scored, key=lambda t: t[0].product_score)
    ]

    labels_beaten = [r["anchor"].product_score for r in rows if r["beats"]]
    labels_lost = [r["anchor"].product_score for r in rows if not r["beats"]]

    # point estimate: interpolate in model-score space between straddling anchors
    by_model = sorted(rows, key=lambda r: r["model_score"])
    below = [r for r in by_model if r["model_score"] <= user_score]
    above = [r for r in by_model if r["model_score"] > user_score]
    straddle: tuple[float, float] | None = None
    if not below:
        implied = min(r["anchor"].product_score for r in rows)
        position = "below the whole panel"
    elif not above:
        implied = max(r["anchor"].product_score for r in rows)
        position = "above the whole panel"
    else:
        lo, hi = below[-1], above[0]
        span = hi["model_score"] - lo["model_score"]
        t = (user_score - lo["model_score"]) / span if span > 0 else 0.5
        implied = lo["anchor"].product_score + t * (
            hi["anchor"].product_score - lo["anchor"].product_score
        )
        position = "within the panel"
        straddle = (
            min(lo["anchor"].product_score, hi["anchor"].product_score),
            max(lo["anchor"].product_score, hi["anchor"].product_score),
        )

    # confidence band: at minimum the straddled interval (placement can't be more
    # precise than the rung spacing), widened by any anchors the model is unsure
    # about (ambiguous win prob)
    ambiguous = [
        r["anchor"].product_score
        for r in rows
        if AMBIGUOUS_LO < r["p_win"] < AMBIGUOUS_HI
    ]
    candidates = list(ambiguous)
    if straddle is not None:
        candidates.extend(straddle)
    else:
        candidates.append(implied)
    band = (min(candidates), max(candidates))

    # per-user inconsistency: beats an anchor labeled higher than one it loses to
    inconsistent = bool(
        labels_beaten and labels_lost and max(labels_beaten) > min(labels_lost)
    )

    return {
        "rows": rows,
        "implied_score": implied,
        "band": band,
        "position": position,
        "inconsistent": inconsistent,
        "n_beaten": len(labels_beaten),
        "n_lost": len(labels_lost),
    }
