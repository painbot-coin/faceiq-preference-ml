"""Map BT theta percentiles to the pre-registered /10 calibration curve.

Anchor points (scoring-gt-core.md §12): median face scores 5.0, top 10% starts at 7.0,
top 1% at 8.0, top 0.1% at 8.5. Monotone PCHIP interpolation between anchors, linear
tails to the [0, 10] bounds.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator

# (percentile in [0,1], scoreOutOf10) — pre-registered, do not tweak post-hoc
ANCHORS: list[tuple[float, float]] = [
    (0.0, 1.0),
    (0.10, 3.0),
    (0.50, 5.0),
    (0.90, 7.0),
    (0.99, 8.0),
    (0.999, 8.5),
    (1.0, 9.0),
]

# Post-hoc display variant (2026-07-05, research log §5.2): identical below the top
# 0.1%; max anchor raised to 9.5 because the cohort includes top-model faces (the
# sample maximum plausibly represents beyond a 1-in-3,000 tail). Rankings unchanged —
# this only stretches the extreme right tail. v1 ANCHORS remain the pre-registered
# canonical curve.
ANCHORS_CAP95: list[tuple[float, float]] = [
    (0.0, 1.0),
    (0.10, 3.0),
    (0.50, 5.0),
    (0.90, 7.0),
    (0.99, 8.0),
    (0.999, 8.5),
    (1.0, 9.5),
]

_interp = PchipInterpolator(
    [p for p, _ in ANCHORS],
    [s for _, s in ANCHORS],
)


def make_scorer(anchors: list[tuple[float, float]]):
    """Return a percentile -> /10 function for an arbitrary anchor set."""
    interp = PchipInterpolator([p for p, _ in anchors], [s for _, s in anchors])

    def score(percentile: float) -> float:
        return float(np.clip(interp(percentile), 0.0, 10.0))

    return score


def percentile_ranks(theta: dict[str, float]) -> dict[str, float]:
    """Fractional rank in [0, 1] per face.

    Exact theta ties are broken arbitrarily (argsort order); with continuous
    MLE theta they do not occur in practice.
    """
    ids = list(theta)
    values = np.array([theta[f] for f in ids])
    order = values.argsort().argsort().astype(float)
    if len(ids) > 1:
        pct = order / (len(ids) - 1)
    else:
        pct = np.array([0.5])
    return {fid: float(p) for fid, p in zip(ids, pct)}


def score_out_of_10(percentile: float) -> float:
    return float(np.clip(_interp(percentile), 0.0, 10.0))


def calibrate(theta: dict[str, float]) -> dict[str, dict[str, float]]:
    """Return faceId -> {theta, percentile, scoreOutOf10}."""
    pct = percentile_ranks(theta)
    return {
        fid: {
            "theta": theta[fid],
            "percentile": pct[fid],
            "scoreOutOf10": score_out_of_10(pct[fid]),
        }
        for fid in theta
    }
