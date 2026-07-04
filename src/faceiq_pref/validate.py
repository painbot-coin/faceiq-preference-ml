"""Validation of BT rankings against Labs overall_score (sanity, not a training target)."""

from __future__ import annotations

from scipy.stats import spearmanr

from .data import Face


def spearman_vs_labs(theta: dict[str, float], faces: dict[str, Face]) -> tuple[float, int]:
    """Spearman rho between BT theta and Labs overall_score.

    Returns (rho, n_faces_with_labs_score). Moderate positive correlation is expected —
    perfect agreement would mean BT just reproduced the legacy formula.
    """
    pairs = [
        (theta[fid], faces[fid].labs_overall_score)
        for fid in theta
        if fid in faces and faces[fid].labs_overall_score is not None
    ]
    if len(pairs) < 3:
        return float("nan"), len(pairs)
    rho, _ = spearmanr([t for t, _ in pairs], [s for _, s in pairs])
    return float(rho), len(pairs)
