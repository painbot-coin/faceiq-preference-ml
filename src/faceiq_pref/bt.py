"""Bradley-Terry MLE refit with connectivity and stability diagnostics.

Fits male and female graphs separately (the pair queue is same-gender only, so the
combined graph is disconnected across genders by construction). Ties count as a
half-win for each side.

Gates (research log §5.1): connected graph, no face < MIN_COMPARISONS resolved games,
80% subsample Spearman rho > 0.95.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import spearmanr

from .data import Matchup

MIN_COMPARISONS = 15
STABILITY_RHO_GATE = 0.95
TIE_WEIGHT = 0.5


@dataclass
class BtResult:
    gender: str
    theta: dict[str, float]  # faceId -> BT strength (log scale, mean 0)
    comparison_counts: dict[str, int]
    connected: bool
    n_components: int
    faces_below_min: list[str] = field(default_factory=list)
    dropped_faces: list[str] = field(default_factory=list)  # under-connected, removed pre-fit
    dropped_matchups: int = 0
    stability_rho: float | None = None


def _pairs_to_wins(
    matchups: list[Matchup],
) -> tuple[list[str], list[tuple[int, int, float]]]:
    """Return (face_ids, weighted win list of (winner_idx, loser_idx, weight))."""
    face_ids = sorted({fid for m in matchups for fid in (m.face_a_id, m.face_b_id)})
    index = {fid: i for i, fid in enumerate(face_ids)}
    wins: list[tuple[int, int, float]] = []
    for m in matchups:
        a, b = index[m.face_a_id], index[m.face_b_id]
        if m.final_outcome == "A":
            wins.append((a, b, 1.0))
        elif m.final_outcome == "B":
            wins.append((b, a, 1.0))
        else:  # tie -> half-win each way
            wins.append((a, b, TIE_WEIGHT))
            wins.append((b, a, TIE_WEIGHT))
    return face_ids, wins


def _connected_components(n: int, wins: list[tuple[int, int, float]]) -> int:
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for w, l, _ in wins:
        rw, rl = find(w), find(l)
        if rw != rl:
            parent[rw] = rl
    return len({find(i) for i in range(n)})


def _fit_theta(n: int, wins: list[tuple[int, int, float]], alpha: float = 0.01) -> np.ndarray:
    """Regularized MM algorithm for weighted Bradley-Terry (Hunter 2004).

    alpha adds a virtual half-win/half-loss vs a pseudo-average opponent, keeping the
    MLE finite even for undefeated faces.
    """
    win_weight = np.full(n, alpha)
    # per-pair aggregated weights: (i, j) -> weight of i beating j
    pair_weight: dict[tuple[int, int], float] = {}
    for w, l, wt in wins:
        pair_weight[(w, l)] = pair_weight.get((w, l), 0.0) + wt
        win_weight[w] += wt

    # games[i][j] = total games between i and j (both directions)
    opponents: dict[int, dict[int, float]] = {}
    for (w, l), wt in pair_weight.items():
        opponents.setdefault(w, {})[l] = opponents.setdefault(w, {}).get(l, 0.0) + wt
        opponents.setdefault(l, {})[w] = opponents.setdefault(l, {}).get(w, 0.0) + wt

    p = np.ones(n)
    for _ in range(1000):
        p_new = np.empty(n)
        for i in range(n):
            denom = 2 * alpha / (p[i] + 1.0)  # regularization vs pseudo-opponent (p=1)
            for j, games in opponents.get(i, {}).items():
                denom += games / (p[i] + p[j])
            p_new[i] = win_weight[i] / denom if denom > 0 else p[i]
        p_new /= np.exp(np.mean(np.log(np.clip(p_new, 1e-12, None))))  # geometric mean 1
        if np.max(np.abs(np.log(np.clip(p_new, 1e-12, None)) - np.log(np.clip(p, 1e-12, None)))) < 1e-6:
            p = p_new
            break
        p = p_new
    return np.log(p)


def _comparison_counts(rows: list[Matchup]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for m in rows:
        counts[m.face_a_id] = counts.get(m.face_a_id, 0) + 1
        counts[m.face_b_id] = counts.get(m.face_b_id, 0) + 1
    return counts


def fit_bt(matchups: list[Matchup], gender: str, drop_below_min: bool = True) -> BtResult:
    """Fit BT for one gender.

    When drop_below_min is set (default), faces with fewer than MIN_COMPARISONS
    resolved games are removed before fitting (iteratively, since dropping their
    edges can push neighbours below the floor). This handles intentionally
    stripped nodes — e.g. the synthetic poison face whose matchups were excluded
    from export — per research log §4.3. Dropped faces are reported, not scored.
    """
    rows = [m for m in matchups if m.gender == gender]
    if not rows:
        raise ValueError(f"no matchups for gender {gender!r}")

    dropped_faces: list[str] = []
    dropped_matchups = 0
    if drop_below_min:
        while True:
            counts = _comparison_counts(rows)
            below = {fid for fid, c in counts.items() if c < MIN_COMPARISONS}
            if not below:
                break
            dropped_faces.extend(sorted(below))
            before = len(rows)
            rows = [m for m in rows if m.face_a_id not in below and m.face_b_id not in below]
            dropped_matchups += before - len(rows)

    face_ids, wins = _pairs_to_wins(rows)
    n = len(face_ids)
    counts = _comparison_counts(rows)

    n_components = _connected_components(n, wins)
    theta = _fit_theta(n, wins)

    return BtResult(
        gender=gender,
        theta={fid: float(theta[i]) for i, fid in enumerate(face_ids)},
        comparison_counts=counts,
        connected=n_components == 1,
        n_components=n_components,
        faces_below_min=[fid for fid, c in counts.items() if c < MIN_COMPARISONS],
        dropped_faces=dropped_faces,
        dropped_matchups=dropped_matchups,
    )


def stability_check(
    matchups: list[Matchup], gender: str, subsample: float = 0.8, seed: int = 7
) -> float:
    """Spearman rho between rankings from two independent 80% pair subsamples."""
    rows = [m for m in matchups if m.gender == gender]
    rng = random.Random(seed)

    def subsample_theta(trial_seed: int) -> dict[str, float]:
        trial_rng = random.Random(trial_seed)
        sample = [m for m in rows if trial_rng.random() < subsample]
        face_ids, wins = _pairs_to_wins(sample)
        theta = _fit_theta(len(face_ids), wins)
        return {fid: float(theta[i]) for i, fid in enumerate(face_ids)}

    t1 = subsample_theta(rng.randint(0, 2**31))
    t2 = subsample_theta(rng.randint(0, 2**31))
    common = sorted(set(t1) & set(t2))
    rho, _ = spearmanr([t1[f] for f in common], [t2[f] for f in common])
    return float(rho)
