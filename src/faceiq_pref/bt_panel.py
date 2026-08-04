"""Bradley-Terry with per-source discrimination, for mixing VLM labels and panel votes.

Why this is not just `bt.py` with extra rows appended: the two sources disagree about
*scale*, not only about individual pairs. Gemini answers a pair almost
deterministically, so plain BT fit on VLM labels infers large theta gaps; a panel of
humans splits 60/40 on the same pair, which implies small gaps. Pooling both under one
scale (what `fit_bt` would do) lets the mix of sources per face bend the ranking —
faces the panel happened to cover get compressed, faces it missed do not.

So give each source its own discrimination:

    P(i beats j | source s) = sigmoid(beta_s * (theta_i - theta_j))

with `beta_vlm == 1` fixing the scale, so theta stays comparable to earlier refits and
`beta_human` is estimated. beta_human < 1 is the panel's noisier, flatter view of the
same latent quantity; the ratio is the "temperature" between them. Every human vote is
one observation, so a 7-5 split contributes 7 wins and 5 losses and lands near
theta_i == theta_j on its own — which is exactly the information a close pair carries.

Fit by L-BFGS on the exact log-likelihood with analytic gradients. Regularization
matches `bt.py`: `alpha` virtual half-win and half-loss against a pseudo-opponent at
theta=0, keeping the MLE finite for undefeated faces.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_expit

from .bt import MIN_COMPARISONS, TIE_WEIGHT, _connected_components
from .data import Matchup

ALPHA = 0.01


@dataclass
class Cells:
    """Aggregated pairwise outcomes: wins_ij / wins_ji per (face i, face j, source)."""

    i: list[int] = field(default_factory=list)
    j: list[int] = field(default_factory=list)
    wij: list[float] = field(default_factory=list)
    wji: list[float] = field(default_factory=list)
    human: list[bool] = field(default_factory=list)

    def add(self, i: int, j: int, wij: float, wji: float, human: bool) -> None:
        self.i.append(i)
        self.j.append(j)
        self.wij.append(wij)
        self.wji.append(wji)
        self.human.append(human)

    def __len__(self) -> int:
        return len(self.i)


@dataclass
class JointBtResult:
    gender: str
    theta: dict[str, float]
    beta_human: float
    comparison_counts: dict[str, int]      # distinct opponents faced
    observation_counts: dict[str, int]     # individual votes/labels behind them
    connected: bool
    n_components: int
    faces_below_min: list[str]
    dropped_faces: list[str]
    dropped_matchups: int
    n_vlm_obs: float
    n_human_obs: float
    converged: bool
    n_iter: int


def _pair_counts(rows: list[Matchup]) -> dict[str, int]:
    """Distinct comparisons per face — one per matchup, however many votes it drew.

    The MIN_COMPARISONS gate means "this face has been placed against enough different
    opponents". Twelve panel votes on two opponents is not the same evidence as twelve
    different opponents, so votes must not inflate this count or the gate is gameable.
    """
    counts: dict[str, int] = defaultdict(int)
    for m in rows:
        counts[m.face_a_id] += 1
        counts[m.face_b_id] += 1
    return dict(counts)


def build_cells(
    matchups: list[Matchup],
    panel_votes: dict[int, tuple[float, float]],
    gender: str,
    drop_below_min: bool = True,
):
    """Aggregate one gender's observations.

    `panel_votes` maps a matchup's pair_index to (wins for face A, wins for face B)
    in human votes, ties already split. Any pair present there has its VLM label
    **dropped** — the panel measured that pair directly with ~12x the evidence, and
    keeping both would let the VLM's near-deterministic call fight the humans.

    Faces under MIN_COMPARISONS distinct opponents are removed before fitting, matching
    `bt.fit_bt`, so this refit is comparable to earlier ones rather than scoring faces
    the baseline declined to score.
    """
    rows = [m for m in matchups if m.gender == gender]

    dropped_faces: list[str] = []
    dropped_matchups = 0
    if drop_below_min:
        while True:
            counts = _pair_counts(rows)
            below = {fid for fid, c in counts.items() if c < MIN_COMPARISONS}
            if not below:
                break
            dropped_faces.extend(sorted(below))
            before = len(rows)
            rows = [m for m in rows
                    if m.face_a_id not in below and m.face_b_id not in below]
            dropped_matchups += before - len(rows)

    face_ids = sorted({fid for m in rows for fid in (m.face_a_id, m.face_b_id)})
    index = {fid: k for k, fid in enumerate(face_ids)}

    agg: dict[tuple[int, int, bool], list[float]] = defaultdict(lambda: [0.0, 0.0])
    obs: dict[str, int] = defaultdict(int)
    n_vlm = n_human = 0.0

    for m in rows:
        a, b = index[m.face_a_id], index[m.face_b_id]
        votes = panel_votes.get(m.pair_index)
        human = votes is not None and (votes[0] + votes[1]) > 0
        if human:
            wa, wb = votes
            n_human += wa + wb
        elif m.final_outcome == "A":
            wa, wb = 1.0, 0.0
            n_vlm += 1.0
        elif m.final_outcome == "B":
            wa, wb = 0.0, 1.0
            n_vlm += 1.0
        else:
            wa = wb = TIE_WEIGHT
            n_vlm += 1.0

        lo, hi, w_lo, w_hi = (a, b, wa, wb) if a < b else (b, a, wb, wa)
        cell = agg[(lo, hi, human)]
        cell[0] += w_lo
        cell[1] += w_hi
        obs[m.face_a_id] += int(wa + wb)
        obs[m.face_b_id] += int(wa + wb)

    cells = Cells()
    for (i, j, human), (wij, wji) in agg.items():
        cells.add(i, j, wij, wji, human)
    return (face_ids, cells, _pair_counts(rows), dict(obs), n_vlm, n_human,
            dropped_faces, dropped_matchups)


def fit_joint_bt(
    face_ids: list[str],
    cells: Cells,
    counts: dict[str, int],
    obs: dict[str, int],
    gender: str,
    n_vlm: float,
    n_human: float,
    dropped_faces: list[str] | None = None,
    dropped_matchups: int = 0,
    alpha: float = ALPHA,
) -> JointBtResult:
    n = len(face_ids)
    i = np.asarray(cells.i)
    j = np.asarray(cells.j)
    wij = np.asarray(cells.wij)
    wji = np.asarray(cells.wji)
    is_human = np.asarray(cells.human)

    def neg_ll(x: np.ndarray) -> tuple[float, np.ndarray]:
        theta = x[:n]
        beta_h = np.exp(x[n])
        beta = np.where(is_human, beta_h, 1.0)

        d = theta[i] - theta[j]
        z = beta * d
        ll = float(np.sum(wij * log_expit(z) + wji * log_expit(-z)))

        p = expit(z)
        g = wij * (1.0 - p) - wji * p          # dLL/dz
        gt = np.zeros(n)
        np.add.at(gt, i, beta * g)
        np.subtract.at(gt, j, beta * g)
        gb = float(np.sum(d[is_human] * g[is_human]) * beta_h)   # chain rule for log beta

        # pseudo-opponent at theta=0, one half-win and one half-loss, source scale 1
        ll += float(alpha * np.sum(log_expit(theta) + log_expit(-theta)))
        gt += alpha * (1.0 - 2.0 * expit(theta))

        grad = np.empty(n + 1)
        grad[:n] = -gt
        grad[n] = -gb
        return -ll, grad

    x0 = np.zeros(n + 1)
    x0[n] = np.log(0.2)      # humans start flatter than the VLM; the fit moves it
    res = minimize(neg_ll, x0, jac=True, method="L-BFGS-B",
                   options={"maxiter": 20000, "maxfun": 40000, "ftol": 1e-12, "gtol": 1e-8})

    theta = res.x[:n] - float(np.mean(res.x[:n]))
    beta_h = float(np.exp(res.x[n]))
    n_components = _connected_components(n, [(a, b, 1.0) for a, b in zip(cells.i, cells.j)])

    return JointBtResult(
        gender=gender,
        theta={fid: float(theta[k]) for k, fid in enumerate(face_ids)},
        beta_human=beta_h,
        comparison_counts=counts,
        observation_counts=obs,
        connected=n_components == 1,
        n_components=n_components,
        faces_below_min=[f for f in face_ids if counts.get(f, 0) < MIN_COMPARISONS],
        dropped_faces=dropped_faces or [],
        dropped_matchups=dropped_matchups,
        n_vlm_obs=n_vlm,
        n_human_obs=n_human,
        converged=bool(res.success),
        n_iter=int(res.nit),
    )
