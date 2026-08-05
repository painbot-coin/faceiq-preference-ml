"""Place a new face on the cohort scale, with a band and a tier — the production read path.

This is the shipping form of what `scripts/reference_set_inference.py` measured. Given a comparator
score for an unseen face and a reference set of cohort faces with known Bradley-Terry theta, it
returns theta, a standard error, a percentile, a /10, a band and a tier.

Why it is a fit and not a lookup. The comparator is *trained* to answer "is A better than B", so the
only thing we are entitled to read out of it is the set of comparisons it makes against faces whose
position we know. Fit theta_new to explain that record:

    P(new beats ref_j) = sigmoid(theta_new - theta_j)

The log-likelihood is strictly concave in one variable, so Newton converges in a few steps, and the
curvature at the optimum gives a standard error free:

    se(theta_new) = 1 / sqrt( sum_j p_j (1 - p_j) )

Adding one node to a BT graph while holding the existing edges fixed *is* this conditional MLE, so
this is the same estimator as "refit Bradley-Terry with the user included" at 1/1000th of the cost
(research log §5.8).

**Only the sign of each comparison is used.** The comparator's score *magnitude* is on an arbitrary
scale we have no reason to trust; its ordering is what was trained and validated. Measured cost of
discarding the magnitude: none (Spearman 0.857 either way).

Two behaviours that are not edge cases in practice, both measured in log §5.8:

* A face that beats **every** reference has no finite MLE. ~16% of top-5% female faces do this. Such
  a face is capped at the top reference plus `UNBOUNDED_MARGIN` and flagged via `bounded=False`;
  never let the optimiser run away instead.
* Placement is roughly twice as accurate mid-scale (0.41 /10) as at either end (0.76-0.92), so the
  band uses the per-face `se` rather than a constant width.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .calibrate import make_scorer

# Fitted on run 4's 2,500 uniform pairs — the only pairs no ranking had been fitted on when they
# were scored. P(higher /10 score wins) = sigmoid(gap / T). See scripts/band_calibration.py.
TEMPERATURE_TEN = 1.920

# Product default: the band edges are the distance at which two thirds of people agree.
DEFAULT_AGREEMENT = 2.0 / 3.0

# How far above the top reference to place a face that beat all of them. Arbitrary by necessity —
# the likelihood is genuinely unbounded there — so it is flagged rather than silently returned.
UNBOUNDED_MARGIN = 1.0

TIERS = 7

# Anchor ladder with the top of the cohort reaching 10. Keeps the two pre-registered anchors people
# already know (median 5.0, top decile starts at 7.0) and stretches only the top percentile, which
# is where the original ladder's ceiling of 9.0 was the binding constraint. `ANCHORS` in
# calibrate.py remains the pre-registered canonical curve; this is a display ladder, and swapping
# between them changes no ordering whatsoever.
ANCHORS_TOP10: list[tuple[float, float]] = [
    (0.0, 1.0),
    (0.10, 3.0),
    (0.50, 5.0),
    (0.90, 7.0),
    (0.99, 8.7),
    (0.999, 9.4),
    (1.0, 10.0),
]


@dataclass
class Placement:
    theta: float
    se: float
    bounded: bool
    percentile: float
    score_ten: float
    band_low: float
    band_high: float
    tier: int
    tier_low: float
    tier_high: float
    agreement: float
    wins: int
    references: int

    @property
    def band_half(self) -> float:
        return (self.band_high - self.band_low) / 2


def band_half_width(agreement: float = DEFAULT_AGREEMENT, t: float = TEMPERATURE_TEN) -> float:
    """Half the /10 gap at which `agreement` of people agree on the ordering.

    Halved so that two faces whose bands stop overlapping differ by the *full* resolvable gap —
    the same convention as a confidence interval, except the width comes from how much people
    disagree rather than from our sample size, which is why it does not shrink with more data.
    """
    return t * math.log(agreement / (1 - agreement)) / 2


def tier_edges(n: int = TIERS) -> list[float]:
    """Tier boundaries on the /10 scale, evenly spaced from 1 to 10.

    `9 / 7 = 1.29` per tier against a measured resolution limit of `T * ln(2) = 1.33`, so
    within-tier differences are genuinely below what people can agree on. That two independent
    measurements — rank resolution (~210 places of 1,430) and the calibration curve — both land on
    ~7 tiers is the justification for the number.
    """
    return [1.0 + i * (9.0 / n) for i in range(n + 1)]


def tier_of(score_ten: float, n: int = TIERS) -> tuple[int, float, float]:
    edges = tier_edges(n)
    for i in range(n):
        if score_ten < edges[i + 1] or i == n - 1:
            return i + 1, edges[i], edges[i + 1]
    return n, edges[-2], edges[-1]


def agreement_vs_score(score_ten: float, other_ten: float, t: float = TEMPERATURE_TEN) -> float:
    """Share of people who would place `score_ten` above `other_ten`."""
    return 1.0 / (1.0 + math.exp(-(score_ten - other_ten) / t))


def agreement_vs_tier_below(score_ten: float, n: int = TIERS) -> tuple[int, float] | None:
    """-> (tier below, share of people who would rank this face above a typical face in it).

    The user-facing sentence, and it deliberately compares against a **tier**, not against a bare
    number. Saying "67% of people would place you above someone scoring 3.8" reads as though some
    raters think this face *is* a 3.8, which is not what the maths says and is a bad thing to imply.
    Comparing against the typical member of the tier below carries the same information without the
    misreading, and it is the comparison the tier system exists to make.

    Tier width (1.29 /10) is almost exactly the 2-in-3 resolvable gap (1.33), so a face in the
    middle of its tier lands near two thirds against the tier below by construction — but this
    returns the *exact* figure for this face's score, which is lower near the bottom of a tier and
    higher near the top. That variation is real and should be shown rather than rounded away.
    """
    tier, _, _ = tier_of(score_ten, n)
    if tier <= 1:
        return None
    edges = tier_edges(n)
    below_centre = (edges[tier - 2] + edges[tier - 1]) / 2
    return tier - 1, agreement_vs_score(score_ten, below_centre, t=TEMPERATURE_TEN)


def fit_theta(
    beats: list[bool], ref_thetas: list[float], max_iter: int = 60
) -> tuple[float, float, bool]:
    """Conditional MLE for a new face's theta. -> (theta, se, bounded)."""
    if not ref_thetas:
        raise ValueError("reference set is empty")
    if all(beats):
        return max(ref_thetas) + UNBOUNDED_MARGIN, float("inf"), False
    if not any(beats):
        return min(ref_thetas) - UNBOUNDED_MARGIN, float("inf"), False

    theta = sum(ref_thetas) / len(ref_thetas)
    info = 0.0
    for _ in range(max_iter):
        grad = info = 0.0
        for won, tj in zip(beats, ref_thetas):
            p = 1.0 / (1.0 + math.exp(-(theta - tj)))
            grad += (1.0 if won else 0.0) - p
            info += p * (1 - p)
        if info < 1e-12:
            break
        step = grad / info
        theta += max(-2.0, min(2.0, step))
        if abs(step) < 1e-9:
            break
    return theta, (1.0 / math.sqrt(info) if info > 0 else float("inf")), True


def place(
    score: float,
    reference_scores: list[float],
    reference_thetas: list[float],
    population_thetas: list[float] | None = None,
    agreement: float = DEFAULT_AGREEMENT,
    anchors: list[tuple[float, float]] | None = None,
) -> Placement:
    """Score a new face against the reference set and return its placement.

    `reference_scores` are the comparator's scores for the same faces as `reference_thetas`, from
    the same checkpoint — mixing checkpoints silently invalidates every comparison.

    **`population_thetas` is not optional in spirit.** The reference set and the population play two
    different roles, and conflating them biases every score. References *estimate theta*, and for
    that job they should be spread evenly over the scale — which deliberately over-samples the thin
    extremes. The percentile then has to be read off the **full cohort's** theta distribution,
    because "fraction of references below you" is only a population percentile if the references are
    population-representative, and a good reference set is not. Getting this wrong pulls a true 90th
    percentile face down to about the 70th; it defaults to the reference thetas only so the function
    still works with a representative draw.
    """
    if len(reference_scores) != len(reference_thetas):
        raise ValueError("reference scores and thetas must be the same length")
    to_ten = make_scorer(anchors or ANCHORS_TOP10)
    population = sorted(population_thetas if population_thetas else reference_thetas)

    beats = [score > rs for rs in reference_scores]
    theta, se, bounded = fit_theta(beats, reference_thetas)

    def pct_of(t: float) -> float:
        lo, hi = 0, len(population)
        while lo < hi:
            mid = (lo + hi) // 2
            if population[mid] < t:
                lo = mid + 1
            else:
                hi = mid
        return lo / len(population)

    pct = pct_of(theta)
    ten = to_ten(pct)

    # Two independent widths, combined in quadrature. The disagreement term dominates by ~5x at a
    # 200-face reference set, which is why reference-set size is not the lever it looks like.
    disagree_half = band_half_width(agreement)
    est_half = 0.0
    if math.isfinite(se):
        # Convert se(theta) to /10 through the population distribution and the anchor curve, so the
        # conversion follows the scale's local slope instead of assuming it is linear.
        est_half = max(0.0, (to_ten(pct_of(theta + se)) - to_ten(pct_of(theta - se))) / 2)
    half = math.hypot(est_half, disagree_half)

    tier, t_lo, t_hi = tier_of(ten)
    return Placement(
        theta=theta, se=se, bounded=bounded, percentile=pct, score_ten=ten,
        band_low=max(1.0, ten - half), band_high=min(10.0, ten + half),
        tier=tier, tier_low=t_lo, tier_high=t_hi,
        agreement=agreement, wins=sum(beats), references=len(reference_thetas),
    )


def stratified_reference_ids(
    percentiles: dict[str, float],
    n: int = 200,
    seed: int = 20260804,
    anchors: list[tuple[float, float]] | None = None,
) -> list[str]:
    """Pick ~`n` reference faces spread evenly across the *score* scale, not the population.

    Spread is the only selection rule that measurably helps (0.03-0.06 /10 over a random draw);
    selecting for tight theta intervals or for panel coverage does **not** help, and panel coverage
    is marginally worse because panel-covered faces cluster in dense parts of the scale (log §5.8).

    Stratifying by **tier** rather than by equal percentile bands is load-bearing at the extremes,
    and the reason is worth stating. The /10 scale is stretched at the top: tier 7 spans the top 1%
    of the cohort, so an even split over percentile puts almost no references up there, and a user
    in the top 1% beats essentially all of them — which yields a rank of 1.0 and no finite MLE. That
    is the mechanism behind the ~16% unbounded rate measured in log §5.8. Sampling per tier puts
    references where the *score* needs resolving.

    The top tier is genuinely thin in the cohort (~14 female faces), so tiers are filled up to
    availability and the shortfall is redistributed over the tiers that have faces to spare. The
    returned count is therefore approximately, not exactly, `n`.
    """
    import random

    rng = random.Random(seed)
    to_ten = make_scorer(anchors or ANCHORS_TOP10)
    by_tier: dict[int, list[str]] = {}
    for fid, p in percentiles.items():
        by_tier.setdefault(tier_of(to_ten(p))[0], []).append(fid)
    for cell in by_tier.values():
        rng.shuffle(cell)

    quota = max(1, n // max(1, len(by_tier)))
    out: list[str] = []
    spare: list[str] = []
    for cell in by_tier.values():
        out += cell[:quota]
        spare += cell[quota:]
    rng.shuffle(spare)
    return out + spare[: max(0, n - len(out))]
