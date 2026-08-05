"""Join off-cohort judgements to placements — the pair table the report and the dashboard share.

`scripts/validate_placement.py report` prints the verdict; the Inference tab's review view shows
the same pairs as photos so the choices can be eyeballed. Both need the identical join, and a
duplicate implementation would drift — the /10 gap that decides "model agreed" here is the same gap
that decides which side of the band a pair falls on there.

Three classes of judgement cannot be scored and must be visible rather than folded into the
accuracy, because each of them moves it in a flattering or unflattering direction for a reason that
has nothing to do with the model:

* **skipped** — "can't tell" is not a wrong answer. Counting it as one deflates accuracy by roughly
  one pair's worth per skip, which is small but is error in the wrong direction on a test whose
  whole purpose is to be honest.
* **tied** — the reference-set MLE reads only how many references a face *beat*, so two photos
  falling between the same pair of adjacent references get the identical /10. There is no prediction
  to score. This is quantisation in the production path, not a rounding artefact, and it is worth
  reporting on its own: those two users would see the same number.
* **repeat** — a re-asked pair measures the labeller, not the model. It is the ceiling, so it must
  leave the numerator entirely.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .placement import DEFAULT_AGREEMENT, TEMPERATURE_TEN, band_half_width


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. Normal approximation is wrong where it matters here — a ceiling of
    34/35 has an asymmetric interval that the textbook +-1.96*sqrt(p(1-p)/n) pushes above 1.0."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = hits / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def pair_table(judgments: list[dict], ten: dict[str, float]) -> pd.DataFrame:
    """One row per judgement, in queue order, with what the placed scores say about it.

    `ten` maps photo filename -> placed /10, and may be **empty**: the judgements stand on their own
    (they are the human ordering) and the dashboard renders them for review before `score` has ever
    run. Such rows come back with `placed=False` and no opinion attached, rather than vanishing —
    a silently shorter table is how a count stops adding up to the queue length.
    """
    first: dict[frozenset, str | None] = {}
    rows = []
    for i, j in enumerate(judgments):
        a, b, winner = j["a"], j["b"], j.get("winner")
        key = frozenset((a, b))
        repeat = key in first
        earlier = first.get(key)
        if not repeat:
            first[key] = winner
        placed = a in ten and b in ten
        sa, sb = (ten[a], ten[b]) if placed else (float("nan"), float("nan"))
        tied = bool(placed and sa == sb)
        model_pick = (a if sa > sb else b) if (placed and not tied) else None
        rows.append({
            "index": j.get("index", i),
            "a": a,
            "b": b,
            "winner": winner,
            "placed": placed,
            "scoreA": sa,
            "scoreB": sb,
            "gap": abs(sa - sb) if placed else float("nan"),
            "modelPick": model_pick,
            "repeat": repeat,
            "skipped": winner is None,
            "tied": tied,
            # None, not False, when there is nothing to agree about — so `.mean()` cannot quietly
            # count an unscoreable pair as a miss.
            "agree": None if (model_pick is None or winner is None) else (model_pick == winner),
            "selfAgree": (None if (not repeat or winner is None or earlier is None)
                          else earlier == winner),
            "firstWinner": earlier,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["scored"] = df["placed"] & ~df["repeat"] & ~df["skipped"] & ~df["tied"]
    return df


def scorable(t: pd.DataFrame) -> pd.DataFrame:
    """Fresh pairs the placement actually makes a prediction on — the accuracy denominator."""
    return t[t["scored"]] if not t.empty else t


def gap_curve(t: pd.DataFrame, bins: int = 5) -> pd.DataFrame:
    """Accuracy by how far apart the placement put the two faces.

    The single most informative table in the study: the band's entire claim is that this rises with
    the gap. A flat curve would mean the score carries no information about how confident to be,
    which is a different (and worse) failure than a low average accuracy.
    """
    s = scorable(t)
    if len(s) < bins * 2:
        return pd.DataFrame()
    edges = np.unique(np.quantile(s["gap"], np.linspace(0, 1, bins + 1)))
    out = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        last = hi == edges[-1]
        m = (s["gap"] >= lo) & ((s["gap"] <= hi) if last else (s["gap"] < hi))
        sub = s[m]
        if sub.empty:
            continue
        hits = int(sub["agree"].sum())
        out.append({
            "gapLow": float(lo), "gapHigh": float(hi), "pairs": len(sub),
            "accuracy": hits / len(sub),
            "predicted": float(np.mean([1 / (1 + math.exp(-g / TEMPERATURE_TEN))
                                        for g in sub["gap"]])),
        })
    return pd.DataFrame(out)


def summarise(t: pd.DataFrame, agreement: float = DEFAULT_AGREEMENT) -> dict:
    """Headline metrics. Every accuracy carries its denominator, per the log's §5.8 rule."""
    out: dict = {"judgments": len(t)}
    if t.empty:
        return out

    reps = t[t["repeat"] & t["selfAgree"].notna()]
    ceiling = float(reps["selfAgree"].mean()) if len(reps) else None
    out["yourRepeatPairs"] = len(reps)
    out["yourSelfAgreement"] = ceiling
    if len(reps):
        lo, hi = wilson(int(reps["selfAgree"].sum()), len(reps))
        out["yourSelfAgreementCi95"] = [lo, hi]

    s = scorable(t)
    fresh = t[~t["repeat"]]
    out["freshPairs"] = len(fresh)
    out["skipped"] = int(fresh["skipped"].sum())
    out["tiedOnPlacedScore"] = int(fresh["tied"].sum())
    out["unplacedPairs"] = int((~fresh["placed"]).sum())
    out["pairsUsed"] = len(s)
    if not len(s):
        return out

    hits = int(s["agree"].sum())
    acc = hits / len(s)
    out["orderingAccuracy"] = acc
    lo, hi = wilson(hits, len(s))
    out["orderingAccuracyCi95"] = [lo, hi]
    if ceiling:
        out["shareOfCeiling"] = acc / ceiling

    resolvable = 2 * band_half_width(agreement)
    out["resolvableGapPoints"] = resolvable
    for label, m in (("withinBand", s["gap"] < resolvable),
                     ("beyondBand", s["gap"] >= resolvable)):
        sub = s[m]
        out[f"accuracy_{label}"] = float(sub["agree"].mean()) if len(sub) else None
        out[f"pairs_{label}"] = len(sub)

    out["predictedAgreement"] = float(np.mean(
        [1 / (1 + math.exp(-g / TEMPERATURE_TEN)) for g in s["gap"]]))
    out["observedAgreement"] = acc
    out["medianGapPoints"] = float(s["gap"].median())
    return out
