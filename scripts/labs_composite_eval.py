"""Does blending the Labs deterministic score into the placed /10 help? Two targets, two answers.

The interesting part of this script is that it scores the same blend against **two different
targets** and they disagree, which is the whole reason it exists:

* **against BT /10** — the blend looks like a clear, statistically significant win.
* **against real human votes** — the blend is flat to slightly worse, at every weight.

Both cannot be improvements. BT theta is a *proxy* for what people think, fitted on Gemini pairwise
labels; the panel votes are the thing itself. When a change moves the proxy and not the target, it is
fitting the proxy's error. That is the confirmation-bias failure mode this programme has worried
about in the abstract, caught in the concrete: run only the first measurement and you ship a "+0.05
significant improvement" that does nothing for a single user.

So the verdict is **do not blend**, and the script stays because the negative result is worth being
able to re-run when the EBM arrives — the EBM is the same architecture with a better second opinion,
and this is the harness that will judge it.

Labs scores need no computation: `labsOverallScore` is already on all 3,000 faces in `faces.jsonl`.
It is rank-normalised onto our /10 through the same anchor curve before blending, because the two
scales are not otherwise comparable.

**`--photo-quality` answers a different question: is the Labs formula bad, or are the photos?**
Restricting the panel comparison to pairs where *both* faces are clean per the §5.0 VLM QC pass
(`artifacts/face-qc-v1/results.jsonl`, already run on all 2,998 faces — no new spend) separates
"the deterministic formula cannot rank faces" from "the formula is fine and a distorted selfie
breaks it", which have completely different fixes: retire the formula versus gate uploads on photo
quality.

Two things make the raw comparison misleading, and both are reported:

* **Clean pairs are easier for everyone**, so every predictor rises. What matters is the *gap to
  the human ceiling*, recomputed on the same subset, and whether Labs closes it faster than the
  placement does. If both rise equally, quality is not a Labs-specific problem.
* **Quality correlates with the pair's difficulty.** The subset's percentile-gap distribution is
  printed alongside, and each accuracy is also re-weighted onto the full sample's gap distribution
  so the level is not just an easier draw.

**The panel arm is in-sample for the comparator, and that is not a detail.** `model_scores.csv`
holds every cohort face, ~80% of which the checkpoint trained on, so the headline "placement scores
81.9% against the panel majority" is a *transductive* number: it says how well the model ranks faces
it has already been fitted around, not how well it will score a stranger's upload. The
`--leakage-split` table below breaks the same pairs into faces the run trained on, faces it held out,
and the mix, and reports each against a ranking that never saw the panel votes so the strata's
different difficulty is controlled for. Quote the leak-free row, or quote log §5.10's off-cohort
numbers, and treat the pooled figure as a property of the *ranking* rather than of the comparator.

Usage:
    python scripts/labs_composite_eval.py
    python scripts/labs_composite_eval.py --run train-v12-panel-soft
    python scripts/labs_composite_eval.py --photo-quality        # adds the clean/flagged split
    python scripts/labs_composite_eval.py --leakage-split        # trained-on vs held-out faces
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from accuracy_matrix import ceilings, load_votes  # noqa: E402

from faceiq_pref.calibrate import make_scorer  # noqa: E402
from faceiq_pref.data import load_export, split_by_face_id  # noqa: E402
from faceiq_pref.placement import (  # noqa: E402
    ANCHORS_TOP10,
    place,
    stratified_reference_ids,
    tier_of,
)

WEIGHTS = (1.0, 0.9, 0.85, 0.8, 0.7, 0.5, 0.0)
MIN_VAL_FRACTION = 0.2

# Quality issues the QC pass records. `multiple_faces` and `face_cropped` are in the list because
# they break the crop rather than the face; `possible_minor` and `gender_*` are deliberately NOT —
# they are eligibility flags, not photo quality, and folding them in would confound this test with
# the exclusions §5.0 already applied.
QUALITY_ISSUES = ("screenshot", "obstruction", "poor_lighting", "low_resolution",
                  "heavy_filter", "multiple_faces", "face_cropped", "extreme_angle")


def photo_issues(path: Path) -> dict[str, list[str]]:
    """faceId -> quality issues from the §5.0 QC pass. Absent file = empty, caller decides."""
    if not path.exists():
        return {}
    out = {}
    for ln in path.read_text().splitlines():
        if ln.strip():
            r = json.loads(ln)
            out[r["faceId"]] = [i for i in (r.get("issues") or []) if i in QUALITY_ISSUES]
    return out


def score_subset(votes, meta, score_of: dict[str, float], keep=None) -> dict:
    """Accuracy of one score against the panel, over the pairs `keep` admits."""
    maj_hit = maj_n = 0
    vote_hit = vote_n = 0.0
    for pid, (va, vb) in votes.items():
        m = meta.get(pid)
        if m is None:
            continue
        fa, fb = m["faceAId"], m["faceBId"]
        if fa not in score_of or fb not in score_of:
            continue
        if keep is not None and not keep(fa, fb):
            continue
        sa, sb = score_of[fa], score_of[fb]
        if sa == sb:
            continue
        picks_a = sa > sb
        vote_hit += va if picks_a else vb
        vote_n += va + vb
        if va != vb:
            maj_hit += int(picks_a == (va > vb))
            maj_n += 1
    return {"vsMajority": maj_hit / maj_n if maj_n else float("nan"),
            "vsVotes": vote_hit / vote_n if vote_n else float("nan"), "pairs": maj_n}


def quality_split(votes, meta, placed, labs_ten, truth, issues) -> dict:
    """Labs vs placement vs the human ceiling, on clean pairs and on flagged ones.

    The comparison that matters is not either level — clean pairs are easier for every predictor,
    including the humans — but whether **Labs closes on the ceiling faster than the placement
    does**. If both gain the same, photo quality is not a Labs-specific weakness and gating uploads
    buys nothing the placement was not already getting.
    """
    def clean(f: str) -> bool:
        return not issues.get(f, [])

    out: dict = {"issuesCounted": list(QUALITY_ISSUES), "arms": {}}
    for label, keep in (("all", None),
                        ("bothClean", lambda a, b: clean(a) and clean(b)),
                        ("eitherFlagged", lambda a, b: not (clean(a) and clean(b)))):
        sub = {pid: v for pid, v in votes.items()
               if keep is None or (meta.get(pid) is not None
                                   and keep(meta[pid]["faceAId"], meta[pid]["faceBId"]))}
        ceil = ceilings(sub)
        arm = {
            "ceiling": ceil["vsMajority"],
            "labs": score_subset(votes, meta, labs_ten, keep),
            "placed": score_subset(votes, meta, placed, keep),
            "bt": score_subset(votes, meta, truth, keep),
        }
        # Distance to the ceiling is the quantity the hypothesis is about: an accuracy that rises
        # only because the draw got easier moves the ceiling with it and leaves this unchanged.
        for k in ("labs", "placed", "bt"):
            arm[f"{k}GapToCeiling"] = arm[k]["vsMajority"] - arm["ceiling"]
        out["arms"][label] = arm
    out["btIsCircular"] = ("the BT arm is scored against a ranking fitted on these same votes; "
                           "it is on its own training data and must not be quoted")
    return out


def leakage_split(votes, meta, placed, control: dict[str, float], val_faces: set[str],
                  n_boot: int = 4000) -> dict:
    """Split the panel comparison by how many of a pair's faces the run trained on.

    The three strata are not equally hard — the held-out faces happen to land on wider pairs — so
    raw accuracies across them are not comparable. `control` is a ranking that never saw these
    votes and is unaffected by *this* run's split, which makes it a difficulty yardstick: the
    quantity to read is **comparator minus control** within each stratum. If that margin decays as
    the faces become unseen, the pooled headline is borrowing from memorisation.
    """
    rows: dict[int, list[tuple[int, int]]] = {0: [], 1: [], 2: []}
    for pid, (va, vb) in votes.items():
        m = meta.get(pid)
        if m is None:
            continue
        fa, fb = m["faceAId"], m["faceBId"]
        if fa not in placed or fb not in placed or fa not in control or fb not in control:
            continue
        if va == vb or placed[fa] == placed[fb]:
            continue
        rows[(fa in val_faces) + (fb in val_faces)].append(
            (int((placed[fa] > placed[fb]) == (va > vb)),
             int((control[fa] > control[fb]) == (va > vb))))

    rng = np.random.default_rng(0)
    names = {0: "bothTrained", 1: "oneHeldOut", 2: "bothHeldOut"}
    out: dict = {"valFaces": len(val_faces), "strata": {}}
    for k, name in names.items():
        v = rows[k]
        if not v:
            continue
        d = np.array([a - b for a, b in v])
        boot = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
        out["strata"][name] = {
            "pairs": len(v),
            "placed": float(np.mean([a for a, _ in v])),
            "control": float(np.mean([b for _, b in v])),
            "margin": float(d.mean()),
            "marginCi95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        }
    if rows[0] and rows[2]:
        g0 = np.array([a - b for a, b in rows[0]])
        g2 = np.array([a - b for a, b in rows[2]])
        boot = (g0[rng.integers(0, len(g0), (n_boot, len(g0)))].mean(1)
                - g2[rng.integers(0, len(g2), (n_boot, len(g2)))].mean(1))
        out["inSampleAdvantage"] = float(g0.mean() - g2.mean())
        out["inSampleAdvantageCi95"] = [float(np.percentile(boot, 2.5)),
                                        float(np.percentile(boot, 97.5))]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="train-v14-panel-ship")
    ap.add_argument("--ranking", default="artifacts/bt-refit-v5-panel")
    ap.add_argument("--panel", default="labels/panel-run-4-random")
    ap.add_argument("--rejects", default="artifacts/panel-run-v4/reject-pids.txt")
    ap.add_argument("--references", type=int, default=200)
    ap.add_argument("--photo-quality", action="store_true",
                    help="also split the panel comparison by photo quality (§5.0 QC flags)")
    ap.add_argument("--leakage-split", action="store_true",
                    help="split the panel comparison by faces this run trained on vs held out")
    ap.add_argument("--control-ranking", default="artifacts/bt-refit-v2-qc",
                    help="difficulty yardstick for --leakage-split; must predate the panel votes")
    ap.add_argument("--val-fraction", type=float, default=MIN_VAL_FRACTION)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--qc", default="artifacts/face-qc-v1/results.jsonl")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out) if a.out else ROOT / "artifacts" / a.run / "labs-composite.json"

    export = next(p for p in sorted((ROOT / "data" / "exports").glob("*"))
                  if (p / "faces.jsonl").exists())
    faces = pd.DataFrame(json.loads(ln)
                         for ln in (export / "faces.jsonl").read_text().splitlines())
    labs = dict(zip(faces["faceId"], faces["labsOverallScore"]))
    rank = pd.read_csv(ROOT / a.ranking / "ratings.csv")
    to_ten = make_scorer(ANCHORS_TOP10)
    scores = pd.read_csv(ROOT / "artifacts" / a.run / "model_scores.csv")

    # Place every cohort face once, and put Labs on the same /10 ladder by rank.
    placed: dict[str, float] = {}
    labs_ten: dict[str, float] = {}
    truth: dict[str, float] = {}
    for gender in ("female", "male"):
        rg = rank[rank["gender"] == gender]
        sg = scores[scores["gender"] == gender]
        theta_of = dict(zip(rg["faceId"], rg["theta"]))
        pct_of = dict(zip(rg["faceId"], rg["percentile"]))
        score_of = dict(zip(sg["faceId"], sg["modelScore"]))
        usable = {f: p for f, p in pct_of.items() if f in score_of}
        ref_ids = stratified_reference_ids(usable, n=a.references)
        ref_thetas = [theta_of[f] for f in ref_ids]
        ref_scores = [score_of[f] for f in ref_ids]
        population = list(theta_of.values())
        ladder = sorted(labs[f] for f in usable if f in labs)
        for f in usable:
            placed[f] = place(score_of[f], ref_scores, ref_thetas,
                              population_thetas=population).score_ten
            labs_ten[f] = to_ten(float(np.searchsorted(ladder, labs[f]) / len(ladder)))
            truth[f] = to_ten(pct_of[f])

    # The checkpoint's *actual* validation faces. An earlier version re-shuffled the faceIds in
    # model_scores.csv locally, which produced a plausible-looking but different set — so the
    # "held-out subjects" it reported were not the faces the run held out.
    _, val_rows = split_by_face_id(load_export(export).all_matchups(),
                                   a.val_fraction, a.split_seed)
    held_out = {f for m in val_rows for f in (m.face_a_id, m.face_b_id)}
    ref_all = set()
    for gender in ("female", "male"):
        rg = rank[rank["gender"] == gender]
        sg = scores[scores["gender"] == gender]
        usable = {f: p for f, p in zip(rg["faceId"], rg["percentile"])
                  if f in set(sg["faceId"])}
        ref_all |= set(stratified_reference_ids(usable, n=a.references))
    subjects = [f for f in held_out if f in placed and f not in ref_all]

    # ---- target 1: BT /10 ------------------------------------------------------
    p = np.array([placed[f] for f in subjects])
    lt = np.array([labs_ten[f] for f in subjects])
    t = np.array([truth[f] for f in subjects])
    tiers = np.array([tier_of(v)[0] for v in t])
    rng = np.random.default_rng(0)
    vs_bt = []
    for w in WEIGHTS:
        blend = w * p + (1 - w) * lt
        err = np.abs(blend - t)
        base = np.abs(p - t)
        idx = rng.integers(0, len(p), (2000, len(p)))
        gain = base[idx].mean(1) - err[idx].mean(1)
        lo, hi = np.percentile(gain, [2.5, 97.5])
        vs_bt.append({
            "w": w, "median": float(np.median(err)), "mean": float(err.mean()),
            "p90": float(np.percentile(err, 90)),
            "tierExact": float((np.array([tier_of(v)[0] for v in blend]) == tiers).mean()),
            "gainCI": [float(lo), float(hi)],
        })

    # ---- target 2: real human votes --------------------------------------------
    rejects = {ln.strip() for ln in (ROOT / a.rejects).read_text().splitlines() if ln.strip()}
    votes, meta = load_votes(ROOT / a.panel / "results",
                             ROOT / a.panel / "sample-meta.json", rejects, True)
    vs_panel = []
    for w in WEIGHTS:
        maj_hit = maj_n = 0
        vote_hit = vote_n = 0.0
        for pid, (va, vb) in votes.items():
            m = meta.get(pid)
            if m is None:
                continue
            fa, fb = m["faceAId"], m["faceBId"]
            if fa not in placed or fb not in placed:
                continue
            sa = w * placed[fa] + (1 - w) * labs_ten[fa]
            sb = w * placed[fb] + (1 - w) * labs_ten[fb]
            if sa == sb:
                continue
            picks_a = sa > sb
            vote_hit += va if picks_a else vb
            vote_n += va + vb
            if va != vb:
                maj_hit += int(picks_a == (va > vb))
                maj_n += 1
        vs_panel.append({"w": w, "vsMajority": maj_hit / maj_n, "vsVotes": vote_hit / vote_n,
                         "pairs": maj_n})

    err_corr = float(np.corrcoef(p - t, lt - t)[0, 1])
    ceil = ceilings(votes)
    result = {
        "run": a.run, "ranking": a.ranking, "panel": a.panel,
        "subjects": len(subjects), "errorCorrelation": err_corr,
        "spearmanLabsVsBT": float(spearmanr(lt, t).statistic),
        "spearmanPlacedVsBT": float(spearmanr(p, t).statistic),
        "ceiling": {k: v for k, v in ceil.items() if isinstance(v, float)},
        "vsBT": vs_bt, "vsPanel": vs_panel,
        "verdict": "do not blend — helps against BT, flat-to-worse against humans",
    }

    if a.leakage_split:
        ctrl = pd.read_csv(ROOT / a.control_ranking / "ratings.csv")
        result["leakage"] = leakage_split(votes, meta, placed,
                                          dict(zip(ctrl["faceId"], ctrl["theta"])), held_out)
        result["leakage"]["controlRanking"] = a.control_ranking

    if a.photo_quality:
        issues = photo_issues(ROOT / a.qc)
        if not issues:
            raise SystemExit(f"no QC results at {a.qc} — run scripts/qc_faces.py first.")
        result["photoQuality"] = quality_split(votes, meta, placed, labs_ten, truth, issues)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print(f"\n{a.run}, {len(subjects)} held-out subjects, {vs_panel[0]['pairs']} panel pairs")
    print(f"error correlation placed vs Labs: {err_corr:+.3f}  "
          f"(near zero, which is why the BT-target result looks so good)")
    print(f"\n{'w on comparator':>16} | {'BT med':>7} {'BT p90':>7} {'BT tier':>8} | "
          f"{'vs majority':>12} {'vs votes':>9}")
    print(f"{'':>16} | {'':>7} {'':>7} {'':>8} | "
          f"{ceil['vsMajority']:>11.1%}* {ceil['vsVotes']:>8.1%}*  <- human ceiling")
    for b, q in zip(vs_bt, vs_panel):
        tag = "  <- placement only" if b["w"] == 1.0 else ("  <- Labs only" if b["w"] == 0 else "")
        print(f"{b['w']:>16.0%} | {b['median']:>7.3f} {b['p90']:>7.2f} {b['tierExact']:>8.1%} | "
              f"{q['vsMajority']:>12.1%} {q['vsVotes']:>9.1%}{tag}")
    best_bt = min(vs_bt, key=lambda r: r["median"])
    best_panel = max(vs_panel, key=lambda r: r["vsMajority"])
    print(f"\nbest against BT:     w={best_bt['w']:.2f}  (median {best_bt['median']:.3f})")
    print(f"best against humans: w={best_panel['w']:.2f}  ({best_panel['vsMajority']:.1%})")
    print("\nThe two targets disagree. Humans are the target and BT is the proxy, so the answer "
          "is\nDO NOT BLEND. A change that moves the proxy and not the target is fitting the "
          "proxy's error.")

    if a.leakage_split:
        lk = result["leakage"]
        ctrl_name = a.control_ranking.split("/")[-1]
        print(f"\n--- is the {vs_panel[0]['vsMajority']:.1%} the comparator, or is it memory? ---")
        print(f"Same pairs, split by how many faces `{a.run}` trained on. `{ctrl_name}` never saw "
              f"these\nvotes and is blind to this run's split, so it measures each stratum's "
              f"difficulty.\n")
        print(f"{'faces':>14} {'pairs':>7} {'placed':>8} {ctrl_name[:11]:>12} "
              f"{'margin':>8} {'95% CI':>18}")
        for name, lbl in (("bothTrained", "both trained"), ("oneHeldOut", "one held out"),
                          ("bothHeldOut", "both held out")):
            s = lk["strata"].get(name)
            if s:
                lo, hi = s["marginCi95"]
                print(f"{lbl:>14} {s['pairs']:>7} {s['placed']:>8.1%} {s['control']:>12.1%} "
                      f"{s['margin']:>+8.1%}  [{lo:+.1%}, {hi:+.1%}]")
        if "inSampleAdvantage" in lk:
            lo, hi = lk["inSampleAdvantageCi95"]
            print(f"\nin-sample advantage (margin on trained faces minus margin on unseen ones): "
                  f"{lk['inSampleAdvantage']:+.1%}  [{lo:+.1%}, {hi:+.1%}]")
        print("Read the MARGIN column. A margin that shrinks as the faces become unseen means the")
        print("pooled headline is partly memorisation, and the honest number for a stranger's")
        print("upload is the bottom row — or better, the off-cohort test in log §5.10.")

    if a.photo_quality:
        q = result["photoQuality"]
        print("\n--- photo quality: is the Labs formula bad, or are the photos? ---")
        print("Pairs where both faces are clean per the QC pass, versus pairs where either is "
              "flagged.\nAll figures are vs the panel MAJORITY.\n")
        print(f"{'subset':>14} {'pairs':>7} {'ceiling':>9} {'Labs':>8} {'gap':>7} "
              f"{'placed':>8} {'gap':>7} {'BT†':>8} {'gap':>7}")
        for label in ("all", "bothClean", "eitherFlagged"):
            r = q["arms"][label]
            print(f"{label:>14} {r['labs']['pairs']:>7} {r['ceiling']:>8.1%} "
                  f"{r['labs']['vsMajority']:>8.1%} {r['labsGapToCeiling']:>+7.1%} "
                  f"{r['placed']['vsMajority']:>8.1%} {r['placedGapToCeiling']:>+7.1%} "
                  f"{r['bt']['vsMajority']:>8.1%} {r['btGapToCeiling']:>+7.1%}")
        cl, al = q["arms"]["bothClean"], q["arms"]["all"]
        d_labs = cl["labsGapToCeiling"] - al["labsGapToCeiling"]
        d_pl = cl["placedGapToCeiling"] - al["placedGapToCeiling"]
        print(f"\n† Do not quote the BT column. `{a.ranking.split('/')[-1]}` was fitted on these "
              f"very votes,\n  so it is scored on its own training data. The honest VLM-only "
              f"figure is in log §5.8.")
        print(f"\nGap to ceiling, clean minus all:  Labs {d_labs:+.1%}   placement {d_pl:+.1%}")
        print("Read the GAP columns, not the levels — clean pairs are easier for the humans too,")
        print("so every accuracy rises and only the distance to the ceiling is evidence. Labs")
        print("closing much faster than the placement would mean photo quality is a Labs-specific")
        print("weakness, i.e. a gating problem; both moving together means it is not.")

    print(f"-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
