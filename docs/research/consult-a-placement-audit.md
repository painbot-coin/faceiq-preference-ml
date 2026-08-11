# Consult (a) — Is reference-set placement the right estimator?

*2026-08-07. First consulting deliverable against `docs/ml-consult-brief.txt` §9a.
Scoreboard uses the corrected §5.11 numbers, not brief §5. No local `data/` /
`artifacts/` / `checkpoints/` in this working tree — this pass is math + code +
recorded evidence, not a re-measure.*

## Success metric (locked with stakeholder 2026-08-07)

**Primary:** on **held-out pairs**, does the production score (comparator → placement →
ordering) pick the face the **panel majority** picked?

Rationale (stakeholder): individuals disagree ~30% of the time (more on close pairs);
majority is the stabler target. A model that matches the crowd is more accurate than
one that matches a single rater.

Operational rules for any experiment:

- Score **majority winner**, not individual ballots (unless reporting the secondary
  “vs votes” column for comparison).
- Pairs must be **held out**: both faces unseen by the checkpoint when claiming
  upload-path accuracy; attach the pair distribution (near-tie vs uniform).
- Do **not** treat “closer to BT /10” or Labs formula as success. Those are debug
  proxies only (Labs-composite lesson, §5.8 / §5.11).
- Ceiling to quote beside the number: one held-out rater vs the majority of the
  others on the **same** pair set (run 4 uniform: ~73.2% majority-level).

## Verdict

**Ship reference-set placement. Do not replace it with a full BT refit.**

It is the right *class* of estimator for production: conditional MLE of a new
face’s θ given win/lose outcomes against fixed known-θ references. What it is
*not* is an estimate of the user’s θ under the human Bradley-Terry generative
model. It estimates the θ that best explains the **comparator’s** record against
those references, then reads that θ on the cohort yardstick.

That distinction is the whole audit.

| Claim in the brief / pipeline docs | Assessment |
|---|---|
| Conditional MLE ≡ add one node, hold existing edges fixed | **True** |
| Full refit would move reference θs by ~0.4% — not worth it | **True enough** to reject full refit as a production path |
| Placement puts the user on the BT scale in a well-founded way | **Yes, as a bridge from comparator signs → θ yardstick** |
| Held-out ρ 0.926 / tier-exact 67% / product 81.9% | **Superseded** — use §5.11 |
| Something better-founded is waiting in “full BT” or soft margins | **No** for full BT; soft margins low priority; ordering gap is (b) |

## What the code does

`src/faceiq_pref/placement.py` → `fit_theta` / `place`:

1. Hard outcomes only: `beats_j = score > reference_score_j` (magnitude discarded).
2. Newton MLE: `P(beat j) = sigmoid(θ − θ_j)`, strictly concave in one variable.
3. `se(θ) = 1 / √Σ p_j(1−p_j)` from observed information.
4. Percentile against **population** θs (not the reference set).
5. `/10` via `ANCHORS_TOP10`; band = hypot(estimation half-width, disagreement half-width at `T = 1.920`).
6. All-win / all-lose → cap at extreme reference ± `UNBOUNDED_MARGIN`, `bounded=False`.

Reference construction (`stratified_reference_ids`) stratifies by **tier**, not
equal percentile mass — required so the top of the `/10` scale has references and
the MLE stays finite (~16% of top-5% female faces otherwise unbounded).

## Why this is well-founded

**Identity with “refit holding the graph fixed.”** Adding the user as a new node
with edges only to the reference faces, while freezing every existing cohort edge,
*is* this 1-D conditional MLE. The pipeline doc’s claim here is correct.

**Why not float all θ.** Letting the ~2,866 cohort θs move under ~200 new edges
is a ~200/47,914 ≈ 0.4% edge-mass perturbation. Even if the constant is rough,
the direction is right: you pay a full joint solve per upload to jiggle the
yardstick under every user. Keeping references fixed is what makes two uploads
comparable on one scale. Full refit is not “more correct” for the product; it is
a different, worse product contract.

**Why signs, not margins.** The siamese head is trained on differences only. Its
score scale is not identified. Measured cost of discarding magnitude: none on
Spearman (0.857 either way, log §5.8). Soft-weighting by `|s − s_j|` would be a
re-introduction of a scale you have already declined to trust.

**Why population ≠ references.** References should over-sample thin tiers so θ is
identifiable at the extremes. Percentile must be read off the full cohort, or a
true ~90th lands near ~70th. The API makes this hard to get wrong if callers pass
`population_thetas`; defaulting to reference θs is a footgun for demos only.

**Why ~200, why spread not “quality.”** Ordering accuracy plateaus by ~50 refs;
`se` keeps shrinking but is already ~5× smaller than the human-disagreement band
at 200. Stratified spread helps ~0.03–0.06 `/10`; selecting tight-θ or
panel-covered faces does not help (panel faces cluster mid-scale).

## The misspecification (state this in any write-up)

The likelihood assumes

```text
Y_j ~ Bernoulli( sigmoid(θ_user − θ_j) )
```

with `Y_j` a true preference. Production uses `Y_j = 1[comparator prefers user to j]`.

So θ_user is **not** “the BT strength a human panel would assign.” It is “the
point on the existing θ axis that best reconciles this model’s pairwise decisions
against the reference ladder.” Transfer onto the human/BT scale is only as good
as those signs.

Consequences already in the programme’s own measurements:

- Placement **cannot repair ordering**. Comparator errors in the 10–45 gap band
  become placement errors. That is question (b), not a reason to change the placer.
- Fisher `se(θ)` is the curvature under the *fitted* (misspecified) model. When
  signs are wrong, true placement error is larger. In practice the disagreement
  term dominates the band (~5×), which is why growing N past 200 is the wrong lever.
- Absolute `/10` vs BT (Gate 1) is a proxy test. Failing it means “noisier at
  reproducing BT’s point score than hoped,” not “the conditional MLE algebra is
  wrong.”

## Corrected scoreboard (do not quote brief §5)

From log **§5.11** / `production-scoring-pipeline.md` gates, `train-v14-panel-ship`,
true `split_by_face_id` on the export:

| | Brief / pre-fix | True held-out |
|---|--:|--:|
| Spearman vs ranking | 0.926 | **0.852** |
| Median error F / M | 0.34 / 0.28 | **0.495 / 0.486** |
| p90 | 0.93 | **1.32 / 1.46** |
| Tier-exact | 67% | **54.4%** |

Gate 1 (ρ > 0.90 and median < 0.5) is **failed** as pre-registered. Checkpoint
ordering also re-opens: `train-v22-ship-0.2` leads slightly on the BT proxy
(ρ 0.857, tier 56.8%); decide on **off-cohort human** labels per standing rule,
not proxy alone.

The **81.9%** panel figure was measured with ~96% of pairs containing a training
face. Leakage split:

| faces | pairs | placement | bt-refit-v2-qc | margin |
|---|--:|--:|--:|--:|
| both trained on | 1,499 | 82.4% | 81.5% | +0.9 |
| one held out | 760 | 81.4% | 82.0% | −0.5 |
| both held out | 111 | 78.4% | 84.7% | **−6.3** |

Upload-path claim that survives: off-cohort ordering (§5.10 / §5.10a), and the
ranking’s crowd-prediction claim — not “placement beats a human by eight points.”

## Alternatives considered

| Alternative | Verdict | Reason |
|---|---|---|
| Keep conditional MLE, N≈200, tier-stratified | **Adopt** | Right class; shared scale; se for bands |
| Full joint BT refit per upload | **Reject** | ≈ same estimator; breaks fixed yardstick |
| Soft / margin-weighted Bernoulli | Low priority | Magnitude untrusted; no Spearman gain measured |
| Isotonic comparator-rank → BT percentile | Near-duplicate | Same monotone signal; weaker se story |
| Close comparator↔BT gap (distill / listwise) | **→ (b)** | Placement is not the lever |
| Gap-routed ensemble at inference | Next product bet | Global order from placement; near-ties elsewhere |

## Extremes, compression, and “attractive faces land at 6–7.5”

These are real, measured, and mostly **not** the same bug. Treat them as three stacked
mechanisms before deciding on more labelling spend.

### 1. Quantisation onto the reference ladder (structural)

Off-cohort (§5.10): **74 photos → 49 distinct `/10`**. The MLE uses only *how many*
references a face beat. Two faces between the same adjacent references get the same
number by construction — not a rounding artefact (`offcohort.py`).

| Lever | Helps? |
|---|---|
| More of the same close-pair labels | **No** |
| More references (→ finer win-count ladder) | Marginally |
| Soft / margin-weighted outcomes | Maybe (re-opens magnitude trust) |
| Ship bands, not decimals | **Yes — product fix** |

### 2. Extremes are weaker (measured twice)

On-cohort (§5.8): median placement error ~2× worse at the ends than mid-scale; ~16% of
top-5% female faces beat every reference (no finite MLE) before tier stratification
(~0.2% after). Off-cohort (§5.10a): inside-band accuracy **above 5.5 `/10` is chance**
(48.8% M / 57.7% F). The **top** of the scale is the weak region that replicates.

Mechanism: tier 7 is ~1% of the cohort (~14 faces per gender). The reference ladder is
thin where “9–10” lives, and sign-only placement cannot say “much better than the top
refs” except via the unbounded cap.

### 3. Compression / hard to reach 9–10 (several causes stacked)

What people experience as “very attractive → 6–7.5” is usually some mix of:

1. **Regression to the mean in placement.** Placed `/10` vs true `/10` has slope
   well below 1 (bundle note: ~0.74 M / 0.81 F). A temperature on the placement link
   was tried; Spearman stays flat (monotone map) and forcing slope → 1 trades away
   median / tier accuracy. Compression here is not a calibration knob bug.
2. **Cohort + anchor curve.** The research set is ~1,500 faces/gender from an app
   funnel, stratified on the *old* formula’s deciles. The absolute top of *this*
   sample is not “population 99.99th.” Early anchors deliberately compressed the
   right tail (ceiling 9.0); `ANCHORS_TOP10` stretches display to 10 but cannot invent
   resolution the sample does not have.
3. **Comparator global compression.** Local pairwise order can look fine while the
   absolute score range is squashed — exactly the failure mode
   `placement_by_checkpoint.py` was written to catch.
4. **Possible systematic low shift off-cohort.** Validation sets sat at median
   ~4.4–4.5 `/10` with most mass in tiers 2–3. Either those uploads are below the
   curated cohort’s middle, or placement is shifted low. **Only hand ranges settle
   this** (`ranges.json` / test 3) — still unrun.

So “difficult to score 9–10” is partly honest resolution (people cannot agree that
finely), partly a thin elite ladder, partly RTM, and possibly a scale shift — not
one missing hyperparameter.

## Is another $30k of A/B labelling worth it?

**Not if it is more of the same recipe** (Gemini-style pool → Prolific close pairs on
the existing 3k cohort). That path was already measured shut for the *user’s* score:

| Route for new votes → production `/10` | Result |
|---|---|
| Better reference θ (ranking ladder v2→v5) | **0.2 pts, downward** (`ranking_value_to_placement.py`) |
| Better / more panel labels in comparator training (`train-v22`) | **+0.00 pts [−0.96, +0.96]** vs panel |
| Headroom on close pairs (VLM vs human) | Still real — labels beat Gemini locally |
| Remaining ~$7.5k buy zone / ~$14k cancelled | Closed 2026-08-04: quality no longer binding |

Human close-pair labelling was the right $4k spend (near-ties are where VLM fails).
Repeating it at $30k improves a ranking and a training signal that **do not currently
reach** the placed score. Reopening needs a **different kind** of label, not a larger
cheque on the same band table.

### If you *do* spend, what could move the symptom

Ranked by fit to “attractive faces stuck at 6–7.5 / no 9–10”:

| Spend | Fits the symptom? | Notes |
|---|---|---|
| **Hand ranges / absolute calibration on ~30–100 known faces** (incl. clearly elite) | **Yes — diagnostic first** | Cheap; separates shift vs true mid-cohort; unlocks test 3 |
| **New elite faces into the graph** (intake of highly attractive faces + pairs among them and into the existing cohort) | **Maybe** | Attacks thin top ladder; still limited by comparator extraction |
| **Elite-vs-elite A/B densification on existing top faces** | Weak alone | Better BT top ranks; placement stays comparator-dominated |
| **More mid-scale close pairs (old buy zone)** | **No for this symptom** | Already closed for production score |
| **Extraction work (b): distill / listwise / gap-routing** | **Yes — main accuracy bet** | Addresses compression without buying pairs |
| Test–retest / multi-photo | Band narrowing, not point 9–10 | Different product feature |

**Working recommendation:** do not commit $30k to Prolific close pairs yet. Spend a
small amount on **absolute ranges for elite + ordinary faces** and on **(b) /
gap-routing**. Only reopen large A/B spend if the ranges show a systematic scale
shift *or* if a new elite cohort is added and you need edges to attach it — and even
then, pre-register a placement/panel metric, not a BT-proxy win.

## Recommendations

1. **Treat (a) as closed on estimator choice:** wire `place()`; do not pursue full
   refit or hand-labelled `/10` anchors on the reference set.
2. **Hard requirements in any production port:** sign-only outcomes; tier-stratified
   refs; `population_thetas` for percentile; unbounded cap + flag; per-user `se` in
   the band (extremes are ~2× worse).
3. **Claims hygiene:** quote Gate 1 as failed; quote upload accuracy from
   leak-free / off-cohort only; keep “beats an individual human at crowd
   prediction” attached to the **ranking**, not the upload path.
4. **Checkpoint:** reopen v14 vs v22 on off-cohort humans when checkpoints are
   available — not on BT-proxy alone.
5. **Accuracy work moves to (b)** and gap-routing, not a new placement theory.
6. **Do not buy $30k more of the same A/B recipe** to fix extremes/compression;
   diagnose with ranges first; elite coverage and extraction are the real levers.
7. **Product:** lead with tier + band; treat decimals as interior to the band so
   quantisation and RTM are not user-visible as false precision.

## Baseline re-run (2026-08-07, this machine)

Bundle merged; success metric = **panel majority on held-out pairs**.

```bash
python scripts/placement_by_checkpoint.py --common-val --references 200 \
  --out artifacts/placement-by-checkpoint-common-val.json
python scripts/labs_composite_eval.py --run train-v14-panel-ship --leakage-split
python scripts/labs_composite_eval.py --run train-v22-ship-0.2 --leakage-split
python scripts/validate_placement.py report --set set-1-female
python scripts/validate_placement.py report --set set-1-male
```

### Gate 1 (BT proxy — debug only): common val, 200 refs, n=489

| checkpoint | ρ | median F/M | tier-exact |
|---|--:|--:|--:|
| `train-v22-ship-0.2` | **0.857** | 0.487 / 0.465 | **56.8%** |
| `train-v14-panel-ship` | 0.852 | 0.495 / 0.486 | 54.4% |

Gate 1 still **fails** pre-registered ρ > 0.90. Confirms §5.11.

### Success metric: vs panel majority (run 4), leakage split

Human ceiling on this draw: **73.2%** majority-level.

| checkpoint | both trained | one held out | **both held out** | in-sample adv. |
|---|--:|--:|--:|--:|
| v14 placed | 82.4% (1499) | 81.4% (760) | **78.4%** (111) | +7.2 pts |
| v14 control `bt-refit-v2-qc` | 81.5% | 82.0% | **84.7%** | — |
| v22 placed | 82.7% (1498) | 80.9% (759) | **80.2%** (111) | +6.3 pts |
| v22 control `bt-refit-v2-qc` | 81.8% | 82.5% | **85.6%** | — |

**Number to beat for upload-path work:** the both-held-out row (~78–80%), not the pooled ~82%.
Ranking still leads placement by ~5–6 pts on those pairs. Off-cohort (one rater, not panel
majority): 84.3% F / 76.4% M — reproduced.

### Next experiments (against majority, held-out)

1. ~~Gap-routed scoring~~ — **measured 2026-08-07, see below.**
2. Soft-margin / more-refs ablation on quantisation — only if majority moves.
3. Absolute ranges (stakeholder) if we need to diagnose 6–7.5 compression separately.
4. **(b) extraction** — make placement match ranking on far pairs for *new* faces.

### Gap-routed ensemble (2026-08-07)

`scripts/gap_routed_eval.py` → `artifacts/<run>/gap-routed.json`.

Success metric: panel majority. Focus stratum: **bothHeldOut** (n=112).

| arm | v14 | v22 |
|---|--:|--:|
| comparator only | 78.6% | 79.5% |
| placement only | 78.6% | 80.4% |
| ranking `bt-refit-v2-qc` only | **84.8%** | **84.8%** |
| best ORACLE (true pct gap → cmp else rank) | **86.6%** (+8.0 vs place) | **87.5%** (+7.1) |
| best PREDICTED (\|Δ/10\| → cmp else place) | 78.6% (**+0.0**) | 79.5% (**−0.9**) |

**Verdict: do not ship predicted gap-routing.** It cannot move the production score.
Oracle headroom is real but almost entirely “use the ranking on far pairs” (at the best
threshold only ~9–15% of pairs take the comparator). New uploads have no ranking row, so
that far arm is unavailable. Hybrid (true gap, far=placement) also gains **0** — confirmation
that the win is the ranking lookup, not clever near-tie switching.

Implication: the held-out gap to the ranking (~5–6 pts) is an **extraction** problem (b),
not a routing problem.

### (b) BT soft-target distillation — measured (2026-08-08)

Arm that the brief asked for and the programme never ran: train targets =
`sigmoid((θ_a − θ_b) / T)` from **vote-blind** `bt-refit-v2-qc`, warm-start from
`train-v14-panel-ship`, backbone frozen (CPU), panel pairs keep human labels.

| | |
|---|---|
| Config | `configs/train-v23-bt-distill.yaml` |
| Checkpoint | epoch 2, `panel_val_accuracy` 0.554 |
| Spearman vs BT | 0.959 (faces) |

**Success metric — bothHeldOut vs panel majority:**

| run | bothHeldOut | vs ranking control |
|---|--:|--:|
| v14 placement | 78.4% | −6.3 |
| **v23 BT-distill** | **78.6%** | −6.2 |
| v22 placement | 80.2% | −5.4 |
| ranking `bt-refit-v2-qc` | 84.8% | — |

**Verdict: head-only BT soft distill does not close the gap** (+0.2 vs v14, noise).
Still ~6 pts behind the ranking on strangers.

**Follow-up (2026-08-10):** same recipe with `freeze_backbone: false` on GPU
(`train-v23b-bt-distill-gpu`) → **82.6%** bothHeldOut majority (n=109) vs ranking control
85.3% (−2.8). Current ship vs v14. Larger backbone `train-v24-arcface-r100` → **78.4%**
bothHeldOut (n=111) with +7.2 pts in-sample advantage — killed (memorisation). Next: short
FT off v23b or non-ArcFace ensemble; not more R100. Details:
[`meruzhan-next-steps.md`](./meruzhan-next-steps.md).

## Pointers

- Code: `src/faceiq_pref/placement.py`, `scripts/reference_set_inference.py`
- Evidence: log §5.8 (placement design), §5.11 (split bug / gates), §5.10 (off-cohort)
- Pipeline: `docs/research/production-scoring-pipeline.md` §§2–3, gates, ship order step 3
- Brief question: `docs/ml-consult-brief.txt` §9a
