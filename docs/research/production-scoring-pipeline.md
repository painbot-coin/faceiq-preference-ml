# Production scoring pipeline — what ships, in what order, and what is still missing

*2026-08-04. The consolidation doc. Everything else in `docs/research/` explains how we learned
something; this one says what a user's photo actually goes through and which pieces exist today.
Keep it short — if a section needs a page, it belongs in the research log with a pointer here.*

**Reading order:** this file for "what are we building", [`README.md`](./README.md) for "what is the
next task", [`programme-direction-review.md`](./programme-direction-review.md) for "is this the right
direction", [`scoring-gt-research-log.md`](./scoring-gt-research-log.md) for the evidence.

---

## The pipeline, end to end

```
        upload
          │
   ┌──────┴──────┐
   │ 1. photo QC │  VLM: real photo? one face? usable? → sigma_photo hint      [NOT BUILT]
   └──────┬──────┘
          │
   ┌──────┴───────────────┐
   │ 2. comparator scores │  ArcFace-R50 siamese, pixels → scalar               [BUILT]
   └──────┬───────────────┘
          │
   ┌──────┴────────────────────────┐
   │ 3. reference-set placement    │  compare vs ~200 cohort faces of known     [MEASURED,
   │    theta_new by MLE + se      │  theta; MLE gives theta and a std error      NOT WIRED]
   └──────┬────────────────────────┘
          │
   ┌──────┴──────────────────┐
   │ 4. percentile → /10      │  pre-registered anchor curve (50th=5.0,         [BUILT]
   │                          │  90th=7.0, 99th=8.0)
   └──────┬──────────────────┘
          │
   ┌──────┴──────────────────┐
   │ 5. band, not a decimal   │  half-width = T·logit(agreement)/2, T = 1.920   [MEASURED,
   └──────┬──────────────────┘                                                   NOT WIRED]
          │
   ┌──────┴──────────────────────────┐
   │ 6. multiple photos → tighter band│  inverse-variance weighting             [BLOCKED]
   └──────────────────────────────────┘
```

### The product surface, as agreed — four claims, three shippable now

Confirmed 2026-08-04. Each line is a claim we can defend, with what it rests on:

| what the user sees | status | rests on |
|---|---|---|
| **A range**, e.g. 7.4–8.1, from the comparator against the reference set | measured, not wired | `T = 1.920` fitted out of sample (§5) + per-user `se(θ)` from the MLE (§3) |
| **A tier**, 1–7 | ✅ defined and computed (§5c) | two independent measurements both landing on ~7 (rank resolution and the calibration curve) |
| **"About two thirds of people would place you above a typical Tier 4 face"** | ✅ computed (§5c-bis) | the calibration curve, applied at the *tier* spacing. Say the tier, never the bare number |
| **A 90% floor claim** — "at least Tier 2" | ✅ arithmetic on the same curve (§5c-ter) | one-sided, so it never shrinks the headline. This is the answer to "let users pick a confidence level", which we should **not** ship as a slider |
| **Exemplar faces per tier** | not sourced (§5c-quater) | licensed / consented / generated images — **never cohort photos**, which are real users' uploads. Cosmetic, so hand-picking is correct here |
| **σ-weighted multi-photo + geometric formula** | blocked / unbuilt | needs test–retest (§6) and the EBM (§"second opinion") |

One correction to the framing worth keeping visible: tiers do **not** "compress as scores go higher" —
tiers are *equal-width* on the /10 scale, and it is the **population** inside them that thins out (tier
7 is the top 1%, tier 4 is 31% of everyone). Equal-population tiers were considered and rejected,
because they would put mid-scale boundaries closer together than the resolution limit and claim a
distinction the data cannot support. The product story is better anyway: "top 1%" beats "septile 7".

Everything except the fourth row is shippable with no new data and no new training.

---

## 1. Photo QC and a per-photo sigma — NOT BUILT

Two separate jobs, and only the first is urgent.

**Gatekeeping** (is this a real, single, usable face?) reuses `scripts/qc_faces.py`, which already
runs this prompt over the cohort at ~$0.002 per image. Straightforward.

**A quality-derived `sigma_photo`** is the interesting one, and it is the input the whole
multiple-photo feature needs. Two routes:

| route | how | status |
|---|---|---|
| **VLM-derived** | ask the QC pass for lighting / blur / angle / occlusion, map to a sigma | **now the only live route.** Easy, but the mapping is a guess until calibrated against something |
| model-derived | `variance_head` predicts `(mu, log sigma^2)` per image and *learns* sigma from data | ❌ **measured 2026-08-04, does not work for this** |

**The model-derived route failed, and the failure is instructive.** `train-v19` ran the variance head
against panel targets. sigma did *not* collapse — it ranges 0.42 to 1.85, CV 0.22 — but it correlates
**+0.52 with the /10 score**, and once that is regressed out the residual is flat across every
photo-quality flag we have:

| photo issue | faces | residual sigma |
|---|--:|--:|
| obstruction | 277 | −0.027 |
| poor_lighting | 179 | −0.015 |
| low_resolution | 221 | −0.005 |
| no issues flagged | 2,066 | +0.003 |
| heavy_filter | 39 | +0.059 |

Against a sigma sd of 0.169 that is nothing. It learned **"the thin top of the ranking is
ill-determined"** — which is true — not **"this photo is bad"**. Nothing in the training signal varies
photo quality, so there was nothing for sigma to attach to.

**If we return to it**, train with quality augmentation — random blur, downsampling, lighting jitter —
so degradation actually appears in the loss. Until then, use the VLM route and treat the mapping as a
tunable guess. Log §5.8.

---

## 2–3. Reference-set placement — the most important unshipped piece

### The flow, in five steps

Precomputed once, offline, and cached — none of this runs per upload:

- **Two reference sets, one per gender.** ~200 cohort faces, tier-stratified, each with a fixed
  Bradley-Terry `θ` from the ranking of record. Per-gender because the pair queue was same-gender
  only, so the two BT graphs are disconnected and their θ scales share no common zero.
- **A comparator score for every reference face**, from the same checkpoint production will use.
  Mixing checkpoints silently invalidates every comparison.
- **`T = 1.920`**, the scale's bluntness (§5).

Then per upload:

1. **Normalise** with the faceiq-labs MediaPipe eye-level crop — the exact pipeline that produced the
   cohort photos. One forward pass gives the user a comparator score `s`.
2. **Compare** `s` against each of the ~200 cached reference scores. That is 200 win/lose outcomes.
   **Only the sign is used**; the score's magnitude is on an arbitrary scale we have no reason to
   trust, and discarding it costs nothing (Spearman 0.857 either way).
3. **Fit** the one θ that best explains those 200 outcomes, under
   `P(user beats ref_j) = sigmoid(θ_user − θ_j)`. One-dimensional and strictly concave, so a few
   Newton steps solve it, and the curvature at the optimum gives a standard error free:
   `se(θ) = 1/sqrt(Σ p_j(1−p_j))`.
4. **Convert** θ to a percentile against the **full cohort's** θ distribution (not the reference
   set's — see §3c-bis), then to /10 through the anchor curve (§4).
5. **Present** as a tier, a band and an agreement sentence (§5).

Adding one node to a BT graph while holding the existing 47,914 edges fixed *is* this conditional MLE,
so step 3 is "refit Bradley-Terry with the user included" at 1/1000th of the cost. The reference θs
stay fixed on purpose: that is what keeps every user on one comparable scale.

**What this buys, stated honestly:** a per-user standard error (which is what the band needs), a
position on the θ scale where distances mean something in log-odds, explicit handling of the faces
that beat every reference, and 200 forward passes instead of 2,866. It does **not** repair the
ordering — if the comparator ranks two faces wrongly, this ranks them wrongly too.

<details><summary>Why ~200 references, why no curation, and why the sets are not split by ethnicity or photo condition</summary>

Measured in log §5.8 (`scripts/reference_set_inference.py`). Ordering accuracy plateaus by 50
references and is flat to 700; only `se` keeps shrinking, and at 200 it is already ~5× smaller than
the disagreement band, so it is not the bottleneck. 10–20 references is genuinely too few. The error
floor is the *comparator's*, not the reference set's.

Curation actively hurts. Each reference's weight in the fit is `p_j(1−p_j)` — 0.25 for an
evenly-matched opponent, collapsing toward zero for a mismatch — so the set self-selects toward faces
near the user and a face you think is mis-ranked contributes almost nothing to anyone it is not near.
Selecting for tight θ intervals or for panel coverage measurably does *not* help; panel coverage is
marginally worse, because panel-covered faces cluster in dense parts of the scale. **Spread across
the scale is the only selection rule that pays** (~0.03–0.06 /10), which is what `stratified_
reference_ids` does.

Do not split the set by ethnicity or by photo condition. Its job is to locate a face on the
*existing* cohort-wide percentile scale; splitting defines a different scale per group, so a user's
score would depend on which set they were compared against. Audit win rates by subgroup instead. For
photo condition the right lever is **σ, not the yardstick**: a poor photo should get a *wider band*,
not a different scale (§1).

</details>

<details><summary>Correction kept on the record: the unanchored scalar was over-blamed</summary>

An earlier version of this section called the unanchored scalar "the most likely reason an upload
scores oddly". That overstated it. The Inference tab already computes a **rank** within the model's
own score distribution, and a rank is shift-invariant — if the head's bias moves every score
together, the percentile does not move at all. Confirmed by measurement: placement matches the raw
scalar's Spearman almost exactly.

The three likelier reasons, in order: the score is **shown as a point** when its honest resolution is
over a point wide; **normalisation parity** (audited ✅, see Ship order); and **genuine per-face model
error, worst at the extremes**.

</details>

---

### 3a. Building the reference set — a config value, not a curation task

`stratified_reference_ids(percentiles, n=200)`. Take all QC-clean ranked faces of one gender,
stratify by **tier**, sample randomly within each, store the `faceId` list and each face's `θ`. That
is the whole artifact — no page to build, no faces to eyeball, no /10 labels to assign.

**Two rules the code must not get wrong**, both found by testing rather than reasoning:

1. **Stratify by tier, not by equal percentile bands.** The /10 scale is stretched at the top — tier 7
   is the top 1%, about 14 faces per gender in this cohort — so an even split over *percentile* puts
   almost no references up there, a top-1% user beats essentially all of them, and the MLE is
   unbounded. Tier stratification puts references where the *score* needs resolving. Tiers fill to
   availability and the shortfall redistributes, so the count is approximately, not exactly, `n`.
2. **Read the percentile off the population, not the references.** References *estimate θ* and should
   over-sample the thin extremes; the percentile conversion needs a population-representative
   distribution. Conflating them is a real bug that pulled a true 90th-percentile face down to about
   the 70th. `place()` takes `population_thetas` separately for this reason.

**If you want to change what a 7 means, change the anchor curve, not the faces** (§4). One function
and one product decision, not 200 individual judgments.

### 3b. Two behaviours to handle in code, not edge cases

| | |
|---|---|
| **Accuracy varies ~2× along the scale** — best in the middle 50%, worst at either end | so the band uses the per-user `se(θ)` the fit already returns, not a constant width |
| **A face that beats *every* reference has no finite MLE** — expected for roughly the top 1%, and much more common if you stratify by percentile instead of tier | cap θ at the top reference plus a margin, flag it (`bounded=False`), and read it as "at least this". Never let the optimiser run away |

Numbers behind both, and the reference-count sweep, are in log §5.8.

### 3c. "Some reference faces are clearly in the wrong tier" — why that is tolerable

The single most common objection to this design, and it has a real answer rather than a reassurance.
Browsing the 200-face set you will find faces that look a tier too low or too high. Three reasons
that does not propagate to the user's score:

1. **A mis-ranked reference is down-weighted automatically.** Reference `j` contributes
   `p_j(1−p_j)` to the fit: 0.25 at most, for an opponent the user is evenly matched against, and
   collapsing toward zero for a mismatch. A face placed a full tier wrong is by definition far from
   most users, so for most users it contributes almost nothing. It only matters for the users it
   is genuinely near — and for them, the error is one comparison out of ~200.
2. **Errors are two-sided and 200 of them average out.** The estimate is a weighted average over
   ~200 comparisons. Some references are too high, some too low; θ̂ is pulled both ways. This is the
   composite view you were reaching for, and it is already what the fit does.
3. **The user's *tier* survives errors that the ordering does not.** You do not need every reference
   correctly placed to know the user beats most of tier 4 and loses to most of tier 6. Tier
   assignment is a much coarser question than fine ranking, which is exactly why we ship tiers.

**What a mis-ranked reference would have to look like to matter:** many faces wrong *in the same
direction*, concentrated in one region of the scale. That is a systematic bias, not noise, and the
fix is a better ranking (more comparisons in that region — the Run 5 buy zone), not hand-editing the
set. Individual faces that look wrong are noise, and hand-removing them substitutes one person's taste
for 95,245 measured votes while removing faces the estimator was already discounting.

**One caveat that is genuinely unresolved.** The ranking's own resolution is ~210 rank places, so a
face that looks "a tier too low" is often *inside* the ranking's stated uncertainty and is not
evidence of an error at all. But it can also be a real error, and we cannot distinguish the two by
looking. If a specific face bothers you, the honest move is to record it and check whether the
off-cohort validation (§7) shows a systematic offset in that region.

### 3c-bis. Where to see and test it

| | |
|---|---|
| Code | `src/faceiq_pref/placement.py` — `place()` returns θ, `se`, percentile, /10, band, tier |
| Inference tab | Upload a photo; selectors for reference ranking, 50 / 200 / all references, agreement level |
| **Browse the whole reference set** | Expander in the same tab: the selected set in θ order, grouped by tier, with a per-tier count against the cohort's own count. Open it once before trusting any placement — it makes the top-tier thinness visible, which is the unbounded-MLE mechanism |
| Which checkpoint | See §3d. Placement error, not val accuracy, is the metric that decides this |

### 3c-ter. Cross-check /10 vs placed /10 — why the Inference tab shows two numbers

They will not match, and the difference is not an error. **Two estimators on two ladders:**

| | cross-check /10 | placed /10 |
|---|---|---|
| estimator | k-NN — mean `scoreOutOf10` of the 5 cohort faces with the nearest comparator score | conditional MLE on the reference set (`fit_theta`) |
| anchor ladder | pre-registered `ANCHORS` (top ≈ 9.0) | `ANCHORS_TOP10` (top = 10.0) |
| uses | the score's *magnitude* | only the *sign* of each comparison |
| gives an se | no | yes |

A few tenths apart is expected and healthy. **More than about a point apart is a signal**: the
placement is fighting its own local neighbourhood, which usually means the photo sits where the
comparator's score ordering and BT's θ ordering genuinely disagree. The placed number is the one to
trust — it is the production path — but a persistent large gap is worth investigating rather than
explaining away.

### 3d. Which checkpoint — `train-v14-panel-ship`, and it is not close

**Decided 2026-08-04 by measurement, not by val accuracy.** `scripts/placement_by_checkpoint.py`
ranks every checkpoint on the question production actually asks: placed on the cohort scale via the
reference set, how far is a held-out face from where the ranking of record puts it? All runs are scored
on the *same* held-out faces (the `val_fraction 0.2` split, which is a subset of the 0.5 split under a
shared seed, so it is genuinely held out for every run).

| checkpoint | median err F / M | p90 | tier exact | Spearman | vs panel majority |
|---|--:|--:|--:|--:|--:|
| **`train-v14-panel-ship`** | **0.34 / 0.28** | 0.93 | 67.2% | 0.926 | **81.7%** |
| **`train-v22-ship-0.2`** | 0.35 / 0.30 | 1.06 | **68.7%** | **0.937** | **81.7%** |
| `train-v19-panel-variance` | 0.47 / 0.39 | 1.20 | 57% | 0.875 | — |
| `train-v13-panel-hard` | 0.44 / 0.42 | 1.28 | 55% | 0.867 | — |
| `train-v12-panel-soft` | 0.45 / 0.43 | 1.27 | 56% | 0.868 | — |
| `train-v17-panel-select` | 0.47 / 0.47 | 1.26 | 57% | 0.869 | — |
| `ensemble-v7-v1` | 0.47 / 0.47 | 1.17 | 61% | 0.898 | — |

**v14 and v22 are interchangeable; use either.** On the target that counts they are a dead tie —
81.7% each against human votes, paired bootstrap over 2,390 pairs giving **+0.00 pts, 95% CI
[−0.96, +0.96]**. They trade small wins on the BT-proxy columns (v22 better on tier accuracy and
Spearman with zero unbounded cases, v14 better on median), and per this document's own standing rule
those differences do not count. v14 stays the dashboard default only because it is the incumbent and
there is no evidence to switch.

**Why v14 wins, and why this was predictable.** It is the only panel run trained at
`val_fraction: 0.2`, so it saw 33,449 training pairs instead of 13,113. Every arm in the v12–v21 series
deliberately ran at 0.5 to hold out enough panel pairs for the human-grounded comparison — which makes
those runs *controlled comparisons of label recipes*, not shippable models. The log said so at the
time; this table is what that warning cashes out to.

**Two consequences.**

- **Use `train-v14-panel-ship` in the Inference tab and for the off-cohort validation.** Set the
  cohort context to the matching run so percentiles come from the same model.
- **The "p90 placement error is 1.33 /10" figure quoted throughout the docs was measured on a
  0.5-split model.** On v14 it is **0.93**. Still wide enough that a point score overclaims, so the
  band/tier argument stands, but the number was pessimistic by ~30%.

**`train-v22-ship-0.2` was that missing run, and it came back null.** v17's label recipe (soft panel
targets, run 4 folded in, human-grounded checkpoint selection) at v14's data volume: identical against
human votes. So **essentially all of v14's advantage over the v12–v21 arms was training-pair count,
not labels.** The label recipe contributed nothing measurable on top.

That is a load-bearing negative. Combined with the reference-θ null (§"Ground truth" below), it means
*additional* panel labels of the kind we have been buying have no demonstrated route to a user's
score — neither through the reference set nor through the comparator's training set. Note the narrow
scope: v14 already contains panel runs 1 and 3, so this tests the *increment* (run 4, soft targets,
better selection), not whether panel labels help at all. Log §5.3.1 established that they do, once.

<details><summary>Reference-count variant, still worth an A/B later</summary>

200 stratified references versus all ~2,866. On cohort faces they land within 0.04 /10 of each other,
so the honest prediction is that they agree; `se(θ)` falls from 0.24 to 0.13, which is already ~5×
below the disagreement band and therefore not the bottleneck. Run it anyway once real uploads exist,
because a material *disagreement* between the two on a given photo is itself a useful signal about
that photo.

</details>

## 4. Percentile → /10 — BUILT, and this is the "anchor curve"

The pre-registered anchor ladder is a **single monotone function from percentile to score**
(`src/faceiq_pref/calibrate.py`), pinned at three points:

| percentile in the cohort | /10 |
|---|--:|
| 50th | **5.0** |
| 90th | **7.0** |
| 99th | **8.0** |

Everything between is interpolated. That function is the *only* thing that decides what a "7" means,
and it applies identically to every face, which is what makes two users comparable.

**"Change the anchor curve" means: move those anchor points.** If the product decision is that the top
of the cohort should reach 9 rather than 8, you change `99th → 9.0` and every score above the 90th
percentile shifts accordingly, consistently, for everyone, in one place. The ordering does not change
at all — only the labels on it.

That is the alternative to hand-labelling reference faces out of 10. Hand-labelling would set the
scale by 200 individual judgments that do not define a function, cannot extend to a face outside the
set, and encode one person's taste. Changing the curve sets it with three numbers that do all of those
things correctly.

Consequence to keep stating either way: **no model can make the top cohort face score above ~9**
under the current ladder, because the ladder says so. "Our best faces only score 7" is an anchor
decision, not a model defect (log §5.6, §5.4).

---

## 5. Bands, not decimals — MEASURED, NOT WIRED

Fitted on run 4's 2,500 uniform pairs, the only pairs no ranking had been fitted on:

```
P(the higher-scoring face wins) = sigmoid(gap / T)        T = 1.920 on the /10 scale
```

**T is a temperature: the score gap at which agreement reaches 73%.** Small T would mean a sharp
scale; ours is blunt, and that bluntness is the finding, not a fitting problem.

| if we want this many people to agree | the two scores must differ by | so the band half-width is |
|---|--:|--:|
| 55% | 0.39 | ±0.20 |
| 60% | 0.78 | ±0.39 |
| **2 in 3** | **1.33** | **±0.67** |
| 3 in 4 | 2.11 | ±1.05 |

**Why half.** Two bands stop overlapping exactly when the two scores differ by the full gap. So
**non-overlapping bands mean at least that many people agree on the ordering; overlapping bands mean
a call the scale cannot make.** Same convention as a confidence interval, except the width comes from
how much *people* disagree rather than from our sample size — which is why it never shrinks with more
data.

**In production this needs no panel.** T was fitted once, offline. At inference the model emits a
score and the band is three lines of arithmetic on top of it.

Both product surfaces are available and they are the same measurement stated two ways:

- **Band:** "You score **5.7 – 7.1**." (a 6.4 at 2-in-3)
- **Sentence:** "You score **6.4**. About **two thirds of people** would place you above someone
  scoring 5.1."

Ship both. The sentence is what makes the band legible.

Two constraints before the first mock:

- **Wider is more honest and gets uncomfortable fast.** At 3-in-4 the band is ±1.05, which around the
  median contains **half the cohort**. There is no width that is both narrow and defensible; choosing
  the agreement level *is* the product decision.
- **Compute the band in percentile space**, then map both edges through the anchor curve. A fixed
  ±0.67 in score space is a much wider slice of the population at 5.0 than at 8.0.

### 5a. Composing the two bands — the arithmetic for one displayed range

Reference-set placement gives `se(theta)`, and the calibration curve gives the disagreement width.
They are different quantities and should be combined in quadrature, not added:

```
half_width_ten = sqrt( estimation_half²  +  ( T · logit(agreement) / 2 )² )
```

With measured values at 200 references, at the median, for a 2-in-3 band:

```
sqrt( 0.14²  +  0.67² )  =  0.68        →  a 6.4 displays as 5.7 – 7.1
```

**The estimation term contributes 0.01 of the 0.68.** That is the whole argument for not chasing
reference-set size, and it is also the reason the band is honest: it is dominated by a property of the
world rather than a property of our sample.

### 5b. Is a 1.33-point band too wide to show? Show tiers instead

A 5.7–7.1 range does look large, and the width is **not** a free parameter — it is set by the
agreement level, which is the one real product dial:

| agreement claimed | band width | a 6.4 displays as | honest reading |
|---|--:|---|---|
| 55% | 0.39 | 6.2 – 6.6 | barely better than a coin flip at the edges |
| 60% | 0.78 | 6.0 – 6.8 | defensible, noticeably tighter |
| **2 in 3** | **1.33** | **5.7 – 7.1** | the natural break point |
| 3 in 4 | 2.11 | 5.3 – 7.5 | contains half the cohort around the median |

So you can show a narrower band, but only by claiming less. There is no width that is both tight and
strong, and pretending otherwise is what the point score was already doing.

**The better resolution is to stop showing a range at all and show a tier.** The ranking supports
~7 distinguishable tiers, and a 2-in-3 band is 1.33 wide — 9 ÷ 1.33 ≈ 7. **Those are the same number
arrived at two independent ways**, which means tiers *are* the band, discretised. A tier label plus
the point score reads as a confident answer while being exactly as honest as the range:

> **7.1 — Tier 6 of 7.** About two thirds of people would place you above someone in Tier 5.

Recommendation: **tier as the headline, point score as the detail, range in a "what does this mean"
expander.** Show all three — the range is what makes the tier legible, and the "% of people would
agree" sentence is what makes the range legible. They are one measurement stated three ways.

### 5c. The tier ladder, defined

**Product decision recorded 2026-08-04: the top of the cohort should reach 10.** The pre-registered
ladder caps at 9.0 and a display variant caps at 9.5, neither of which lets the best face be a 10.
Implemented as `ANCHORS_TOP10` in `src/faceiq_pref/placement.py` — it keeps the two anchors people
already know and stretches only the top percentile, which is where the ceiling actually bound:

| percentile | pre-registered | **top-10 ladder** |
|---|--:|--:|
| 50th | 5.0 | **5.0** (unchanged) |
| 90th | 7.0 | **7.0** (unchanged) |
| 99th | 8.0 | **8.7** |
| 99.9th | 8.5 | **9.4** |
| max | 9.0 | **10.0** |

Two alternatives were computed and rejected: raising *only* the max leaves nobody scoring 9 and
exactly one face at 10, which reads as broken; spreading the whole scale balances tiers slightly
better but moves the 90th anchor to 7.6, a bigger change to a number already published. **Swapping
ladders changes no ordering whatsoever** — only the labels on it.

Tiers are then equal-width on that scale, **1.29 /10 each against a measured resolution limit of
1.33**, so within-tier differences are genuinely below what people agree on:

| tier | /10 range | cohort percentile | share | cumulative |
|---|---|---|--:|---|
| **7** | 8.7 – 10.0 | 99.0 – 100% | 1.0% | top 1% |
| **6** | 7.4 – 8.7 | 93.8 – 99.0% | 5.2% | top 6% |
| **5** | 6.1 – 7.4 | 77.6 – 93.8% | 16.2% | top 22% |
| **4** | 4.9 – 6.1 | 47.0 – 77.6% | 30.6% | top 53% |
| **3** | 3.6 – 4.9 | 17.4 – 47.0% | 29.6% | top 83% |
| **2** | 2.3 – 3.6 | 5.5 – 17.4% | 11.9% | top 95% |
| **1** | 1.0 – 2.3 | 0 – 5.5% | 5.5% | — |

Population per tier is deliberately **not** equal. Equal-population septiles would put tier
boundaries in the middle of the scale closer together than the resolution limit, which claims a
distinction the data cannot support. Unequal tiers with honest boundaries beat balanced tiers with
dishonest ones — and "top 1%" / "top 6%" / "top 22%" is a better product story than "septile 7"
anyway.

**Tiers do not compress as scores rise.** They are equal-width on the /10 scale; it is the
*population* inside them that thins. Worth stating because the opposite is the intuitive reading, and
it changes what the tier means.

### 5c-bis. The user-facing sentence — say the tier, never the bare number

The mechanically obvious sentence is wrong as copy, and it shipped in the dashboard before being
caught. Corrected 2026-08-04 in `agreement_vs_tier_below()`.

| | |
|---|---|
| ❌ | "You score 5.2. About 67% of people would place you above someone **scoring 3.8**." |
| ✅ | "**5.2 — Tier 4 of 7.** About 62% of people would place you above a typical **Tier 3** face. You place above 58% of the cohort." |

Both are arithmetically true. The first reads as though **some raters think this face is a 3.8**,
which is not what the maths says at all — 3.8 is a *different, lower-scoring face*, one resolvable gap
down. For someone clearly above average, naming a number that far below their own is not just
uninformative, it actively undermines the score above it. The tier version carries identical
information and cannot be misread that way.

**Why the percentage varies and should be shown varying.** Tier width is 1.29 /10 and the 2-in-3
resolvable gap is 1.33, so a face in the *middle* of its tier lands at about two thirds against the
tier below by construction. Near the bottom of a tier it is ~62%, near the top ~71%. Showing the
real figure rather than always saying "two thirds" is both more honest and a genuine improvement
incentive: it tells someone mid-tier that they are close to the next one.

**And show the plain percentile too.** "You place above 58% of the cohort" needs no calibration curve,
no temperature and no tier system to be understood, so it is the sentence that survives if anyone
disputes the rest.

### 5c-ter. Should users choose the confidence level? No — but give them a floor claim

The dashboard exposes an agreement slider because it is a research tool. **Do not ship it as a
slider.** A user who can dial confidence will dial it to whichever end flatters them, and a number
that moves when you drag something is not a rating. Fix one default and offer *one* stronger statement
alongside it.

The reason 2-in-3 is the right default is not taste. Convert each agreement level into "how many tiers
down is the face you are being compared against":

| agreement | /10 gap | **tiers down** | half-width | reads as |
|--:|--:|--:|--:|---|
| 55% | 0.39 | 0.30 | ±0.19 | a coin flip dressed up |
| 60% | 0.78 | 0.61 | ±0.39 | half a tier — no natural referent |
| **2 in 3** | **1.33** | **1.04** | **±0.67** | **exactly one tier down** |
| 3 in 4 | 2.11 | 1.64 | ±1.05 | between one and two tiers |
| 80% | 2.66 | 2.07 | ±1.33 | two tiers down |
| 90% | 4.22 | 3.28 | ±2.11 | three tiers down |

Tier width is 9/7 = 1.286 and `T·ln2` = 1.331, a ratio of **1.035**. So at 2-in-3 the comparison is
*one tier down*, to within 3%. That is not a coincidence — it is the same fact that produced seven
tiers from two independent directions — and it makes the default sentence land on the scale's own
natural unit.

**Why "90% of people agree you are above tier X and below tier Y" reads worse than it sounds.** At 90%
the comparison is 3.3 tiers away, so the honest version is "90% of people would place you above a
typical Tier 1 face" — true, unarguable, and unflattering to the point of being insulting. High
confidence forces a distant comparison, and a distant comparison is a claim about a completely
different-looking person. This is the arithmetic reason the confidence level is not a product dial.

**What to ship instead: two statements at fixed levels.**

1. **The default read, at 2-in-3.** "6.4 — Tier 5 of 7. About 62% of people would place you above a
   typical Tier 4 face."
2. **A floor claim, at 90%, framed as a floor rather than a band.** "We are 90% confident you are at
   least Tier 2." One-sided, strong, and it never shrinks the headline. It answers "how sure are you"
   without inviting the user to renegotiate the answer.

### 5c-quater. Tier exemplar faces — good idea, cannot use cohort photos

Showing example faces per tier is the single best thing we could do for legibility: a tier is abstract,
and seven faces make it concrete instantly. **But the reference set cannot be the exemplars.** Cohort
photos are real people's uploads from faceiq-labs — third-party personal data with no consent for
public display, which is why they never enter git and why the panel treated them as sensitive. Showing
them to users is off the table regardless of how useful it would be.

The fix is a clean separation that also happens to remove the objection in §3c:

| | reference set | tier exemplars |
|---|---|---|
| purpose | **measurement** — estimates the user's θ | **illustration** — shows what a tier looks like |
| source | cohort faces, θ known | licensed, synthetic, or consented images |
| count | ~200 per gender | ~1–3 per tier per gender |
| chosen by | tier-stratified random sample | hand-picked for how well they represent the tier |
| if one looks wrong | leave it; it is down-weighted (§3c) | **swap it** — nothing depends on it numerically |
| shown to users | never | yes, that is the point |

Exemplars are scored through the same pipeline so their tier label is real, then hand-audited freely,
because they are cosmetic. That is the one place hand-picking is not only allowed but correct — and it
gives you a legitimate outlet for the judgment you wanted to apply to the reference set.

Sourcing, in preference order: **consented volunteers** (cleanest, and we can capture them under the
same crop); **licensed stock** (fast, but stock photography skews attractive and well-lit, which will
distort the low tiers); **generated faces** (no consent problem at all, and a generator can be steered
per tier, but they must be scored and audited or they will simply encode the generator's bias).

**How a generated face gets its tier.** Exactly the same path a user takes — that is the point, and
it is also the safeguard:

1. Generate a batch, deliberately over-generating (aim for ~30 candidates per tier slot).
2. Run each through the **production pipeline unchanged**: faceiq-labs crop → comparator → 200
   references → MLE → percentile → /10 → tier. Nothing bespoke.
3. Keep the ones whose placement is *confident and central*: **bounded MLE**, small `se(θ)`, and a
   score near the middle of the tier rather than within ~0.2 of an edge. An exemplar sitting on a
   boundary will look wrong to half your users no matter how correct the number is.
4. **Then look at them, and reject freely.** This is the audit step and it is allowed here precisely
   because exemplars are cosmetic. If a face is placed Tier 5 and looks Tier 6, throw it out and use
   the next candidate. Nobody's score changes.

Two failure modes worth naming before you start. **Generator bias**: image models produce a narrow,
symmetric, well-lit aesthetic, so the low tiers will be hard to fill and the high tiers will fill too
easily — over-generate at the bottom, and expect to reach for licensed or consented photos there.
And **the uncanny tell**: a synthetic face that reads as synthetic undermines the tier it is
illustrating, which is a product problem, not a scoring one, and it is caught by looking.

Do **not** hand-assign tiers to exemplars. If an exemplar's displayed tier is chosen by eye rather
than by the model, the ladder stops being a picture of what the system does and becomes a picture of
what we wish it did — and the first user who scores 6.9 next to an exemplar you promoted by hand is
looking at an inconsistency we created.

Ship order: this is a **step 1 companion**, not a later phase. Tiers without exemplars are the weakest
part of the product surface, and this needs no model work.

Two adjustments the code must make rather than using a constant width:

- **Widen at the extremes.** Placement error is 0.41 /10 in the middle 50% and 0.76–0.92 at the ends
  (§3b), so scale the estimation term by the subject band, or simply use the per-user `se(theta)` the
  fit already returns.
- **Compute in percentile space** and map both edges through the anchor curve, since a fixed ±0.67 in
  score space is a much wider slice of the population at 5.0 than at 8.0.

**How many bands?** Two independent measurements say about **seven**. The ranking's own resolution:
median 95% rank interval is 237 of 1,422 places, **0.00%** of adjacent pairs are distinguishable, and
you must move ~210 places for a real difference — so 1,430 ÷ 210 ≈ 7 tiers. And the calibration curve:
a 2-in-3 band is 1.33 points wide, and a 1–10 scale holds about seven of those. Two unrelated routes
to the same number is the strongest argument in the programme for banding.

---

### 5d. Rejected variant: blend the user's score toward their tier's mean

Proposed: once the user lands in a tier, average their score with the scores of everyone in that tier,
weighted toward the user, so a few mis-ranked faces wash out and the answer is a composite rather than
a false precision. **Measured 2026-08-04 and it is a no-op — but for a reason worth keeping.**

On 499 held-out faces with `train-v14`:

| weight on the user | median err | p90 | tier exact |
|--:|--:|--:|--:|
| **100% (current)** | **0.319** | 0.93 | 67.3% |
| 80% | 0.319 | 0.93 | 67.3% |
| 70% | 0.314 | 0.93 | 67.3% |
| 50% | 0.331 | 0.94 | 67.3% |

Nothing moves. Splitting by whether the tier was assigned correctly shows why:

| | median err at 100% | at 70/30 | |
|---|--:|--:|---|
| tier correct (n=336) | 0.226 | **0.192** | better |
| tier wrong (n=163) | 0.646 | **0.719** | worse |

**The blend is a bet on the tier being right.** It pays when it is and loses when it is not, and at 67%
tier accuracy the two almost exactly cancel. It also cannot improve tier accuracy *at all* — 67.3% at
every weight — because averaging toward a tier's own mean never moves anyone out of that tier. And the
noise it was meant to cancel is already in the tier mean, so it imports the mis-ranked faces rather
than washing them out.

**The instinct is right and already implemented.** "Do not over-trust one precise number, get a
composite" is exactly what the MLE does: θ̂ is a weighted average over ~200 comparisons, and the band
states the residual uncertainty. The formal version of the proposal — empirical-Bayes shrinkage toward
the *population* mean rather than the tier mean, which does not depend on the tier being right — was
also computed: at `se(θ) = 0.24` against a population θ variance of 14.7–17.7, it moves θ by **0.3%**.
Correct, principled, and negligible.

**So the way to make tiers look accurate is to make tier assignment accurate**, and the lever there is
the checkpoint: 67% (v14) versus 55–57% (the 0.5-split arms). See §3d.

## 6. Multiple photos → a tighter band — BLOCKED, and worth recording properly

**The idea is sound and the obvious implementation is wrong.** Intersecting the intervals from each
photo is not how independent estimates combine: it is overconfident when photos agree and returns an
**empty** band when they disagree. The correct operation is **inverse-variance weighting**.

```
theta_hat = Σ(theta_i / sigma_i²) / Σ(1 / sigma_i²)        sigma_hat² = 1 / Σ(1 / sigma_i²)
```

Intuition: three people estimate your height — one with a tape (sigma 0.5 cm), one by eye (5 cm), one
glancing from across the room (15 cm). You would not average them equally. Weight by `1/sigma²`
(precision): 4, 0.04, 0.004. The tape gets 100× the weight of the eyeball. Among all unbiased weighted
averages this provably has the smallest variance, and **precisions add**, so n equally good photos
give `sigma/sqrt(n)` — three photos narrow by 42%, not to a point.

**The floor, and why this is blocked.** Photos of the same face are not independent draws. With
correlation `rho` between their errors:

```
sigma_hat² = sigma² · (1 + (n−1)·rho) / n     →     sigma²·rho    as n → ∞
```

More photos cannot beat `sigma·sqrt(rho)`. That floor is the person's irreducible score uncertainty,
and **measuring it needs a test–retest study** — the same person, two photos — which we deprioritised
2026-08-04 because the second views do not exist for every cohort face. So the feature has an unknown
asymptote: at plausible values (sigma 0.45, rho 0.4) three photos narrow the photo term 23% and the
*displayed* band only 6%, because the disagreement term does not move at all.

**When to ship:** after §2–5 are live and a single photo produces a trustworthy band. This is a
multiplier on a working score, not a way to fix a broken one. Live model with sliders in the
dashboard's **Score bands** tab — set sigma to 0.1 to see the case where it is not worth building.

---

## 7. Validating the whole thing on faces from outside the cohort — PHOTOS IN, BLOCKED ON GENDER

`scripts/validate_placement.py` + the **"Validate on unseen faces"** block in the Inference tab.
Built 2026-08-04, smoke-tested end to end.

**Status 2026-08-04 evening: READY TO JUDGE.** Harsh delivered 1,000 labs front photos with a
gender column. After fetching and auditing, two queues are built and waiting:

| set | distinct faces | queue | rejected |
|---|---|---|---|
| `set-1-female` | 82 (70 drawn) | 350 pairs + 35 repeats | 4 cohort re-uploads, 22 repeat identities |
| `set-1-male` | 601 (70 drawn) | 350 pairs + 35 repeats | 109 cohort re-uploads, 180 repeat identities |

**The id-level exclusion list was not enough, and this is the headline.** Zero of the 1,000 face
ids appear in the cohort, so Harsh applied the list correctly — but **113 of the 998 photos (11%)
are cohort people uploading again**, caught only by comparing faces with ArcFace. Ids cannot see
this: the export carries no user id, and a new upload gets a new face id. Had we skipped the audit,
one in nine faces in the test that exists to prove off-cohort generalisation would have been a face
the model trained on, and the result would have been inflated in exactly the flattering direction.

Two supporting findings from the same pass:

- **The `/faces/<id>/` path segment is an account, not a person.** 20 accounts carry both male and
  female photos, so people are scoring their friends. Deduping on it would have thrown away real
  faces; identity clustering is the right tool and it removed 202 repeats.
- **Normalisation parity holds, measured.** The normaliser moves these real uploads **7.85** mean
  abs pixel value against **7.66** for cohort photos, face detected 25/25 on both. Same crop
  pipeline, no preprocessing shift, so §4's normalisation suspect is not live here.

Why 70 faces drawn from 601: at a fixed judging budget it is pairs *per face* that decides whether a
face's position is determined, so 350 pairs over 70 faces (5.0 each) supports the local BT fit while
350 over 601 (0.6 each) would not. The draw is random, not score-ordered, to stay representative.

The threshold is **0.45 cosine**, below the density valley at ~0.47, chosen deliberately
conservative: a false rejection costs one photo out of hundreds we do not need, a false acceptance
contaminates the headline. Both directions were eyeballed — matches at 0.95+ are plainly the same
person, and the 0.50–0.53 band is genuinely ambiguous, which is why the cut sits under it. Note the
cut could *not* be derived from cohort impostor statistics the obvious way, because the cohort turns
out to contain duplicate people and conflicting gender labels on the same person (log §5.9).

**Why this outranks every number in the research log.** Every accuracy figure we have is measured
against BT θ fitted on the same 3,000 faces the comparator trained on. Held-out *pairs* are honest
about new comparisons; they say nothing about new **faces from a different source**, which is all
production ever sees. Different camera, different lighting, different demographics, no VLM label
anywhere in the chain. If placement degrades there, no amount of cohort-internal validation would
have told us.

Your instinct — hand-label ~50–100 unseen faces with a *range* rather than a number — is right, and
it needs one correction plus one addition.

**The correction: judge pairs, not scores.** Absolute labels reintroduce the exact subjectivity the
programme spent $4,155 removing — your "6 to 7" and mine differ by a constant nobody can measure, so
an absolute-label test conflates "the model is wrong" with "we disagree about what a 7 is". Pairwise
judgements need no shared scale, so they test ordering cleanly. Ranges are still worth collecting, but
for a *different* question (below), not as the primary test.

**The addition: repeat some pairs.** Roughly 10% of the queue is re-asked with the sides flipped, and
your agreement with yourself on those is the **ceiling**. If you only agree with yourself 80% of the
time then 80% is the score to beat, and any figure quoted against 100% is quoted against an impossible
target. This is the same correction that moved the panel ceiling from 100% to 74.9% (log §5.8), and
without it this study would have reproduced that mistake.

The three tests are reported separately because they fail for different reasons and have different
fixes — reporting one number for all three is how you spend a month on the model to fix an arithmetic
problem:

| test | needs | a failure means | fix |
|---|---|---|---|
| **1. Ordering** — does the placed score reproduce your pairwise choices, as a share of your own ceiling? | pairs only | the comparator does not generalise off-cohort | model work — and it would be the first evidence that more labels are not the answer |
| **2. Band** — accuracy on pairs closer than 1.33 /10 vs further apart; and does the curve's predicted agreement match observed? | pairs only | `T = 1.920` was fitted on cohort pairs and does not transfer | refit `T` on this data |
| **3. Calibration** — hand ranges, split into a **shift** (everything placed 1.2 low) and a **spread** (each face wrong differently) | ranges | shift → the anchor ladder is wrong; spread → the model is wrong | re-anchor (free) vs retrain (expensive) — which is exactly why they must not be reported as one number |

Test 2 is the one that has never been checked anywhere and is cheapest to check here: the band's whole
claim is that pairs inside 1.33 /10 should be near chance and pairs outside it should be reliable. On
synthetic data the harness returns 64.7% within vs 93.1% beyond, so the test discriminates.

Test 3 also reports **band coverage** — at the 2-in-3 setting roughly two thirds of your ranges should
fall inside the band. Much higher and the band is padding; much lower and it is overclaiming. And it
reports **your own median range width** next to the band width, which is the most direct answer
available to "is a 1.33-point band too wide": if your own honest ranges are 1.2 wide, the band is not
the thing that is imprecise.

**What you actually do**, start to finish, ~40 minutes of your time:

| step | where | what it costs you |
|---|---|---|
| 1. Collect 50–100 photos, one clear front-facing face each, from outside the cohort. ✅ **done** — 644 in `data/validation/set-1/photos/` via `fetch_validation_photos.py`, pending the gender split | your filesystem | the only real work |
| 2. `make-pairs` builds a judging queue and secretly re-asks ~10% of pairs with the sides flipped | terminal | seconds |
| 3. **Judge.** Dashboard → **Inference** tab → scroll to **"Validate on unseen faces"**. Two photos, click the more attractive one, ~300 times. **Do not look at any score while doing this** — that is the whole point | dashboard | ~20 min |
| 4. `score` runs every photo through the production path (normalise → comparator → 200 references → MLE → percentile → /10 → band → tier). It never sees your judgements | terminal | ~1 min |
| 5. `report` compares the two and prints the three verdicts | terminal | instant |

You are not grading faces and you are not checking whether scores "look right". **You are producing an
independent human ordering of faces the system has never seen, so the report can ask whether the
system reproduces it.** Everything else — bands, calibration, the ceiling — is computed from those same
clicks.

```bash
# download, refusing cohort ids and splitting by gender
python scripts/fetch_validation_photos.py --csv ~/Downloads/data-lab-front-photos-1000.csv --set set-1
# drop cohort re-uploads and repeat identities by face, not by id
python scripts/audit_validation_identities.py --set set-1-female --set set-1-male \
    --threshold 0.45 --apply
python scripts/validate_placement.py make-pairs --set set-1-female --pairs 350 \
    --repeat 0.1 --max-photos 70

#   judge in the dashboard: Inference tab -> "Validate on unseen faces"  (~26 min)

python scripts/validate_placement.py score  --set set-1-female --gender female \
    --checkpoint checkpoints/train-v14-panel-ship/best.pt \
    --context artifacts/train-v14-panel-ship
python scripts/validate_placement.py report --set set-1-female
```

Step 3's optional extra: after judging, type a *range* ("6 to 7") for as many photos as you have
patience for. Ranges are what test 3 needs and nothing else uses them, so skip them if you only care
whether the ordering generalises.

Sample size: ~300 pairs puts the ordering accuracy CI at about ±5pp, enough to separate "generalises"
from "degrades" and **not** enough to rank two checkpoints against each other. Do not use it for
checkpoint selection. One caveat to state up front — the raters are you, so this measures agreement
with *one* person's taste; the panel measures the crowd's. A pass here means the system reproduces a
consistent human on unseen faces, which is the claim that is currently untested, not that it
reproduces the population.

---

## The second-opinion question: comparator + deterministic formula

Proposed: score with both, and let the gap between them be the band. **Build the second source, but
do not make it the band.**

**Why not the band.** A band should represent uncertainty; the gap between two models is disagreement
between two biases. If both are wrong in the same direction you get a narrow band around a wrong
answer, which is the worst failure mode available. And §5.6 measured blends: broad ones are *worse*
than the comparator alone.

**Why the objection is weaker than it looks, though — and this is the fair part of the proposal.**
§5.6's blends failed specifically because their components' errors were *correlated*: the comparator,
BT and Labs were all fitted on the same Gemini labels, so they share a blind spot. A landmark-based
EBM would not — it reads geometry, the network reads pixels. Both estimate the same target, so the
errors are not independent, but they are far less correlated than anything blended so far, and that is
exactly the condition under which combination helps.

**So:** build it (two-line exporter change — `Face.frontLandmarks` is already stored on every upload),
score it on the same panel-vote metric so it is directly comparable, and use it three ways:

1. **Disagreement as a flag** — large gaps mark faces to investigate, not a band to display.
2. **Explainability** — per-feature contributions ("your jaw contributes +0.9"), which the network
   cannot give.
3. **Plan B** — a scorer that stands alone if the neural path stalls, benchmarked against the same
   ground truth.

If the two sources are later *measured* to have low error correlation, combining them is back on the
table — as a better point estimate with a narrower `sigma`, feeding §5's band, not as the band itself.

### The Labs composite — measured, and the answer is no

Run 2026-08-04, `scripts/labs_composite_eval.py`. `labsOverallScore` was already on all 3,000 faces,
so this cost nothing but the running. Applied post-hoc — the reference-set MLE runs first and
untouched, then `w · placed + (1−w) · labs`, with Labs rank-normalised onto our /10 first. Geometric
weighting lands within 0.003 of arithmetic; both scores sit in a narrow range, so the choice does not
matter.

| w on comparator | BT median err | BT tier exact | vs panel majority | vs panel votes |
|--:|--:|--:|--:|--:|
| **1.00 (placement only)** | 0.319 | 67.3% | **81.9%** | **70.8%** |
| 0.85 | **0.283** | **69.9%** | 81.7% | 70.8% |
| 0.80 | 0.295 | 69.5% | 81.5% | 70.7% |
| 0.50 | 0.412 | 61.3% | 80.6% | 70.2% |
| 0.00 (Labs only) | 0.652 | 42.9% | 71.7% | 64.6% |

**Read the last two columns, not the first two.** Against BT the blend is a clean win — median error
down 11%, tier accuracy up 2.6 points, bootstrap CI comfortably excluding zero, replicating on a
second checkpoint. Against **real human votes** it is flat to slightly worse at every weight.

The two cannot both be improvements. **BT is a proxy for what people think; the panel votes are the
thing itself.** A change that moves the proxy and not the target is fitting the proxy's error. The
error correlation between the two sources is +0.008 — essentially independent, which is normally
exactly the condition for a blend to pay, and it is why the BT-target result looks so convincing.

This is the confirmation-bias failure mode the programme worried about in the abstract, caught in the
concrete. Had we only run the BT measurement, we would have shipped a "statistically significant
11% improvement" that helps not one user. **New rule: any composite, ensemble or post-hoc adjustment
must be scored against panel votes before it counts.** BT-target numbers are for debugging.

**What this does and does not say about the EBM.** It is weak evidence at most. Labs `overall_score`
is a legacy hand-tuned formula, and a purpose-built energy model on landmark geometry is a different
proposition. But the harness is now built and the bar is explicit: beat 81.9% / 70.8% on panel votes,
not BT. The plumbing in the Inference tab is deliberately generic so the EBM drops into the same slot.

**The rule that does not change:** Labs `overall_score` is never a training target. Held-out
comparison only.

### Distortion-dependent weighting between the two sources

Worth recording because it is the sharpest version of the two-source idea, and it does not have the
"gap as band" problem. A landmark/geometric formula degrades in a *specific, predictable* way that the
network does not share: selfie perspective distortion, extreme yaw and heavy crop all corrupt measured
proportions while leaving the pixels perfectly legible to a CNN. So the two sources have
**condition-dependent** reliability, which is exactly the situation inverse-variance weighting is for:

```
score = ( score_nn / sigma_nn²  +  score_geo / sigma_geo² ) / ( 1/sigma_nn² + 1/sigma_geo² )
```

where `sigma_geo` grows with measured distortion — yaw angle, focal-length estimate, face-to-frame
ratio, all derivable from the landmarks themselves. A well-framed frontal photo might weight
50/50; a close-range selfie might fall to 30% geometric, which was the instinct behind this. That is
the principled form of "weigh it depending on the distortion", and it is the same maths as §6 rather
than a second mechanism.

**Preconditions before building it**, in order: the EBM has to exist and be scored on the panel-vote
metric; its error correlation with the comparator has to be *measured* (if it is high, combining buys
nothing — this is what killed §5.6's blends); and `sigma_geo` needs calibrating against something,
which means checking that predicted-distortion actually tracks geometric error on our own cohort.
Until those three, treat it as a design sketch, not a plan.

---

## Ground truth: what we have and whether to buy more

**What we have is not throwaway, and it is the most durable asset in the programme.** 47,914
machine-labelled pairs plus **95,245 human votes on 10,280 pairs from 963 raters**, validated three
ways: the ranking beats an average person at naming the crowd's choice on a typical pair (81.7% vs
73.2%), agreement rises monotonically with ranking distance across all six bands on data no model was
fitted on, and the /10 calibration curve is monotone out of sample. Every fallback needs it — a
deterministic formula needs it to be benchmarked against, an EBM needs it to fit, the comparator needs
it to train. It survives every downstream choice.

The one qualification, stated precisely: it is accurate **to about 210 rank places, or seven tiers**.
An excellent tier-assignment instrument, a poor fine-ranking one. Build at tier resolution.

**Buying more: yes, but only inside a 10-point gap, and for the ranking rather than the comparator.**

| | |
|---|---|
| Drawn and shipped | `labels/panel-run-5-buyzone/` — 5,000 pairs at 0–10, in `faceiq-rating/data/pairs.json` |
| Cost | **$1,286** — 300 raters × 100 pairs = 6.00 votes/pair |
| Headroom | +6.2 pt (0–2), +6.2 (2–5), +3.9 (5–10) |
| Remaining after | only 10–20: 7,247 pairs, ~$2,100 at 6 votes |
| Cancelled permanently | everything above 20 — headroom is negative, three methods agree |

Fields to fill in `prolific-soft-launch-form.md` **§10**. Expect it to move the *ranking*, not the
comparator: log §5.8 measured the comparator already beating the ranking under a 10-point gap. A
sharper ruler is the point.

---

## How we know it works — four gates, three already passed

"Is it accurate" has no single answer, because there are four different claims and each fails
differently. Written as **pass/fail thresholds set in advance**, because a threshold chosen after
seeing the number is not a test.

**Gate 1 — internal consistency. ✅ PASSED.** Does placement reproduce the ranking of record on faces
the model never trained on? Threshold was Spearman > 0.90 and median error < 0.5 /10. Measured with
`train-v14-panel-ship`: **ρ = 0.926, median 0.34 / 0.28, p90 0.93, tier-exact 67%**. This is a
plumbing test — it proves the estimator is wired correctly, and nothing more. Failing it would have
meant a bug.

**Gate 2 — human agreement on cohort faces. ✅ PASSED, and this is the strongest number we have.**
On run 4's uniform pairs, does the system pick the crowd's winner? The bar is not 100%, it is what a
*person* achieves: one rater against the majority of the others scores **73.2%**. The system scores
**81.9%**. It is already better than an average individual human at predicting what a crowd will
prefer, on pairs drawn from the whole population. That is the claim the product rests on.

**Gate 3 — off-cohort generalisation. ❌ NOT TESTED. The only untested link in the chain.** Every
number above is measured on faces from the same 3,000-face pool. Production only ever sees faces from
elsewhere. §7 is this test. Thresholds, set now:

| | pass | investigate | fail |
|---|---|---|---|
| ordering, as a share of your own repeat-pair ceiling | ≥ 90% | 80–90% | < 80% |
| accuracy on pairs > 1.33 /10 apart | ≥ 75% | 65–75% | < 65% |
| accuracy on pairs < 1.33 /10 apart | 50–65% *(near chance is the correct answer here)* | — | > 75%, which would mean the band is too wide |
| systematic shift, if you collect ranges | < 0.5 /10 | 0.5–1.0 | > 1.0 → re-anchor |

**Gate 4 — the product holds up in front of users. Not yet designed.** Distinct from accuracy, and it
can fail while gates 1–3 all pass. Things to watch once live: the share of uploads returning an
unbounded MLE (expect ~1%, alarm above 5% — it means the reference set does not cover your users);
score drift between the cohort's distribution and your users' (a very different intake population
makes "top 22%" a lie); and repeat uploads of the same person landing in different tiers, which is
the visible face of the test–retest gap in §6.

**The rule that keeps these honest**, learned the hard way from the Labs composite above: **anything
claiming an improvement must be scored against panel votes.** BT-target metrics are for debugging
only. A change that moves the proxy and not the target is fitting the proxy's error, and it will look
statistically significant while doing nothing for anyone.

---

## Ship order

| # | Step | Cost | Blocked on |
|---|---|--:|---|
| ~~1~~ | ~~Comparator arms~~ ❌ **five arms, five ties — closed.** Labels, targets, checkpoint selection, variance head and 4× resolution all land 1.8–2.1 pts behind BT; the 10–20 band never moved off ~55%. **Stop trying to train the global ordering into the comparator** | ~$15 GPU | done 2026-08-04, log §5.8 |
| **1** | **Ship the band / tier** — `T = 1.920`, computed in percentile space, displayed as one of ~7 tiers (§5b), with the default sentence at 2-in-3 and a 90% floor claim (§5c-ter) | free | nothing. The agreement level is **settled at 2-in-3**, because that is exactly one tier down (§5c-ter) — it was the last open product call here |
| **1-a** | **Tier exemplar faces** — 1–3 per tier per gender, from consented / licensed / generated sources, never cohort photos (§5c-quater) | sourcing only | nothing model-side. Tiers without exemplars are the weakest part of the surface, and this is where hand-picking is *correct* |
| ~~2~~ | ~~Normalisation parity audit~~ ✅ **verified 2026-08-04, three-way.** `faceiq-labs/src/lib/utils/mediapipe.ts` `makeHeadshot`, `faceiq-labs/sagemaker/front-model/code/inference.py`, and `faceiq_pref/preprocess.py` all use identical constants (`OUTPUT_SIZE 1024`, `HEAD_FILL 0.65`, `EYE_HEIGHT 0.5`, `HAIR_ALLOW 0.12`), identical MediaPipe indices (33 / 263 / 10 / 152), the same eye-midpoint rotation and the same `/0.55` fallback. Only difference found: the interpolation threshold is `>` in the ML port and `>=` in SageMaker, which differs only at exactly 1:1 scale where both are identity | free | done — **so odd upload scores are not a crop bug.** The remaining check is that the production upload path actually calls one of these three, rather than a fourth implementation |
| 3 | **Wire reference-set placement** (200 refs, tier-stratified) into the inference path. Handle the two behaviours in §3b: widen the band at the extremes using the per-user `se(θ)`, and cap faces with no finite MLE | free | **checkpoint settled: `train-v14-panel-ship`** (§3d). Buys the standard error the band wants, a θ-scale position, and 200 passes instead of 2,866 |
| **3-a** | **`train-v22-ship-0.2`** — v17's label recipe at `val_fraction: 0.2`, i.e. v14's data volume with the better labels | ~$2 GPU | one config file. The only cheap experiment left with obvious upside, and it feeds straight into step 3 |
| 3b | **A/B the 200-reference vs all-2,866 variants** side by side in the dashboard on real uploads (§3d) | free | step 3 |
| **3c** | **Off-cohort validation** (§7) — 50–100 unseen photos, ~300 pairwise judgements, ~20 min of your time. Harness built and smoke-tested; needs photos | free | nothing. **Do this before step 6.** It is the only test of whether any of this survives contact with faces from a different source, and a failure here changes what to build next |
| 4 | Photo QC gate on upload | ~$0.002/image | — |
| 5 | Run 5 — buy the 0–10 zone | $1,286 | nothing; drawn and shipped |
| 6 | Landmarks → EBM second opinion | ~free | two-line exporter change in faceiq-labs. **This is the next substantial build after 1–3**, and the first thing that adds genuinely new information rather than reading existing information better |
| 7 | `sigma_photo` — VLM-derived, since the variance head measured out (§1) | ~$0.002/image | step 6 gives the distortion features it needs |
| 8 | Multi-photo inverse-variance weighting | — | test–retest, deprioritised |
