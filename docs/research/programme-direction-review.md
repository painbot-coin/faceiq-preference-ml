# Programme direction review — is this working, when do we stop, and what is plan B

*2026-08-04, written the day run 4 landed. This is a deliberately adversarial read of the whole
programme: not "how do we keep going" but "should we, and would we know if we shouldn't".*

Every number here is measured and cited to the research log. Where something is an opinion or an
inference I say so.

---

## 0. Bottom line

**The approach works, the ground-truth question is answered, and the bottleneck is no longer data.**

Three things are now established well enough to stop debating:

1. **Pairwise human ground truth was the right instrument.** It found a blind spot in the machine
   labels that nothing else could have found (the VLM is at a literal coin flip on near-ties, log
   §5.5), and it fixed it (+8.0 points of held-out accuracy in the closest band, §5.8).
2. **The ranking is not circular and not decorative.** Bands cut on a VLM-derived ranking predict
   human agreement monotonically across all six bands, 53.2% → 83.6%, measured on votes that
   ranking never saw (§5.8). This was the single most dangerous unexamined assumption in the
   programme and it survived the test.
3. **The labelling budget is spent.** Not "we ran out of money" — **we ran out of pairs worth
   buying.** $14,367 of planned spend was cancelled on measurement, $7,463 of validated spend
   remains, and even that is now second in line.

And one thing has changed direction:

4. **The neural comparator is now the weakest link in the chain, by measurement.** On a typical
   pair it scores 67.7% where the ranking it was distilled from scores 69.5% and a single human
   scores 74.9% (§5.8). The ordering it is missing is *already in the labels we own*. **More ground
   truth cannot fix this**, and buying more would look like progress while changing nothing.

**Directional verdict: keep going, stop labelling, spend the next month on the model and on the two
unmeasured product risks (test–retest, and the band policy).** The thing to be worried about is not
that we are on the wrong track — it is that we have been measuring the track with a metric that
moves 30 points depending on which pairs we look at, and we only just found that out.

---

## 1. What $4,400 actually bought — stated two ways, because one way is misleading

Total programme spend to date: ~$124 of Gemini labelling and **$4,155** of human panel votes across
three studies — run 2 $1,900, run 3 $1,286, run 4 $969 — plus a $106 soft launch and a few dollars
of GPU. 976 raters paid, 963 in the fit.

| | on the pairs we bought (near-ties) | on a **typical** pair |
|---|--:|--:|
| ranking, machine labels only | 51.5% | **70.6%** |
| ranking now (`bt-refit-v5-panel`) | 57.9% | ~71.7% |
| **what the $4,155 bought** | **+6.4 points** | **≈ +1.1 points** |
| individual-human ceiling | 62.7% | 70.4% |

*Both columns are **vote-level** — the share of individual ballots agreeing with the pick — so they are
comparable with each other and with every `panel_run_delta.py` figure in the log. The 81.5%/74.9% pair
quoted elsewhere is the **majority-level** metric, which asks a different question and runs ~10 points
higher; see log §5.8's metric warning. The population gain is the v5 held-out gain per band reweighted
by run 4's band mix, so call it "about a point" rather than 1.1.*

**Read at face value, "$4,155 for one point" looks bad, and that reading is wrong** — for a reason
that is the most useful thing in this document.

### Which pair distribution is the *product's* distribution?

A user's /10 is their percentile in the cohort. To place them at ±0.25 points you have to order
them correctly against everyone within 0.25 points of them — and around the median that is **11% of
the cohort**, roughly 155 faces, every one of which is a near-tie. So:

> **Score precision *is* the near-tie problem.** Population accuracy only protects against gross
> errors. Every point of *resolution* in the number a user sees comes from the hardest pairs that
> exist.

Measured, from run 4's out-of-sample curve (`scripts/band_calibration.py`):

| a /10 gap of | how many people agree with that ordering | share of the cohort inside that band of the median |
|---|--:|--:|
| 0.25 | 53.3% | 11% |
| 0.50 | 56.5% | 23% |
| 1.00 | 62.7% | 48% |
| 2.11 | 75.0% | 50% (half the cohort) |

So the $404/point we paid for near-tie labels is the price of the thing the product sells, and the
$11,140/point run 4 paid for uniform pairs is the price of a thing it does not. **That is the
justification for past spend and simultaneously the reason to stop: we bought the pairs that matter
and we have measured that the remaining ones do not.**

---

## 2. The confirmation-bias audit — the concern was justified, and here is the evidence both ways

The worry, stated plainly: after each study we go looking for a number that says "keep spending",
find one, and mistake it for progress.

**This has partly happened.** The pattern is visible in the log itself. Three times, a headline
metric saturated and was replaced by a new metric that had headroom:

| when | metric declared "the one that matters" | what happened to it |
|---|---|---|
| §5.3 | pairwise accuracy vs the export label | saturated at ~78%, and was measuring agreement with Gemini, not with people |
| §5.5 | pairwise accuracy vs human votes | reached ~96% of its ceiling; §5.7 declared it "has no room left to show progress in" |
| §5.7 | **margin recovery** (% of vote-share ceiling) | declared 38.9% with "2.6× to go" — and on run 4's population sample the *same* ranking sits at **82.3% of ceiling** |

That last row is the clearest instance. §5.7's "the one axis with real headroom left" was a
statement about deliberately-hard pairs that read like a statement about the model. It was not
wrong, it was unqualified — and an unqualified metric that always shows headroom is
indistinguishable from a rationalisation.

**But the process also caught itself, which is the part that matters.** Run 4 was explicitly
designed so that three favourable readings *could* come back negative, with the falsification
condition written down before launch ("if headroom comes back below ~2 points with a tight
interval, the remaining $15.3k is dead money"). Two of the three came back negative:

- the $15.3k was killed, and
- the comparator turned out to be *behind* the ranking on typical pairs, which nobody expected.

A programme where every study confirms the last one is not measuring anything. This one just spent
$969 to cancel $14,367 and to demote its own model. That is the behaviour you want.

### Three rules to keep it honest, all cheap

1. **Freeze the metric suite now and stop adding to it.** Four numbers, every one reported with its
   pair distribution: near-tie accuracy, population accuracy, margin recovery, and per-face
   estimation interval. If a new metric is proposed, it must be justified as *additional*, never as
   a replacement for one that saturated.
2. **State the deliverable and the falsification condition before spending.** Run 4 did this and it
   is the only reason its negative results were legible instead of disappointing. It is now item 0
   in `panel-study-playbook.md` §6.
3. **Never compare two runs' raw accuracies.** `compare_panel_evals.py` bootstraps over pairs; the
   naive comparison overstates confidence badly enough to manufacture results. Several
   "improvements" in the log (v13 vs v15, v12 vs v13) are ties once done properly, and they are
   labelled as ties.

### The dilution worry, quantified

The concern that panel labels are a minority of the training signal is real and now measured:
**19.5% of train pairs carry a human target** with run 4 folded in (up from 14.7%), and 1,035 of
those flip the export's winner. The one experiment on it — `train-v15`, which upweighted human rows
3× — was **inconclusive** (+0.016 margin recovery, CI [−0.005, +0.037]). So "the human signal is
too quiet" remains a plausible but unproven diagnosis, and it is cheap to settle with one more arm
at weight 6.0 once `train-v16` establishes a baseline.

---

## 3. First principles: is the source data right?

The 3,000-face cohort was never audited as a *design choice*, only as a data-quality problem. Here
is the audit.

### What the cohort is good enough for

| question | verdict | evidence |
|---|---|---|
| Can it support a stable ranking? | **Yes, comfortably.** | One connected component per gender, no face under 15 comparisons, subsample stability ρ 0.992 across five refits (§5.1, §5.8) |
| Is the comparison graph dense enough? | **Yes, and saturated.** | `cohort_capacity.py`: accuracy goes as `0.588 − 0.072/√k` in comparisons per face (R² 0.79). 47,914 matchups is k = 33.4 at 57.9%; **doubling to 100k buys +0.4 points**, and infinitely many tops out at **58.8%** — only **+0.9** over what we have, against 5.1 points of remaining headroom to the 70.6% ceiling. More pairs on these faces is a dead end |
| Is it ethnically defensible? | **Adequate, not good.** | 69.2% white, under the ≤80% audit gate. Six groups clear ~50 faces per gender; south_asian female (47) is borderline and the native-american/pacific-islander tail (14) cannot support a quota at all |
| Is the /10 anchoring valid within it? | **Yes by construction, and that is the trap.** | Percentile is uniform within the cohort whatever the mix, so composition cannot inflate the /10. The anchors (50th→5.0, 90th→7.0, 99th→8.0) are a *pre-registered choice*, which is why no model can make the best cohort face exceed ~9 (§5.6) |

**So drawing more or "better" faces is not the lever.** The graph is saturated, the ranking is
stable, and per-face precision is limited by *effective* comparisons (10.1 per face) rather than by
cohort size. Redrawing would cost the entire labelling investment and buy almost nothing measurable.

### What the cohort is structurally incapable of, and this is the real finding

**It is one front-view photo per person, for 3,000 distinct people.** That single design decision
makes three things unmeasurable, and no amount of labelling touches any of them:

1. **Test–retest stability.** Does the same person get the same score from a different photo? This
   is the biggest production trust risk in the programme — a user who re-uploads and moves 1.5
   points stops believing the number — and we have **never measured it**. Fixable for **$0 in
   labels**: the other views (`<sourceFaceId>_<view>.webp`) already exist in Labs blob storage; it
   needs an export and a scoring run.
2. **The band-narrowing feature.** "Upload more photos and the band tightens" is entirely a
   statement about photo-level variance. See §5 — it is the *only* component multiple photos can
   reduce, and it is precisely the one we have not measured. **The feature cannot be designed until
   this export happens.**
3. **Absolute (population) calibration.** A within-cohort percentile equals a population percentile
   only if the pool was drawn from the population. It was drawn from the Labs upload funnel — people
   who chose to submit their face to a rating app. If that pool sits above or below the population,
   every /10 is shifted. This is not fixable from any data we own; it needs an externally-sampled
   reference set, and it is a known, bounded, statable limitation rather than a bug.

Two smaller cleanups the raters found for us, both unresolved: some faces still look AI-generated
after QC removed 133, and at least one cohort member is a recognisable public figure (reputation
contaminates an attractiveness judgement, and it is a privacy question).

> **Verdict on first principles: the initial data was the right shape for the ranking question and
> the wrong shape for the product question.** The fix is not more faces. It is *more photos of the
> same faces*, which is an export, not a study.

---

## 4. When to stop labelling — the answer, with the numbers

**Already reached, for three of the four bands, and here is the rule that produced that.**

Stop condition (pre-registered): a study's marginal gain on withheld pairs falls below **1 point of
accuracy per $500**. Applied per pair type, which is the correction run 4 forced:

| study | pairs bought | spend | marginal gain | points per $500 | verdict |
|---|---|--:|--:|--:|---|
| run 2 | 2,980 mixed | $1,900 | +3.26 | 0.86 | (retrospective) |
| run 3 | 4,800 near-ties | $1,286 | **+3.18 ±0.23** | **1.24** | buy more like this |
| **run 4** | 2,500 **uniform** | $969 | **+0.09 ±0.36** | **0.046** | **stop** |

Run 3 and run 4 were bought in the same month at the same rate and differ by **27×**. A global stop
rule would have averaged them into a number that describes neither. **The rule is per band, and the
band boundary is measured:**

| percentile gap | headroom over the free label | remaining cost | verdict |
|---|--:|--:|---|
| 0–2 | **+6.2** [+4.1, +8.3] | $563 | buy |
| 2–5 | **+6.2** [+4.4, +7.9] | $971 | buy |
| 5–10 | **+3.9** [+2.3, +5.6] | $1,724 | buy |
| 10–20 | **+3.6** [+1.7, +5.4] | $4,205 | buy |
| 20–45 | **−0.9** [−2.1, +0.4] | $8,269 | **no** |
| 45–100 | **−1.3** [−2.2, −0.5] | $6,098 | **actively worse than free** |

Three independent methods put the boundary in the same place (headroom over the label,
`panel_run_delta`, and the refit's held-out gains by band), which is the standard §5.7's
inverted-table incident taught us to demand.

**So the honest answer to "would we know when to stop?" is yes, and we would know it band by band
rather than all at once.** The $20,000 ceiling in the original plan was never the constraint. The
whole validated remaining programme is **$7,463**, and it is currently blocked behind the model.

### The one thing that could reopen the buy zone

If `train-v16` shows the comparator *can* use wide-band labels once it stops losing that band, the
20–45 verdict does not change (headroom is over the *label*, not over the model). But if the
comparator's 0–10 advantage grows with more near-tie labels, the $1,534 for the 0–2 and 2–5 bands
becomes the cheapest measurable improvement available. **That is the next spend, if any, and it is
1/13th of what we thought we were committing to.**

---

## 5. The band paradigm — the maths, and which parts we can actually do

This is the right direction, and the version described in your notes has one mathematical error and
one hidden dependency. Both are fixable.

### There are three different bands, and they behave completely differently

| band | what it means | does it shrink? | status |
|---|---|---|---|
| **Estimation** | how well we know *this face's* θ from the comparisons we have | Yes, as 1/√(comparisons) | **Measured**: median 95% interval **0.91 /10 points**, 241 rank places (`bt_uncertainty.py`, §5.6) |
| **Disagreement** | people genuinely differ about the same two faces | **No. Never.** It is a property of the population | **Measured out-of-sample** (`band_calibration.py`, below) |
| **Photo** | how much the score moves between two photos of the same person | Yes, as 1/√(photos) — *down to a floor* | **Unmeasured, and unmeasurable from this export** |

Conflating these is what makes band discussions go wrong. The estimation band is about our data; the
disagreement band is about the world; the photo band is about the user's upload. Only the third one
is what "upload more photos" touches.

### How many bands the ranking actually supports: about seven

Two independent limits, measured (`scripts/rank_resolution.py`, log §5.8):

| limit | what it says | buyable? |
|---|---|---|
| **estimation** | median 95% rank interval is **237 places of 1,422**; **0.00%** of adjacent pairs are distinguishable; you must move **~210 places** for a real difference | partly — SE falls as 1/√k, so tripling coverage gets 237 → ~140 |
| **crowd** | `T = 1.920`: faces within **0.39 /10 points** are a coin flip to real people | **no, at any price** |

1,430 faces ÷ 210 places ≈ **7 tiers per gender.** That is what the evidence supports, and it lands in
almost the same place as the calibration curve: at 2-in-3 agreement the band is 1.33 points wide, and a
1–10 scale holds about seven of those. **Two unrelated measurements converging on ~7 bands is the
strongest argument in this document for the band product.**

It is also the answer to "why does a face sit next to faces from a different tier?" — the sort prints a
total order the evidence does not support. That is the expected appearance of an honest fit, not a defect.

### The disagreement band, measured (this is publishable today)

Fitting `P(the higher score wins) = sigmoid(Δ/10 gap / T)` on run 4's 2,500 pairs — the only pairs
no ranking had been fitted on — gives **T = 1.920** on the /10 scale
(`artifacts/panel-run-v4/band-calibration.json`):

| a /10 gap of | fraction of people who agree | inverse: agreement level | gap needed |
|---|--:|---|--:|
| 0.10 | 51.3% | 55% | 0.39 |
| 0.25 | 53.3% | 60% | 0.78 |
| 0.50 | 56.5% | **2 in 3** | **1.33** |
| 1.00 | 62.7% | **3 in 4** | **2.11** |
| 2.00 | 73.9% | 80% | 2.66 |
| 3.00 | 82.7% | 90% | 4.22 |

**The product statement this licenses**, and it is far better than a decimal because it is true:
*"You score 6.4. Two out of three people would place you above someone at 5.1."* That needs no new
data, no new model, and no new labels.

**And the band that goes with it: half-width = `T · logit(agreement) / 2`.** At 2-in-3 that is
±0.67, so a 6.4 displays as **5.7–7.1**. The halving is what makes it a *band* rather than a
sentence: two bands stop overlapping exactly when the two scores differ by the full 1.33 points, so
**non-overlapping bands mean at least two thirds of people agree on the ordering, and overlapping
bands mean a call the scale cannot make.** Same convention as a confidence interval, except the
width comes from how much *people* disagree rather than from our sample size — which is why it does
not shrink as we collect more data.

Two consequences worth designing for before the first mock:

- **Wider bands are more honest, and they get uncomfortable fast.** At 3-in-4 agreement the band is
  ±1.05, which around the median contains **half the cohort**. There is no width that is both
  narrow and defensible; the choice of agreement level *is* the product decision.
- **The band must be asymmetric near the ends of the scale.** The /10 is a percentile ladder
  (§5.6), so a fixed ±0.67 in score space is a much wider slice of the population at 5.0 than at
  8.0. Compute the band in percentile space and map both edges through the anchor curve, rather
  than adding and subtracting in /10 points as the illustration above does.

### The multi-photo consolidation idea — right instinct, wrong rule

The example in your notes — bands 7.0–7.5, then 7.2–7.8, then 7.3–7.9, **intersecting** to pinpoint
7.6 — is not how independent estimates combine, and using intersection will fail in two directions:

- when the photos agree it produces an interval that is **too narrow** (overconfident), because
  intersecting three 95% intervals is not a 95% interval;
- when the photos disagree it produces an **empty** interval, which will happen often at any real
  photo-level variance.

The correct operation is inverse-variance weighting (equivalently, a Gaussian posterior update):

```
θ̂  = Σ(θᵢ / σᵢ²) / Σ(1 / σᵢ²)
σ̂² = 1 / Σ(1 / σᵢ²)
```

With `n` photos of equal precision `σ` this gives `σ̂ = σ/√n` — so three photos narrow the band by
**42%**, not to a pinpoint. And photos of the same face are *not* independent draws, so with
correlation `ρ` between their errors:

```
σ̂² = σ² · (1 + (n−1)ρ) / n     →     σ²ρ  as n → ∞
```

**That floor, `σ√ρ`, is the person's irreducible score uncertainty, and measuring it is exactly what
the test–retest export gives you.** Until then the feature has an unknown asymptote: it might narrow
a ±0.45 band to ±0.26 with three photos, or it might narrow it to ±0.42 and stop. Both are
consistent with everything we currently know.

Two more constraints worth designing around now:

- **The disagreement band does not shrink with photos and must be shown separately**, or the product
  will eventually claim a precision no amount of uploading can earn. Total width is
  `√(σ_est² + σ_photo²/n + σ_taste²)`, and `σ_taste` is the term that dominates:
  T = 1.92 means ±0.4 points is already only a 60%-agreement claim.
- **Photo quality should modulate `σ_photo`, not the score.** That is a heteroscedastic head
  (predict `μ` and `log σ²` per image), and **it is already fully implemented and has never been
  run**: `variance_head` scores each face as `N(μ, σ²)` with pairwise logit
  `(μ_A − μ_B)/√(1 + var_A + var_B)`, wired through `model.py`, `train.py` and `eval.py`, with a
  config sitting unused at `configs/train-v11-arcface-variance.yaml`. That config predates the panel
  labels, so it needs `panel_labels` adding — but this is a one-line edit and a training run, not a
  project. It is the clean version of "a margin of error based on photo quality", and it is the
  cheapest unexplored idea in the programme.

### Can the neural net produce a band? Yes — three ways, cheapest first

1. **Calibration band (free, today).** The net outputs a score; the band comes off the T = 1.920
   curve above. No model change. This is the disagreement band and it is the one users should see.
2. **Ensemble band (cheap, this week).** We already have four checkpoints. Their per-face score
   spread is a usable epistemic band, and their pairwise disagreement rate is measurable in an hour.
3. **Heteroscedastic head (one training run, code already written).** `variance_head: true` +
   `panel_labels` on `configs/train-v11-arcface-variance.yaml`. This is the only version that makes σ
   depend on *this photo*, which is what the multi-photo feature needs — and it has been sitting
   implemented and unrun since before the panel existed.

### Can Bradley-Terry produce bands from panel votes? Yes, and better than from VLM labels

A vote share is a direct estimate of `p` with known binomial variance, which the VLM's single binary
label never was. Two levels:

1. **Closed-form per-face standard error** — for face *i*, the observed Fisher information is
   `I(θᵢ) = Σⱼ βₛ² · pᵢⱼ(1 − pᵢⱼ)`, so `se(θᵢ) = 1/√I(θᵢ)`. Cheaper than the current bootstrap and
   **more correct at the extremes**: it returns an infinite interval for an undefeated face, which
   is the truth (§5.6 records that the bootstrap gave those 44 faces *narrow* intervals at an
   arbitrary θ — the opposite of the truth, because a perfect record survives every resample).
   Recommended replacement for `bt_uncertainty.py`'s row-resample.
2. **Crowd band, not estimation band** — for a face at percentile *q*, read off the fraction of the
   population that would place it above a face at percentile *q′*. That is the calibration curve
   again, and it is what turns a rank into a sentence a user can trust.

> **Recommendation: ship the disagreement band now, from the measured curve; treat the multi-photo
> narrowing as blocked on the test–retest export; and implement the closed-form BT standard error
> when convenient.** Do not build interval intersection.

**All of this is now in the dashboard** — the **Score bands** tab has the fitted curve against the
raw observed points, a slider that turns a score plus an agreement level into the band and the
sentence, and a live model of the multi-photo maths where `σ_photo` and `ρ` are sliders precisely
because they are the two numbers we do not know. Slide `σ_photo` to 0.1 to see the case where the
feature is not worth building.

---

## 6. If the approach fails — the actual fallbacks, ranked

Worth writing down, but note the framing has changed: **the ground-truth approach did not fail.** It
delivered a validated ranking that beats an individual human on a typical pair. The open risk is
narrower — whether the *neural comparator* can carry that ranking to unseen faces.

| # | If this fails… | Fallback | Honest assessment |
|---|---|---|---|
| 1 | The comparator cannot close the 10–45 band | **Gap-routed ensemble**, not a blend | **New and untested, and the first evidence for it just arrived.** §5.6 measured *global* blends as worse, but run 4 shows the comparator wins under a 10-point gap and the ranking wins 10–45. Routing on predicted gap is a different operation from averaging, and it is measurable with data we already have. **Try this before anything else in this table.** |
| 2 | The comparator plateaus below the ranking everywhere | Bigger backbone / higher resolution / longer schedule | Cheap and unexplored. Current arms are ArcFace-R50 at **112 px** for 4–8 epochs at lr 1e-5. For a task whose signal is fine facial structure, 112 px is a plausible ceiling all by itself. Test before concluding anything about data. |
| 3 | Neural approach fails entirely | **Landmarks → EBM on feature *differences*** | Genuinely viable, and cheaper than it looks: Labs already computes and stores `Face.frontLandmarks` / `mediapipeLandmarks` on **every** upload, so there is no extraction project and no inference-path gap. Formulate it as `P(A beats B)` from feature differences, not a score from absolutes, so it is scored on the same panel-vote metric. Gives the per-feature explanations ("your jaw contributes +0.9") the product wants. |
| 4 | Everything model-side fails | **Deterministic formula, benchmarked against our ground truth** | This is the answer to "there would be nothing to benchmark it against". There would be: 95,245 human votes on 10,280 pairs is a permanent, model-independent test set. **The ground-truth work retains its full value under every fallback** — that is the strongest argument for having done it, and it is why it was the right first investment rather than the last. |
| — | *Rejected:* 300 celebrity training faces + 200 held out | — | Do not. Three faults: 300 identities will overfit identity rather than attractiveness; celebrity photos carry production-value and fame confounds (the exact reason two candidate worked examples were rejected in the runbook, where a red-carpet photo beat an amateur selfie); and 500 faces cannot support a percentile ladder. It also throws away the 47,914-pair graph we already own. |

---

## 7. What I would actually do next, in order

| # | Action | Cost | Why here |
|---|---|--:|---|
| ~~1~~ | ~~`train-v16-panel-run4`~~ ✅ **done 2026-08-04, and it settled the question by failing.** 67.43%, a tie with v12, 10–20 band unmoved | $2 GPU | Panel coverage of the train split went 14.7% → 19.4% with the new pairs aimed at the weak band, and the weak band did not move. The 10–45 deficit is now *measured* to be extraction, not data. |
| 1 | 🟡 **Three arms, running now.** `train-v17-panel-select` (checkpoint selection fixed — see below), `train-v19-panel-variance` (the never-run variance head, with panel labels), `train-v18-arcface-224` (resolution) | ~$10 GPU | The ranking recovers the 10–45 ordering from the same labels, so the information is provably present. These are the three cheapest explanations for why the network is not extracting it. |
| 2 | **Reference-set inference** — score a new face by comparing it against a fixed panel of cohort faces, rather than reading the raw scalar | free | The comparator was *trained* pairwise and is currently *read* as an absolute score. This is the most likely explanation for "the ranking looks wrong when I upload a face" and it needs no training. |
| 3 | **Gap-routed ensemble**, measured on the 631 leak-free run-4 pairs | free | Uses data we already have. If routing recovers the ranking's 10–45 accuracy while keeping the comparator's 0–10 edge, the product problem is solved without a training run. |
| 4 | **Ship the disagreement band** from the measured T = 1.920 curve; retire the two-decimal score | free | Fully specified by §5.8. Removes the "our best faces only score 7" and "6.00 vs 6.05" problems in one move, honestly. |
| 5 | **Landmarks in the exporter** (two lines) → EBM on feature differences | ~free | Unlocks explainability and fallback 4 at almost no cost — and gives the second opinion a two-source band would need. |
| 6 | The **$1,534** 0–2 and 2–5 tranche | $1,534 | Only after 1–3. Still the highest-headroom labels available (+6.2 pts), and the right buy **if the goal is a better ranking** (which is our ground truth) rather than a better comparator. |
| ~~7~~ | ~~Test–retest export~~ | ~~$0~~ | **Deprioritised 2026-08-04 (product call)** — the second views do not exist for every face and the overhead is not worth it now. Accept knowingly that the multi-photo band-narrowing feature stays unbuildable until this exists. |
| ~~8~~ | ~~$15.3k of wide-band labels~~ | ~~$15,328~~ | **Cancelled.** Negative headroom, three methods agreeing. |

**Total to answer every open question above: under $1,600, most of it optional.** The programme's
expensive phase is over, and it ended because it succeeded, not because it stalled.

### Checkpoint selection — fixed 2026-08-04, and it was a real bug

`train.py` chose `best.pt` on `val_accuracy`, which is scored against the export's **Gemini** labels.
Log §5.3.1 had already measured that a run can gain 2 points against real people while `val_accuracy`
moves 0.4 — so the selection metric was structurally blind to the improvement every panel run exists to
produce, and v16's top four epochs sat within 0.6 points of each other on it.

`train()` now computes **`panel_val_accuracy`** each epoch — the share of real human votes agreeing with
the model, over val-split pairs the panel covered — and selects on it. Verified two ways: it reproduces
`eval_vs_panel.py` to within 0.02 points on the same checkpoint, and a smoke run caught the two metrics
disagreeing in the expected direction (epoch 2 higher on `val_accuracy`, lower on human votes). Every arm
before v17 shares the old flaw, so past comparisons remain fair among themselves but may all be understated.

---

## 8. The three things most likely to be wrong in this document

Stated so they can be checked rather than discovered later.

1. **"About a point on population" understates the value of the panel labels if the product's pair
   distribution is even harder than run 3's.** §1 argues the product distribution is near-tie
   dominated, which makes the $404/point figure the relevant one. That argument is *derived* from
   the density of the /10 scale (11% of the cohort within ±0.25 points), not measured end to end. A
   direct test exists: measure score-assignment error for held-out faces as a function of how many
   near-neighbour comparisons they have.
2. **The comparator-vs-ranking comparison is not symmetric, and the asymmetry flatters the ranking.**
   BT was fit on these faces' VLM labels; the comparator was scored only on faces it never trained
   on. That is deliberate — it is the production comparison, since BT cannot score a new upload at
   all — but it means "the ranking beats the network by 1.8 points" overstates how much of that gap
   is a modelling failure rather than a fitting advantage. The clean version is to fit BT on a face
   subset too, and it has not been done.
3. **Run 4's per-band comparator cells are thin.** The 0–2 and 2–5 rows rest on 19 and 28 pairs.
   Direction is trustworthy; magnitudes in those two rows are not.
