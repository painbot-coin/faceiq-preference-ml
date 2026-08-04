# Panel pilot — what 667 raters told us

> **Scope: runs 1–3 only** (hard/close pairs, deliberately selected). Three later findings revise how to
> read these numbers: log **§5.6** (per-face uncertainty and calibration), log **§5.7** (binary
> accuracy is saturated; margin recovery is the metric with headroom, and Gemini's confidence is not a
> usable router), and log **§5.8** (run 4).
>
> ⚠️ **Every number in this document is a worst case, and run 4 measured by how much.** These pairs were
> chosen to be near-ties, so they are the hardest that exist. On a **uniform random** draw of 2,500 pairs
> the same ranking scores **81.5%** against a **74.9%** individual-human ceiling — it beats the average
> person on a typical pair — where here it scores 51.5% against 62.7%. Both are correct; they are answers
> to different questions. **A 30-point swing from pair selection alone means no accuracy figure in this
> programme is quotable without stating which pairs it was measured on.** The run-4 summary is below;
> the full version is log §5.8.

**Date:** 2026-08-01 (run 4 addendum 2026-08-04) · **Cost:** $3,186 across two studies (plus ~$106 for a
12-rater soft launch, and $969 for run 4) 
**Data:** 667 Prolific raters, 7,780 pairs, 65,894 votes — 673 were paid, 6 excluded from the fit
**Reproduce:** `scripts/analyze_panel_run.py`, `scripts/panel_run_delta.py`,
`scripts/refit_bt_panel.py` → `artifacts/panel-run-v3/`, `artifacts/bt-refit-v4-panel/`

We paid a demographically representative panel to re-judge face pairs that our Gemini
pipeline had already labelled, to answer one question: **are the VLM labels behind our
ranking good enough to keep using?**

Answer: **yes for easy pairs, no for hard ones.** On pairs where two faces are close,
the machine-built ranking was at a **coin flip** — it genuinely could not tell which
face people prefer. Two rounds of human labelling have taken it to **57.9%**, roughly
half the distance to the ceiling set by how much raters disagree with each other.

The second round was the real test, and it passed: **$1,286 bought +3.2 points, and the
returns had not started to flatten.**

---

## The headline — two studies, measured the hard way

Scored on 1,200 pairs held out **whole**: not one of their votes went into any fit, so a
ranking can only do better on them by having learned better *face* scores. That is the
same mechanism a future study would depend on for pairs we never buy.

| ranking | predicts a real person's choice on a pair it never saw |
|---|--:|
| machine labels only | 51.5% — a coin flip |
| + run 2 ($1,900) | 54.7% |
| **+ run 3 ($1,286)** | **57.9%** |
| ceiling — raters disagree above this | 62.7% |

Run 3's step is **+3.18 points, positive on 5 of 5 random splits (±0.23)**. At **$404
per point** it was *cheaper* per point than run 2's $583, because we aimed it only at
hard pairs.

**Is it worth continuing? Yes, and we can show it rather than assert it.** Feeding a
growing share of run 3's data into the fit:

| human votes in the fit | accuracy on withheld pairs | points per 1,000 votes |
|--:|--:|--:|
| 5,550 | 55.4% | — |
| 11,103 | 56.0% | 0.110 |
| 16,650 | 56.9% | 0.163 |
| 22,204 | **57.9%** | **0.179** |

**The last dollar bought more than the first.** Returns are still accelerating, not
diminishing — the model seems to need a critical mass of human votes per face before the
signal locks in. Our pre-registered stop condition was "less than 1 point per $500"; run
3 delivered **1.24 points per $500**, four times the threshold. So we buy one more
round.

---

## 1. The ranking is validated

For every pair, we asked how often humans picked the face our Bradley-Terry model rates
higher. The relationship is clean and monotone across the whole range:

| gap between the two faces | on our /10 scale | pairs | humans agree |
|---|--:|--:|--:|
| 0–5 percentile points | 0.11 | 1,237 | 51.9% |
| 5–10 | 0.39 | 450 | 53.8% |
| 10–20 | 0.72 | 339 | 60.5% |
| 20–30 | 1.28 | 141 | 66.0% |
| 30–45 | 1.83 | 159 | 71.4% |
| 45–60 | 2.59 | 121 | 76.5% |
| 60–80 | 3.51 | 95 | 89.6% |
| 80–100 | 5.03 | 29 | 93.4% |

Nothing about that curve is guaranteed. A ranking built from machine labels could easily
have been flat. It rises from chance to 93%, which means **the ordering is measuring
something real about human preference.**

## 2. But it has a resolution limit, and it is coarse

Read the top of that table again. Two faces **1 point apart on our /10 scale** are close
to a coin flip to real people — 54%. You need roughly a **2.5-point gap before 3 in 4
people agree.**

This is not a flaw in our model, it is a fact about the task: people simply do not agree
about faces that are similarly attractive. It does mean **sub-point differences in the
score are not human-meaningful** and should not be presented as if they were. Ranking
someone 7.3 rather than 7.1 is precision we cannot defend.

## 3. On the easy pairs, Gemini beats the average person

Restricted to pairs the VLM called clear:

| | agrees with the panel majority |
|---|--:|
| an average individual rater | 74.6% |
| **Gemini** | **81.0%** |
| our BT model | 81.4% |

Gemini is **+6.4 points better than a single human** at predicting what the crowd
thinks. Note what this does and does not say: the crowd is the standard, and Gemini
beats one *member* of the crowd, not the crowd itself. The practical consequence is
still decisive — **paying a person to label a clear pair buys us nothing.** Those
~37,000 labels stay as they are.

## 4. On the hard pairs, humans know something our model does not

This is the finding that changes our roadmap, and it is the opposite of what the
first pass of analysis concluded.

Our model thinks the "close" pairs are near-ties. Humans disagree with each other on
them — which looked like noise. It isn't:

- A pair's human vote share **replicates at 0.66** across two disjoint halves of
  raters (Spearman-Brown). That is a real measurement, not noise.
- **39% of close pairs have 75%+ agreement**, against 15% expected if people were
  guessing. Humans see a clear winner on four in ten pairs our model calls a tie.
- Yet the model's predicted direction on those pairs is **worthless** — correlation
  with the human verdict is +0.02, i.e. zero.

We ruled out the obvious artifacts: left/right position bias is 0.7 points (real but
negligible), and ethnicity accounts for a ~7-point spread against ~20 points of real
per-pair variation. **So there is genuine preference signal on the hard pairs that
our machine labels are blind to.**

## 5. The model was also badly overconfident

Where the old model said one face wins 88% of the time, humans split 58/42. The *ordering*
was right; the *confidence* was inflated several-fold. Anything user-facing built on win
probabilities would have overstated certainty.

---

## 6. We refit the ranking with the panel's votes, and it worked

`scripts/refit_bt_panel.py` → `artifacts/bt-refit-v3-panel/`

Each human vote enters as one observation, so a 7–5 split pulls two faces together and a
12–0 pushes them apart. Because Gemini answers almost deterministically while a crowd
splits, the two sources are given separate sensitivity settings rather than being pooled
naively — otherwise the faces the panel happened to cover would get squashed relative to
the faces it missed.

The fitted numbers say humans are **3–4× less decisive than Gemini** on the same pairs,
consistently across both genders — an independent confirmation of the overconfidence above.

**The honest test.** Fitting on the votes and then scoring against them proves nothing, so
we withheld half the raters, refit on the other half, and scored on the withheld half:

| pairs | before (VLM only) | after (+ panel votes) | gain |
|---|--:|--:|--:|
| **hard / close** | **50.6%** | **58.3%** | **+7.7** |
| medium | 53.7% | 56.8% | +3.1 |
| low-confidence | 56.9% | 59.1% | +2.2 |
| easy / clear | 71.2% | 71.6% | +0.3 |
| all | 57.5% | 61.1% | +3.7 |

Read the first and last rows together. On hard pairs the old ranking was **at chance** —
it had no idea. It now beats chance by 8 points. On easy pairs nothing changed, because
there was nothing to fix. The gain landed exactly where we predicted it would.

The ranking itself barely moved (correlation 0.991 with the previous version, median shift
1.8 percentile points, 97 of 2,866 faces moving more than half a point on the /10 scale),
and it passes every structural gate. **So this is a precision upgrade in the crowded middle,
not a reshuffle** — which is the right outcome, since the old ranking was already validated
at the extremes.

---

## The network now learns from the panel — and beats the labels it used to copy

Everything above is about the **Bradley-Terry ranking**. The product ships a **neural
comparator**, and for a long time we had never measured that network against a human. Its
headline number, 78.4% held-out accuracy, is scored against the export's own labels — which
are Gemini's on ~98% of rows. So "78.4%" means *"it reproduces Gemini well"*, and we now know
Gemini is at chance on close pairs.

`scripts/eval_vs_panel.py` measures the thing we care about, on 1,928 panel pairs where
**both faces are in the network's own validation split**, so nothing it trained on. Scoring
the old checkpoint that way exposed the real problem: it was indistinguishable from Gemini.
`train.py` read `finalOutcome` and nothing else, so the panel's 65,894 votes flowed only into
BT — there was no code path from a human vote into a training batch.

That path now exists (`src/faceiq_pref/panel.py` → `TrainConfig.panel_labels`), and the same
eval on the retrained arms is the first evidence that the label spend reaches the model:

| predictor | agrees with real human votes | vs Gemini labels, paired |
|---|--:|---|
| ceiling — another rater | 59.1% | — |
| **`train-v13` panel hard majority** | **56.4%** | **+2.00 pts, 95% CI [+0.60, +3.41]** |
| `train-v12` panel vote share | 55.8% | +1.41 pts, 95% CI [−0.03, +2.89] |
| BT from Gemini labels only (`v2-qc`) | 55.0% | — |
| the Gemini labels themselves | 54.4% | — |
| `train-v10`, same config, Gemini labels | 54.3% | −0.13 pts, 95% CI [−1.55, +1.31] |

`train-v10` is an exact control: byte-identical config, only the train split's label source
differs. Both panel arms separate from it cleanly (soft +1.54, hard +2.13, both 95% CIs
excluding zero). **The gap between Gemini's labels and the human ceiling was 4.7 points, and
the hard-majority arm closes 43% of it.**

Intervals come from `scripts/compare_panel_evals.py`, which bootstraps over **pairs** rather
than votes. That matters: twelve raters on one pair are not twelve independent observations of
the model's skill, and resampling votes directly would shrink the interval by roughly √12 and
manufacture significance.

**Two results here are worth not overselling.** First, the soft-vs-hard ablation is a tie
(−0.59 pts, 95% CI [−1.47, +0.28]) — the value came from *correcting the winner*, and we have
no evidence the float target's "admit uncertainty" property paid for itself. Prefer the
simpler target until something distinguishes them. Second, the gain is concentrated where we
bought labels, and the network is still slightly *worse* than Gemini on obvious pairs:

| percentile gap | pairs | Gemini | `v10` control | `v13` hard |
|---|--:|--:|--:|--:|
| 0–2 | 383 | 50.7% | 53.1% | 53.3% |
| 2–5 | 510 | 51.4% | 51.6% | **54.7%** |
| 5–10 | 557 | 53.4% | 51.7% | **55.0%** |
| 10–20 | 307 | 53.2% | 54.6% | **55.9%** |
| 20–45 | 109 | 65.9% | 61.0% | 64.0% |
| 45–100 | 62 | 79.7% | 78.6% | 78.6% |

The money was made in the 2–20 band — close but not coin-flip — which is exactly the region
the top-up studies targeted. The two widest bands hold only 109 and 62 pairs, so read them as
"no gain here" rather than as a measured regression.

**Caveat on the absolute numbers.** These arms run at `val_fraction: 0.5`, chosen so that
1,928 panel pairs are fully held out instead of 333. That halves the training data, so 56.4%
is a *controlled comparison*, not the shippable number. Refit the winning label recipe at
`val_fraction: 0.2` before shipping.

## What we do next

**1. Stop buying labels for easy pairs.** Gemini beats an individual human there and costs
a fraction of a cent. Settled.

**1b. What the "gap bands" mean.** A band is how far apart our ranking puts the two faces,
in percentile points within their gender — not a percentage difference in score. Translated
into /10 points and measured against what raters actually did:

| percentile gap | ≈ /10 gap | pairs | humans agree |
|---|--:|--:|--:|
| 0–5 | 0.14 | 959 | 54.3% |
| 5–10 | 0.39 | 672 | 63.7% |
| 10–20 | 0.70 | 681 | 69.4% |
| 20–30 | 1.21 | 241 | 72.2% |
| 30–45 | 1.81 | 185 | 75.8% |
| 45–60 | 2.54 | 116 | 82.3% |
| 60–80 | 3.58 | 100 | 89.9% |
| 80–100 | 5.03 | 26 | 95.8% |

Two readings matter. It is monotone from chance to 96%, which a machine-label-derived
ranking had no obligation to be. And it sets the **resolution limit**: about 2 /10 points
before 3 in 4 people agree, with sub-half-point differences near coin flips. That is a fact
about human perception, not a defect we can spend our way out of — it is the reason to show
bands rather than decimals in the product.

**2. Spend the human budget on the blind spot — wide, not deep.** 24% of our 47,904 pairs
sit within 10 percentile points, the band where the model is still weakest.

We measured how to spend it rather than guessing. Holding out 750 pairs whose votes never
entered any fit, then spending the same ~$700 three different ways:

| design | accuracy on unseen hard pairs |
|---|--:|
| 2,250 pairs × 6 votes | **53.8%** |
| 1,687 pairs × 8 votes | **53.7%** |
| 1,125 pairs × 12 votes | 52.0% |

**Same money, ~1.8 points better from covering more pairs with fewer votes each.** That
kills the "top up to 25 votes" idea we were considering — past about 8 votes, extra votes
on a pair we already have are the worst available use of a dollar.

It also makes the blind spot far cheaper than first thought. At 6 votes per pair the
*entire* sub-10-point band is **$3,680**, not the $15,300 I quoted when assuming 25 votes.
Relabelling everything would be $15,000 and three-quarters of it would buy pairs where
Gemini already matches the crowd.

**✅ Done — that slice was run 3, and it held.** 4,800 new hard pairs at 6 votes, 300
raters, $1,286. The breadth-over-depth call was right and the returns had not flattened
(see the headline above). Two things the run also settled:

* **Gemini's confidence is not a usable trigger.** On these pairs it agrees with the
  crowd 51.3% overall, and *no better when confident*: 53.5% on `high`, 48.2% on
  `medium`. It is confidently wrong. The only reliable way to find pairs worth buying is
  the percentile gap in our own latest ranking.
* **The app's coverage bug is closed.** Every one of the 4,800 pairs got 6 or 7 votes
  (run 2 ranged 8–56). Nothing was over-served, so no money was wasted.

**~~Next: one more round of ~4,800 hard pairs~~ — run 4 was a uniform random draw instead, and it ended
the labelling programme.** The gate above did clear (the retrained arms beat their Gemini labels by 2.0
points against human votes), but before spending on more hard pairs it was worth $969 to find out what a
human vote is worth *per band* on pairs nobody had selected. See the run 4 addendum below.

**3. Retrain the comparator on human targets — ✅ done, and it worked.**
A 4–2 pair should teach the network "these are nearly equal", not "A wins". Three things
had to change in the trainer, none of them large:

* let a label come from panel vote share rather than `finalOutcome`;
* accept a float target (the loss is already `BCEWithLogitsLoss`, which takes soft targets
  natively — the restriction is in how the dataset builds labels, not in the loss);
* ~~stop dropping ties~~ — **this one turned out to be unnecessary.** `skip_ties` only drops the
  113 pairs where *Gemini* said "tie". The 788 pairs where the *crowd* split exactly evenly were
  never dropped, because Gemini named a winner on all of them; they were being trained as
  confident 0/1 labels when the truth is exactly 0.5. Soft targets alone fix them.

**Done 2026-08-01.** Vote share now reaches the trainer via `panel_labels` in `TrainConfig` and
`src/faceiq_pref/panel.py`, applied to the **train split only** so `val_accuracy` stays comparable
across runs. Half the panel targets land between 0.35 and 0.65 — that is the shape of the signal
the export could not carry. Both arms ran against the existing `train-v10` control:
`train-v12-panel-soft` (vote share) and `train-v13-panel-hard` (crowd winner, rounded), which
between them separate "fixing which face" from "admitting how much".

**Outcome: fixing which face is what pays; admitting how much is unproven.** Hard 56.4%, soft
55.8%, control 54.3%, and the two arms are not separable from each other. Only 1,912 of 13,113
train pairs (14.6%) carry a human target and 893 of those flip the export's winner, so a
2-point gain from a 7% correction rate is a large effect per corrected pair — consistent with
the corrections landing precisely where Gemini was at chance. Note also that `val_accuracy`
barely moves (77.5% soft, 77.5% hard, 77.1% control): the val split keeps Gemini's labels, so
it *cannot* see this improvement. That is the clearest illustration of why we needed a
human-grounded eval at all.

The 7,780 panel pairs are also our permanent human-grounded test set, the first yardstick
we have that isn't machine-generated. `scripts/eval_vs_panel.py` reports against it and
refuses to score pairs the checkpoint trained on.

**4. Decide how to display the score.** Given the ~2.5-point resolution limit, showing one
decimal place implies precision we cannot support. Bands or a coarser scale would be honest.
**Run 4 now specifies this exactly**: a 0.25 /10 difference is a coin flip (53.0%), 1 point is 65.4%,
2 points is 82.7%.

**5. No spend on comparison coverage.** Worth stating because we previously flagged it: the
ranking passes its structural gates — one connected component per gender, no face under 15
comparisons, 0.99 stability. The graph was never the bottleneck.

---

## Addendum — run 4, and the three things it changed (2026-08-04)

2,500 pairs drawn **uniformly at random**, 303 raters, 31,237 judgments, **$969**. The point was to stop
measuring ourselves on pairs we had chosen. Full detail in log **§5.8**.

### The ranking's distances are real — this was the biggest open worry

Every spend decision above is cut on percentile gaps from a Bradley-Terry fit over **Gemini's** labels.
The fair objection is that this is circular: if Gemini's ordering is wrong, "close" and "far" are arbitrary
labels and the whole band table is decoration. Run 4 tests it, because the bands were *recorded* rather
than imposed and the human votes are data the ranking has never seen:

| percentile gap | pairs | crowd majority share | two halves of raters agree | **ranking agrees with the crowd** |
|---|--:|--:|--:|--:|
| 0–2 | 68 | 67.8% | 71.2% | **53.2%** |
| 2–5 | 98 | 70.9% | 77.0% | **56.5%** |
| 5–10 | 198 | 71.4% | 76.0% | **57.8%** |
| 10–20 | 479 | 70.4% | 75.1% | **61.9%** |
| 20–45 | 883 | 74.6% | 80.5% | **69.6%** |
| 45–100 | 774 | 84.9% | 93.1% | **83.6%** |
| *pure coin flips would give* | | *61.5%* | *~50%* | *50%* |

**Strictly monotone, all six bands, out of sample.** A circular partition cannot do that. Overall
`spearman(gap, how decisive the crowd is) = +0.408`.

One piece of fine structure matters for planning. **Crowd agreement is a threshold, not a smooth
gradient**: majority share is flat at 70–71% across 2–5, 5–10 and 10–20, then rises. So below about 20
percentile points the crowd is uniformly ~70% cohesive *however close the ranking says the pair is*, while
the ranking's own accuracy still climbs (53.2 → 61.9) — which is precisely why those pairs are worth
buying and the wider ones are not. Do not expect a study on 0–2 pairs to yield *cleaner* labels than one on
10–20 pairs; expect it to yield more *surprising* ones.

### On a typical pair, the rating already beats the average person

Asking *"did it pick the face more raters picked?"* — same predictor, same metric, two pair distributions:

| predictor | on these near-tie pairs | **on a typical pair** |
|---|--:|--:|
| the raw Gemini label | 51.2% | **80.0%** |
| BT ranking (Gemini labels only) | 52.9% | **81.7%** |
| one human vs the crowd (ceiling) | 60.8% | 73.2% |

The ranking is **+8.5 points better than a randomly chosen human** at naming the crowd's choice on a
typical pair. That is the number the product claim rests on, and it is the first one in the programme
that was not measured on hand-picked near-ties.

> ⚠️ **Two accuracies, and mixing them invents a 10-point gap.** The table above scores against the
> *majority label*, where a 9–3 pair is one clean win and 100% is reachable. The comparator numbers below
> score against *individual ballots*, where a perfect predictor on that same 9–3 pair gets only 75%. On
> typical pairs, vote-level, on identical pairs: comparator **67.7%**, Gemini label 68.8%, ranking
> **69.5%**, one-human ceiling **69.9%**. The same ranking is +8.5 above a person on the first metric and
> +0.2 on the second. Full 2 × 2 in `artifacts/panel-run-v4/accuracy-matrix.json`.

### And the money has run out of places to go

Human votes are worth buying where the free label is weak, and nowhere else:

| percentile gap | what a human vote adds over the free Gemini label | verdict |
|---|--:|---|
| 0–2 | **+6.2** [+4.1, +8.3] | buy |
| 2–5 | **+6.2** [+4.4, +7.9] | buy |
| 5–10 | **+3.9** [+2.3, +5.6] | buy |
| 10–20 | **+3.6** [+1.7, +5.4] | buy |
| 20–45 | **−0.9** [−2.1, +0.4] | no |
| 45–100 | **−1.3** [−2.2, −0.5] | **actively worse than free** |

**$14,367 of planned labelling was cancelled on this table**, leaving $7,463 of validated spend inside 20
points. Two other methods reach the same boundary independently: run 4 itself moved the ranking by only
**+0.09 points for $969** ($11,140/point, against $404 for run 3's hard pairs), and the refit's held-out
gains go +8.0 points at 0–2, +0.8 at 10–20, and **exactly +0.0** at 45–100.

### The uncomfortable finding, which is the useful one

**On typical pairs the neural comparator is 1.8–3.1 points *behind* the ranking it was distilled from**
(paired, every interval below zero), losing the 10–45 band by 3–7 points while winning under 10. Since the
ranking recovers that ordering from labels we already own, the missing accuracy is not missing ground
truth. **The bottleneck has moved from data to the model**, and no further labelling addresses it.

That also settles a question about our own process. Run 4 was designed so that three previously favourable
readings *could* come back negative, and two of them did. A programme where every study confirms the last
one is not measuring anything.

## Caveats

- The 2,980 pairs are a stratified sample deliberately overweighted toward hard pairs,
  so the cohort-wide agreement rate is higher than the 67% headline.
- Run 2's coverage came out at 8–56 votes per pair instead of a flat 12 (a concurrency
  bug in the rating app). Fixed before run 3, which landed at a clean 6–7. It widened
  run 2's error bars; it did not bias the results.
- 6 of the 673 paid raters had their votes dropped for chance-level agreement; only 1 was
  recommended for non-payment, since that is the only case with behavioural proof (a
  236 ms median, five times faster than the 1st percentile of the cohort).
  Attention-check failures alone are a poor quality signal — 3 of our 20 checks turned
  out to be genuinely contested pairs and have been replaced, and the correlation
  between check failures and real agreement is only +0.18.
- **Fast clicking is not a data-quality problem**, now measured on 299 raters: the
  fastest third of raters agree with the crowd 61.5% of the time against 62.8% for the
  slowest third. Speed is the instructed behaviour.
- The 62.7% ceiling is estimated from only 5–6 other votes per pair, which understates
  it slightly. "Roughly half the reachable gap closed" is the honest phrasing; the
  literal arithmetic says 56%.
- Two raters volunteered that some faces looked AI-generated, after our QC pass already
  removed 133. One recognised a public figure. Both are open cleanup items.
- Whether the hard-pair signal is "attractiveness" or some shared human bias we
  haven't identified is not settled by this study. It is real and replicable; its
  nature is an open question.
- The +7.7 point gain is measured with **half** the panel's votes, since the other half
  was withheld to test it. The shipped ranking uses all of them, so it should be at least
  this good — but that is inference, not measurement.
- We have **not** validated that the returns hold all the way to 12,000 pairs. We know they
  had not flattened by 2,250. That gap is exactly what the staged $1,000 slice is for, and
  it is why we are not committing the full $3,680 up front.
- ~~We did not download Prolific's rater demographics~~ ✅ pulled for runs 2–3, 667/667 matched. The
  cohort effect is real but small (+2.4 pts for 55+, +1.4 White, +1.0 Female) and confounded with rater
  consistency — see `panel-study-playbook.md` §8.
- **Run 4's addendum numbers rest on a smaller leak-free subset than they look.** The population
  accuracies for the *comparator* come from 631 pairs (7,411 votes) — the ones whose both faces sit in
  that checkpoint's own validation split — not from all 2,500. The ranking and Gemini figures use all
  2,500. Per-band comparator cells at 0–2 and 2–5 hold only 19 and 28 pairs, so read those two rows as
  direction, not magnitude.
- **The comparator-versus-ranking comparison is not symmetric, and the asymmetry favours the ranking.**
  BT was fit on the Gemini labels of these very faces; the comparator was scored only on faces it never
  trained on. That is deliberate, because it is the *production* comparison — BT cannot score a new user's
  photo at all, so the transductive advantage is not something we can ship. But it does mean "the ranking
  beats the network" overstates how much of the gap is a modelling failure.
- **Run 4's pairs are spent.** They were the only pairs no ranking had been fitted on, which is what made
  the calibration curve honest; `bt-refit-v5-panel` has now absorbed them. Re-testing calibration on a
  future ranking needs a fresh uniform draw.
