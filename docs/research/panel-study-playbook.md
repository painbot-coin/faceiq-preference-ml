# Panel study playbook — how we decide to spend, and how we know it worked

The repeatable loop for buying human labels. Written after run 2 (**372** raters, 2,980 pairs,
$1,900), **validated by run 3** (**301** raters, 4,800 pairs, $1,286), and **bounded by run 4**
(**303** raters, 2,500 uniform pairs, $969), which found the edge of the buy zone. Read this before
proposing another study.

> **Headline as of 2026-08-04: the buy zone has a hard outer edge and we have reached it.** Human votes
> are worth buying inside a 20-point percentile gap (+3.6 to +6.2 points of headroom over the free label)
> and are worth *nothing or less* outside it (−0.9 and −1.3). **$14,367 of planned spend was cancelled.**
> Remaining validated spend is **$7,463**, and even that is currently blocked by §2b — the neural
> comparator is behind the ranking on typical pairs, so the next dollar belongs to training, not labels.

*Rater counts are submitted sessions, which equal distinct Prolific IDs in every run — the
exclusion groups worked, so no one rated in two. 976 paid across the three, 963 in the v5 fit
after exclusions. Earlier drafts said "360" and "300"; those were the study targets.*

**Unit economics:** run 2 $1,900 / 36,290 votes = $0.052/vote; run 3 $1,286 / 29,924 votes
= **$0.043/vote**; run 4 $969 / 29,351 votes = **$0.033/vote**, ~$3.20 per rater for ~105 pairs. All
filled in hours, so throughput is not a constraint — money is.

**Where we are after three studies** (measured on pairs held out whole, `panel_run_delta.py`, on the hard
pairs runs 2–3 targeted): machine labels alone 51.5% → run 2 54.7% → run 3 **57.9%**, against a 62.7%
human ceiling. About half the reachable gap is closed, at $404/point on hard pairs.

**And on a *typical* pair the ranking was already better than a person before any of that** — 81.5%
against a 74.9% individual-human ceiling (log §5.8). Both statements are true. Which one to quote depends
entirely on which pairs you mean, and the difference is 30 points.

---

## 1. The decision rule

Do not buy labels because more data sounds better. Buy them only where a **measured**
curve says the next dollar still moves accuracy. Run 2 gave us three findings that
together fix the policy:

| finding | consequence |
|---|---|
| Gemini beats an individual human on clear pairs (81.0% vs 74.6%) | never pay for clear pairs |
| VLM-only ranking was at **chance** (50.6%) on close pairs | close pairs are the only blind spot worth money |
| Breadth beats depth per dollar (below) | ~6–8 votes/pair, spread wide — **not** 12–25 |

Run 3 added a fourth, and it is the one that changes how a study is *targeted*:

| finding | consequence |
|---|---|
| Gemini's confidence does not predict its accuracy on hard pairs — 51.3% overall, 53.5% when `high`, 48.2% when `medium` | **never trigger a buy on VLM confidence.** The percentile gap in the newest refit is the only usable selector |

**The breadth-vs-depth measurement** (`scripts/panel_value_curve.py`, evaluated on 750
pairs whose votes never entered any fit) — same ~$700 spent three ways, 5 seeds:

| design | accuracy on unseen hard pairs |
|---|--:|
| 2,250 pairs × 6 votes | **53.80% ±0.38** |
| 1,687 pairs × 8 votes | **53.69% ±0.85** |
| 1,125 pairs × 12 votes | 52.04% ±1.65 |

Same money, ~1.8 points better from spreading it wider. This **retires the earlier
"top up to 25 votes" plan** — depth past ~8 votes is the worst available use of budget
for improving the ranking. Keep 12+ votes only where you need a *per-pair* label good
enough to be a test-set target (we already have that: run 2's 2,980 pairs).

**Stop condition.** Run `panel_run_delta.py` after every study — it is the cleaner test,
because it withholds whole *pairs* rather than raters, so the measured gain has to travel
through better face scores. Stop buying when a study's marginal gain falls below
**1 point of held-out accuracy per $500**.

| study | pairs bought | spend | marginal gain | points per $500 | verdict |
|---|---|--:|--:|--:|---|
| run 2 | 2,980 mixed strata | $1,900 | +3.26 | 0.86 | (measured retrospectively) |
| **run 3** | 4,800 hard (0–10 gap) | **$1,286** | **+3.18 ±0.23** | **1.24** | **buy more like this** |
| **run 4** | 2,500 **uniform** | **$969** | **+0.09 ±0.36** | **0.046** | **stop buying this kind** |

**Run 4 is the stop condition firing, and reading it correctly matters.** It returned 0.046 points per
$500 against a 1.0 threshold — 27× worse than run 3 and **$11,140 per point** — with a completely *flat*
dose curve (25% → 100% of its pool moved −0.02, +0.11, −0.00) where run 3's was accelerating. That is not
"labelling has stopped working". It is "labelling **the pairs run 4 drew** has stopped working", and 66% of
run 4 was deliberately in the skip zone. The stop rule must therefore always be applied **per pair type**,
never globally: run 3's hard pairs and run 4's uniform pairs were bought in the same month at the same rate
and differ by a factor of 27.

**Run 4 was still worth its money**, and the distinction is how future measurement studies get justified:
it was bought to *measure*, not to *improve*. It cancelled $14,367 of planned spend, produced the only
non-circular calibration curve the programme will get, and delivered the first population accuracy figure.
Judging it by `panel_run_delta.py` alone would be judging it against a goal it never had. **State a
study's deliverable before it launches, and score it on that.**

Run 3's own dose curve is *accelerating*, not flattening — 0.110 → 0.163 → **0.179**
points per 1,000 votes across its four quarters. The plausible reason: BT needs a critical
mass of human observations per face before human-informed theta stabilises, so early votes
underpay and later ones compound. Practical implication: **do not judge a study by its
first quarter, and do not split a planned buy into slices so thin that none of them
reaches critical mass.**

`panel_value_curve.py` still answers a different and narrower question — breadth vs depth
*shape* — and only run 2 has enough votes per pair to run it. Do not re-run it on a
6-vote study.

## 2. Which matchups to buy

Rank every export pair by `|percentile_A − percentile_B|` from the **newest** refit and
take from the bottom. Against `bt-refit-v3-panel` over 47,904 scored pairs:

| gap | pairs | share | at 6 votes |
|---|--:|--:|--:|
| 0–5 pts | 6,205 | 13.0% | $1,949 |
| 5–10 pts | 5,507 | 11.5% | $1,730 |
| 10–20 pts | 9,126 | 19.1% | $2,867 |
| 20+ pts | 27,066 | 56.5% | *don't* |

The whole sub-10-point blind spot at 6 votes is **$3,680** — not the $15,300 I first
quoted, which assumed 25 votes/pair. Exclude the 2,980 pairs run 2 already covered:
breadth means *new* pairs.

Two constraints carried over from run 2's draw (`scripts/select_panel_pairs.py`): keep the
ethnicity floor, and exclude QC-failed faces. Both are already in that script.

### 2a. Why the percentile gap, and not Gemini's confidence

The obvious cheaper idea is to let the labeler flag its own hard cases. **Measured, and it does not
work** (log §5.7, `scripts/label_information.py`). Across 7,780 panel-covered pairs, crowd decisiveness
(mean `|vote share − 0.5|`) is flat against the confidence tier — 0.213 high, 0.203 medium, 0.185 low —
and **66.5% of the very closest band (0–2 pts) is labelled "high" confidence**. Gemini does not know when
it is guessing, so its confidence cannot target a study. The same result rules out re-prompting for a
graded label or a rating band such as "5.8–6.3": the blind spot is in the discrimination, not in the
output format, so a finer format inherits it. Confidence is still fine for *filtering* (§4.3's audit
rate), just never for *selection*.

The percentile gap, by contrast, separates cleanly, which is why the rule above is the rule. **Cut the
bands on the VLM-only ranking (`bt-refit-v2-qc`), never on a panel-fitted one** — see the warning below.

**Refreshed 2026-08-04 with run 4's unbiased pairs** (log §5.8). Sample sizes in the two wide bands went
from 389 and 262 *selected* pairs to 1,272 and 1,036 including a uniform subsample, and the verdict changed
from "unresolved" to a hard **no**:

| pct gap | pairs | bought | remaining | cost @12 | VLM vs votes | 1 rater (ceiling) | headroom (95% CI) | verdict |
|---|--:|--:|--:|--:|--:|--:|--:|---|
| 0–2 | 2,663 | 1,692 | 971 | **$563** | 50.2% | 56.4% | **+6.2** [+4.1, +8.3] | **BUY** |
| 2–5 | 3,848 | 2,175 | 1,673 | **$971** | 51.6% | 57.7% | **+6.2** [+4.4, +7.9] | **BUY** |
| 5–10 | 5,387 | 2,416 | 2,971 | **$1,724** | 53.2% | 57.1% | **+3.9** [+2.3, +5.6] | **BUY** |
| 10–20 | 8,936 | 1,689 | 7,247 | **$4,205** | 54.8% | 58.5% | **+3.6** [+1.7, +5.4] | **BUY** |
| 20–45 | 15,524 | 1,272 | 14,252 | $8,269 | 68.4% | 67.5% | **−0.9** [−2.1, +0.4] | **do not buy** |
| 45–100 | 11,546 | 1,036 | 10,510 | $6,098 | 83.3% | 81.9% | **−1.3** [−2.2, −0.5] | **SKIP** |

**Headroom** is the column that decides spend: the crowd ceiling (one rater vs the majority of the others)
minus what the free VLM label already scores. Buy where the whole interval clears zero.

- **Buy zone, 0–20 gap:** **$7,463** finishes all four bands (12,862 unlabelled pairs). All four intervals
  are entirely above zero, so this is validated spend — but see §2b, it is currently *deferred* behind
  fixing the comparator.
- **Above a 20-point gap: never. This is closed, not deferred.** The 45–100 interval is entirely *below*
  zero, meaning a purchased vote is worse than the free label: one rater agrees with the crowd 81.9% of the
  time where Gemini agrees 83.3%. 20–45 is centred negative with a best case of +0.4 pt at $8,269. On run
  4's pairs alone the two bands read −1.3 [−2.9, +0.3] and −1.4 [−2.4, −0.5]. **$14,367 cancelled.**

**Three independent methods agree on that boundary**, which is the standard the inverted-table trap below
taught us to demand:

| method | what it says |
|---|---|
| headroom over the free label (this table) | +6.2 at 0–2 falling to −1.3 at 45–100 |
| `panel_run_delta.py` on run 4 (66% wide pairs) | +0.09 ±0.36 pts for $969 = **$11,140/point** |
| `bt-refit-v5-panel` held-out gains by band | +8.0 at 0–2, +0.8 at 10–20, **+0.0** at 45–100 |

That is the answer to "why not just panel-label all 52,414": **half the export is already at human level
for free, and on the widest quarter a human vote is actively worse.** Full coverage is $24,446 and most of
the marginal dollars buy duplicates or damage.

**Spend the buy zone in tranches, cheapest-first** (0–2 → 2–5 → 5–10 → 10–20), re-running
`panel_run_delta.py` between tranches against the standing stop rule of <1 point per $500. Log §5.7–§5.8
have the full argument.

Machine-readable copy of this table, including the bootstrap intervals and verdicts:
`artifacts/label-information-v2/report.json` (`byPercentileGap`, `byBandCost`), plus
`report-run4only.json` for the unbiased subsample alone. The superseded
`artifacts/label-information-v1/report-v2qc-bands.json` is kept as the pre-run-4 record. Regenerate with
`scripts/label_information.py`. `select_topup_pairs.py` defaults to `--max-gap 20`, **refuses** a
panel-fitted `--ratings`, and prints every draw's band composition against this table.

### 2c. Does the band structure mean anything? Yes — measured on an unselected sample

The whole policy above rests on percentile gaps computed from a Bradley-Terry fit over **VLM** labels. If
that ordering were arbitrary, the bands would be arbitrary and every number in §2a would be an artefact of
how we drew pairs. Run 4 tested it the only way that can settle it: uniform random pairs, bands *recorded*
rather than imposed, scored against human votes the ranking has never seen
(`scripts/gap_agreement_curve.py`, log §5.8).

| percentile band | pairs | crowd majority share | split-half | **ranking agrees with the crowd** |
|---|--:|--:|--:|--:|
| 0–2 | 68 | 67.8% | 71.2% | **53.2%** |
| 2–5 | 98 | 70.9% | 77.0% | **56.5%** |
| 5–10 | 198 | 71.4% | 76.0% | **57.8%** |
| 10–20 | 479 | 70.4% | 75.1% | **61.9%** |
| 20–45 | 883 | 74.6% | 80.5% | **69.6%** |
| 45–100 | 774 | 84.9% | 93.1% | **83.6%** |
| *coin-flip null* | | *61.5%* | *~50%* | *50%* |

Two things to carry forward:

1. **The ordering is real.** Agreement is strictly monotone across all six bands and
   `ρ_s(gap, decisiveness) = +0.408` unbinned. A circular partition cannot produce a monotone
   out-of-sample curve.
2. **Crowd cohesion is a threshold, not a gradient.** Majority share is flat at 70–71% across 2–5, 5–10 and
   10–20, then rises. So below ~20 points the crowd is uniformly ~70% cohesive *no matter how close the
   ranking says the pair is* — which is why the ranking's own accuracy still climbs there while consensus
   does not, and why those are the pairs worth buying. Do not expect a study on 0–2 pairs to produce
   "cleaner" labels than one on 10–20 pairs; expect it to produce *more surprising* ones.

> **Warning — this table inverts if you cut the bands wrong.** Computed against `bt-refit-v4-panel` it
> reports headroom of −5.7 pt at 0–2 and +13.6 pt at 20–45, i.e. "spend $8.8k on wide pairs", which is the
> worst available choice. v4-panel absorbed the panel votes, so pairs where the crowd overruled Gemini had
> their θ pushed apart and migrated into wider bands, dragging their VLM errors with them. Band cuts,
> strata, and pair draws must come from a ranking that has not seen the votes under analysis.

## 2b. Before the next study: can the model consume what we already bought?

Added after run 3, because we nearly bought a fourth round without checking. A study is
only worth money if the gain reaches the **product**, and the product is the neural
comparator, not the BT ranking. Two separate links have to hold:

Added after run 3, because we nearly bought a fourth round without checking. **The gate is now
cleared (2026-08-01)** — the record below is kept because the check is permanent, not because
it is still failing.

| link | status |
|---|---|
| human votes → better BT ranking | ✅ measured, +3.18 per study *on hard pairs*; +0.09 on uniform pairs |
| better BT ranking → better neural comparator | ✅ **measured 2026-08-01, +2.00 pts vs its own labels** |
| comparator reaches the ranking's own accuracy | ❌ **failing as of 2026-08-04 — this is now the gate** |

> **⛔ The gate has closed again, for a different reason. Do not buy the buy zone yet.** Log §5.8 measured
> both on a *population* sample and the comparator is **1.8–3.1 points behind the VLM-only ranking**
> (paired, every interval below zero), losing the 10–45 percentile band by 3–7 points while winning under
> 10. The ranking recovers that ordering from labels we already own, so the missing accuracy is not missing
> ground truth — it is extraction. Buying the $7,463 buy zone now would improve a ranking the product
> cannot ship (BT cannot score a face it has never seen) and hand the comparator more of the labels it is
> already the *best* at using. Run `configs/train-v16-panel-run4.yaml` and re-measure first.

The original failure: `train.py` built labels from the export's `finalOutcome`, so panel votes
never entered training, and the comparator agreed with human votes 54.3% against 54.4% for the
Gemini labels it copied — it had learned the VLM's blind spot rather than fixed it.

After wiring panel votes in (`panel_labels` → `src/faceiq_pref/panel.py`) and retraining the
same config, the comparator reaches **56.4%** against a 59.1% ceiling: +2.00 pts over the
Gemini labels (paired 95% CI [+0.60, +3.41]) and +2.13 pts over its exact control, closing 43%
of the reachable gap. The gain sits in the 2–20 percentile-gap band — the pairs these studies
buy. Full result in research log §5.3.1.

**Standing rule: never buy a round of labels the trainer cannot read.** Before authorising
a study, confirm the previous study's votes are (a) in the ranking and (b) in a training run
whose accuracy is measured against humans, not against the export. Two practical notes for
re-running this check:

* Use `scripts/compare_panel_evals.py`, which bootstraps over **pairs**. Comparing two runs'
  raw accuracies as if independent, or resampling votes, both overstate confidence badly.
* `val_accuracy` is useless for this — all three arms sat at 77.1–77.5% because val keeps
  Gemini's labels. A run can improve materially against humans and show nothing on val.

## 3. Recommended next study — **none. Fix the comparator first**

> **Updated 2026-08-04, after run 4 came in.** There is **no study to authorise right now**, and that is a
> deliberate stop rather than an absence of ideas. The only validated spend left is the $7,463 buy zone
> (§2a), and §2b's gate blocks it: the comparator is 1.8–3.1 points *behind* the ranking on population
> pairs, so more labels would improve a ranking the product cannot ship. Run
> `configs/train-v16-panel-run4.yaml`, re-measure with `eval_vs_panel.py` on run 4's pairs, and revisit.
>
> **When a study does become the right move, these are the three candidates, in order:**
>
> | candidate | cost | deliverable | precondition |
> |---|--:|---|---|
> | 0–2 and 2–5 buy-zone tranche | **$1,534** | +6.2 pt headroom bands, 2,644 pairs | §2b gate clears |
> | 18–34 age A/B on a fixed 500-pair subset of run 4 | **$429** | isolates the rater-age effect with pairs held constant (§8) | none — runnable any time |
> | fresh uniform pairs for a *second* calibration test | **~$400** | re-tests calibration on a future ranking; run 4's pairs are spent, they are inside `bt-refit-v5-panel` now | only after a new ranking ships |

<details><summary>Superseded: run 4's design rationale (kept — the reasoning is reusable for any measurement study)</summary>

> **Decided 2026-08-02.** Run 4 is *not* another hard-pair round. It is a uniform random draw,
> drawn and ready at `labels/panel-run-4-random/pairs.json` (2,500 new pairs + 20 carried golds,
> median percentile gap **29.9**). Generate with `scripts/select_random_pairs.py`; fill the
> Prolific form from `prolific-soft-launch-form.md` **§9**.
>
> **Two earlier instructions in this section were wrong and are corrected:**
>
> 1. "Draw against `bt-refit-v4-panel`" — **no.** v4 absorbed the panel votes, so its percentile
>    gaps already encode the crowd's corrections and any band cut on it is circular. Log §5.7
>    measures the damage: the headroom table inverts, recommending the single worst band in the
>    set. `select_topup_pairs.py` and `select_random_pairs.py` now both **refuse** a panel-fitted
>    `--ratings`. Use `bt-refit-v2-qc`.
> 2. "Stratified across /10 gaps" — **no.** Stratifying destroys the population-accuracy estimate,
>    which is the study's main deliverable. The draw is deliberately *unstratified*: uniform over
>    unbought pairs, no gap filter, no ethnicity floor. Bands are *recorded*, not *imposed*.

Why this rather than a fourth hard-pair round — all measurements, not preferences:

* The comparator sits at **56.9%** against a **59.1%** ceiling on hard pairs; ~2.2 points remain
  on that pair type.
* §5.6: the binding limit is **per-face precision**, not label volume. Each face has 34
  comparisons but only **10.1 effective** ones, and the median /10 is uncertain by **±0.45**. More
  pairs spread thinly does not shrink that.
* §5.7: headroom above a 20-point gap is **unresolved** (+0.1 [−2.5, +2.5] and −1.1 [−2.9, +0.6])
  on 389 and 262 *biased* pairs, and that uncertainty gates ~**$15.3k**. **66% of this draw lands
  in those two bands**, so one $1,286 study either releases or kills that spend.
* It is the only study that can produce an **honest calibration curve**: run 4's pairs are the
  only ones in the programme no ranking has been fitted on. Run `rating_calibration.py` *before*
  any refit consumes them — that property is spent permanently on first use.

Design, and why depth breaks §2's breadth-first rule here: breadth wins when the goal is to *move*
the ranking, but the crowd ceiling is a leave-one-out statistic and the calibration curve bins
observed win rates, so both want votes per pair. At 6 votes the "majority of the others" is 5 noisy
votes, which biases the ceiling down and *understates* headroom. Hence **2,500 pairs × 12 votes**,
the same 300 raters and same ~$1,286 as run 3.

**Keep it representative, and test age separately.** Our panel is 38.1% aged 55+ and 11.9% aged
18–24, which is what a census-matched US sample looks like. Restricting age would turn a population
number into a subgroup number and break comparability with runs 2–3, and the measured age effect is
small (in-group gap +2.4 pts for 55+, −0.5 for 35–44). Test it afterwards as a clean A/B: the same
**500-pair subset** given to an **18–34** standard sample — no 300-participant floor applies, so
100 raters × 100 pairs ≈ **$429**. Form §9.3 has the detail.

**How it turned out (log §5.8):** every deliverable landed. 303 raters, 31,237 judgments, $968.58, 11–13
votes/pair. Population accuracy 81.5% (ranking) vs a 74.9% ceiling; a monotone out-of-sample calibration
curve; and the two wide bands resolved to a hard no, cancelling $14,367. The design lesson worth reusing:
**a measurement study should be scored on its stated deliverable, not on `panel_run_delta.py`** — run 4
moved the ranking by only +0.09 pts and was still the highest-leverage $969 in the programme.

</details>

<details><summary>Superseded: the staged hard-pair round (run 4 as originally scoped)</summary>

**Stage it.** 4,800 new pairs from the 0–10 point band at ~6 votes/pair = 30,000 votes ≈
**$1,286**, 300 raters. Then re-measure before committing the rest.

The size is set by a platform floor, not by choice: **Prolific will not run a representative
sample below 300 participants.** 300 raters × 100 pairs is 30,000 votes, so the draw has to
be ~4,800 pairs to keep votes-per-pair near 6. Spending those raters on a smaller draw would
mean 9.4 votes/pair, which the equal-budget test above shows is the wasteful direction.

Why staged rather than the full $3,680 at once: the curve was still rising at run 2's
margin, but "still rising at 2,250 pairs" does not prove "still rising at 12,000." One
$1,000 slice answers that for real, and if it holds, the remainder is cheap.

Verified against the current export — after excluding run 2's 3,000 pairs, the worked
examples and QC-failed faces, **10,079 pairs** remain within 10 percentile points, so the
draw is not pool-limited. The 4,800-pair draw touches 2,757 faces at **3.5 new comparisons
each**, with 338 faces appearing only once. Watch that number on every draw: the gain travels
through better face scores, so faces with a single new comparison contribute least.

</details>

Current pool for a hard-pair round, if one is bought later: **13,702** unbought pairs sit within 20
percentile points (§2a's buy zone), costing $7,952 at 12 votes each. Take them cheapest-band-first.

## 4. The pipeline — what actually has to happen

Yes, a new study needs a **new draw**; `pairs.json` is a static file shipped to the app,
not a database query.

Use **`select_topup_pairs.py`**, not `select_panel_pairs.py`. The latter draws run 2's
designed experiment — five fixed stratum shares plus an overlap stratum for cross-checking
prior labels — which was right for *asking* whether the VLM labels hold up. A top-up is a
different sample: one band, and it must exclude what we already bought.

```bash
# 1. draw new hard pairs against the newest refit, excluding what we already bought
python scripts/select_topup_pairs.py --export data/exports/<runId> \
    --ratings artifacts/bt-refit-v3-panel/ratings.csv \
    --max-gap 10 --n 4800 \
    --exclude-pairs labels/panel-pilot/pairs.json \
    --out labels/panel-run-3
# emits pairs.json (ships, blinded) + sample-meta.json (answer key, gitignored)
# and prints the vote/rater/cost estimate for the draw

# 2. golds ride along unchanged — 20, revision 2, already validated at ~90 views each
cp labels/panel-pilot/golds.json labels/panel-run-3/golds.json

# 3. ship to the app and set votes/pair via the session length
cp labels/panel-run-3/{pairs,golds,examples}.json ../faceiq-rating/data/
#    votes/pair = raters x PAIRS_PER_SESSION / pairs. For 6 votes on 3,200 pairs:
#    190 raters x 100 = 19,000 -> ~6 votes/pair. Leave PAIRS_PER_SESSION at 100.
cd ../faceiq-rating && git add data lib && git commit -m "run 3 data" && git push

# 4. new completion code in Vercel, then redeploy. Verify /api/monitor shows the new
#    pair count at 0 votes before publishing on Prolific.

# 5. pull the data back (psql, not /api/export — it dies above ~1.5k rows)
#    see runbook Stage 6 for the exact query

# 6. analyse this study on its own — rater QC, gold health, gates
python scripts/analyze_panel_run.py --results labels/panel-run-N/results \
    --panel-dir labels/panel-run-N --out artifacts/panel-run-vN
#    writes TWO reject lists: reject-pids.txt (drop from fits) and
#    prolific-rejections.txt (refuse payment — behavioural proof only)

# 7. apply the stop condition from §1 BEFORE refitting: did this study's money buy
#    anything that generalises to pairs we did not buy?
python scripts/panel_run_delta.py --export data/exports/<runId> \
    --prior labels/panel-pilot/results labels/panel-pilot/sample-meta.json \
    --prior labels/panel-run-3/results labels/panel-run-3/sample-meta.json \
    --new   labels/panel-run-N/results labels/panel-run-N/sample-meta.json \
    --rejects artifacts/panel-run-v1/reject-pids.txt \
              artifacts/panel-run-v3/reject-pids.txt \
              artifacts/panel-run-vN/reject-pids.txt \
    --spend 1286 --out artifacts/panel-run-delta-vN

# 8. refit with ALL studies' votes pooled — repeat the two flags per study, same order
python scripts/refit_bt_panel.py --export data/exports/<runId> \
    --panel-results labels/panel-pilot/results labels/panel-run-3/results \
                    labels/panel-run-N/results \
    --panel-meta    labels/panel-pilot/sample-meta.json \
                    labels/panel-run-3/sample-meta.json \
                    labels/panel-run-N/sample-meta.json \
    --rejects artifacts/panel-run-v1/reject-pids.txt \
              artifacts/panel-run-v3/reject-pids.txt \
              artifacts/panel-run-vN/reject-pids.txt \
    --exclude-faces artifacts/face-qc-v1/exclude-faces.csv \
    --exclude-genders artifacts/face-qc-v1/gender-fixes.csv \
    --baseline artifacts/bt-refit-v2-qc/ratings.csv \
    --out artifacts/bt-refit-vN-panel

# 9. register the study in app/dashboard.py -> PANEL_RUNS so the tab can show it
```

**Keep `--baseline` on `bt-refit-v2-qc`** (the last VLM-only fit). Pointing it at a panel
refit makes the held-out table unreadable: that baseline was fit on *every* rater for the
pairs it owned, so it scores its own training data on earlier strata and appears to win.
The script warns about this and `metrics.json` records a `baselineCaveat`, but the honest
before/after is step 7, not step 8.

**Do not wipe the database between runs.** Set `CURRENT_STUDY_ID` in Vercel to the new
Prolific study id; `buildQueue()` and `/api/monitor` both scope to it, so past runs stay in
the tables and stop being counted. Archive each study to its own
`labels/panel-run-N/results/` (already gitignored — the rows carry Prolific ids).

**Queue behaviour is handled and now proven.** `buildQueue()` reserves pairs for in-flight
sessions (fixed 2026-07-31). Run 3 landed at **6 votes on 3,676 pairs and 7 on 1,124** —
sd 0.43 against run 2's 5.63. Consider it closed.

## 5. How the data joins

Everything keys on **`pairIndex`**, the export's stable index for a matchup. `pairs.json`
carries it, the app returns it, `sample-meta.json` maps it to face ids and strata, and
`refit_bt_panel.py` consumes `{pairIndex: (wins_A, wins_B)}`.

So **multiple studies accumulate by merging vote tallies on `pairIndex`** — no schema
change, no migration, and a pair bought twice simply gets more votes. Pass repeated
`--panel-results`/`--panel-meta` pairs (same order) and the fitter treats them as one
pooled panel; run 3's refit did exactly this over 7,780 pairs and 65,894 votes. Faces gain
precision from every pair they appear in, which is the mechanism the whole strategy rests
on.

Each study needs its own `--panel-meta` because `pairId` (the app's key) is only unique
*within* a draw, while `pairIndex` (the export's key) is global. The meta file is the
translation, which is why results and meta must stay paired.

Rejected raters are dropped by `prolificPid` before tallying, so QC decisions from any run
apply retroactively without touching the raw archive.

## 6. What a study must report before we spend again

0. **Its stated deliverable, written down before launch.** Added after run 4, which moved the ranking by
   +0.09 points and was still the best $969 in the programme because its job was measurement. A study
   bought to *improve* is judged on item 1; a study bought to *measure* is judged on whether the
   measurement came back tight enough to act on. Deciding which after the fact is how confirmation bias
   gets in.
1. **Marginal gain on withheld pairs** — `panel_run_delta.py`, against the §1 stop
   condition, **and reported per pair type**. Run 3 (hard) returned 1.24 points per $500 and run 4
   (uniform) returned 0.046 in the same month. A single global figure would have averaged them into a
   meaningless number.
2. **The dose curve** — is this study's own contribution still rising at 100%?
3. **Held-out gain by stratum** — fit on half the raters, score on the other half.
   `refit_bt_panel.py` writes this to `metrics.json` → `heldOut.byStratum`.
4. **All §5.1 BT gates** — connected graph, no face under 15 distinct opponents,
   subsample stability ρ > 0.95. A refit that fails these does not ship regardless of
   accuracy.
5. **Coverage histogram** — votes per pair. A wide spread means money was wasted on
   over-served pairs.
6. **Gold health** — every gold measured at its new view count; retire and replace
   anything under 70% with `scripts/replace_golds.py`.
7. **Rater QC** — two lists, and keep them apart. Drop votes on converging statistical
   signals; refuse *payment* only on behavioural proof (impossible median, straight-lining).
   Gold failures alone are a poor screen on a **near-tie** draw (correlation with real agreement +0.18,
   gap 2.2–2.4 pts) but a decent one on a **wide** draw (run 4: gold-failers trailed by 7.8 pts). Golds
   *are* wide pairs, so the screen only measures the task when the draw looks like the golds. Weight it
   accordingly.
8. **Feedback messages** — 14 in run 3 and 14 in run 4, and several surfaced real issues (a misleading
   tie label, "Skip vs Save & Continue" ambiguity on ties, residual AI-looking faces, a recognisable
   public figure). Read them.
9. **Population accuracy alongside hard-pair accuracy** — since run 4 the same predictor is known to move
   30 points on pair selection, so a bare accuracy figure is not reportable. `analyze_panel_run.py` on run
   4's pairs is the population reference.

## 7. Things that are settled — don't relitigate

- **Human votes above a 20-point percentile gap**: never buy. Not "unresolved", not "last in line" —
  **negative**. §2a: −0.9 [−2.1, +0.4] at 20–45 and −1.3 [−2.2, −0.5] at 45–100 on an unbiased sample,
  confirmed by two other methods. $14,367 cancelled 2026-08-04.
- **Whether the percentile bands are circular**: they are not. §2c — monotone 53.2% → 83.6% agreement
  across bands on a uniform draw scored against votes the ranking never saw.
- **Clear pairs**: never buy. Gemini is better than one human and ~free. On a *population* sample the raw
  Gemini label beats an individual human by 5.0 points and the ranking by 6.6.
- **VLM confidence as a selector**: useless. Run 3 measured `high` 53.5% and `medium` 48.2%
  against the crowd — confidently wrong. Confirmed at full scale over all 7,780 panel pairs
  in §2a: decisiveness is flat across tiers and two thirds of the closest band is "high".
  Select on percentile gap only.
- **Re-prompting the VLM for a graded label / rating band** ("5.8–6.3" instead of a winner):
  ruled out by the same measurement. A finer output format cannot help a labeler that cannot
  tell a near-tie from a blowout in the first place. See log §5.7.
- **Fast clicking**: not a quality problem. Across 299 run 3 raters the fastest third agree
  with the crowd 61.5% vs 62.8% for the slowest third (`corr = −0.12`). 36% of judgments
  land under 2 seconds because that is the instructed behaviour.
- **Wiping the database between runs**: unnecessary. `CURRENT_STUDY_ID` scopes the queue
  and the monitor.
- **Depth past ~8 votes/pair**: the worst available use of budget for improving the ranking.
- **Comparison coverage / graph connectivity**: not a bottleneck. One component per
  gender, no face under 15 comparisons, stability 0.99.
- **A denser VLM matchup graph** (the "go to 80–100k matchups" idea). Measured in
  `scripts/cohort_capacity.py` by subsampling the graph and scoring against human votes:

  | matchups | per face | predicts a human vote |
  |--:|--:|--:|
  | 11,978 | 8.3 | 56.41% |
  | 23,957 | 16.7 | 56.93% |
  | 35,936 | 25.0 | 57.04% |
  | 47,914 | 33.4 | 57.90% |

  A BT ability estimate's standard error falls as \(1/\sqrt{k}\) in comparisons per face, so
  accuracy approaches its asymptote in \(1/\sqrt{k}\) — not linearly. Fitting that shape
  gives `accuracy = 0.588 − 0.072/√k` (R² 0.79), so **doubling the graph to ~100k matchups buys
  +0.4 points** and even an infinite number tops out at **58.8%**, i.e. **+0.9 points** over the
  57.9% we have.

  *(Corrected 2026-08-04: this line previously said "+0.04 points" for 100k, a factor-of-ten
  slip. The conclusion is unchanged — 0.4 points of a 5.1-point remaining headroom, bought with a
  labeller that is at chance where the headroom lives — but the arithmetic is worth having right.
  Reproduce with `scripts/cohort_capacity.py`; k = 2 × matchups ÷ 2,866 faces, so 47,914 matchups
  is k = 33.4 and 100k is k = 69.8.)*

  The bound that does not depend on the fit: the **ceiling** — the best score any ranking
  can get, since the same pair goes to raters who contradict each other — is **70.6%**. Total
  remaining headroom is 5.1 points, and the whole density curve can reach 0.9 of them. A naive
  straight line predicted +3.7 points from doubling the graph, which would consume three-quarters
  of all remaining headroom using a labeller that is *at chance* on the pairs where the headroom
  lives. That is why the linear read is wrong.

  **Why \(1/\sqrt{k}\), in one line:** a face's θ is estimated from its k comparisons, and the
  error of any average over k noisy observations shrinks as \(1/\sqrt{k}\). So the 25th comparison
  on a face is worth about \(1/\sqrt{25}-1/\sqrt{26}\) — a fortieth of what the 1st was worth. The
  graph is not "full", it is **past the point where adding to it is measurable**.

  Per-stratum ceilings, for sizing any future buy:

  | stratum | ceiling | v3 now | headroom |
  |---|--:|--:|--:|
  | close | 68.4% | 64.3% | 4.2% |
  | mid | 68.8% | 62.8% | 6.0% |
  | clear | 76.6% | 72.2% | 4.4% |
  | ALL | 70.6% | 65.5% | 5.1% |
- **Cohort composition does not inflate scores.** Faces are near-uniform across Labs deciles
  (D1 12.0%, D10 12.5%, mid 8.5–10.6%). Composition cannot inflate a /10 anyway: percentile
  is by construction uniform *within* the cohort whatever the mix, and the /10 comes from
  applying the pre-registered anchor curve to that percentile. Ours lands at median 5.00,
  p90 7.00. Separately, 41% of matchups deliberately pair faces from the same decile (5.2%
  are D10-vs-D10) — that is the informative structure BT wants, not a defect.

- **Speed-based auto-rejection**: leave Prolific's setting off. See the fast-clicking bullet
  above; we screen on agreement, not on the clock.

## 8. Open questions a future study should answer

**~~Rater demographics.~~ Pulled, and it answers two questions at once.** Prolific's export
for studies 2 and 3 matched **667 of 667** raters on `prolificPid`
(`scripts/analyze_demographics.py`, `artifacts/panel-demographics/summary.json`).

The representative sample delivered: 51.2% female, 63.1% White / 11.9% Black / 11.0% Mixed /
7.6% Other / 6.3% Asian, all close to census.

**On age there is no sampling error — there is a choice we made without noticing.** Our panel
tracks US adults to within 0.6 points in every band:

| band | ours | US 18+ | a plausible product-user mix |
|---|--:|--:|--:|
| 18–24 | 12.0% | 11.5% | ~35% |
| 25–34 | 17.1% | 17.5% | ~40% |
| 35–44 | 17.0% | 16.5% | ~15% |
| 45–54 | 15.8% | 15.5% | ~7% |
| 55+ | **38.2%** | **39.0%** | ~3% |

So Prolific delivered exactly what we ordered. But "US adults" and "our users" are different
populations, and we have been answering the first question. Reweighting every vote to the young
mix above moves the vote-share vector to ρ 0.80 and flips the winner on **14.2%** of pairs — which
looks alarming until you run the null. Shuffling the age labels across raters, keeping band sizes
fixed, flips **12.7%** (sd 0.41%) all by itself, because down-weighting 38% of the panel to 3%
throws away most of the effective sample. The genuinely age-attributable part is **~1.5 points of
flips**, statistically solid (observed ρ sits 4.8 sd below the null) but small.

Read that as: age has a real effect, of about the same size as the in-group advantage below, and it
is not worth re-running anything for. It *is* worth stating which population we claim about.
Three ways to handle it, in increasing cost:

1. **Post-stratify at analysis time** (free, what the numbers above do). Only trustworthy while the
   target mix is not too far from the sample; the 12.7% noise floor is the price of a big reweight.
2. **Age quotas via a custom Prolific audience** — you can filter on age, unlike the representative
   sample. Costs more per participant and forfeits the representativeness claim.
3. **Balanced age cells**, sized so each supports its own estimate. This is the design Alex's
   per-audience feature needs anyway, so cost it there rather than retrofitting it here.

One caveat: the product mix in that table is a guess. Replace it with Labs' actual user age
distribution before drawing any conclusion about what our users would say.

On whether cohort-specific taste exists, the test is matched by construction: train on half a
cohort and predict its other half, versus training on an *equally sized* set of raters from
outside that cohort and predicting the same held-out votes. Noise cancels, so any gap is
cohort signal.

| cohort | predicts its own | out-group predicts it | gap |
|---|--:|--:|--:|
| 55+ | 64.25% | 61.89% | **+2.36** (4.0 sd) |
| White | 63.72% | 62.33% | **+1.39** (3.1 sd) |
| Female | 63.62% | 62.57% | **+1.04** (2.3 sd) |
| Other | 63.39% | 59.75% | +3.64 (1.3 sd) |
| Black | 62.64% | 60.33% | +2.30 (1.1 sd) |
| Asian | 58.48% | 59.38% | −0.91 (noise) |

So the effect is **real but small** — one to two and a half points where we have the power to
see it, against 4.8 points of total headroom between the comparator and the human ceiling.
Two caveats before anyone builds on it. It is confounded with rater *consistency*: a cohort
whose members are individually more reliable predicts itself better whether or not its taste
differs, and 55+ raters took visibly longer per judgment. And the small cells have no power at
all — Asian (41 raters) and Other (50) produce error bars of 3 points, wider than the effect.

Verdict for the per-audience feature: directionally supported, nowhere near buildable. Each
cohort would need roughly the coverage the pooled panel has now, which is 8–16× more
recruiting in the small cells, and Prolific's representative sample cannot target them.
Revisit once the comparator itself clears the ceiling gap.

**Cohort cleanliness.** Run 3 raters volunteered two problems unprompted: some faces still
look AI-generated (after QC removed 133), and at least one person in the cohort is a
recognisable public figure. The first adds noise; the second contaminates an attractiveness
judgement with reputation.

## 9. The one open calibration risk

Composition cannot distort the ranking, but it *can* distort the absolute /10, in exactly one
way: a within-cohort percentile only equals a population percentile if Labs' deciles were
computed on the population. They were computed on Labs' users, who self-selected into a
face-rating app. If that pool sits above the population, every /10 we print is shifted down —
our "5.0" could be a population 6 or 7.

Nothing in our data can settle this; it needs an outside reference. The mechanism already
exists: the dashboard's **Anchor panel** (§5.4 Path B), where you assign product /10 scores to
known faces and place cohort faces on that ladder instead of on cohort percentiles. Worth
resolving before any absolute score ships. It is independent of, and does not undermine, any
of the ranking results above.
