# Human panel pilot — plan & decision framework

**Status:** ✅ **Executed and superseded** (drafted 2026-07-14; three runs completed 2026-07-29 → 08-01).
Kept as the historical record of *why* we decided to buy human labels and what we pre-registered before
seeing any data — useful when writing up, and the honest reference for which predictions held.

**Read these instead for anything current:**

| For | Read |
|---|---|
| Results and what they mean | `panel-pilot-findings.md`, and `scoring-gt-research-log.md` §5.5 (canonical) |
| How to run the next study | `panel-study-playbook.md` |
| Operational detail, per-run outcomes, bugs | `panel-pilot-runbook.md` |
| Prolific form fields | `prolific-soft-launch-form.md` |

**Two of this doc's framing assumptions turned out to be wrong**, recorded here rather than edited away:
the human ceiling on close pairs is ~59%, not the ~85% guessed in §1.1 below, so the model was *not* near
a recoverable ceiling; and majority agreement turned out to be the wrong statistic entirely — the signal
is in the vote *share*.

**Purpose:** decide — for a few thousand dollars instead of tens of thousands — whether
the current VLM-labeled pipeline (Gemini Flash pairwise labels → BT → neural comparator)
is measuring what humans perceive, or whether systematic label bias requires human
relabeling before production.

---

## 1. Why a pilot before scaling

The model plateaus at ~78% held-out pairwise accuracy against a single VLM labeler that
itself agrees with one human auditor 84.9% of the time (750-pair seeded sample, log §4.3).
Three unknowns block the "spend more on labels" decision:

1. **The human ceiling is unmeasured.** We don't know how often one human agrees with the
   human *majority* on these pairs. If that number is ~85% on close pairs, the model may
   already be at the recoverable ceiling and "78% vs one noisy labeler" is not
   underperformance.
2. **Random vs systematic VLM error is undistinguished.** Random error in the ~15%
   disagreement washes out through BT aggregation (~35 comparisons/face). Systematic
   error (consistent VLM preference for/against a feature or demographic) does NOT wash
   out — it shifts theta and the comparator learns it as signal. The overall 84.9% cannot
   distinguish these; stratified disagreement analysis can.
3. **Controversy is unmeasured.** Product wants per-face confidence bands. Real bands come
   from inter-rater variance, which one labeler cannot provide.

## 2. Questions the pilot answers (each with its metric)

| # | Question | Metric | Why it matters |
|---|----------|--------|----------------|
| Q1 | What is the human-human ceiling? | Mean rater-vs-panel-majority agreement, overall and by theta-gap bucket | Sets the fair benchmark for the model's 78% |
| Q2 | Is Gemini biased, and where? | VLM-vs-panel-majority agreement, stratified by demographic attributes and pair closeness | Distinguishes random from systematic label error |
| Q3 | Does the comparator track human consensus? | Model-vs-panel-majority agreement (compare to model-vs-Gemini 78%) | Model may agree with consensus BETTER than with its own training labels (noise averaging) — would validate the pipeline as-is |
| Q4 | What does controversy look like? | Per-pair vote entropy / split distribution | Feeds confidence-band design; validates whether model-ensemble spread predicts real controversy |

## 3. Design

### Pair sample (~2,000–3,000 pairs from the existing export)

Stratified, not random — the pilot must cover the regions where the answers differ:

- **Theta-gap buckets** (from `bt-refit-v1` ratings): close (|dtheta| bottom third),
  medium, far. Oversample close pairs — that's where labeler quality matters.
- **Demographic coverage**: after attribute-tagging the 3,000 cohort faces (ethnicity,
  hair color, eye color, skin tone via FairFace or a VLM pass — the bias-audit
  prerequisite), ensure every major group appears in enough pairs (target >= 150 pairs
  per group) to detect group-level VLM bias.
- **VLM confidence strata**: include low/medium/high-confidence VLM pairs.
- **Overlap with the human audit**: include the ~750 already-audited pairs (free
  cross-check of the original auditor against the panel).
- Both genders, same-gender pairs only (matches training data).

### Raters

- **10–15 ratings per pair.** Literature: aggregated attractiveness ratings stabilize
  quickly; 10–15 gives a reliable majority + usable variance. (SCUT-FBP5500 used 60
  raters/face for absolute scores; pairwise winners need far fewer.)
- **Diverse panel** — mixed gender, age, ethnicity of raters; log rater demographics for
  rater-pool sensitivity checks.
- **Task format**: identical to VLM task — two photos, "who is more attractive?", forced
  choice + optional "too close to call" (maps to tie).
- **Quality control**: seeded gold pairs (far-theta-gap pairs where the answer is
  unambiguous), attention checks, per-rater agreement-with-majority monitoring. Reject
  raters below threshold on golds; standard vendor tooling covers this.

### Vendor & cost

| Option | Est. cost/judgment | 3,000 pairs x 12 raters = 36k judgments | Notes |
|--------|--------------------|------------------------------------------|-------|
| Prolific (research panel, self-serve) | $0.03–0.05 | **$1.1k–1.8k** + platform fee | Best demographics control; we build the rating UI (or use jsPsych/Gorilla) |
| Managed vendor (e.g. Surge, Toloka managed) | $0.05–0.15 | $1.8k–5.4k | They handle UI + QC |
| Scale AI | higher | likely $5k–15k+ | Overkill for subjective forced-choice; built for objective annotation |

Realistic all-in pilot budget: **$3k–8k** including UI setup time and re-runs.

**Privacy gate (resolve BEFORE contracting):** the research cohort's 3,000 faces were
cleared for VLM labeling; confirm ToS/privacy policy permits showing them to third-party
human raters under vendor DPA/NDA. The 1M+ user-photo pool is a SEPARATE, stricter
question — do not assume vendor labeling of user photos is permitted; needs legal review.

## 4. Decision gates

Compute after the pilot:

- `H` = mean human-vs-majority agreement (the ceiling), by theta-gap bucket
- `G` = Gemini-vs-majority agreement, overall + stratified by demographic group
- `M` = model(v8/ensemble)-vs-majority agreement

| Outcome | Interpretation | Action |
|---------|----------------|--------|
| `G ≈ H` overall AND no demographic stratum where `G` drops materially (e.g. >5 pp below its overall) | VLM labels are as good as one human; errors ~random | **Keep VLM pipeline.** Scale cheaply with multi-VLM ensemble voting (below). No human relabeling. |
| `G ≈ H` overall BUT specific strata show systematic gaps | Targeted label bias | Human-relabel ONLY affected strata + close pairs (small fraction of 52k), refit BT, retrain |
| `G` well below `H` broadly (e.g. >8 pp) | Single-VLM labeling is the bottleneck | Budget full human panel labeling; the 52k relabel becomes justified. Also test multi-VLM consensus first — may close most of the gap at ~1% of the cost |
| `M > G` (model agrees with humans more than Gemini does) | Comparator already averaged out VLM noise | Strong validation — pipeline works; publish this number |

**Controversy deliverable (independent of gates):** per-pair vote-split dataset →
(a) correlate with model-ensemble score spread (v8 + v1 + v7 z-scored std) — if
correlated, ensemble spread becomes the free production confidence signal;
(b) correlate with BT theta-gap — cheap proxy available today.

## 5. Related workstreams (parallel, cheap)

1. **Bias audit on existing data (~$0, prerequisite).** Attribute-tag 3,000 cohort faces;
   check BT theta and model-score residual distributions by group; stratify the existing
   750-pair audit disagreements by group. Also generates the strata for the pilot sample.
2. **Multi-VLM ensemble labeling test (~$50–200).** Relabel the pilot's 3,000 pairs with
   3–5 diverse VLMs (Gemini, GPT-4V-class, Claude, Qwen-VL); majority vote + disagreement
   as confidence. Compare consensus-vs-panel against single-Gemini-vs-panel. If the gap
   closes, this is the scaling path for future labeling (52k pairs x 5 models is still
   ~100x cheaper than humans).
3. **External model cross-check (half a day).** Score the 3,000 cohort faces with
   MEBeauty's pretrained predictor (modern PyTorch, `pytorch_predict.py`;
   2,550 multi-ethnic faces x ~300 diverse raters) and, if conversion is easy,
   SCUT-FBP5500 ResNet-18. Compare tau/rho vs BT and vs v8; inspect where an
   independently-trained model disagrees with ours by demographic group.
   **License constraint: both datasets/models are non-commercial-research-only — 
   diagnostics and cross-checks are fine; fine-tuning them into the product is not.**
4. **Anchor panel smoke test (already planned, §5.4 Path B).** Orthogonal to label
   quality — tests deployment mechanics. Dashboard already validates panel monotonicity
   (model order vs assigned product scores) and flags inverted anchors. Use widely-spaced
   rungs (3.0/4.5/5.5/6.5/7.5/8.5) with unambiguous exemplars.

## 6. Timeline sketch

| Week | Work |
|------|------|
| 1 | Bias audit (attribute tagging + stratified analysis); legal/privacy check; pick vendor |
| 2 | Pair sampling script; rating UI (if Prolific); gold pairs; multi-VLM relabel of pilot pairs |
| 3 | Panel collection (typically days on Prolific for 36k judgments) |
| 4 | Analysis vs decision gates; controversy/ensemble-spread validation; write up in research log §5.5 |

## 7. What this pilot does NOT decide

- Whether to label the 1M user-photo pool (blocked on legal, and on this pilot's outcome).
- Anchor panel composition (separate smoke test).
- Architecture changes (v1–v10 established that labels, not architecture, are the binding
  constraint).
