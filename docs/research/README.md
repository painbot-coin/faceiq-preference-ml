# Scoring GT — Research Docs

**Canonical methodology:** [`scoring-gt-core.md`](./scoring-gt-core.md). (An older `scoring-gt-plan.md`
is long gone — if you see it referenced anywhere, it means core.)

**Superseded plans and closed sub-studies:** [`archive/`](./archive/README.md) — read its index before
opening anything in there, so you know what replaced it.

**ML training code:** separate repo `faceiq-preference-ml` (see [`scoring-gt-training.md`](./scoring-gt-training.md)) — not in faceiq-labs.

---

## Doc maintenance (after every implementation)

When you ship a feature, migration, or research gate, **read the relevant docs first**, then **update their stated status** so nothing stays “planned” or “TBD” after it exists in code.

| If you changed… | Update |
|-----------------|--------|
| Prisma models / migrations | [`scoring-gt-schema.md`](./scoring-gt-schema.md) — §1 model list, relationship diagram, field tables; §2 phase sufficiency; §3 section status; §6 migration checklist |
| Admin UI routes or API | [`vlm-pilot-spec.md`](./archive/vlm-pilot-spec.md) — §6 routes, §8 API table |
| Phase completion / QA results | [`scoring-gt-research-log.md`](./scoring-gt-research-log.md) — matching §2–§6 section (Results, Decision, Artifacts); §8.2 manifest; §9 gate row |
| Process or “what next” | This README — Focus table, phase-by-phase table, build checklists |
| Canonical methodology only | [`scoring-gt-core.md`](./scoring-gt-core.md) — when the *plan* changes, not every bugfix |

**Checklist before closing a PR / task:**

1. Grep docs for **“Not started”**, **“TBD”**, **“planned”**, **“need §”**, **“UI TBD”** in the area you touched — fix or narrow the wording.
2. Promote **§3 “Planned”** blocks in the schema doc to **§1 “in scope today”** (or mark **implemented** with migration id) when tables ship.
3. Cross-link **migration filename**, **route path**, and **artifact ids** (cohort, sample set, run) in the research log §8.2 where applicable.
4. Do **not** edit attached plan files in `.cursor/plans/` — update the research docs above instead.

Treat doc drift as a blocker: if code and docs disagree, the docs are wrong until updated.

---

## Focus **right now** (2026-08-04 — run 4 is in and it changed the plan)

**The labelling question is closed. The bottleneck moved to the model.** Run 4 (uniform random pairs,
study `6a71110e4d5c6eba96c17d21`, 303 raters, $968.58) delivered all three things it was bought for, and
two of them were bad news for the previous plan. Full write-up in log **§5.8**; strategy in
[`programme-direction-review.md`](./programme-direction-review.md).

| What run 4 settled | Result |
|---|---|
| Is the percentile-band structure circular? | **No.** Bands cut on the VLM-only ranking predict human agreement monotonically across all six bands, **53.2% → 83.6%**, on votes that ranking never saw. `ρ_s(gap, decisiveness) = +0.408`. |
| Population accuracy — what the rating scores on a *typical* pair | **81.5%** for the ranking, **79.9%** for the raw Gemini label, against a **74.9%** individual-human ceiling *on the same metric*. **The ranking beats the average person by 6.6 points at naming the crowd's choice.** |
| Is the remaining $15.3k of labelling worth it? | **No.** Headroom is **−0.9 [−2.1, +0.4]** at 20–45 and **−1.3 [−2.2, −0.5]** at 45–100. **$14,367 cancelled.** |
| Honest /10 calibration | 0.25 pt gap = **53.0%** (coin flip), 1 pt = 65.4%, 2+ pt = **82.7%**. Monotone, tight. But `σ(Δθ)` is **overconfident by 26 points** — temperature-scale before publishing any probability. |
| Where does the neural comparator stand on typical pairs? | **1.8–3.1 points *behind* the ranking** (paired, all intervals below zero). It wins under a 10-pt gap and loses 10–45. |

Steps 1–4 below are ✓ **complete**. Labeling (`cmr1mr0m7000196d57zi3vcgn`, 52,414 pairs, 84.9% audit)
finished 2026-07-04; BT and the comparator followed; **three** Prolific studies (plus a $106 soft launch)
bought **95,245 usable human votes** from 963 raters for **$4,155**; the comparator was retrained on those
votes; the rating was validated (**§5.6**); and the label economics are now measured to a verdict
(**§5.7**, **§5.8**).

**Current state — everything below is done:**

| | Where it stands | Doc |
|---|---|---|
| Face photo QC | ✓ 133 of 3,000 faces excluded | log **§5.0** |
| Ranking of record | ✓ **`bt-refit-v5-panel`** — all three panel runs, 10,280 pairs / 95,245 votes, all gates | log **§5.8** |
| Human panel runs 1–4 | ✓ $404/pt for hard pairs (run 3); **$11,140/pt for uniform pairs (run 4)** | log **§5.5**, **§5.8** |
| Comparator, before panel labels | 54.3% vs a 59.1% human ceiling — it had only learned to match Gemini | log **§5.3** |
| Comparator, after | ✓ **56.4%** on hard pairs (`train-v13`); **67.67%** on population pairs (`train-v12`) | log **§5.3.1**, **§5.8** |
| Per-face ranking uncertainty | ✓ median 95% interval **0.91 /10 points**, **241 rank places** | log **§5.6** |
| /10 calibration | ✓ **honest, out-of-sample** — monotone, but overconfident by 26 pts | log **§5.8** |
| Composites (blends) | ✓ measured; broad blends **worse**, Labs + comparator best at 56.91% but not significant | log **§5.6** |
| Label information | ✓ **buy inside 20 pctile points, never outside** — three independent methods agree | log **§5.7**, **§5.8** |

### How to read "we spent $4,155 and I don't see a result"

The result is real, and the metric was hiding it in **both** directions.

**Every number the programme quoted was a worst case.** All three earlier studies deliberately bought
near-ties, so the "56.9% against a 59.1% ceiling" figure describes the hardest pairs that exist, not
typical ones. On a uniform draw (log §5.8), same predictor, same metric:

| predictor — *"did it pick the face more raters picked?"* | near-tie pairs | **typical pairs** |
|---|--:|--:|
| chance | 50.0% | 50.0% |
| the raw Gemini label | 51.2% | **80.0%** |
| BT ranking, VLM-only | 52.9% | **81.7%** |
| one human vs the crowd (ceiling) | 60.8% | 73.2% |

**~29 points from pair selection alone**, so **an accuracy without a stated pair distribution is not a
number.** On a typical pair the ranking beats a randomly chosen person at naming the crowd's choice.

⚠️ **There are two "accuracies" and mixing them creates a fake 10-point gap.** The table above asks *did
it pick the majority's face* (a 9–3 pair is one clean win, so 100% is reachable). The comparator's numbers
are reported on the other metric — *what share of raw ballots agree* — where a perfect predictor on that
same 9–3 pair scores only 75%. On typical pairs, vote-level, on identical pairs:

| predictor — *"what share of ballots agree?"* | typical pairs |
|---|--:|
| neural comparator (`train-v12`) | 67.7% |
| the raw Gemini label | 68.8% |
| BT ranking, VLM-only | **69.5%** |
| one human vs the crowd (ceiling) | 69.9% |

Same ranking: **+8.5 points above a person** on the first metric, **+0.2** on the second. Both true,
different questions. Full 2 × 2 in `artifacts/panel-run-v4/accuracy-matrix.json`
(`scripts/accuracy_matrix.py`).

**And the money went where it was supposed to.** Human votes bought +8.0 points of held-out accuracy in
the closest band and **+0.0** in the widest (log §5.8, v5 held-out table). That is the whole thesis of the
programme, measured: pay for the pairs machines cannot resolve, take the rest for free.

### What is actually unknown (this is the bottleneck, and it is no longer ground truth)

1. **Why the comparator is behind the ranking on typical pairs.** 67.4–67.7% vs 69.50% (paired −1.84 to
   −2.09, both intervals below zero), with the deficit concentrated in the 10–45 percentile band. The
   ranking recovers that ordering *from the same labels*, so this is an extraction failure — capacity,
   schedule, loss shape, checkpoint selection, or 112 px input — not a data shortage. **Confirmed by
   experiment 2026-08-04**: `train-v16` added run 4's wide-band votes, lifting panel coverage of the train
   split from 14.7% to 19.4%, and the 10–20 band did not move. **More labels cannot fix it.** Log **§5.8**.
2. ~~**Test–retest stability**~~ — **deprioritised 2026-08-04 (product call).** It would need a second
   photo view per person exported from Labs, which does not exist for every face and is more overhead than
   it is worth right now. Consequence to accept knowingly: the "upload more photos, your band narrows"
   feature stays unbuildable, because photo-level variance is the only component extra photos can shrink
   (`programme-direction-review.md` §5). Revisit if users report their score moving between uploads.
3. **The /10 anchors** — the /10 is a percentile ladder (50th = 5.0, 90th = 7.0, 99th = 8.0), so *no*
   model or blend can make the top cohort face score above ~9. "Our best faces only score 7" is an anchor
   decision, not a model defect. See log **§5.6**.
4. **Cohort validity at the source.** 3,000 faces, one front-view photo each, drawn from the Labs upload
   funnel. That inherits the funnel's selection and cannot express within-person variation. No amount of
   labelling touches it. See [`programme-direction-review.md`](./programme-direction-review.md) §3.

### Priority order for the next work block

| Priority | Task | Cost | Why it is ranked here |
|---|---|---|---|
| ~~1~~ | ~~Run `train-v16-panel-run4`~~ ✅ **done 2026-08-04 — and it confirmed the diagnosis by failing to help.** 67.43%, a tie with `train-v12` (+0.25 [−0.95, +1.46]), still **2.09 pts behind the ranking**, 10–20 band still 8.5 pts adrift | $2 of GPU | Panel coverage of the train split went 14.7% → 19.4% with the new pairs aimed at exactly the weak bands, and it changed nothing there. **The 10–45 deficit is now measured to be an extraction failure, not a label shortage** — no labelling programme can close it. Log §5.8 |
| **1** | 🟡 **Fix the comparator's 10–45 band — three arms running 2026-08-04.** ✅ (a) checkpoint selection moved off `val_accuracy` onto a human-grounded `panel_val_accuracy`, verified against `eval_vs_panel` to 0.02 pts; 🟡 (b) `train-v17-panel-select` = v16 under the new rule; 🟡 (c) `train-v19-panel-variance` = the never-run variance head with panel labels; 🟡 (d) `train-v18-arcface-224` = resolution | **~$10 of GPU** | The ranking recovers this ordering from the same labels, so the information is provably there. This is the only thing standing between us and a shippable rating, and none of it needs new data. |
| **1a** | **Gap-routed ensemble** — comparator under a 10-pt gap, ranking above it | **free** | Measurable today on the 631 leak-free run-4 pairs. §5.6 ruled out *global* blends; routing is a different operation and the band table is the first evidence for it. May solve the product problem without any training at all. |
| **1b** | **Reference-set inference** — score a new face by comparing it against a fixed panel of cohort faces and reading off where it lands, instead of trusting the raw scalar | **free** | Uses the comparator the way it was trained (pairwise) rather than the way it is currently read (absolute), and is the most likely explanation for "the ranking looks off when I upload a face". |
| **2** | **Ship bands, not decimals** — temperature-scale against §5.8's curve, then define the band policy | free | Now fully specified by measurement: 0.25 /10 is a coin flip, 2 points is 83%, and the point score is overconfident by 26 pts. See `programme-direction-review.md` §5 for the band maths. |
| **3** | **Anchor-ladder decision** (§5.4) | free | Decides whether top faces can exceed 8/10. This is the thing that reads as "the ranking is wrong". Product call. |
| **4** | **Landmarks → EBM** for explainable per-feature contributions | **~free** | Labs already stores `Face.frontLandmarks` / `mediapipeLandmarks` on every upload — no extraction project and no inference-path problem. The only work is adding the columns to `export-gt-run.ts`'s `select` and pinning point order. See §5.7. |
| **5** | **The 0–20 buy zone, in tranches** — 12,862 unlabelled pairs | **~$7,463** | Still validated spend (+3.6 to +6.2 pts, all four intervals clear of zero) but **deferred behind priority 1**: it improves a ranking the product does not ship, and priority 1 decides whether the comparator can even use it. Cheapest-first: 0–2 $563 → 2–5 $971 → 5–10 $1,724 → 10–20 $4,205. |
| ~~6~~ | ~~Human votes above a 20-point gap~~ | ~~$15,328~~ | ❌ **Cancelled 2026-08-04.** Headroom is negative on an unbiased sample; at 45–100 the whole interval is below zero. $14,367 saved. |

**Ruled out by measurement, do not revisit without new evidence:** buying human votes above a 20-point
percentile gap (**§5.8** — negative headroom, and three independent methods agree); asking the VLM for
graded labels or rating bands (§5.7 — its confidence has no discrimination to grade with); cutting bands,
strata or pair draws on a panel-fitted ranking (§5.7 — it inverts the recommendation;
`select_topup_pairs.py` now refuses); shipping a composite blend (§5.6); quoting an accuracy without its
pair distribution (§5.8 — it moves 30 points).

For the strategic picture — is this approach working, when do we stop labelling, what is plan B, how do
rating *bands* work — read [`programme-direction-review.md`](./programme-direction-review.md).

For anything panel-related read [`panel-study-playbook.md`](./panel-study-playbook.md) (policy),
[`panel-pilot-runbook.md`](./panel-pilot-runbook.md) (ops),
[`panel-pilot-findings.md`](./panel-pilot-findings.md) (results), and
[`prolific-soft-launch-form.md`](./prolific-soft-launch-form.md) (the Prolific fill sheet — §9 is run 4).
Earlier plans now live in [`archive/`](./archive/README.md).

<details><summary>Original phase-4 step list (all complete)</summary>

| Step | Where | Doc |
|------|-------|-----|
| **1. Export matchup JSON** | **faceiq-labs** — paginated API | `vlm-pilot-spec.md` · run export route |
| **2. BT refit + stability + /10 calibration** | **`faceiq-preference-ml`** (Python) | `scoring-gt-training.md` · core §6–§7 |
| **3. Validate vs Labs `overall_score`** | ML repo notebooks | research log §5.2 |
| **4. Train preference comparator** | **`faceiq-preference-ml`** | `scoring-gt-training.md` |
| **5. (Optional) Persist BT to DB** | faceiq-labs admin | schema §3.3 — not built yet |

</details>

| Role | Document | Why |
|------|----------|-----|
| **Primary — read for “what next”** | [`scoring-gt-training.md`](./scoring-gt-training.md) + [`scoring-gt-core.md`](./scoring-gt-core.md) **§6–§8, §15 step 7** | Export → BT → train |
| **Write as you go** | [`scoring-gt-research-log.md`](./scoring-gt-research-log.md) **§5+** | BT results, training summary |
| **Export API reference** | [`vlm-pilot-spec.md`](./archive/vlm-pilot-spec.md) | `GET …/export` |
| **DB / future BT tables** | [`scoring-gt-schema.md`](./scoring-gt-schema.md) §3.3 | When persisting ratings back to admin |

**Do not** start another VLM batch on this run unless re-labeling. **Do** scaffold `faceiq-preference-ml` as a sibling repo and copy exports to `data/exports/`.

**Completed (no longer focus):** 3k curation (`cmqz06xr20001bydxaazuojp8`), pair queue k=35, full 52.5k VLM batch, human audit gate.

---

## What each doc is for

**Ten live docs.** Everything else is in [`archive/`](./archive/README.md).

| Document | Role | When to open it |
|----------|------|-----------------|
| [`README.md`](./README.md) | **Start here** — current state, what is unknown, priority order | First thing in a new context |
| [`programme-direction-review.md`](./programme-direction-review.md) | **Strategy** — are we going the right way, when to stop, what plan B is, the band maths | Before committing money or changing direction |
| [`scoring-gt-research-log.md`](./scoring-gt-research-log.md) | **Notebook** — what you did, numbers, decisions | **During and after** each step; §5.8 is the newest finding |
| [`scoring-gt-training.md`](./scoring-gt-training.md) | **This repo's charter** — the offline ML checklist | "What is the next ML task?" |
| [`scoring-gt-core.md`](./scoring-gt-core.md) | **North star** — scale, sampling, BT, consensus, phases | Planning a work block; §6–§8 for BT/calibration/training method |
| [`panel-study-playbook.md`](./panel-study-playbook.md) | **Labelling policy** — what to buy, what never to buy, §2a headroom bands | Before spending money on labels |
| [`prolific-soft-launch-form.md`](./prolific-soft-launch-form.md) | **Prolific fill sheet** — every field, per run. §9 = run 4 | Setting up or launching a study |
| [`panel-pilot-runbook.md`](./panel-pilot-runbook.md) | **Ops** — draw → ship → deploy → monitor → archive | Running a study end to end |
| [`panel-pilot-findings.md`](./panel-pilot-findings.md) | **Panel results** — runs 1–3 | Interpreting panel numbers |
| [`scoring-gt-schema.md`](./scoring-gt-schema.md) | **DB map** — faceiq-labs models, future BT tables | Persisting ratings back to admin (§3.3, not built) |

**Rule:** the **log** is always a sidecar — write to it as you go. **Core** and **schema** are lookup,
not sequential reading. If a doc's status line goes stale, fix it in the same change that made it stale.

---

## Phase-by-phase: primary doc → move on when

| Phase | Primary doc | Also use | Log (fill when done) | Schema (when) |
|-------|-------------|----------|----------------------|-----------------|
| **0 — Prompt & consensus** ✓ pilot done | `core` §5.1, §13 #1–3 · `vlm-pilot-spec` §6–§9 | `research-log` §2.2–§2.3 | §2.1 archive pilot · §2.2 prompt chosen · §2.3 panel validated | **§3.1** before wiring consensus at scale |
| **1 — Pool import** | `core` §2, §13, §15–§16 | `vlm-pilot-spec` §3 import | §3.1 pool QC | Usually **no migration** (existing cohort tables) |
| **2 — 3k curation + audit** | `core` §16 | Pairwise Lab **`/research/pairwise/cohorts/[id]/sample`** | §3.2 curation · §3.3 audit pass | **§3.2** ✓ (`20260629180000_vlm_pilot_sample_set`) |
| **3 — Labeling (52.5k)** ✓ | `core` §14–§15 · `vlm-pilot-spec` §15 | Admin batch + Review | §4.1–§4.3 ✓ | export exclude migration; review fields |
| **4 — BT + calibration + export** ✓ | `core` §4, §7, §15 step 7 · **`scoring-gt-training.md`** | Export API | §5.1–§5.2 ✓ | §3.3 BT tables optional |
| **4b — Face QC + human panel** ✓ **closed** | `panel-study-playbook.md` | `panel-pilot-runbook.md` | §5.0, §5.5, §5.7, **§5.8** ✓ | — |
| **5 — NN training** ← **now, and it is the bottleneck** | **`scoring-gt-training.md`** | `core` §8 · `programme-direction-review.md` | §5.3 (v1–v15 ✓; **`train-v16` is the open one**) | ML repo only |
| **6 — Primary human eval** | `research-log` **§6** | `core` for context | §6 full write-up (partial in §6.4) | — |

---

## Quick decisions

| Question | Open |
|----------|------|
| What should I do this week? | Run `configs/train-v16-panel-run4.yaml` on the GPU box, then `eval_vs_panel.py` on run 4's pairs |
| Should we buy more labels? | **No.** `panel-study-playbook.md` §2a and §2b — the buy zone has a measured outer edge and the remaining $7,463 is blocked on the model |
| Is this whole approach working? | `programme-direction-review.md` — yes, and the bottleneck moved from data to the model |
| How do we show a rating band? | `programme-direction-review.md` §5 — three different bands, only one is measurable today, and interval intersection is the wrong rule |
| Which ranking do I use? | **`bt-refit-v5-panel`** to score. **`bt-refit-v2-qc`** to cut bands or calibrate — it is the only fit that predates the human votes |
| How does the admin UI / batch work? | `vlm-pilot-spec.md` (labeling done; use export route) |
| Which Prisma model / migration next? | `scoring-gt-schema.md` §3.3 (optional BT persist) |
| What do I write down after a run? | `research-log.md` §9 gate row → matching §2–§6 |

---

## Doc index

| Document | Purpose |
|----------|---------|
| [`scoring-gt-core.md`](./scoring-gt-core.md) | Canonical plan — phases, 3k × 35, BT, VLM consensus |
| [`scoring-gt-research-log.md`](./scoring-gt-research-log.md) | Living log — record at gates; artifact pointers |
| [`programme-direction-review.md`](./programme-direction-review.md) | Strategy — direction, stop rules, plan B, band maths |
| [`vlm-pilot-spec.md`](./archive/vlm-pilot-spec.md) | Pairwise admin + pilot (done) + §15 scale batch |
| [`scoring-gt-schema.md`](./scoring-gt-schema.md) | DB map + migrations before scale |
| [`scoring-gt-training.md`](./scoring-gt-training.md) | Offline preference model (separate repo) |
| [`panel-study-playbook.md`](./panel-study-playbook.md) | Labelling policy — §2a buy/skip bands, §2b the spend gate, §2c non-circularity |
| [`panel-pilot-findings.md`](./panel-pilot-findings.md) | Panel results, runs 1–4 |
| [`panel-pilot-runbook.md`](./panel-pilot-runbook.md) | Study ops, end to end |
| [`prolific-soft-launch-form.md`](./prolific-soft-launch-form.md) | Prolific fill sheet + §9.5 pull-and-analyse commands |

---

## Dashboard — where each finding is illustrated

`streamlit run app/dashboard.py --server.port 8502`

| Tab | Shows |
|---|---|
| **Human panel** | Per-study results, rater QC, the "was it worth the money" ladder and dose curve, and the **two-accuracies** explainer (majority-level vs vote-level — read it before quoting any figure) |
| **Score bands** | The fitted agreement curve against the raw observed points, `T = 1.920`, a score → band widget with the product sentence, and a live model of multi-photo narrowing with `σ_photo` and `ρ` as sliders |
| **Label spend** | The buy/skip band table with intervals, the $7,463 / $14,367 split, and the three independent methods that agree on the boundary |
| BT rankings / Refit diagnostics | Per-face scores, gates, stability, anchor curve |
| Training runs / Model inspection / Model gallery | Per-run histories, checkpoint scores, visual cross-checks |
| Composite | The measured negative result on blends |
