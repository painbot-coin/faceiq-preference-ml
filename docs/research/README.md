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

1. ~~**Why the comparator is behind the ranking on typical pairs**~~ — **answered 2026-08-04, and the
   answer changes the design.** Five arms (labels, targets, checkpoint selection, variance head, 4×
   resolution) all tie at 1.8–2.1 pts behind BT with the 10–20 band stuck near 55% against 63.2%, and the
   arm with *no* panel labels loses the same bands. BT solves a **global** system over 47,914 comparisons;
   the comparator only sees **local** pairs and must induce an absolute function from 1,430 faces per
   gender. **So stop asking it to reproduce the global ordering.** It beats BT under a 10-point gap — use
   it there, and place new faces against the ranking (priority 1). Log **§5.8**.
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
| ~~1~~ | ~~Fix the comparator's 10–45 band by training~~ ❌ **five arms, five ties, closed 2026-08-04.** More labels (v16), soft vs hard targets (v12/v13), fixed checkpoint selection (v17), a variance head (v19), and 4× resolution (v20/v21) all land 1.8–2.1 pts behind BT with the 10–20 band stuck near 55% against 63.2% | ~$15 of GPU | `train-v10`, which saw **no** panel labels, loses the same bands. BT solves a *global* system over 47,914 comparisons; the comparator only ever sees *local* pairs. It is not a tuning problem. Log **§5.8** |
| **1** | **Ship the band as a tier** — `T = 1.920`, half-width `T·logit(agreement)/2` in percentile space, displayed as one of **~7 tiers**, at 2-in-3 agreement plus a 90% floor claim | **free** | p90 placement error is **0.93 /10** on the best checkpoint (the widely-quoted 1.33 came from a `val_fraction 0.5` arm — see below), so a point score still overclaims and this fixes its presentation for nothing. Two independent measurements say ~7 tiers, and 2-in-3 agreement is *exactly one tier down* (`T·ln2 / (9/7) = 1.035`), which settles the last open product call. |
| **1a-new** | **Use `train-v14-panel-ship`, not the newest checkpoint** — and queue `train-v22-ship-0.2` | **free / ~$2 GPU** | New 2026-08-04. `scripts/placement_by_checkpoint.py --common-val` ranks checkpoints on *placement* error rather than val accuracy, scoring all of them on the same held-out faces. v14 places at **0.34 / 0.28** median /10 with **67%** tier-exact against **0.44–0.47** and **55–57%** for every v12–v21 arm — because those arms ran at `val_fraction 0.5` to hold out panel pairs and so trained on 13k pairs instead of 33k. They were controlled label-recipe comparisons, never production candidates. `train-v22-ship-0.2` combines v17's labels with v14's data volume and is the only cheap experiment left with obvious upside. |
| **1a** | **Normalisation parity audit** — does the production upload path apply the exact faceiq-labs MediaPipe eye-level crop the cohort went through? | **free** | A per-face error that no ranking scheme cancels, and the cheapest real bug to rule out. **Largely answered 2026-08-04**: 644 real user uploads from today, pulled straight from labs blob storage, are 1024×1024 like the cohort, detect a face 25/25, and the normaliser moves them **7.85** mean abs pixel value against **7.66** for cohort photos. Same crop pipeline. Strong evidence rather than proof — equal residual magnitude is not a pixel-identical crop — but it demotes preprocessing as a suspect for odd upload scores. |
| **1b** | **Wire reference-set placement into inference** — 200 stratified cohort faces of known θ, MLE for θ_new, `se` from the curvature | **free** | Buys the per-user standard error the band wants, a θ-scale position, and explicit handling of the ~16% of top-tier faces with no finite solution. Note it does **not** repair the score: a rank is already shift-invariant, so it matches the raw scalar's ordering (0.857 F). See `production-scoring-pipeline.md` §2–3. |
| **1b-bis** | **Validate off-cohort** — ✅ **ready to judge, 2026-08-04.** Two queues built, 350 pairs + 35 repeats each, ~26 min apiece: `set-1-female` (82 clean faces) and `set-1-male` (601). **The audit is the story**: the id-level exclusion list was applied correctly and still let through **113 of 998 photos (11%) that are cohort people re-uploading** — invisible to ids, caught by ArcFace. Normalisation parity also confirmed on real uploads (7.85 vs 7.66) | **free** | **The only untested link in the chain.** Every accuracy figure in this log is measured against BT θ fitted on the same 3,000 faces the comparator trained on — honest about new *comparisons*, silent about new *faces from a different source*, which is all production sees. Separates ordering / band / calibration, and separates a fixable **scale shift** from an expensive **spread**. See `production-scoring-pipeline.md` §7 and log §6.3. |
| **1c** | **Gap-routed ensemble** — comparator under a 10-pt gap, ranking above it | **free** | Measurable on the 631 leak-free run-4 pairs. §5.6 ruled out *global* blends; routing is a different operation. Caveat: at inference the true gap is unknown, so measure the oracle-routed ceiling first and drop it if that is small. |
| **1d-new** | **Multi-photo narrowing is buildable now** — and the photo band has its first measurement | **free** | New 2026-08-04, log **§5.9**. The cohort was believed to be one photo per person, which is why this was shelved; it is not. It contains **363 people photographed two or more times**, i.e. a test–retest set we already owned. Scored through `bt-refit-v5-panel`, the same person's two photos land **0.65 /10 apart at the median, 1.54 at p90** — so photo-to-photo spread *exceeds* the model's own p90 placement error of 0.93, and asking a user for a second photo is worth more than another 0.1 /10 out of the comparator. |
| **2** | **Ship bands, not decimals** — temperature-scale against §5.8's curve, then define the band policy | free | Now fully specified by measurement: 0.25 /10 is a coin flip, 2 points is 83%, and the point score is overconfident by 26 pts. See `programme-direction-review.md` §5 for the band maths. |
| **3** | **Anchor-ladder decision** (§5.4) | free | Decides whether top faces can exceed 8/10. This is the thing that reads as "the ranking is wrong". Product call. |
| **4** | **Landmarks → EBM** for explainable per-feature contributions | **~free** | Labs already stores `Face.frontLandmarks` / `mediapipeLandmarks` on every upload — no extraction project and no inference-path problem. The only work is adding the columns to `export-gt-run.ts`'s `select` and pinning point order. See §5.7. |
| ~~5~~ | ~~**The 0–20 buy zone**~~ — 12,862 unlabelled pairs | ~~$7,463~~ | ⛔ **CLOSED 2026-08-04. Both routes measured shut on the same day.** The labels are still genuinely better than the free ones (+3.6 to +6.2 pts headroom, all intervals clear of zero) — that was never the question. `ranking_value_to_placement.py`: swapping the ranking supplying reference θ from zero human votes to all 65,894 moves the production score **0.2 pts, downward** (θs enter with weight ≤0.25 each over 200 references, so they average out). `train-v22-ship-0.2`, v17's label recipe at v14's data volume: **+0.00 pts [−0.96, +0.96]** against human votes. So label quality reaches a user's score through neither the reference set nor the comparator's training set. **Label quality has stopped being the binding constraint.** Reopening needs a different *kind* of label — test–retest, or a demographically different panel — not more pairs from the same pool. |
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

**Eleven live docs.** Everything else is in [`archive/`](./archive/README.md).

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
| [`production-scoring-pipeline.md`](./production-scoring-pipeline.md) | The shipping pipeline — built / measured / blocked, in order |
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
| **Inference** | Reference-set placement for an uploaded photo (tier, band, placed /10, per-user `se`), a **browser for the whole reference set** in θ order with per-tier counts, the **Labs composite** side-by-side with its measured verdict, and the **off-cohort validation labeller** |
| BT rankings / Refit diagnostics | Per-face scores, gates, stability, anchor curve |
| Training runs / Model inspection / Model gallery | Per-run histories, checkpoint scores, visual cross-checks |
| Composite | The measured negative result on blends |

The **Anchor panel** tab was removed 2026-08-04 along with `src/faceiq_pref/anchors.py`. It was the
§5.4 Path A idea — hand-assign /10 scores to a few faces and place photos on that ladder — and it is
superseded by reference-set placement, which uses 200 faces with *measured* θ and no hand-assigned
scores. Scale-level corrections belong in the anchor curve, not in per-face judgments.
