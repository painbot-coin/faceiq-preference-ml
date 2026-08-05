# Scoring Ground Truth — Research Log

**Purpose of this document:** A living record of *what we did*, *why*, and *what we decided* at each phase of the scoring ground-truth program. This is a **working lab notebook**, not the final study write-up. Fill sections as work completes.

**Related plans:** [`scoring-gt-core.md`](./scoring-gt-core.md) · [`vlm-pilot-spec.md`](./archive/vlm-pilot-spec.md) · [`scoring-gt-training.md`](./scoring-gt-training.md)

---

## How to write in this log

**Methodology → prose.** Explain each step in full sentences and short paragraphs under clear headings (`#### Methodology`, `#### Results`, etc.). A reader should understand *what happened and why* without decoding pipe tables.

**Data → tables or external files.** Use markdown tables only for numeric results, cohort counts, model comparisons, and checklists. Full exports go to `~/research-data/scoring-gt/artifacts/` or the admin DB — log a path and one-line summary, not 52k rows.

**Each validation sub-study should include:**

- **Overview** — one paragraph: what this step was and where it sits in the pipeline
- **Research question** — what we needed to learn
- **Methodology** — how we ran it (human-readable; mention admin routes and DB only where useful for traceability)
- **Success criteria** — what would count as pass/fail before seeing results
- **Results** — what we observed (prose + small data table if helpful)
- **Decision** — go / no-go / next step
- **Artifacts** — run IDs, batch IDs, export paths, hashes
- **Notes** — limitations, surprises, open questions

**Workflow with Cursor:** After a phase completes, ask the agent to *“fill §2.1 from the DB”* or paste a short summary. Save full tables via admin **Copy table** → `artifacts/phase-N/*.tsv`.

**Target size:** Stay under ~500 lines. If a section grows past ~30 lines of results, move detail to `artifacts/<phase>/README.md` and link it.

**Formal study later:** This log feeds a methods appendix and pre-registration. The publishable claim lives in §6; everything else is supporting evidence.

---

## Where data lives

Raw labeling data is in Prisma (`vlm_pilot_*` tables — see [`scoring-gt-schema.md`](./scoring-gt-schema.md)). Charts, CSVs, and checkpoints live under `~/research-data/scoring-gt/artifacts/` and `checkpoints/`. Face photos never go in git.

| What | Where | In this log |
|------|-------|-------------|
| Summary metrics | One paragraph + optional small table in §2–§6 | Yes |
| Full comparison CSVs | `artifacts/` or DB export | Path + hash only |
| Charts | `artifacts/` as PNG/SVG | Path or embed |
| Model weights | `checkpoints/` | Path + hash in §8.2 |

---

## Table of contents

1. [Program charter](#1-program-charter)
2. [Phase 0 — Labeler validation (VLM)](#2-phase-0--labeler-validation-vlm)
3. [Phase 1 — Cohort construction & sampling](#3-phase-1--cohort-construction--sampling)
4. [Phase 2 — Ground-truth labeling at scale](#4-phase-2--ground-truth-labeling-at-scale)
5. [Phase 3 — Rankings, calibration & training](#5-phase-3--rankings-calibration--training)
6. [Phase 4 — Primary evaluation (publishable experiment)](#6-phase-4--primary-evaluation-publishable-experiment)
7. [Phase 5 — Deployment & monitoring (optional)](#7-phase-5--deployment--monitoring-optional)
8. [Cross-cutting log](#8-cross-cutting-log)
9. [Documentation timing checklist](#9-documentation-timing-checklist)
10. [Sub-experiment inventory](#10-sub-experiment-inventory)

---

## 1. Program charter

*Document once at kickoff; revise only when scope changes.*

### 1.1 Research question (top-level)

<!-- What are we ultimately trying to prove or disprove? -->

### 1.2 Why pairwise / why a preference model / why VLMs

<!-- Narrative for "why did we use VLMs?" — update as evidence accumulates in §2 -->

### 1.3 Primary vs supporting work

The **held-out human evaluation** (§6) is the only publishable main result. Everything else supports methods and infrastructure:

- **Labeler validation** (VLM pilot, prompt, consensus) — can we obtain reliable pairwise labels at scale?
- **GT construction** (sampling, labeling, Bradley–Terry) — what is the ground-truth ranking for this cohort?
- **Model training** — can a neural comparator learn the preference structure? (methods summary only)

### 1.4 Scope boundaries (v1)

<!-- N, k, same-gender policy, deferred Phase 2 items -->

### 1.5 Pre-registered decisions

<!-- Lock before main labeling: calibration anchors, k=35, labeler model, success thresholds -->

| Decision | Value / rule | Locked date |
|----------|--------------|-------------|
| Labeler model | `gemini-2.5-flash` (single model, no panel) | 2026-06-29 |
| Labeler prompt (Phase 0 gate) | `pairwise-v2` (Holistic) | 2026-06-29 |
| **GT scale prompt** | **`pairwise-v3-gt`** (structure-first; supersedes v2 for dry/full GT after 1k audit) | 2026-06-30 |
| Audit set for prompt gate | Larger pilot run `cmqvmn3z800a46mdv66rbvclh` (80 pairs, 78 scored) | 2026-06-29 |

### 1.6 Ethics, privacy, data handling

<!-- No PII, cohort frozen, who can access -->

---

## 2. Phase 0 — Labeler validation (VLM)

Before labeling ~52k pairs for ground truth, we must validate that a vision-language model can agree with human pairwise judgments at acceptable cost. Phase 0 runs small pilots in `/admin/pairwise`, compares models and prompts against a fixed human-labeled audit set, and locks the labeler configuration for scale.

---

### 2.1 VLM pilot (complete)

**Status:** Complete — archived 2026-06-29 from live DB.

#### Overview

We ran a series of small pairwise pilots to answer a basic feasibility question: can a VLM pick the more attractive face in a side-by-side A/B comparison, and at what cost? Each pilot used the same end-to-end workflow in the admin pairwise tool, with all data stored in isolated research tables (`vlm_pilot_*`), separate from production face records.

#### Research question

Can Flash-class VLMs match human pairwise judgments often enough (~75%+) to justify using them as the primary labeler for ~52k ground-truth comparisons, at a per-pair cost that scales to roughly hundreds of dollars rather than tens of thousands?

#### Methodology

The pilot pipeline always followed the same sequence. First, we imported a cohort from a Data Lab export (a zip containing `manifest.jsonl` and front-face images). The import created a cohort record and one row per face, with photos uploaded to blob storage. We then created a **run** for that cohort — a container with a reproducible seed and a target number of pairs.

Pair generation ran server-side before any labeling. Using a deterministic hash of the run seed, the system created a fixed set of A/B comparisons: which two faces appear together, their display order, and whether face A appears on the left or right. This layout was frozen before any model or human saw the pairs, so all evaluators saw identical stimuli.

Human labeling came next. A researcher labeled every pair in the admin labeling queue, choosing left, right, or tie. These labels became the **ground truth** for scoring VLM accuracy. VLMs were never run until all human labels were complete.

We then ran **VLM batches**: for each pair, each selected model received the same two photos and prompt, and returned a winner, confidence, and rationale. Results were stored per comparison, model, and prompt version. The results dashboard aggregated accuracy and cost; the review queue supported case-by-case inspection of disagreements.

We executed this pipeline three times at increasing scale: a 52-face / 40-pair original pilot, a 100-face / 80-pair model comparison run, and an 80-pair test draw from a ~20k parent pool.

#### Success criteria

Pass if a Flash-tier model reached **≥75% agreement** with human labels (human ties excluded from the accuracy denominator) at **≤~$0.004 per single-model call**, with a credible path to ~$100 or less for 52.5k pairs.

#### Results

All three runs reached `vlm_complete` status with full human labeling. The 52-face pilot was male-heavy (48M / 4F) and is useful as a feasibility check, not for fine-grained model ranking. The **100-face / 80-pair run** is the primary audit set for model selection.

**Cohorts and runs:**

| Cohort | Faces | Run seed | Pairs | Human ties | Status |
|--------|-------|----------|-------|------------|--------|
| `pilot set` | 52 | `vlm-pilot-v1` | 40 | 1 | `vlm_complete` |
| `Larger pilot` | 100 | `vlm-pilot-v1-2` | 80 | 2 | `vlm_complete` |
| `Large 20k Batch` | 19,989 | `vlm-pilot-test3` | 80 | 0 | `vlm_complete` |

The ~20k cohort is the parent pool for Phase 1 stratified sampling; only 80 test pairs were labeled there so far.

On the 80-pair audit run, prompt `pairwise-v2`, human ties excluded:

| Model | Accuracy | ~$/pair | ~$/10k proj. |
|-------|----------|---------|--------------|
| Gemini 2.5 Flash | 83.3% (64/78) | $0.0017 | ~$17 |
| Gemini 3 Flash | 76.9% | $0.0029 | ~$29 |
| Gemini 3.1 Flash-Lite | 74.4% | $0.0008 | ~$8 |
| Claude Opus 4.8 | 76.9% | $0.0209 | ~$209 |
| GPT-5.3 Chat | 74.4% | $0.0086 | ~$86 |

Human labeling took roughly 4–10 minutes per run depending on pair count. Accuracy on pool-drawn pairs from the 20k cohort was lower (~66–73%), which we expect given broader diversity and harder matchups.

#### Decision

**Go.** Flash-class Gemini models meet the accuracy and cost bar. Proceed to prompt tuning, then scale labeling with a **single model** (Gemini 2.5 Flash — see §2.3).

#### Artifacts

- Run IDs: `cmqvimh740036aoefld3y2oj0` (40-pair), `cmqvmn3z800a46mdv66rbvclh` (80-pair audit), `cmqyw6ywb00hz9gdvqsmjvcih` (20k-pool test)
- Spec snapshot: `vlm-pilot-spec.md` §14
- DB tables: `vlm_pilot_runs`, `vlm_pilot_comparisons`, `vlm_pilot_vlm_results`, `vlm_pilot_metric_snapshots`

#### Notes

Opus matched some Flash models on accuracy but at ~10× cost. The original 40-pair pilot aligns with the spec snapshot (~76–86% for top Flash models). Full per-batch breakdown: query DB or export from `/admin/pairwise/runs/[id]/results`.

---

### 2.2 Prompt validation (complete)

**Status:** Complete — 2026-06-29. Three prompt-sweep batches across two runs; labeler prompt locked to `pairwise-v2`.

#### Overview

With Gemini 2.5 Flash chosen as the scale labeler, we ran controlled prompt sweeps: the same five prompt versions on the same pairs within each run, holding the model fixed. The goal was to see whether any v3 variant beat the `pairwise-v2` (Holistic) baseline by enough to change the production prompt before ~52.5k labeling.

We ran sweeps on two different human-labeled runs:

1. **Audit run** — Larger pilot (`vlm-pilot-v1-2`, 100-face cohort, 80 pairs). This is the pre-registered gate: curated faces, researcher labels, same pairs we used for model selection.
2. **Pool test run** — 20k parent pool (`vlm-pilot-test3`, 80 pairs drawn from ~20k faces). Harder, more diverse matchups; useful for sanity-checking generalization, not for the primary go/no-go gate.

#### Research question

Which prompt version maximizes agreement with human pairwise labels on the audit set, without regressing on tie rate or cost?

#### Methodology

Each sweep used admin **prompt sweep → individual model** (Gemini 2.5 Flash). Five prompts per batch: `pairwise-v2`, `pairwise-v3-minimal`, `pairwise-v3-pillar`, `pairwise-v3-opposite-sex`, `pairwise-v3-weighted`. One VLM call per (pair × prompt). Accuracy computed from frozen metric snapshots; human ties excluded from the denominator. Batches were partial (398–399/400 calls) with one failed call per batch — immaterial for ranking prompts.

#### Success criteria

A variant beats `pairwise-v2` by **≥2 percentage points** on the **audit run** (`cmqvmn3z800a46mdv66rbvclh`), with no meaningful increase in VLM tie rate.

#### Results

**Audit run** (batch `cmqyy45gi02d89`, run `cmqvmn3z800a46mdv66rbvclh`, 2026-06-29):

| Prompt | Accuracy | VLM ties | Cost (batch) |
|--------|----------|----------|--------------|
| **v2 · Holistic (current)** | **87.2%** (68/78) | 1 | $0.14 |
| v3 · Pillar detail | 85.9% (67/78) | 1 | $0.15 |
| v3 · Pillar + outlier weight | 85.5% (65/76) | 1 | $0.16 |
| v3 · Minimal | 80.8% (63/78) | 1 | $0.08 |
| v3 · Opposite-sex lens | 80.8% (63/78) | 1 | $0.15 |

**v2 wins on the audit set** by 1.3 pp over the nearest v3 variant (Pillar). v3 Minimal is cheapest (~$0.001/pair) but **6.4 pp below v2** on this run. No variant met the ≥2 pp success criterion.

**Pool test run — sweep 1** (batch `cmqyxp4s801qq9`, run `cmqyw6ywb00hz9gdvqsmjvcih`, 2026-06-29):

| Prompt | Accuracy | VLM ties |
|--------|----------|----------|
| **v3 · Minimal** | **76.3%** (61/80) | 0 |
| v3 · Pillar detail | 73.8% (59/80) | 0 |
| v3 · Pillar + outlier weight | 73.8% (59/80) | 0 |
| v3 · Opposite-sex lens | 70.9% (56/79) | 0 |
| v2 · Holistic | 67.5% (54/80) | 0 |

**Pool test run — sweep 2** (batch `cmqyyep0s02zo9`, same run, 2026-06-29):

| Prompt | Accuracy | VLM ties |
|--------|----------|----------|
| **v3 · Minimal** | **73.8%** (59/80) | 0 |
| v2 · Holistic | 72.5% (58/80) | 0 |
| v3 · Pillar detail | 72.5% (58/80) | 0 |
| v3 · Opposite-sex lens | 71.8% (56/78) | 0 |
| v3 · Pillar + outlier weight | 71.3% (57/80) | 0 |

On the pool-drawn pairs, **v3 Minimal wins both sweeps** (by ~1–9 pp over v2 depending on batch). Overall accuracy is ~10–15 pp lower than the audit run for all prompts — consistent with harder matchups from the 20k pool, not a prompt-ranking failure.

#### Is the cross-run flip noise or signal?

**Mostly noise from small N and different pair samples**, with a plausible content explanation:

- **Sample size:** Each estimate is ~78–80 scored pairs. A swing of 4–8 pp is only **3–6 labels** — well within binomial noise and partial-batch variance.
- **Different pairs:** The audit run and pool run use **different human-labeled comparisons** (different seeds, different face pools). Prompts can legitimately rank differently on easy vs hard matchups (e.g. Minimal may suffice when faces are far apart; Holistic may help on close calls).
- **Consistent pattern:** v2 leads on **curated close-call audit** pairs; v3 Minimal leads on **pool-hard** pairs while being worse on audit. Pillar variants track v2 closely on audit (within 1–2 pp) but do not exceed it.
- **Tie rates:** All prompts: 0–1 VLM ties per 78–80 pairs — no tie-regression concern.

We did **not** re-run enough sweeps on the audit set to compute confidence intervals. One audit sweep is sufficient for a **pragmatic lock** given v2’s clear lead there and its status as the long-standing baseline.

#### Decision

**Lock `pairwise-v2` (Holistic) for Phase 0 gate and pilot runs.**

**Lock `gemini-2.5-flash` as the sole labeler model** (from §2.1 and §2.3).

**Update (2026-06-30):** After 1k GT dry-run rationale audit (§4.3), **scale GT labeling uses `pairwise-v3-gt`**, not v2 — see §1.5. v2 remains the Phase 0 pre-registration baseline.

Rationale in order:

1. **Pre-registered gate:** Success was defined on the audit run. v2 is best there (87.2%); no v3 variant cleared ≥2 pp improvement.
2. **Conservative for scale:** At ~52.5k pairs, a 6 pp audit regression from switching to Minimal would mean thousands of wrong labels. Holistic instructions may help on ambiguous pairs even if Minimal wins on easy pool draws.
3. **Cross-run disagreement:** Treat pool-run Minimal wins as **generalization signal to monitor**, not grounds to switch — re-evaluate only if scale spot-checks show Holistic underperforming on production pairs.
4. **Cost:** Minimal is ~40% cheaper per call but total scale cost remains ~$90 vs ~$150 for v2 at current rates — both acceptable; accuracy dominates.

**Deferred:** v3 Pillar variants for a future ablation if Holistic spot-checks underperform mid-labeling.

#### Artifacts

- Audit sweep: batch `cmqyy45gi02d89gdv4zoxod7p`, run `cmqvmn3z800a46mdv66rbvclh`
- Pool sweeps: batches `cmqyxp4s801qq9gdvk14tv8dx`, `cmqyyep0s02zo9gdvevd3932z` on run `cmqyw6ywb00hz9gdvqsmjvcih`
- Prompt definitions: `src/lib/vlm-pilot/prompt.ts`
- DB: `vlm_pilot_metric_snapshots` where `batchMode = prompt_sweep`
- Admin export: Results → By prompt → Copy table per run

#### Notes

Phase 0 exit criteria are met: model validated (§2.1), panel rejected (§2.3), prompt locked (§2.2). **Proceed to Phase 1** (stratified 3k draw from parent pool `cmqx9npqe00007hdp54e00vem`).

---

### 2.3 Multi-model consensus validation (complete — no-go for scale)

**Status:** Complete for v1 decision purposes. Panel infrastructure shipped; not selected for scale labeling.

#### Overview

We tested whether aggregating votes from a three-model panel (majority consensus per pair) improves accuracy enough to justify tripling API cost. The admin supports panel mode, individual model-compare mode, and prompt sweep with an optional panel toggle.

#### Research question

Does a 3-model panel beat the best single model on the same pairs, at a cost we can afford for ~157k calls at scale?

#### Methodology

In **panel batch mode**, three models each label every pair independently. The system aggregates votes into a single consensus outcome (`consensusOutcome`, `consensusWinnerFaceId`, `panelVoteJson`) stored in `vlm_pilot_comparison_consensus`. Per-model results are retained for comparison. Metric snapshots record both panel-level and per-model accuracy after each batch.

We ran panel batches on the 80-pair audit run using the production panel: Gemini 2.5 Flash, Gemini 3 Flash, and Gemini 3.1 Flash-Lite, all on prompt `pairwise-v2`.

#### Success criteria

Panel accuracy **greater than** the best single model on the **same pairs**, with manageable cost at 3× call volume (~157k calls for full GT).

#### Results

On the 80-pair audit run, Gemini 2.5 Flash alone outperformed panel consensus:

| Config | Accuracy | Notes |
|--------|----------|-------|
| Gemini 2.5 Flash (legacy batch) | 85.9% (67/78) | Best single-model result |
| Gemini 2.5 Flash (panel batch) | 83.1% (64/77) | Same pairs, panel batch invocation |
| Panel consensus | 80.5% (62/77) | 3-model vote aggregation |
| Panel batch cost | ~$0.42 | 79 pairs × 3 models; 1 missing call (partial batch) |

Head-to-head on panel-batch pairs: 2.5 Flash was correct and panel wrong on 4 pairs; panel correct and 2.5 wrong on 2 pairs. Unanimous agreement rate was ~71%. Panel did not justify 3× spend for raw accuracy.

#### Decision

**No-go for scale panel.** Use **Gemini 2.5 Flash single-model** labeling for the 52.5k ground-truth run. Panel mode remains in the admin for optional experiments (e.g. confidence routing) but is not the production labeler path.

#### Artifacts

- Panel batch: `cmqyw03ij00019gdve6iw79pu` on run `cmqvmn3z800a46mdv66rbvclh`
- Consensus rows: `vlm_pilot_comparison_consensus`

#### Notes

Partial batches must be retried before any downstream Bradley–Terry work (`vlm-pilot-spec.md` §15). Panel value may exist for routing low-confidence pairs to human review, but that was not validated here.

---

## 3. Phase 1 — Cohort construction & sampling

**Phase 0 complete (2026-06-29).** Locked labeler for all scale work: **`gemini-2.5-flash`** + **`pairwise-v2`** (single model — panel rejected §2.3). See §2.1–§2.3 and §1.5.

**Phase 1 complete (2026-06-30).** Approved 3k sample `cmqz06xr20001bydxaazuojp8` on parent pool `cmqx9npqe00007hdp54e00vem`. See §3.2–§3.3.

**Phase 2 dry run complete (2026-06-30).** 1k pair queue + VLM batch + human audit sample → **go** to full 52.5k with **`pairwise-v3-gt`**. See §4.1–§4.3.

**Next:** Create **full-scale queue** on approved sample → VLM batch (~52.5k) → export for BT refit (§5).

---

### 3.1 Parent pool export & QC (imported)

**Status:** Imported — full decile QC pending.

#### Overview

We imported a large Data Lab export as the frozen parent pool from which the 3k active cohort will be drawn. The pool must be **de-identified for research** (no names, emails, or account IDs in the manifest), gender-balanced enough to stratify, and documented with export filters and row counts.

**PII** = *Personally Identifiable Information* — anything that could identify a real person (name, email, phone, account ID tied to identity, etc.). **“PII-free”** here means the export/manifest we use for GT research carries **face photos + scores + demographics**, not user account fields. That keeps the research cohort shareable internally without exposing who uploaded each scan. It also means **we do not currently have `userId` in `exportPayload`** — only `analysis_id` per face.

#### Methodology

Export delivered as manifest + images. Imported via `/admin/pairwise/import` into cohort `Large 20k Batch`. An 80-pair test run from this pool validated that VLM labeling works on pool-drawn matchups (harder than the curated 100-face pilot).

#### Results

**19,989 faces** imported 2026-06-28. Gender split approximately even (9,992 male / 9,997 female). Score decile distribution not yet aggregated (large payloads — defer to offline export). VLM accuracy on 80 pool-drawn pairs was ~66–73%, lower than the curated pilot, which is expected rather than a failure of import QC.

#### Decision

**Pool accepted** for stratified sub-sampling. Full decile audit deferred to the curation gate (§3.2–§3.3).

#### Artifacts

- Cohort ID: `cmqx9npqe00007hdp54e00vem`
- Test run: `cmqyw6ywb00hz9gdvqsmjvcih`

---

### 3.2 Stratified sub-sample + curation (3k active cohort)

**Status:** **Approved** (2026-06-30) — locked at **3,000 active** faces. Ready for pair-queue generation on this sample set only.

#### Overview

Pairwise Lab **Sample curation** at `/research/pairwise/cohorts/[cohortId]/sample`: seeded stratified draw (gender × decile targets from core §15), histogram dashboards (pool vs sample vs target), paginated photo review (remove-only with reason), manual **Fill gaps**, audit checklist, and **Approve sample** lock.

A **draft** sample set remains editable indefinitely: members, histograms, and review state live in `VlmPilotSampleSet` / `VlmPilotSampleSetMember` rows. Nothing is discarded until you **Approve** (locks) or a future **archive + re-draw** flow is used. Closing the browser or returning days later is fine.

#### Research question

Does the active cohort represent the intended attractiveness spread for Bradley–Terry and training?

#### Methodology

**Where it runs:** UI → `POST /api/admin/pairwise/cohorts/[cohortId]/sample/draw`. Server lib: `src/lib/vlm-pilot/sample/` (`deciles.ts`, `draw-rules.ts`, `stratified-draw.ts`, `seed.ts`).

**Step A — Decile labeling (parent pool, once per face)**  
On first draw, if `decileBin` is null:

1. Read Labs **`overall_score` / `overallScore`** from each face’s `exportPayload` (not harmony or other pillars). Denormalize to `labsOverallScore`.
2. Within each **gender separately**, sort by score ascending and assign **D1–D10** as equal-count deciles (~10% of that gender per bin).  
   - **D1 = bottom 10% within gender in this export**, not “absolute 1/10 on a universal scale.”  
   - In our pool, D1 female spans roughly **1.0–5.0** Labs overall because the export is score-skewed; median pool female overall is ~**6.35**.
3. Write `decileBin` + `labsOverallScore` back to `vlm_pilot_faces` (chunked updates).

**Step B — Target grid (core §15)**  
Default **N = 3,000** → **1,500 male + 1,500 female**. Per gender, non-uniform decile targets (stored in `drawRulesJson`), e.g. ~188 D1, ~144 D2, … ~159 D5–D6 (middle-heavy), … ~186 D10. Tails D1/D10 slightly **oversampled** vs flat 150/decile for BT cross-tier coverage.

**Step C — Seeded stratified draw**  
For each `(gender, decile)` cell:

1. Query parent cohort faces with matching `gender` + `decileBin`, excluding faces already picked.
2. Draw `n = target[decile]` faces **without replacement** using deterministic hashes:  
   `sha256("{seed}:{gender}:{decile}:{drawIndex}")` → index into sorted candidate ids.  
   Same seed + same pool → same face ids (reproducible).
3. Insert `VlmPilotSampleSet` (`status: draft`), ~3k `VlmPilotSampleSetMember` rows (`inclusionSource: auto_draw`), and a `draw` event.

**Step D — Manual curation (ongoing, before approve)**  
Default **included** until removed. **Remove** requires reason (`duplicate`, `synthetic`, `quality`, `other`). Removals do **not** auto-replace; **Fill gaps** batch-draws replacements from the same parent cell (seeded sub-prefix `fill_gaps:{n}`). Review grid supports gender/decile/**source** filters (use **Gap fill (new)** after Fill gaps — newest replacements first, **New** badge on cards).

**What we remove (and what we do not):**  
Quality gate only — not re-scoring attractiveness or chasing population representativeness.

| Remove when | Examples | Reason code |
|-------------|----------|-------------|
| Duplicate / near-identical reshoot | Obvious same-person duplicate spotted while browsing | `duplicate` |
| Synthetic / non-real | AI, anime, heavy filters, obvious renders | `synthetic` |
| Unusable photo | Blur, crop, occlusion, bad lighting | `quality` |
| Wrong metadata | Mislabeled gender vs appearance | `other` (+ note) |

**Goal:** A clean ~3k face set for pairwise labeling and Bradley–Terry — real front photos, no junk synthetics/quality failures, stratification preserved after fill. Deciles stay **relative rank within gender in this export**; we are not verifying absolute `/10` scores by eye.

**Duplicates:** The export has no account/user linkage — only one row per `analysis_id`. We do **not** run automated duplicate detection and will **not** re-export to add user IDs. During paginated review, remove a face as `duplicate` only when the same person is **visually obvious** (near-identical reshoot). Same person with a clearly different photo is fine. Residual overlap is acceptable and not an approve blocker.

**Training note:** Phase 3 Jupyter work exports approved members to the ML repo. If curation or labels improve later, we can **swap the approved sample export** and re-run training without redoing Phase 0 labeler validation — the pipeline is designed for that iteration.

**Step E — Approve (later)**  
When audit checklist is green → **Approve sample** → `status: approved`, read-only. Blocks scale pair queue until then (pair queue not built yet).

**Pilot runs on same cohort:** Existing 80-pair VLM pilot run on this pool is independent (`VlmPilotRun`); no re-import required.

See `scoring-gt-core.md` §15–§16. Schema: `scoring-gt-schema.md` §1.5. Migration: `20260629180000_vlm_pilot_sample_set`.

#### Success criteria

Per-cell sample counts match draw rules within tolerance after fill; manual spot-check ~30 thumbnails/decile; curation sign-off; approved sample set ID recorded for future scale runs (`VlmPilotRun.sampleSetId` gate).

#### Results

**Draw executed 2026-06-29** on parent cohort `Large 20k Batch`. **Approved 2026-06-30** after manual curation + fill gaps.

| Metric | Value |
|--------|--------|
| Sample set ID | `cmqz06xr20001bydxaazuojp8` |
| Name | `3k active cohort-v1` |
| Seed | `gt-draw-v1` |
| Target size | 3,000 |
| Active members (final) | **3,000 / 3,000** |
| Status | **`approved`** (2026-06-30T01:07:06Z) |

**Curation removals:** **35** excluded before approve — synthetic **16**, quality **14**, duplicate **5**. Two fill-gap passes added **35** replacements (`34` + `1`); final active: **2,966** initial draw + **34** gap fill.

**Final cell counts:** all 20 gender×decile cells match targets (e.g. female/male D1 **188**, D10 **186**, middle cells **130–159** each).

**Race distribution (approved):** white **69.2%** (2,077/3,000) — passes ≤80% audit bound. Hispanic 242, east_asian 196, black 179, middle_eastern 163, south_asian 128, others ≤8 each.

**Manual review:** general visual pass complete; removed obvious synthetics, quality failures, and visually obvious duplicates only (no automated dedup; see Duplicates note in Methodology).

#### Decision

**Go** — proceed to **pair queue generation** on approved sample `cmqz06xr20001bydxaazuojp8` (§4.1). Do **not** label from the full 20k parent pool or the old 80-pair test run (`sampleSetId: null`).

#### Artifacts

- UI: `/research/pairwise/cohorts/cmqx9npqe00007hdp54e00vem/sample` (read-only)
- Parent cohort: `cmqx9npqe00007hdp54e00vem`
- **Approved** sample set: `cmqz06xr20001bydxaazuojp8` · seed `gt-draw-v1`
- Migration: `20260629180000_vlm_pilot_sample_set`
- Code: `src/lib/vlm-pilot/sample/stratified-draw.ts`, `deciles.ts`, `draw-rules.ts`

---

### 3.3 Pre-label audit

**Status:** **Passed** at approve (2026-06-30).

#### Overview

Final gate before pair generation on the curated 3k. The sample page runs the checklist live and **blocks Approve** until all items pass.

#### Methodology

Automated checks (see `src/lib/vlm-pilot/sample/audit.ts`):

- ≥100 active faces in **D1** and **D10** per gender
- No empty decile per gender
- Race/ethnicity distribution logged; largest group ≤80%
- Manual: spot-check ~30 thumbnails per decile during review (not enforced in code)

#### Results

| Check | Result |
|-------|--------|
| Female D1 ≥100 | **Pass** — 188 active |
| Female D10 ≥100 | **Pass** — 186 active |
| Male D1 ≥100 | **Pass** — 188 active |
| Male D10 ≥100 | **Pass** — 186 active |
| All deciles non-empty (both genders) | **Pass** |
| Largest race group ≤80% | **Pass** — white 69.2% |
| Manual visual pass | **Pass** — researcher sign-off at approve |

#### Decision

**Proceed to §4.1 pair queue** on sample set `cmqz06xr20001bydxaazuojp8`.

#### Artifacts

- Approve event in `vlm_pilot_sample_set_events` (2026-06-30T01:07:06Z) with `raceDistribution` payload

---

## 4. Phase 2 — Ground-truth labeling at scale

~52.5k pairwise comparisons (3k faces × 35 comps/face), labeled by **`gemini-2.5-flash` + `pairwise-v3-gt`**, with human spot-check on a sample before and after scale.

**Current stage (2026-06-30):** Dry run **passed** (§4.3). **Full 52.5k queue not yet created** — next action is **Create full-scale queue** on sample page, then **Start GT labeling batch** on that run.

---

### 4.1 Pair queue design

**Status:** Complete on approved sample — dry run generated; full run pending.

#### Overview

Ground-truth labeling needs ~52.5k pairwise comparisons on the approved 3k sample: each face appears in **k = 35** undirected edges, **same gender only**, with left/right layout frozen before any model sees the pair. The queue is built **before** VLM labeling and stored in `vlm_pilot_comparisons`. Bradley–Terry refit (§5.1) assumes a sparse, connected graph with balanced opponent exposure — not a uniform random draw of pairs.

#### Methodology

**Implementation:** `src/lib/vlm-pilot/pair-queue.ts` · version **`bt-stratified-v2`** · invoked from `createGtLabelingRun` (`gt-run.ts`).

**Decile bins (proxy for strength).** Each face carries a gender-specific decile D1–D10 from the stratified sample draw (Labs overall score rank within gender). Deciles are coarse — not perfect attractiveness — but they stratify the pool so we can **balance who plays whom** without peeking at BT parameters.

**Per-face edge budget (k = 35).** Each quota slot is filled with a pair whose *actual* decile relationship matches the slot when possible:

| Slot | Share of k | Target count | Pair type |
|------|------------|--------------|-----------|
| Cross-decile | 30% | ~11 | Opponent in a different decile |
| Within-decile | 40% | ~14 | Opponent in the same decile |
| Flex | ~30% | ~10 | Fills remaining degree; prioritizes under-covered decile-pair cells |

**Why opponent decile balance matters for BT.** Bradley–Terry jointly estimates a latent strength θᵢ for every face. A face that only faces weak opponents will win often, but that does **not** automatically inflate θᵢ — wins are down-weighted against weak θⱼ. The real risk of imbalanced opponents is **variance and slow convergence**: some nodes get less informative comparisons than others. We therefore **balance opponent-decile exposure per face** (each face should meet opponents across many deciles, not always D1 vs D10) and **balance global decile-pair cell counts** (the 55 undirected cells Dᵢ×Dⱼ, i ≤ j, per gender).

**Algorithm (deterministic, seeded).** Process male and female pools separately.

1. **Connectivity seed** — Within each decile, chain faces (sorted by id). Between adjacent deciles Dd and Dd+1, add one bridge edge (hash-picked endpoints). Ensures the comparison graph is connected in score space, not merely by UUID order.

2. **Greedy fill to k** — Repeated rounds; faces with lowest degree first. For each needy face, choose an opponent decile in priority order:
   - Lowest per-face opponent-decile count (spread exposure).
   - Lowest global decile-pair cell count (spread Dᵢ×Dⱼ coverage).
   - For cross / flex cross: prefer mid-range decile gaps (|Δ| ∈ [2, 5]) over extreme blowouts — a BT information heuristic (near-50/50 matchups carry more Fisher information than D1 vs D10 walkovers).
   - Tie-break with `sha256(seed:…)` so the queue is reproducible.

3. **Opponent within decile** — Among valid candidates (same gender, not already paired, degree headroom), prefer lower degree, then lower cell load; hash tie-break.

4. **Left/right debiasing** — Canonical storage uses sorted face ids (`faceAId < faceBId`). Display side is `hash(seed, layout, pairIndex) % 2` → `showFaceAOnLeft`, frozen at queue generation.

**Dry run cap.** `gt_dry_run` stops at 1,000 total pairs (same algorithm, truncated). Full run targets N×k/2 = 52,500.

**Stored spec.** Each run persists `pairQueueSpecJson` (algorithm version, k, shares, stats: cross/within/flex counts, degree min/max/avg, decile-pair cell spread).

#### Results

**GT dry run** (`gt_dry_run`) on approved sample `cmqz06xr20001bydxaazuojp8`:

| Field | Value |
|-------|-------|
| Run ID | `cmr00irf500018ldzuqsavzlh` |
| Seed | `gt-label-dry-v1` |
| Pairs generated | 1,000 (cap on full k=35 queue) |
| Algorithm | `bt-stratified-v2` |
| Active faces | 3,000 |
| Created | 2026-06-30T02:13:28Z |

Dry-run queue stats (truncated graph — expected lower cross-decile mix at 1k cap): 1,000 pairs stored; batch connectivity seed dominated within-decile edges at this cap.

**Full scale (`gt_full`):** **complete** — **52,500** pairs · run `cmr1mr0m7000196d57zi3vcgn` · VLM labeled 2026-07-03.

#### Artifacts

- Code: `src/lib/vlm-pilot/pair-queue.ts` (`PAIR_QUEUE_ALGORITHM_VERSION = bt-stratified-v2`)
- Approved sample: `cmqz06xr20001bydxaazuojp8` · seed `gt-draw-v1`
- Dry run: `cmr00irf500018ldzuqsavzlh` · seed `gt-label-dry-v1`

---

### 4.2 Main labeling run (scale batch)

**Status:** Full **52,500 / 52,500** VLM complete (2026-07-03). Audit gate passed 2026-07-04 — ready for §5.1 export.

#### Overview

Largest data-collection step: VLM labels every comparison in the run (~1k dry run, ~52.5k full). Batching is idempotent — completed pairs are skipped on resume.

#### Methodology

**Current (v1):** Admin **Start GT labeling batch** → server processes 250 pairs per invocation, chains in `after()` until pending = 0. If the process stops (dev server closed, deploy, timeout), **re-click Start** — safe because `(comparisonId, vlmModel, promptVersion)` is unique and only missing results are queued.

**Full 52.5k (local research session):**

1. Sample page → **Create full-scale queue** (seed `gt-label-full-v1`).
2. Run → VLM tab → **Start GT labeling batch** with **`pairwise-v3-gt`** (auto for GT runs).
3. Keep laptop awake; **Resume** if worker stalls.

**Labeler:** **`pairwise-v3-gt`** · **`gemini-2.5-flash`** (single model, no panel).

#### Results

**Dry run VLM batch** (2026-06-30):

| Metric | Value |
|--------|-------|
| Run | `cmr00irf500018ldzuqsavzlh` |
| Prompt / model | `pairwise-v2` · `gemini-2.5-flash` |
| VLM calls completed | **998 / 1,000** |
| Batch | `cmr01nepc0001atdkc7a3ery5` |
| Spend (v2 dry) | ~**$1.82** |

**Full GT run** (2026-07-03):

| Metric | Value |
|--------|-------|
| Run ID | `cmr1mr0m7000196d57zi3vcgn` |
| Purpose / seed | `gt_full` · `gt-label-full-v1` |
| Pairs queued | **52,500** (3,000 faces × k=35 / 2) |
| VLM labeled | **52,500 / 52,500** (`vlm_complete`) |
| Prompt / model | **`pairwise-v3-gt`** · **`gemini-2.5-flash`** |
| AUTO_SKIPPED | **35** (one poison face `cmqxa1yun0lhe7hdpzb92acyj` — Invalid JSON from Gemini) |
| Total batch spend | ~**$123.54** |
| Metric snapshots | **0** (P2035 on batch completion — all pairwise labels persisted; snapshot job needs chunking fix) |

#### Decision

**Go** to human GT audit (§4.3) and export prep. Exclude poison/synthetic matchups via Review **Exclude from GT export** before Bradley–Terry.

#### Artifacts

- Full run: `cmr1mr0m7000196d57zi3vcgn`
- Export: `GET /api/admin/pairwise/runs/cmr1mr0m7000196d57zi3vcgn/export?promptVersion=pairwise-v3-gt&offset=&limit=500`

---

### 4.3 Human review & override analysis

**Status:** Complete — go/no-go passed (2026-07-04). Dry-run audit archived below.

#### Overview

After the full VLM batch finished, we spot-checked Gemini’s pairwise labels against a researcher’s judgment before trusting the run for Bradley–Terry. The goal was not to re-label 52.5k pairs by hand, but to estimate how often the model is wrong, remove known-bad edges, and decide whether the remaining labels are good enough to export.

#### Research question

On the full **`pairwise-v3-gt`** run, does **`gemini-2.5-flash`** agree with a human auditor often enough (~85%+, ties excluded) that we can export ~52k edges for BT refit without a full manual re-label?

#### Methodology

Auditing happened in the Pairwise Lab **Review** tab on run `cmr1mr0m7000196d57zi3vcgn`. The workflow had three layers.

First, we **excluded bad matchups** rather than deleting them. Synthetic faces, poison-face AUTO_SKIPPED edges, and other unusable pairs were soft-dropped with **Exclude from GT export** (`gtExportExcludedAt`), so they stay in the DB for traceability but drop out of export and BT.

Second, we ran a **risk pass** on low- and medium-confidence VLM picks before the main sample. That pass intentionally showed a lower agreement rate (~72% on low/med-only filters) because it surfaces the hardest comparisons first.

Third, we audited a **seeded random sample** of the full run. The UI draws `N` pair indices deterministically from run seed `gt-label-full-v1` (same logic as `buildAuditSamplePairIndices`). For each pair the researcher clicked **Correct — model right** or **Wrong — pick winner**; labels persist on `humanOutcome`, `humanWinnerFaceId`, and `humanLabeledAt`. Accuracy is computed only on pairs where both human and VLM picked a clear winner (ties excluded). Overrides include tie disagreements.

Final audit configuration: **1,000-pair seeded sample**, **all confidence levels**, through **2026-07-04**. Numbers below were pulled from the live DB on 2026-07-04.

#### Success criteria

Pre-registered gate: **≥~85%** sample agreement (ties excluded) on a random audit of hundreds–low thousands of pairs; extrapolated full-run overrides in the **low thousands**, not ~10k+. Dry run on `pairwise-v2` had hit **~92%** on 100 pairs (§4.3 archive below).

#### Results

**Export pool after removals.** The run still has **52,500** comparison rows. **86** matchups are excluded from export (all tagged `synthetic`, including the 35 AUTO_SKIPPED poison-face edges). **52,414** pairs remain export-eligible.

**Main audit sample (1,000 seeded random).** The researcher completed **750 of 1,000** pairs in this sample. On those 750:

- **121 overrides** (researcher disagreed with the model)
- **84.9% accuracy** (ties excluded) — effectively at the **≥85%** gate
- **~8,470 extrapolated overrides** across the full 52.5k if the same error rate held everywhere (likely pessimistic)

**All human labels on the run (1,032 pairs total).** This includes the 750 sample audits plus ~282 extra labels from the earlier low/medium risk pass outside the sample. Run-wide accuracy on all 1,032 reviewed pairs is **83.3%** (193 overrides).

**Accuracy by VLM confidence (on all 1,032 human-labeled pairs).** High-confidence picks hold up best; medium is the weak tier:

| Confidence | Audited | Accuracy (ties excl.) | Overrides |
|------------|---------|------------------------|-----------|
| High | 723 | **87.0%** (620 / 713 scored) | 102 |
| Medium | 261 | **74.1%** (189 / 255 scored) | 72 |
| Low | 48 | **76.3%** (29 / 38 scored) | 19 |

**Dry-run reference (2026-06-30, `pairwise-v2`, 100-pair sample):** ~92% agreement, ~8% extrapolated override rate — motivated the switch to **`pairwise-v3-gt`** for full scale.

#### Decision

**Go** to Bradley–Terry export on **52,414** export-eligible pairs.

The 750-pair sample hit **84.9%**, meeting the ~85% gate within rounding. High-confidence labels audit at **~87%**, which is acceptable for BT at this scale. Medium-confidence edges drive most disagreement; consider **excluding or down-weighting medium (and low) confidence** in the first BT export if refit stability looks noisy — not required to block progress.

No further manual audit is needed unless BT diagnostics show systematic failure modes (connectivity, face-level clusters of error).

#### Artifacts

- Run: `cmr1mr0m7000196d57zi3vcgn` · seed `gt-label-full-v1` · status `vlm_complete`
- Review UI: `/research/pairwise/runs/cmr1mr0m7000196d57zi3vcgn/review`
- Export: `GET /api/admin/pairwise/runs/cmr1mr0m7000196d57zi3vcgn/export?promptVersion=pairwise-v3-gt`
- DB snapshot: 2026-07-04 (live query) — 86 excluded, 52,414 export-eligible, 1,032 human-labeled, 750/1,000 sample reviewed at 84.9%

#### Notes

- Review UI pagination bug (blank screen at high page offsets) was fixed 2026-07-04; remaining 250 sample pairs were not required for go/no-go.
- Gender mislabels in spot-checks (~handful): accepted as low risk per §3.2.
- Next: §5.1 Bradley–Terry refit on filtered export.

---

**Archive — dry-run audit (2026-06-30)**

On dry run `cmr00irf500018ldzuqsavzlh` with prompt `pairwise-v2`, a 100-pair audit yielded ~92% agreement and ~4,200 extrapolated overrides on 52.5k. That result, plus rationale quality issues on v2, led to **`pairwise-v3-gt`** for the full run.

---

## 5. Phase 3 — Rankings, calibration & training

Bradley–Terry refit on finalized comparisons, calibration to `/10`, preference model training in `faceiq-preference-ml`.

---

### 5.0 Pre-refit face photo QC (VLM)

**Status:** **Complete** (2026-07-27, `artifacts/face-qc-v1/`). **133 of 3,000 faces removed.**
Precondition for every refit from `bt-refit-v2-qc` onward, and for all panel draws.

#### Overview

§3.3's pre-label audit sampled the cohort; it did not screen every face. Raters in the pilot
then flagged what a sample misses — faces that look AI-generated, and photos where the recorded
gender is wrong. Both contaminate everything downstream: a mislabelled gender puts a face in the
wrong same-gender comparison pool, so its θ is fitted against opponents it should never have met.

#### Methodology

`scripts/qc_faces.py` sends every cohort face to `gemini-2.5-flash` for structured metadata —
usability, apparent gender, whether the image is a real photograph, apparent age band, quality
defects, plus apparent ethnicity and Fitzpatrick tone for cohort description. Flagged faces go to
a human review sheet (`scripts/qc_review_sheet.py`, `app/qc_review.py`); `scripts/qc_handoff.py`
turns confirmed decisions into two CSVs. Faces in **either** CSV are dropped from the fit.

#### Results

2,998 faces scored (2 VLM errors), **508 flagged** for review:

| flag | n | | quality defect (non-blocking) | n |
|---|--:|---|---|--:|
| possible minor | 307 | | obstruction | 288 |
| screenshot | 82 | | low resolution | 239 |
| not a real photo | 66 | | screenshot | 234 |
| gender mismatch | 54 | | poor lighting | 184 |
| gender ambiguous | 21 | | heavy filter | 53 |
| multiple faces | 21 | | multiple faces | 28 |

Confirmed removals: **71** unusable (33 not a real photo, 31 possible minor, 5 gender-ambiguous,
2 multiple faces) plus **62** with mislabelled gender (51 actually male, 11 female). The gender
group is *dropped rather than reassigned* — every one of their matchups was drawn inside the wrong
gender pool, so there is nothing valid to re-fit them on. Total **133 faces, 4,500 matchups**;
47,914 of 52,414 survive.

Cohort description from the same pass (never a training input): apparent ethnicity 60% white, 10%
middle-eastern, 9% hispanic, 7% east-asian, 5% mixed, 5% black, 3% south-asian; Fitzpatrick
II–III 74%; apparent age 63% 18–24.

`bt-refit-v2-qc` on the cleaned set: all gates passed, ρ_stab 0.9921 F / 0.9920 M, ρ vs Labs
0.746 F / 0.760 M, 1,422 F + 1,444 M faces scored.

#### Decision

**Exclude all 133.** Photos also fed to the panel draws, so `select_panel_pairs.py` and
`select_topup_pairs.py` filter on the same CSVs — no rater ever saw an excluded face.
Corresponding faces were disabled in faceiq-labs via `apply-gt-face-decisions.ts`.

#### Artifacts

- `artifacts/face-qc-v1/{results.jsonl,flags.csv,summary.json,decisions.jsonl}`
- `artifacts/face-qc-v1/{exclude-faces.csv,gender-fixes.csv}` — the two inputs every refit reads
- `scripts/{qc_faces,qc_review_sheet,qc_handoff,join_face_attributes}.py`, `app/qc_review.py`

#### Notes

Run 3 raters still reported occasional AI-looking faces, so the `not_real_photo` screen is not
airtight — tracked as an open item in `panel-pilot-runbook.md`. The 307 "possible minor" flags were
mostly false positives on young-looking adults; only 31 survived human review, but the flag is
worth keeping at that precision given the consequence of a miss.

---

### 5.1 Bradley–Terry refit & stability

**Status:** First refit **complete — all gates passed** (2026-07-04, `faceiq-preference-ml` · `artifacts/bt-refit-v1`).
**Superseded four times since; the ranking of record is now `artifacts/bt-refit-v5-panel`** (2026-08-04,
§5.8). All gates passed at every step. The methodology below still describes the v1 fit; the later refits
change the *inputs* and, from v3 on, add a per-source discrimination term (§5.5).

⚠️ **`bt-refit-v2-qc` is not superseded for one specific job.** It is the last fit that predates every
human vote, so it is the only ranking that may be used to cut bands, strata or pair draws, and to
calibrate against panel votes. Using a panel-fitted refit for those inverts the answer (§5.7).

| refit | date | change | ρ vs previous |
|---|---|---|--:|
| `bt-refit-v1` | 2026-07-04 | first fit, VLM labels + 1,032 audited rows | — |
| `bt-refit-v1-cap95` | 2026-07-05 | display-only recalibration, θ byte-identical | 1.000 |
| `bt-refit-v2-qc` | 2026-07-2x | **133 faces excluded** by VLM photo QC (AI-looking, mislabelled gender, unusable) + gender fixes | — |
| `bt-refit-v3-panel` | 2026-07-31 | + 372 raters' votes on 2,980 pairs, joint fit, β_human free | 0.991 |
| `bt-refit-v4-panel` | 2026-08-01 | + 301 more raters on 4,800 hard pairs (pooled) | 0.9905 |
| **`bt-refit-v5-panel`** | **2026-08-04** | **+ 296 more raters on 2,500 uniform random pairs — 10,280 pairs / 95,245 votes / 963 raters total** | **0.9667 vs v2-qc** |

#### Overview

Bradley–Terry turns ~52k pairwise outcomes into a **relative strength θ** per face. This is the authoritative ground-truth ranking for the 3k cohort. The refit runs on **finalized winners** from the export, not raw VLM rows in isolation.

#### Methodology

**Input:** Export of run `cmr1mr0m7000196d57zi3vcgn` produced by `faceiq-labs/scripts/export-gt-run.ts` — **52,414** eligible rows as JSONL shards + `manifest.json` (sha256-verified) + all 3,000 face photos, landing in `faceiq-preference-ml/data/exports/`. For each row, the BT winner is:

- **`humanWinnerFaceId`** when `humanLabeledAt` is set (1,032 audited rows), else  
- **`vlmWinnerFaceId`** from `pairwise-v3-gt` / `gemini-2.5-flash`

**Tie rule (locked at first refit):** half-win for each side (0.5/0.5); 113 ties in the export.

**Fit:** custom regularized MM algorithm (Hunter 2004; α=0.01 virtual games) in `faceiq-preference-ml/src/faceiq_pref/bt.py`. **Male and female fitted separately** (same-gender queue → cross-gender graph disconnected by construction). Faces with **< 15** resolved comparisons are dropped pre-fit (iteratively) rather than scored on thin evidence.

**Model & estimation (formal).** Each face $i$ carries a latent strength $p_i > 0$; reported scores are $\theta_i = \log p_i$. The Bradley–Terry model puts

$$
\Pr(i \succ j) \;=\; \frac{p_i}{p_i + p_j}.
$$

Let $w_{ij}$ be the weighted win count of $i$ over $j$ (1 per decisive win; a tie contributes $\tfrac{1}{2}$ to each direction) and $n_{ij} = w_{ij} + w_{ji}$ the total games between $i$ and $j$. The log-likelihood is

$$
\ell(\mathbf{p}) \;=\; \sum_{i \neq j} w_{ij} \bigl[ \log p_i - \log (p_i + p_j) \bigr],
$$

maximized by the MM (minorize–maximize) fixed-point iteration of Hunter (2004), augmented with $\alpha$ virtual wins and $\alpha$ virtual losses against a pseudo-opponent of fixed strength $1$:

$$
p_i^{(t+1)} \;=\; \frac{W_i + \alpha}{\dfrac{2\alpha}{p_i^{(t)} + 1} \;+\; \displaystyle\sum_{j \neq i} \frac{n_{ij}}{p_i^{(t)} + p_j^{(t)}}},
\qquad W_i = \sum_{j \neq i} w_{ij},
\qquad \alpha = 0.01 .
$$

The regularization keeps the MLE finite for undefeated faces (unregularized, $\hat{p}_i \to \infty$). Each MM step provably does not decrease $\ell$. Strengths are identifiable only up to scale, so each iterate is renormalized to geometric mean $1$ (equivalently $\sum_i \theta_i = 0$); convergence is declared when $\max_i \lvert \theta_i^{(t+1)} - \theta_i^{(t)} \rvert < 10^{-6}$.

**Stability gate (core §14):** Refit twice on independent Bernoulli(0.8) subsamples of pairs → Spearman ρ between the resulting θ rank lists **> 0.95**. Also: graph connectivity, no face with < 15 resolved comparisons.

> **This gate is a whole-ranking gate, not a per-face one.** §5.6 shows the two are far apart: at ρ = 0.9931 the median face's rank is still uncertain by **241 places** and its /10 score by **±0.45**, because ρ is dominated by the correctly ordered extremes. Read ρ as "the ranking is reproducible in aggregate", never as "this face's number is right".

#### Success criteria

- Connected comparison graph per gender  
- Stability ρ > 0.95 on 80% subsamples  
- Spearman ρ vs Labs `overall_score` documented (sanity, not training target)

#### Results

First refit (2026-07-04, no confidence filter):

| Metric | Female | Male |
|--------|--------|------|
| Faces scored | 1,499 | 1,500 |
| Graph components | **1** (connected) | **1** (connected) |
| Faces below 15 comparisons | 0 (1 dropped — see note) | 0 |
| **Stability ρ (80% subsample)** | **0.9925** | **0.9925** |
| Spearman ρ vs Labs `overall_score` | 0.745 | 0.752 |

The single dropped female face is the **poison face** `cmqxa1yun0lhe7hdpzb92acyj` — its 35 AUTO_SKIPPED edges were excluded from export, leaving 11 resolved games (below the 15 floor). Expected per §4.3; BT correctly refuses to score it.

Moderate ρ ≈ 0.75 vs Labs is the healthy regime: correlated with the legacy formula but not reproducing it.

**Independent cross-check (2026-07-04):** refitting the same edges with `choix.ilsr_pairwise` (ILSR, a different BT estimator) reproduces the shipped ranking at Spearman ρ = **0.9995** (female) / **0.9994** (male). The two libraries' regularization parameters are not comparable (choix's `alpha=0.01` shrinks far harder than our virtual-game α=0.01; near-zero values recover the common MLE), and choix cannot weight tie edges — refitting our MM on the same tie-dropped edges matches the shipped refit at ρ = 0.99997, confirming tie handling is immaterial. Notebook: `faceiq-preference-ml/notebooks/explore.ipynb`.

#### Decision

**Go** — BT ranking accepted as GT v1 (2,999 faces). Calibrated /10 scores written alongside θ. Next: preference comparator training (§5.3); optionally re-run with `--confidence high` as an ablation.

#### Artifacts

- Export: `faceiq-preference-ml/data/exports/cmr1mr0m7000196d57zi3vcgn/` — 6 JSONL shards, manifest, 3,000 images (174 MB, not in git)
- Exporter: `faceiq-labs/scripts/export-gt-run.ts` (idempotent, resumable image download)
- Refit: `faceiq-preference-ml/artifacts/bt-refit-v1/{ratings.csv,metrics.json}` · CLI `scripts/run_bt.py`
- Dashboard: `streamlit run app/dashboard.py` (rankings, diagnostics, calibration, training)

---

### 5.2 Calibration & validation vs Labs overall_score

**Status:** **Complete** (2026-07-04 — ran as part of `bt-refit-v1` inside `scripts/run_bt.py`).

#### Methodology

BT strengths $\theta$ have an arbitrary scale — only their order is meaningful — so the `/10` score is defined through the **percentile rank**, mapped onto the **pre-registered anchor curve** (core §12, locked before any results were seen).

**Percentile rank.** Within each gender, for face $i$ among $n$ scored faces:

$$
r_i \;=\; \frac{\operatorname{rank}(\theta_i) - 1}{n - 1} \;\in\; [0, 1],
$$

where $\operatorname{rank}$ is ascending (1 = weakest).

**Score curve.** $s_i = f(r_i)$, with $f$ the monotone PCHIP (piecewise-cubic Hermite) interpolant through the pre-registered anchors, clipped to $[0, 10]$:

| Percentile $r$ | 0.0 | 0.10 | 0.50 | 0.90 | 0.99 | 0.999 | 1.0 |
|---|---|---|---|---|---|---|---|
| Score $f(r)$ | 1.0 | 3.0 | 5.0 | 7.0 | 8.0 | 8.5 | 9.0 |

Monotonicity of PCHIP guarantees the score preserves the BT order exactly. The right tail compresses deliberately: with ~1,500 faces per gender the single best face sits at roughly the top $0.07$ percentile, so the sample cannot resolve rarer tails (a 1-in-100k face); capping at 9.0 avoids extrapolating beyond the data. Implementation: `faceiq-preference-ml/src/faceiq_pref/calibrate.py`.

**Validation vs Labs.** Spearman ρ between $\theta$ and the legacy Labs `overall_score` per gender (`src/faceiq_pref/validate.py`) — a sanity check, never a training target: ρ ≈ 1 would mean the GT merely reproduced the formula being replaced; ρ ≈ 0 would indicate a defect.

#### Results

- Calibrated `scoreOutOf10` written for all 2,999 scored faces alongside θ and percentile in `artifacts/bt-refit-v1/ratings.csv`; median = 5.0 and max = 9.0 per gender by construction.
- **Spearman ρ vs Labs `overall_score`: 0.745 (female, n=1,499) / 0.752 (male, n=1,500)** — the intended moderate regime (correlated with, but not duplicating, the legacy formula). Computed over **all scored faces** (not a subsample; the 80% subsample protocol is the §5.1 stability gate only).

#### Extended BT-vs-Labs analytics (2026-07-04, full population)

Rank agreement, score-scale agreement, and localization of disagreement — all faces, per gender:

| Metric | Female (n=1,499) | Male (n=1,500) |
|---|---|---|
| Spearman ρ (rank) | 0.745 | 0.752 |
| Kendall τ (rank) | 0.561 | 0.566 |
| Pearson r (`scoreOutOf10` vs Labs) | 0.742 | 0.754 |
| Labs score mean ± std | 6.31 ± 1.10 | 6.13 ± 1.14 |
| BT `/10` mean ± std | 4.98 ± 1.54 | 4.98 ± 1.54 |
| Mean signed diff (BT − Labs) | −1.33 | −1.15 |
| Median absolute rank displacement | 158 / 1,499 | 152 / 1,500 |
| p90 absolute rank displacement | 530 | 532 |
| Top-50 set overlap | 36% | 48% |
| Top-150 set overlap | 59% | 70% |
| Bottom-150 set overlap | 49% | 45% |
| Decile-bin agreement (exact / ±1) | 27.5% / 61.4% | 29.7% / 61.1% |

Within-decile Spearman (θ vs Labs, inside each Labs decile): ~0.24–0.41 at the extremes (D1–D2, D9–D10) but **~0.0–0.15 across the middle (D4–D7)**.

**Interpretation:**

- Pearson on calibrated scores ≈ Spearman on ranks (both ~0.75), so rank-level and score-level agreement tell the same story; the calibrated score is a monotone function of rank, and both scales are roughly linearly related.
- The absolute offset (Labs mean 6.1–6.3 vs BT median 5.0) is **by construction, not a finding**: the pre-registered curve pins the median face at 5.0, while the legacy formula is inflated (its median sits above 6). Absolute BT−Labs differences are therefore not meaningful; only order comparisons are.
- Agreement is concentrated at the **tails**: the two systems largely concur on who is clearly high or low (D1–D2, D9–D10 within-decile ρ up to 0.41; top-150 overlap 59–70%), but are **near-uncorrelated inside the middle deciles** — the fine ordering of average faces is where the pairwise GT diverges most from the formula, and where it plausibly adds the most new information (middle deciles are also where per-pair label noise is highest, per the §4.3 audit).
- Top-50 overlap of 36–48% means the very top of the hierarchy is substantially re-ranked, not just reshuffled by noise — median displacement is ~155 rank positions (~10% of the list), p90 ~530.

These are descriptive diagnostics, not gates; no action required. Script inline in session notes; numbers reproducible from `ratings.csv` alone.

#### Elite crowding — cohort design concern (2026-07-05)

**Observation:** Top-decile faces (celebrities, professional models) often read **~7.3–7.5 /10** despite Labs scores of 8+ and high win rates (e.g. ~86% in 35 matchups). Dashboard review flagged Sean O'Pry–tier and Taylor Hill–tier faces in the **~95th percentile** band, not the 99th+ band where scores reach 8.0+.

**Mechanism (not a BT bug):** Two stacked effects:

1. **Cohort-relative ranking.** BT θ and percentile are computed **within this 3k sample**. ~22% of each gender sits in Labs deciles 9–10 (~330 faces); ~186 males are decile-10 alone. Elites compete mostly against other elites (~35 same-gender matchups each), so even an 85%+ win rate can land at only the ~95th percentile when 70+ faces rank higher. In a general-population sample with few elites, the same face would sweep most matchups and sit at 99th+ percentile.
2. **Calibration maps percentile → /10**, not absolute world attractiveness. Pre-registered anchors pin 90th → 7.0, 99th → 8.0, so 95th → ~7.4 **by construction**. The `/10` is a display label on cohort rank, not a claim about "deserves 8+ globally."

**Impact on comparator training:** **Low risk for the training objective.** The neural model learns **pairwise winners only** — it never sees `/10`. It learns *relative taste* ("this elite face beats that elite face"), which is valuable GT signal. A low displayed `/10` on a celebrity does **not** teach the model "this person is a 7.5"; it only reflects where that face sits in **this** crowded elite pool. Matchup accuracy and θ ordering are what matter for training and for rank-agreement evaluation (Kendall τ vs BT).

**Impact on production `/10`:** **Real concern if research-cohort `/10` is shipped directly to users.** Production absolute scores should come from **reference-set inference** (§5.4): compare the new face against an anchor panel with **product-facing** known `/10` values, then aggregate implied scores. The comparator supplies relative judgments; anchors supply the absolute scale.

**Future data gathering (recommended):**

| Issue | Current v1 cohort | Suggested v2+ |
|-------|---------------------|---------------|
| Tail density | ~22% in deciles 9–10 per gender (stratified for balance) | **Thinner tails** — fewer faces at the top (and bottom) deciles; avoid near-uniform decile fill |
| Elite block | Many professional / hyper-attractive faces in one pool | **Manual curation** — cap elite count; separate "elite ladder" subset if fine top-end ordering is needed |
| Sampling goal | Learn preference across full decile range | Match sampling to **production population** (mostly average users), with a small held-out elite ladder for top-end calibration |

**Recalibration stopgap (e.g. map 95th → 8.0):** **Not pursued for v1.** Arbitrary without a production anchor policy; does not affect training; ranking θ unchanged. Acceptable to defer until a deliberate anchor ladder or v2 cohort exists. The `cap95` variant (max 9.5 only) remains an optional display CSV; v1 pre-registered curve stays canonical for the paper.

**What actually matters for v1 → production:**

1. **Matchup label quality** (~85% VLM agreement; human overrides where audited) — primary training signal.
2. **BT θ ranking** — authoritative relative GT; stability gates passed.
3. **Production anchor panel** — manually curated and/or BT-fitted on a **production-representative** cohort; anchors may assign known `/10` (e.g. a reference face fixed at 8.5) independent of where that face percentile-sits in the research export.

#### Decision

- Calibration accepted as pre-registered; **no post-hoc anchor adjustments** to the canonical v1 curve.
- **Amendment (2026-07-05) — `cap95` display variant.** A second, explicitly-labeled anchor set was added with the max anchor raised 9.0 → 9.5 (all other anchors identical). Rationale: the cohort includes professional top-model faces, so the sample maximum plausibly represents a rarer tail than 1-in-3,000. Effects verified: **ranking and θ byte-identical**; only 22 / 2,999 faces change score by > 0.005 (all above the ~99.6th percentile); median stays 5.0. Produced by `scripts/recalibrate.py --refit artifacts/bt-refit-v1 --anchors cap95` → `artifacts/bt-refit-v1-cap95/` (no BT refit involved; fully revertible by preferring the v1 CSV). v1 remains the pre-registered curve of record for the paper; cap95 is the product-facing display variant.
- **Persistence:** refit and training results remain **filesystem-versioned in the ML repo** (`artifacts/bt-refit-vN/`, `artifacts/train-vN/`) and are browsed via the Streamlit dashboard. No local database. Persisting into the labs DB (schema §3.3 `VlmPilotBtRefit` / `VlmPilotFaceRating`) is **deferred** to a future one-shot `faceiq-labs/scripts/import-bt-refit.ts` that reads `ratings.csv` + `metrics.json`, run once per *accepted* refit — the ML repo itself never touches the DB.

#### Artifacts

- `faceiq-preference-ml/artifacts/bt-refit-v1/{ratings.csv, metrics.json}` (anchors recorded in `metrics.json`)
- Curve + validation code: `src/faceiq_pref/{calibrate.py, validate.py}`
- Dashboard (calibration curve, BT-vs-Labs scatter, rankings with photos): `streamlit run app/dashboard.py --server.port 8502`

---

### 5.3 Preference model training

**Status:** In progress — 10 training runs + 1 ensemble complete (2026-07-08 → 07-12). Best **single-model** val pairwise accuracy: **78.4%** (train-v8, ArcFace R50 e2e, best @ ep 5 of 16). Best **rank agreement**: ensemble-v7-v1 (τ 0.858, ρ 0.967). Training moved to EC2 g5.xlarge (A10G) 2026-07-12 — ~45 min per 8-ep ArcFace e2e run vs ~5 h on Mac MPS; v8 long run ~90 min on GPU.

> **⚠️ 2026-08-01 — that 78.4% is measured against the wrong yardstick, and the comparator does not
> beat it.** Val accuracy is scored against export `finalOutcome`, which is Gemini's label on ~98% of
> rows, so "78.4%" means *"it reproduces Gemini"*. §5.5 established that Gemini is at chance on close
> pairs, so the metric cannot see the model's blind spot. Measured for the first time against real human
> votes (`scripts/eval_vs_panel.py`, on panel pairs with **both** faces in the checkpoint's own val split):
>
> | predictor | agrees with human votes |
> |---|--:|
> | comparator `train-v10` (1,928 pairs, 16,376 votes) | 54.29% |
> | the Gemini labels it trained on | 54.43% |
> | BT from Gemini labels only (`v2-qc`) | 55.02% |
> | ceiling — another rater | 59.09% |
>
> By gap band the comparator sits at 51–55% under 20 percentile points and only reaches 79% on obvious
> pairs. `train-v8` agrees: 53.1% on its smaller leak-free slice. **The network learned the VLM's
> function, blind spot included.**
>
> Root cause: `PairDataset` built labels from `final_outcome`, so **none of the 65,894 human votes ever
> entered a training batch**. Fixed 2026-08-01 — `panel_labels` in `TrainConfig` +
> `src/faceiq_pref/panel.py` now supply a **vote share** as a soft target on the train split only (val
> keeps export labels, so `val_accuracy` stays comparable across runs). Arms run:
> `train-v12-panel-soft` (vote share) and `train-v13-panel-hard` (crowd winner only, rounded), both
> cloned from `train-v10-arcface-val50` so **v10 is a control we already paid for**.
>
> `val_fraction: 0.5` in those configs is deliberate: 1,934 panel pairs are leak-free at 0.5 versus 333
> at 0.2, which is the difference between ±1.1 and ±2.7 points of measurement error. The cost is training
> on 13.1k pairs instead of 33.5k, so **absolute accuracy from these arms is not the shippable number** —
> refit the winning label scheme at 0.2 for the model we ship.

##### 5.3.1 Comparator trained on human targets — result (2026-08-01)

**The label spend reaches the model.** Both panel arms beat the Gemini labels they replaced, on the
same 1,928 leak-free pairs / 16,376 votes used above. Intervals are a **paired bootstrap over pairs**
(`scripts/compare_panel_evals.py`, 10,000 resamples): the arms are scored on identical pairs, and
resampling *votes* instead of pairs would shrink intervals by ≈√12 and manufacture significance,
since 12 raters on one pair are not 12 independent observations of model skill.

| run | train label | vs humans | val acc | vs Gemini labels (paired) |
|---|---|--:|--:|---|
| `train-v10-arcface-val50` (control) | export `finalOutcome` | 54.29% | 77.07% | −0.13 pts, CI [−1.55, +1.31] — n.s. |
| `train-v12-panel-soft` | vote share (float) | 55.84% | 77.53% | +1.41 pts, CI [−0.03, +2.89] |
| **`train-v13-panel-hard`** | crowd winner (rounded) | **56.42%** | 77.46% | **+2.00 pts, CI [+0.60, +3.41]** ✅ |
| ceiling — another rater | — | 59.09% | — | — |

Head-to-head, paired: v13 − v10 = **+2.13 pts** CI [+1.24, +3.02]; v12 − v10 = **+1.54 pts**
CI [+0.66, +2.44]. Both separate from the control. **v13 − v12 = +0.59 pts CI [−0.28, +1.47] — not
separable.** The Gemini→ceiling gap is 4.66 pts and v13 closes **43%** of it.

Four findings worth carrying:

1. **Correcting the winner is what pays; the float target is unproven.** The ablation was designed to
   split "which face" from "by how much" and it came out a tie. Prefer the simpler hard-majority
   target until something distinguishes them — the "admit uncertainty" hypothesis is *not* supported.
2. **`val_accuracy` cannot see this.** All three runs sit at 77.1–77.5% because val keeps Gemini's
   labels. A model that got materially better at predicting humans moved val accuracy by 0.4 points,
   inside noise. This is the case for `eval_vs_panel.py` existing.
3. **Small correction, large effect.** Only 1,912 of 13,113 train pairs (14.6%) carry a human target
   and 893 (6.8% of train) flip the export's winner. A 2-point gain off a 6.8% correction rate means
   the corrections land where Gemini was worst, as intended.
4. **Gains are concentrated in the 2–20 percentile-gap band** — close but not coin-flip, exactly what
   the top-up studies bought. On wide gaps the network remains slightly below Gemini (64.0% vs 65.9%
   at 20–45; 78.6% vs 79.7% at 45–100), on 109 and 62 pairs respectively — read as "no gain here",
   not a measured regression.

| percentile gap | pairs | Gemini | `v10` control | `v12` soft | `v13` hard |
|---|--:|--:|--:|--:|--:|
| 0–2 | 383 | 50.7% | 53.1% | 52.5% | 53.3% |
| 2–5 | 510 | 51.4% | 51.6% | 53.4% | **54.7%** |
| 5–10 | 557 | 53.4% | 51.7% | 54.0% | **55.0%** |
| 10–20 | 307 | 53.2% | 54.6% | **56.8%** | 55.9% |
| 20–45 | 109 | 65.9% | 61.0% | 65.1% | 64.0% |
| 45–100 | 62 | 79.7% | 78.6% | 77.1% | 78.6% |

**Not yet done:** refit the hard-majority recipe at `val_fraction: 0.2` for the shippable checkpoint,
and re-run the §6.3 held-out *face* test. `skip_ties` was left at `true` deliberately — it drops only
the 23 rows where Gemini said "tie" (15 panel-covered), 0.8% of the targets, and flipping it would
cost the exact-control property. The near-ties that mattered were never dropped; they arrived as
confident 0/1 labels, which is what the soft targets address.

#### Methodology

**Data flow (labeling → training):**

1. **Labeling** — VLM writes pairwise winners to `vlm_pilot_vlm_results`; optional human audit overrides on `vlm_pilot_comparisons.human*`.
2. **Export** — Paginated `GET /api/admin/pairwise/runs/{runId}/export` (or SQL dump) → local Parquet/JSONL in `~/research-data/scoring-gt/artifacts/`. Export is **after labeling** (or partial for dry run), not live during VLM batch.
3. **Finalize labels** — Training uses `humanWinnerFaceId` when audited, else `vlmWinnerFaceId`. Confidence optional as sample weight.
4. **Bradley–Terry refit (Python, §5.1)** — Separate statistical step on the **same pairwise win/loss edges**; outputs rank/score θᵢ per face. Not neural training.
5. **Neural comparator (faceiq-preference-ml)** — Mini-batch gradient descent over **fixed exported pairs** (image A, image B, winner); many **epochs** over the same dataset. BT ranks used for evaluation and calibration to `/10`, not usually as the primary loss target.

**Architecture justification (recorded 2026-07-04, pre-training):**

- The comparator is a **single-input pairwise model** (a scalar scorer $s(\cdot)$ applied siamese-style; logit $= s(A) - s(B)$), not a pair-input model that ingests both images jointly. Single-input scoring guarantees **transitivity by construction** — real-valued scores cannot produce preference cycles (A ≻ B ≻ C ≻ A), which pair-input models can, and a coherent global ranking is exactly the product goal.
- The loss (sigmoid over the score difference + binary cross-entropy on the winner) is the **RankNet** objective. RankNet's known limitation — it weights all pairwise inversions equally instead of prioritizing top-of-list accuracy (LambdaRank/nDCG-style) — is **acceptable and arguably preferable here**: the goal is a well-calibrated score across the whole face distribution, not a retrieval-style top-k list. LambdaRank considered and rejected on these grounds.
- **Order-symmetry guard:** the two arms share weights exactly, and left/right presentation was randomized upstream at labeling; training should verify no positional bias (e.g. accuracy on A-wins vs B-wins pairs should match). This is the most common silent failure of this architecture.
- Distance-metric siamese variants (contrastive/triplet loss) are a **different branch** of the siamese family aimed at verification/one-shot tasks; they do not apply to scalar-utility learning and are out of scope.

**Training signal vs display scores (2026-07-05):**

- **Training label:** matchup winner only (`finalOutcome`). No `/10`, no Labs `overall_score`.
- **BT `/10` and θ:** evaluation, dashboard, and future anchor metadata — **not** the loss target.
- **Elite crowding** (§5.2) compresses research-cohort `/10` at the top; training still benefits from elite-vs-elite pairwise structure. See §5.4 for how production `/10` is assigned via anchors, not research percentiles.

**Common setup (all runs):** split by face id, seed 42, 80/20 → 33,449 train / 2,068 val pairs scored; ties skipped (113); labels = `finalOutcome` only; RankNet BCE loss; AdamW; MPS (Apple Silicon). Runs are directly comparable — identical split and eval.

#### Results

| Run | Backbone | Mode | Key config | Best ep | Val loss | Val acc | Kendall τ | Spearman ρ |
|---|---|---|---|---|---|---|---|---|
| train-v1 | ResNet-18 (ImageNet) | e2e | 224px, lr 1e-4, wd 1e-4, 10 ep | 7 | 0.525 | 76.6% | 0.832 | 0.956 |
| train-v2 | ResNet-50 (ImageNet) | e2e | wd 5e-4, 8 ep | 2 | 0.495 | 75.4% | 0.801 | 0.944 |
| train-v3 | DINOv2 ViT-S/14 | frozen probe | lr 1e-3, 10 ep | 3 | 0.573 | 72.4% | 0.720 | 0.892 |
| train-v4 | ResNet-18 | e2e | wd 3e-4, 7 ep | 6 | 0.505 | 76.7% | 0.826 | 0.953 |
| train-v6 | ArcFace R50 (w600k) | frozen probe | 112px, lr 1e-3, 10 ep | 4 | 0.568 | 74.6% | 0.805 | 0.940 |
| **train-v7** | ArcFace R50 (w600k) | e2e | 112px, lr 1e-5, bs 32, 8 ep | 8 | 0.467 | 78.1% | 0.842 | 0.962 |
| **train-v8** | **ArcFace R50** | **e2e** | **v7 + 16 ep (GPU)** | **5** | **0.453** | **78.4%** | 0.840 | 0.962 |
| train-v9 | ArcFace R50 | e2e, high-conf train filter | v7 recipe; train 33,449 → 28,422 pairs (high-conf VLM + all human; val unchanged) | 8 | 0.526 | 78.1% | 0.828 | 0.957 |
| ensemble-v7-v1 | ArcFace R50 + ResNet-18 | eval-only ensemble | z-scored per-face scores averaged (`scripts/ensemble_eval.py`) | — | — | 78.1% | **0.858** | **0.967** |

(v5 = DINOv2 e2e config exists, not yet run. v8 was first started on Mac and stopped at ep 2/16 — 77.7%, redundant once GPU came online; full 16 ep completed on GPU 2026-07-12.)

**Findings so far:**

- Signal confirmed on first run (v1: 76.6% vs ~85% label ceiling; ρ 0.96 vs BT).
- Capacity is not the bottleneck: ResNet-50 (v2) < ResNet-18 (v1).
- Frozen embedding probes underperform e2e: DINOv2 probe 72.4%, ArcFace probe 74.6% — pretrained features alone don't encode preference; fine-tuning matters.
- **Face-specific backbone + fine-tune wins:** ArcFace e2e (v7) = 78.1%; extending to 16 epochs (v8) peaked at **ep 5 → 78.4%** (+0.3 pp), then val accuracy drifted down (76.5–77.3% ep 8–12) — ArcFace e2e **does** overfit if run too long; early stopping / shorter schedule recommended.
- ResNet runs overfit after ~epoch 7; ArcFace e2e (lr 1e-5) overfits later but still peaks before 16 epochs.
- **High-confidence filtering didn't help (v9):** dropping medium/low-confidence VLM train pairs (~15% of train) matched v7 on val accuracy (78.14% vs 78.09% — noise) but was slightly worse on rank agreement (τ 0.828 vs 0.842, ρ 0.957 vs 0.962). Medium-confidence label noise is not the current bottleneck; the extra pairs are worth keeping.
- **Ensemble is the best ranker (ensemble-v7-v1):** averaging z-scored v7 (ArcFace) + v1 (ResNet-18) face scores left val accuracy at 78.1% (tied — near the recoverable ceiling given label noise) but produced the **best rank agreement of any run**: τ 0.858, ρ 0.967 (v7 alone: 0.842 / 0.962). Diverse backbones cancel per-model ranking errors even when pairwise accuracy saturates.

**Future training tests (queued):**

- [x] **High-confidence-only training** — done as **train-v9** (2026-07-12): no gain over v7; see findings. Filter kept in `train.py` (`confidence_filter`) for future ablations.
- [x] **Ensemble** — done as **ensemble-v7-v1** (2026-07-12) via new `scripts/ensemble_eval.py`: accuracy flat, rank agreement best-in-class; see findings.
- [ ] Stratified eval by VLM confidence / BT θ-gap (easy-vs-hard pairs) — diagnostic, defines "clear winner" performance.
- [ ] Positional-bias check (accuracy on A-wins vs B-wins pairs) — order-symmetry guard from methodology.
- [ ] DINOv2 e2e (v5 config, ready) — lower priority after ArcFace e2e won.
- [ ] LR schedule (cosine/step decay) on the winning recipe.

#### Artifacts

- Checkpoints: `faceiq-preference-ml/checkpoints/train-vN*/best.pt`; metrics + eval: `artifacts/train-vN*/{metrics,eval}.json` + `model_scores.csv`
- Configs: `faceiq-preference-ml/configs/train-v*.yaml`; run logs: `artifacts/train-v*-full-run.log`
- Best single model: `checkpoints/train-v8-arcface-e2e-long/best.pt` (78.4%, ep 5)
- Best ranker (ensemble): `ensemble-v7-v1` eval in `artifacts/ensemble-v7-v1/eval.json` (τ 0.858, ρ 0.967)
- Dashboard: `streamlit run app/dashboard.py --server.port 8502` → Training runs tab

---

### 5.4 Anchor-ladder inference (pre-deployment smoke test)

**Status:** Not started.

#### Methodology

<!-- New face vs anchor set → /10 band. -->

**Design notes for production inference (recorded 2026-07-04, before implementation):**

In production the comparator scores a **new, unseen face** by comparing it against a **reference panel** of anchor faces with known `/10` scores — the comparator alone outputs only relative scores with no absolute scale. Literature guidance to apply when this is built (siamese-regression reference-set inference, PMC10469421):

- **Panel size is not "more is better."** Reference-set inference degrades past a moderate panel size (~5–10 references in the source study) because dissimilar/distant references add bias, not signal. Choose panel size empirically on held-out cohort faces; do not default to "compare against everything."
- **Panel composition:** anchors should span the score range (a ladder), e.g. faces at fixed product-facing percentiles per gender with high comparison counts — the `isAnchor` flag in schema §3.3 is reserved for exactly this.
- **Cheap per-prediction uncertainty:** the variance of the predicted deltas across the panel is a confidence signal (high spread = unreliable prediction) — no ensemble needed. Log it alongside the point estimate.
- **Two inference options to compare:** (a) average implied score across panel comparisons; (b) fit the new face's θ by mini-BT against the panel outcomes. Validate both against held-out BT θ before choosing.

Note the asymmetry with training: training needs no reference panel (labels are fixed exported pairs); the panel exists only at deployment/inference time.

**Anchor sourcing — two valid paths (2026-07-05, post elite-crowding review):**

| Path | When to use | How |
|------|-------------|-----|
| **A. New curated cohort + BT** | Long-term; production-population-aligned GT | Smaller tails (§5.2); export → BT refit → `/10` from calibration; select ladder faces from stable θ percentiles |
| **B. Manually assigned anchor panel** | Near-term product; decouple absolute scale from research cohort | Curate ~5–10 faces per gender; assign **product `/10`** by committee (e.g. reference model fixed at 8.5); store in DB (§3.3 `isAnchor`). Comparator predicts win/loss vs each anchor; aggregate to user `/10`. Research-cohort BT scores on those same photos are optional cross-check only |

Path B is the intended **v1 production bridge**: the model learns *who beats whom* from 52k matchups; anchors define *what `/10` means* for users. Elite crowding in the research export does not block training; it only means **do not copy research `scoreOutOf10` verbatim into the consumer app** — use Path A or B for production labels.

**Open for §5.4 implementation:** panel size sweep; gender-matched anchors; log variance-based confidence; smoke test on held-out cohort faces vs human `/10`.

#### Results

<!-- Consistency vs BT on held-out cohort faces. -->

---

### 5.5 Human panel ground truth (Prolific runs 1–3)

**Status:** **Three runs complete** (2026-07-29 → 08-01). 667 raters, **65,894 votes** on **7,780** pairs,
**$3,186** spent. Ranking refit twice and re-gated (`artifacts/bt-refit-v{3,4}-panel`). Marginal value of
the newest run measured out-of-sample and **positive**: +3.18 pts at $404/pt. **The comparator has not yet
consumed any of it** — see §5.3.

Detailed operational record lives in `panel-pilot-runbook.md`; the repeatable decision loop in
`panel-study-playbook.md`; the partner-facing summary in `panel-pilot-findings.md`.

#### Overview

§2.1–§2.3 validated the VLM against *our own* audit labels. That leaves the load-bearing question
unanswered: does the ranking predict what **ordinary people** actually choose? We built a standalone
rating app (`faceiq-rating`, Next.js + Prisma on Vercel) and bought pairwise judgments from Prolific's
US representative sample, ~100 pairs per rater with 20 gold attention checks.

#### Research question

1. Does the BT ranking predict human pairwise choice, and how does that vary with the θ gap?
2. Where does Gemini agree with the crowd, and where is it blind?
3. Does buying human votes measurably improve the ranking on pairs we did **not** buy?

#### Methodology

**Design: incomplete block.** No rater sees all pairs. Each rater draws ~100 pairs from the active
study's pool; the queue targets even coverage so every pair accumulates ~6–12 votes from *different*
people. This buys population generalisability and a rater-bias audit that a fixed panel of 12 cannot.

| run | studyId | raters | pairs | votes/pair | spend |
|---|---|--:|--:|--:|--:|
| 1 — soft launch | `6a6b7dc47e3421b6e4b5ca76` | 12 | 100 | 12 | ~$40 |
| 2 — panel pilot | `6a6ba4da4825f473a8f65364` | 372 | 2,980 | 8–56 (uneven) | $1,900 |
| 3 — hard-pair top-up | `6a6d1e7e3999a35d0dc956b9` | 301 | 4,800 | 6.2 | $1,286 |

**Ranking fit.** Human votes cannot be pooled with VLM labels as if equally sharp. `bt_panel.py` fits a
joint BT with **per-source discrimination**: $\Pr(i \succ j \mid s) = \sigma(\beta_s(\theta_i - \theta_j))$,
$\beta_{\text{vlm}} \equiv 1$, $\beta_{\text{human}}$ free. On panel pairs the VLM label is **dropped** and
replaced by the human votes. Ties are half-votes, matching §5.1.

**Marginal-value protocol** (`panel_run_delta.py`) — the only measure we act on. Hold out **whole pairs**
from the newest run, so no held-out pair's votes enter any fit and a gain can only travel through better
face θ. Then compare a ladder: VLM only → + prior runs → + this run. 5 seeds, 25% held out.

#### Success criteria

- BT gates of §5.1 still pass on the joint fit (connectivity, ≥15 comparisons, ρ_stab > 0.95)
- Marginal accuracy gain on held-out pairs > 0, replicated across seeds
- Stop condition: **< 1 pt per $500** ends the programme

#### Results

**1. The ranking is validated, and it has a resolution limit.** Human agreement with BT's favourite rises
monotonically from chance to 93% as the gap widens — which a machine-label-derived ranking had no
obligation to do. It also takes **~2.5 /10 points before 3 in 4 people agree**; sub-point differences are
a coin flip. That is a property of human perception, not a model defect, and it argues for shipping bands
rather than decimals.

**2. Gemini matches or beats the crowd on clear pairs and is at chance on hard ones.** On run 3's
`topup-0-10` stratum (gap < 10 percentile points) VLM-only scored **51.3%**. Its own confidence field does
not locate the blind spot, so confidence is not a valid trigger for buying labels.

**3. Humans carry real signal on hard pairs — the initial reading was wrong.** Run 2 first looked like
"close pairs are irreducibly subjective" because majority agreement was near chance. That conflated *low
majority* with *no signal*. Vote **shares** replicate at split-half reliability **0.655**, and 39% of close
pairs reach ≥75% agreement. The information is in the margin, not the winner.

**4. Both refits passed all gates.**

| | v3-panel (run 2) | v4-panel (runs 2+3) |
|---|---|---|
| β_human (F / M) | 0.25 / 0.32 | **0.417 / 0.368** |
| implied temperature vs VLM | ~3–4× flatter | 2.40 / 2.72 |
| ρ_stab (F / M) | > 0.95 | 0.9931 / 0.9924 |
| ρ vs Labs `overall_score` | — | 0.752 / 0.766 |
| ρ vs previous refit | 0.991 (vs v2-qc) | 0.9905 (vs v3) |
| faces moving > 0.5 /10 | — | 149 |

β_human < 1 in every fit: **humans are consistently less decisive than the VLM at the same θ gap**, so
pooling the two sources naively would have over-weighted the human votes.

**5. Run 3 bought real, out-of-sample accuracy.** On 1,200 pairs whose votes entered no fit:

| model | accuracy on held-out human votes |
|---|--:|
| VLM only | 51.47% ± 0.74 |
| + prior runs (run 2) | 54.73% ± 0.69 |
| **+ run 3** | **57.91% ± 0.66** |

Marginal gain **+3.18 pts** (paired sd 0.23, **5/5 seeds positive**) for $1,286 → **$404/point**, cheaper
than run 2. The dose curve *accelerates* — 55.41 → 56.01 → 56.92 → 57.91 at 25/50/75/100% of the votes —
so we are not yet in diminishing returns, and the stop condition is ~4× away.

**6. Rater demographics (pulled 2026-08-01).** 667/667 matched. Representative sample delivered on sex
(51.2% F) and ethnicity (63.1% White, 11.9% Black, 11.0% Mixed, 7.6% Other, 6.3% Asian), but **38.1% of
raters are 55+** — census-plausible, probably older than the product's users. A matched in-group vs
out-group test finds a **real but small** cohort effect: +2.36 pts for 55+ (4.0 sd), +1.39 White, +1.04
Female, noise on the small cells. Confounded with rater consistency. Relevant to §5.2's absolute-`/10`
risk, not to the ranking.

#### Decision

- ~~**Ranking of record is `bt-refit-v4-panel`. Draw future studies against its percentiles.**~~
  **Both halves superseded.** `bt-refit-v5-panel` is the ranking of record (§5.8), and draws must
  *never* use a panel-fitted refit — §5.7 measured that it inverts the headroom table. Draw against
  `bt-refit-v2-qc`.
- ~~**Keep buying hard pairs**~~ — correct at the time (dose curve accelerating, stop condition 4×
  off) and still correct *within a 20-point gap*, but §5.8 found the outer edge: above 20 points
  headroom is negative and $14,367 was cancelled. The remaining $7,463 is also now behind the
  comparator in priority.
- **Score studies on vote share, never majority.** The margin is the signal.
- **No spend** on comparison-graph density or cohort breadth; `cohort_capacity.py` shows neither binds.
- **Per-audience labels deferred**, with numbers rather than an opinion: 8–16× more recruiting in the
  small cells, and Prolific's representative sample cannot target them.
- Rater QC: reject on **behavioural proof only** (1 rater, 236 ms median). Fast clicking and gold failures
  alone are poor quality signals — gold pass rate does not track leave-one-out agreement.

#### Artifacts

- App: `faceiq-rating` (Vercel) · run separation via `CURRENT_STUDY_ID`; queue fix in `lib/assignment.ts`
- Raw votes (gitignored — personal data): `labels/panel-pilot/results/`, `labels/panel-run-3/results/`
- Analysis: `scripts/analyze_panel_run.py` → `artifacts/panel-run-v{1,3}/`
- Refits: `scripts/refit_bt_panel.py` + `src/faceiq_pref/bt_panel.py` → `artifacts/bt-refit-v{3,4}-panel/`
- Marginal value: `scripts/panel_run_delta.py` → `artifacts/panel-run-delta-v3/run-delta.json`
- Value curve / capacity: `artifacts/panel-value-v1/`, `artifacts/cohort-capacity-v1/`
- Demographics: `scripts/analyze_demographics.py` → `artifacts/panel-demographics/summary.json`
- Dashboard: `app/dashboard.py` → "Human panel" tab (`PANEL_RUNS`)

#### Notes

Two bugs worth remembering. Run 2's coverage ran 8–56 votes/pair because `buildQueue()` ranked pairs on
*recorded* judgments only, so concurrent arrivals all reserved the same zero-vote pairs — a thundering
herd. Fixed by counting in-flight reservations (30-min TTL); run 3 came back at 6–7 votes/pair. And the
first split-half reliability figure (0.957) was inflated by averaging vote shares across splits *before*
correlating; the honest figure is 0.655.

---

### 5.6 Rating validation — per-face uncertainty, calibration, composites (complete)

**Status:** **Complete** (2026-08-01), no new labels bought. Answers "is the rating good, and where is it
wrong" with numbers. Headline: **the ranking's global stability (ρ = 0.9931) hid per-face uncertainty of
±0.45 /10 points**, and **no composite beats the network alone**.

#### Research question

§5.1's gates pass and §5.5 shows the ranking predicts human choice above chance, yet inspection kept
turning up faces that "look ranked wrong". Three things were unmeasured:

1. Per-face uncertainty. `stability_check` gives **one** ρ per gender; nothing says which faces the data
   cannot place.
2. Whether the /10 number is **calibrated** — does a face we call 7.0 beat one we call 5.0 as often as we
   imply? Pairwise accuracy is a diagnostic, not a product metric.
3. Whether **averaging** sources (Labs + comparator + BT, as proposed) helps or hurts.

#### Methodology

`scripts/bt_uncertainty.py` bootstraps the comparison graph (200 replicates per gender, refit through
`fit_joint_bt`) in two deliberately separate modes, because the causes differ:

- `resample` — comparison rows with replacement: *sampling* variability.
- `relabel` — comparison set fixed, VLM labels flipped at the **measured** per-band disagreement rate:
  *label* sensitivity. Read as an upper bound, since flipping at rate $e$ adds noise rather than redrawing
  the label. Panel-covered pairs are immune, because `build_cells` already discards the VLM label there.

Neither bootstrap can express **non-identification**: an undefeated face has no losses in *any* resample,
so its interval comes out narrow and high — false confidence, not confidence. Those faces get a flag
instead, and the artifact is visible in the output (median interval 5.8 pts for flagged faces vs 16.9 for
the rest).

`effectiveComparisons` weights each observation by $(1-2e)^2$, the Fisher information a label with error
rate $e$ carries relative to a perfect one.

Calibration (`scripts/rating_calibration.py`) reads the same votes two ways: share of votes won by the
higher-scored face per **/10 gap band** (product-facing), and a **reliability diagram** against
$\sigma(\beta\,\Delta\theta)$.

Composites (`src/faceiq_pref/composite.py`, `scripts/composite_eval.py`, dashboard **Composite** tab) blend
any sources by percentile- or z-average **within gender**, then score every blend and component on
**leak-free** pairs — the intersection of components' val splits — with a **paired** bootstrap over pairs.
Geometric mean is not offered: θ is signed and the mean is meaningless on an interval scale.

#### Results

**Per-face uncertainty** (`artifacts/bt-refit-v4-panel/uncertainty.csv`, 2,866 faces):

| quantity | median | p90 |
|---|--:|--:|
| 95% interval, percentile points (`resample`) | **16.8** | 24.3 |
| 95% interval, percentile points (`relabel`) | 12.4 | 20.7 |
| 95% interval, **/10 points** | **0.91** | 1.36 |
| rank interval, places (of ~1,430 per gender) | **241** | — |
| comparisons per face | 34 | — |
| **effective** comparisons per face | **10.1** (8.6 VLM + 1.4 panel) | — |

- **Two faces closer than ~17 percentile points apart are not ordered by this data.** Near the middle of
  the scale that is ~0.9 /10 points, so "a 6.2 outranks a 5.8" is not a claim the evidence supports. This
  is the direct explanation for faces appearing misplaced: mostly they are *unplaced*.
- 856 faces (30%) have intervals wider than 20 percentile points; **125** sit under 5 effective
  comparisons; **44** are undefeated or once-beaten and therefore **not identified from above**.
- `relabel` intervals are *narrower* than `resample`. Close-pair label noise is near-symmetric and
  partially cancels, whereas dropping ~37% of rows does not — which comparisons we ran matters more than
  the noise in the close ones.

**Calibration** (`artifacts/bt-refit-v4-panel/calibration.json`, 7,780 panel pairs). Higher-scored face's
share of human votes, by /10 gap:

| /10 gap | `bt-refit-v2-qc` (VLM-only, honest) | `bt-refit-v4-panel` (**circular**) |
|---|--:|--:|
| 0–0.25 | 51.5% | 54.5% |
| 0.25–0.5 | 52.5% | 63.1% |
| 0.5–1 | 54.5% | 71.1% |
| 1–1.5 | 61.8% | 78.8% |
| 1.5–2 | 68.8% | 80.0% |
| 2+ | **81.1%** | 89.4% |

- The VLM-only ranking is **badly overconfident**: where it implies 84.9% it delivers **53.1%** (−31.7 pts).
- Applying the panel's measured temperature ($\beta_{\text{human}} = 0.4171$) helps but does **not** fix it
  (−31.7 → −19.7). So the per-face θ *positions* are wrong on close pairs, not merely the scale.
- `bt-refit-v4-panel` is calibrated within ±3 pts up to the 0.8 bin — but it **absorbed every one of these
  pairs**, so this is measured on its own training data and is not evidence.
- **Caveat that limits all of the above:** panel pairs were deliberately enriched for *close* pairs
  (only ~230 of 7,780 exceed a 45-point percentile gap). These are worst-case, not population, figures.

**Composites** — scored on 1,928 leak-free pairs, 16,376 votes, crowd ceiling **59.09%**:

| predictor | agreement |
|---|--:|
| `train-v13-panel-hard` | **56.42%** |
| BT `bt-refit-v2-qc` | 55.02% |
| 6-way equal blend | 54.43% |
| Labs `overall_score` (legacy) | 53.65% |
| `external-scut` | 52.76% |
| `external-mebeauty` | 52.10% |
| `external-humanaes-1b` | 50.04% |

- **The 6-way equal-weight blend loses to its best component**: −1.99%, 95% CI [−3.33%, −0.67%] (paired
  bootstrap). Adding near-chance components drags the average down. Averaging is not free lunch here
  because the components' errors are not independent — comparator and BT were both fit on VLM labels.
- Narrower blends are **indistinguishable** from the comparator alone: comparator + BT + Labs gives
  +0.51% [−0.40%, +1.43%]; comparator + BT gives +0.25% [−0.41%, +0.92%].
- **Best blend measured: Labs + comparator, 50/50 → 56.91%** (+0.49% over the comparator alone, CI
  [−0.36%, +1.32%]). Positive point estimate, not significant. Worth re-testing if the panel grows,
  because Labs is the one component whose errors are *plausibly* independent — it is a hand-built formula
  over different features, not another fit to the VLM labels. Scores dumped for inspection to
  `artifacts/composite-v1/labs-plus-v13-scores.csv` (2,866 faces); it moves the median face 0.39 /10
  points from BT, and 40% of faces move more than 0.5.
- Blending **sibling comparators** does not help either: v13 + v15 + v12 + BT reaches 56.57% against
  `train-v15-panel-weighted` alone at 56.55% (+0.02%, CI [−0.61%, +0.66%]). Models trained on the same
  labels make the same mistakes, which is the correlated-error argument in one line.
- `train-v15-panel-weighted` (panel votes up-weighted 3×) scores **56.55%**, a hair above `train-v13`'s
  56.42% and inside the noise — the up-weighting is not a separable improvement.
- **Reference-set transfer is poor in both directions.** Models trained on independently collected human
  ratings sit near chance on our panel (SCUT 52.8%, MEBeauty 52.1%, HumanAesExpert 50.0%) while our
  cohort-trained comparator reaches 56.4% against a 59.1% ceiling. Rank agreement vs `bt-refit-v4-panel`
  refreshed on the QC'd 2,866 faces: SCUT τ = 0.285 / ρ = 0.418, MEBeauty τ = 0.208 / ρ = 0.310.

#### The /10 range is set by the anchors, not by the model

Worth stating plainly because it caused real confusion. The /10 is a **percentile ladder**: `calibrate.py`
pins the 50th percentile to 5.0, the 90th to 7.0 and the 99th to 8.0, then interpolates. So the blend's /10
distribution comes out at *exactly* median 5.00, p90 7.00, p99 8.00, max 9.00 — and so does every other
blend, and so does BT.

Consequence: **no change of model or blend can make the best face in the cohort score 9.5.** Swapping
sources only reshuffles *which* face gets a 7.0. "Our top faces only score 7" is a statement about the
anchors and about the cohort being top-decile already, not about model quality. Changing it is the deferred
**anchor-ladder** decision (§5.4), and it is a product/positioning choice, not a modelling one.

#### Decisions

- **No composite ships yet.** The comparator alone is the best *significant* predictor; the Labs +
  comparator blend is the best point estimate (+0.49%) but its CI includes zero. Broad blends are
  measurably worse. Revisit Labs + comparator when the panel is larger, or with a component whose errors
  are plausibly independent (landmark/EBM), and only on a paired test.
- **Quote bands, not points.** Given a median /10 interval of 0.91, the honest product unit is a band of
  roughly **1 point**, not two decimals.
- **`external-*` models remain diagnostics.** Non-commercial research licences, and near chance anyway.
- **Publishing a calibrated /10 is blocked on a held-out calibration test.** Every panel pair is inside
  `bt-refit-v4-panel`'s fit, so its calibration cannot be honestly measured today.

#### What a shipped rating is judged on

Pairwise accuracy on hard pairs is a *diagnostic*. Product metrics, with current status:

| metric | definition | status |
|---|---|---|
| ordering by gap band | agreement with human votes vs the gap between two faces | **measured** — `train-v13` runs **53.3%** at a 0–2 pt gap rising to **78.6%** at 45+ (`artifacts/train-v13-panel-hard/panel-eval.json`, bands cut on the VLM-only `bt-refit-v2-qc`). An earlier revision of this row read "50.8% → 88.3%"; 50.8% was the *VLM* column and 88.3% matched no artifact |
| score calibration | does a stated 7.0 beat a stated 5.0 at the implied rate | **measured but circular** on the shipping ranking; honest number needs fresh pairs |
| band width | smallest /10 gap that is reliably ordered | **measured** — ~1 point (median interval 0.91) |
| per-face uncertainty | interval on an individual's number | **measured** — ±0.45 /10 median, ±0.68 at p90 |
| **test–retest stability** | same person, different photo, same score | **never measured**, and not possible from this export (one `_front` photo per person). Needs a small extra export of other views — no new labels |

#### Landmark features — feasibility of an external extraction (assessed 2026-08-01, not built)

`insightface` is already a dependency and gives 106-point 2D landmarks, but those are face-*alignment*
points, optimised for cropping and pose normalisation rather than for anatomically salient positions
(gonial angle, canthal tilt, alar base, philtrum). If we want the second kind, the extraction has to come
from somewhere else — plausibly a collaborator running a dedicated model.

**The join is trivial and needs no schema work.** `faces.jsonl` carries
`imagePath = images/<faceId>.webp`, so the image *filename is the join key*: any per-image output keyed by
filename merges straight onto `faceId`. Precedent already in the repo is `scripts/join_face_attributes.py`,
which joins an external manifest on `sourceFaceId`. The whole cohort is 3,000 images, and a 106-point
feature table is well under 1 MB. So "send the images out, get a CSV back, join it" is a genuinely small
piece of work.

Three things that are **not** small, and should be settled before sending anything:

1. **These are real people's face photos.** Sending the cohort to a third party is a privacy and consent
   decision, not an engineering one — the same reason `data/` is a hard-rule git exclusion. Check what the
   onboarding consent actually permits before exporting.
2. **The extractor must run in our inference path.** A one-off extraction on the cohort is enough to *fit*
   an EBM, but scoring a new user's upload needs the same landmarks at request time. If the extractor is a
   collaborator's local script, the EBM can be built and validated but never shipped — worth agreeing on a
   runnable artifact (container, weights, or ONNX) up front rather than after the fact.
3. **Fix the landmark schema before extraction, not after.** Point count, ordering, coordinate space, and
   whether values are pose-normalised all have to be pinned, or the join succeeds and the features are
   silently incomparable across faces.

Formulation when it happens: the EBM should predict **P(A beats B) from feature *differences***, not a
score from absolute features. That preserves the pairwise structure everything else in this programme is
built on, keeps it comparable to the comparator on the same panel-vote metric, and yields exactly the
per-feature contributions ("your jaw contributes +0.9") that motivated the request.

#### Artifacts

- Uncertainty: `scripts/bt_uncertainty.py` → `artifacts/bt-refit-v4-panel/uncertainty.{csv,json}`
- Calibration: `scripts/rating_calibration.py` → `artifacts/bt-refit-v4-panel/calibration.json`
- Composites: `src/faceiq_pref/composite.py`, `scripts/composite_eval.py` → `artifacts/composite-v1/`
  (`report.json`, `labs-plus-v13.json`, `labs-plus-v13-scores.csv`, `reference-set-v1{3,4}.json`,
  `arms-v15.json`)
- Dashboard: `app/dashboard.py` → **Composite** tab — blend scoring, the blend's own /10 distribution,
  and visual validation (contact sheet by score band with per-face intervals, "most suspicious" view)
- Per-face blend scores for eyeballing: `composite_eval.py --dump-scores`

#### Notes

A row-resample bootstrap is the wrong tool for the top of a ranking and quietly says the opposite of the
truth: an undefeated face keeps a perfect record in every replicate, so it earns a *narrow* interval at an
arbitrary θ (v4-panel's top face sits at θ = 18.04 against a p99 of 8.73). Trust `notIdentified` over the
interval for those 44 faces. The general lesson for §5.1's gates: a global stability ρ of 0.9931 is
compatible with the median face's rank being uncertain by 241 places, because ρ is dominated by the
correctly ordered extremes.

---

### 5.7 Label information — margin recovery, VLM confidence as a router, spend pricing (complete)

**Status:** **Complete** (2026-08-01), no new labels bought. Headline: **binary accuracy was the wrong
metric and it hid the real headroom.** The comparator sits at ~96% of the crowd ceiling on *picking
winners* but only **38.9% of the ceiling on predicting the margin**. Separately, **Gemini's confidence is
useless as a router** (it cannot tell close pairs from far ones), while the **percentile gap works**, and
completing human coverage of every pair where the VLM is a coin flip costs **≈$3.6k**.

#### Research question

§5.5 left three things unresolved that were each about to drive a spend decision:

1. Is a single score "faulty by nature" when one rater agrees with the crowd only 59.1% of the time? A
   score saying 6.00 vs 6.05 is making a *quantitative* claim — that the crowd splits ~evenly — and binary
   accuracy throws that claim away rather than testing it.
2. Would a **graded** VLM label (a band, or a margin) fix the blind spot more cheaply than human votes?
   That only works if the VLM has any idea when two faces are close.
3. What does each labelling strategy actually cost, per pair type?

#### Methodology

`scripts/label_information.py`. Three measurements, one artifact.

**Margin recovery.** Spearman between a predictor's *score gap* and the crowd's *observed vote share*. The
observed share is itself noisy at ~8.5 votes/pair, so a perfect predictor cannot correlate 1.0 with it.
The ceiling is derived rather than assumed: split each pair's ballots into halves at vote level, correlate
the halves (200 draws), correct half-samples up to full sample with Spearman–Brown, then take the square
root to convert a reliability into the maximum correlation an external predictor can reach. That gives
**reliability 0.580 → ceiling 0.762**, and every row is reported as a fraction of it.

**Leakage guard.** Panel vote shares became *train* targets in v12+, so a model scored on a pair it trained
on is reciting a target, not predicting one. Margin rows are restricted to the intersection of the runs'
validation splits, rebuilt from each run's stored `val_fraction`/`split_seed` (**1,928 pairs**). This is not
cosmetic: the same table computed on all 7,780 panel pairs reports v15 at **0.430**, versus **0.296**
honestly — a 45% overstatement. Sections 2–3 study the VLM and the ranking, neither of which trained on
these votes, so they keep all 7,780.

#### Findings

Margin recovery, leak-free (1,928 pairs, ceiling 0.762):

| predictor | r(score gap, vote share) | % of ceiling |
|---|---|---|
| NN `train-v15-panel-weighted` | 0.296 | 38.9% |
| NN `train-v13-panel-hard` | 0.280 | 36.7% |
| NN `train-v12-panel-soft` | 0.245 | 32.1% |
| BT `bt-refit-v2-qc` (VLM-only, honest) | 0.215 | 28.2% |
| Labs `overall_score` | 0.200 | 26.3% |
| VLM label × confidence tier | 0.174 | 22.8% |
| BT `bt-refit-v4-panel` | 0.719 | **CIRCULAR** — fit on these votes |

1. **The single number is not the problem.** The comparator carries **1.7× the margin information of the
   VLM label it was trained on** (+0.150, 95% CI [+0.096, +0.202], paired bootstrap over pairs), and beats
   both the honest VLM-only ranking and Labs. Distilling 52k binary labels produced something strictly
   richer than the labels — the network sees closeness the labels never stated.
2. **The headroom moved.** 96% of ceiling on winners vs 38.9% on margin means further work on *picking
   winners* is close to pointless, while margin fidelity has ~2.6× to go. This is the first metric in the
   programme with real room in it.
3. **The panel arms are mostly indistinguishable.** v15 − v13 = +0.016, CI [−0.005, +0.037] — **not**
   distinguishable, so `panel_weight: 3.0` is unproven. v13 − v12 = +0.035, CI [+0.016, +0.054] — hard
   majority targets beat soft vote-share targets, which is counterintuitive and worth a rerun before it is
   trusted. Note this contradicts nothing in §5.5: those arms were also indistinguishable on accuracy.

VLM confidence versus crowd difficulty (all 7,780 panel pairs):

| confidence | pairs | decisiveness | VLM agrees w/ votes |
|---|---|---|---|
| high | 5,460 | 0.213 | 55.6% |
| medium | 2,290 | 0.203 | 50.0% |
| low | 30 | 0.185 | 48.2% |

4. **Gemini's confidence is not a router.** Decisiveness (mean \|share − 0.5\|) is flat across tiers, and
   **66.5% of the closest band (0–2 pct gap) is labelled "high" confidence**. The model does not know when
   it is guessing. This kills the "ask for a rating band instead of a winner" idea as a cheap fix: a graded
   label inherits the same blind spot, because the blindness is in the discrimination, not the output
   format. Confidence is still useful for *filtering* (§4.3's 87% audit rate), just not for *targeting*.
5. **The percentile gap is the router**, and it is the one already in use.

##### Where a human vote is worth buying — bands must be cut on the VLM-only ranking

The decisive column is **headroom**: the crowd ceiling (leave-one-out, one rater vs the majority of the
others) minus the free VLM label's agreement. Where headroom is ~0 the VLM already performs at single-human
level and a purchased vote buys a duplicate. Bands cut on `bt-refit-v2-qc` (VLM-only, **not** contaminated
by panel votes), priced at $0.0483/vote and 12 votes/pair:

| pct gap | pairs | bought | unlabelled | cost | VLM vs votes | 1 rater (ceiling) | headroom (95% CI) | verdict |
|---|--:|--:|--:|--:|--:|--:|--:|---|
| 0–2 | 2,663 | 1,624 | 1,039 | $603 | 50.1% | 56.3% | **+6.2** [+4.1, +8.4] | **BUY** |
| 2–5 | 3,848 | 2,077 | 1,771 | $1,028 | 51.6% | 57.4% | **+5.9** [+4.1, +7.7] | **BUY** |
| 5–10 | 5,387 | 2,218 | 3,169 | $1,839 | 52.9% | 56.4% | **+3.5** [+1.9, +5.3] | **BUY** |
| 10–20 | 8,936 | 1,210 | 7,726 | $4,483 | 52.6% | 56.4% | **+3.8** [+1.5, +6.0] | **BUY** |
| 20–45 | 15,524 | 389 | 15,135 | $8,781 | 66.0% | 66.0% | +0.1 [−2.5, +2.5] | unresolved |
| 45–100 | 11,546 | 262 | 11,284 | $6,547 | 82.2% | 81.1% | −1.1 [−2.9, +0.6] | unresolved |

6. **Buy the 0–20 gap.** All four buy-zone bands clear zero with the whole interval — this is a validated
   spend, not a directional hunch. Finishing them costs **$7,952** over 13,702 unlabelled pairs (confirmed
   by an actual `select_topup_pairs.py` draw).
7. **Above a 20-point gap is *unresolved*, not proven worthless** — an earlier revision of this section
   overstated it. Both intervals straddle zero because we own only 389 and 262 pairs there. What *is*
   established is a cap: the best case is **+2.5 pt at 20–45** and **+0.6 pt at 45–100**, against +1.5 to
   +8.4 in the buy zone, at the two highest per-band costs in the table ($8,781 and $6,547). Worst value
   per dollar available, so it stays last regardless of where inside the interval the truth sits.

**Is the sample representative of its band?** No, and the bias helpfully runs *against* the conclusion. Our
purchased pairs were drawn as near-ties under a different ranking, so their VLM-confidence mix is skewed
versus their band: **+9.6 pt more high-confidence at 0–2** (which flatters the VLM and *understates* buy-zone
headroom) and **−7.2 pt less at 20–45** (which handicaps the VLM and *overstates* skip-zone headroom). Both
distortions push toward the recommendation we are making, so correcting them would widen the gap rather than
close it. A uniform draw is still the clean fix — see the sequencing note below.

##### Recommended sequencing — the random-pair study first

The $7,952 does not have to be committed at once, and it should not be first. A **uniform** random-pair
study of ~700 pairs (~$400) does triple duty, and its natural band composition is exactly the coverage we
lack: 227 pairs at 20–45 and 169 at 45–100, roughly doubling the thin cells, versus 305 spread across the
buy zone.

1. Population accuracy — the first non-worst-case number in the programme (§5.5's ceiling is measured only
   on deliberate near-ties).
2. A non-circular calibration curve — `bt-refit-v4-panel` absorbed every pair we own, so §5.6's reliability
   diagram cannot be honestly tested on them.
3. **Unbiased headroom in the two unresolved bands**, resolving whether the remaining $15,328 is dead money
   before any of it is spent.

> **All three delivered — see §5.8.** Population accuracy 81.5% against a 74.9% human ceiling; a monotone
> out-of-sample calibration curve; and the $15,328 resolved to **dead money** (headroom −0.9 and −1.3, the
> second interval entirely below zero). Two of §5.7's conclusions below are narrowed as a result: the
> "unresolved" verdict on the wide bands is now a **no**, and this section's headline that margin recovery
> has "2.6× to go" is a statement about *hard pairs only* — on a population sample the same ranking sits at
> 82.3% of its ceiling, not 28.2%.

Then buy the 0–20 zone in tranches, cheapest-first (0–2 at $603, 2–5 at $1,028, 5–10 at $1,839, 10–20 at
$4,483), re-running `panel_run_delta.py` between tranches against the existing stop rule of <1 point per
$500.

**Drawn and ready (2026-08-02).** `scripts/select_random_pairs.py` →
`labels/panel-run-4-random/pairs.json`: 2,500 new pairs + the 20 carried golds, uniform over the 40,101
unbought QC-clean ranked pairs, **no gap filter and no ethnicity floor** — stratifying would destroy the
population estimate the study exists to produce, so bands are *recorded*, not *imposed*. Median percentile
gap **29.9** versus 4.9 for run 3; that difference is the experiment. The draw's band mix tracks its pool to
within 2.6 points, and **1,657 of 2,500 pairs (66%) land in the two unresolved bands**.

Sized at **2,500 pairs × 12 votes, 300 raters, ≈$1,286** — the same spend as run 3, because the
representative-sample floor of 300 participants fixes the vote budget at 30,000 regardless, leaving only the
pair count as a lever. This **breaks the breadth-beats-depth rule deliberately**: that rule optimises how
much a study *moves* the ranking, whereas the crowd ceiling is a leave-one-out statistic and the calibration
curve bins observed win rates, so both want depth. At 6 votes the "majority of the others" is 5 noisy votes,
which biases the ceiling downward and *understates* headroom; the existing table was measured at ~8.5
votes/pair, so 12 is the comparable floor. Prolific fields in `prolific-soft-launch-form.md` §9. Verified the
draw satisfies `lib/pairs.ts`'s boot assertion (0 golds missing, 0 worked examples leaked).

**Published 2026-08-03.** Prolific study **`6a71110e4d5c6eba96c17d21`**, 300 places, representative US,
total **$968.58**. Pre-launch checks passed: `/api/monitor` returned `run: 6a71110e4d5c6eba96c17d21`,
`coverage: {"0": 2500}`, `judgments: 0`. It went out at a ~7 min estimate and ≈$2.26/submission rather than
the planned $3.00 at 9 min — ≈$19.4/hr at the stated time, on policy, but run 2's p90 of 9.4 min means the
slowest decile earns nearer $14.5/hr. Note representative places cannot be increased post-publish, so 300
is final.

**Run 4's pairs are the only ones in the programme no ranking has been fitted on.** Refitting BT on them
spends that permanently, so `rating_calibration.py` must run *before* any refit consumes them. This is the
one path to a publishable calibrated /10, which §5.6 currently blocks as circular.

**Age stays representative for this run.** Runs 2–3 were 38.1% aged 55+ and 11.9% aged 18–24
(`artifacts/panel-demographics/summary.json`) — what census-matching produces. Restricting age converts the
population number into a subgroup number and breaks comparability with runs 2–3, and the measured effect is
small (in-group agreement gap **+2.4 pts** for 55+, −0.5 for 35–44, +0.7 for 25–34). The clean follow-up is a
fixed-pair A/B: the same 500-pair subset to an 18–34 *standard* sample, which carries no 300-participant
floor, so 100 raters × 100 pairs ≈ **$429**.

**Methodological trap, recorded because it inverted the recommendation.** Running the same table with bands
cut on `bt-refit-v4-panel` reports headroom of **−5.7 pt at 0–2 and +13.6 pt at 20–45** — the exact
opposite conclusion, and it would have sent $8.8k at the worst band in the table. Cause: v4-panel absorbed
the panel votes, so pairs where the crowd overruled Gemini had their θ pushed apart and *migrated into
wider bands*, carrying their VLM errors with them. **Any band cut, stratification, or pair selection must
use a ranking that has not seen the votes being analysed.** The same hazard as the calibration finding in
§5.6, in a place nobody was watching for it.

#### Interpretation — what 59.1% actually means

The ceiling is **one rater versus the crowd majority on deliberately-selected near-ties**, not "people
only agree 59% of the time". Disagreement there is the *correct* reading of two nearly-equal faces, not
noise to remove. So a low binary number and a good rating are compatible, and the honest product claim is
about margins and bands, not about winning individual coin flips. The margin metric above is the one that
should gate future training runs; §5.6's calibration check is the one that should gate the /10 mapping.

#### Artifacts

- `scripts/label_information.py` → `artifacts/label-information-v1/report.json` (v4-panel bands, kept only
  as the record of the circular version) and `report-v2qc-bands.json` (**the honest one — use this**)

#### Follow-on: landmarks are already in the Labs database

Recorded because §5.6 wrongly listed landmark extraction as external work gated on shipping constraints.
`faceiq-labs` already computes and stores landmarks on **every** user upload — `Face.frontLandmarks`,
`Face.sideLandmarks`, `Face.mediapipeLandmarks` (MediaPipe 468-point), all `Json?` columns, produced by the
existing SageMaker front-model and the landmarking cascade in `src/components/landmarking/`. So there is no
extraction project, no third-party round trip, and no inference-path problem: the extractor is already in
production on the path a new upload takes.

What is actually missing is one line in the exporter. `scripts/export-gt-run.ts` selects only `decileBin`,
`labsOverallScore`, `gender`, and the image path, so `faces.jsonl` has no landmark columns. Adding
`frontLandmarks` (and `mediapipeLandmarks`) to the two `select` blocks and re-running the faces half of the
export yields landmarks for all 3,000 cohort faces immediately. Point ordering and coordinate space still
have to be pinned before the join, per §5.6.

#### Notes

`TrainConfig(**ckpt["config"])` at seven call sites made any checkpoint trained after a field was added
unloadable by older code (the dashboard died on `variance_head`). Replaced with
`TrainConfig.from_saved()`, which drops unknown keys with a warning; dropping an architecture-bearing key
still fails loudly at `load_state_dict`, which is the behaviour we want. Also note Streamlit does not
reload imported modules on script change — after editing `src/faceiq_pref/`, restart `streamlit run` or the
dashboard keeps a stale class in `sys.modules`.

---

### 5.8 Run 4 — the uniform random-pair study: population accuracy, honest calibration, and the end of the spend question (complete)

**Status:** **Complete** (2026-08-04). Prolific study `6a71110e4d5c6eba96c17d21`, 303 sessions, **31,237
judgments** (29,351 real votes usable after rater exclusions, 1,486 gold) over ~4.6 hours, **$968.58**.
Archived to `labels/panel-run-4-random/results/` (gitignored).

Four headline results, and they do not all point the same way:

1. **The ordering hypothesis is confirmed out-of-sample.** On a uniform draw the ranking's agreement with
   the popular vote rises **monotonically** across all six percentile bands, 53.2% → 83.6%.
2. **The first population accuracy in the programme, and it is far better than the worst-case numbers we
   have been quoting.** The VLM-only ranking predicts the human **majority** on a *typical* pair **81.5%**
   of the time against a **74.9%** individual-human ceiling on the same metric — the ranking beats the
   average person by 6.6 points at identifying what the crowd thinks.
3. **The $15.3k spend question is closed, negatively.** Headroom above a 20-point gap is **−0.9
   [−2.1, +0.4]** and **−1.3 [−2.2, −0.5]**. The second interval is entirely below zero: buying human votes
   on far-apart pairs makes the label *worse* than the free Gemini label. **$14,367 of planned labelling is
   dead money.**
4. **The neural comparator is the weak link, not the labels.** On the same population sample the comparator
   scores **66.5–67.7%** of **individual votes** where the VLM-only ranking scores **69.5%** on the identical
   pairs and metric — a paired deficit of **−1.8 to −3.1 pts, every interval below zero.** It wins only
   inside a 10-point gap and loses the 10–45 band. The ordering information is already in the data we own;
   the network is not extracting it.

> ⚠️ **Two different accuracies are in play and they are not interchangeable. Do not put a number from one
> next to a number from the other** — an earlier draft of this section did, and it produced a spurious
> 14-point gap between the ranking and the comparator.
>
> | metric | what it asks | ceiling on a uniform draw | who reports it |
> |---|---|--:|---|
> | **vs the majority label** | did the predictor pick the face *more* raters picked? A 9–3 pair is one clean win, so 100% is reachable | **73–75%** | `analyze_panel_run.py` |
> | **vs individual votes** | what share of the raw ballots agree with the pick? On that same 9–3 pair a *perfect* predictor scores 75% | **70.4%** | `eval_vs_panel.py`, `panel_run_delta.py` |
>
> Measured side by side on run 4 (`scripts/accuracy_matrix.py` →
> `artifacts/panel-run-v4/accuracy-matrix.json`), excluded raters dropped:
>
> | predictor | vs majority label | vs individual votes |
> |---|--:|--:|
> | Gemini label | 80.0% | 69.5% |
> | BT `bt-refit-v2-qc` | **81.7%** | **70.6%** |
> | ceiling (one rater) | 73.2% | 70.4% |
>
> So the same ranking is **+8.5 points better than a person** at naming the crowd's choice and
> **+0.2 points** — level — at predicting one person's ballot. Both are true; they are answers to
> different questions. Result 2 above is the majority-level claim, result 4 is the vote-level one, and
> the comparator is only ever compared against a vote-level baseline on identical pairs.

#### Research question

§5.7 left three things unresolved, all of which required a pair sample that had never been selected on:

1. **Population accuracy.** Every number in §5.5–§5.7 (56.9% vs a 59.1% ceiling) was measured on pairs
   *chosen* to be near-ties, so it is worst-case by construction and says nothing about a typical face pair.
2. **Honest calibration.** `bt-refit-v4-panel` absorbed all 7,780 pairs we owned, so its excellent
   reliability diagram was measured on its own training data.
3. **Unbiased headroom in the 20–45 and 45–100 bands**, which gated ~$15.3k.

And one question that turned out to be the most important of the four, raised while reviewing the
programme: **is the whole band structure circular?** The percentile gaps are cut on a Bradley-Terry fit
over VLM labels. If the VLM's ordering is wrong, then "close pairs" and "far pairs" are arbitrary
partitions and every spend decision built on them is decorative.

#### Methodology

The draw (`scripts/select_random_pairs.py`, seed 20260802) is **uniform over the 40,101 unbought,
QC-clean, ranked export pairs — no gap filter and no ethnicity floor.** Both would have been selection
effects that make a population figure unquotable, so bands were *recorded* in `sample-meta.json` and never
imposed. The draw's band mix tracks its pool to within 2.6 points, which is the check that it really is
uniform. Median percentile gap **29.9** versus 4.9 in run 3.

Depth was set at **12 votes/pair** rather than run 3's 6, deliberately breaking the breadth-beats-depth
rule: two of the three deliverables (a leave-one-out ceiling, a binned calibration curve) are statistics
about per-pair vote *distributions*, and at 6 votes the "majority of the others" is 5 noisy votes, which
biases the ceiling down and *overstates* headroom.

The analysis order was fixed in advance and matters: **`rating_calibration.py` ran before any refit
consumed these pairs.** That property is spent permanently on first use, and it is the only path to a
publishable calibrated /10 that §5.6 blocked as circular.

#### Success criteria

Pre-registered in `prolific-soft-launch-form.md` §9.5: *"if headroom at 20–45 and 45–100 comes back below
~2 points with a tight interval, the remaining $15.3k is dead money and labelling stops at the 0–20 buy
zone."*

#### Results

##### Data quality — the queue fix and the rater screen both held

| | |
|---|---|
| Sessions | 303 (296 completed, 7 abandoned; abandoned votes are real and kept) |
| Distinct raters | 303, none with a second session |
| Votes per pair | **11 on 282 pairs, 12 on 2,185, 13 on 33** — sd 0.35, every pair covered |
| Test rows in the archive | 0 |
| Rating time, median | 5.6 min (p25 4.3, p75 6.8) — per-judgment median 2.1 s, 46% under 2 s |
| Rater agreement with the other 11 | mean **75.1%**, sd 7.6% |
| Determined majority | 2,397 of 2,500 (103 dead splits, 4%) — down from 13% in run 3 |
| Mean majority share | **74.0%** (run 2: 66%) · unanimous **10%** (run 2: 1%) |

The rise in majority share and the collapse in dead splits are the study design working: run 2–3 drew
near-ties on purpose, run 4 drew whatever the export contains.

##### Rater QC — 4 dropped from the fit, 1 recommended for Prolific rejection

| prolificPid | n | golds | agreement | median | fast | ties | action |
|---|--:|--:|--:|--:|--:|--:|---|
| `6a4aaa3ca9cd50f4e76237c3` | 105 | 3/5 | 58% | **294 ms** | 88% | 2% | **reject on Prolific** |
| `6a0f76e7789f0f1981c76d6d` | 105 | 2/5 | **43%** | 1,278 ms | 82% | 37% | pay, drop votes |
| `6a040eed69084b1ee1f6a64f` | 105 | 2/5 | 52% | 3,310 ms | 15% | 5% | pay, drop votes |
| `69c826eee2b0cff78396b165` | 105 | 2/5 | 57% | 2,735 ms | 10% | 2% | pay, drop votes |

Only `6a4aaa3c` clears the behavioural bar §5.5 set for refusing payment: a 294 ms median is ~4× below the
cohort's 1st percentile. The other three are statistical cases — pay them, exclude their votes. 17 raters
tripped a single signal and are all approved.

**The gold screen discriminated much better on this draw.** Gold-failing raters average **67.7%**
agreement against **75.5%** for passers — a 7.8-point gap, against 2.4 points in run 2 and 2.2 in run 3.
That is a real finding about the *screen*, not the raters: golds are wide-gap pairs, so on a run whose
pairs are mostly wide, gold performance and task performance measure the same thing. On an all-near-tie
draw they measure different things and the screen is close to noise. **Reading: weight golds more heavily
on wide draws, and almost not at all on hard draws.**

**One gold to retire.** `cmr1mr5cw06y796d59gfjfwv7` came in at **63% over 78 views** (49 left / 18 right /
11 tie), the only one under 70%. It was flagged in run 1 at 1/2 views, passed run 3 at 77–80%, and has now
failed on a third independent cohort. Retire it before the next launch; the other 19 sit at 79–99%.

##### The ordering hypothesis — confirmed, and it has a threshold, not a gradient

`scripts/gap_agreement_curve.py` (new) → `artifacts/panel-run-v4/gap-curve.json`. Bands cut on
`bt-refit-v2-qc`, the **VLM-only** ranking, which has never seen a human vote — so this is a genuinely
non-circular test of whether a VLM-derived ordering predicts what people do.

| percentile band | pairs | decisiveness | majority share | split-half | **ranking agrees** |
|---|--:|--:|--:|--:|--:|
| 0–2 | 68 | 0.178 | 67.8% [65.3, 70.2] | 71.2% | **53.2%** |
| 2–5 | 98 | 0.209 | 70.9% [68.1, 73.8] | 77.0% | **56.5%** |
| 5–10 | 198 | 0.214 | 71.4% [69.5, 73.3] | 76.0% | **57.8%** |
| 10–20 | 479 | 0.204 | 70.4% [69.2, 71.7] | 75.1% | **61.9%** |
| 20–45 | 883 | 0.246 | 74.6% [73.6, 75.6] | 80.5% | **69.6%** |
| 45–100 | 774 | 0.349 | 84.9% [83.9, 85.9] | 93.1% | **83.6%** |
| *coin-flip null at these ballot counts* | | *0.115* | *61.5%* | *~50%* | *50%* |

1. **Agreement with the ranking is strictly monotone across all six bands, 53.2% → 83.6%.** Overall
   `spearman(percentile gap, crowd decisiveness) = +0.408` with no binning. The experiment works: distance
   in a VLM-derived ranking is a real statement about how much people will agree, validated by votes that
   ranking never saw. **The circularity worry is answered — a circular partition cannot produce a monotone
   out-of-sample curve.**
2. **But crowd cohesion is a threshold, not a gradient.** Majority share is *flat* at 70–71% across 2–5,
   5–10 and 10–20, then rises at 20–45 and jumps at 45–100. Split-half stability has the same shape
   (77.0 → 76.0 → 75.1, then 80.5 → 93.1). So "farther apart ⇒ more agreement" is true across the whole
   range but **essentially absent below ~20 percentile points**. Below that threshold the crowd is
   uniformly ~70% cohesive regardless of how close the ranking says the faces are — which is exactly why
   the ranking's own accuracy still climbs there (53.2 → 61.9) while human consensus does not.
3. **Even the closest band carries real signal.** 67.8% majority share against a **61.5%** coin-flip null
   at these ballot counts is ~6 points of genuine agreement, reproducing §5.5's finding on an unselected
   sample. A dead-split pair is a real observation that two faces are level, not a failed measurement.

##### Population accuracy — the first non-worst-case numbers, and they are good

`analyze_panel_run.py`, 2,500 uniform pairs, `stratum` set to the band:

| band | pairs | split-half | human `H` | Gemini `G` | `G−H` | BT (VLM-only) | maj share |
|---|--:|--:|--:|--:|--:|--:|--:|
| random-0-2 | 68 | 67.4% | 63.0% | 58.7% | −4.2% | 52.4% | 63.5% |
| random-2-5 | 98 | 68.7% | 67.2% | 50.0% | −17.2% | 61.1% | 67.2% |
| random-5-10 | 198 | 71.2% | 69.4% | 61.8% | −7.6% | 63.4% | 67.9% |
| random-10-20 | 479 | 68.2% | 66.4% | 67.1% | +0.8% | 73.5% | 68.0% |
| random-20-45 | 883 | 74.3% | 73.1% | 82.0% | +9.0% | 82.3% | 72.0% |
| random-45-100 | 774 | 90.2% | 85.0% | 94.7% | +9.8% | 94.7% | 82.9% |
| **ALL** | **2,500** | **77.4%** | **74.9%** | **79.9%** | **+5.0%** | **81.5%** | **74.0%** |

4. **On a typical pair the ranking is better than a randomly chosen human at naming the crowd's choice.**
   81.5% versus a 74.9% leave-one-out individual ceiling *on the same metric*, +6.6 points. The free Gemini
   label is +5.0. Both figures are honest: `bt-refit-v2-qc` never saw a human vote, and these pairs entered
   no fit. At *vote* level the same comparison is 70.6% versus 70.4% — level, not ahead. See the metric
   warning above; neither number is wrong and neither may be quoted without saying which it is.
5. **Every headline number the programme has quoted was a worst case, by a factor we can now state.** The
   same ranking, on the same metric, scores 52.9% on run 3's near-tie pairs and 81.7% on a uniform draw.
   When a metric moves ~29 points on pair selection alone, "accuracy" without a stated pair distribution is
   not a number. **Quote both from now on: population accuracy for product claims, near-tie accuracy for
   research progress.** The full 2 × 2 (metric × pair distribution) is in
   `artifacts/panel-run-v4/accuracy-matrix.json`.

##### Honest /10 calibration — the ordering is right, the stated probability is not

`rating_calibration.py` against `bt-refit-v2-qc`, run **before** the refit
(`artifacts/panel-run-v4/calibration.json`):

| /10 score gap | pairs | votes | higher score wins | 95% CI |
|---|--:|--:|--:|---|
| 0–0.25 | 139 | 1,634 | 53.0% | [50.5, 55.4] |
| 0.25–0.5 | 208 | 2,431 | 58.0% | [56.0, 59.9] |
| 0.5–1 | 457 | 5,352 | 61.2% | [59.9, 62.5] |
| 1–1.5 | 429 | 5,033 | 65.4% | [64.0, 66.7] |
| 1.5–2 | 322 | 3,784 | 71.0% | [69.5, 72.4] |
| 2+ | 945 | 11,117 | 82.7% | [82.0, 83.4] |

6. **This is the publishable calibration curve, and it is monotone with tight intervals.** It is also the
   resolution statement the product needs: a **0.25 /10 difference is a coin flip (53.0%)**, and it takes
   **~2 points before 4 in 5 people agree**. Consistent with §5.5's coarser hard-pair version and with
   §5.6's 0.91-point uncertainty interval.

   Fitting `P(higher score wins) = σ(Δ/10 gap / T)` on the same pairs (`scripts/band_calibration.py`,
   new) gives **`T = 1.920` on the /10 scale**, which converts a score gap into a band directly:

   | agreement level | /10 gap required | share of the cohort inside that band of the median |
   |---|--:|--:|
   | 55% | 0.39 | 8% |
   | 60% | 0.78 | 18% |
   | **2 in 3** | **1.33** | 32% |
   | **3 in 4** | **2.11** | 50% |
   | 90% | 4.22 | 82% |

   This is the honest product sentence, and it needs no new data: *"you score 6.4; two out of three
   people would place you above someone at 5.1."* The band-versus-decimal argument is now
   quantitative rather than aesthetic. Note the three *different* bands this can be confused with —
   estimation, disagreement, and photo-level — in `programme-direction-review.md` §5; only the
   disagreement band is measured here, and it is the one that never shrinks.
7. **The ranking is still badly overconfident as a probability model.** `σ(Δθ)` implies 85.2% in the
   0.8–0.9 bin where people actually vote **59.1%** (−26.2 pts), and 98.4% in the top bin where they vote
   74.5% (−23.9). The *ordering* is trustworthy; the *stated confidence* is not. Any published probability
   or band must be temperature-scaled against this curve — the fitted `T ≈ 5.8` from §5.5 is the right
   order of magnitude and can now be refit out-of-sample.

##### Headroom — the $15.3k question, resolved

`label_information.py` → `artifacts/label-information-v2/report.json`, all three runs pooled (10,280
pairs, 95,245 votes), bands on `bt-refit-v2-qc`:

| pct gap | pairs | bought | remaining | cost @12 | VLM vs votes | 1 rater | **headroom (95% CI)** | verdict |
|---|--:|--:|--:|--:|--:|--:|--:|---|
| 0–2 | 2,663 | 1,692 | 971 | $563 | 50.2% | 56.4% | **+6.2 [+4.1, +8.3]** | **BUY** |
| 2–5 | 3,848 | 2,175 | 1,673 | $971 | 51.6% | 57.7% | **+6.2 [+4.4, +7.9]** | **BUY** |
| 5–10 | 5,387 | 2,416 | 2,971 | $1,724 | 53.2% | 57.1% | **+3.9 [+2.3, +5.6]** | **BUY** |
| 10–20 | 8,936 | 1,689 | 7,247 | $4,205 | 54.8% | 58.5% | **+3.6 [+1.7, +5.4]** | **BUY** |
| 20–45 | 15,524 | 1,272 | 14,252 | $8,269 | 68.4% | 67.5% | **−0.9 [−2.1, +0.4]** | **do not buy** |
| 45–100 | 11,546 | 1,036 | 10,510 | $6,098 | 83.3% | 81.9% | **−1.3 [−2.2, −0.5]** | **SKIP** |

On run 4's pairs *alone* — the unbiased subsample, no contamination from earlier hard-pair draws — the two
wide bands read **−1.3 [−2.9, +0.3]** and **−1.4 [−2.4, −0.5]**
(`artifacts/label-information-v2/report-run4only.json`).

8. **Labelling stops at a 20-point gap. Permanently.** The 45–100 interval is entirely below zero, so a
   human vote there is not merely a duplicate of the free label, it is *worse* than it: at 12 votes a
   single rater agrees with the crowd 81.9% of the time where Gemini agrees 83.3%. 20–45 is centred
   negative with an interval that no longer straddles anything interesting (best case +0.4 pt at $8,269).
   **$14,367 of the planned $22k is dead money**, and the pre-registered threshold ("below ~2 points with a
   tight interval") is met with room to spare. §5.7's caution that these bands were merely *unresolved* is
   now closed.
9. **§5.7's bias diagnosis was correct in direction and magnitude.** It predicted our hard-pair sample was
   +9.6 pts over-enriched for high-confidence VLM labels at 0–2 (flattering the VLM, understating buy-zone
   headroom) and −7.2 pts under-enriched at 20–45 (overstating skip-zone headroom). The uniform draw
   measures 42.6% high-confidence at 0–2 against 63.7% pooled, and 94.2% at 20–45 — both distortions
   confirmed, both running the direction predicted.
10. **The remaining buy zone is $7,463** across 12,862 unbought pairs inside 20 points, cheapest-first:
    0–2 $563 → 2–5 $971 → 5–10 $1,724 → 10–20 $4,205.

##### Marginal value of run 4 itself — near zero, exactly as designed

`panel_run_delta.py` withholds whole pairs (25%, 5 seeds), so none of their votes enter any fit
(`artifacts/panel-run-delta-v4/run-delta.json`):

| ranking | predicts a vote on a pair it never saw | step |
|---|--:|--:|
| machine labels only | 71.20% ±0.67 | — |
| + runs 2 and 3 | 72.16% ±0.69 | +0.96 |
| **+ run 4** | **72.25% ±0.66** | **+0.09** |

11. **Run 4 bought +0.09 ±0.36 points, positive on 2 of 5 seeds — $11,140 per point**, against run 3's
    $404. Its dose curve is *flat* (25% → 100% of the pool moves −0.02, +0.11, −0.00), where run 3's was
    accelerating. This is the **stop condition firing at 0.046 points per $500 against a 1.0 threshold**,
    and it is the third independent confirmation of the headroom table: uniform pairs do not move the
    ranking. (The 71.20% baseline is not comparable to run 3's 51.5% — different withheld pairs, easier by
    construction.)
12. **Run 4 was still worth its $968.58,** and the distinction matters for how future studies are
    justified: it was bought as a **measurement**, not as an improvement. It closed $14,367 of spend,
    produced the only honest calibration curve the programme will get, and delivered the first population
    accuracy. Judging it by `panel_run_delta.py` would be judging it on a goal it never had.

##### BT refit v5 — the new ranking of record

`refit_bt_panel.py` with all three runs: **10,280 pairs, 95,245 votes, 963 raters (9.3/pair)**. All §5.1
gates pass — 1 component per gender, no face under 15 distinct opponents, stability ρ 0.9924 F / 0.9914 M.
`beta_human` 0.338 F / 0.286 M. Spearman 0.9667 vs `bt-refit-v2-qc` over 2,866 faces; median /10 movement
0.26, **599 faces move more than half a point**.

Held-out validation (fit on half the raters, score the disjoint half, 47,865 withheld votes) is the
cleanest statement of where human money goes:

| stratum | VLM-only | + panel | gain |
|---|--:|--:|--:|
| random-0-2 | 53.1% | 61.1% | **+8.0** |
| random-2-5 | 56.6% | 61.9% | **+5.2** |
| random-5-10 | 57.3% | 63.0% | **+5.7** |
| random-10-20 | 62.6% | 63.4% | +0.8 |
| random-20-45 | 69.5% | 69.6% | +0.1 |
| random-45-100 | 83.0% | 83.0% | **+0.0** |
| close (runs 2–3) | 51.5% | 60.3% | +8.7 |
| topup-0-10 (run 3) | 52.1% | 58.4% | +6.2 |
| ALL | 60.2% | 64.5% | +4.3 |

13. **A completely different method reaches the same band verdict.** Headroom-over-VLM (measurement 8) and
    out-of-sample vote prediction (this table) are independent computations, and both put the panel's
    entire contribution inside a 10-point gap, decaying to *exactly zero* at 45–100. Three methods now
    agree, which is the standard §5.7's inverted-table trap taught us to demand.

##### The comparator on a population sample — the honest bad news

`eval_vs_panel.py` on run 4's pairs, restricted to pairs whose *both* faces are in each checkpoint's own
val split. The three `val_fraction: 0.5` arms share an identical 631-pair, 7,411-vote subset, so
`compare_panel_evals.py` pairs them properly (`artifacts/panel-arms-compare-run4.json`).

**Everything in this table is vote-level**, so it is comparable within itself and *not* to the 81.5% above.
The 631-pair subset is representative: BT scores 69.50% here against 70.6% on all 2,500 pairs, and the
ceiling is 69.90% against 70.4%.

| predictor | population accuracy | paired vs BT (VLM-only) |
|---|--:|---|
| NN `train-v10` (control, VLM labels) | 66.47% | **−3.05 [−4.70, −1.48]** |
| NN `train-v13-panel-hard` | 66.93% | **−2.59 [−4.19, −1.05]** |
| NN `train-v12-panel-soft` | **67.67%** | **−1.84 [−3.38, −0.31]** |
| export label (Gemini) | 68.84% | — |
| **BT `bt-refit-v2-qc`** | **69.50%** | — |
| ceiling (another rater) | 69.90% | — |
| BT `bt-refit-v4-panel` | 71.12% | *circular — fit on these votes* |

By band, and this is the actionable part:

| band | pairs | v10 control | **v12 soft** | v13 hard | BT (VLM-only) |
|---|--:|--:|--:|--:|--:|
| 0–2 | 19 | 50.2% | **55.1%** | 54.2% | 50.2% |
| 2–5 | 28 | 51.8% | **58.8%** | 55.5% | 51.8% |
| 5–10 | 58 | 58.2% | **59.4%** | 57.8% | 54.8% |
| 10–20 | 119 | 53.1% | 56.8% | 53.9% | **63.2%** |
| 20–45 | 206 | 66.2% | 66.2% | 66.2% | **69.5%** |
| 45–100 | 201 | 80.5% | 80.4% | 80.8% | **81.7%** |

14. **The comparator is a lossy copy of the ordering it was distilled from.** It is significantly *worse*
    than the VLM-only ranking on a population sample — all three intervals below zero — and the deficit is
    concentrated in 10–45, where the ranking is 3–7 points ahead. The panel labels demonstrably help
    (v12 − v10 = **+1.20 pts [+0.04, +2.35]**, separable) and the help lands exactly in 0–10, the band
    those labels came from. **So the bottleneck for the wide bands is not ground truth. The ordering is
    already in the labels we own; the network is failing to extract it.** No amount of further panel spend
    addresses that.
15. **Soft targets beat hard targets on a population sample, reversing the 2026-08-01 decision.** v12 −
    v10 is separable at +1.20; v13 − v10 is **−0.46 [−1.50, +0.58]**, a tie with the control. On hard pairs
    v13 beat v12; on population pairs the order flips. The reason is mechanical: run 4-type pairs are
    decisive, so their vote shares are far from 0.5 and carry margin information that rounding to 0/1
    destroys. **Future arms should use the soft target, and every arm should be scored on both pair
    distributions before a decision is recorded.**
16. **This is the cleanest antidote to the confirmation-bias risk in this programme.** Run 4 was designed
    to make three previously-favourable readings falsifiable, and it falsified two of them: the comparator
    is not "the rating" on a typical pair, and $14,367 of planned spend is worthless. A study that can only
    return good news is not a measurement.

#### Decision

| # | Decision |
|---|---|
| 1 | **`bt-refit-v5-panel` is the ranking of record.** All gates pass; 599 faces move > 0.5 /10 from v2-qc. |
| 2 | **Human labelling stops at a 20-point percentile gap, permanently.** Buy zone remaining: **$7,463**, cheapest-band-first. **$14,367 cancelled.** |
| 3 | **No further hard-pair study until the comparator can consume what we own.** §2b's standing rule ("never buy a round the trainer cannot read") now binds in a stronger form: the network is 1.8–3.1 points *behind* the ranking on population pairs, so the next dollar belongs to training, not labels. |
| 4 | **Switch the panel target back to soft (vote share).** Reverses 2026-08-01. `configs/train-v16-panel-run4.yaml` is the arm. |
| 5 | **Quote two accuracies, always, each labelled with its metric.** Majority-level population: ranking **81.7%** vs a 73.2% one-rater ceiling. Vote-level population: ranking **69.5%**, comparator **67.7%**, ceiling **69.9%**. Near-tie, vote-level: comparator 56.9% vs 59.1%. A single unqualified accuracy is meaningless — selection moves it ~29 points and the metric choice another ~10. |
| 6 | **The /10 resolution statement is now measured out-of-sample:** 0.25 points is a coin flip, 1 point is 61%, 2 points is 83%. Publish bands, not decimals — and temperature-scale before publishing any probability. |
| 7 | **Retire gold `cmr1mr5cw06y796d59gfjfwv7`** (63% over 78 views, third failure). Reject `6a4aaa3ca9cd50f4e76237c3` on Prolific; pay the other three and drop their votes. |
| 8 | **Weight the gold screen by draw type.** It separates quality at 7.8 pts on wide draws and ~2 pts on near-tie draws, because golds are wide pairs. |

#### Follow-up: `train-v16` folded run 4 into the loss, and the diagnosis held (2026-08-04)

Decision 3 above said the comparator's 10–45 deficit was an *extraction* failure rather than a label
shortage. That was an inference from the fact that BT recovers the same ordering from the same labels.
`train-v16-panel-run4` tests it directly: identical recipe to `train-v12-panel-soft`, with run 4's votes
added to `panel_labels`. Panel coverage of the train split went **14.7% → 19.4%** (1,927 → 2,546 pairs),
the extra pairs concentrated in exactly the wide bands where the network was weakest, and the leak-free
eval set grew 1,928 → 2,559 pairs. Best epoch 6, `val_accuracy` 0.7750.

Population accuracy on run 4's pairs, paired over the identical 631-pair subset
(`compare_panel_evals.py` → `artifacts/panel-arms-compare-run4.json`):

| arm | population accuracy | paired vs BT (VLM-only) | vs `train-v12` |
|---|--:|---|---|
| `train-v10` control | 66.47% | −3.05 [−4.70, −1.48] | — |
| `train-v13-panel-hard` | 66.93% | −2.59 [−4.19, −1.05] | — |
| **`train-v16-panel-run4`** | **67.43%** | **−2.09 [−3.61, −0.55]** | **+0.25 [−0.95, +1.46] — a tie** |
| `train-v12-panel-soft` | 67.67% | −1.84 [−3.38, −0.31] | — |
| BT `bt-refit-v2-qc` | 69.50% | — | — |
| ceiling (another rater) | 69.90% | — | — |

By band, against the ranking:

| band | pairs | `train-v12` | **`train-v16`** | BT (VLM-only) |
|---|--:|--:|--:|--:|
| 0–2 | 19 | 55.1% | **56.0%** | 50.2% |
| 2–5 | 28 | **58.8%** | 55.5% | 51.8% |
| 5–10 | 58 | **59.4%** | 57.9% | 54.8% |
| 10–20 | 119 | 56.8% | 54.7% | **63.2%** |
| 20–45 | 206 | 66.2% | 66.4% | **69.5%** |
| 45–100 | 201 | 80.4% | **81.5%** | 81.7% |

17. **More labels in the weak bands did not fix the weak bands, which is the confirming result.** v16 is
    statistically indistinguishable from v12 (+0.25, CI spans zero) and remains **2.09 points behind the
    ranking**, with 10–20 still 8.5 points adrift (54.7% vs 63.2%). Decision 3 is now *measured*: the
    10–45 deficit is not a ground-truth shortage, so no labelling programme can close it. **This is the
    strongest evidence the programme has that the next dollar belongs to the model.**
18. **The one band that did respond is the widest.** 45–100 moved 80.4% → 81.5%, essentially matching BT's
    81.7% — 66% of run 4's draw sat in the two widest bands, so the labels landed where they were aimed and
    the network used them *there*. It is the middle of the scale, 10–45, that it cannot learn from any
    supervision we have tried.
19. **Checkpoint selection was on the wrong metric, and every arm in this family shared the flaw.**
    `train.py` saved `best.pt` on `val_accuracy`, which is scored against the **export's Gemini labels** —
    the exact metric §5.3.1 established cannot see a panel-driven improvement. v16 peaked there at epoch 6
    while epochs 5, 7 and 10 were within 0.6 points, so the saved checkpoint was close to arbitrary with
    respect to the metric we care about. The comparison across arms stayed fair because all of them did it,
    but every arm may have been leaving something on the table.

    **Fixed 2026-08-04.** `train()` now computes **`panel_val_accuracy`** every epoch — the share of real
    human votes agreeing with the model's pick, over val-split pairs with ≥ `panel_min_votes` — and selects
    `best.pt` on it whenever there are at least 200 such pairs (2,548 at `val_fraction 0.5`, 447 at 0.2).
    `val_accuracy` is still computed and recorded, because it is the only figure comparable to the
    pre-panel runs. The checkpoint now stores `selected_on` and `selected_score`, and `metrics.json`
    carries `selected_epoch`, `panel_val_pairs` and `best_panel_val_accuracy`.

    Two validations before trusting it. The new metric reproduces `eval_vs_panel.py` to within 0.02 points
    on the same checkpoint (**58.96%** inline vs **58.98%** offline on v16 — the gap is the 11 tie-pairs
    `skip_ties` drops). And a 2-epoch smoke run showed the two metrics genuinely disagreeing: epoch 2
    scored **higher** on `val_accuracy` (0.6407 vs 0.6367) and **lower** on human votes (0.5725 vs 0.5848),
    so the old code would have kept the worse checkpoint. `train-v17-panel-select` is v16 rerun under the
    new rule and measures what it was costing us.

#### Why neighbouring faces in the ranking look mis-ordered — it is precision, not labels

Prompted by a direct and fair complaint: *"a face sits next to faces that are obviously a different tier
of attractiveness — is that a fault in the labels, or in the algorithm?"* **Neither, and the data answers
it without a new experiment** (`scripts/rank_resolution.py`, new, over
`artifacts/bt-refit-v4-panel/uncertainty.csv`):

| | female | male |
|---|--:|--:|
| faces ranked | 1,422 | 1,444 |
| median **effective** comparisons per face | 10.1 | 10.2 |
| median 95% rank interval | **237 places** (17% of the list) | **244 places** (17%) |
| adjacent pairs the data can tell apart | **0.00%** | **0.00%** |
| ranks apart before a difference is real | **~208** | **~210** |

20. **Not one adjacent pair in the entire ranking is statistically distinguishable, and you have to move
    ~210 places before the ordering is supported by evidence.** The list prints a total order because
    sorting has to emit *something*; that printed order is the over-precise part, not the model. A face at
    rank 400 and a face at rank 500 are, as far as 34 comparisons each can establish, the same face. The
    ranking supports roughly **1,430 ÷ 210 ≈ 7 tiers per gender**, not 1,430 ranks — which is an
    independent argument for the band product arriving at almost exactly the ~7 bands the calibration
    curve implies.
21. **Two separate limits produce that, and only one is buyable.** *Estimation*: each face has ~34 raw but
    only ~10 **effective** comparisons, and standard error falls as 1/√k, so tripling coverage would shrink
    the 237-place interval only to ~140. *Crowd*: `T = 1.920` means faces within 0.39 /10 points are a
    coin flip to real people, so below that there is no ordering to find at any price. The complaint is
    the visible symptom of both.
22. **Consequence for reading the ranking by eye.** Scrolling the sorted list and finding a jarring
    neighbour is the *expected* appearance of a correctly-fit model with honest uncertainty. The diagnostic
    that would indicate a real defect is different: a face whose 95% interval is unusually wide, or that
    carries the `notIdentified` / `lowInfoOpponents` flags in `uncertainty.csv`. Check those columns before
    concluding a face is mis-ranked.

#### Five arms, none of which closed the 10–45 band (2026-08-04)

Decision 3 said the comparator's mid-gap deficit was an *extraction* failure. Five arms then attacked it
from five directions. **All five are ties**, and the pattern is now the finding rather than any individual
result. Population accuracy on run 4's pairs, all vote-level, all paired over the identical 631-pair
leak-free subset (`artifacts/panel-arms-compare-run4.json`, `artifacts/resolution-compare-run4.json`):

| arm | what it changed | population accuracy | vs BT (paired) |
|---|---|--:|---|
| `train-v10` | control — VLM labels only | 66.47% | −3.05 [−4.70, −1.48] |
| `train-v13` | hard panel target | 66.93% | −2.59 [−4.19, −1.05] |
| `train-v19` | **variance head** (never run before) | 67.39% | −2.11 [−3.63, −0.57] |
| `train-v16` | **+ run 4's votes** (coverage 14.7% → 19.4%) | 67.43% | −2.09 [−3.61, −0.55] |
| `train-v17` | **checkpoint selection fixed** | 67.63% | −1.89 [−3.49, −0.32] |
| `train-v12` | soft panel target | 67.67% | −1.84 [−3.38, −0.31] |
| — | | | |
| `train-v20` | resnet50 @ **112** px | 64.11% | −5.40 [−7.32, −3.51] |
| `train-v21` | resnet50 @ **224** px | 64.27% | −5.23 [−7.14, −3.40] |
| **BT `bt-refit-v2-qc`** | | **69.50%** | — |
| ceiling (another rater) | | 69.90% | — |

The 10–20 band, which is the specific deficit: v12 56.8%, v16 54.7%, v17 55.7%, v19 54.6%, v20 56.0%,
v21 53.5% — against BT's **63.2%**. Nothing moved it.

27. **Resolution is ruled out, and this was the last perceptual hypothesis.** The source photos are
    **1024 × 1024**, so 112 px uses ~1.2% of the available pixels — the hypothesis was live. A matched pair
    varying nothing but `image_size` gives **64.11% vs 64.27%, a tie (−0.15, CI [−2.07, +1.78])**, and on
    the 10–20 band 224 px is *worse* (53.5% vs 56.0%). Note the direction: 224 px did help slightly on the
    near-tie-dominated training metric (`panel_val_accuracy` 0.5674 → 0.5756) and not at all on the wide
    bands, which is the opposite of what a detail-limited mid-gap would predict.
    *(Recorded separately: `arcface_r50` **cannot** take 224 px — its converted ONNX graph ends in a fixed
    `Linear(25088, 512)` sized for a 7×7 feature map. Hence the resnet50 pair. resnet50 is a weaker
    backbone in absolute terms, so these two arms are comparable only to each other.)*
28. **The variance head learned score-dependent uncertainty, not photo quality.** σ does vary (0.42 to
    1.85, CV 0.22, so it did not collapse) but it correlates **+0.52 with the /10 score** — it learned that
    the thin top of the ranking is poorly determined, which is true. Regress that out and the residual is
    flat across every photo-quality flag the QC pass recorded: obstruction −0.027, poor_lighting −0.015,
    low_resolution −0.005, no-issues +0.003, against a σ sd of 0.169. **So this does not give the
    per-photo σ the multi-photo band needs.** The plausible fix if we return to it is training with
    quality augmentation (random blur, downsampling, lighting jitter) so degradation is actually in the
    signal; without it there is nothing for σ to attach to.
29. **The converging read.** Five failures across labels, targets, selection, uncertainty modelling and
    resolution — plus the fact that `train-v10`, which saw **no** panel labels at all, loses the same
    bands — say this is not a tuning problem. BT solves a **global** system over 47,914 comparisons; the
    comparator sees only **local** pairs and must induce an absolute function from them, on 1,430 faces
    per gender. **The likely conclusion is that the comparator should not be asked to reproduce the global
    ordering at all.** Ask it for local comparisons, which it is measurably good at (it beats BT under a
    10-point gap), and take the global ordering from BT via reference-set placement. That reframes the
    "deficit" as a design error in how the model is *read*, not a defect in the model — see the next
    section, which was measured before this conclusion was reached.

#### Reference-set inference — how a new face should actually be placed (2026-08-04)

The comparator is *trained* pairwise and *read* as an absolute score. A siamese network's output has no
anchored zero and no anchored scale, because only differences enter the loss, so any bias shift — or any
distribution shift between cohort photos and a user upload — moves every score together while changing not
one pairwise comparison. That is the most likely reason an upload scores oddly while pairwise accuracy
looks fine, and it is a *placement* failure rather than an *ordering* failure.

`scripts/reference_set_inference.py` (new) uses the model the way it was trained: compare the new face
against N cohort faces of known θ and fit `P(new beats ref_j) = σ(θ_new − θ_j)` by maximum likelihood. The
problem is one-dimensional and strictly concave, so Newton converges in a few steps and the curvature at
the optimum gives a standard error for free, `se(θ) = 1/√Σ p_j(1−p_j)`.

Measured with `train-v16`, subjects being **val-split faces the model never saw** and references being
seen faces — exactly the production arrangement (`artifacts/train-v16-panel-run4/reference-set.json`):

| references | Spearman vs BT (F / M) | median error /10 | p90 error /10 | se(θ) |
|---|--:|--:|--:|--:|
| **raw scalar (what we do today)** | **0.857 / 0.830** | — | — | — |
| 10 | 0.845 / 0.820 | 0.80 | 1.85 | 1.03 |
| 25 | 0.853 / 0.817 | 0.57 | 1.57 | 0.71 |
| 50 | 0.857 / 0.827 | 0.54 | 1.41 | 0.48 |
| **200** | **0.857 / 0.831** | **0.53** | **1.33** | **0.24** |
| 707 / 732 (all available) | 0.857 / 0.830 | 0.53 | 1.33 | 0.13 |

23. **~200 references is the operating point.** Ordering accuracy plateaus by 50 and is flat to 700; past
    200 only `se` keeps shrinking, and `se` is estimation precision, not accuracy — the error floor is
    **0.53 /10 median, 1.33 p90** whatever the set size, which is the *comparator's* limit rather than the
    reference set's. Using the whole 2,866-face ranking is defensible and costs 14× the forward passes for
    nothing measurable.
24. **10–20 references is genuinely too few** — 0.845 vs 0.857 Spearman and 0.80 vs 0.53 median error. The
    original "reference panel of 10–20" sketch would have underperformed the raw scalar it was meant to fix.
25. **No curation is needed, and curating would be worse.** Each reference's weight in the fit is
    `p_j(1−p_j)`: 0.25 for an evenly-matched opponent, collapsing toward zero for a mismatch. The set
    self-selects toward faces near the subject, so a hand-picked ladder adds selection bias and buys
    nothing a stratified sample of the whole ranking does not already give. This also disposes of the worry
    that hand-picking "who is an 8" would bake in one person's taste.
26. **Discarding the scalar's magnitude costs nothing.** The fit uses only the *sign* of each comparison
    and still matches the raw scalar on Spearman. So the anchoring is free — same ordering, plus a
    calibrated position on the cohort scale and a standard error the band can use.

**What this measurement cannot show, and it is the important caveat.** Both methods were scored on
*cohort* faces. The raw scalar's hypothesised failure is distribution shift on *user uploads*, which
cannot be simulated without uploads of known ground truth. So reference-set inference is proven **not
worse** and proven to need only ~200 references; its benefit on real uploads is untested. Ship it as free
insurance against a failure mode we cannot currently measure, not as a measured win.

**Two further measurements that decide the engineering** (same script, per-gender output):

`se(θ)` converted to the /10 scale — the estimation band a 200-face reference set actually produces:

| position | /10 | estimation half-width | disagreement half-width at 2-in-3 |
|---|--:|--:|--:|
| p25 | 3.99 | ±0.08 | ±0.67 |
| median | 5.00 | ±0.14 | ±0.67 |
| p75 | 6.10 | ±0.10 | ±0.67 |
| p90 | 7.00 | ±0.07–0.11 | ±0.67 |

30. **Estimation uncertainty is ~5× smaller than human disagreement, so reference-set size is
    optimising the wrong term.** Composed in quadrature, `√(0.14² + 0.67²) = 0.68` — the estimation
    part contributes 0.01 of the total. That is both why 200 is enough and why the band is honest: its
    width is a property of the world, not of our sample size.

Placement quality by where the subject sits (200 references):

| subject band | median error /10 | p90 | no finite MLE |
|---|--:|--:|--:|
| bottom 25% | **0.92** F / 0.76 M | 1.88 | 0% |
| middle 50% | **0.41** F / 0.50 M | 1.01 | 0% |
| top 25–5% | 0.51 / 0.62 | 1.28 | 0% |
| **top 5%** | 0.87 / 0.64 | 1.92 | **15.8% (F)** |

31. **Placement is twice as accurate in the middle of the scale as at either end, and more references
    do not fix it** (the same table at all 707 references is unchanged). Two production consequences:
    the band must **widen at the extremes** rather than being a constant width — use the per-user
    `se(θ)` the fit already returns — and ~16% of top-5% female faces **beat every reference**, which
    has no finite MLE. Cap θ at the top reference plus a margin and flag it rather than letting the
    optimiser run away.

**A full BT refit per user is unnecessary — it is the same estimator.** Adding the user as a new node
and re-solving while holding the existing 47,914 edges fixed *is* the conditional MLE, which is
`fit_theta`. Letting the 2,866 reference θ float too would move them by roughly 200/47,914 ≈ 0.4%,
numerically irrelevant, at the cost of a full refit per upload instead of one 1-D Newton solve. Keeping
the reference θ fixed is also what keeps every user on one comparable scale.

**Do not hand-label the reference faces out of 10.** It substitutes one person's anchors for 95,245
measured votes, discards the θ scale's metric structure, and is redundant with the pre-registered
percentile→/10 curve. If the intent is to change *what a 7 means*, change that curve — one function,
the §5.4 anchor-ladder decision — not 200 individual judgments. Curating for "a clean progression" is
also picking noise: 0.00% of adjacent pairs are distinguishable (measurement 20).

**Reference *spread* matters, reference *quality* does not.** Median placement error in /10 by
selection rule (715 F / 712 M unseen subjects):

| reference selection | female | male |
|---|--:|--:|
| random 200 | 0.544 | 0.573 |
| **stratified 200 — even across 8 percentile bands, random within** | **0.516** | **0.514** |
| stratified 200, tightest θ interval within band | 0.524 | 0.543 |
| stratified 200, most **panel** votes within band | 0.549 | 0.555 |
| stratified 200, most total comparisons within band | 0.541 | 0.536 |
| all 707 / 732 | 0.514 | 0.551 |

32. **Do not filter the reference set to panel-covered or well-measured faces.** Stratifying across the
    score range buys 0.03–0.06 /10; selecting for measurement quality buys nothing and selecting for
    panel votes is marginally *worse*. Expected on reflection: each reference's θ enters as a fixed
    constant with weight at most 0.25 spread over 200 references, so individual θ errors average out,
    while panel-covered faces are exactly those drawn into near-tie pairs — selecting them concentrates
    references in dense parts of the scale and costs the coverage that does matter.

**Correction to measurement 26's framing.** It said the raw scalar's problem is that it is unanchored.
That is weaker than stated: the dashboard's Inference tab already computes a **rank** within the
model's own score distribution (`(cohort_scores < user_score).mean()`), and a rank is
**shift-invariant** — a global bias shift changes nothing. Which is also why reference-set placement
matches the raw scalar's Spearman almost exactly: both are monotone functions of the same rank. So what
it buys is narrower than "fixes the score" — a principled per-user **standard error** (the tab gives a
point estimate with none), a position on the interpretable **θ** scale, explicit handling of the ~16%
of top-tier faces with no finite solution, and 200 forward passes instead of 2,866. The likelier causes
of an odd-looking upload score, in order, are: **the point score being shown at all** when p90 error is
1.33 /10; **normalisation parity** with the faceiq-labs MediaPipe crop, which is a per-face error no
ranking cancels; and genuine model error, worst at the extremes.

**One reference set, not one per subgroup or per photo condition.** The set's job is to locate a face on
the *existing* cohort percentile scale; splitting it by ethnicity or by lighting would define a different
scale per group, so a user's score would depend on which set they were compared against and two users
would not be comparable. Photo condition is real but belongs on **σ, not the yardstick**: a poor photo
should widen the band, which is what `variance_head` (`train-v19`) is for. Make the set representative of
the cohort, stratify it across the score range, and audit win rates by subgroup rather than engineering
separate scales.

#### Artifacts

| What | Where |
|---|---|
| Raw archive (gitignored) | `labels/panel-run-4-random/results/{judgments,sessions}.jsonl` — 31,237 + 303 rows |
| `train-v16` (run 4 in the loss) | `configs/train-v16-panel-run4.yaml` · `artifacts/train-v16-panel-run4/{metrics,panel-eval,panel-eval-run4}.json` · 67.43%, a tie with v12 |
| **Reference-set inference** | `scripts/reference_set_inference.py` → `artifacts/train-v16-panel-run4/reference-set.json` · ~200 refs is the operating point; 0.53 /10 median placement error |
| **Production placement** | `src/faceiq_pref/placement.py` — the shipping form of the above: θ, `se`, percentile, /10, band, tier, plus tier-stratified reference sampling and `ANCHORS_TOP10`. Wired into the dashboard's Inference tab |
| **Off-cohort validation harness** | `scripts/validate_placement.py` (+ Inference tab labeller) · built and smoke-tested 2026-08-04, **unrun — needs 50–100 photos from outside faceiq-labs.** Separates ordering / band / calibration, and scores against the labeller's own repeat rate rather than against 100%. See production-scoring-pipeline §7 |
| **Checkpoint choice for production** | `scripts/placement_by_checkpoint.py` → `artifacts/placement-by-checkpoint.json` · **`train-v14-panel-ship` wins by a wide margin** — see the note below |
| **Labs composite** | `scripts/labs_composite_eval.py` → `artifacts/<run>/labs-composite.json` · **do not blend** — improves the BT proxy, not human agreement |
| **Value of ranking quality to the score** | `scripts/ranking_value_to_placement.py` → `artifacts/ranking-value-to-placement.json` · **flat across the whole refit ladder** — see below |
| **Rank resolution** | `scripts/rank_resolution.py` → `artifacts/bt-refit-v4-panel/rank-resolution.json` · 0% of adjacent pairs distinguishable; ~210 places for a real difference; ~7 supported tiers per gender |
| Rater QC, bands, golds | `artifacts/panel-run-v4/{raters.csv,majority-labels.csv,reject-pids.txt,prolific-rejections.txt}` |
| **Honest calibration** | `artifacts/panel-run-v4/calibration.json` — run before any refit consumed run 4 |
| Gap–agreement curve | `scripts/gap_agreement_curve.py` → `artifacts/panel-run-v4/gap-curve.json`, `gap-curve-pooled.json` |
| **Band widths from the curve** | `scripts/band_calibration.py` → `artifacts/panel-run-v4/band-calibration.json` — `T = 1.920` on the /10 scale; 2-in-3 agreement needs a **1.33** point gap, 3-in-4 needs **2.11** |
| Refreshed headroom | `artifacts/label-information-v2/{report.json,report-run4only.json}` |
| Marginal value | `artifacts/panel-run-delta-v4/run-delta.json` |
| **Ranking of record** | `artifacts/bt-refit-v5-panel/{ratings.csv,metrics.json}` |
| Comparator population eval | `artifacts/train-v1{0,2,3,4}-*/panel-eval-run4.json` + `panel-pairs-run4.csv`; paired in `artifacts/panel-arms-compare-run4.json` |
| Next training arm | `configs/train-v16-panel-run4.yaml` (not yet run — needs the GPU box) |

#### Notes

**The uniform draw is spent.** Run 4's pairs are now inside `bt-refit-v5-panel`, so they can never again
serve as an out-of-sample calibration set. The curve in measurement 6 is the one we have; re-testing
calibration on a future ranking requires buying fresh random pairs (~$400 at 700 pairs is enough for the
curve alone, since it does not need the 300-participant representative floor).

**`refit_bt_panel.py` silently ignores QC exclusions unless told.** `--exclude-faces` /
`--exclude-genders` have no defaults, so the first v5 attempt ranked 2,999 faces instead of 2,866 and
recorded `qcExcludedFaces: 0` in `metrics.json`. Gates still passed, which is what makes it dangerous.
Check that field on every refit; better, give the arguments the same defaults `eval_vs_panel.py` uses.

**We were about to ship the wrong checkpoint, and `bestValAccuracy` is why.** Ranking every checkpoint
on *placement* error — the production question — instead of on val accuracy inverts the ordering.
`scripts/placement_by_checkpoint.py --common-val` scores every run on the same held-out faces (the
`val_fraction 0.2` split, a subset of the 0.5 split under a shared seed, so genuinely held out for all
of them):

| checkpoint | median /10 err F / M | p90 | tier exact | Spearman |
|---|--:|--:|--:|--:|
| **`train-v14-panel-ship`** | **0.34 / 0.28** | **0.93** | **67.2%** | **0.926** |
| `train-v19-panel-variance` | 0.47 / 0.39 | 1.20 | 56.6% | 0.875 |
| `train-v13-panel-hard` | 0.44 / 0.42 | 1.28 | 55.3% | 0.867 |
| `train-v12-panel-soft` | 0.45 / 0.43 | 1.27 | 55.9% | 0.868 |
| `train-v17-panel-select` | 0.47 / 0.47 | 1.26 | 57.2% | 0.869 |
| `ensemble-v7-v1` | 0.47 / 0.47 | 1.17 | 60.7% | 0.898 |

The cause is not subtle and was flagged in §6.4 before it was measured: **v14 is the only panel arm
trained at `val_fraction: 0.2`**, so it saw 33,449 pairs against the others' 13,113. Every v12–v21 arm
ran at 0.5 deliberately, to hold out ~2,548 panel pairs rather than ~330 for the human-grounded
comparison. That made them controlled comparisons of *label recipes* and never production candidates.
Ten points of tier accuracy and 0.13 /10 of placement error is what the 0.5 split cost, and it went
unnoticed because no arm was ever scored on placement.

Two corrections follow. **The "p90 placement error is 1.33 /10" figure quoted across the docs was
measured on a 0.5-split model; on v14 it is 0.93** — still wide enough that a point score overclaims,
so the band argument stands, but the number was pessimistic by ~30%. And **the best label recipe has
never been trained at 0.2**: `configs/train-v22-ship-0.2.yaml` is v17's recipe at v14's data volume,
~$2 of GPU, and it is the last cheap experiment with an obvious upside.

**Blending the placed score toward its tier's mean is a measured no-op, and the reason is instructive.**
Proposed as a way to let a few mis-ranked reference faces wash out. On 499 held-out faces with v14,
median error moves 0.319 → 0.314 at a 70/30 blend and tier-exact accuracy does not move at all (67.3%
at every weight, because averaging toward a tier's own mean never moves anyone out of that tier).
Splitting by whether the tier was right shows the mechanism: where the tier is correct the blend helps
(0.226 → 0.192) and where it is wrong it hurts (0.646 → 0.719), and at 67% tier accuracy those cancel.
It is a bet on the tier assignment, and it imports the mis-ranked faces through the tier mean rather
than washing them out. The formal version — empirical-Bayes shrinkage toward the *population* mean,
which does not depend on the tier being right — moves θ by 0.3% at `se = 0.24` against a population θ
variance of 14.7–17.7. Correct, principled, negligible. The composite the proposal wanted is already
what the MLE computes.

**A better ranking does not make a better score, and that reroutes the entire spend argument.**
`scripts/ranking_value_to_placement.py` holds the comparator fixed at `train-v14-panel-ship` and swaps
only the ranking supplying reference θ, walking the refit ladder from zero human votes to all 65,894:

| ranking supplying reference θ | panel runs in it | vs majority | vs votes |
|---|---|--:|--:|
| `bt-refit-v2-qc` | none (VLM only) | 82.1% | 70.9% |
| `bt-refit-v3-panel` | run 1 | 82.1% | 70.9% |
| `bt-refit-v4-panel` | runs 1+3 | 82.1% | 70.9% |
| `bt-refit-v5-panel` | runs 1+3+4 | 81.9% | 70.8% |

Spread across the ladder: **0.2 points, downward.** The bias ran toward finding an effect — run 4's
votes are inside v5, so evaluating v5 on run 4 pairs is partly in-sample. The mechanism is the same
`p(1−p)` weighting that makes reference curation pointless: θs enter as fixed constants with weight at
most 0.25 each over 200 references, so improvements average out and the comparator's ordering dominates.

That left exactly one route for ground-truth spend to reach a user's score — the comparator's
**training set** — and `train-v22-ship-0.2` tested it the same afternoon. v17's label recipe (soft
panel targets, run 4 folded in, human-grounded checkpoint selection) at v14's data volume:

| checkpoint | median err F / M | tier exact | Spearman | **vs panel majority** |
|---|--:|--:|--:|--:|
| `train-v14-panel-ship` | 0.34 / 0.28 | 67.2% | 0.926 | **81.7%** |
| `train-v22-ship-0.2` | 0.35 / 0.30 | 68.7% | 0.937 | **81.7%** |

**+0.00 pts, 95% CI [−0.96, +0.96]** on a paired bootstrap over 2,390 pairs. A dead tie, which means
essentially all of v14's advantage over the 0.5-split arms was **training-pair count, not label
quality**. (The two trade small wins on the BT-proxy columns; per the rule below those do not count,
and the two checkpoints are interchangeable.)

**Both routes are therefore measured shut, and the $7,463 buy zone is closed.** The headroom table
remains correct that a purchased vote beats the free label on those pairs — label quality has simply
stopped being the binding constraint. Scope: v14 already contains panel runs 1 and 3, so this tests
the *increment*, not whether panel labels help at all; §5.3.1 measured +2.00 pts for wiring them in
the first time. The programme was right to buy them and is now past the point where more of the same
pays. Reopening would need a different *kind* of label — test–retest pairs, or a demographically
different panel — not more pairs from the same pool.

**The Labs composite is the cleanest example of proxy-chasing this programme has produced, and it
nearly shipped.** `scripts/labs_composite_eval.py` blends the Labs deterministic score into the placed
/10 post-hoc, at weight `w` on the comparator:

| w on comparator | BT median err | BT tier exact | vs panel majority | vs panel votes |
|--:|--:|--:|--:|--:|
| **1.00 (placement only)** | 0.319 | 67.3% | **81.9%** | **70.8%** |
| 0.85 | **0.283** | **69.9%** | 81.7% | 70.8% |
| 0.80 | 0.295 | 69.5% | 81.5% | 70.7% |
| 0.00 (Labs only) | 0.652 | 42.9% | 71.7% | 64.6% |

Against **BT** it is a clear win: median error down 11%, tier accuracy up 2.6 points, bootstrap CI
`[+0.032, +0.061]` excluding zero, and it replicates on `train-v12-panel-soft` (55.9% → 60.3% tier).
Against **real human votes** it is flat to slightly worse at every weight. The error correlation
between the two sources is **+0.008** — essentially independent, normally the exact condition for a
blend to pay, and the reason the BT result is so persuasive.

Both cannot be improvements. BT θ is a proxy fitted on Gemini labels; the panel votes are the target.
A change that moves the proxy and not the target is fitting the proxy's error. Had only the first
measurement been run, a "statistically significant 11% improvement" would have gone into the pipeline
and helped nobody. **Standing rule from here: any composite, ensemble or post-hoc adjustment is scored
against panel votes before it counts. BT-target numbers are for debugging.**

This is weak evidence about the EBM, whose input is genuinely different, but the bar is now explicit
and the harness exists: beat 81.9% / 70.8% on panel votes.

**The user-facing sentence had to be rewritten, and the bug was in the copy rather than the maths.**
The first version read *"67% of people would place you above someone scoring 3.8"*, which is
arithmetically exact and reads as though some raters think the face **is** a 3.8. It does not: 3.8 is a
different, lower-scoring face one resolvable gap down. Replaced by `agreement_vs_tier_below()`, which
compares against a typical face in the tier below and reports the *exact* share for that score (~62% at
the bottom of a tier, ~71% at the top) rather than always saying "two thirds". Recorded because it is
the general failure mode of shipping a calibrated quantity: the number was right and the sentence built
on it implied something we never measured.

**Reference sets are per-gender and cannot be pooled.** The pair queue was same-gender only, so the two
BT graphs are disconnected by construction and their θ scales share no common zero — every placement
draws references of the subject's own gender, and every /10 is a position within that gender's cohort.
The Inference tab now has a browser for the whole selected reference set in θ order, grouped by tier,
which also makes the top-tier thinness (~14 faces per gender) visible as the mechanism behind the
unbounded-MLE rate.

**Three raters volunteered task-validity observations worth keeping.** One asked whether "Skip" or "Save
& Continue" was correct for a tie (a UI ambiguity, same family as run 3's tie-button wording); one wrote
*"quite a few where neither seemed attractive — just a matter of personal taste"*, which is the honest
description of a uniform draw and not a complaint; one wrote *"this was disturbing"*, worth noting for the
ethics section of any write-up.

---

### 5.9 The cohort is not 3,000 people — duplicate identities, and the test–retest set we already owned (complete, 2026-08-04)

Found while building the off-cohort validation set, not by looking for it. To check whether the new
labs photos contained anyone from the cohort, both sides were embedded with **ArcFace R50**
(`buffalo_l w600k`, a face *recognition* model, so it measures who someone is rather than what they
look like). The threshold was to be calibrated off the cohort's own pairs on the assumption that
3,000 faces are 3,000 different people. The largest such similarity came back **1.000**.

**The cohort contains the same person more than once.** Two independent readings, one of which needs
no threshold at all:

| measure | result |
|---|---|
| **byte-identical image files** (sha256, no model involved) | **35 groups, 73 rows** — the same file stored under 2–3 different `sourceFaceId`s |
| complete-linkage identity clusters at cos ≥ 0.60 | **363 people appear more than once**, 1,052 rows involved, largest group 20 |
| ditto at cos ≥ 0.70 (very strict) | 480 redundant rows |

Complete linkage, not single linkage: an early single-linkage pass reported a 45-face cluster that
turned out to be one heavily-photographed blonde woman with a dozen unrelated lookalikes *chained*
onto her through intermediate similarities. Requiring every member to be mutually above the cut
removes that failure mode, and the surviving two-face clusters are unambiguous on sight — the same
celebrity twice, the same selfie twice.

**This defeats `split_by_face_id`, and it does not matter.** The hard rule in CLAUDE.md splits
train/val by face id to stop a face appearing on both sides. One person owning several face ids
walks straight through it: at `val_fraction 0.2` — the shipping checkpoint's split — **32.7% of val
faces have their own identity present in train**, touching **54.2% of val pairs**. That is the exact
leak the rule exists to prevent.

Measured on `train-v14-panel-ship`, the inflation is **zero**:

| val pairs | n | accuracy |
|---|---|---|
| identity also in train | 1,035 | 77.9% |
| genuinely unseen identity | 862 | **78.0%** |

So no reported accuracy needs revising. That is itself evidence for §5.8's diagnosis: a model that
had memorised individuals would score higher on the half it had seen, and this one does not — it is
learning population-level appearance and failing to *extract* fine ordering, exactly as five training
arms suggested. Worth keeping as the standing answer to "is the comparator just memorising faces".

> **What it does cost.** ~10% of a 200-face reference set (17 female, 21 male) is a second photo of
> someone already in it, which quietly doubles those people's weight in the MLE. Small, and worth
> fixing when the reference set is next rebuilt, but not a correctness bug.

**The genuinely valuable part: we have owned a test–retest set all along.** §5.6 and the training
charter both record that photo-to-photo stability was unmeasurable because "the cohort is one front
photo per person", and the multi-photo band-narrowing feature was shelved on that basis. That premise
was false. 363 identities carry two or more *different* photos, each independently ranked by BT, which
is precisely a repeat measurement.

Scoring each identity's photos through `bt-refit-v5-panel`:

| same person, different photo | /10 |
|---|---|
| median spread | **0.65** |
| p90 spread | **1.54** |
| share differing by more than 1.0 | 28% |
| max | 4.23 |

**The p90 photo-to-photo spread (1.54) is larger than the model's p90 placement error (0.93).** Which
photo a user uploads moves their score more than which checkpoint we ship does. Three consequences:

1. It is independent support for shipping a **band** rather than a point score — and it is the first
   estimate of the *photo* band, which until now was the one band with no measurement behind it.
2. It re-opens **multi-photo inverse-variance narrowing** as a buildable feature rather than a
   proposal waiting on data.
3. It reframes the remaining error budget. Chasing another 0.1 /10 out of the comparator is worth
   less than asking the user for a second photo.

Caveat on the figure: the spread mixes genuine photo-to-photo variation with BT's own estimation
noise on each face, so 0.65 is an **upper bound** on the photo component. Separating them needs each
face's BT standard error, which `bt_uncertainty.py` already produces — not yet done.

Artifacts: `scripts/audit_validation_identities.py` (embedding, thresholding, cluster reports),
`artifacts/.cache/cohort-arcface.npz` (3,000 cohort embeddings, cached).

---

## 6. Phase 4 — Primary evaluation (publishable experiment)

**This is the experiment that answers the product/research claim.** Everything in §2–§5 supports this section as methods. Evaluation faces and human ratings must be **held out** from labeling, prompt tuning, and training.

### 6.1 Primary research question

<!-- e.g. How accurately does the preference model predict human-assessed attractiveness on held-out faces? -->

### 6.2 Hypotheses & pre-registered metrics

| Metric | Definition | Success threshold (if any) |
|--------|------------|----------------------------|
| MAE vs human /10 | | |
| Spearman ρ | | |
| Pairwise accuracy vs human | | |
| Other | | |

### 6.3 Evaluation protocol

<!-- Held-out sample size, rater instructions, blind procedure. -->

**Protocol A — off-cohort placement validation (built 2026-08-04, unrun).**
`scripts/validate_placement.py`; full rationale in
[`production-scoring-pipeline.md` §7](./production-scoring-pipeline.md). This is the held-out *face*
test §6.3 has been asking for since it was written, and it is stronger than the version originally
sketched here because the faces come from a **different source entirely** — not a held-out slice of
the same 3,000, so no shared camera, lighting, demographic draw or VLM label.

| | |
|---|---|
| Sample | 50–100 photos from outside faceiq-labs, one face each, same gender as the reference set used |
| Task | ~300 pairwise judgements (**not** absolute /10 labels — see below), ~20 min |
| Ceiling | ~10% of pairs re-asked with sides flipped; the labeller's self-agreement is the denominator |
| Optional | hand /10 **ranges** per face, for the calibration test only |
| Reported | ordering accuracy as a share of ceiling; accuracy within vs beyond the 1.33 resolvable gap; predicted vs observed agreement; Kendall τ vs a BT fit on the labeller's own judgements; scale **shift** vs residual **spread**; band coverage |

**Why pairwise and not absolute /10.** Absolute labels reintroduce the subjectivity the pairwise
programme exists to remove: two labellers' "6 to 7" differ by an unmeasurable constant, so an
absolute-label test cannot separate "the model is wrong" from "we disagree about what a 7 means".
Pairwise judgements need no shared scale. Ranges are still collected, but they answer a different
question — whether the *anchor ladder* is offset, which is free to fix — and the report deliberately
splits median error into a **shift** (re-anchor) and a **spread** (retrain), because a single combined
number is how a month gets spent on the model to fix arithmetic.

**Why the repeat pairs are not optional.** Without them the natural denominator is 100%, which is the
same mistake that made the panel look like it was failing until the leave-one-out ceiling came in at
74.9% (§5.8). The harness refuses to interpret a share-of-ceiling above 105%, reporting instead that
the labeller's own noise is what is being measured.

**Stated limitation.** The rater is one person, so this measures agreement with one consistent taste,
not with the population — that is what the panel measures. A pass means the system generalises to
unseen faces from an unseen source; it does not upgrade to a population claim.

### 6.4 Results

<!-- Main outcome tables and figures. -->

**Partial result available now — the ranking half of the claim is measured (§5.5).** The panel
studies already deliver a held-out human test: on 1,200 pairs whose votes entered no fit, accuracy
against real human choice rose 51.47% → 54.73% → **57.91%** as each study's votes were added, and
the ranking of record (`bt-refit-v4-panel`) passes every §5.1 gate. Human ceiling on those pairs is
~59%, so the ranking is close to what a single rater achieves.

**The model half now has a partial result too (§5.3.1, 2026-08-01).** The comparator trained on the
panel's labels agrees with human votes **56.4%** of the time against a 59.1% ceiling, beating the
Gemini labels it replaced by +2.00 pts (paired 95% CI [+0.60, +3.41]) and its own exact control by
+2.13 pts. Before the fix it scored 54.3%, indistinguishable from those labels.

**Two things still block a primary result here.** The winning arm runs at `val_fraction: 0.5`, so its
absolute accuracy is a controlled comparison rather than a shippable number — refit at 0.2. And the
panel pairs are a held-out *pair* test, not the held-out *face* test §6.3 specifies. Sequence: refit
v13's label recipe at 0.2, run the §6.3 face-level protocol, then promote those numbers here.

**Both numbers above are worst cases, and §5.8 now gives the population figures.** On a uniform random
draw of 2,500 pairs (run 4) that entered no fit, agreement with the human **majority** is **81.5%** for the
ranking and **79.9%** for the raw Gemini label, against a **74.9%** individual-human ceiling on the same
metric — the ranking beats the average person at naming the crowd's choice on a typical pair. The
comparator, scored **vote-level** on the leak-free subset of the same pairs, reaches **67.67%** against
**69.50%** for the ranking and a **69.90%** ceiling *on that metric*, i.e. it is 1.8 points *behind* the
ranking (paired 95% CI [−3.38, −0.31]). **Whatever goes in §6.4 must state both the pair distribution and
the metric**: selection moves a figure ~29 points and majority-versus-vote level moves it ~10 more.

Note for the write-up: the panel pairs are a *held-out pair* test, not the held-out *face* test §6.3
specifies. Both are needed — the pair test says "does it predict this choice", the face test says
"does it generalise to people it has never seen".

### 6.5 Discussion

<!-- Interpretation, limitations, comparison to Labs formula. -->

### 6.6 Conclusion

<!-- Is the model accurate enough? Was the VLM-based GT approach justified? -->

---

## 7. Phase 5 — Deployment & monitoring (optional)

*Post-launch; not required for initial paper.*

#### Notes

<!-- Production rollout date, drift monitoring, confidence band calibration. -->

---

## 8. Cross-cutting log

### 8.1 Decision log

| Date | Decision | Rationale | Alternatives considered |
|------|----------|-----------|-------------------------|
| 2026-06-29 | Single-model Gemini 2.5 Flash for scale; no 3-model panel | Panel 80.5% vs 2.5 Flash 83–86% on audit; 3× cost | Panel for confidence routing (deferred) |
| 2026-06-29 | Lock prompt `pairwise-v2` (Holistic) for Phase 0 gate | 87.2% on audit run; v3 Minimal wins pool runs only (~4–9 pp) — different pairs, small N; no v3 beat v2 by ≥2 pp on audit | v3 Minimal (cheaper, wins hard pool pairs); v3 Pillar (within 1.3 pp on audit) |
| 2026-06-30 | **GT scale prompt → `pairwise-v3-gt`** | 1k dry rationale audit: v2 over-weighted skin/expression/grooming; structural tiers added | Stay on v2 for full 52.5k (rejected) |
| 2026-06-30 | **Go full 52.5k labeling** | 100-pair human audit on dry run: ~8 overrides (~92% agree); batch infra validated (998/1000) | Re-draw 3k sample (rejected — handful gender mislabels OK) |
| 2026-07-04 | **Go BT export** after full-run audit | 750/1k sample at **84.9%**; 86 excluded; 52,414 eligible | Re-audit to 1,000 (not required) |
| 2026-07-04 | **BT refit in Python ML repo** (not admin UI first) | Faster path; admin §3.3 tables deferred | Inline BT in Next.js |
| 2026-07-2x | **Exclude 133 faces on VLM photo QC** → `bt-refit-v2-qc` | AI-looking, mislabelled gender, unusable crops contaminate both ranking and training | Keep and down-weight (rejected — no clean weight) |
| 2026-07-30 | **Buy human pairwise labels from Prolific** (§5.5) | Our own audit cannot answer "do ordinary people agree"; product claim is about people, not our labels | Expand internal audit (rejected — same two annotators) |
| 2026-07-31 | **Incomplete block design**, ~100 pairs/rater, not a fixed panel of 12 | Population generalisability + rater-bias audit; avoids fatigue | Fixed panel rating all pairs (rejected) |
| 2026-07-31 | **Joint BT with per-source discrimination**; VLM labels dropped on panel pairs | β_human 0.37–0.42 ⇒ humans are 2.4–2.7× less decisive than the VLM at equal θ gap; naive pooling over-weights them | Pool as equal observations (rejected — mis-calibrates) |
| 2026-07-31 | **Score studies on vote share, not majority** | Majority near chance on close pairs while shares replicate at 0.655; the signal is the margin | Majority label (rejected — discards the margin) |
| 2026-08-01 | **Keep buying hard pairs; stop condition < 1 pt / $500** | Run 3 marginal +3.18 pts out-of-sample at $404/pt, 5/5 seeds, dose curve accelerating | Stop and train only (deferred, not rejected) |
| 2026-08-01 | **Gate the next study on the trainer consuming the last one** | Comparator measured at 54.3% vs humans ≈ its 54.4% VLM labels; panel votes never reached training | Buy run 4 now (rejected — money the model ignores) |
| 2026-08-01 | **Gate cleared — run 4 is unblocked** | Retrained comparator hits 56.4% vs humans, +2.00 pts over its Gemini labels (paired CI [+0.60, +3.41]), closing 43% of the gap to ceiling | Keep the gate closed (rejected — the model demonstrably learns from these labels) |
| 2026-08-01 | **Use the hard-majority panel target, not the vote share** | Arms not separable (+0.59 pts, CI [−0.28, +1.47]); prefer the simpler target absent evidence | Soft float target (deferred — plausible but unproven) |
| 2026-08-01 | **Soft targets on the train split only**; val keeps export labels | Rewriting val labels moves the yardstick and the model in one experiment | Panel labels in val too (rejected — uninterpretable) |
| 2026-08-01 | **Defer per-audience (viewer-cohort) labels** | Effect real but small (+1.0 to +2.4 pts) and confounded with rater consistency; small cells need 8–16× recruiting Prolific cannot target | Build per-cohort labels now (rejected — premature) |
| 2026-08-01 | **Ship no composite; the comparator alone is the rating** | 6-way equal blend −1.99% vs comparator (CI [−3.33, −0.67]); narrower blends indistinguishable. Components' errors are correlated — comparator and BT were both fit on VLM labels | Percentile-average of Labs + comparator + BT (rejected — measured, not better); geometric mean (rejected — undefined on signed θ) |
| 2026-08-01 | **Quote a ~1-point band, not a two-decimal /10** | Median per-face 95% interval is 0.91 /10 points; faces closer than ~17 percentile points are not ordered by the data | Publish the point score (rejected — implies precision we do not have) |
| 2026-08-01 | **Block a published "calibrated /10" on a held-out calibration test** | `bt-refit-v4-panel` absorbed all 7,780 panel pairs, so its excellent reliability is measured on its own training data; the honest VLM-only ranking is overconfident by up to 31.7 pts | Publish v4's calibration curve (rejected — circular) |
| 2026-08-01 | **Gate future training runs on margin recovery, not binary accuracy** (§5.7) | Comparator is at ~96% of ceiling on picking winners but 38.9% on predicting vote share; binary accuracy has no room left to show progress in | Keep quoting pairwise accuracy as the headline (rejected — saturated) |
| 2026-08-01 | **Do not ask the VLM for graded labels or rating bands** (§5.7) | Its confidence is flat against crowd decisiveness (0.213/0.203/0.185) and 66.5% of the closest band is "high" confidence — it cannot tell close pairs from far ones, so a graded output inherits the same blind spot | Re-label with a band/margin prompt (rejected — measured, no discrimination to grade with) |
| 2026-08-01 | **Buy human votes inside a 20-point percentile gap, in tranches, after a random-pair study** (§5.7) | Headroom over the free VLM label is +3.5 to +6.2 pts at 0–20 with **all four 95% intervals clear of zero**; at 20–45 (+0.1 [−2.5, +2.5]) and 45+ (−1.1 [−2.9, +0.6]) it is *unresolved* on 389 and 262 pairs, and capped at +2.5/+0.6 pt at the two priciest bands. Buy zone $7,952; the rest is $15,328 of unproven value | Panel-label the whole export (rejected — $23.3k, most of it duplicating free labels); commit the full $7,952 up front (rejected — a ~$400 uniform study resolves the open bands first) |
| 2026-08-01 | **Cut bands/strata only on a ranking that has not seen the votes** (§5.7) | The same headroom table computed on `bt-refit-v4-panel` inverts (−5.7 pt at 0–2, +13.6 pt at 20–45) because panel-overruled pairs migrate into wider bands; it would have directed $8.8k at the worst band | Use the newest refit for band cuts (rejected — circular) |
| 2026-08-01 | **Landmark work is an exporter change, not an extraction project** (§5.7) | `faceiq-labs` already stores `Face.frontLandmarks`/`sideLandmarks`/`mediapipeLandmarks` on every upload via the existing SageMaker model; only `export-gt-run.ts`'s `select` omits them | Commission external landmark extraction (rejected — already in production) |
| 2026-08-04 | **Stop buying human labels above a 20-point percentile gap — permanently** (§5.8) | Headroom on an unbiased uniform draw is −0.9 [−2.1, +0.4] at 20–45 and **−1.3 [−2.2, −0.5]** at 45–100, the second interval entirely below zero: a purchased vote there is *worse* than the free label. Confirmed independently by `panel_run_delta` (run 4 = +0.09 pts, $11,140/pt) and by v5's held-out gains (+0.0 at 45–100). **$14,367 cancelled** | Keep the bands "unresolved" and buy a probe tranche (rejected — three methods agree); buy the whole export (rejected long since) |
| 2026-08-04 | **`bt-refit-v5-panel` is the ranking of record** | Pools all three panel runs (10,280 pairs, 95,245 votes, 963 raters); all §5.1 gates pass, ρ_stab 0.9924/0.9914; ρ 0.9667 vs v2-qc with 599 faces moving > 0.5 /10 | Stay on v4 (rejected — leaves 29k votes unused) |
| 2026-08-04 | **The next dollar goes to training, not labels** (§5.8) | On a population sample the comparator is **1.8–3.1 pts behind** the VLM-only ranking (paired, all intervals below zero), losing the 10–45 band by 3–7 pts. The ordering is already in the labels we own and the network is not extracting it, so more ground truth cannot fix it. **Confirmed the same day by `train-v16`**, which added run 4's wide-band votes (train-split coverage 14.7% → 19.4%) and left the 10–20 band unmoved | Buy the $7,463 buy-zone tranche now (deferred — it would improve a ranking the product does not ship) |
| 2026-08-04 | **Stop asking the comparator to reproduce the global ordering; place new faces against the ranking instead** (§5.8) | Five arms — more labels, soft vs hard targets, fixed checkpoint selection, a variance head, and 4× resolution — all tie at 1.8–2.1 pts behind BT, and the 10–20 band never moves off ~55% against BT's 63.2%. `train-v10`, which saw no panel labels at all, loses the same bands. BT solves a global system over 47,914 comparisons; the comparator sees only local pairs. It beats BT *under* a 10-pt gap, so use it for what it is good at and take the global ordering from BT via reference-set placement (~200 refs, measured) | Keep tuning the comparator (rejected — five arms, no movement); ship the raw scalar (rejected — unanchored by construction) |
| 2026-08-04 | **Resolution is ruled out** (§5.8) | Matched resnet50 pair varying only `image_size`: 64.11% at 112 px vs 64.27% at 224 px, a tie (−0.15, CI [−2.07, +1.78]), and 224 px is *worse* on the 10–20 band. Source photos are 1024×1024 so the hypothesis was live, and it is now closed. Note `arcface_r50` physically cannot take 224 px | Do the head surgery on ArcFace to test 224 there (rejected — the clean resnet50 pair already answers it) |
| 2026-08-04 | **The variance head does not give a per-photo σ** (§5.8) | σ varies (CV 0.22) but correlates +0.52 with the score and its residual is flat across every photo-quality flag (−0.027 to +0.059 against a σ sd of 0.169). It learned that the thin top of the ranking is ill-determined, not that a photo is bad. Multi-photo band narrowing therefore has no σ to work with yet | Ship σ as a photo-quality band (rejected — measured, it is not measuring that); train with quality augmentation (deferred — plausible, untested) |
| 2026-08-04 | **Stop selecting checkpoints on `val_accuracy`** (§5.8) | It is scored against the export's Gemini labels, which §5.3.1 established cannot see a panel-driven improvement; v16's top four epochs sit within 0.6 points of each other on it, so `best.pt` is near-arbitrary with respect to the metric that matters. All arms share the flaw, so past comparisons stay fair | Keep the current selection (rejected — free to fix, and it may be costing every arm real accuracy) |
| 2026-08-04 | **Revert to the soft (vote-share) panel target** (§5.8) | Reverses 2026-08-01, which was decided on hard pairs only. On population pairs v12-soft − v10-control = **+1.20 [+0.04, +2.35]** (separable) while v13-hard − control = −0.46 [−1.50, +0.58] (a tie). Rounding a decisive vote share to 0/1 destroys the margin information | Keep `panel_hard: true` (rejected — measured worse where it matters) |
| 2026-08-04 | **Every accuracy figure must state its pair distribution *and* which of the two accuracy metrics it is** (§5.8) | Pair selection alone moves the same ranking ~29 points (52.9% on near-ties, 81.7% on a uniform draw), and majority-level versus vote-level moves it ~10 more (81.7% vs 70.6% on the same pairs). An earlier draft mixed the two and invented a 14-point gap between the ranking and the comparator | Keep quoting the near-tie number alone (rejected — understates by ~29 pts); quote population alone (rejected — hides research progress); pick one metric and drop the other (rejected — they answer different questions, and the product needs both) |
| 2026-08-04 | **The percentile-band structure is validated, not circular** (§5.8) | Bands cut on the VLM-only ranking predict human agreement monotonically across all six bands (53.2% → 83.6%) on votes that ranking never saw, `spearman(gap, decisiveness) = +0.408`. A circular partition cannot produce a monotone out-of-sample curve | Abandon percentile-gap targeting as self-referential (rejected — measured, it holds) |
| 2026-08-04 | **Weight the gold screen by draw type** (§5.8) | Gold-failing raters trail by 7.8 pts on run 4's wide draw versus 2.4 in run 2 and 2.2 in run 3, because golds *are* wide pairs. The screen measures task quality on wide draws and close to noise on near-tie draws | One fixed gold threshold for every run (rejected — punishes honest raters on hard draws) |

### 8.2 Reproducibility manifest

| Artifact | Location / hash | Version |
|----------|-----------------|---------|
| Parent pool export | Cohort `cmqx9npqe00007hdp54e00vem` | 2026-06-28 |
| Sample-set schema + UI | Migration `20260629180000_vlm_pilot_sample_set`; `/research/pairwise/cohorts/[cohortId]/sample` | 2026-06-29 |
| Approved sample | `cmqz06xr20001bydxaazuojp8` · seed `gt-draw-v1` · **approved** 2026-06-30 | 2026-06-30 |
| Audit run (80 pairs) | `cmqvmn3z800a46mdv66rbvclh` | |
| Phase 0 prompt | `pairwise-v2` (Holistic) | locked 2026-06-29 |
| **GT scale prompt** | **`pairwise-v3-gt`** | locked 2026-06-30 |
| VLM model (scale) | `gemini-2.5-flash` | locked 2026-06-29 |
| GT dry run | `cmr00irf500018ldzuqsavzlh` · `gt-label-dry-v1` · 1,000 pairs | 2026-06-30 |
| GT dry VLM batch | `cmr01nepc0001atdkc7a3ery5` · 998/1000 v2 · ~$1.82 | 2026-06-30 |
| GT full run | `cmr1mr0m7000196d57zi3vcgn` · `gt-label-full-v1` · 52,500 pairs · **vlm_complete** 2026-07-03 · ~$123.54 | 2026-07-03 |
| GT export exclude | Migration `20260703100000_vlm_pilot_gt_export_exclude` · 86 rows excluded · **52,414** eligible | 2026-07-04 |
| Human audit (full) | 750/1,000 seeded sample · **84.9%** accuracy · 121 overrides · run-wide 1,032 human labels | 2026-07-04 |
| Pair queue (full) | `bt-stratified-v2` · k=35 · 52,500 · run above | 2026-07-03 |
| GT export (ML handoff) | `faceiq-preference-ml/data/exports/cmr1mr0m7000196d57zi3vcgn` · 52,414 rows · 6 shards · 3,000 images · sha256 in manifest | 2026-07-04 |
| BT refit v1 | `faceiq-preference-ml/artifacts/bt-refit-v1` · ρ_stab 0.9925 both genders · vs Labs 0.745 F / 0.752 M · gates passed | 2026-07-04 |
| Face photo QC | `artifacts/face-qc-v1/{exclude-faces.csv,gender-fixes.csv}` · 133 faces excluded | 2026-07-2x |
| Rating app | `faceiq-rating` on Vercel · run separation via `CURRENT_STUDY_ID` | 2026-07-30 |
| Panel run 2 | study `6a6ba4da4825f473a8f65364` · 372 raters · `artifacts/panel-run-v1/` | 2026-07-31 |
| Panel run 3 | study `6a6d1e7e3999a35d0dc956b9` · 301 raters · `artifacts/panel-run-v3/` | 2026-08-01 |
| BT refit v4 (superseded by v5) | `artifacts/bt-refit-v4-panel` · β_human 0.417 F / 0.368 M · ρ_stab 0.9931 / 0.9924 · vs Labs 0.752 / 0.766 · gates passed | 2026-08-01 |
| Run 3 marginal value | `artifacts/panel-run-delta-v3/run-delta.json` · +3.18 pts held-out, 5/5 seeds, $404/pt | 2026-08-01 |
| Comparator vs humans (pre-fix) | `artifacts/train-v{8,10}-*/panel-eval.json` · 54.3% vs 59.1% ceiling | 2026-08-01 |
| **Comparator trained on panel labels** | `artifacts/train-v1{2,3}-panel-*/panel-eval.json` · v13 56.4%, v12 55.8%, control 54.3% · per-pair dumps in `panel-pairs.csv` | 2026-08-01 |
| Paired arm comparison | `artifacts/panel-arms-compare.json` · `scripts/compare_panel_evals.py` · 10k pair-clustered resamples | 2026-08-01 |
| Rater demographics | `artifacts/panel-demographics/summary.json` · 667/667 matched · raw CSVs gitignored | 2026-08-01 |
| **Per-face BT uncertainty** | `artifacts/bt-refit-v4-panel/uncertainty.{csv,json}` · `scripts/bt_uncertainty.py` · 200 resample + 200 relabel replicates/gender, seed 11 · median 16.8 pct-pt / 0.91 /10 interval | 2026-08-01 |
| **/10 calibration** | `artifacts/bt-refit-v4-panel/calibration.json` · `scripts/rating_calibration.py` · 7,780 panel pairs · v2-qc overconfident −31.7 pts; v4-panel circular | 2026-08-01 |
| **Composite scoring** | `artifacts/composite-v1/{report,reference-set-v13,reference-set-v14,arms-v15}.json` · `src/faceiq_pref/composite.py` · 1,928 leak-free pairs, 2k paired resamples | 2026-08-01 |
| **Labs + comparator blend** | `artifacts/composite-v1/labs-plus-v13.json` + `labs-plus-v13-scores.csv` · 56.91% (+0.49%, CI [−0.36, +1.32]) · per-face /10 for 2,866 faces | 2026-08-01 |
| Ship refit + weighted arm | `artifacts/train-v14-panel-ship/` (val_fraction 0.2) · `artifacts/train-v15-panel-weighted/` (panel_weight 3.0, 56.55%) | 2026-08-01 |
| External cross-checks (refreshed vs v4) | `artifacts/external-{scut,mebeauty}/eval.json` · SCUT τ 0.285, MEBeauty τ 0.208 on 2,866 faces | 2026-08-01 |
| Panel run 4 (uniform random) | study `6a71110e4d5c6eba96c17d21` · 303 raters · 31,237 judgments · $968.58 · `artifacts/panel-run-v4/` | 2026-08-04 |
| **Honest /10 calibration (non-circular)** | `artifacts/panel-run-v4/calibration.json` · `bt-refit-v2-qc` × 2,500 unfitted uniform pairs · 0.25 pt = 53.0%, 2+ pt = 82.7% · σ(Δθ) overconfident by 26 pts | 2026-08-04 |
| Gap–agreement curve (ordering validated) | `artifacts/panel-run-v4/gap-curve.json` · `scripts/gap_agreement_curve.py` · monotone 53.2% → 83.6%, ρ_s(gap, decisiveness) = +0.408 | 2026-08-04 |
| Headroom v2 (spend question closed) | `artifacts/label-information-v2/{report.json,report-run4only.json}` · 10,280 pairs / 95,245 votes · 45–100 headroom −1.3 [−2.2, −0.5] | 2026-08-04 |
| Run 4 marginal value | `artifacts/panel-run-delta-v4/run-delta.json` · +0.09 ±0.36 pts, 2/5 seeds, $11,140/pt, flat dose curve | 2026-08-04 |
| **BT refit v5 (ranking of record)** | `artifacts/bt-refit-v5-panel` · β_human 0.338 F / 0.286 M · ρ_stab 0.9924 / 0.9914 · ρ 0.9667 vs v2-qc · gates passed | 2026-08-04 |
| Comparator population accuracy (vote-level) | `artifacts/train-v1{0,2,3,4}-*/panel-eval-run4.json` · v12 67.67%, v13 66.93%, v10 66.47% vs BT 69.50% and a 69.90% ceiling · paired in `artifacts/panel-arms-compare-run4.json` | 2026-08-04 |
| **Accuracy 2 × 2 (metric × pair distribution)** | `artifacts/panel-run-v4/accuracy-matrix.json` · `scripts/accuracy_matrix.py` · the fix for a mixed-metric error in an earlier draft of §5.8 | 2026-08-04 |
| Band widths from the calibration curve | `artifacts/panel-run-v4/band-calibration.json` · `scripts/band_calibration.py` · T = 1.920 on the /10 scale | 2026-08-04 |
| Training split | | |
| Model checkpoint | | |
| Evaluation set | | |

### 8.3 Open questions & risks

<!-- Photo quality confounds, gender policy, position bias, tail compression, etc. -->

Raised by §5.6 and still open:

- **Test–retest stability is unmeasured, and cannot be measured from this export.** Same person, different
  photo, should get the same score; a user who re-uploads and moves 1.5 points stops believing the number.
  Checked 2026-08-01: all 3,000 rows are the `_front` view of 3,000 **distinct** `sourceFaceId`s, so there
  is no second photo of anyone here. Measuring it needs a small extra export of other views
  (`<sourceFaceId>_<view>.webp` already exist in Labs blob storage) — **no new labels and no Prolific
  spend**, just an export plus a scoring run. **Now the single highest-value open item**, since §5.8 closed
  the labelling question and this is the only untested axis that can invalidate the product.
- ~~**Calibration cannot be honestly tested**~~ ✅ **closed by §5.8 (2026-08-04).** Measured on run 4's
  2,500 unfitted uniform pairs: monotone from 53.0% at a 0.25 /10 gap to 82.7% above 2 points, but `σ(Δθ)`
  is overconfident by up to 26 points, so a published probability needs temperature scaling. **The property
  is now spent** — run 4's pairs are inside `bt-refit-v5-panel`, so re-testing a future ranking needs fresh
  random pairs (~$400 buys enough for the curve alone).
- **The comparator is behind the ranking on typical pairs** (§5.8): 67.7% vs 69.50% against a 69.90%
  ceiling, paired deficit −1.84 [−3.38, −0.31], concentrated in the 10–45 percentile band. Since the
  ranking gets it right from the same labels, this is an extraction failure — capacity, schedule, loss, or
  resolution — not a ground-truth shortage. **The top open ML question.**
- **44 faces are not identified from above** (undefeated or once-beaten). Their θ is set by the α = 0.01
  regulariser, not by evidence, and no bootstrap reveals this — only the record does.
- **The top of the ranking is thin.** Only 30 faces score above 8.0/10, and the ranking is fit per gender,
  so elite-vs-elite ordering rests on very few informative comparisons.
- ~~**Panel pairs are enriched for close pairs**~~ ✅ **closed by §5.8.** Population accuracy is now
  measured on a uniform draw. Majority-level: ranking 81.7%, Gemini 80.0%, one-rater ceiling 73.2%.
  Vote-level: ranking 69.5%, Gemini 68.8%, comparator 67.7%, ceiling 69.9%. Every earlier figure was a
  worst case by ~29 points. **Always state the pair distribution and the metric** — the 2 × 2 is in
  `artifacts/panel-run-v4/accuracy-matrix.json`.
- **The cohort is one photo per person, front view only, from the Labs upload funnel** — so it inherits
  that funnel's selection (who uploads, what lighting, what camera) and cannot express within-person
  variation. This is the deepest untested assumption in the programme and no amount of labelling touches
  it. See `programme-direction-review.md` §3.

### 8.4 Scratch / meeting notes

<!-- Promote stable findings to formal sections above -->

---

## 9. Documentation timing checklist

| Gate | Record now | Can defer |
|------|------------|-----------|
| **Pilot complete** | §2.1 results + decision | — |
| **Prompt chosen** | §2.2 + lock prompt hash | — |
| **Consensus validated** | §2.3 | — |
| **Pool imported** | §3.1 + quality notes | — |
| **Sample curation approved** | §3.2 + §3.3 | — (UI + schema shipped 2026-06-29) |
| **Pair queue generated** | §4.1 dry + full ✓ | — |
| **Labeling started** | §4.2 full ✓ | — |
| **Labeling complete** | §4.2 final + §4.3 ✓ | — |
| **BT export ready** | §5.1 + §5.2 ✓ (2026-07-04) | — |
| **Model trained** | §5.3 summary | Failed runs unless informative ✓ (v1–v10, 2026-07-12) |
| **Face photo QC** | §5.0 ✓ (2026-07-27) | — |
| **Human panel studies** | §5.5 ✓ (runs 1–3, 2026-08-01) | — |
| **Comparator trained on human targets** | §5.3 human-grounded eval ✓ (v12–v15, 2026-08-01) | — |
| **Rating validation (uncertainty, calibration, composites)** | §5.6 ✓ (2026-08-01) | Landmark/EBM features (deferred) |
| **Label information / spend pricing** | §5.7 ✓ (2026-08-01) | — |
| **Population accuracy + honest calibration + spend verdict** | §5.8 ✓ (run 4, 2026-08-04) | — |
| **Test–retest stability** | §5.6 production-metrics table · **← now**, blocked on a second-view export | — |
| **Comparator closes the 10–45 band gap vs the ranking** | §5.3.2 · **← now**, `train-v16` on the GPU box | — |
| **Deploy smoke test** | §5.4 | — |
| **Primary evaluation** | **§6 in full** (§6.4 has a partial) | — |
| **Publication draft** | §1.2 updated with §6 answer | §8 scratch cleanup |

---

## 10. Sub-experiment inventory

| # | Sub-study | Phase | Type | In paper as |
|---|-----------|-------|------|-------------|
| 1 | VLM pilot | 0 | Validation | Methods — labeler selection |
| 2 | Prompt A/B on audit set | 0 | Validation | Methods — prompt |
| 3 | 3-model consensus vs single model | 0 | Validation | Methods — consensus |
| 4 | Parent pool export QC | 1 | QC | Methods — dataset |
| 5 | Stratified 3k draw + audit | 1 | QC | Methods — sampling |
| 6 | Pre-label cohort audit | 1 | QC | Methods — sampling |
| 7 | Pair queue design verification | 2 | QC | Methods — graph design |
| 8 | Main labeling (~52.5k pairs) | 2 | Data collection | Methods — GT construction |
| 9 | Human review / override analysis | 2 | Validation | Methods — label quality |
| 10 | BT refit + stability | 3 | Validation | Methods — ranking |
| 11 | BT vs Labs scatter | 3 | Validation | Methods — validation |
| 12 | Preference model training | 3 | Model build | Methods — training |
| 13 | Anchor-ladder smoke test | 3 | Validation | Methods — inference |
| 15 | Human panel runs 1–3 (Prolific, 667 raters) | 3 | Data collection + validation | Methods — human GT; Results — ranking validation |
| 16 | Joint BT with per-source discrimination | 3 | Model build | Methods — ranking |
| 17 | Marginal value of a labeling round (held-out pairs) | 3 | Validation | Results — cost/benefit |
| 18 | Comparator vs human votes (leak-free) | 3 | Validation | Results — model quality |
| 19 | Viewer-cohort effect (in-group vs out-group) | 3 | Validation | Discussion — personalisation |
| 20 | Per-face ranking uncertainty (bootstrap + relabel) | 3 | Validation | Results — ranking precision; Limitations |
| 21 | /10 score calibration vs panel votes | 3 | Validation | Results — calibration |
| 22 | Composite (blended) ratings vs single model | 3 | Validation | Results — negative result |
| 23 | Margin recovery vs vote-share reliability ceiling | 3 | Validation | Results — model quality; Methods — metric choice |
| 24 | VLM confidence as a pair router (negative result) | 3 | Validation | Methods — label routing; Results |
| 25 | Uniform random-pair study (run 4) — population accuracy | 3 | Data collection + validation | **Results — headline accuracy**; Methods — sampling |
| 26 | Out-of-sample /10 calibration on unfitted pairs | 3 | Validation | Results — calibration; Limitations |
| 27 | Percentile gap → crowd agreement, unselected sample | 3 | Validation | Results — ranking validity (answers circularity) |
| 28 | Where a human label beats a machine label, by band (spend curve) | 3 | Validation | Discussion — cost/benefit; Methods — label economics |
| 29 | Comparator vs ranking on population pairs (negative result) | 3 | Validation | Results — model quality; Discussion — what to fix next |
| **14** | **Held-out human evaluation** | **4** | **Primary experiment** | **Results — main claim** |

---

## Appendix: Sub-study template

Copy when adding an ad-hoc entry:

```markdown
### [Title] ([status])

#### Overview
[One paragraph: what this step was.]

#### Research question
[What we needed to learn.]

#### Methodology
[How we ran it — prose, not a table.]

#### Success criteria
[Pass/fail bar before results.]

#### Results
[What we observed. Small table only if numeric.]

#### Decision
[Go / no-go / next step.]

#### Artifacts
[IDs, paths, hashes.]

#### Notes
[Limitations, surprises, open questions.]
```
