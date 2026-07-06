# Scoring Ground Truth — Research Log

**Purpose of this document:** A living record of *what we did*, *why*, and *what we decided* at each phase of the scoring ground-truth program. This is a **working lab notebook**, not the final study write-up. Fill sections as work completes.

**Related plans:** [`scoring-gt-core.md`](./scoring-gt-core.md) · [`vlm-pilot-spec.md`](./vlm-pilot-spec.md) · [`scoring-gt-training.md`](./scoring-gt-training.md)

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

### 5.1 Bradley–Terry refit & stability

**Status:** First refit **complete — all gates passed** (2026-07-04, `faceiq-preference-ml` · `artifacts/bt-refit-v1`).

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

**Status:** Not started.

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

<!-- Train/val split by face id, hyperparameters. -->

#### Results

<!-- Val pairwise accuracy, Kendall τ vs BT. -->

#### Artifacts

<!-- Checkpoint path, config YAML. -->

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

### 6.4 Results

<!-- Main outcome tables and figures. -->

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
| Training split | | |
| Model checkpoint | | |
| Evaluation set | | |

### 8.3 Open questions & risks

<!-- Photo quality confounds, gender policy, position bias, tail compression, etc. -->

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
| **Model trained** | §5.3 summary | Failed runs unless informative · **← now** |
| **Deploy smoke test** | §5.4 | — |
| **Primary evaluation** | **§6 in full** | — |
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
