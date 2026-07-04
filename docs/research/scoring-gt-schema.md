# Scoring GT — Database Schema Map

**Purpose:** Which Prisma models belong to the scoring GT / pairwise experiment — and what to add before scale. Everything else in `schema.prisma` is **unrelated** (prod `Face`, billing, community, etc.).

**Research UI:** `/research/pairwise`  
**Prisma location:** bottom of `schema.prisma` — `// VLM Pilot — pairwise attractiveness research`

---

## 1. Models in scope (today)

All tables are prefixed `vlm_pilot_*` in Postgres. **Eleven models** for this experiment (as of 2026-06-29).

| Prisma model | DB table | What it stores |
|--------------|----------|----------------|
| **`VlmPilotCohort`** | `vlm_pilot_cohorts` | Imported pool (name, face count, import time, notes) |
| **`VlmPilotFace`** | `vlm_pilot_faces` | One row per face: photo URL, gender, race, full `exportPayload` JSON; **`decileBin`**, **`labsOverallScore`** (pool stratification) |
| **`VlmPilotRun`** | `vlm_pilot_runs` | Labeling run on a cohort (seed, pair count, status); optional **`sampleSetId`** for scale runs |
| **`VlmPilotComparison`** | `vlm_pilot_comparisons` | One A/B pair: faces, presentation, **human label** |
| **`VlmPilotVlmBatch`** | `vlm_pilot_vlm_batches` | One VLM batch job: models, prompts, **batch mode**, frozen config |
| **`VlmPilotVlmResult`** | `vlm_pilot_vlm_results` | **Per model, per pair:** winner, confidence, rationale, raw JSON, tokens, cost, latency |
| **`VlmPilotComparisonConsensus`** | `vlm_pilot_comparison_consensus` | **Per pair × batch × prompt:** aggregated panel vote + audit JSON |
| **`VlmPilotMetricSnapshot`** | `vlm_pilot_metric_snapshots` | **Frozen experiment summary:** accuracy, cost, splits (for UI revisit + research log) |
| **`VlmPilotSampleSet`** | `vlm_pilot_sample_sets` | Stratified 3k draw from parent pool: seed, targets, status, approve gate |
| **`VlmPilotSampleSetMember`** | `vlm_pilot_sample_set_members` | Face membership in a sample set (decile at inclusion, soft exclude) |
| **`VlmPilotSampleSetEvent`** | `vlm_pilot_sample_set_events` | Curation audit log (draw, remove, fill_gaps, approve) |

### Relationship diagram

```text
VlmPilotCohort
  ├── VlmPilotFace (many)
  │     └── VlmPilotSampleSetMember (many) ← active or excluded sample rows
  ├── VlmPilotSampleSet (many)
  │     ├── VlmPilotSampleSetMember (many)
  │     ├── VlmPilotSampleSetEvent (many)
  │     └── VlmPilotRun (many) ← scale runs reference approved set via sampleSetId
  └── VlmPilotRun (many)
        ├── VlmPilotComparison (many)
        │     ├── faceA, faceB → VlmPilotFace
        │     ├── humanWinner → VlmPilotFace
        │     ├── VlmPilotVlmResult (many) ← one row per model call
        │     └── VlmPilotComparisonConsensus (many) ← one row per batch × prompt
        ├── VlmPilotVlmBatch (many)
        │     ├── VlmPilotVlmResult (many)
        │     ├── VlmPilotComparisonConsensus (many)
        │     └── VlmPilotMetricSnapshot (many)
        └── VlmPilotMetricSnapshot (many)
```

### What is **not** in prod schema

- No `Face`, `User`, or analysis tables are used for GT labeling.
- Research data is isolated — safe to experiment without touching prod face rows.

---

## 1.1 Code constants (not in DB)

Panel and prompt-sweep **defaults** live in code so prompt phase uses a locked panel without re-selecting models in the UI.

| Constant | File | Purpose |
|----------|------|---------|
| `VLM_PAIRWISE_PRODUCTION_PANEL` | `src/lib/vlm-pilot/models.ts` | **Fixed 3-model panel** for prompt-sweep batches (`batchMode: prompt_sweep`) |
| `VLM_PAIRWISE_PANEL_PROMPT` | `src/lib/vlm-pilot/models.ts` | **Fixed prompt** for panel-test batches (`batchMode: panel`) — currently `pairwise-v2` |
| `VLM_PAIRWISE_TEST_MODELS` | `src/lib/vlm-pilot/models.ts` | Models shown in panel-test UI checkboxes |
| Prompt text bodies | `src/lib/vlm-pilot/prompt.ts` | Prompt versions; catalog labels in `prompt-versions.client.ts` |

**After panel validation (§2.3 in research log):** update `VLM_PAIRWISE_PRODUCTION_PANEL` to the chosen model IDs, record decision + `batchId` in [`scoring-gt-research-log.md`](./scoring-gt-research-log.md), then run prompt sweeps.

**After prompt validation (§2.2):** update `VLM_PAIRWISE_PANEL_PROMPT` (and optionally add/remove entries in `prompt.ts` / `prompt-versions.client.ts`).

Each batch still stores the **actual** `modelIds` and `promptVersion(s)` used in `VlmPilotVlmBatch.modelIds`, `promptVersion`, and `configJson` — so historical runs remain auditable even if constants change later.

---

## 1.2 Batch modes (`VlmPilotVlmBatch.batchMode`)

| Value | UI | Behavior |
|-------|-----|----------|
| `panel` | Panel test | User selects models; prompt locked to `VLM_PAIRWISE_PANEL_PROMPT` |
| `prompt_sweep` | Prompt sweep | Models locked to `VLM_PAIRWISE_PRODUCTION_PANEL`; user selects prompt variants |
| `legacy` | (old runs) | Pre–panel/prompt-split batches; Cartesian product of models × prompts |

**`configJson`** (frozen at batch start):

```json
{
  "modelIds": ["gemini-2.5-flash", "..."],
  "promptVersions": ["pairwise-v2"],
  "panelPrompt": "pairwise-v2",
  "fixedPanel": true
}
```

**Migration:** `prisma/migrations/20260629120000_vlm_pilot_panel_consensus/migration.sql`

---

## 1.3 Consensus row (`VlmPilotComparisonConsensus`)

Written when a batch completes (`persistBatchConsensusAndSnapshots` in `src/lib/vlm-pilot/snapshots.ts`). Vote rules: `src/lib/vlm-pilot/consensus.ts` (scoring-gt-core §5.1).

| Field | Type | Notes |
|-------|------|-------|
| `comparisonId` | FK | → `VlmPilotComparison` |
| `batchId` | FK | → `VlmPilotVlmBatch` |
| `promptVersion` | `String` | One consensus row per prompt in the batch |
| `modelIds` | `String[]` | Panel snapshot at compute time |
| `consensusOutcome` | `String` | `A` \| `B` \| `tie` |
| `consensusConfidence` | `String` | `high` \| `medium` \| `low` |
| `consensusWinnerFaceId` | `String?` | FK → `VlmPilotFace` |
| `panelVoteJson` | `Json` | `{ votes, voteConfidences, pattern, votePattern, ... }` |
| `computedAt` | `DateTime` | |

**Unique:** `(comparisonId, batchId, promptVersion)` — supports multiple prompt A/B runs without overwrite.

---

## 1.4 Metric snapshot (`VlmPilotMetricSnapshot`)

Frozen go/no-go row per batch (and per-model drill-down). Powers Results / Run overview revisit and **Copy for research log**.

| Field | Type | Notes |
|-------|------|-------|
| `runId` | FK | |
| `batchId` | FK | |
| `batchMode` | `String` | `panel` \| `prompt_sweep` \| `legacy` |
| `promptVersion` | `String?` | Set per prompt in sweeps |
| `vlmModel` | `String?` | **`null` = panel consensus row**; set for per-model rows |
| `accuracy` | `Float?` | vs human labels (ties excluded) |
| `scoredComparisons` | `Int?` | |
| `correct` | `Int?` | |
| `vlmTieCount` | `Int?` | |
| `unanimousRate` | `Float?` | Panel consensus rows only |
| `splitCount` | `Int?` | Pairs with split votes → review queue signal |
| `totalCostUsd` | `Float` | |
| `payloadJson` | `Json?` | Full metrics blob for audit |
| `createdAt` | `DateTime` | |

**Research log mapping:** paste `runId`, `batchId`, snapshot `id`, accuracy, cost into [`scoring-gt-research-log.md`](./scoring-gt-research-log.md) §2.2 (prompts) or §2.3 (panel) and §8.2 manifest.

---

## 1.5 Sample set curation (implemented)

**Migration:** `prisma/migrations/20260629180000_vlm_pilot_sample_set/migration.sql`  
**UI:** `/research/pairwise/cohorts/[cohortId]/sample`  
**Server lib:** `src/lib/vlm-pilot/sample/` (deciles, draw, histogram, audit, fill-gaps, approve)

Parent pool (~20k) vs active GT cohort (~3k): reproducible stratified draw + manual curation (core §15–§16). One **draft** sample set per cohort; **approve** locks edits.

### `VlmPilotSampleSet`

| Field | Type | Notes |
|-------|------|-------|
| `id` | cuid | |
| `parentCohortId` | FK | → `VlmPilotCohort` |
| `name` | String | e.g. `Large 20k Batch sample` |
| `seed` | String | Reproducible auto-draw (`sha256` sub-seeds per cell) |
| `targetSize` | Int | Default **3000** |
| `drawRulesJson` | Json | Per-decile targets, gender split — see `draw-rules.ts` |
| `status` | String | `draft` \| `approved` \| `archived` |
| `approvedAt` | DateTime? | Lock gate before pair queue generation |
| `approvedByUserId` | String? | User who approved |
| `createdAt` | DateTime | |

**Index:** `(parentCohortId, status)` — find latest draft/approved per cohort.

### `VlmPilotSampleSetMember`

Join table: a face can be in the parent pool without being in the sample. **Default included** after draw; **soft-remove** via `excludedAt` (not deleted).

| Field | Type | Notes |
|-------|------|-------|
| `id` | cuid | |
| `sampleSetId` | FK | |
| `faceId` | FK | → `VlmPilotFace` |
| `decileBin` | Int | D1–D10 at time of inclusion |
| `inclusionSource` | String | `auto_draw` \| `manual_add` \| `gap_fill` |
| `excludedAt` | DateTime? | Set on remove from draft sample |
| `exclusionReason` | String? | `duplicate` \| `synthetic` \| `quality` \| `other` |
| `exclusionNote` | String? | Free text |
| `createdAt` | DateTime | |

**Unique:** `(sampleSetId, faceId)`. **Active member:** `excludedAt IS NULL`.

**Deferred (not in v1 schema):** `addedByUserId`, `manual_swap` inclusion source — manual add-from-pool UI not built yet; use `manual_add` when added.

### `VlmPilotSampleSetEvent`

| Field | Type | Notes |
|-------|------|-------|
| `id` | cuid | |
| `sampleSetId` | FK | |
| `action` | String | `draw` \| `remove` \| `fill_gaps` \| `approve` \| `manual_add` (future) |
| `faceId` | String? | Set for `remove` |
| `payloadJson` | Json | Cell counts, reasons, fill summary |
| `userId` | String? | Approver / actor when available |
| `createdAt` | DateTime | |

### Extended pool / run fields

**`VlmPilotFace`** (computed on first draw if missing):

| Field | Type | Notes |
|-------|------|-------|
| `decileBin` | Int? | D1–D10 within gender (from `overall_score` / `overallScore`) |
| `labsOverallScore` | Float? | Denormalized from `exportPayload` for queries |

**Index:** `(cohortId, gender, decileBin)` — cell draws and histograms.

**`VlmPilotRun`:**

| Field | Type | Notes |
|-------|------|-------|
| `sampleSetId` | String? | FK → `VlmPilotSampleSet`; **future gate:** scale runs must reference an **approved** set |

### API routes (sample curation)

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/api/admin/pairwise/cohorts/[cohortId]/sample` | Latest set + histograms + audit |
| `POST` | `/api/admin/pairwise/cohorts/[cohortId]/sample/draw` | `{ seed, targetSize?, name? }` |
| `GET` | `/api/admin/pairwise/sample-sets/[sampleSetId]/members` | Paginated review grid |
| `GET` | `/api/admin/pairwise/sample-sets/[sampleSetId]/pool-browse` | Pool faces not in sample (future manual add) |
| `PATCH` | `/api/admin/pairwise/sample-sets/[sampleSetId]/members/[faceId]` | `{ action: "exclude", reason, note? }` |
| `POST` | `/api/admin/pairwise/sample-sets/[sampleSetId]/fill-gaps` | Restore under-target cells |
| `POST` | `/api/admin/pairwise/sample-sets/[sampleSetId]/approve` | Lock if audit passes |
| `GET` | `/api/admin/pairwise/sample-sets/[sampleSetId]/events` | Curation audit trail |

**Cohort GET** also returns `activeSampleSet` summary (status, active count).

---

## 2. Phase → schema sufficiency

| Phase | Status | Schema sufficient? |
|-------|--------|-------------------|
| **Panel test** (audit run, multi-model + consensus) | **Now** | **Yes** — §1.1–§1.4 + migration `20260629120000` |
| **Prompt sweep** (fixed panel, vary prompts) | **Now** | **Yes** — same; lock panel in `models.ts` |
| **Stratified 3k sub-sample** from 20–30k pool | **Now** (UI + schema shipped) | **Yes** — §1.5 + migration `20260629180000` |
| **52k labeling + human review** | **Done** | **Yes** — export API + `gtExportExcludedAt`; human labels on comparison rows |
| **Bradley–Terry refit + /10** | **Next (ML repo)** | **No** — optional §3.3 rating tables for admin persist |
| **Export for training** | **Now** | `GET …/export` implemented; optional §3.4 export job table |
| **NN training** | Offline | **Not in this DB** — separate ML repo (see training doc) |

---

## 3. Planned schema additions (before scale)

Implement as Prisma migrations when you start each phase. Order matters.

### 3.1 Final labels on comparison (deferred — scale only)

**When:** Before 52k labeling + BT refit.  
**Why:** BT refit and training export need **one final winner per pair** after human review — distinct from per-batch consensus rows.

Per-batch consensus is already in **`VlmPilotComparisonConsensus`** (§1.3). At scale, add **final** fields on `VlmPilotComparison`:

| Field | Type | Notes |
|-------|------|-------|
| `finalOutcome` | `String?` | After human review / auto-accept |
| `finalWinnerFaceId` | `String?` | FK → `VlmPilotFace` |
| `finalizedAt` | `DateTime?` | When pair locked for BT |
| `reviewReason` | `String?` | `split` \| `low_confidence` \| `audit` \| `override` |
| `humanOverrideAt` | `DateTime?` | If reviewer changed consensus |

**Status flow (extend `status`):**  
`pending_human` → `human_done` → `vlm_done` → `consensus_done` → `in_review` → `final`

Individual model rows stay in **`VlmPilotVlmResult`**. Panel audit stays in **`VlmPilotComparisonConsensus`**.

---

### 3.2 Stratified sub-sample — **implemented** (see §1.5)

**Status:** Shipped 2026-06-29. Migration `20260629180000_vlm_pilot_sample_set`. Full field tables, API routes, and UI path are in **§1.5**.

**Research log:** fill §3.2–§3.3 after QA on parent pool `cmqx9npqe00007hdp54e00vem` (draw → remove → fill gaps → approve).

**Not yet in schema:** `addedByUserId` on members; `manual_swap` / add-from-pool picker (deferred — `pool-browse` API exists for future use).

---

### 3.3 Bradley–Terry ratings — new `VlmPilotBtRefit` + `VlmPilotFaceRating`

**When:** After labeling finalized.  
**Why:** Store authoritative θ and `scoreOutOf10` per refit; support stability checks.

**New model: `VlmPilotBtRefit`**

| Field | Type | Notes |
|-------|------|-------|
| `id` | cuid | |
| `runId` or `sampleSetId` | FK | Which labeling scope |
| `comparisonCount` | Int | Pairs used |
| `refitAt` | DateTime | |
| `configJson` | Json | Calibration curve version, tie handling |
| `stabilitySpearman` | Float? | 80% subsample check |

**New model: `VlmPilotFaceRating`**

| Field | Type | Notes |
|-------|------|-------|
| `id` | cuid | |
| `btRefitId` | FK | |
| `faceId` | FK → `VlmPilotFace` | |
| `btTheta` | Float | Raw BT estimate |
| `scoreOutOf10` | Float | After calibration curve |
| `percentileRank` | Float | |
| `comparisonCount` | Int | Resolved games for this face |
| `isAnchor` | Boolean | Top/bottom anchor ladder candidates |

---

### 3.4 Export jobs (optional)

**When:** Before notebook handoff at scale.

**New model: `VlmPilotExportJob`**

| Field | Type | Notes |
|-------|------|-------|
| `sampleSetId` or `runId` | FK | |
| `exportType` | String | `matchups` \| `faces` \| `ratings` |
| `rowCount` | Int | |
| `manifestHash` | String | SHA-256 of export file |
| `storagePath` | String? | Local path or blob URL (not in git) |
| `exportedAt` | DateTime | |

`VlmPilotMetricSnapshot` is **already implemented** (§1.4).

---

## 4. What stays **outside** the database

| Data | Where |
|------|-------|
| NN training epochs, loss curves, checkpoints | Separate ML repo (`faceiq-preference-ml`) |
| Exported zip / JSON for notebook | Local disk or blob |
| Narrative decisions (“why prompt v4”, “why this panel”) | `scoring-gt-research-log.md` |
| Prompt full text (version controlled) | `src/lib/vlm-pilot/prompt.ts` — log **prompt id + git commit** in §8.2 |
| Locked panel model IDs (after decision) | `src/lib/vlm-pilot/models.ts` — log chosen IDs + validating `batchId` in §2.3 |
| User photos at rest | Blob URLs in `VlmPilotFace.frontPhotoUrl` |

---

## 5. Quick lookup — ignore these in `schema.prisma`

When scanning the 3600+ line schema, **skip everything except** the `Vlm Pilot` block (~lines 3493–3731). Common unrelated models:

- `Face`, `Analysis`, `Harmony*` — production app
- `User`, `Subscription`, `Stripe*` — billing
- `Community*`, `CreatorLeague*` — other features
- `Chat*`, `Conversation*` — AI chat

Search shortcut: `VlmPilot` or `vlm_pilot_`.

---

## 6. Migration checklist

- [x] **Panel + prompt experiments:** `VlmPilotComparisonConsensus`, `VlmPilotMetricSnapshot`, `VlmPilotVlmBatch.batchMode` + `configJson` — migration `20260629120000_vlm_pilot_panel_consensus`
- [x] **3k sample curation:** `VlmPilotSampleSet`, `VlmPilotSampleSetMember`, `VlmPilotSampleSetEvent`; face decile fields; `VlmPilotRun.sampleSetId` — migration `20260629180000_vlm_pilot_sample_set` — UI `/research/pairwise/cohorts/[cohortId]/sample`
- [x] **GT export exclude:** `gtExportExcludedAt`, `gtExportExcludeReason` on `VlmPilotComparison` — migration `20260703100000_vlm_pilot_gt_export_exclude`
- [x] **Export API:** `GET /api/admin/pairwise/runs/[runId]/export` — excludes soft-dropped rows; includes human + VLM fields
- [ ] **Optional before admin BT UI:** §3.1 `finalOutcome` fields (human* used as proxy today)
- [ ] **Optional:** §3.3 ratings tables (persist offline BT results)
- [ ] **Optional:** §3.4 export jobs

Cross-link completed migrations here and in [`scoring-gt-research-log.md`](./scoring-gt-research-log.md) §8.2.

---

## 7. Experiment workflow (panel → prompts)

```text
1. Panel test (UI: batchMode = panel)
   → Results: consensus accuracy, per-model rows, split count
   → Research log §2.3 + copy summary from Run overview

2. Lock panel in code
   → Edit VLM_PAIRWISE_PRODUCTION_PANEL in models.ts
   → (Optional) commit message references batchId from step 1

3. Prompt sweep (UI: batchMode = prompt_sweep)
   → Uses locked panel automatically; toggle prompt variants only
   → Results: By prompt view; research log §2.2

4. Lock prompt in code
   → Edit VLM_PAIRWISE_PANEL_PROMPT (and prompt.ts if adding variants)
   → Proceed to scale labeling (separate schema + infra work)
```

You do **not** need new schema to lock the panel for prompt phase — only the code constant update (step 2). Tell whoever is implementing to update `VLM_PAIRWISE_PRODUCTION_PANEL` with the chosen model IDs and note the decision in the research log.
