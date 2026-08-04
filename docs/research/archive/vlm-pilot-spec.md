# VLM Pilot — Initial Spec

Validate **A)** whether VLMs can reliably pick the more attractive face in A/B, **B)** cost per comparison, **C)** which model (+ prompt) to use for the full batch.

**Status (2026-06):** Pilot **passed**. On 40 human-labeled pairs (`pairwise-v2`), top Flash models reached **~76–86%** accuracy vs labels; cost **~$0.0008–$0.004/pair** (single call). See §14 for results snapshot. Next: **prompt experiments** + **3-model consensus panel** at scale → full GT pipeline in `scoring-gt-core.md`.

**Scope (v1):** Data Lab export → **~52 front faces** → **40 deterministic pairs** → human labels → VLM batch → accuracy + cost.

Not production GT. **Do not commit export zip/folder to the repo** — import via admin UI into isolated DB tables.

**Admin location:** `/research/pairwise` (see §6 routes; legacy alias `/admin/pairwise` in older notes).

---

## 1. Scale & photo count

| Item | Count (current export) |
|------|------------------------|
| Manifest rows | **100** (52 FRONT + 48 SIDE image rows) |
| **Usable faces for A/B** | **52** (one per `analysis_id` with `view=FRONT`) |
| Local image files | **100** `.webp` in `images/` |
| Pairwise matchups | **40** |
| Photos per comparison | **2** (Face A + Face B front images) |
| Your manual labels | **40** |
| VLM API calls (per model) | **40** |
| VLM API calls (3 models) | **120** |

40 pairs fits in 52 faces. Larger exports can import up to ~100 fronts when available.

---

## 2. Flow

```
Import Data Lab zip (~52 front faces)
        │
        ▼
Create pilot run (seed) → generate 40 pairs (deterministic, stored in DB)
        │
        ▼
You label all 40 pairs (human ground truth) — required before VLM batch
        │
        ▼
Run N VLM configs in parallel on same 40 pairs → store winner, rationale, cost
        │
        ▼
Dashboard: accuracy vs your labels, $/pair, pick model + prompt
        │
        ▼
Go / no-go for full ~10k labeling
```

**Order matters:** Human labels first (or at least complete for all 40) so VLM results are always scored against fixed GT. VLMs see the **same pair order and same left/right layout** as you did.

---

## 3. Import schema

Import Data Lab export (zip or folder). **Do not commit export files.**

### Export layout (FaceIQ Data Lab)

```
faceiq-export/
  README.txt
  manifest.jsonl      ← one row per image (FRONT or SIDE)
  manifest.csv
  images/             ← local .webp files named {analysis_id}_front.webp
```

Current bundle: **100 rows**, **52 unique analyses with FRONT**, **48 with FRONT+SIDE**. All VLM assessment columns are **null** in this export. **No landmarks.**

### Import rules

1. Read `manifest.jsonl` (or `manifest.csv`).
2. Keep rows where **`view === "FRONT"`** only (pilot A/B uses front photos).
3. One `VlmPilotFace` per **`analysis_id`** (not per `image_id`).
4. Match **`filename`** to uploaded file in `images/` → store as **`frontPhotoStorageUrl`** (upload to blob or app storage on import).
5. Skip duplicate `analysis_id`.

### Field mapping (manifest → DB)

| Manifest field | DB / use | Pilot required |
|----------------|----------|----------------|
| `analysis_id` | `sourceFaceId` | **Yes** |
| `filename` | join to `images/` on import | **Yes** |
| `prod_gender` | `gender` | **Yes** (VLM prompt) |
| `prod_race` | `race` (parse JSON string, e.g. `["white"]` → `white`) | **Yes** |
| `overall_score`, pillar scores | `exportPayload` only | Optional (reference; **not** VLM labels) |
| `has_front`, `has_side` | `exportPayload` | Optional |
| `image_id`, `image_key`, `stage` | `exportPayload` | Optional |
| `vlm_ethnicity`, `acne`, `description`, etc. | `exportPayload` | Optional (all null in current export) |

**Not in this export:** `frontPhotoUrl`, landmarks, `createdAt`, prod `Face.id`.

### Reject / skip if

- No matching file in `images/` for `filename`
- Missing `analysis_id` or `prod_gender`
- Duplicate `analysis_id` in same import
- Row is `SIDE` only (use FRONT row as canonical face)

### Example manifest row (FRONT)

```json
{
  "analysis_id": "cml638pm000kg1jfmt5owrqyo",
  "image_id": "973538dd-59cc-44fd-a719-a30d0d1ead47",
  "view": "FRONT",
  "filename": "cml638pm000kg1jfmt5owrqyo_front.webp",
  "image_key": "raw/images/cml638pm000kg1jfmt5owrqyo_front.webp",
  "stage": "RAW",
  "overall_score": 6.7,
  "harmony_score": 6.8,
  "prod_gender": "male",
  "prod_race": "[\"white\"]",
  "has_front": true,
  "has_side": true,
  "vlm_ethnicity": null,
  "acne": null
}
```

### Import UI

- Upload **zip** (extract `manifest.jsonl` + `images/`) or select folder via file picker with same structure
- Preview: N FRONT faces found, gender breakdown, first 5 thumbnails
- On confirm: upload images → insert `VlmPilotCohort` + `VlmPilotFace` rows

---

## 4. Database models (research-only)

Separate from prod `Face`. Prefix tables e.g. `vlm_pilot_*`.

**Canonical reference (11 models, migrations, sample-set API):** [`scoring-gt-schema.md`](../scoring-gt-schema.md) §1 and §1.5. Below is the original pilot subset; sample curation adds `VlmPilotSampleSet*`, face decile fields, and `VlmPilotRun.sampleSetId`.

### `VlmPilotCohort`

One import batch.

| Column | Purpose |
|--------|---------|
| `id` | cuid |
| `name` | e.g. `"faceiq-export-2026-06-24"` |
| `importedAt` | timestamp |
| `faceCount` | denormalized |
| `notes` | optional |

### `VlmPilotFace`

| Column | Purpose |
|--------|---------|
| `id` | cuid (internal) |
| `cohortId` | FK |
| `sourceFaceId` | `analysis_id` from manifest |
| `frontPhotoUrl` | URL after upload (from local `images/{filename}`) |
| `gender`, `race` | from `prod_gender`, `prod_race` |
| `exportPayload` | full FRONT manifest row JSON |

Unique: `(cohortId, sourceFaceId)`.

### `VlmPilotRun`

One experiment (40 pairs from one cohort).

| Column | Purpose |
|--------|---------|
| `id` | cuid |
| `cohortId` | FK |
| `seed` | string, e.g. `"vlm-pilot-v1"` |
| `pairCount` | 40 |
| `status` | `draft` → `pairs_generated` → `human_labeled` → `vlm_complete` |
| `createdAt` | |

### `VlmPilotComparison`

Immutable row per pair — **source of truth for order and layout**.

| Column | Purpose |
|--------|---------|
| `id` | cuid |
| `runId` | FK |
| `pairIndex` | 0..39, **display order** |
| `faceAId` | FK → `VlmPilotFace` |
| `faceBId` | FK → `VlmPilotFace` |
| `presentationSeed` | derived from run seed + pair id |
| `showFaceAOnLeft` | boolean — fixed before any labeling |
| `humanWinnerFaceId` | FK, null until you label; null if tie |
| `humanOutcome` | `A` \| `B` \| `tie` |
| `humanLabeledAt` | |
| `status` | `pending_human` → `human_done` → `vlm_done` |

Unique: `(runId, pairIndex)`.  
Unique canonical pair optional: `(runId, minFaceId, maxFaceId)` if deduping.

**Resolved URLs for UI/VLM:** join `faceA` / `faceB` for `frontPhotoUrl`. Store ids, not duplicated URLs.

### `VlmPilotVlmResult`

One row per (comparison × model × prompt variant).

| Column | Purpose |
|--------|---------|
| `id` | cuid |
| `comparisonId` | FK |
| `vlmModel` | e.g. `gemini-3-flash` |
| `promptVersion` | e.g. `pairwise-v1` |
| `vlmWinnerFaceId` | FK or null if tie |
| `vlmOutcome` | `A` \| `B` \| `tie` |
| `confidence` | high \| medium \| low |
| `rationale` | text |
| `rawResponse` | JSON |
| `inputTokens`, `outputTokens` | |
| `costUsd` | computed |
| `latencyMs` | |
| `createdAt` | |

---

## 5. Deterministic pair generation

Run **once** when creating a `VlmPilotRun`. Persist all rows before any UI labeling.

### Algorithm

```text
1. Load face internal ids for cohort; sort lexicographically (reproducible)
2. seed = run.seed (e.g. "vlm-pilot-v1")
3. pairs = []; seen = {}; i = 0
4. While pairs.length < 40:
     h = md5(seed + ":" + i)
     idxA = int(h[0:8])  % N
     idxB = int(h[8:16]) % N
     if idxA == idxB: i++; continue
     (faceA, faceB) = canonical order (lower id first) for dedupe key
     if key in seen: i++; continue
     seen.add(key)
     showFaceAOnLeft = int(h[16:18]) % 2 == 0
     insert VlmPilotComparison { pairIndex, faceA, faceB, showFaceAOnLeft, … }
     i++
5. Set run.status = pairs_generated
```

Same seed + same cohort → **same 40 pairs, same order, same left/right**.  
Implement as API `POST /runs/:id/generate-pairs` or auto on run create.

**Do not** generate pairs in the UI on the fly. Script/server only, writing to DB.

---

## 6. Admin UI — `/research/pairwise`

Follow existing admin patterns. Components under `src/components/admin/pairwise/`. API under `/api/admin/pairwise/*`. RBAC: `research.lab.view`. Nav: `PairwiseLabNav.tsx`.

### Routes

| Path | Page | Purpose |
|------|------|---------|
| `/research/pairwise` | Dashboard | Cohort list, run list, links |
| `/research/pairwise/import` | Import | Upload zip → preview → confirm |
| `/research/pairwise/cohorts/[cohortId]` | Cohort detail | Face grid, create pilot run, link to sample |
| `/research/pairwise/cohorts/[cohortId]/sample` | **Sample curation** | 3k draw, histograms, review, fill gaps, approve |
| `/research/pairwise/runs/[runId]` | Run overview | Pair table, status, actions |
| `/research/pairwise/runs/[runId]/label` | Human labeling | A/B queue (pairIndex order) |
| `/research/pairwise/runs/[runId]/vlm` | VLM batch | Model/prompt select, run, progress |
| `/research/pairwise/runs/[runId]/results` | Results | Accuracy, cost, model ranking |
| `/research/pairwise/runs/[runId]/review` | Review | Pair-by-pair: your pick vs AI, rationale, disagreements |

### 6.0 Sample curation (`/cohorts/[cohortId]/sample`)

- **Run draw** — seeded stratified sample (~3k); deciles computed on first draw if missing
- **Review grid** — paginated photos; remove with reason (`duplicate`, `synthetic`, `quality`, `other`)
- **Fill gaps** — manual restore of under-target gender × decile cells (not auto on remove)
- **Audit checklist** — blocks approve until tails, non-empty deciles, race distribution pass
- **Approve sample** — locks draft set (read-only)

DB: `scoring-gt-schema.md` §1.5. Server lib: `src/lib/vlm-pilot/sample/`.

### 6.1 Cohort import (`/import`)

- Upload **zip** (must contain `manifest.jsonl` + `images/`)
- Validate schema; show row-level errors
- Preview: FRONT face count, gender breakdown, first 5 thumbnails
- Confirm → upload images to blob → insert cohort + faces

### 6.2 Create run (cohort detail)

- Select cohort (or pre-selected on cohort page)
- Seed (default `vlm-pilot-v1`), pair count (default 40, max = face count)
- **Generate pairs** → redirect to run overview with pair table

### 6.3 Human labeling (`/runs/[id]/label`)

- Queue comparisons in **`pairIndex` order**
- Side-by-side photos respecting `showFaceAOnLeft`
- Labels: show **Left** / **Right** (map to face A/B internally) + **Tie**
- Keyboard: `1` / `2` / `t` or arrow keys
- Progress: `12 / 40`
- PATCH on each save → `humanWinnerFaceId` (null if tie), `humanLabeledAt`, `status = human_done`
- Editable until run status `vlm_batch_started`

### 6.4 VLM batch (`/runs/[id]/vlm`)

- Gate: all comparisons human-labeled (`humanLabeledAt` set)
- Multi-select models; each batch **appends** new results (reruns accumulate)
- Prompt version dropdown (current: `pairwise-v2`)
- Run batch server-side; poll progress per batch (`pairs × selected models`)
- Store one `VlmPilotVlmResult` per comparison × model × batch invocation

### 6.5 Results (`/runs/[id]/results`)

Per model (+ prompt):

| Metric | Source |
|--------|--------|
| Accuracy | `vlmWinnerFaceId === humanWinnerFaceId` (ties excluded or counted separately) |
| Tie rate | `vlmOutcome = tie` |
| Avg / total cost | `costUsd` |
| Avg latency | `latencyMs` |
| Projected 10k cost | `totalCost / 40 × 10000` |

**Go criteria (tune):** e.g. ≥ 75% accuracy on pilot, acceptable $/pair → promote to full GT pipeline. **Met** — proceed to scale + prompt iteration.

### 6.6 Review (`/runs/[id]/review`)

- Side-by-side photos with your pick vs selected model pick
- Filter: disagreements only
- Shows VLM rationale + confidence for case-by-case audit

---

## 7. VLM call contract

Same prompt template as `scoring-gt-core.md` §5. Pass:

- Image left / image right (respect `showFaceAOnLeft`)
- `genderA`, `genderB`
- `promptVersion` id for reproducibility

Response JSON: `winner`, `confidence`, `rationale` → map to `vlmWinnerFaceId`.

Log tokens + compute `costUsd` per call for criterion **B**.

---

## 8. API — `/api/admin/pairwise`

| Method | Path | Notes |
|--------|------|--------|
| `POST` | `/cohorts/import` | multipart zip |
| `GET` | `/cohorts` | list |
| `GET` | `/cohorts/:id` | faces + metadata + `activeSampleSet` |
| `GET` | `/cohorts/:id/sample` | latest sample set, histograms, audit |
| `POST` | `/cohorts/:id/sample/draw` | `{ seed, targetSize?, name? }` |
| `GET` | `/sample-sets/:id/members` | paginated review grid (`gender`, `decile`, `page`) |
| `GET` | `/sample-sets/:id/pool-browse` | pool faces not in sample |
| `PATCH` | `/sample-sets/:id/members/:faceId` | `{ action: "exclude", reason, note? }` |
| `POST` | `/sample-sets/:id/fill-gaps` | restore under-target cells |
| `POST` | `/sample-sets/:id/approve` | lock if audit passes |
| `GET` | `/sample-sets/:id/events` | curation audit trail |
| `POST` | `/runs` | `{ cohortId, seed, pairCount }` |
| `POST` | `/runs/:id/generate-pairs` | idempotent if pairs exist |
| `GET` | `/runs/:id` | run + status counts |
| `GET` | `/runs/:id/comparisons` | ordered by `pairIndex`, include face URLs |
| `PATCH` | `/comparisons/:id/human` | `{ winnerFaceId \| null, outcome: A\|B\|tie }` |
| `POST` | `/runs/:id/vlm-batch` | `{ modelIds[], promptVersion }` — async job |
| `GET` | `/runs/:id/vlm-batch/status` | progress, cost so far |
| `GET` | `/runs/:id/results` | aggregated metrics per model |
| `GET` | `/runs/:id/review` | pair-level human vs VLM (`?modelId=`, `?disagreementsOnly=true`) |

---

## 9. What you do manually

1. Import zip once (~52 fronts from current export)  
2. Create run → generate 40 pairs  
3. Label all 40 at `/admin/pairwise/runs/[id]/label` (≈ 5–10 min)  
4. Run VLM batch at `/admin/pairwise/runs/[id]/vlm`  
5. Read results → pick model or pivot to crowdsourcing  

You are the rater for this pilot. The 40 labels **are** the ground truth.

---

## 10. Out of scope (this spec — v1 pilot only)

- ~~Full 1k cohort / 10k pairs~~ → moved to `scoring-gt-core.md` §13–§15 (scale phase)
- Bradley–Terry refit (optional later on same data)
- Crowdsourcing / trap pairs
- Preference model training
- Committing export zip to git

---

## 11. Implementation order

1. Prisma models (`vlm_pilot_*`) + manual migration SQL  
2. `/api/admin/pairwise/*` — import, pair generation, human PATCH  
3. `/admin/pairwise/import` + cohort pages  
4. `/admin/pairwise/runs/[id]/label` — human queue  
5. VLM batch route + existing VLM client (`scoring-gt-core.md` §5 prompt)  
6. Results + review pages + nav entry  

---

## 12. Build readiness

**Shipped:** pilot (≤80 pairs) + **full GT single-model run** (52.5k pairs, `pairwise-v3-gt`, export + review + export-exclude).  
**Next:** offline BT + training in `faceiq-preference-ml` (see `scoring-gt-training.md`).  
**Deferred:** 3-model panel at scale (§15 panel row) — not used for this run.

---

## 13. Success outputs

| Question | Deliverable |
|----------|-------------|
| **A) Can VLM do pairwise?** | ✓ ~76–86% on pilot (Flash models) |
| **B) Cost?** | ✓ ~$8–30 / 10k pairs (single call, Flash) |
| **C) Which model?** | ✓ Gemini 2.5 / 3 / 3.1 Flash-Lite tier; prompt + consensus next |

Winning config → full GT pipeline in `scoring-gt-core.md`.

---

## 14. Pilot results snapshot (2026-06)

Single-call accuracy vs 40 human labels (`pairwise-v2`, ties excluded from denominator):

| Model | Accuracy | ~$/pair | ~$ / 10k proj. |
|-------|----------|---------|----------------|
| Gemini 2.5 Flash | ~86% | $0.0016 | ~$16 |
| Gemini 3 Flash | ~80% | $0.0030 | ~$30 |
| Gemini 3.1 Flash-Lite | ~77% | $0.0008 | ~$8 |
| Claude Opus 4.8 | ~77% | $0.021 | ~$209 |
| GPT-5.3 Chat | ~74% | $0.0086 | ~$86 |

**Conclusion:** Flash-class Gemini models are **accurate enough and cheap enough** for production-scale pairwise labeling. Opus is not cost-competitive at similar accuracy. Next levers: **prompt tuning** and **multi-model consensus** (see `scoring-gt-core.md` §5.1).

---

## 15. Scale VLM batch — before ~157k panel calls

**Scope:** Full GT labeling — ~52.5k pairs × 3 models ≈ **157k** sync calls (~$400–600 Flash). Pilot batch path (`POST …/vlm-batch` → `runVlmPilotBatch` in `after()`) is **not sufficient as-is**.

### How it works today (pilot)

- One API POST starts **one** serverless `after()` job (`src/lib/vlm-pilot/batch.ts`).
- **Concurrency 5**; calls go through **Vercel AI Gateway** (`vlm-pilot-pairwise` tag).
- Runs on **Vercel** (not your laptop), but a single invocation **times out** (~5–13 min) — far short of 157k calls.
- On error (429, 503, timeout): **logged, no retry**; batch may end **`partial`**.
- Re-clicking “Run batch” **appends** results — does **not** skip already-succeeded `(comparison × model × prompt)` → duplicate spend risk.

### Required before scale (not built yet)

| Item | Why |
|------|-----|
| **Chunked multi-batch runner** | Many batch invocations (e.g. 500–2k calls each), cron/worker or chained `after()` — not one 157k job |
| **Idempotent task queue** | Enqueue only missing `(comparisonId, modelId, promptVersion)`; never redo successes |
| **Retry + backoff** | 429 / 503 / timeout: exponential backoff, max retries per task |
| **`maxSpendUsd` cap** | Hard stop (planned in core §5.3; not in code) |
| **Coverage gate** | Run “complete” only when **100%** required panel slots filled; block BT refit until then |
| **Failure + cost monitor** | Extend status UI: failed/pending count, coverage %, 429 rate (cost partial today via `VlmBatchPanel`) |
| **Raise pair cap** | `PAIRWISE_MAX_PAIR_COUNT` still **80** — must lift for ~52.5k pairs |
| **3-model consensus aggregation** | Panel vote → `consensusOutcome` before BT (core §5.1) |

### Rate limits (expect 429s; plan for them)

- **Vercel AI Gateway:** no published RPM; paid credits > free. Limits often pass through to upstream. [Pricing](https://vercel.com/docs/ai-gateway/pricing)
- **Gemini (upstream):** RPM / TPM / RPD per **project** — check live limits in AI Studio. [Rate limits](https://ai.google.dev/gemini-api/docs/rate-limits). At concurrency 5, RPM is usually fine; TPM (2 images/call) and spend-rate windows matter more.
- **Not a distillation ban** — throttling, not account flags. Retry + chunking is the fix.

### Failed calls ≠ wasted cycle

Partial spend on successes is kept. Gaps block consensus/BT until **idempotent retry** fills them. Do not treat `partial` as done.

### Transport options (later)

- **Near term:** hardened chunked batch on **Gateway** (same as pilot).
- **Optional at scale:** **Gemini Batch API** direct — separate quota, ~50% cheaper, async; new integration.

---

## 16. GT export (post-labeling → ML repo)

**When:** After VLM batch complete + audit gate (research log §4.3).

**Endpoint:**

```http
GET /api/admin/pairwise/runs/{runId}/export?promptVersion=pairwise-v3-gt&modelId=gemini-2.5-flash&offset=0&limit=2000
```

**Full run:** `cmr1mr0m7000196d57zi3vcgn` — paginate until all **52,414** export-eligible rows retrieved (`excludedCount` in response = rows soft-dropped via Review).

**Row fields:** face ids, photo URLs, VLM outcome/confidence, human audit fields (`humanWinnerFaceId`, `isHumanOverride`), exclude metadata.

**Finalize label for BT/training (in ML repo):** use `humanWinnerFaceId` when `humanLabeledAt` set, else `vlmWinnerFaceId`.

**Storage:** `~/research-data/scoring-gt/artifacts/gt-full-export/` — never commit to git. See `scoring-gt-training.md`.
