# Scoring Ground Truth — Core Research Plan

Pairwise-first plan: collect A/B matchups → Bradley–Terry rankings → train a **preference model** → rate users via an **anchor ladder**. Replaces tuning the harmony + pillar formula.

Pillar assessments stay user-facing; not required for Phase 1.

---
## 1. Phases (pairwise-first)

| Phase | What | Output |
|-------|------|--------|
| **Export** | ~10k–50k pool → stratified sub-sample **2k–5k faces** | Frozen cohort in data lab |
| **VLM pilot** | ✓ **Done** — 40 pairs, model + prompt validated | Flash tier + `pairwise-v2` |
| **Prompt + consensus** | Prompt A/B tests; **3-model panel** per pair | Consensus winner + confidence |
| **Labeling** | **~52k** A/B matchups (3k faces × 35 comps; 3× VLM panel) | Winner per pair |
| **Rankings** | Bradley–Terry refit + calibration curve | `scoreOutOf10` per face + 50–100 anchors |
| **Training** | Preference model on (photo A, photo B) → winner | Deployable comparator |
| **Production** | New user vs anchor ladder | Explainable /10 + confidence band |

**Scale (default v1):** **3,000 faces × 35 comparisons/face ≈ 52,500 pairs** → ~157k panel VLM calls (~$400–600 Flash). Formula: `pairs = (N × k) / 2`. See §14–§15.

Phase 2+ (deferred): pointwise regression, landmarks/ratios/VLM ablation — see §9.

---

## 2. Export fields & import

Import at `/admin/pairwise/import` (zip: `manifest.jsonl` + `images/`). Field spec below.

| Parameter | Value |
|-----------|-------|
| Parent pool export | **~30k faces** (minimum 20k) |
| Active cohort (we sub-sample) | **3,000 faces** |
| Comparisons per face | **35** |
| Total pairs | **~52,500** |
| VLM panel calls (×3) | **~157,500** |

---

### Import manifest fields

| Field | Required | What we use it for | Notes / limitations |
|-------|----------|-------------------|---------------------|
| `analysis_id` | **Yes** | Stable face id | Becomes `sourceFaceId` |
| `filename` | **Yes** | Join to image file | e.g. `{analysis_id}_front.webp` |
| `view` | **Yes** | Filter | Must be `"FRONT"` |
| `prod_gender` | **Yes** | VLM prompt; same-gender pairing; optional NN conditioning | `male` / `female` |
| `prod_race` | **Yes** | Cohort diversity audit; optional Phase 2 ablation | JSON array string e.g. `["white"]`. **Not a ranking label.** |
| `overall_score` | **Yes** | **Stratified sampling** (decile bins) + cross-decile pair quotas + post-hoc validation vs BT | **Not a training label.** Required for spread — do not omit. |
| `harmony_score`, pillar scores | Recommended | Stored in payload; validation scatter vs BT later | Not used for stratification or training v1 |
| `createdAt` | **Yes** | Cohort vintage / date filters | Document cutoff dates in README |
| `frontLandmarks` | Optional | Quality hint; Phase 2 ablation; **re-runnable from photos** | JSON if available |
| `sideLandmarks` | Optional | Phase 2 only | Include if cheap |
| Front image bytes | **Yes** | VLM + NN training | `images/{analysis_id}_front.webp` |

**Do NOT export:** `userId`, email, name, billing, or any PII.

**Landmark validation (optional — prefer keeping the row if photo is good):**

- If exported: `frontLandmarks` non-null, expected point count, no all-zero/NaN  
- Prefer analyses after **auto-landmark rollout** when pool size allows  
- **If landmarks missing or suspect:** still export the face; we will re-run landmarking on the image  

**README must include:**

- Export timestamp  
- Exact SQL / filters and **row count after each filter stage**  
- `AUTO_LANDMARK_ROLLOUT_DATE` (and assessment date if used)  
- Final row count, gender breakdown, score min/median/max  
- Known limitations (e.g. “assessment date filter removed to preserve N=28k”)  

---

**Why each class of data matters:**

| Data | Purpose |
|------|---------|
| **Photos** | VLM labeling + preference model input |
| **`overall_score`** | **Only** to build a representative sample (decile stratification) and validate BT at the end — **never** fed as the training target |
| **`prod_gender`** | Same-gender matchups; VLM context; optional model conditioning |
| **`prod_race`** | Ensure cohort diversity; audit only in v1 — **must not** drive attractiveness ranking |
| **`frontLandmarks`** | Optional quality hint; recomputable from photos |
| **`createdAt`** | Reproducible cohort definition; enforce auto-landmark era |

---

### Field reference (schema)

| Field | Source (prod) | Required | Notes |
|-------|---------------|----------|-------|
| `analysis_id` | Face / analysis id | **Yes** | Stable id; becomes `sourceFaceId` |
| `filename` | `{analysis_id}_front.webp` | **Yes** | Must exist in `images/` |
| `view` | `"FRONT"` | **Yes** | Skip SIDE-only rows |
| `prod_gender` | `gender` | **Yes** | VLM prompt + pairing |
| `prod_race` | `race` (JSON array string) | **Yes** | e.g. `["white"]` |
| `overall_score` | Labs overall at export | **Yes** | Stratification + validation only |
| `harmony_score`, pillar scores | Labs | Recommended | `exportPayload`; validation only |
| `frontPhotoUrl` or raw image bytes | Blob / storage | **Yes** | Upload to export zip `images/` |
| `createdAt` | Face created | **Yes** | Date filters + cohort vintage |
| `frontLandmarks` | MediaPipe / stored landmarks | Optional | Nice to have; **re-runnable from photos** if missing — do not shrink pool |
| `sideLandmarks` | If side exists | Optional | Phase 2 |

**Do not export:** `userId`, email, name, billing, or any PII.

**Export deliverable checklist:**

1. `manifest.jsonl` — one FRONT row per face  
2. `images/*.webp` — front photos, filenames match manifest  
3. `README.txt` — export date, filters applied, row count  
4. Optional `manifest.csv` — same columns as jsonl  

**Pool vs active cohort:**

| Set | Target size | Purpose |
|-----|-------------|---------|
| **Parent pool** | **~30k** faces (min 20k) | Stratified draw; held frozen |
| **Active GT cohort** | **3k** faces | Pairwise labeling + BT refit + training export |

Sub-sample from parent with a **documented seed** + stratification rules (§15).

### Phase 2+ — regression / ablation (later)

Add side photos, pillar scores, auto-VLM assessment JSON, ratios. Require post-auto-landmark and post-auto-VLM-assessment dates when using assessments (exclude self-assessed).

---

## 3. Core data model (concepts)

These entities live in the research database — separate from production.

### Sample set

A frozen cohort: seed, size, optional gender filter, optional parent set (for sub-sampling).

- **Parent pool:** ~10k faces, exported once
- **Child set:** ~1k sub-sample drawn from parent with a new seed

### Sample face

One face in a sample set. Holds a frozen snapshot of exported fields (`exportPayload`) so research is self-contained.

### Pairwise comparison

Two faces compared head-to-head.

| Concept | Values / notes |
|---------|----------------|
| Outcome | A wins, B wins, or tie |
| VLM result | winner, confidence (high / medium / low), raw response |
| Human override | Optional correction on top of VLM |
| Status | pending → vlm_done → human_review → final |

### ELO rating

Per face in a sample set:

| Field | Purpose |
|-------|---------|
| `elo` | Raw rating |
| `scoreOutOf10` | Mapped GT — **anchors + validation**, not pairwise training label |
| `comparisonCount` | How many comparisons resolved |
| `labsOverallScore` | Copy of Labs overall at export — for validation scatter |

---

## 4. ELO / Bradley–Terry system

Pairwise attractiveness is modeled with **Bradley–Terry** (statistically rigorous) and **Elo** (online updates for UI feedback). For binary wins they are equivalent:

\[
P(A \text{ beats } B) = \frac{1}{1 + 10^{(R_B - R_A)/400}}
\]

### Two-layer rating pipeline

| Layer | When | Purpose |
|-------|------|---------|
| **Online Elo** | During labeling | Fast UI feedback as comparisons resolve |
| **Bradley–Terry MLE refit** | After batch finalization | **Ground truth export** — handles sparse graphs, maximum likelihood |

Export `scoreOutOf10` from the **BT refit**, not from stale online totals. Human overrides → update comparison row → **full refit** (no incremental cascade).

### Upset factor (no “big win” multiplier)

Forced-choice A/B only observes **who won**, not margin. Do **not** add separate “big win” multipliers.

Elo already encodes difficulty:

- Beat someone much **higher-rated** → large rating gain (upset)
- Beat someone much **lower-rated** → small gain
- Lose to someone much **lower-rated** → large penalty

That is exactly “beating an attractive person means more.”

### Ties and close calls

Three-outcome model (Rao–Kupper / Bradley–Terry with ties):

| Outcome | ELO update |
|---------|------------|
| A wins clearly | A +K, B −K |
| B wins clearly | opposite |
| Tie / too close | Partial credit (~K/4 each, or 0.5 / 0.5 expected score) |

VLM returns `winner: A | B | tie` plus `confidence: high | medium | low`.

Routing:

- `tie` or `confidence: low` → smaller ELO delta + **human review queue**
- `confidence: high` → auto-accept unless random audit flags it

Tell the VLM: pick a winner when there is any meaningful difference; use `tie` only when genuinely ~50/50.

### Mapping ELO → score out of 10

After BT refit on the active 1k cohort:

**Recommended — percentile rank → fixed calibration curve** (not raw Elo):

```text
scoreOutOf10 = calibrationCurve(percentile_rank(θ))
```

Define the curve **before the experiment** (median → 5.0, top 1% → 8.0, etc.). Raw Elo values are not portable — only rank order matters.

Always run **sanity check**: Spearman ρ and scatter plot of `scoreOutOf10` vs Labs `overallScore` before trusting GT.

### Comparison count (avoid full round-robin)

For **1k faces**, full round-robin ≈ 500k pairs — too many.

Target **~20 comparisons per face** (~10k pairs for 1k cohort). Formula: `pairs = (N × k) / 2`. Full round-robin = `N × (N − 1) / 2` = 499,500 — impractical.

- Generate all pairs **deterministically from seed before labeling**
- Pair by rating uncertainty + random exploration (connected graph)

---

## 5. VLM pairwise labeling

**Pilot complete** (`vlm-pilot-spec.md` §14): Flash models ~76–86% vs human on 40 pairs; ~$8–30 / 10k single-call pairs. **Proceed with prompt experiments + 3-model consensus at scale.**

Run **front photos only** for A/B.

### 5.1 Multi-observer consensus (recommended for production)

Instead of one VLM call per pair, run a **panel of 3 independent observers** on the same matchup. Default panel (from pilot):

1. **Gemini 2.5 Flash**  
2. **Gemini 3 Flash**  
3. **Gemini 3.1 Flash-Lite**

Each call uses the same prompt + photos; store all three in `VlmPilotVlmResult` (already supported). Derive a **consensus outcome** before BT refit:

| Vote pattern | Consensus | Confidence | Action |
|--------------|-----------|------------|--------|
| **3–0** unanimous | Majority side | `high` | Auto-accept |
| **2–1** majority | Majority side | `medium` | Auto-accept |
| **2–1** with dissenter = `tie` | Majority side | `medium` | Auto-accept (tie vote ignored) |
| **1–1–1** (A, B, tie) | `tie` | `low` | Human review queue |
| **1–1** split (A vs B, no tie) | `tie` | `low` | Human review OR second-round tiebreak |
| **3× tie** | `tie` | `high` | Accept as tie (genuinely ambiguous) |

**Optional tiebreak (2nd round):** if 1–1 split, call a 4th model (e.g. Gemini 3.5 Flash) or route to human review. Cheaper default: **human review only on ~5–15%** of pairs (splits + low single-model confidence).

**Why this works:** Independent errors are uncorrelated; majority vote lifts effective accuracy above any single model (~86% → often **90%+** in practice). Cost: **3× calls** — still ~**$24–90 / 10k pairs** on Flash tier (pilot economics).

**Implementation note:** Add `consensusOutcome`, `consensusConfidence`, `panelVoteJson` on comparison or a derived table after batch; BT refit uses **consensus**, not individual model votes.

### System prompt (template)

```text
You are evaluating facial attractiveness in a pairwise A/B study.

Task: Given two front-facing photos (Face A left, Face B right), decide which
face would be rated MORE ATTRACTIVE in a general population preference study.

Context:
- Face A is {genderA}, Face B is {genderB}.
- When genders differ, judge from the perspective of typical opposite-sex
  attraction. When same gender, judge general facial attractiveness.
- Use ONLY visible facial features: proportions, symmetry, skin clarity,
  dimorphic cues appropriate to gender, overall harmony. Ignore hairstyle,
  clothing, background, filters, and makeup intensity unless they obscure
  bone structure.

Rules:
- Return winner: "A" | "B" | "tie"
- "tie" ONLY if truly indistinguishable (~50/50); prefer a winner when unsure
- confidence: "high" | "medium" | "low"
- Do NOT score 1–10 here; this is relative preference only
- Do NOT use race or ethnicity as ranking criteria

Output STRICT JSON:
{
  "winner": "A" | "B" | "tie",
  "confidence": "high" | "medium" | "low",
  "rationale": "one sentence, observable features only"
}
```

### Prompt design

| Include | Avoid |
|---------|-------|
| Holistic attractiveness | Long ratio lists (that is harmony scoring, not A/B) |
| Opposite-gender framing when mixed pairs | Race/ethnicity as ranking criteria |
| Tie option with strict guidance | Absolute 1–10 scores in the same call |
| Confidence for review routing | Identity or user metadata |

**Same-gender pairs:** “general facial attractiveness.” Optionally run **stratified queues** (e.g. male faces judged “as perceived by women”) but validate on a human subset — VLMs are weak at modeling gender-specific taste.

### Batch job controls

- Hard cap: max spend, max comparisons
- Resume from pending queue
- Route `tie` / low confidence → human review before marking final
- Log cost per comparison for budget tracking

### Budget estimate (updated post-pilot)

Cost scales with **pairwise comparisons × observers per pair**.

| Scenario | Pairs | Observers | VLM calls | Est. cost (Flash) |
|----------|-------|-----------|-----------|-------------------|
| Baseline | 10k | 1 | 10k | ~$8–30 |
| **Recommended** | 10k | 3 | 30k | ~$24–90 |
| Scale | 50k | 3 | 150k | ~$120–450 |
| Scale + tiebreak 10% | 50k | 3.1 avg | ~165k | ~$130–500 |

Human review (~5–15% of pairs at scale) = labor, not API.

---

## 6. UI pages (concepts)

### Cohort management

- **Sample set list** — frozen cohorts (10k pool + 1k sub-samples; seed, size, date)
- **Sub-sample creator** — pick parent pool → new seed → create 1k child set
- **Import validation** — face count, required fields, landmark overlay preview

### Cohort preview

- **Face grid** — thumbnails, gender, race, Labs `overallScore`
- **Distribution preview** — Labs overall score histogram for the active cohort (before ELO exists)

### Ground truth labeling

- **Pairwise A/B UI** — two front photos side-by-side; pick more attractive or tie; keyboard shortcuts
- **Pair queue** — smart pairing (ELO uncertainty + random exploration); completion %
- **VLM batch panel** — run VLM on pairs; show progress, cost, errors
- **Human review queue** — filter: VLM tie, low confidence, random audit, all pending
- **Override controls** — set human winner; mark comparison final

### ELO + validation

- **Live ELO feedback** — rating updates as comparisons resolve (online layer)
- **GT dashboard** — scatter / table: `scoreOutOf10` vs Labs `overallScore`; Spearman ρ
- **Refit trigger** — run Bradley–Terry MLE on finalized comparisons
- **Export GT** — download dataset for notebook: face id, features, elo, scoreOutOf10, labsOverallScore

### Notebook handoff

- **Matchup export** — pairs + winners for training
- **Face export** — ids, photos, BT scores, anchor flags

---

## 7. Core operations (functions)

| Operation | Input | Output |
|-----------|-------|--------|
| **Create sample set** | seed, size, optional parent set, optional gender filter | Sample set + sample faces |
| **Sub-sample from pool** | parent set id, new seed, size (e.g. 1000) | Child sample set |
| **Generate pair queue** | sample set, pairing strategy (uncertainty + exploration) | Ordered list of face pairs |
| **Record human comparison** | pair id, winner (A / B / tie) | Updated comparison + online ELO |
| **Run VLM batch** | pending pairs, model id, budget cap | VLM outcomes stored per pair |
| **Route to review** | comparisons with tie / low confidence | Human review queue |
| **Apply human override** | comparison id, human winner | Comparison marked for refit |
| **Finalize comparisons** | comparison ids | Status → final |
| **Refit Bradley–Terry** | all final comparisons in sample set | MLE ratings per face |
| **Map to /10** | BT ratings, mapping strategy (percentile recommended) | `scoreOutOf10` per face |
| **Validate vs Labs** | scoreOutOf10, labsOverallScore | Spearman ρ, scatter data |
| **Export GT dataset** | sample set id | JSON/CSV for notebook |
| **Check ranking stability** | consecutive refits | Boolean: change < ε |

---

## 8. Pairwise model training (Phase 1)

### Data collection vs model training

```text
Labeling:  10k matchups (who won)  →  BT refit  →  scoreOutOf10 + anchors
Training:  ~8k matchup rows         →  gradient descent  →  s(photo) comparator
Deploy:    new face vs 50–100 anchors  →  /10 from win/loss bands
```

The NN learns the same preference structure BT extracted, but from pixels. It outputs implicit scores `s(A)`, `s(B)` via a shared backbone; **higher score wins**.

### What we train and measure

| | **Pairwise model (Phase 1)** | **Pointwise regression (Phase 2+)** |
|--|------------------------------|-------------------------------------|
| **Input** | Photo A + Photo B | One photo (+ optional features) |
| **Label** | **Who won** (A / B / tie) | BT `scoreOutOf10` |
| **Loss** | **Pairwise cross-entropy** | MAE / MSE |
| **Validate** | **Pairwise accuracy** on held-out pairs; Kendall τ vs BT | MAE vs BT on held-out faces |

**Not MAE on /10 for the comparison model.** BT scores are for anchors, sanity checks, and optional pointwise experiments — not the primary training label.

### Training loop

1. Split by **face id** (not random pairs) → ~8k train / ~2k val matchups  
2. Each batch: `(photo_A, photo_B, winner)` → forward → `s_A`, `s_B`  
3. Cross-entropy loss; backprop until val pairwise accuracy plateaus  
4. Optional: anchor-ladder inference test vs BT on held-out faces  

### Paradigms

| Approach | Phase 1? |
|----------|----------|
| **Pairwise** (who wins) | **Yes** |
| **Pointwise** (photo → /10) | Phase 2+ |
| **Listwise** (full list NDCG) | Skip for now |

### Inference

Compare new user vs **50–100 BT-rated anchor faces**. Aggregate wins/losses into an explainable /10 interval (inconsistent results = wider band, not failure).

### Metadata at training time (gender, race, etc.)

**Training label:** only **who won the matchup** (A / B / tie). Not `overall_score`, not BT θ.

**Phase 1 v1 (default):** **Photos only** — shared backbone `s(photo)`; cross-entropy on winner. The model learns attractiveness cues from pixels, same as the VLM. Gender/race are **not** required inputs for v1.

**Phase 1 optional conditioning (recommended after v1 baseline):**

| Metadata | Use | Caution |
|----------|-----|---------|
| **`gender`** | Concat to backbone or separate heads; matchups are same-gender so the model learns **within-gender** preference | Helps M–M vs F–F calibration |
| **`race` / ethnicity** | **Audit + stratified sampling only in v1** | **Do not** train the model to rank races. Any Phase 2 conditioning is for fairness testing / ablation only — never “ethnicity X is less attractive” |
| **Landmarks / ratios** | Phase 2 ablation | Compare image-only vs image+geometry |

**VLM labeling already uses gender** in the prompt (opposite-sex framing when mixed; we use same-gender pairs in v1). The **consensus winner** inherits that context; the stored training row is still `(photoA, photoB, winner)`.

**Why not feed `overall_score` into training?** That would teach the model to reproduce the old formula, not human/VLM pairwise truth. We only use Labs scores to **pick a balanced sample** and **validate** BT afterward.

---

## 9. Phase 2+ (deferred)

Pointwise photo → /10 regression and feature ablation (landmarks, ratios, VLM assessments). Not blocking Phase 1. Do not use Labs pillar scores as inputs when training a replacement model.

---

## 10. Key decisions

| Question | Decision |
|----------|----------|
| Primary path | Pairwise preference model + anchor ladder |
| Pool / GT size | **~30k pool**, **3k active**, **35 comps/face**, **~52.5k pairs** |
| VLM labeling | ✓ Pilot passed; **3-model consensus** + prompt iteration |
| Crowdsourcing QC | Fallback if consensus accuracy insufficient; ~10 trap pairs |
| GT source | Bradley–Terry refit on **consensus** outcomes |
| /10 mapping | Percentile → fixed calibration curve (median ≈ 5) |
| Explicit face scores | **No** — pairwise only; Labs score for stratification only |
| Corrections | Update comparison row → full BT refit (no cascade) |
| **Prompt audit set** | Keep **original 40-pair pilot run** for prompt A/B — do not re-draw from 20k each time |
| **20k parent pool (now)** | Browse / validate only — **no scale labeling** on full pool |
| **3k active cohort** | Stratified auto-draw → **manual curation UI** → lock → then pair queue (§16) |
| **Pool quality** | Expect Labs-score bias + dupes + synthetics — curation fixes what stratification cannot |

---

## 11. Workflow summary (pairwise-first)

```
Frozen face pool (~10k–50k, stratified export)
        │
        ▼
Sub-sample active cohort (~2k–5k faces, seeded + stratified — §15)
        │
        ▼
┌─── VLM PILOT ─── ✓ DONE ─────────────────────────────┐
│  40 pairs, Flash models 76–86%, $/pair validated       │
└──────────────────────────────────────────────────────┘
        │
        ▼
Prompt experiments (pairwise-v3, v4, …) on held-out audit set
        │
        ▼
Full labeling: ~20k–50k matchups × 3-model panel → consensus
        │
        ├── Human review (splits, low confidence — ~5–15%)
        │
        ├── Bradley–Terry refit → scoreOutOf10 + anchor set (50–100)
        │
        └── Export matchup dataset → notebook
                │
                ├── Train preference model (pairwise loss)
                │
                └── Validate vs BT rankings + anchor-ladder inference
```

The end goal: replace the hand-tuned overall formula with a preference model trained on pairwise ground truth, deployed via anchor comparisons for explainable /10 ratings.

---

## 12. Reference notes

### Matchup math

| | Formula | N=1000 example |
|--|---------|----------------|
| Target pairs | `(N × k) / 2` | k=20 → **10,000** VLM calls |
| Full round-robin | `N × (N − 1) / 2` | **499,500** (skip) |

### BT vs online Elo

- **Online Elo** — UI preview during labeling only  
- **BT refit** — authoritative GT from all final comparison outcomes  
- Override a matchup → update row → **full refit** (seconds, no cascade)

### Calibration curve (example anchor points)

| Percentile | scoreOutOf10 |
|------------|--------------|
| 50th | 5.0 |
| 90th | 7.0 |
| 99th | 8.0 |
| 99.9th | 8.5 |

Fix curve before the experiment. Inject ~10 tail **anchor faces** into cohort — a 1k random sample alone won't observe true 9–10 extremes.

### VLM pilot

✓ **Complete.** See `vlm-pilot-spec.md` §14. Full GT labeling ✓ 2026-07-03. Next: export + BT in `faceiq-preference-ml`.

### Crowdsourcing QC

~10 gold-standard trap pairs per session; discount or drop raters who fail them.

### Known limits

- Photo quality (lighting, filters) confounds A/B  
- Gender stratification policy TBD  
- Randomize A/B left-right presentation (position bias)

### Minimal audit log

Seed, pair list, model + prompt ids, each comparison outcome, BT refit timestamp, export hashes.

---

## 13. Scale phase — status & what to build next

**Completed (2026-07-04):**

1. ✓ **Sample curation** — approved 3k sample `cmqz06xr20001bydxaazuojp8`  
2. ✓ **Pair queue** — k=35, `bt-stratified-v2`, ~52.5k pairs  
3. ✓ **VLM batch at scale** — single model `gemini-2.5-flash`, prompt **`pairwise-v3-gt`**, run `cmr1mr0m7000196d57zi3vcgn`  
4. ✓ **Human audit gate** — 750-pair seeded sample, **84.9%** accuracy; **86** matchups excluded from export  

**Now (in order):**

5. **Export finalized matchups** — paginated `GET /api/admin/pairwise/runs/[runId]/export` → local `~/research-data/scoring-gt/artifacts/` (excludes `gtExportExcludedAt` rows automatically)  
6. **BT refit + stability** — in **`faceiq-preference-ml`** (Python); label rule: `humanWinnerFaceId` if `humanLabeledAt`, else `vlmWinnerFaceId`; optional high-confidence filter  
7. **Calibration** — map BT percentile → pre-registered `/10` curve (§12 calibration table)  
8. **Validate** — Spearman ρ vs Labs `overall_score`; 80% subsample stability (ρ > 0.95)  
9. **Train preference model** — same export rows; split by face id (§8)  
10. **(Optional later)** Admin BT persist — schema §3.3 `VlmPilotBtRefit` + dashboard  

**Deferred:** 3-model panel consensus at scale (157k calls) — single-model GT path chosen 2026-06-29.  

---

## 14. How many faces and matchups?

### Two jobs, one dataset

Every labeled matchup serves **both** purposes:

| Purpose | What you need | Label |
|---------|---------------|-------|
| **Bradley–Terry ranking** | Sparse connected graph; enough edges per face for stable θ | Consensus winner per pair |
| **Preference model training** | Many diverse `(photo A, photo B, winner)` rows | **Same winner** — the matchup outcome **is** the training label |

You do **not** need every face to play every other face for either job. BT infers global order from partial observations (transitivity: if A beats B and B beats C, A is inferred above C without a direct A–C game). The NN learns from the same win/loss rows; more rows = better, but they don’t have to form a complete graph.

### Core formula

```text
pairs = (N × k) / 2

N = faces in active cohort
k = average comparisons per face
```

### How many comparisons per face (k)?

| k | Role | When to use |
|---|------|-------------|
| **~15** | Floor | Minimum for BT to run; high variance per face |
| **~20** | Baseline | Original plan; workable but thin at tails |
| **~25–30** | Strong | Good BT stability + training volume |
| **~35** | **Default v1** | **3k × 35 — our starting plan** |
| **~40** | Ambitious | Clearer tier separation |
| **50+** | Diminishing returns | Small BT gain; mostly extra training rows — only if budget is trivial |

**Rule of thumb:** pick **k** from ranking needs first; training data scales automatically (`pairs = N×k/2`).

For **3k faces at k=35** → **~52.5k pairs** → **~157k VLM calls** (3-model panel) → **~$400–600** Flash.

### Why k=20 can feel “thin”

A face with 20 random opponents might never directly play the #1 face — but BT still places it correctly if it:

- Plays someone who played someone in the top tier (graph path), and  
- Gets **cross-bin pairs** in the queue (see §15 — ~30% of each face’s games across score deciles).

Raising **k to 30** and enforcing **cross-decile pairing** fixes “never matched against the best” better than raising **N** alone.

### Full round-robin?

**No.** For N=3k, all pairs = **4.5M**. Unnecessary. Sparse graph + BT is the whole point.

### Face count (N) vs matchup count

| Plan | N | k | Pairs | Panel VLM calls | Notes |
|------|---|---|-------|-----------------|-------|
| Minimum | 1k | 20 | 10k | 30k | Cheap; first scale test |
| **Default v1** | **3k** | **35** | **52.5k** | **157.5k** | **Starting plan** |
| More training rows | 3k | 40 | 60k | 180k | Extra NN data, same N |
| Wider face coverage | 5k | 20 | 50k | 150k | Weaker per-face BT |

**Prefer higher k over higher N** when choosing between them (up to ~35).

### Stability check (after labeling)

Before trusting BT export:

- Refit BT twice on random 80% subsamples of pairs → Spearman ρ between rank lists **> 0.95**  
- No face with **< 15** resolved comparisons  
- Graph connectivity: every face reachable from every other via comparison edges  

### Do we need explicit /10 labels on faces?

**No.** The system never asks anyone to score a face 9 or 9.5.

| What | Role |
|------|------|
| **Pairwise outcomes** (A/B/tie) | **Training + BT labels** — the only required human/VLM signal |
| **BT refit** | Produces **relative ranking** θ per face |
| **Calibration curve** | Maps rank **percentile** → `/10` for display (median → 5.0, top 1% → 8.0, etc.) |
| **Labs `overall_score` at export** | **Stratification + validation only** — check Spearman ρ vs BT; not a training target |

### Generalization beyond the sample

The comparator learns **which visible features predict “more attractive” in head-to-head comparisons**, not absolute scores tied to your cohort.

- A face **more attractive than anyone in the sample** still wins vs anchors if it exhibits the same feature directions (symmetry, harmony, skin clarity, etc.).
- You do **not** need a literal “9.5” in the dataset if the **high tail** is represented: very attractive faces that consistently beat most others. BT rank + calibration curve assigns them a high `/10` at deploy time.
- Risk: if the sample **never** includes highly attractive faces, the model may compress the top of the scale. Fix with **tail coverage in sampling** (§15), not manual score labels.

---

## 15. Sampling plan — faces, spread, pairs

### End-to-end checklist

1. **Engineer exports parent pool** (~**30k** faces) → Data Lab zip per §2  
2. **Quality filter** — drop missing front image, invalid landmarks, obvious non-front  
3. **Stratified draw** → **3k active cohort** (algorithm below)  
4. **Audit histogram** — decile counts per gender; fix before labeling  
5. **Generate pair queue** — **k=35**/face, **~52.5k pairs**, cross-bin quotas, deterministic seed  
6. **Label** — 3-model VLM panel → consensus; human review on splits  
7. **BT refit** → export rankings + **matchup training JSON** (same winners)  

---

### Step 1–2: Parent pool filters

- Front photo exists and passes basic quality  
- **`frontLandmarks` present** — optional (we can re-landmark from photos)  
- Optional: `createdAt` after auto-landmark rollout date  
- Store full manifest row + landmarks in `exportPayload`  

---

### Step 3: Stratified draw (how it works)

**Goal:** active cohort mirrors attractiveness spread; **not** uniform random.

**Bins:** within each **gender** separately, split pool by Labs `overall_score` into **10 deciles** (use quintiles if pool < 5k per gender).

**Target active size:** **N = 3,000** (1,500 M + 1,500 F, or your policy).

**Per-gender allocation** (example for 1,500 faces):

| Decile | Labs score slice | Target % | Faces |
|--------|------------------|----------|-------|
| D1 | Bottom 10% | 13% | ~195 |
| D2 | 10–20% | 10% | ~150 |
| D3–D4 | 20–40% | 18% | ~270 |
| D5–D6 | 40–60% | 22% | ~330 |
| D7–D8 | 60–80% | 18% | ~270 |
| D9 | 80–90% | 10% | ~150 |
| D10 | Top 10% | **13%** | ~195 |

**Oversample tails (D1, D10) by ~30%** vs uniform — ensures attractive and unattractive faces exist for BT and cross-tier matchups.

**Draw procedure (reproducible):**

```text
1. Filter pool → quality-passed faces
2. Split by prod_gender
3. Assign each face a decile D1..D10 from overall_score (gender-specific quantiles)
4. For each (gender, decile) cell:
     n = target_count[decile]
     sample n faces uniformly at random WITH seed (e.g. sha256("gt-cohort-v1" + gender + decile))
5. Union → active cohort (~3k)
6. Write manifest: cohort_id, seed, per-cell counts
```

**Optional manual inject:** add 20–50 curated high-quality faces into D10 before draw if prod top decile is thin.

**Do not** manually assign 9/9.5 scores — decile binning from Labs score is enough for **sampling only**.

---

### Step 4: Pre-label audit

Before generating pairs, confirm:

- [ ] Each gender has **≥100 faces in D1 and D10**  
- [ ] No decile empty  
- [ ] Race/ethnicity distribution logged (no single group > 80% unless intentional)  
- [ ] Spot-check 30 random thumbnails per decile  

---

### Step 5: Pair queue generation

**Target:** **k = 35** comparisons per face → **~52.5k pairs** for N=3k.

**Mix (per face, ~35 games):**

| Pair type | Share | Count |
|-----------|-------|-------|
| **Cross-decile** | **~30%** | ~10 |
| **Within-decile** | ~40% | ~14 |
| **Uncertainty** (after ~10 online games) | ~30% | ~11 |

**Hard constraints:**

- Graph **connected** (minimum spanning tree + extra edges)  
- No duplicate pairs  
- **v1: same-gender pairs only** (M–M, F–F) — see below  
- Deterministic from seed — same seed → same queue  

**You do not need every face vs every top face.** Cross-decile quota + BT transitivity handles tier placement.

### Gender pairing — v1 same-gender, why, and what we defer

**v1 default:** same-gender matchups. **Separate BT rankings** per gender (≈1.5k M + ≈1.5k F in the 3k cohort). Export pool stays ~50/50 M/F with decile stratification **within each gender**.

**Why same-gender for v1**

- One clear question per pair → less VLM/human noise, better consensus  
- Dimorphic cues differ by gender; mixed pairs blend incompatible signals  
- Matches the simplest deploy path: user vs same-gender anchors  

**Cross-gender is valid but a different task**

People do judge “who’s more attractive” across genders (e.g. in a couple). That signal could teach traits that transfer within gender. Mixed pairs are **deferred** because they require an explicit frame (“general appeal” vs “as perceived by X”) and a single BT graph across genders is hard to interpret.

**Pool size concern (1.5k vs 3k)**

- Active cohort is 3k total, but **ranking runs per gender** (~1.5k each) — same as stratifying a 1.5k study twice  
- The **30k parent pool** is what gives decile spread; the 3k draw is 150 per gender per decile (with tail oversampling), which is enough for k=35  
- A unified 3k pool with cross-gender pairs is a **future option** (mixed anchor ladder, equal M/F anchor set) — not v1  

**Deploy note:** anchor comparison does not have to stay same-gender forever. A mixed anchor panel (balanced M/F, tail coverage) is compatible with same-gender **training** data; revisit after v1 ranks exist.

---

### “Really attractive” faces

You do **not** need to fix anyone at 9.0 or 9.5. You **do** need faces that **would win most pairwise matchups** in the cohort:

- **Automatic:** top decile by Labs score + manual spot-check 20–50 thumbnails  
- **Manual inject (optional):** 20–50 curated high-quality reference faces (celebrity/snapshot pipeline, internal picks) added to pool as **anchor candidates** — still labeled only via pairwise, not point scores  
- **Validation:** after BT refit, top 1% by θ should correlate with Labs top decile (ρ check)

---

### Summary numbers (default plan)

| Parameter | Value |
|-----------|-------|
| Parent pool | **~30k** faces (min 20k) |
| Active cohort | **3k** faces (stratified by gender × decile) |
| Comparisons / face | **35** |
| Total pairs | **~52.5k** |
| VLM panel calls | **~157.5k** (3 models) |
| Est. Flash cost | **~$400–600** |
| Training rows | **~52.5k** `(A, B, winner)` tuples (same labels as BT) |

---

## 16. Active sample curation (3k from parent pool)

**Decision (2026-06):** The parent pool (~20k) is **not** labeled directly. We draw a **3k active sample**, **curate it manually** in admin, **lock** it, then generate the ~52.5k pair queue only on the approved sample.

### Why manual curation is required

Automatic stratified sampling (§15) is the **starting point**, not the final word. Real pools often have issues stratification alone cannot fix:

| Issue | Example | Curation action |
|-------|---------|-----------------|
| **Labs score bias** | Pool skews attractive (overall_score inflation) | Swap in lower-decile faces from pool; oversample D1 manually |
| **Duplicate / near-duplicate faces** | Same person, different analysis | Remove one; optionally flag `duplicateOf` |
| **Synthetic / AI / anime** | Non-photographic faces | Exclude; pull replacement from same decile cell |
| **Bad quality** | Heavy filter, non-front, obscured | Exclude; fill from pool |
| **Thin lower tail** | Few genuinely low-attractiveness faces | Manually add from D1–D2 browse |

Attractive over-representation in the parent pool is **acceptable** for the pool itself — the 3k active set must still **cover the decile grid** after curation (especially D1 and D10).

### Workflow

```text
Parent cohort (~20k) — import only, never scale-label
        │
        ▼
Auto stratified draw (seeded) → draft SampleSet (~3k target)
        │
        ▼
Curation UI: review grid by decile / gender / flags
        │     remove · add from pool · swap · fill cell gaps
        │     mark exclusions (reason: duplicate | synthetic | quality | other)
        ▼
Histogram + checklist pass (§15 step 4)
        │
        ▼
Approve & lock SampleSet → only then generate pair queue + VLM batch
```

**Rules:**

- **Do not** generate scale pairs or run VLM batch on the full 20k cohort.  
- **Do not** assign manual /10 scores — pairwise labeling only after lock.  
- Replacements are drawn from the **same parent cohort**, same gender, same target decile cell when possible.  
- Every add/remove is logged (who, when, reason) for reproducibility.  
- Once **approved**, the sample is frozen; changes require a new sample-set version or explicit unlock.

### Admin UI — dedicated curation route

**Decision:** Build curation as a **separate route**, not an overload of the run/pair flow.

| Route | Purpose |
|-------|---------|
| `/admin/pairwise` | Cohort list, import |
| `/admin/pairwise/cohorts/[cohortId]` | Pool overview, create **pilot** runs (≤80 pairs) |
| **`/admin/pairwise/cohorts/[cohortId]/sample`** | **Sample curation** — draw, edit, approve 3k active set |
| `/admin/pairwise/runs/[runId]` | Labeling, VLM batch, review (on **approved sample** runs only) |

The cohort page stays for pool browsing and small pilot runs. The **sample** page owns decile histograms, exclusion queue, add-from-pool picker, and the **Approve for labeling** gate.

### Prompt testing vs scale (do not conflate)

| Phase | Cohort | Pairs | Action |
|-------|--------|-------|--------|
| **Prompt testing (now)** | Original **52-face pilot** cohort | **40 fixed** (80 max) | Prompt A/B on same audit pairs |
| **Optional extra audit** | 20k pool | One **fixed** 40–80 run, locked seed | Only if pilot audit is too narrow |
| **Scale GT** | **Approved 3k sample** | **~52,500** | After curation UI + cap raise |

You do **not** need to run 40–80 pairs from the 20k pool for prompt testing unless you deliberately want a second, locked audit set. Prefer the original pilot run for comparability.
