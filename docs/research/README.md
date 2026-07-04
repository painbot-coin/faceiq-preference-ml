# Scoring GT — Research Docs

**Ignore:** [`scoring-gt-plan.md`](./scoring-gt-plan.md) — deprecated; use [`scoring-gt-core.md`](./scoring-gt-core.md) instead.

**ML training code:** separate repo `faceiq-preference-ml` (see [`scoring-gt-training.md`](./scoring-gt-training.md)) — not in faceiq-labs.

---

## Doc maintenance (after every implementation)

When you ship a feature, migration, or research gate, **read the relevant docs first**, then **update their stated status** so nothing stays “planned” or “TBD” after it exists in code.

| If you changed… | Update |
|-----------------|--------|
| Prisma models / migrations | [`scoring-gt-schema.md`](./scoring-gt-schema.md) — §1 model list, relationship diagram, field tables; §2 phase sufficiency; §3 section status; §6 migration checklist |
| Admin UI routes or API | [`vlm-pilot-spec.md`](./vlm-pilot-spec.md) — §6 routes, §8 API table |
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

## Focus **right now** (post–full labeling + audit gate passed)

**Phase 3 labeling + §4.3 audit:** ✓ complete (2026-07-04). Run `cmr1mr0m7000196d57zi3vcgn` — **52,414** export-eligible pairs, **84.9%** audit accuracy on 750-pair sample.

| Step | Where | Doc |
|------|-------|-----|
| **1. Export matchup JSON** | **faceiq-labs** — paginated API | `vlm-pilot-spec.md` · run export route |
| **2. BT refit + stability + /10 calibration** | **`faceiq-preference-ml`** (Python) | `scoring-gt-training.md` · core §6–§7 |
| **3. Validate vs Labs `overall_score`** | ML repo notebooks | research log §5.2 |
| **4. Train preference comparator** | **`faceiq-preference-ml`** | `scoring-gt-training.md` |
| **5. (Optional) Persist BT to DB** | faceiq-labs admin | schema §3.3 — not built yet |

| Role | Document | Why |
|------|----------|-----|
| **Primary — read for “what next”** | [`scoring-gt-training.md`](./scoring-gt-training.md) + [`scoring-gt-core.md`](./scoring-gt-core.md) **§6–§8, §15 step 7** | Export → BT → train |
| **Write as you go** | [`scoring-gt-research-log.md`](./scoring-gt-research-log.md) **§5+** | BT results, training summary |
| **Export API reference** | [`vlm-pilot-spec.md`](./vlm-pilot-spec.md) | `GET …/export` |
| **DB / future BT tables** | [`scoring-gt-schema.md`](./scoring-gt-schema.md) §3.3 | When persisting ratings back to admin |

**Do not** start another VLM batch on this run unless re-labeling. **Do** scaffold `faceiq-preference-ml` as a sibling repo and copy exports to `data/exports/`.

**Completed (no longer focus):** 3k curation (`cmqz06xr20001bydxaazuojp8`), pair queue k=35, full 52.5k VLM batch, human audit gate.

---

## What each doc is for

| Document | Role | When to open it |
|----------|------|-----------------|
| [`scoring-gt-core.md`](./scoring-gt-core.md) | **North star** — scale, sampling, BT, consensus, phases | Planning any work block; re-read §13 before starting a new phase |
| [`scoring-gt-research-log.md`](./scoring-gt-research-log.md) | **Notebook** — what you did, numbers, go/no-go | **During and after** each step (§9 gates); fill §1 once at kickoff |
| [`vlm-pilot-spec.md`](./vlm-pilot-spec.md) | **Admin + VLM ops** — import, label queue, batch, results | Using `/admin/pairwise`; scale batch prep (§15) |
| [`scoring-gt-schema.md`](./scoring-gt-schema.md) | **DB map** — models, migrations, sample-set API | Implementing Prisma/admin features; “where does X live?” |
| [`scoring-gt-training.md`](./scoring-gt-training.md) | **Offline ML only** | BT export copied to `faceiq-preference-ml` |

**Rule:** One **primary** doc per phase (core or training). **Log** is always a sidecar. **Schema** is lookup, not sequential reading.

---

## Phase-by-phase: primary doc → move on when

| Phase | Primary doc | Also use | Log (fill when done) | Schema (when) |
|-------|-------------|----------|----------------------|-----------------|
| **0 — Prompt & consensus** ✓ pilot done | `core` §5.1, §13 #1–3 · `vlm-pilot-spec` §6–§9 | `research-log` §2.2–§2.3 | §2.1 archive pilot · §2.2 prompt chosen · §2.3 panel validated | **§3.1** before wiring consensus at scale |
| **1 — Pool import** | `core` §2, §13, §15–§16 | `vlm-pilot-spec` §3 import | §3.1 pool QC | Usually **no migration** (existing cohort tables) |
| **2 — 3k curation + audit** | `core` §16 | Pairwise Lab **`/research/pairwise/cohorts/[id]/sample`** | §3.2 curation · §3.3 audit pass | **§3.2** ✓ (`20260629180000_vlm_pilot_sample_set`) |
| **3 — Labeling (52.5k)** ✓ | `core` §14–§15 · `vlm-pilot-spec` §15 | Admin batch + Review | §4.1–§4.3 ✓ | export exclude migration; review fields |
| **4 — BT + calibration + export** ← **now** | `core` §4, §7, §15 step 7 · **`scoring-gt-training.md`** | Export API | §5.1–§5.2 | §3.3 BT tables optional |
| **5 — NN training** | **`scoring-gt-training.md`** | `core` §8 | §5.3 | ML repo only |
| **6 — Primary human eval** | `research-log` **§6** | `core` for context | §6 full write-up | — |

---

## Quick decisions

| Question | Open |
|----------|------|
| What should I build / do this week? | **`scoring-gt-training.md`** — export → BT → smoke train |
| How does the admin UI / batch work? | `vlm-pilot-spec.md` (labeling done; use export route) |
| Can we run 157k panel calls yet? | **N/A** — single-model GT run complete; panel deferred |
| Which Prisma model / migration next? | `scoring-gt-schema.md` §3.3 (optional BT persist) |
| What do I write down after a run? | `research-log.md` §9 gate row → matching §2–§6 |
| How do I train the comparator? | `scoring-gt-training.md` (only after export) |

---

## Doc index

| Document | Purpose |
|----------|---------|
| [`scoring-gt-core.md`](./scoring-gt-core.md) | Canonical plan — phases, 3k × 35, BT, VLM consensus |
| [`scoring-gt-research-log.md`](./scoring-gt-research-log.md) | Living log — record at gates; artifact pointers |
| [`vlm-pilot-spec.md`](./vlm-pilot-spec.md) | Pairwise admin + pilot (done) + §15 scale batch |
| [`scoring-gt-schema.md`](./scoring-gt-schema.md) | DB map + migrations before scale |
| [`scoring-gt-training.md`](./scoring-gt-training.md) | Offline preference model (separate repo) |
