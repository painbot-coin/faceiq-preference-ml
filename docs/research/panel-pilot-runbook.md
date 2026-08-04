# Panel pilot — runbook (do-this-in-order checklist)

*2026-07-25 · The executable companion to
[`archive/panel-pilot-execution-plan.md`](./archive/panel-pilot-execution-plan.md) (what & why, now
archived) and [`archive/rating-app-build-plan.md`](./archive/rating-app-build-plan.md) (the app, now
built). This file is the "come back to it and keep going" checklist: exact commands, decision rules,
and handoff points.*

> **Still current for ops, but read the newer docs for policy.** This runbook was written around
> runs 1–3. Run 4 (uniform random pairs, study `6a71110e4d5c6eba96c17d21`) closed 2026-08-04 — its
> outcome is in Stage 5 below and log **§5.8**. For **which pairs to buy** use
> [`panel-study-playbook.md`](./panel-study-playbook.md) §2a — the answer changed on 2026-08-04, nothing
> above a 20-point percentile gap is ever worth buying. For **Prolific fields and the launch
> sequence** use [`prolific-soft-launch-form.md`](./prolific-soft-launch-form.md) §9, whose §9.5 now
> carries the full pull-and-analyse command sequence. The stage-by-stage
> mechanics below (draw → ship → deploy → `/?test=1` → monitor → archive) are unchanged.
>
> **There is no run 5 queued, on purpose.** The only validated spend left is $7,463 inside the 0–20 band,
> and it is blocked until the comparator can match the ranking it is distilled from (playbook §2b).

Repos involved:

| Repo | Role |
|---|---|
| **faceiq-labs** | canonical DB. All metadata *fixes* and the GT re-export happen here. |
| **faceiq-preference-ml** (this repo) | QC pass, pair sampling, answer keys, analysis. |
| **faceiq-rating** | the rater-facing web app. Receives 3 JSON files, returns judgments. Live: `https://faceiq-rating.vercel.app` (production alias — **not** a per-deployment URL, see Stage 5) |

---

## Stage 0 — Face metadata QC  ✅ DONE 2026-07-25

Full pass complete: **2,998 / 3,000** scored (2 API errors — rerun the same
command to retry those two), **508 flagged (16.9%)** for human review. Outputs in
`artifacts/face-qc-v1/`:

| Flag | count | action |
|---|---:|---|
| `possible_minor` | 307 | hard exclude if confirmed (review every one) |
| `screenshot` | 82 | exclude |
| `not_real_photo` | 66 | exclude |
| `gender_mismatch` | 54 | relabel + drop edges |
| `gender_ambiguous` | 21 | human decides |
| `multiple_faces` | 21 | exclude |

Much higher than the "few dozen" the execution plan assumed — budget an afternoon
for flag review, not an hour. Contact sheets: `artifacts/face-qc-v1/review/*.png`.

Runs **here**, read-only against the export. Fixes are applied in faceiq-labs.
VLMs are only asked objective questions (gender, real photo, age band, usability,
ethnicity, skin tone) — never anything about attractiveness.

```bash
source .venv/bin/activate
uv pip install -e '.[vlm]'          # google-genai, first time only

# smoke test (40 faces, ~25s, pennies)
python scripts/qc_faces.py --export data/exports/cmr1mr0m7000196d57zi3vcgn \
    --env-file ../faceiq-labs/.env --limit 40 --out artifacts/face-qc-smoke

# full pass (3,000 faces, ~15-25 min, a few dollars) — resumable, rerun to retry errors
python scripts/qc_faces.py --export data/exports/cmr1mr0m7000196d57zi3vcgn \
    --env-file ../faceiq-labs/.env --out artifacts/face-qc-v1 --workers 8

# contact sheets of flagged faces only, one PNG per reason
python scripts/qc_review_sheet.py --qc artifacts/face-qc-v1 \
    --export data/exports/cmr1mr0m7000196d57zi3vcgn
```

**Env var needed:** `GEMINI_API_KEY` only, and it stays where it is —
`--env-file ../faceiq-labs/.env` reads it at run time, so no secret is copied
into this repo and there is nothing to gitignore here. (`GOOGLE_GENERATIVE_AI_API_KEY`
and `GOOGLE_AI_API_KEY` also work.) This script calls the Google API **directly**
via `google-genai`; labs' AI-gateway wrapper in `src/lib/vlm-precompute/client.ts`
is TypeScript app plumbing for retries/fallbacks at scale — and for VLM
precompute (the pass that produced the 52k matchup labels) it already goes
**direct-first** when `GEMINI_API_KEY` is set, per its own comment: "the
gateway's `google/gemini-3-flash` slug is intermittently 503 while the direct
`gemini-3-flash-preview` endpoint is reliable." So direct is what the matchups
actually ran on; a one-off 3,000-call QC pass has no reason to differ.

Outputs in `artifacts/face-qc-v1/`: `results.jsonl` (one row per face),
`flags.csv` (review list), `summary.json` (counts + distributions),
`review/*.png` (contact sheets).

**Check progress while it runs:**

```bash
wc -l artifacts/face-qc-v1/results.jsonl    # target: 3000
# when finished, summary.json and flags.csv appear alongside results.jsonl
```

The run is **resumable** — if it stops, rerun the same command; faces already
in `results.jsonl` are skipped.

**Cost:** one Gemini Flash call per face ≈ **$3–8 total** for 3,000 faces
(image + short JSON prompt). Pennies for a 40-face smoke test. The key is read
from `../faceiq-labs/.env` via `--env-file`; top up Google AI / Cloud billing
there if calls start failing with quota or billing errors (the script will show
API errors in `results.jsonl`). Two transient errors so far in the current run;
rerun picks up anything missing.

### Triage rules — what to do with each flag

| Flag | Decision | Why |
|---|---|---|
| `possible_minor` | **Hard exclude the face** (and its edges). No judgement call. | Ethics/legal — we cannot send a minor's photo to paid third-party raters. Prompt errs younger on purpose; a human confirms each one. |
| `not_real_photo` | Exclude via the existing `synthetic` exclusion mechanism. | 16 were removed at curation, pilot-500 surfaced 1 more. |
| `screenshot` / `multiple_faces` | Exclude via `quality`. | Photo-of-a-screen or two faces makes "which face?" undefined. |
| `gender_mismatch` (VLM ≠ label) | **Both**: correct the gender in labs, *and* drop that face's existing comparison edges from GT. | See below — relabelling alone does not fix the damage. |
| `gender_ambiguous` | Human decides: keep with corrected label, or exclude. | Usually a styling/angle artefact. |
| quality notes (`obstruction`, `poor_lighting`, `low_resolution`, tight crop) | **Ignore** — recorded in `results.jsonl`, deliberately not flagged. | Raters are instructed to look past photo quality; excluding these would bias the cohort toward good photography. |

### Why a gender mislabel means dropping edges, not just relabelling

Matchups were generated *within* the labelled gender. A face labelled male that
is actually female was therefore compared against ~35 males — so those rows are
in reality cross-gender comparisons, which is exactly what the design excludes.
Flipping the label fixes future matchup generation but cannot retroactively fix
who the face was already compared against, so:

1. Correct `gender` in faceiq-labs (matters for any future draw).
2. Exclude that face's existing edges from the GT export.
3. Accept that the face drops out of BT for now (it will fall under the
   15-comparison gate). BT and the comparator are robust to losing ~35 edges.
4. If we later generate new matchups adaptively (execution plan Step 7), the face
   re-enters naturally, now correctly pooled.

**Materiality rule:** if under ~2–3% of faces are affected, fix and move on. If
much worse, stop and discuss re-drawing the cohort.

### What the 40-face smoke test suggests (calibration, not a result)

6 of 40 flagged: 2 possible minors, 1 gender mismatch, 1 not-real-photo,
2 screenshots. Gender agreed on 39/40. If that rate holds, expect **several
hundred** flags over 3,000 faces — materially more than the "few dozen" the
execution plan assumed, so budget an afternoon for review, not an hour. Ethnicity
and skin-tone distributions come out of the same pass for free.

### Also needs checking: duplicate faces

pilot-500 turned up one pair where both raters wrote "same person" —
`cmr1mr15000q396d5crhkyclk`. A duplicate face is two BT nodes for one person,
which corrupts the ranking. The per-face VLM pass cannot catch this (it only
sees one image at a time); it needs an embedding pass over all 3,000 faces
flagging near-duplicate pairs. **Not yet written** — the `.[arcface]` extra is
the intended tool.

## Stage 0.5 — Ethnicity, from the Labs manifest  ✅ DONE 2026-07-25

Self-declared ethnicity turned out to be recoverable **entirely offline** — no
prod DB, no re-export, no VLM. The Labs face manifest
(`ab-data/manifest.part-NN-of-05.jsonl`, 5 shards × 4,000 rows = the 20k pool
the 3k cohort was drawn from) carries `prod_race` and `prod_gender`, and its
`analysis_id` is exactly the export's `sourceFaceId`:

```bash
python scripts/join_face_attributes.py \
    --export data/exports/cmr1mr0m7000196d57zi3vcgn \
    --manifests ~/Downloads/ab-data --out labels
```

**Result: 3,000/3,000 faces matched (100%), every one with a non-empty
ethnicity, 0 disagreements between `prod_gender` and the export's `gender`.**
White is 69.2% — the same "69% white" quoted in the execution plan, which
confirms this is the same source data, and it sits under the ≤80% single-group
audit gate.

The join is **additive and read-only**: it reads `faces.jsonl` + the shards and
writes one new file, `labels/face-attributes.json` (`faceId ->
{sourceFaceId, ethnicities[], prodGender}`). The export, the pair rows and BT
inputs are untouched, so there is nothing to lose. Gitignored — pseudonymous
demographic data about real users, and regenerable in a second.

`prod_race` is an **array**: mixed-ethnicity users declared more than one
(`east_asian+white`, `black+white`), which a single VLM guess would have
flattened. Values already use the labs `Race` vocabulary. One face is
`prefer_not_to_say` and one is `other` — treat both as unknown.

Faces per group per gender (primary declaration), which is what the Stage 2
ethnicity floor has to work with:

| group | male | female | total |
|---|---:|---:|---:|
| white | 1044 | 1033 | 2077 |
| hispanic | 112 | 130 | 242 |
| east_asian | 78 | 118 | 196 |
| black | 84 | 95 | 179 |
| middle_eastern | 92 | 71 | 163 |
| south_asian | 81 | 47 | 128 |
| native_american / pacific_islander / other | 8 | 6 | 14 |

Each face sits in ~35 pairs, so ~50 faces per gender is enough to support 150
pairs. Six groups clear that; **south_asian female (47) is borderline** and the
native_american/pacific_islander tail (14 faces total) cannot support a floor at
all — pool it as "other" and report it, rather than forcing a quota.

### Review results  ✅ DONE 2026-07-26 — all 508 triaged

| Action | faces |
|---|---:|
| keep | 375 |
| exclude | 71 |
| relabel gender | 62 (51 →male, 11 →female) |

Per group: `possible_minor` 254 keep / 31 exclude (the prompt errs young as
designed), `not_real_photo` 32/33, `screenshot` 62 keep / 0 exclude (that flag
produced nothing actionable — drop or loosen it next pass), `gender_mismatch`
all 54 relabelled, `multiple_faces` 15 keep / 2 exclude / 4 relabel,
`gender_ambiguous` 12 keep / 5 exclude / 4 relabel.

**Impact on the comparison graph** (`python scripts/qc_handoff.py --qc
artifacts/face-qc-v1 --export data/exports/cmr1mr0m7000196d57zi3vcgn`):

| | |
|---|---|
| Faces losing all edges (exclude + relabel) | 133 (**4.4%** of cohort) |
| Cohort after exclusions | 2,929 faces |
| Pairs dropped | 4,500 of 52,414 (**8.6%**) |
| Pairs remaining | 47,914 |
| Surviving faces under the 15-comparison gate | 63 (62 of them at zero) |

4.4% is above the execution plan's 2–3% materiality rule but not "much worse",
and it splits into 2.4% genuinely excluded plus 2.1% metadata relabels — the
cohort is not rotten, gender labelling was noisy. Proceed, and note it in
research log §5.

**Reading the §5.1 gate correctly afterwards:** the 62 relabelled faces stay in
`faces.jsonl` but have zero comparisons by construction, so a naive "no face
under 15 comparisons" check will fail. Evaluate the gate over faces that have at
least one comparison; the 62 are expected dropouts and re-enter only if Step 7
generates new matchups for them.

### Where to review the flags

```bash
streamlit run app/qc_review.py --server.port 8503
```

Grid of flagged faces, one group at a time, each card showing the photo plus the
VLM's verdicts (export label vs apparent gender, age band, issues). Four buttons
per face: **Keep**, **Exclude**, **→M**, **→F**. Decisions append to
`artifacts/face-qc-v1/decisions.jsonl` — resumable, quit and come back.

The static `review/*.png` contact sheets are for skimming a whole group at once;
the Streamlit app is where decisions actually get recorded.

When finished, hit **Write handoff files** in the sidebar. That emits the two
lists faceiq-labs consumes:

| File | Contents |
|---|---|
| `exclude-faces.csv` | `faceId, labsReason` — reason mapped to labs' vocabulary (`synthetic` \| `quality` \| `duplicate` \| `other`) |
| `gender-fixes.csv` | `faceId, correctGender` |

## Stage 1 — Fix in labs and re-export  (faceiq-labs)

1. Apply the Stage 0 decisions: correct genders, mark excluded faces (the
   exporter skips pairs with `gtExportExcludedAt` set).

   **Where the data lives.** The faceiq-labs *repo* holds no data — but its
   `.env` has `DATABASE_URL` pointing at the prod Postgres (`db.prisma.io`), so
   scripts run from your local checkout read and write **prod over the network**.
   That is what `export-gt-run.ts` does. The 5 `manifest.part-NN-of-05.jsonl`
   files are a separate, older file dump of the ~20k source pool (used in Stage
   0.5 for ethnicity); they are not the labs database and cannot be written to.

   **The GT pipeline lives on the `faceiq-scoring-research` branch**, not `main`
   — `scripts/export-gt-run.ts`, `src/lib/vlm-pilot/`, and the `VlmPilot*` models
   only exist there. Check out that branch before running anything (as of
   2026-07-27 `main` was checked out with unrelated uncommitted work in
   `src/lib/side-prenormalization/`).

   The built-in helper `setComparisonGtExportExcluded()` marks **one comparison**
   at a time, but review decisions are per *face*. So
   `scripts/apply-gt-face-decisions.ts` (written 2026-07-27) fans each face
   decision across every comparison it appears in, and applies the gender fixes:

   ```bash
   # dry run first — prints counts, writes nothing
   npx tsx scripts/apply-gt-face-decisions.ts --run cmr1mr0m7000196d57zi3vcgn \
     --excludes ../faceiq-preference-ml/artifacts/face-qc-v1/exclude-faces.csv \
     --genders  ../faceiq-preference-ml/artifacts/face-qc-v1/gender-fixes.csv
   # then, once the numbers match Stage 0's impact report (4,500 comparisons):
   #   ... --commit
   ```

   Expect ~4,500 comparisons excluded and 62 faces relabelled. It writes to
   **prod**, so read the dry-run output before passing `--commit`.
2. **Add `race` to the faces export** while you're in there. `VlmPilotFace.race`
   exists in the labs DB (self-declared, imported from `prod_race`) but
   `scripts/export-gt-run.ts` selects only `gender`, so it never reaches
   `faces.jsonl`. Two-line change, and it makes future exports self-contained —
   but it is **no longer blocking**: Stage 0.5 already recovered ethnicity for
   all 3,000 faces without touching the DB. Precedence stays: self-declared first,
   VLM `apparent_ethnicity` only to fill gaps and cross-check. The VLM's
   `skin_tone_fitzpatrick` is new data either way (skin tone is stored nowhere).
   The QC prompt uses the same vocabulary as labs `Race`
   (`src/types/face.ts`: east_asian, south_asian, black, hispanic,
   middle_eastern, native_american, pacific_islander, white, mixed, other) so
   the buckets line up with the app and with rater strata.
3. Re-run the exporter:
   `npx tsx scripts/export-gt-run.ts --run cmr1mr0m7000196d57zi3vcgn --out <dir>`.
   **It reads the labs prod Postgres** (`import prisma from '../src/lib/database'`
   → `DATABASE_URL` in `faceiq-labs/.env`) — there is no intermediate file, so
   this step needs prod DB access and can only run from that repo. Nothing needs
   to be re-sent to this repo for it.
   **The run ID does not change** — `--run` is an input, and re-exporting simply
   re-reads the current DB state for that run (new counts, new hashes, now with
   `race`). A new run ID would only appear if the VLM labelling itself were rerun.
   Its only purpose now is applying the Stage 0 exclusions; if Stage 0 turns up
   nothing material, the current export stands and Stage 2 can start immediately.
4. Copy the export here, and write a one-paragraph QC note into research log §5.

## Stage 1.5 — Refit BT with the QC exclusions  ✅ DONE 2026-07-27

**Done locally, without touching prod.** `run_bt.py` now takes the two QC lists
and drops every matchup touching those faces, so the cleaned refit does not have
to wait on the labs write or the re-export:

```bash
python scripts/run_bt.py --export data/exports/cmr1mr0m7000196d57zi3vcgn \
    --exclude-faces   artifacts/face-qc-v1/exclude-faces.csv \
    --exclude-genders artifacts/face-qc-v1/gender-fixes.csv \
    --out artifacts/bt-refit-v2-qc
```

Result — `artifacts/bt-refit-v2-qc/`, **all §5.1 gates passed**:

| | female | male |
|---|---|---|
| Faces ranked | 1,422 | 1,444 |
| Graph components | 1 (connected) | 1 (connected) |
| Faces below 15 comparisons | 0 | 0 |
| Stability rho (80% subsample) | 0.9921 | 0.9920 |

47,914 matchups, 2,866 faces ranked. One extra female face
(`cmqxa1yun0lhe7hdpzb92acyj`) dropped as under-connected after the exclusions.
Losing 8.6% of pairs did **not** disconnect the graph or breach any gate, and
rho barely moved from v1's 0.9925 — the ranking is robust to the cleanup.

**Stage 2 can draw against `artifacts/bt-refit-v2-qc/ratings.csv` now.** The labs
write + re-export (Stage 1) is still worth doing so future exports are correct at
source, but it is no longer on the critical path.

### Why the refit is mandatory before drawing

Easy to skip, and it would quietly corrupt the draw. The panel strata are
win-probability buckets (close / mid / clear / …) computed from Bradley-Terry
thetas. Those thetas currently live in `artifacts/bt-refit-v1-cap95/` and were
fit on the **uncleaned** 52,414-pair export. Stage 1 changes that graph:

| Stage 1 action | Effect on BT graph |
|---|---|
| Exclude a face (minor, synthetic, screenshot, …) | Drop **all** of that face's existing matchup edges (~35 each). Face falls under the ≥15-comparison gate and drops out of rankings. |
| Gender mislabel | Correct label in labs **and** drop all existing edges (they were within-gender matchups that are cross-gender in reality). Same ≥15 gate. |
| Relabel only, no exclusion | Edges stay; only metadata fixes. |

We do **not** generate replacement matchups for dropped edges — the pilot draws
from whatever pairs remain in the cleaned export. Losing ~35 edges × a few dozen
bad faces out of 52k is within tolerance (materiality rule: <2–3% of faces).

But the thetas of faces that *survived* still shift, because their opponents
changed. Drawing "close pairs" off stale thetas would put the wrong pairs in
front of raters. So the order is fixed:

1. Stage 1 re-export (fewer pairs, same run ID).
2. **Refit BT here:**
   ```bash
   python scripts/run_bt.py --export data/exports/<runId>
   ```
3. Re-check §5.1 gates on the new fit (graph connected, no face under 15 resolved
   comparisons, 80% subsample Spearman rho > 0.95). The ≥15 gate is the one
   most likely to move.
4. Stage 2 draw reads the **new** `ratings.csv` for buckets/thetas/winProb.

`sample-meta.json` from the draw then carries post-cleaning thetas — the answer
key and the strata stay aligned.

## Stage 2 — Draw the panel sample  ✅ DONE 2026-07-27

`scripts/select_panel_pairs.py` (new; `select_pilot_pairs.py` stays as the
pilot-500 record). Reads thetas from **`artifacts/bt-refit-v2-qc/ratings.csv`**
and drops every pair touching a faceId in `artifacts/face-qc-v1/exclude-faces.csv`
or `gender-fixes.csv` — those pairs are invalid regardless of whether the
labs-side exclusion has been applied yet — plus the three `example-pairs.json`
pairIds, whose answers raters are shown in the instruction sheet.

```bash
python scripts/select_panel_pairs.py --export data/exports/cmr1mr0m7000196d57zi3vcgn
```

Result — 3,000 pairs, seed 20260727, from 2,463 distinct faces:

| stratum | pairs | pool available |
|---------|------:|---------------:|
| close (p < 0.60) | 1,050 | 1,863 |
| mid (0.60–0.80) | 750 | 4,502 |
| clear + very_clear | 600 | 33,266 |
| lowconf (medium/low VLM confidence) | 300 | 6,824 |
| overlap (GT audit + pilot-500) | 300 | 1,446 |

Gender 1,499 M / 1,501 F. Skipped 4,500 QC-excluded, 10 unranked, 3 examples.

Ethnicity coverage (pairs containing at least one face of that group; floor 150):
white 2,736 · hispanic 469 · east_asian 354 · black 327 · middle_eastern 305 ·
south_asian 241. Every floor was met by the natural draw, so the swap-based
top-up never fired — worth knowing, because a swap would have perturbed the
strata slightly.

The **overlap** stratum is deliberately *not* fresh: it re-asks pairs the GT
audit and pilot-500 already answered, which is what makes the panel comparable to
the labels it is meant to replace, and it is where the golds live.

Emits `pairs.json` (blinded — pairId, gender, left/right faceId + photoUrl;
ships) and `sample-meta.json` (answer key: stratum, bucket, winProb, thetas, VLM
label, ethnicity, which side face A landed on — **stays here, gitignored**).

## Stage 3 — Golds and examples  ✅ DONE 2026-07-27

**Golds — 33** (`scripts/make_golds.py` → `golds.json`, `golds-review.json`):

```bash
python scripts/make_golds.py                          # after the draw
streamlit run app/gold_review.py --server.port 8504   # eyeball, then Apply
python scripts/make_golds.py --decisions artifacts/panel-golds-v1/decisions.jsonl
```

A gold is a pilot-500 pair that was unanimous (Dit + Alex + Gemini) **and**
very_clear at high VLM confidence. Unanimity alone is not enough: 63 of the 220
unanimous pairs are *close* pairs where three raters happened to land the same
way, and failing a paid rater on one of those would be indefensible. The strict
rule leaves 37 candidates, of which 33 survived into the draw (3 lost to QC
exclusions, 1 became a worked example). Expected side is stored relative to the
**panel's** left/right, which is shuffled independently of pilot-500 — 13 left,
20 right.

**Review outcome 2026-07-29: 13 rejected, 20 golds ship.** Raters see 5 golds each
(`GOLD_FRACTION = 0.05`) drawn from those 20, so in the soft launch each gold gets
seen ~3 times — enough to catch one that is secretly ambiguous — and ~86 times
across run 2. Twenty is the floor, not a comfortable margin: if the soft launch
retires two or three, top up before run 2.

Top-up options, cheapest first:

1. `make_golds.py --loose` accepts any unanimous pilot-500 pair, which adds the
   clear/mid/close buckets (220 candidates total). Review them in the UI and keep
   only the obvious ones — but note a *close* pair is a bad gold by construction,
   so expect a high rejection rate.
2. Better: draw fresh candidates from the whole export rather than pilot-500 —
   very large theta gap plus high VLM confidence, no human votes needed — and
   review those. Not written yet; the pilot-500 restriction is what caps the pool
   at 37 candidates in the first place.

**A gold must be in the current draw.** `lib/pairs.ts` throws at boot on a gold
missing from `pairs.json`, and on 2026-07-29 that nearly shipped: the review UI had
cached its candidate list from a superseded draw and served a pair that had since
been promoted to a worked example. `gold_review.py` now re-reads the file every
rerun and intersects with `pairs.json`, and `make_soft_launch.py` refuses to build
a subset whose golds aren't all in the draw. Regenerating from the recorded
decisions is always safe:

```bash
python scripts/make_golds.py --decisions artifacts/panel-golds-v1/decisions.jsonl
```

**Examples — 3** (`examples.json`, `example-pairs.json`):

```bash
python scripts/make_examples.py --list 4     # QC-clean candidates + contact sheet
python scripts/make_examples.py --clear cmr1mrj9y0sld96d5vwedgxka \
    --close cmr1mrj1t0s8a96d5hrbousc7 --tie cmr1mrcct0ibe96d5vxiolkb4
```

Candidates are now gated on Stage 0 QC (both faces usable, issue-free, gender
confidently as labelled) — that alone removed the browser-chrome and
mislabelled-gender pairs that blocked this in the first attempt, cutting the
`clear` pool to 19 QC-clean candidates.

The three picks were chosen by eye from those candidates, on two rules the data
can't express:

- **No celebrities and no stock/editorial photos.** The top two `clear`
  candidates paired an amateur selfie against a red-carpet photo of a recognisable
  actress; two others carried an Alamy watermark. An example where the polished
  photo wins teaches raters to reward production value — the exact confound the
  brief tells them to ignore.
- **Both photos the same *kind* of photo.** All three chosen pairs are ordinary
  casual portraits at similar framing and age band, so the visible difference is
  the face.

| kind | pairId | p | Dit / Alex / Gemini |
|------|--------|--:|---------------------|
| easy call | `cmr1mrj9y0sld96d5vwedgxka` | 1.000 | left / left / left |
| close call | `cmr1mrj1t0s8a96d5hrbousc7` | 0.502 | left / left / right |
| too close to call | `cmr1mrcct0ibe96d5vxiolkb4` | 0.508 | left / right / left |

Preview: `artifacts/panel-pilot/examples-final.png`. Explanations stay
non-analytic — naming features ("better cheekbones") would teach raters to score a
checklist, which is the behaviour this whole effort replaces.

Both example pairs 2 and 3 are pairs where the humans and Gemini *disagree*,
which is the honest way to show a close call: the instruction sheet gives the
human answer without pretending the pair was obvious.

**Open: the exclusion is pair-level, not face-level.** All three example *pairIds*
are correctly absent from the 3,000-pair draw, but the six example *faces* recur
in **13 other drawn pairs** (0.43%), found while validating the run 2 ship on
2026-07-30. Every rater is shown those faces with a stated answer, so in those 13
pairs a face arrives pre-endorsed. Two defensible calls:

- **Accept it.** 13 of 3,000 is far below the noise floor run 1 measured (only 1%
  of pairs were unanimous, mean majority share 63%), the anchor is weak once the
  opponent changes, and it costs nothing to leave.
- **Drop the 13** (recommended if cheap). 2,987 pairs still gives 12.05 votes per
  pair at 360 sessions, `sample-meta.json` can stay a superset since analysis
  joins on `pairId`, and it removes an easy "why didn't you exclude those?" from
  the writeup.

Either way, make `make_examples.py` write example *faceIds* alongside `pairIds` so
the next draw can exclude at face level from the start.

## Stage 4 — Ship the data to the app  (faceiq-rating)

**Soft launch — ship the 100-pair subset, not the full 3,000.** Already built by
`scripts/make_soft_launch.py` into `labels/panel-pilot/soft-launch/`: 100 real
pairs (stratum-proportional, 50/50 gender) **plus all 33 golds**, 133 pairs
total.

```bash
python scripts/make_soft_launch.py            # regenerate if the draw changes
cp labels/panel-pilot/soft-launch/{pairs,golds,examples}.json ../faceiq-rating/data/
cd ../faceiq-rating && git add data && git commit -m "soft-launch data" && git push
```

For the real launch, ship the full files from `labels/panel-pilot/` instead.
Vercel redeploys on push. There is **no database seeding and no live link**
between the repos — pairs travel as a static file, judgments come back as JSONL.

Why the subset: `buildQueue()` hands each rater the *least-loaded* pairs with a
random tiebreak. That self-balances toward 12 votes/pair over a full run, but a
12-rater soft launch can't reach 12 votes on 3,000 pairs no matter how the queue
is built — 12 raters drawing 100 pairs each from a 3,000-pair
file get near-disjoint slices: ~1,200 pairs with a single vote each, instead of
100 pairs with twelve. Gold pass-rates would still work; inter-rater agreement and
majority stability — the actual point of the soft launch — would not be
computable. The golds ride along whole because `lib/pairs.ts` throws if
`golds.json` names a pairId missing from `pairs.json`.

Also before launch:

- Set the real `PROLIFIC_COMPLETION_CODE` (from Prolific study setup) and a real
  `ADMIN_PASSWORD` (`openssl rand -base64 24`) in the Vercel project.
- Leave `PAIRS_PER_SESSION` at 100 for both runs (see the cost table in Stage 5).
- ~~Gate the "Start a test session" button and keep test rows out of queue
  coverage~~ ✅ done 2026-07-29 in faceiq-rating: `buildQueue()` now counts only
  sessions with `studyId != "test"`, and the button renders only on `/?test=1`, so
  a rater who opens the bare URL can't start an unpaid run. Test the app with
  `http://localhost:3000/?test=1` (or the same param on the deployment).

## Stage 5 — Run  (Prolific)

### There are three runs, and only the first two are this project

| Run | Size | Cost | Purpose |
|---|---|---|---|
| **1. Soft launch** | 12 raters × 100 pairs (same 100 for everyone) | ~$106 at $20/hr | Find UI bugs, get a real median response time, sanity-check golds. Throwaway data. |
| **2. Panel pilot** | 3,000 pairs × 12 votes = 360 submissions | ~$1.9k at $12/hr, ~$3.2k at $20/hr | **The actual experiment.** Produces the majority-vote GT the Step 6 gates are computed on. |
| **3. Scale-up** | ~15.3k noisy pairs × 5 votes ≈ 77k judgments | ~$4–6k | Execution plan Step 7. **Conditional** — only if the Step 6 gate passes. Separate budget and decision. |

Runs 1 and 2 are the same study design at different sizes; run 3 is a later,
bigger project that reuses the same app. Account funded $150 on 2026-07-25 —
covers run 1 with ~$44 to spare; run 2 needs a further top-up, sized once the
soft launch gives us a real median response time (that median, not a guess, is
what decides whether $12/hr is still a fair rate).

### Study setup

Create **one project** (`Project Type: One-off studies`) holding both studies.
To **exclude soft-launch raters from the full run** — they have already seen 100
of the 3,000 pairs and would otherwise vote on them twice — use the study form's
`Add to participant group` field on study 1 (create `faceiq-soft-launch`), then
add a filter on study 2 excluding that group.

- Project name (internal): `FaceIQ — face preference panel`
- Study 1 name (participant-visible): `Photo comparison task (~20 min)`
- Study 2 name (participant-visible): `Photo comparison task (~20 min) — main run`

Study names are visible to participants; keep them plain. Revealing the topic is
fine and unavoidable, but nothing in the title or description should hint at
*criteria* — no mention of bone structure, symmetry, features. The consent and
instruction screens already carry the framing we want.

**Bug reports** have two channels, both live:

1. An optional free-text box on the app's done screen (built 2026-07-25). It
   sits under the completion code, saves to `Session.feedback`, and surfaces in
   an amber panel on `/monitor` and in `?table=sessions` exports. Typing in it
   cancels the auto-return timer, so nobody gets bounced mid-sentence; leaving
   it blank still auto-returns after 12s so no one strands their submission.
2. Prolific's participant→researcher messaging, for anyone who quits early and
   never reaches the done screen.

Put channel 2 in the **study description**, not the consent block — consent
should stay narrowly about data use, and description text is what participants
actually read before accepting. Suggested line:

> If anything goes wrong — a photo that won't load, a page that freezes — please
> message us through Prolific. There is also an optional feedback box at the end.

### ⛔ Blocker: Vercel deployment protection (check as anonymous user)

If **you** can open the URL and see "Missing study link", that only proves the
app works — you are almost certainly logged into Vercel in the same browser, so
team auth passes silently. An anonymous participant is not:

```bash
curl -s -o /dev/null -w "%{http_code}\n" \
  "https://<your-domain>/?PROLIFIC_PID=test&STUDY_ID=test&SESSION_ID=test"
```

You want **200**. A **302** to `vercel.com/sso-api` means Prolific raters would
hit a Vercel login wall. Fix **before** publishing:

| Option | Cost | What to do |
|---|---|---|
| **Turn off Vercel Authentication** (recommended for this throwaway study) | $0 | Project → Settings → Deployment Protection → toggle **Require log in** OFF. The app already has `ADMIN_PASSWORD` on `/monitor` and `/api/export`. |
| **Standard Protection** (what you have now) | $0 | Protects preview URLs; production **custom domains** are exempt. A bare `*.vercel.app` deployment URL may still require team login for outsiders. |
| **Password Protection** | Usually paid (Pro+) | Shared password in the study description — works but ugly; every rater needs the password. |
| **All Deployments / Advanced** | **$150/mo** | Overkill. Do not pay this for a one-off panel pilot. |

Simplest path: disable Vercel Authentication on this project, re-curl as above,
then use the URL with Prolific params in the study link.

### Study form — every field, soft launch values

**→ `docs/research/prolific-soft-launch-form.md`** is the fill sheet: every field
in Prolific's form, in order, with the exact string to paste, plus what changes
for run 2. Kept as one document so it can't drift from a duplicate table here.

Two things from it that matter beyond the form itself:

- The study URL is `https://faceiq-rating.vercel.app` (project production alias,
  verified 200 anonymously 2026-07-29) **with** the `PROLIFIC_PID` / `STUDY_ID` /
  `SESSION_ID` params. Never a per-deployment URL like
  `faceiq-rating-3tvlk95q4-face-iq-labs.vercel.app` — pinned to one build, so it
  silently serves stale data after the next push.
- Nothing in the name or description may hint at *criteria* (symmetry, bone
  structure, features): naming them would prime the analytic mindset the whole
  study is designed to avoid.
2. **Sampling method.** Prolific has three, chosen inside study creation — there
   is no separate "screening" step to do first:
   - *Standard* — anyone. Use for the soft launch.
   - *Quota* — balanced on **sex only**.
   - *Representative* — US/UK census-matched on age + sex + ethnicity, min 300
     participants, max ~1,500 US.

   The full run needs ~360 participants, which clears the 300 minimum, so **one
   US representative sample** delivers execution-plan Step 4's balance in a
   single study. Caveat: census-proportional means ~60% white raters, so it does
   *not* give equal ethnicity cells. If the cross-ethnicity bias audit comes out
   underpowered, add a booster study with an ethnicity prescreener rather than
   fighting the quota system. Custom screening (a paid mini-study) is not needed.

3. **What a "submission" is: one rater completing one session.** With
   `PAIRS_PER_SESSION = 100` plus 5% golds, that's 105 judgments per submission,
   *not* 12. Submissions = total judgments ÷ 100.

   | Calculator field | Soft launch | Full run |
   |---|---|---|
   | Number of submissions | 12 | 360 |
   | Reward per hour | $20 | decide after the soft launch |
   | Time per submission | 20 mins | measured median + buffer |
   | ≈ total cost | ~$114 | $2,060 (@$12/hr) – $3,430 (@$20/hr) |

   **Rate reality check:** Prolific's audience snapshot says competing studies
   pay **$18–20/hr**, well above the $12 the execution plan assumed. Pay the
   competitive rate for the soft launch — it is only 12 people, it fills in
   minutes, and a slow-filling debug study wastes days. For the full run the
   lever is *time*, not rate: reward = hourly × estimated minutes, so if the
   observed median is 12 min rather than 20, the same $20/hr costs $4 per
   submission instead of $6.67. Measure first, then set it.

   360 = 3,000 pairs × 12 votes ÷ 100 pairs per session. 20 min = 105 items at
   ~6.5s (~11 min) + consent/instructions/examples for a first-timer (~5 min),
   rounded up on purpose: Prolific compares the estimate to actual completion
   times and flags underpayment. Platform fee is **42.8%** on top of rewards
   (corporate rate; 33.3% is academic/non-profit only). At $15/hr the full run is
   ~$2,570. Both sit inside the plan's $2–3k envelope.

4. Fund the workspace before launching — studies will not go live on a $0
   balance.
5. Soft launch: 12 submissions against the 100-pair file. Watch `/monitor`: gold
   pass-rates, median response time, <2s click-through, tie rate, comment volume.
   **Re-derive "time per submission" from the observed median** before the full
   run instead of trusting the 6.5s estimate.
6. Fix, then full launch.

### Run 1 outcome — soft launch, 2026-07-30  ✅ DONE

Published mid-afternoon; **12/12 submissions complete in ~13 minutes**, all
approved and paid. Archived to `labels/panel-pilot/soft-launch/results/`
(gitignored — carries Prolific ids), then the database was truncated so run 2
starts at zero coverage. Rerun the numbers any time with:

```bash
python scripts/analyze_soft_launch.py
```

| Measure | Result | Reading |
|---|---|---|
| Sessions | 12 complete, 105 judgments each (1,260 total) | No dropouts, no partials. |
| Rating time, median | **5.9 min** (p25 4.5, p75 6.7, max 11.6) | Session clock starts at "I consent", so this excludes instructions. Sets run 2 at **12 min / $4.00**, ~$1.3k under the old estimate. |
| Per-judgment median | 2.2 s | Faster than the 6.5 s guess the whole cost model was built on. |
| Judgments under 2 s | 42% | **Not a failure signal** — see below. |
| Golds | **51/60 = 85%** | Matches the 84.9% human-vs-VLM agreement from the original GT audit. |
| Raters under the 60% gold floor | **0** | Nobody to reject. Two raters at 3/5. |
| Real pairs | 100, exactly 12 votes each | The overlap the soft launch was carved for. |
| Unanimous real pairs | **1%** (mean majority share 63%) | The headline finding — see below. |
| Comments | 35 on 1,260 (3%) | Volume is manageable to read by hand at run 2 scale (~1k). |
| Bug reports | 1 (photos not loading on first two pairs) | Fixed; redeploy before run 2. |

**42% fast clicks with an 85% gold rate means speed is not cheating.** The two
raters with the weakest golds (3/5) were also the fastest (~65% under 2 s), which
is the profile to watch — but 3/5 is inside binomial noise at an 85% true rate,
and rejecting on it would punish honest instinct raters. Keep Prolific's
auto-reject-fast-submissions **off** for run 2 and screen on golds only:
≤2/5 is the reject line.

**Only 1% of real pairs were unanimous, and the average winning side held just
63% of the 12 votes.** This is the most important thing run 1 taught us, and it is
about the *task*, not the app: human attractiveness judgments genuinely disperse.
Two consequences for Step 6:

- Majority-vote GT is noisier than the plan assumed, so read the human-vs-majority
  ceiling `H` as the realistic bar. Expecting Gemini to beat a 63%-consensus
  target is expecting it to beat humans.
- Per-pair vote entropy stops being a diagnostic and becomes a headline result.
  Report the distribution, and consider whether 12 votes is enough resolution on
  `close`-stratum pairs before committing to run 3's 5-votes-per-pair design.

Caveat on the 63%: the soft-launch 100 are stratum-proportional, so they include
`close` pairs by construction. Recompute on run 2's full 3,000 and split by
stratum before treating it as the cohort-wide number.

**Three golds are now under suspicion** — `cmr1mr5cw06y796d59gfjfwv7` (1/2),
`cmr1mr5tr083a96d5m7edvf9s` (1/3), `cmr1mrf1p0m3o96d5be5szpcp` (2/4). Each was
seen 2–4 times, so this is weak evidence, not a verdict; a gold that clean raters
miss is a bad gold, not a bad rater. One gold (`cmr1mrh2r0pgb96d5qamu7jnq`) was
never shown at all, which is just the luck of 5-from-20 sampling across 12 sessions.

*Resolved by run 2, which put ~90 views on every gold instead of 2–4.* One of these
three (`cmr1mr5tr083a…`) was confirmed broken and retired; the other two passed at
77–80% and were kept. Run 1's weak signal pointed the right way but could not
settle it — **don't retire golds on double-digit view counts.**

### Run 2 outcome — panel pilot, 2026-07-30  ✅ DATA IN

**372 sessions, 38,314 judgments, filled in 99 minutes** (19:31–21:10 UTC, peak 50
starts/minute). 362 completed, 10 abandoned; 372 distinct raters, no repeats, no
test rows. Archived to `labels/panel-pilot/results/` (gitignored). Analysis:

```bash
python scripts/analyze_panel_run.py      # -> artifacts/panel-run-v1/
```

#### The gates — close pairs fail, clear pairs pass

| stratum | pairs | split-half | human `H` | Gemini `G` | `G−H` | BT theta | maj share |
|---|--:|--:|--:|--:|--:|--:|--:|
| close | 1,050 | **63.4%** | 64.6% | **51.2%** | −13.4% | 50.2% | 66.2% |
| mid | 750 | 64.4% | 65.2% | 56.6% | −8.6% | 57.8% | 66.8% |
| **clear** | 600 | **77.6%** | 74.6% | **81.0%** | **+6.4%** | 81.4% | 75.5% |
| lowconf | 300 | 62.6% | 63.0% | 53.9% | −9.1% | 60.1% | 66.3% |
| overlap | 280 | 70.8% | 70.2% | 70.1% | −0.1% | 75.9% | 70.2% |
| ALL | 2,980 | 67.1% | 67.2% | 60.7% | −6.5% | 61.9% |

**Step 7's gate — "split-half majority stability on close pairs meaningfully above
our two-person 68.5%" — is not met. Close pairs come in at 63.4%, below the
baseline.** 2,777 of 2,980 pairs got a determined majority; the remaining 203 (7%)
are dead splits.

> **Correction (2026-07-31): the gate is the wrong measurement, and the
> "irreducibly subjective" reading it produced is wrong.** Split-half *majority*
> stability collapses to the binary winner and throws away the margin. When the true
> split is near 50/50, majority reproducibility is mathematically pinned near chance
> **no matter how much real signal the votes carry** — so a low number here measures
> the metric's blind spot, not the data's quality. Measuring the quantity that
> actually carries the information (the vote *share*) reverses the conclusion. See
> §"Close pairs re-examined" below and `panel-pilot-findings.md`.

**The other half of the hypothesis is confirmed emphatically.** On clear pairs
Gemini agrees with the human majority **81.0%** against a human ceiling of 74.6% —
the VLM is **+6.4 points better than an average individual human** at predicting
what the crowd thinks. The existing BT fit (trained on VLM labels) independently
lands at 81.4% there. So the ~37k high-confidence clear/very_clear VLM labels are
not just adequate, they are better than paying humans for the same pairs.

Gemini's own confidence is a *useless* discriminator (high 61.4%, medium 56.5%),
which matters because the plan intended to relabel by confidence. Stratum is the
signal; confidence is not.

#### Close pairs re-examined — real signal the VLM labels are blind to

Prompted by a good challenge from Dit: if close pairs are contested, isn't the
*contestedness itself* information, and can't BT absorb it? Yes — and the data says
more than that. Measuring vote **share** instead of binary majority:

| stratum | share reliability (SB) | max-share obs | vs p=.5 null | pairs ≥75% agree | null |
|---|--:|--:|--:|--:|--:|
| close | **0.655** | 70.4% | +7.8 | **38.9%** | 14.7% |
| mid | 0.674 | 70.7% | +8.3 | 40.2% | 14.8% |
| clear | 0.858 | 78.4% | +16.0 | 61.4% | 14.5% |
| lowconf | 0.683 | 70.3% | +7.8 | 40.0% | 14.4% |
| overlap | 0.789 | 74.6% | +12.2 | 52.6% | 15.2% |

Close-pair vote shares **replicate at 0.655** across disjoint rater halves, and 39%
of close pairs draw 75%+ agreement against 15% expected under guessing. Back out the
noise and true `p` on close pairs has sd ≈ **0.20** around 0.5 — substantial real
variation.

**And BT predicts none of it.** Spearman between theta gap and human share *within*
the close stratum is **+0.019**; the dose-response is flat across every gap bin. So
this is not BT being roughly right with noise on top — it is information BT does not
have.

Artifacts ruled out: global side bias is 49.3/50.7 (detectable at n=33,540, but 0.7
points cannot manufacture a 24-point excess), and mixed-ethnicity close pairs spread
only ~6.9 points across groups against ~20 points of real per-pair variation.

**BT is also overconfident by 5.8×.** Fitting `p = σ(Δθ/T)` against human votes gives
`T = 5.84`: where BT says 88%, humans split 58/42. Empirically, agreement with BT's
favourite by percentile gap — the headline curve, monotone from chance to 93%:

| percentile gap | /10 gap | pairs | humans agree |
|---|--:|--:|--:|
| 0–5 | 0.11 | 1,237 | 51.9% |
| 5–10 | 0.39 | 450 | 53.8% |
| 10–20 | 0.72 | 339 | 60.5% |
| 20–30 | 1.28 | 141 | 66.0% |
| 30–45 | 1.83 | 159 | 71.4% |
| 45–60 | 2.59 | 121 | 76.5% |
| 60–80 | 3.51 | 95 | 89.6% |
| 80–100 | 5.03 | 29 | 93.4% |

The ordering is validated end to end. The resolution is coarse: **~2.5 /10 points
before 3 in 4 people agree**, sub-point differences are a coin flip.

**Consequences, replacing the earlier "don't relabel close pairs / spend on clear
coverage" line, which was backwards:**

1. ✅ **Done — `artifacts/bt-refit-v3-panel`.** `scripts/refit_bt_panel.py` fits θ jointly
   over VLM labels and individual human votes with a per-source discrimination
   (`β_vlm ≡ 1`, `β_human` free), so the two sources' different decisiveness can't bend
   the ranking through uneven panel coverage. `β_human` = 0.322 F / 0.246 M — humans are
   3–4× flatter, independently reproducing the `T` finding. All §5.1 gates pass
   (1 component, no face < 15 distinct opponents, stability ρ 0.992/0.991).
   Held-out (fit on half the raters, score on the disjoint half), by stratum:

   | stratum | VLM-only | + panel | gain |
   |---|--:|--:|--:|
   | **close** | **50.6%** | **58.3%** | **+7.7** |
   | mid | 53.7% | 56.8% | +3.1 |
   | lowconf | 56.9% | 59.1% | +2.2 |
   | clear | 71.2% | 71.6% | +0.3 |
   | overlap | 62.9% | 62.3% | −0.5 |
   | ALL | 57.5% | 61.1% | +3.7 |

   Close pairs move off *literal chance*, clear pairs don't move — the gain lands
   precisely where the blind spot was. Ordering is preserved (ρ 0.991 vs v2, median move
   1.8 pctile, 97/2,866 faces shift > 0.5 on the /10 scale), so this is added precision
   in the crowded middle, not a reshuffle.
2. Future *human* budget goes to close pairs, not breadth. A p≈0.5 comparison carries
   `p(1−p)=0.25` Fisher information about Δθ versus 0.09 at p=0.9 — close comparisons
   are the most informative per unit, not the least. ~25 votes/pair reaches 0.8 share
   reliability. Costed at run 2's actual **$0.052/vote** (=$1,900/36,290), against v3
   percentiles over the 47,904 scored export pairs:

   | option | pairs | votes | cost |
   |---|--:|--:|--:|
   | **top up the 1,050 already drawn to 25** | 1,050 | +13 ea | **$715** |
   | the whole sub-10-percentile blind spot | 11,712 (24%) | 25 | $15,300 |
   | relabel the entire export with a 12-crowd | 47,904 | 12 | $30,100 |

   The last row is the trap — ~three quarters of it buys pairs where Gemini already
   matches the crowd. Do the $715 top-up first and check the marginal return before
   committing to $15k.
3. Train the comparator on soft targets; hard labels on 55/45 pairs teach false
   confidence.
4. **No budget on graph connectivity** — `bt-refit-v2-qc` already has 1 component per
   gender, no face under 15 comparisons, stability ρ 0.992. It was never the bottleneck.
5. Don't surface sub-point /10 differences as meaningful.

#### Rater QC — reject 2, approve the rest

Gold pass rate alone would have condemned ~23 raters. It should not:

- Raters who **failed** the gold screen average **65.2%** agreement with the other
  eleven. Raters who **passed** average **67.6%**. A 2.4-point gap means the gold
  screen is measuring luck, not quality. With 5 golds each, an honest rater at the
  task's ~85% accuracy fails 3+ by chance ~2.7% of the time — about 10 people per
  360 by construction.
- **Three golds are broken** and manufacture failures: `cmr1mr2t702st…` 59%,
  `cmr1mril80rl0…` 60%, `cmr1mr5tr083a…` 60% (run 1 flagged the last one on 3
  views — it was right). Retiring them rescues 6 of the 23.

**Replaced 2026-07-31** — `scripts/replace_golds.py` (rerunnable; `--write` to
apply, backup at `golds.json.bak`, retired pairs recorded in the file itself):

```bash
python scripts/replace_golds.py            # dry run, prints the plan
python scripts/replace_golds.py --write
cp labels/panel-pilot/golds.json ../faceiq-rating/data/golds.json
```

The replacements come from a **better source than the originals**. The first 20
rested on three judges agreeing (Dit, Alex, Gemini). Run 2 surfaced 37 non-gold
pairs where 12–28 independent strangers were *unanimous* — that is what
"unmistakable" actually looks like, measured rather than assumed. Picks are
restricted to the `clear` stratum with the BT fit and the VLM both agreeing, and
chosen to match the retired pairs' left/right split (1 left, 2 right) so a
one-key rater still cannot pass:

| new gold | side | unanimous votes |
|---|---|--:|
| `cmr1mrdqo0k6r96d51k15r116` | left | 28 |
| `cmr1mrb5g0gg596d5nb73d9zq` | right | 18 |
| `cmr1mroij114b96d5da385iny` | right | 16 |

Gold set stays at **20** (8 left / 12 right), `revision: 2`, all present in
`pairs.json`, no overlap with the worked examples.

**Retire-and-replace is now a routine every run should end with**, not a one-off:
each run measures its own golds at ~90 views and hands back a fresh pool of
unanimous candidates.

So reject only where independent signals agree — gold failure *and* agreement
below µ−2σ (53.3%), or straight-lining, or impossible speed:

| prolificPid | n | golds | agreement | median | fast | ties |
|---|--:|--:|--:|--:|--:|--:|
| `63b18d94775ad4aad221103c` | 105 | 1/5 | **49%** | 1,335 ms | 86% | 0% |
| `6a178bf9ee4c5e4a39bc027c` | 105 | 2/5 | **49%** | 1,857 ms | 56% | 22% |

49% on a forced binary choice *is* chance. Everyone else gets approved, including
all 29 single-signal "watch" raters. Full table in
`artifacts/panel-run-v1/raters.csv`, PIDs in `reject-pids.txt`.

**42% of judgments under 2 seconds is not a problem** — same as run 1, and the
population agreement of 67.5% (sd 7.1) sits right at the human ceiling. Speed is
the instructed behaviour.

#### Coverage came out uneven — a real app bug for run 3

Votes per pair ranged **8 to 56** (mean 12.24) instead of a flat 12. Every pair
cleared 8, so nothing is unusable, but ~25% of pairs got 8 votes while a handful
got 40+.

Cause: `buildQueue()` balances on *judgments already recorded*, and 50 raters
starting per minute means hundreds of queues were built from identical all-zero
counts before anyone had submitted anything. The pairs drawn during the opening
five minutes were handed out repeatedly.

**Fixed 2026-07-31** in `faceiq-rating/lib/assignment.ts`. `buildQueue()` now
ranks on **judgments recorded + pairs reserved by live sessions**, where a live
session is one that is incomplete and started within `RESERVATION_TTL_MS` (30 min
— comfortably past the 6-minute median, and the TTL stops an abandoned session
parking 100 pairs forever). Only the *unjudged* part of a live assignment counts as
a reservation, so nothing is double-counted.

Verified by replaying run 2's real arrival and completion times, with judgments
landing progressively across each session rather than instantly:

| policy | min | max | mean | sd | pairs <10 | pairs >20 |
|---|--:|--:|--:|--:|--:|--:|
| old (judgments only) | 8 | 43 | 12.24 | 4.88 | 1,113 | 202 |
| **new (+ reservations)** | **12** | **13** | 12.24 | **0.43** | **0** | **0** |
| observed run 2 | 8 | 56 | 12.24 | 5.63 | 1,264 | 232 |

The old-policy simulation reproduces the observed spread closely, which confirms
the diagnosis; the new policy flattens it to 12–13.

### Run 3 outcome — 4,800 hard pairs, 2026-08-01  ✅ DATA IN

First study of the repeatable loop in `panel-study-playbook.md`, and the first run
of the `CURRENT_STUDY_ID` scoping — **no database wipe**, run 2's rows still sit in
the same tables. Study `6a6d1e7e3999a35d0dc956b9`, 301 sessions, 31,421 judgments
(29,924 real + 1,497 gold) over 4 hours. 299 completed, 2 abandoned. Archived to
`labels/panel-run-3/results/` (gitignored — the `.gitignore` rule was widened to
`labels/panel-run-*/**/results/`).

```bash
python scripts/analyze_panel_run.py --results labels/panel-run-3/results \
    --panel-dir labels/panel-run-3 --out artifacts/panel-run-v3
python scripts/panel_run_delta.py --export data/exports/<runId> \
    --prior labels/panel-pilot/results labels/panel-pilot/sample-meta.json \
    --new   labels/panel-run-3/results labels/panel-run-3/sample-meta.json \
    --rejects artifacts/panel-run-v1/reject-pids.txt \
              artifacts/panel-run-v3/reject-pids.txt --spend 1286
python scripts/refit_bt_panel.py --export data/exports/<runId> \
    --panel-results labels/panel-pilot/results labels/panel-run-3/results \
    --panel-meta labels/panel-pilot/sample-meta.json \
                 labels/panel-run-3/sample-meta.json \
    --rejects artifacts/panel-run-v1/reject-pids.txt \
              artifacts/panel-run-v3/reject-pids.txt \
    --baseline artifacts/bt-refit-v3-panel/ratings.csv \
    --out artifacts/bt-refit-v4-panel
```

#### The queue fix worked

**Votes per pair: 6 on 3,676 pairs, 7 on 1,124. Nothing else.** Run 2's spread was
8–56 (sd 5.63); this is sd 0.43, matching the simulation's prediction exactly. Every
one of the 4,800 pairs was covered, and no pair was over-served. Treat this as
closed.

#### The gates — the VLM is not merely weak on hard pairs, it is blind

| stratum | pairs | split-half | human `H` | Gemini `G` | `G−H` | BT theta (v3) | maj share | dead splits |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| topup-0-10 | 4,800 | 59.5% | 62.7% | **51.3%** | **−11.4%** | 59.3% | 69.3% | 13% |

Gemini lands at **51.3%** against the human majority — a coin flip — and its own
confidence does not rescue it: `high` 53.5%, `medium` 48.2%. It is confidently
wrong. That kills the idea of using VLM confidence to triage which pairs need
humans; **percentile gap in the current refit is the only usable trigger**.

Dead splits rose to 13% from run 2's 7%, which is arithmetic rather than a quality
change: 6 votes can tie 3–3 far more easily than 12 can tie 6–6. It costs nothing —
the refit consumes vote *shares*, not majority labels, so a 3–3 is a real
observation that two faces are level.

#### Marginal value — the number that decides whether to keep spending

`panel_run_delta.py` withholds **whole pairs** from the newest study (25%, 5 seeds),
so not one of their votes enters any fit. A model can only score better on them by
having learned better *face* scores — the same mechanism that any future study would
have to rely on for pairs we never buy.

| ranking | predicts a vote on a pair it never saw | step |
|---|--:|--:|
| machine labels only | 51.5% ±0.7 | — |
| + run 2 (36,290 votes, $1,900) | 54.7% ±0.7 | +3.26 |
| **+ run 3 (22,204 votes, $1,286)** | **57.9% ±0.7** | **+3.18** |
| human ceiling (`H`, leave-one-out) | 62.7% | — |

Paired across seeds the run 3 step is **+3.18 ±0.23, positive on 5 of 5 splits**.
That is **$404 per accuracy point**, against $583 for run 2 — targeting hard pairs
made each dollar **31% more productive**, and each *vote* 60% more productive
(0.143 vs 0.090 points per 1,000 votes).

Cumulatively the two studies have closed **56% of the reachable gap** (51.5% → 57.9%
against a 62.7% ceiling). The ceiling is itself measured with only 5–6 other votes
per pair, so it slightly understates the truth and 56% slightly overstates our
progress; "about half" is the honest phrasing.

#### No sign of diminishing returns yet — the dose curve is still bending upward

Feeding a growing share of run 3's pool into the fit and scoring the same withheld
pairs:

| votes in the fit | accuracy | step | points per 1,000 votes |
|--:|--:|--:|--:|
| 5,550 | 55.41% | — | — |
| 11,103 | 56.01% | +0.61 | 0.110 |
| 16,650 | 56.92% | +0.91 | 0.163 |
| 22,204 | **57.91%** | **+1.00** | **0.179** |

The last increment is the *largest*. BT appears to need a critical mass of human
observations per face before human-informed theta stabilises, so early votes
underpay and later ones compound. **The stop condition in the playbook (<1 point per
$500) is not close to triggering** — run 3 delivered 3.18 points for $1,286, i.e.
1.24 points per $500. Buy again on the same shape.

#### BT refit v4 — pooled, and pooling is a sum not a join

`refit_bt_panel.py` now takes repeated `--panel-results`/`--panel-meta` pairs. Runs
merge on the export's `pairIndex`, so disjoint draws union and a pair bought twice
accumulates votes; no schema change and no re-analysis of earlier runs is needed to
add a study. Pooled panel: **7,780 pairs, 65,894 votes, 667 raters (8.5 per pair)**.

All gates passed. `beta_human` rose to 0.417 (female) / 0.368 (male) from run 2's
0.25–0.32, i.e. humans look *less* flat than before — expected, since the pooled
panel now carries many more independent observations per face. Spearman vs v3 is
0.9905 over 2,866 faces; median /10 movement 0.14, 149 faces move more than half a
point.

**One trap worth recording.** Passing `--baseline artifacts/bt-refit-v3-panel` makes
the rater-holdout table read as if v4 *lost* ground on run 2's strata (close
64.1% → 60.2%). It did not: v3 was fit on **every** run 2 rater including the
withheld half, so on pairs it owns it is scoring its own training data. The only
honest cells in that table are strata the baseline never saw (`topup-0-10`:
55.0% → 58.1%). The dashboard now prints a warning whenever the baseline is itself a
panel refit, and `metrics.json` carries a `baselineCaveat`. For a clean
before/after, use `panel_run_delta.py`.

#### Gold health — the replacements fixed the floor

All 20 golds now pass at **≥70%** (range 70–97%, ~75 views each). The three retired
after run 2 sat at 59–60%, and nothing has taken their place at the bottom. Revision
2 of the gold set is sound; no further replacement needed this run.

#### Rater QC — and the distinction the file name was hiding

4 raters dropped from the fit, **1** recommended for rejection on Prolific.

| prolificPid | n | golds | agreement | median | fast | ties | action |
|---|--:|--:|--:|--:|--:|--:|---|
| `69eeacbb750a87c832863dfd` | 105 | 2/5 | **41%** | **236 ms** | 98% | 1% | **reject** |
| `68114b50b766b8f5b3b6cc8e` | 105 | 2/5 | 48% | 2,289 ms | 39% | 2% | pay, drop votes |
| `645906dc264ac9eb66b7f3ec` | 105 | 2/5 | 50% | 1,722 ms | 63% | 5% | pay, drop votes |
| `65fcaa30c61e9b7657625912` | 21 | 0/1 | — | 3,588 ms | 10% | **90%** | dropout, nothing to action |

`analyze_panel_run.py` now writes **two** lists, because these were always two
different decisions and one file name was doing both jobs:

* `reject-pids.txt` — votes excluded from every fit. Costs the rater nothing, so a
  statistical case is enough.
* `prolific-rejections.txt` — payment refused. Permanent and appealable, so it
  requires *behavioural* proof: a median under 700 ms or one button all session.
  Failing golds and disagreeing with the crowd can both be luck or unusual taste.

Only `69eeacbb` qualifies: a 236 ms median is below the **1st percentile of the
whole cohort** (1,138 ms) by a factor of five.

**Speed still does not predict quality, now on 299 raters.** Splitting by fast-click
rate: slowest third 62.8% agreement, middle 64.1%, fastest third 61.5%.
`corr(fastRate, agreement) = −0.12`. The 36% of judgments under 2 seconds is the
instructed behaviour, not a defect. Golds remain weak too:
`corr(goldRate, agreement) = +0.18`, and gold-failing raters average 60.9% agreement
against 63.1% for gold-passers.

#### Rater feedback worth acting on

14 messages, 12 positive. Three carry information:

* *"I wish there was a button for 'neither', it felt wrong using 'too close to
  call'"* — **not a misunderstanding**, and this rater scored above the cohort on
  every metric (4/5 golds, 65.9% agreement, 21% fast vs 36% cohort). The task is
  relative, and for Bradley-Terry a tie at the bottom and a tie at the top are the
  same observation: no preference between these two. Worth a wording tweak to the
  tie button (e.g. "no preference / too close") since run 3's draw is *all* near-tie
  pairs and some are level at the bottom.
* Two raters spontaneously flagged **AI-looking faces** ("I wondered how many were
  AI", "some of the AI morphed faces appeared unrealistic"), after Stage 0 QC already
  removed 133. The exclusion list is not complete.
* One rater **recognised a face** ("Hateeq the tiktok blackpill dude"). Public
  figures in the cohort are a validity risk — reputation contaminates an
  attractiveness judgement — and a privacy consideration. Worth a pass for
  recognisable people.

### Run 4 outcome — 2,500 uniform random pairs, 2026-08-04  ✅ DATA IN, STUDY CLOSED

Study `6a71110e4d5c6eba96c17d21`, **303 sessions, 31,237 judgments** (29,351 real usable + 1,486 gold) over
~4.6 hours, 296 completed / 7 abandoned, **$968.58**. Archived to `labels/panel-run-4-random/results/`.
Full write-up in research log **§5.8**; the exact command sequence is in
`prolific-soft-launch-form.md` §9.5.

The first study bought to **measure** rather than to improve, and the ops notes that come out of that:

- **Coverage: 11 votes on 282 pairs, 12 on 2,185, 13 on 33.** sd 0.35. The queue fix from run 2 has now
  held across two runs at different depths — treat it as closed for good.
- **`analyze_panel_run.py` needs `golds.json` in the panel dir.** A random draw does not emit one, so
  `cp ../faceiq-rating/data/golds.json labels/panel-run-4-random/golds.json` first. Run 4 carries the same
  20 golds with their original left/right sides, so the shipped file is the right one.
- **`refit_bt_panel.py` has no default for `--exclude-faces` / `--exclude-genders`.** Omit them and it
  ranks 2,999 faces instead of 2,866, records `qcExcludedFaces: 0`, and **still passes every gate**. Check
  that field after every refit.
- **`Session` has no `judgmentCount` column** — the psql dump in Stage 6 synthesises it with a subselect.
- **Every query must filter `studyId`.** Runs 2–4 coexist in one database and four `test` rows from the
  pre-launch check were sitting in there too.

**Gold health: 19 of 20 pass at 79–99%. Retire `cmr1mr5cw06y796d59gfjfwv7`** — 63% over 78 views (49 left,
18 right, 11 tie). Run 1 flagged it at 1/2 views, run 3 passed it at 77–80%, and a third independent cohort
has now failed it. That is the pattern the "don't retire golds on double-digit view counts" rule was meant
to catch: two weak signals plus one strong one, not one weak one.

**Rater QC: 4 dropped from the fit, 1 recommended for Prolific rejection** (`6a4aaa3ca9cd50f4e76237c3`,
294 ms median, 88% fast, 3/5 golds — roughly 4× under the cohort's 1st percentile). The other three
(`6a0f76e7…`, `6a040eed…`, `69c826ee…`) are statistical cases: pay, exclude the votes.

**The gold screen behaved differently on this draw, and it is worth understanding.** Gold-failing raters
averaged **67.7%** agreement against **75.5%** for passers — a 7.8-point gap, versus 2.4 in run 2 and 2.2
in run 3. Golds are wide-gap pairs by construction, so on a run whose pairs are mostly wide they measure
the same skill as the task; on an all-near-tie draw they do not. **Weight golds by draw type**, and do not
carry a fixed threshold across draws of different difficulty.

**Rater feedback: 14 messages, mostly positive.** Three carry information: one asked whether "Skip" or
"Save & Continue" was correct on a tie (same family as run 3's tie-button wording — the two controls are
ambiguous together); one wrote *"quite a few where neither seemed attractive, just a matter of personal
taste"*, which is the honest description of a uniform draw rather than a complaint; one wrote *"this was
disturbing"*, worth recording for the ethics section of any write-up.

## Stage 6 — Pull the data back and decide

```bash
curl -H "x-admin-password: $ADMIN_PASSWORD" \
  -o labels/panel-pilot/judgments.jsonl "https://<deploy>/api/export?table=judgments"
curl -H "x-admin-password: $ADMIN_PASSWORD" \
  -o labels/panel-pilot/sessions.jsonl  "https://<deploy>/api/export?table=sessions"
```

**`?table=judgments` does not survive scale — use psql instead.** On 2026-07-30 it
returned 500 (then reset mid-stream on retry) at only 1,477 rows: the route
buffers every row into one string inside a serverless function, so it hits the
duration/response ceiling. Run 2 will produce ~37,800 judgments, ~25× that. The
sessions table is small and still fine over HTTP. Pull judgments straight from
Postgres, which is also faster and needs no redeploy:

```bash
cd ../faceiq-rating && set -a && . ./.env && set +a
psql "$DATABASE_URL" -At -o judgments.jsonl -c 'select row_to_json(t) from (
  select j.id as "judgmentId", j."sessionId", s."prolificPid", s."studyId",
         j."pairId", j.choice, j."responseTimeMs", j."presentationOrder",
         j."isGold", j.comment,
         to_char(j."createdAt" at time zone '"'"'UTC'"'"', '"'"'YYYY-MM-DD"T"HH24:MI:SS.MS"Z"'"'"') as "createdAt"
  from "Judgment" j join "Session" s on s.id = j."sessionId"
  order by j."createdAt") t;'
```

Also dump `Session` with `assignedPairIds` intact (the HTTP route drops it for a
count): that column is the only record of which pairs each rater was *offered*
versus judged, which you need to tell a dropout from a skip.

Verify before wiping anything: `sum(session.judgmentCount)` must equal the
judgment line count, and `judgmentId` must be unique.

Drop rows where `prolificPid` starts with `test` or `studyId == "test"`. Join
against `sample-meta.json` (strata/thetas/VLM labels) and Prolific's demographic
export on `prolificPid`. Then compute the Step 6 gates: human-vs-majority `H`,
Gemini-vs-majority `G`, model-vs-majority `M` per bucket and per demographic
stratum, split-half majority stability, per-pair vote entropy. Write to research
log §5.5.

---

## Open items

| Item | Owner | Notes |
|---|---|---|
| ~~Full 3,000-face QC pass~~ | Dit | ✅ done — 508 flags; review `artifacts/face-qc-v1/review/*.png` |
| ~~Flag review~~ | Dit | ✅ done — 375 keep / 71 exclude / 62 relabel |
| ~~Per-face exclusion script in faceiq-labs~~ | — | ✅ `scripts/apply-gt-face-decisions.ts`, dry-run first |
| Run the labs script + re-export | Dit | needs `faceiq-scoring-research` branch; **no longer blocking** |
| ~~Refit BT with QC exclusions~~ | — | ✅ `artifacts/bt-refit-v2-qc`, all gates passed |
| Duplicate-face detection pass | — | not written; ArcFace embeddings |
| ~~Ethnicity for the cohort~~ | Dit | ✅ done — 3,000/3,000 via Stage 0.5 join |
| `race` added to labs GT exporter | Dit | two lines; nice-to-have now that Stage 0.5 covers it |
| ~~Panel strata draw~~ | — | ✅ `scripts/select_panel_pairs.py` — 3,000 pairs, all floors met |
| ~~`golds.json` generator~~ | — | ✅ `scripts/make_golds.py` — 33 golds |
| ~~Review the golds~~ | Dit | ✅ 2026-07-29 — 13 rejected, **20 golds** ship (at the floor; top-up path in Stage 3) |
| ~~Ship soft-launch data to faceiq-rating~~ | Dit | ✅ pushed + deployed 2026-07-29; verified live on `faceiq-rating.vercel.app` |
| ~~Gate test sessions~~ | — | ✅ faceiq-rating: `?test=1` gate + test rows excluded from queue counting |
| ~~Worked examples chosen~~ | Dit | ✅ 3 picked, QC-clean; `artifacts/panel-pilot/examples-final.png` |
| ~~Soft-launch subset~~ | — | ✅ `labels/panel-pilot/soft-launch/` — 100 real + 33 gold |
| ~~Ship soft-launch files to faceiq-rating~~ | Dit | ✅ done; run 1 published + paid 2026-07-30 |
| ~~Run 1 soft launch~~ | Dit | ✅ 12/12 in ~13 min, 85% golds, 0 rejects — see Stage 5 outcome |
| ~~Archive run 1 + wipe db~~ | — | ✅ 1,477 judgments + 16 sessions to `soft-launch/results/`, tables truncated |
| ~~Ship full 3,000-pair file~~ | — | ✅ `data/pairs.json` in faceiq-rating = 3,000 pairs, 20 golds all present |
| ~~Image-load handling in faceiq-rating~~ | Dit | ✅ shipped 2026-07-30 — buttons gated until both photos render, error + retry UI, 12s hang timeout |
| ~~Verify run 2 deploy~~ | — | ✅ 2026-07-30 — `/api/monitor` reports 2,980 real pairs all at 0 votes, 0 sessions; anonymous URL returns 200 |
| ~~Re-review 3 suspect golds~~ | Dit | ✅ superseded by run 2 — 20 golds measured at ~90 views each; 3 retired and replaced 2026-07-31 |
| ~~Replace the 3 broken golds~~ | — | ✅ `scripts/replace_golds.py --write`; still 20 golds (8L/12R), `revision: 2`, shipped to `faceiq-rating/data/` |
| ~~Uneven coverage from concurrent starts~~ | — | ✅ fixed 2026-07-31 — `buildQueue()` reserves live sessions' pairs; replay of run 2 gives 12–13 votes/pair vs 8–43 |
| 13 pairs reuse a worked-example face | Dit | accepted — 0.43% of the draw; example *pairs* are excluded but 6 faces recur |
| `/api/export?table=judgments` 500s at scale | — | Broke at 1,477 rows; run 2's 37.8k needed the psql path in Stage 6. Paginate the route or delete it |
| ~~Top up Prolific~~ | Dit | ✅ run 2 published and paid 2026-07-31 |
| ~~Prolific run 2 study + completion code~~ | Dit | ✅ `C1QWP0PC`, 360/360 submitted |
| ~~Deploy the queue + gold changes~~ | Dit | ✅ live for run 3 — coverage came out 6–7 votes/pair (was 8–56), all 20 golds ≥70% |
| ~~Run separation without wiping~~ | — | ✅ `CURRENT_STUDY_ID` scopes `buildQueue()` and `/api/monitor`; run 2 + run 3 coexist in one database |
| ~~**Refit BT with panel vote shares**~~ | — | ✅ 2026-07-31 `artifacts/bt-refit-v3-panel`, all gates. Held-out close-pair accuracy **50.6% → 58.3%**; ρ 0.991 vs v2. See `panel-pilot-findings.md` §6 |
| Pull Prolific rater demographics | Dit | Free export, keyed on `prolificPid`. Needed to answer "do preferences differ by rater age/sex/ethnicity" — we cannot today |
| Point Stage 2 draws at **v4** | — | `select_panel_pairs.py` and the strata definitions still read `bt-refit-v2-qc/ratings.csv`. Run 3 drew against v3 via `select_topup_pairs.py`; run 4 must draw against `bt-refit-v4-panel`, since 149 faces moved more than half a /10 point and the close set is therefore not the same set |
| ~~**Wire panel votes into `train.py`**~~ | — | ✅ 2026-08-01. `panel_labels` / `panel_rejects` / `panel_min_votes` / `panel_prior` / `panel_hard` in `TrainConfig`, plus `src/faceiq_pref/panel.py`. Vote share becomes a soft target on the **train split only** so `val_accuracy` stays comparable; 1,927 of 13,136 train pairs (14.7%) take one at `val_fraction 0.5`. Laplace prior stops a 6-0 sweep becoming a target of 1.0 |
| Run the two label arms | Dit | `train-v12-panel-soft` (vote share) and `train-v13-panel-hard` (crowd winner, rounded) against the existing `train-v10` control. ~20–25 min each on the g5.xlarge, then `eval_vs_panel.py`. **This is what gates run 4** |
| ~~Human-grounded eval for checkpoints~~ | — | ✅ `scripts/eval_vs_panel.py` — scores a checkpoint against panel votes, restricted to pairs whose *both* faces are in that checkpoint's own val split |
| Retrain comparator on soft targets | — | After the wiring above. Hold config fixed at `train-v8-arcface-e2e-long` and vary only the labels; ~6.6 min/epoch on CUDA, so ~100 min for 16 epochs |
| Raise `val_fraction` for panel-heavy eval | — | At 0.2 only 331 panel pairs are leak-free; at 0.5 it is 1,928. Either train the eval run at 0.5, or split so panel faces land in val |
| ~~Run 3~~ | Dit | ✅ 2026-08-01 — 4,800 *new* hard pairs × 6.2 votes, **301** raters (299 completed), $1,286. Chose breadth over depth on the value curve. Marginal gain **+3.18 pts** on withheld pairs, $404/point |
| ~~Refit BT on run 2 + run 3~~ | — | ✅ `artifacts/bt-refit-v4-panel`, all gates; ρ 0.9905 vs v3. `refit_bt_panel.py` now pools runs on `pairIndex` |
| ~~Run 4 = another ~4,800 hard pairs~~ | Dit | ✅ **superseded and completed as a uniform random draw instead** — 2,500 pairs × 12 votes, 303 raters, $968.58, 2026-08-04. See the Run 4 outcome above and log §5.8 |
| ~~Point Stage 2 draws at v4~~ | — | ✅ resolved the other way: draws must use a ranking that has **not** seen the votes. `select_random_pairs.py` and `select_topup_pairs.py` both refuse a panel-fitted `--ratings`; run 4 drew against `bt-refit-v2-qc` |
| ~~Re-derive the /10 calibration~~ | Dit | ✅ done out-of-sample on run 4's unfitted pairs (log §5.8): 0.25 /10 = 53.0%, 1 pt = 65.4%, 2+ = 82.7%, and `σ(Δθ)` overconfident by 26 pts. **Band the score; do not show decimals.** Temperature-scale before publishing any probability |
| Retire gold `cmr1mr5cw06y796d59gfjfwv7` | Dit | 63% over 78 views, third independent failure. Gold set drops to 19 — top up before the next launch via `replace_golds.py` using run 4's unanimous non-gold pairs as the candidate pool |
| Reject `6a4aaa3ca9cd50f4e76237c3` on Prolific | Dit | 294 ms median, 88% fast clicks — ~4× below the cohort 1st percentile. Pay `6a0f76e7…`, `6a040eed…`, `69c826ee…` and drop their votes |
| **Run `train-v16-panel-run4`** | Dit | Config written. Folds run 4 into `panel_labels` on the **soft** target (reverses the 2026-08-01 hard-target decision, which was measured on hard pairs only). Panel coverage of the train split 14.7% → 19.5%; leak-free eval set 1,928 → 2,559 pairs. **This is the top priority in the whole programme** — log §5.8 measured the comparator 1.8–3.1 pts *behind* the ranking on typical pairs |
| Give `refit_bt_panel.py` QC defaults | — | `--exclude-faces` / `--exclude-genders` default to nothing, so a refit silently ranks 2,999 faces and passes all gates. Copy `eval_vs_panel.py`'s defaults |
| Fix the tie controls | Dit | Two separate reports now: run 3's "too close to call" wording, and run 4's "was I supposed to hit Skip or Save & Continue?". The two controls are ambiguous *together*, not just individually mislabelled |
| ~~Pull Prolific demographics~~ | Dit | ✅ Both studies, 667/667 raters matched. Saved to `labels/*/results/demographics.csv` (gitignored — personal data) |
| Per-viewer-demographic labels (Alex) | — | Deferred with numbers now, not just an opinion. A matched in-group vs out-group test finds a **real but small** cohort effect: +2.36 pts for 55+, +1.39 White, +1.04 Female, and pure noise on Asian (41 raters) and Other (50). Confounded with rater consistency. Small cells need 8–16× more recruiting and Prolific's representative sample cannot target them |
| Rater age mix | — | **Not a sampling error.** 38.2% of raters are 55+ against 39.0% of US adults; every band within 0.6 pts. But "US adults" ≠ "our users". Reweighting to a young mix flips 14.2% of winners, of which **12.7% is the noise floor** of the reweight itself (permutation null) — only ~1.5 pts is age. Real (4.8 sd) but small. Decide the target population deliberately; don't re-run anything. Options in `panel-study-playbook.md` §8 |
| Reword the tie button | Dit | A rater flagged that "too close to call" felt wrong for "neither is attractive". Same observation for BT, but the label misleads on an all-near-tie draw |
| Second AI-face QC pass | Dit | Two run 3 raters volunteered that some faces looked AI-generated, after Stage 0 removed 133. The exclusion list is incomplete |
| Screen for recognisable people | Dit | A rater identified a public figure in the cohort. Reputation contaminates an attractiveness judgement, and it is a privacy question |
| Re-derive the /10 calibration | Dit | Resolution is ~2.5 points for 75% agreement; decide whether to band the score rather than show decimals |
| ~~Feedback box on the done screen~~ | — | ✅ done — `Session.feedback`, shown on `/monitor` |
| Privacy/legal sign-off for third-party raters | Dit | execution plan Step 1.7 |
