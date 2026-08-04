# Panel pilot — execution plan (next actionable steps)

*2026-07-23 · Follows from `pilot-500-results-brief.md` (results) and
`human-panel-pilot-plan.md` (design rationale + decision gates). This doc is the
"what do we actually do, in order" version.*

## The plan in one paragraph

Clean the face metadata first (Step 0), then run a paid panel pilot: ~3,000
existing pairs × **12 votes per pair** (~36k judgments, ~$2–5k all-in on
Prolific), stratified toward close/mid pairs. Twelve *votes*, not twelve
*people* — see "Why not a fixed panel of 12?" under Step 4. If the pilot's majority votes are stable and beat our
two-person 68% agreement on hard pairs, we panel-relabel the noisy ~15k slice of
the existing 52k (close/mid pairs + medium-confidence labels), **keep** the ~37k
high-confidence clear/very_clear VLM labels (pilot showed they're at human
ceiling), blend, refit BT, retrain the comparator. Only if that shows improvement
do we commit to larger-scale labeling — and new pairs get generated adaptively
near the BT margin, not as a flat random 100k.

---

## Step 0 — Data cleaning BEFORE sampling panel pairs (~2–3 days, ~$10 compute)

We pay per judgment; don't pay raters to judge junk. Known issues: a handful of
face-level **gender mislabels** (face tagged male that is visually female → ends up
in M–M matchups; the pair logic itself is fine, 0 cross-gender pairs exist in the
export) and possibly a few **synthetics** that survived curation (16 were removed;
pilot-500 surfaced 1 more).

1. **VLM metadata QC pass over all 3,000 faces** (not preference labels — VLMs are
   good at objective checks): per face ask "apparent gender?" + "is this a real
   photograph of a real person (not AI-generated/anime/render)?" + "usable quality?".
   One cheap Gemini Flash call per face ≈ a few dollars.
2. **Human review of flags only.** Expect a few dozen flags; Dit or Alex eyeballs
   them in an hour.
3. **Fix in faceiq-labs** (canonical DB): correct gender metadata, mark bad faces
   with the existing exclusion mechanism (`synthetic`/`quality` + exclude edges from
   GT export), re-export.
4. **Materiality rule:** if <~2–3% of faces are bad, fix and move on — BT and the
   comparator are robust to dropping a face's ~35 edges. Only if it's much worse do
   we discuss re-drawing the cohort (currently no evidence for that).

Deliverable: cleaned export + a one-paragraph QC note in the research log.

---

## Step 1 — Requirements to lock BEFORE requesting any quote

Yes — vendors quote against a spec. We need, in writing:

1. **Task spec (1 page).** Two photos side by side; forced choice "who is more
   attractive?" + "too close to call" (tie); one optional free-text comment on hard
   pairs. Same-gender pairs only. Identical framing to the VLM prompt so results
   are comparable.
2. **Labeling instruction sheet.** Short — this is a gut-preference task, NOT a
   checklist task. Instructions bias raters; keep to: judge facial attractiveness
   only, ignore photo quality/background/clothing where possible, no "right answer",
   work quickly (first impression), ties are allowed but should be rare.
   Include 3 worked examples (one clear, one close, one tie). Draft from the
   pilot-500 labeler UI text we already used.
3. **Sample definition:** ~3,000 pairs from the cleaned export (see Step 2).
4. **Ratings per pair:** 12 (literature says 10–15 stabilizes pairwise majorities).
5. **QC design:** ~5% gold pairs seeded into every rater's queue. We already have
   the golds for free: pilot-500 very_clear pairs where Dit + Alex + Gemini all
   agree (46+ available). Reject/replace raters below ~80% on golds; monitor
   per-rater agreement-with-majority and response-time floors (<2s = clicking
   through).
6. **Rater demographic quotas** (Step 4).
7. **Privacy/legal sign-off** — required before any vendor sees photos. The 3k
   research cohort was cleared for VLM labeling; confirm ToS/DPA covers third-party
   human raters. (The 1M user-photo pool is a separate, stricter question — not in
   scope.)

---

## Step 2 — Pair sample (~3,000 pairs from the existing export)  ✅ DRAWN 2026-07-27

No new matchups needed for the pilot. Stratified draw
(`scripts/select_panel_pairs.py`, seed 20260727 — 3,000 pairs, 2,463 faces,
1,499 M / 1,501 F, every stratum and ethnicity floor hit; see runbook Stage 2):

| Stratum | share | why |
|---|---|---|
| close (p < 0.60) | ~35% | where labels are noise; the pilot's main question |
| mid (0.60–0.80) | ~25% | second-noisiest segment |
| clear + very_clear | ~20% | measures ceiling + validates keeping VLM labels |
| medium/low VLM confidence (any bucket) | ~10% | pilot-500 says these are junk — confirm at scale |
| overlap with the 750-pair human audit + pilot-500 | ~10% | free cross-check of prior labels against the panel |

Ensure demographic coverage per group (≥150 pairs per major ethnicity group).
**Resolved 2026-07-25:** no VLM pass needed for this — self-declared ethnicity
for all 3,000 faces was joined offline from the Labs manifest (runbook Stage
0.5). Six groups have enough faces to support the floor; south_asian female is
borderline and the native_american/pacific_islander tail is too small, so pool
it. The VLM QC pass still contributes skin tone, which is stored nowhere else.

Attention checks and instruction-sheet examples came out of the same draw: 33
golds (`scripts/make_golds.py`) and 3 worked examples (`scripts/make_examples.py`),
both restricted to QC-clean faces. Runbook Stage 3 has the selection rules.

## Step 3 — Tooling: do we build a web app?

**Recommended: Prolific + our own lightweight rating page.** We control UX and data;
Prolific handles recruitment, demographics, quotas, payments, consent.

- **Not** the Streamlit labeler (single-user, no concurrency).
- **Not** inside faceiq-labs — no research tooling in the prod app. Build a
  **standalone throwaway rating app** (own tiny repo, own Vercel project, own
  votes table/KV store). It only needs: pair viewer, keyboard shortcuts, progress
  bar, one row written per judgment. Photos load from the existing public blob
  URLs, so it needs no access to prod data or DB.
- **No user accounts needed.** Prolific passes `PROLIFIC_PID` (+ study/session IDs)
  as URL params — that's the rater identity. Completion code at the end closes the
  loop. Estimated build: **2–4 dev-days** including QC logic (gold insertion,
  timing capture).
- Alternative if we want zero build: managed vendor (Surge/Toloka managed) brings
  their own UI at ~2–3× the per-judgment price and gives us less control over rater
  demographics and raw data. Worth getting one quote as a comparison point.

## Step 4 — Raters: who, and what we collect about them

**Not random — quota-stratified.** If the panel skews (e.g. mostly young US men),
"majority preference" bakes that skew into the GT. Balance:

- **Gender: ~50/50** male/female raters, and ensure both rate both pair genders —
  this directly measures whether e.g. male and female raters disagree about male
  faces (they might; that's data, not noise).
- **Ethnicity:** cohort is 69.2% white (confirmed against the manifest join), so
  cross-ethnicity rating patterns are a first-class question. **Correction:**
  Prolific does *not* do arbitrary ethnicity quotas — its quota sample balances
  **sex only**. Ethnicity balance comes from a *representative* sample
  (US census-proportional, min 300 participants), or from separate prescreened
  studies. Plan on representative + an optional booster study per runbook Stage 5.
- **Age:** spread across 18–45+ bands.
- **Region:** mostly US/EU for v1 (product market), note as a known limitation.

**Collected per rater (via Prolific's screening fields + a 30-second intake):**
age band, gender, ethnicity, country, and optionally sexual orientation (relevant
to "attractiveness as perceived by whom"; sensitive — collect only via Prolific's
existing opt-in screening, never our own form, and store only as aggregate strata).

**Collected per judgment:** pairId, choice (left/right/tie), response time,
presentation order, rater ID (pseudonymous), timestamp.

**Why this matters beyond QC:** with rater demographics attached to every vote we
can compute per-group majorities and cross-group agreement — i.e., "do preferences
differ by rater gender/ethnicity, and where?" That is (a) the bias audit for
Gemini, (b) a future product option (preference-conditioned scoring), and (c) a
dataset nobody else has at this granularity. It costs nothing extra to collect.

### Why not a fixed panel of 12 rating all 3,000?

Worth stating explicitly, because "3,000 pairs × 12" reads like twelve people
doing everything, and because that is *literally what run 1 was* — the soft
launch was a **complete design**: 12 raters, the same 100 pairs, everyone rated
every pair. Run 2 is an **incomplete (overlapping) design**: 360 raters, 100 pairs
each, and every pair still collects exactly 12 votes — just from a different
twelve people. `buildQueue()` enforces this by always handing out the
*least-judged* pairs, so coverage converges to 12 per pair rather than drifting.

A fixed panel of 12 covering all 3,000 would actually be **cheaper** (12 × ~2.5 h
× $20 ≈ $800 vs ~$1.9k) and would give textbook inter-rater reliability, since
every rater shares every item. We are not doing it for three reasons:

1. **Twelve people cannot carry a population claim.** The GT we are building is
   "what do people find more attractive," and it feeds a product scoring users.
   With n=12 the majority label is twelve individuals' taste, permanently baked
   in. With 360 census-matched raters each pair's 12 votes are a random draw from
   a population, so the majority *estimates* something that generalises. Run 1
   made this concrete: only **1%** of pairs were unanimous and the winning side
   averaged **63%** of 12 votes. When dispersion is that high, whose twelve
   opinions you sampled matters enormously.
2. **The bias audit becomes impossible.** Step 4's whole point is measuring
   whether preferences differ by rater gender and ethnicity. That needs hundreds
   of raters spread across demographic cells. Twelve people have no cells.
3. **3,000 judgments in one sitting is ~2.5–3.5 hours.** Fatigue would degrade
   quality monotonically through the session and confound pair difficulty with
   position in the queue, and Prolific dropout on multi-hour tasks is severe.

**What the incomplete design costs us, and why it doesn't matter here.** Two
specific raters share only ~3 pairs on average (100 × 100/3,000), so
rater-vs-rater agreement statistics are out of reach. Every gate in Step 6 is
computed either per-pair or rater-vs-majority, and both survive intact:

| Quantity | Needs the same raters? | Why it works |
|---|---|---|
| Majority label per pair | No | 12 independent votes per pair, by construction |
| Split-half stability | No | Split *that pair's* 12 votes into 6 + 6, compare majorities, average over 3,000 pairs |
| Gemini-vs-majority `G` | No | Per-pair comparison |
| Human-vs-majority `H` | No | Leave-one-out: each vote vs the majority of the other 11 |
| Per-rater quality screen | No | Each rater has 100 own-vote-vs-majority comparisons, plus 5 golds |
| Vote entropy | No | Per-pair distribution |
| Krippendorff-style rater reliability | **Yes** | Not estimable — and not one of our gates |

The one real limitation is **subgroup power**, not design validity. A
census-proportional 360 gives roughly 6 female and 6 male votes per pair (fine),
but only ~1.6 Black and ~0.7 Asian votes per pair, so *per-pair* minority-group
majorities are not estimable. Aggregate group agreement rates still are (~47 Black
raters × 105 judgments ≈ 4.9k votes). If the cross-ethnicity audit needs per-pair
resolution, that is what the runbook's booster study is for.

## Step 5 — Cost

| Item | estimate |
|---|---|
| 3,000 pairs × 12 raters = 36,000 judgments @ ~$0.03–0.05 (Prolific, ~5s/judgment at fair hourly rate) | $1.1k–1.8k |
| Prolific platform fee (~33%) | $0.4k–0.6k |
| Quota targeting premium + re-runs/rejected raters (~15%) | $0.2k–0.4k |
| Rating page build (internal dev time, 2–4 days) | internal |
| **All-in** | **~$2k–3k** (budget $5k ceiling) |

Managed-vendor alternative: $3.5k–6k for the same volume, less control.

## Step 6 — Run + analyze (1–2 weeks end to end)

1. Soft launch: 100 pairs × 12 raters (~$100). Check gold pass-rates, timing
   distributions, tie rates, UI bugs. Fix, then full launch (Prolific typically
   fills 36k judgments in days).
2. Analysis = the decision-gate computation already specified in
   `human-panel-pilot-plan.md` §4: human-vs-majority ceiling `H`, Gemini-vs-majority
   `G`, model-vs-majority `M`, per bucket and per demographic stratum; plus
   majority-vote stability (split-half: does majority-of-6 replicate majority-of-
   other-6?) and per-pair vote entropy vs model-ensemble spread.
3. Write results to research log §5.5.

**What run 2's 3,000 labels are actually for** — three things, in order of value:

1. **A decision.** Do the Step 7 gate. This is the primary deliverable: it decides
   whether to spend $4–6k relabeling the noisy 15.3k slice, or to stop paying for
   close-pair labels altogether.
2. **A permanent human-GT test set.** 3,000 pairs with majority human labels and
   full vote distributions is the benchmark every future BT refit and comparator
   retrain gets measured against. This is reusable forever and is why the data is
   worth keeping even if the gate fails.
3. **Label replacement for those 3,000 pairs**, joined into the export in place of
   their VLM labels. Real but modest on its own: 3,000 of 52,414 pairs is 5.7% of
   the graph, so refitting BT on it alone will barely move rankings. The volume
   that changes the model comes from Step 7's 15.3k, not from here.

So it is **not** "just an inter-rater agreement check," but it is also **not** the
retraining run. It is the measurement that authorises the retraining run.

## Step 7 — The gate, then scale (only if passed)

> **RESULT 2026-07-30: the gate FAILED on close pairs and PASSED on clear pairs.**
> Close-pair split-half stability came in at **63.4%** against the 68.5% bar, with
> a human ceiling of 64.6% — people don't agree with each other there, so the
> labels aren't the problem. Meanwhile Gemini hit **81.0%** on clear pairs versus a
> 74.6% human ceiling, i.e. **better than an average human**. Read the "Fail"
> bullet below for close/mid and the "Pass" bullet for clear. Net effect: **do not
> run the $4–6k relabel** — it was aimed almost entirely at the close/mid slice
> that humans cannot resolve either. Full numbers in the runbook, "Run 2 outcome".

> ### ⚠️ SUPERSEDED 2026-07-31 — the close-pair "fail" was a bad gate, not a bad result
>
> **The gate itself was wrong**, so do not act on the verdict above or on the "Fail"
> bullet below. It scored close pairs on *majority* agreement, which is near chance
> there by construction — that measures whether a coin lands the same way twice, not
> whether the pair carries signal. Scored on **vote share** instead, close pairs
> replicate at split-half **0.655** and 39% reach ≥75% agreement. The signal was
> always there; the statistic hid it.
>
> What actually happened next, and why it was right: we spent on close pairs anyway.
> Run 3 bought 4,800 of them for $1,286 and returned **+3.18 points** of held-out
> accuracy (5/5 seeds), with the dose curve still accelerating. Had we followed the
> "Fail" branch we would have stopped the programme at exactly the point it started
> paying.
>
> Current policy lives in `panel-study-playbook.md`; results in
> `panel-pilot-findings.md` and `scoring-gt-research-log.md` §5.5. This section is
> kept unedited below as the pre-registered gate, so the write-up can be honest about
> which predictions failed.

**Gate:** split-half majority stability on close pairs meaningfully above our
two-person 68.5%, and majorities that replicate across rater subgroups.

- **Pass →** panel-relabel the noisy slice of the existing 52k:
  **11,391 close+mid pairs + 3,929 medium/low-confidence clear pairs ≈ 15.3k pairs
  × 5 votes ≈ 77k judgments ≈ $4k–6k.** Blend with the kept 37,083 high-confidence
  clear/very_clear VLM labels (yes — the VLM is genuinely good there: 80–88% vs
  human consensus). Refit BT, retrain comparator, measure vs the panel-majority
  test set the pilot produced.
- **Improvement confirmed →** then and only then scale further: generate NEW
  matchups adaptively near the current BT margin (close/mid by construction) and
  send them straight to the panel. A flat random 100k would be ~58% very_clear —
  wasted budget regardless of who labels it.
- ~~**Fail (close-pair majorities don't replicate) →** close pairs are irreducibly
  subjective: treat them as soft ties in training, stop paying anyone to label
  them, and invest in more clear-pair coverage instead.~~
  **Struck 2026-07-31 — see the superseded note above.** Majorities not replicating
  turned out to mean the majority was the wrong statistic, not that the pairs were
  subjective. Two halves of this bullet survive in current policy for unrelated
  reasons: close pairs *should* be soft targets in training (now implemented via
  `panel_labels`), and clear-pair coverage is genuinely not worth buying — but
  because Gemini already handles it, not because humans failed.

## Owner checklist (proposed)

| # | Task | Owner | Blocked by |
|---|------|-------|-----------|
| 0 | VLM metadata QC pass + fix mislabels in labs, re-export | Dit | — |
| 1 | Privacy/legal check for third-party raters | Dit | — |
| 2 | Instruction sheet + task spec (1-pager) | Dit | — |
| 3 | Pair sampling script extension + gold set | Dit | 0 |
| 4 | Standalone rating app (separate repo/deploy) + Prolific integration | dev | 2 |
| 5 | Prolific study setup w/ quotas; soft launch | Dit | 1,3,4 |
| 6 | Full run + gate analysis | Dit | 5 |
