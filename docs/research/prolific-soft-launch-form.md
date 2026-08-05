# Prolific study form — runs 1 and 2

Field-by-field fill sheet, in the order Prolific's study creation form presents
them. §1–§6 are **run 1 (soft launch)**, kept as the record of what was actually
published on 2026-07-30. **§7 is the run 2 sheet** — the panel pilot, 3,000 pairs
× 12 votes, and the one you fill next.

Run 1 is complete: 12/12 submissions in ~13 minutes, all paid out. The numbers it
produced (median 5.9 min of rating, 85% gold pass rate) are what set run 2's
length and reward below, and they cut the estimate from 20 minutes to 12 —
roughly **$1.3k off** the original projection.

Companion docs: `panel-pilot-runbook.md` Stage 5 (why these choices), Stage 4
(shipping the data — do that **before** publishing).

## 1. Data collection type

**External study link.** Not AI Task Builder, not Survey Builder — the task is
our own Next.js app.

## 2. Study details

| Field | Value |
|---|---|
| Study name (participant-visible) | `Quick photo comparison task (~20 min)` |
| Internal study name | `panel soft launch v1 — 100 pairs × 12 raters` |
| Study label | **Decision making** |
| Devices | **Desktop only** — untick Mobile and Tablet |
| Audio / Camera / Microphone / Software download | all unticked |
| Content warning | None |

Replace what's in the draft now. `Human judgment & response quality evaluation`
has two problems: it's vague enough that Prolific may query it, and "response
quality evaluation" tells raters they're the ones being evaluated, which is how
you get performance rather than instinct.

Desktop-only is not cosmetic. The layout puts two photos side by side with
keyboard shortcuts, and response time is one of the numbers we're measuring — a
phone would distort it even where the layout survives.

**Study description** (paste as-is):

> In this short task you will see pairs of face photographs, side by side, and
> pick whichever person you find more attractive. There is no right answer — we
> are interested in your immediate, instinctive reaction, so please work quickly
> rather than deliberating. You can also mark a pair as too close to call.
>
> The task takes about 20 minutes and runs on desktop only. Full instructions and
> a few worked examples are shown before you start, and stay available from a
> button throughout.
>
> If anything goes wrong — a photo that won't load, a page that freezes — please
> message us through Prolific. There is also an optional feedback box at the end.

Nothing in the name or description may hint at *criteria* — no symmetry, bone
structure, or features. Naming them primes the analytic checklist mindset this
whole ground-truth effort exists to replace. Revealing the topic is fine and
unavoidable; revealing how to score is not.

## 3. Data collection

| Field | Value |
|---|---|
| Study URL | see below |
| Recording Prolific IDs | **URL parameters** |
| Custom screening | **No** |
| Process submissions | **Manually review** |
| Add to participant group | create **`faceiq-soft-launch`** |
| Completion path | **Redirect URL** (the default) |

Study URL — paste exactly, placeholders included:

```
https://faceiq-rating.vercel.app/?PROLIFIC_PID={{%PROLIFIC_PID%}}&STUDY_ID={{%STUDY_ID%}}&SESSION_ID={{%SESSION_ID%}}
```

Use `faceiq-rating.vercel.app`, the project production alias — verified 200 for
an anonymous visitor on 2026-07-29. Do **not** use a per-deployment URL like
`faceiq-rating-3tvlk95q4-face-iq-labs.vercel.app`: those are pinned to one build
and will not pick up later data pushes. The params are mandatory; without
`PROLIFIC_PID` the app shows "Missing study link" and the rater cannot start.

**Manually review**, not auto-approve: the ~80% gold gate needs a human in the
loop. Approve generously when you do — rejections damage a rater's standing on
the platform, so reserve them for clear bad faith.

`faceiq-soft-launch` is the exclusion hook for run 2. These 12 people will have
seen 100 of the 3,000 pairs, and letting them vote on those again in the main run
would double-count one opinion.

**Completion code:** the form shows `CGT1P6T0`. That exact string must be set in
Vercel as `PROLIFIC_COMPLETION_CODE` (Project → Settings → Environment Variables →
Production), then redeploy so it takes effect. If the code shown when you publish
differs from what's in Vercel, every rater ends the task with a code Prolific
rejects.

## 4. Recruit participants

| Field | Value |
|---|---|
| Participants | **12** |
| Filtering | **Choose filters** → Country of residence = United States · Approval rate ≥ 95% · First language English |
| Study distribution | **Standard sample** |
| Credentials | **No** |
| Total times a participant can complete | **Once** |
| Automatically reject exceptionally fast submissions | **No** |

Keep the filters light — every one slows the fill, and the pool is 312k active.
The demographic balance work belongs to run 2's representative sample, not here.

Auto-reject stays off: legitimate per-item responses are ~6 seconds, so a
platform-level speed rule would fire on honest raters. Our own gold pass-rate and
the 2-second-floor click-through flag on `/monitor` do this better because they
look at per-item behaviour instead of total duration.

## 5. Reward

| Field | Value |
|---|---|
| How long will your study take | **20 mins** |
| Reward | **$6.67** (= $20.00/hr) |

The task is 105 items (100 real pairs + 5 golds) at roughly 6.5 s each ≈ 12
minutes, plus consent and instructions ≈ 15 minutes realistic. Declare 20 to keep
headroom: Prolific asks for a top-up if the *median* exceeds your estimate, and
the auto-timeout scales off this number too.

$20/hr is the bottom of the competitive band Prolific quotes for this audience
($20–26/hr, high competition). Cost: 12 × $6.67 = $80.04 plus the 33% platform
fee ≈ **$106 all-in**, against your $150 balance.

You could pay $4.33 (=$13/hr, ≈$69 all-in) and save ~$37, but the soft launch is
a blocking step — you're waiting on it to find bugs before spending $2k — and
underpaying in a high-competition audience buys a slow fill. Pay for speed here.

## 6. Before you hit Publish

1. Ship the data (runbook Stage 4) and confirm the deploy picked it up. The app
   must be serving the 120-pair soft-launch file, not the 40-pair placeholder.
2. `PROLIFIC_COMPLETION_CODE` = the code on this form, in Vercel Production.
3. `ADMIN_PASSWORD` set (`openssl rand -base64 24`) — it gates `/monitor` and
   `/api/export`.
4. `PAIRS_PER_SESSION` — leave unset. The code defaults to 100.
5. Do a full test run yourself at
   `https://faceiq-rating.vercel.app/?test=1` → **Start a test session**. Those
   judgments are tagged `studyId = "test"` and are excluded from queue coverage
   and from analysis.
6. Anonymous access check (a logged-in browser proves nothing):

```bash
curl -s -o /dev/null -w "%{http_code}\n" \
  "https://faceiq-rating.vercel.app/?PROLIFIC_PID=test&STUDY_ID=test&SESSION_ID=test"
```

Want **200**. A 302 to `vercel.com/sso-api` means Deployment Protection is back
on and raters would hit a login wall.

## 7. Run 2 — the panel pilot (fill this next)

**This is the actual experiment.** 3,000 pairs × 12 votes = 360 submissions,
producing the majority-vote human ground truth the execution plan's Step 6 gates
are computed on. Run 1's data is archived and the database is wiped, so run 2
starts from zero coverage.

Everything not listed below is **unchanged from §1–§6** — external study link,
Decision making label, desktop only, URL parameters, manual review, no custom
screening, no auto-reject, once per participant.

### 7.1 The nine fields that change

| Field | Run 2 value |
|---|---|
| Study name (participant-visible) | `Quick photo comparison task (~10 min)` |
| Internal study name | `panel pilot v1 — 3,000 pairs × 12 votes` |
| Participants | **360** |
| Study distribution | **Representative sample** — US, census-matched on age + sex + ethnicity |
| Filtering | **greyed out — none available.** See §7.1a |
| Add to participant group | create **`faceiq-panel-run2`** |
| How long will your study take | **10–12 mins** — see §7.2 |
| Reward | **$3.33 at 10 min / $4.00 at 12 min** (both = $20.00/hr) |
| Completion code | whatever run 2's form shows — **must match Vercel** (see §7.5) |

The `(~10 min)` in the study name and the estimate differ on purpose: the name is
what a participant skims, the estimate is what Prolific's underpayment check
measures against. Round the name down to the honest typical experience, keep the
paid estimate at or above it.

### 7.1a Representative samples accept no filters — confirmed 2026-07-30

Selecting **Representative sample** greys out the entire screener section. This is
by design, not a UI bug: Prolific's docs state prescreeners are "only available
for balanced and standard samples," because any extra filter breaks the census
proportions the sample exists to guarantee. Three consequences, none fatal:

**1. No approval rate ≥ 95%.** Run 1 had this filter; run 2 cannot. Quality
control falls entirely to our own gold gate, which is the better instrument
anyway — it measures behaviour on *this* task rather than platform history. Keep
`Process submissions: manually review` and budget a few percent of submissions for
rejection and replacement.

**2. Excluding the soft-launch raters still works** — it lives in a *different*
section. Prescreening filters are greyed out, but **Recruit participants → Block
participants → Exclude participants from previous studies** is separate and
remains available on a representative sample. Select `panel soft launch v1 — 100
pairs × 12 raters` there. Confirmed present in the run 2 draft on 2026-07-30.

Note this is a nice-to-have, not load-bearing. An earlier draft of this doc called
it mandatory on the assumption that a returning rater would re-see the same 100
pairs. They wouldn't: run 2 assigns 100 pairs out of 2,980, so a returning
soft-launch rater would overlap their own previous slice by only
100 × 100/2,980 ≈ **3.4 pairs** — worst case ~41 of 36,000 votes, 0.11%. Use the
blocklist because it's free, not because the study depends on it. One caveat from
Prolific's docs: only participants who finished the earlier study *before you
publish* are excluded.

**3. No "First language English."** Irrelevant for a visual forced-choice task
with a short instruction sheet, in a US census sample.

**Don't switch to a standard/balanced sample to win the filters back.** Balanced
samples balance **sex only**, which would leave rater ethnicity at whatever the
active Prolific pool happens to be — and cross-ethnicity rating patterns are a
first-class deliverable (execution plan Step 4). The census match is the reason
we're paying for a representative sample; the filters are not worth trading it for.

**One representative-sample gotcha to watch:** if the study is still awaiting
submissions 48 hours after launch, Prolific offers **reallocation**, which drops
the age-bracket restrictions (and sometimes sex/ethnicity) on unfilled places to
speed up the fill. It works, but it silently degrades representativeness. If you
accept it, note it in research log §5.5 as a limitation on the demographic
analysis — that analysis is the whole reason for this sample type. Also note
representative samples are the one study type whose places **cannot** be increased
after publishing, so get the 360 right the first time.

### 7.2 Why 12 minutes, and what it saves

Measured from run 1 (`scripts/analyze_soft_launch.py`, 12 completed sessions):

| Measure | Run 1 actual |
|---|---|
| Rating time, median | **5.9 min** |
| Rating time, p25 / p75 | 4.5 / 6.7 min |
| Rating time, slowest rater | 11.6 min |
| Per-judgment median | 2.2 s |

Those are **rating time only** — the session row is created when the rater clicks
"I consent — start rating", so consent, instructions, and the three worked
examples all happen before the clock starts. Prolific measures from study
acceptance, so add reading time: median ≈ 6 + ~3 = **~9–10 minutes** end to end.

**8 minutes is tight.** It leaves ~2 minutes for a first-timer to read the consent
block, the instruction sheet, and three worked examples, and 5.9 min of rating is a
*floor* we measured, not an estimate. The form is explicit about the consequence:
"if the median completion time exceeds this estimate we will ask you to make
additional payments."

Timeout is **not** a concern — an 8-minute estimate yields a 25-minute maximum
allowed time, and run 1's slowest rater spent 11.6 min rating (~15 total). The only
real risk is the top-up request.

**Settle it with run 1's own number:** the completed study page on Prolific shows
the actual median completion time, measured from acceptance — the same basis run 2
will be judged on. Decision rule:

| Run 1 median on Prolific | Set the estimate to |
|---|---|
| ≤ 7 min | 8 is fine, keep it ($1,414) |
| 7–9 min | 10 min ($1,769) |
| > 9 min | 12 min ($2,057) |

Declaring **12** gives headroom without being the 20 we guessed blind:

| Length × rate | Reward | 360 subs | + 42.86% fee |
|---|---|---|---|
| ~~20 min @ $20/hr~~ (old guess) | ~~$6.67~~ | ~~$2,401~~ | ~~**$3,430**~~ |
| **12 min @ $20/hr** ← safest | **$4.00** | $1,440 | **$2,057** |
| **10 min @ $20.6/hr** ← recommended | **$3.44** | $1,238 | **$1,769** |
| 8 min @ $20.6/hr — see the risk above | $2.75 | $990 | **$1,414** |

**The platform fee is 42.86%, not the 33% an earlier draft of this doc used.** The
run 2 draft form prices 360 × $2.75 = $990.00 with a $424.29 platform fee, which
is 42.857%. Run 1 confirms it independently: $150.00 funded − (12 × $6.67 ×
1.4286 = $114.34) = **$35.66**, exactly the balance Prolific shows. 33.3% is the
academic/non-profit rate and does not apply to this workspace. Every all-in figure
elsewhere in the panel docs that used 33% is understated by ~7 points.

The gap between 8 and 10 minutes is **$355**. Weigh that against being asked to
top up 360 submissions mid-run.

Take the $20/hr row. Run 2 uses a **representative** sample, which fills far more
slowly than run 1's standard sample — Prolific has to find census-matched people
in the slow cells (older, non-white, lower-incidence combinations), and
underpaying there costs days, not dollars. The saving already comes from cutting
the *time* estimate in half; don't also cut the rate and stall the fill.

Verify the fee percentage against your run 1 invoice before budgeting: run 1
billed ≈$106 on $80.04 of reward, i.e. **33%**. If your workspace is on the 42.8%
corporate rate instead, the recommended row becomes **$2,056**. Either way it
lands inside the execution plan's $2–3k envelope, where the old 20-minute
estimate did not.

**Top up the workspace before publishing.** The balance after run 1 is **$35.66**,
so you need to add roughly **$1,380** (8 min), **$1,735** (10 min), or **$2,025**
(12 min). Prolific blocks publishing on an insufficient balance. Prolific's help
docs say representative samples are the one study type whose places cannot be
increased after publishing, while the confirm-publish dialog says the maximum
number of submissions *can* be increased — don't rely on either. 360 is fixed by
the design anyway (3,000 pairs × 12 votes ÷ 100 per session), so get it right up
front. Everything *other* than submission count is genuinely immutable once
published, including the description.

### 7.3 Keep `PAIRS_PER_SESSION` at 100 — it is load-bearing

Run 1 showed 105 items takes under 6 minutes, so a longer session is tempting:
150 pairs would need only 240 submissions. **Don't.** Prolific's representative
sample has a **300-participant minimum**, and 240 falls under it — you would lose
the census matching that run 2 exists to get. 100 pairs × 360 sessions ÷ 3,000
pairs = exactly 12 votes per pair, and 360 clears the minimum with room.

### 7.4 What run 1 says about rater screening

Run 1's 12 raters scored **51/60 golds (85%)** — the same 85% the human audit got
against the VLM in the original GT run, which is reassuring: it means ~15% miss
rate is the task's natural noise floor, not sloppiness.

Set the rejection rule accordingly:

- **≤2 of 5 golds → reject candidate.** Nobody in run 1 hit this.
- **3 of 5 → approve.** Two raters landed here (`62e02660…`, `67103e4e…`), both
  with ~65% of judgments under 2 seconds. Fast plus mediocre golds is the profile
  to watch, but 3/5 on a 5-item screen is within binomial noise at an 85% true
  rate. Approve and let the majority vote absorb them.
- **Fast clicking alone is never grounds.** 42% of all run 1 judgments were under
  2 seconds and the gold rate was still 85%. This is an instinct task; speed is
  the intended behaviour.

Approve generously — a rejection damages a rater's platform standing, and with
360 submissions the majority vote is robust to a few weak raters in a way that
individual rejections can't improve.

With 360 sessions × 5 golds ÷ 20 golds, **each gold gets seen ~90 times** in run
2, so per-gold pass rates finally become statistically meaningful. Check them
mid-run — see the runbook's Stage 3 note on the three golds run 1 already put
under suspicion.

### 7.5 Before you hit Publish (run 2)

Same checklist as §6, plus what's specific to this run:

0. **Strip the `>` characters out of the study description.** The §2 text is
   written as a markdown blockquote, and pasting it verbatim carries the `> `
   prefixes and the bare `>` spacer lines into Prolific, which renders them
   literally. Paste the prose only. Also make the description's stated duration
   match the estimate you set in §7.1 — the draft says "about 10 minutes" while
   the estimate field says 8.
1. **Data shipped** — ✅ verified live 2026-07-30: `/api/monitor` reports
   **2,980 real pairs, all at 0 votes** (2,980 + 20 golds = 3,000; `realPairs`
   excludes golds by design, so 2,980 is the correct number to see here).
2. **Database wiped** — ✅ 2026-07-30, and re-cleaned after end-to-end testing.
   Confirmed 0 sessions / 0 judgments. Leftover rows would count as coverage and
   starve those pairs of real votes.
3. **Image-load fix deployed** — ✅ shipped 2026-07-30. Photos must render before
   the answer buttons enable; run 1's one bug report was a rater judging a pair
   whose photos hadn't loaded.
4. **Completion code matches** — ✅ verified end-to-end 2026-07-30: production
   `/api/session/complete` returns **`C1QWP0PC`**, the code on the run 2 draft.
   Worth re-checking if you ever regenerate it, because the env var is read at
   runtime — a code changed in Prolific but not in Vercel hands all 360 raters a
   string Prolific rejects. Verify by completing a `studyId = "test"` session and
   reading the code the API returns, rather than trusting the Vercel dashboard.
5. **Test run** at `/?test=1`. Test rows are excluded from queue coverage and from
   `/monitor`, but **delete them anyway** before publishing so the live numbers
   start from a clean zero:

   ```bash
   psql "$DATABASE_URL" -c 'delete from "Judgment" where "sessionId" in
     (select id from "Session" where "studyId" = '"'"'test'"'"');'
   psql "$DATABASE_URL" -c 'delete from "Session" where "studyId" = '"'"'test'"'"';'
   ```

6. **Anonymous 200 check** (the §6 curl) — ✅ 200 on 2026-07-30. Vercel Deployment
   Protection can come back on after a project settings change.
7. **Confirm the four toggles that are easy to leave wrong:** Study label =
   *Decision making*; Devices = **Desktop only** (untick Mobile and Tablet);
   Process submissions = **Manually review** (not Approve and pay — the gold gate
   needs a human); Representative sample criteria = **United States of America
   (Sex, Age, Ethnicity)**, not one of the Political Affiliation or Regional
   variants.

---

# 8. Run 3 — the close-pair top-up

**What changed and why.** Run 2 answered the question it was built for, and the answer
narrowed the target: machine labels already beat an individual human on easy pairs, and sat
at a coin flip on near-ties. So run 3 buys **only near-ties, only pairs nobody has rated**,
and at **6 votes each rather than 12** — measured, not guessed: at a fixed budget, 2,250
pairs × 6 votes beat 1,125 pairs × 12 by 1.8 points on held-out hard pairs.

Data is drawn and shipped: `labels/panel-run-3/pairs.json`, 4,820 pairs = **4,800 new**
(median percentile gap 4.9) + the 20 golds carried over with their original left/right
sides. Golds must ride along or `lib/pairs.ts` throws at boot.

### 8.1 The fields that change from run 2

| Field | Run 3 value |
|---|---|
| Study name (participant-visible) | `Quick photo comparison task (~6 min)` |
| Internal study name | `panel run 3 — 4,800 new close pairs × 6 votes` |
| Participants | **300** — Prolific's representative-sample floor, see §8.2 |
| How long will your study take | **9 mins** — see §8.3 |
| Reward | **$3.00** (= $20.00/hr at the stated 9 min) |
| Add to participant group | create **`faceiq-panel-run3`** |
| Block participants | exclude **`faceiq-panel-run2`** *and* the soft-launch group |
| Completion code | **generate a new one** and mirror it into Vercel (§7.5) |

Everything else is unchanged from §7: representative US sample (Sex, Age, Ethnicity),
desktop only, Decision making, manually review, no auto-reject, once per participant.

### 8.2 Why 4,800 pairs — a Prolific floor drove the size

**A representative sample will not run below 300 participants** (Prolific's range is
300–4,500; confirmed in the UI 2026-07-31). The first version of this plan wanted 200,
which is not available.

That leaves two ways to spend 300 raters, and the run-2 measurements pick one. Keeping the
draw at 3,200 pairs would give 30,000 ÷ 3,200 = **9.4 votes per pair** — depth we already
know is close to worthless, since at a fixed budget 2,250 pairs × 6 votes beat 1,125 × 12.
Buying **more pairs** instead keeps every rater on the efficient part of the curve, so the
draw grew to 4,800 and votes-per-pair stayed at 6.

The bigger draw is better in a second way that matters for the transfer effect the whole
strategy depends on: 4,800 pairs touch 2,757 faces at **3.5 new comparisons each** (vs 2.5
at 3,200), and only 338 faces appear just once (vs 709). Gains travel through face scores,
so faces with a single new comparison contribute least.

The app gives each rater `PAIRS_PER_SESSION` = 100 real pairs and adds golds on top at 5%,
so a session is 105 screens and contributes **100 real votes**.

| | |
|---|--:|
| Real pairs to cover | 4,800 |
| Real votes per session | 100 |
| Participants (Prolific floor) | 300 |
| Votes collected | 30,000 |
| **Votes per pair** | **6.25** |
| Reward × 300 | $900.00 |
| Prolific fee (42.86%) | $257.16 → $385.74 |
| **Total** | **≈ $1,286** ($0.0429/vote) |

More than the $857 the 200-rater version would have cost, but it buys 50% more pairs at the
same per-vote rate, and it is still a fraction of the ~$3,700 the whole blind spot would
take. After this run, roughly half of the sub-10-point band will be covered.

### 8.3 Why 9 minutes and $3.00, when run 2 paid $3.33–$4.00

Run 2's 362 completed sessions took a **median of 5.7 minutes** (mean 6.3, p90 9.4) for the
same 105 screens. It was estimated at 10–12 minutes, so raters were effectively earning
$35–42/hr. Run 3's task is identical in length, so the estimate can come down to the
measured reality without touching the experience.

At a 9-minute estimate the stated rate is $20.00/hr and the *effective* rate at the 5.7-min
median is ≈$32/hr — comfortably generous, far above Prolific's fairness floor, and it keeps
the 10% of raters who take longer than 9 minutes still paid fairly. Do not cut below 8
minutes: p90 is 9.4 and Prolific's underpayment check compares against the estimate.

### 8.4 Two new pre-publish steps

1. **Set `CURRENT_STUDY_ID` in Vercel** to run 3's Prolific study id (visible in the study
   URL once created), then redeploy. This is new in run 3 and it replaces wiping the
   database: coverage counting and `/monitor` scope to the named run, so run 2's 38,314
   judgments stay archived in place without making run 3's pairs look already covered.
   Verify with `/api/monitor` — `run` should echo the id, `totals.judgments` should be 0,
   and `coverage` should read `{"0": 4800}`.
2. **Do not wipe.** Run 2's data is separable by `studyId` and is the permanent human-GT
   test set. Only delete `studyId = "test"` rows, per §7.6 step 5.

---

## 9. Run 4 — the uniform random-pair study (fill this next)

**This run is not like runs 2 and 3.** Those bought near-ties to *improve* the ranking. This one
buys a uniform sample to *measure* it, and it is the only study that can produce three things we
cannot get from any data we own (log §5.7):

1. **Population accuracy** — what the rating scores on a *typical* pair. Every number quoted so
   far (56.9% against a 59.1% ceiling) is measured on hand-picked hard pairs and is worst-case by
   construction.
2. **An honest calibration curve** — `bt-refit-v4-panel` absorbed all 7,780 pairs we own, so its
   /10 reliability is currently measured on its own training data.
3. **A verdict on ~$15.3k of future labelling** — headroom above a 20-point percentile gap is
   +0.1 [−2.5, +2.5] and −1.1 [−2.9, +0.6] on only 389 and 262 pairs, which are a *biased*
   subsample. **66% of this draw lands in exactly those two bands.**

**Status: ✅ COMPLETE 2026-08-04.** Study id **`6a71110e4d5c6eba96c17d21`**, published 2026-08-03,
303 sessions (296 completed), **31,237 judgments**, **$968.58**. Archived to
`labels/panel-run-4-random/results/`, analysed in full — **research log §5.8** is the write-up. Outcome in
one line: every deliverable landed, and two of the three overturned the previous plan.

| deliverable | result |
|---|---|
| Population accuracy | **81.5%** (ranking) / **79.9%** (Gemini) vs a **74.9%** individual-human ceiling |
| Honest calibration | monotone, 0.25 /10 gap = 53.0% → 2+ = 82.7%; but `σ(Δθ)` overconfident by **26 pts** |
| The $15.3k verdict | **dead money.** Headroom −0.9 [−2.1, +0.4] at 20–45, **−1.3 [−2.2, −0.5]** at 45–100 |
| Bonus — is the band structure circular? | **No.** Agreement monotone across all six bands, 53.2% → 83.6% |
| Bonus — comparator on typical pairs | **1.8–3.1 pts behind the ranking.** The bottleneck moved to the model |

Coverage came out **11–13 votes/pair** (282 / 2,185 / 33 pairs), sd 0.35 — the queue fix continues to
hold. 4 raters dropped from the fit, 1 (`6a4aaa3ca9cd50f4e76237c3`, 294 ms median) recommended for
rejection. One gold to retire: `cmr1mr5cw06y796d59gfjfwv7` at 63% over 78 views.

**As-published vs as-planned, for the record:** the study went out at a **~7 min** estimate and
**≈$2.26/submission** (implied by $968.58 ÷ 1.4286 ÷ 300), not the $3.00 at 9 min planned below.
That is ≈**$19.4/hr** at the stated 7 minutes, so it sits on policy — but run 2's **p90 was 9.4
min**, so the slowest ~10% of raters earn nearer $14.5/hr. Watch for underpayment flags or slow
fill; §7.1a's reallocation warning still applies, and representative places **cannot** be
increased after publishing.

`labels/panel-run-4-random/pairs.json`, 2,520 pairs = **2,500 new** + the same 20 golds carried with
their original left/right sides. Median percentile gap **29.9** vs 4.9 in run 3 — that difference
*is* the experiment.

The draw is **uniform and unstratified on purpose**: no percentile-gap filter and **no ethnicity
floor**, unlike runs 2–3. A floor swaps pairs to hit per-group counts, which is precisely the
selection effect that would make the resulting accuracy figure unquotable as a population number.
Bands are *recorded* in `sample-meta.json`, never *imposed*. Consequence to expect in the data:
far fewer very close pairs than run 3 and many more far-apart ones — 66% of the draw lands in the
20–45 and 45–100 bands, which is where the open spend question lives.

| band | drawn | draw % | pool % | headroom | what it buys |
|---|--:|--:|--:|--:|---|
| 0–2 | 68 | 2.7% | 2.6% | +6.2 pt | confirms the known buy zone |
| 2–5 | 98 | 3.9% | 4.4% | +5.9 pt | confirms the known buy zone |
| 5–10 | 198 | 7.9% | 7.9% | +3.5 pt | confirms the known buy zone |
| 10–20 | 479 | 19.2% | 19.3% | +3.8 pt | confirms the known buy zone |
| 20–45 | 883 | 35.3% | 37.7% | +0.1 pt | **resolves the open $15.3k question** |
| 45–100 | 774 | 31.0% | 28.1% | −1.1 pt | **resolves the open $15.3k question** |

Draw % tracking pool % to within 2.6 points is the check that the draw really is uniform.

### 9.1 The fields that change from run 3

| Field | Run 4 value |
|---|---|
| Study name (participant-visible) | `Quick photo comparison task (~8 min)` |
| Internal study name | `panel run 4 — 2,500 uniform random pairs × 12 votes` |
| Participants | **300** — still the representative-sample floor |
| How long will your study take | **9 mins** — task length is unchanged at 105 screens |
| Reward | **$3.00** (= $20.00/hr at the stated 9 min) |
| Add to participant group | create **`faceiq-panel-run4`** |
| Block participants | exclude **`faceiq-panel-run2`**, **`faceiq-panel-run3`**, and the soft-launch group |
| Completion code | **generate a new one** and mirror it into Vercel (§7.5) |
| `CURRENT_STUDY_ID` | run 4's Prolific study id — `/api/monitor` `coverage` must read `{"0": 2500}` |
| `PAIRS_PER_SESSION` | **100** — leave it. 300 raters × 100 = 30,000 votes = 12.0 votes/pair on 2,500 pairs |

Everything else is unchanged: **representative** US sample (Sex, Age, Ethnicity), desktop only,
Decision making, manually review, no auto-reject, once per participant.

> **Want younger raters?** You cannot on a representative sample — Prolific greys out every
> age filter (§7.1a). Census-matching *is* why 38% of runs 2–3 were 55+. To force younger you
> must switch to a **standard** sample and set Age = 18–34 (or 18–44). That is allowed, costs
> the same, and drops the 300-participant floor — but it also drops the *population* accuracy
> claim this study exists to buy. Prefer: keep this run representative, then do the $429
> 18–34 A/B in §9.3. If you override anyway, change Internal study name to note it and expect
> the accuracy number to be quoted as "among 18–34 US raters," not "population."

### 9.2 Why 12 votes per pair, breaking run 3's breadth-first rule

Run 3's rule — **breadth beats depth** — was measured and is still correct *for its purpose*:
2,250 pairs × 6 votes beat 1,125 × 12 by 1.8 points at fixed budget. That rule optimises how much
a study *moves the ranking*. Run 4's purpose is measurement, and two of its three deliverables
depend on per-pair depth:

- The **crowd ceiling** is a leave-one-out statistic: one rater against the majority of the
  others. At 6 votes that majority is 5 noisy votes, which biases the ceiling *downward* and so
  *understates* headroom. The existing table was measured at ~8.5 votes/pair, so run 4 needs at
  least that to be comparable, and 12 is cleanly better.
- The **calibration curve** bins observed win rates. Vote-share resolution is 1/12 at twelve
  votes versus 1/6 at six — a materially finer grid for the same money.

Population accuracy is the one deliverable that wants breadth, and 2,500 pairs already gives it a
±1.9-point interval, which is tighter than the effect sizes in play.

| | |
|---|--:|
| Real pairs to cover | 2,500 |
| Real votes per session | 100 |
| Participants (Prolific floor) | 300 |
| Votes collected | 30,000 |
| **Votes per pair** | **12.0** |
| Reward × 300 | $900.00 |
| Prolific fee (42.86%) | $385.74 |
| **Total** | **≈ $1,286** ($0.0429/vote) |

Same cost as run 3. The 300-participant floor fixes the spend at ~$1,286 whether we need the
votes or not, so the only real lever is how many pairs to spread them across — and 2,500 is what
makes 12 votes/pair fall out.

### 9.3 Age: keep this one representative, test age separately

Worth asking, because our panel skews much older than FaceIQ's likely users. Run 2+3 demographics
(`artifacts/panel-demographics/summary.json`):

| age band | raters | share |
|---|--:|--:|
| 55+ | 249 | **38.1%** |
| 35–44 | 112 | 17.1% |
| 25–34 | 112 | 17.1% |
| 45–54 | 103 | 15.7% |
| 18–24 | 78 | **11.9%** |

**Keep run 4 representative anyway**, for three reasons. The study's headline deliverable is a
*population* accuracy number — restricting age turns it into a subgroup number and forfeits the
main point. Recruitment must match runs 2–3 or the accuracy figures are not comparable. And the
measured age effect is small: in-group vs out-group agreement gaps are **+2.4 pts for 55+** (SD
0.6), −0.5 for 35–44, +0.7 for 25–34 — real but minor, and confounded with rater consistency
(log §5.5).

**Then test age properly, and cheaply.** The clean design is a fixed pair set given to two
panels, so age is the only thing varying: take a **500-pair subset of run 4** and re-run it on an
**18–34 filter**. A standard (non-representative) sample has **no 300-participant floor**, so
100 raters × 100 pairs = 10,000 votes = 20 votes/pair for **$300 + 42.86% ≈ $429**. Comparing
vote shares pair-by-pair against run 4's representative panel isolates the age effect without
contaminating the population estimate. Do not fold this into run 4 as a second arm — that halves
the power of both.

### 9.4 Launch sequence (run 4) — exact order, with commands

**Step 1 is already done** (2026-08-02): the draw is generated *and* imported into
`faceiq-rating/data/pairs.json`, validated, and the app builds. Start at step 2.

#### ✅ Step 1 — Generate and ship the data (DONE)

```bash
# in faceiq-preference-ml
python scripts/select_random_pairs.py \
    --export data/exports/cmr1mr0m7000196d57zi3vcgn \
    --n 2500 --out labels/panel-run-4-random

# ship it (run 3's file is archived at labels/panel-run-3/pairs.json and git-tracked,
# so this overwrite is reversible)
cp labels/panel-run-4-random/pairs.json ../faceiq-rating/data/pairs.json
```

`golds.json` and `examples.json` need **no change** — the draw carries the same 20 golds with
their original left/right sides. Validated against `lib/pairs.ts`'s contract:

| check | result | why it matters |
|---|---|---|
| golds present in pairs.json | 20/20 | `lib/pairs.ts` **throws at boot** if any is missing |
| gold left/right orientation preserved | 20/20 | `golds.json` stores the answer as a *side*; re-randomising marks half the attention checks backwards |
| duplicate pairIds | 0 | duplicates silently shadow each other in the `byId` Map |
| `realPairIds` (queue size) | 2,500 | must match the `coverage` figure in step 4 |
| blank photoUrls | 0 | a blank URL is an unanswerable screen |
| worked examples leaked into scored pool | 0 | examples must not be paid pairs |
| `npx tsc --noEmit` | clean | — |
| `npm run build` | ✓ compiled | the build imports the module, so this **executes** the boot assertion |

#### Step 2 — Deploy

Deploy and confirm the **production alias** serves the new file, not just a preview build. (Run 2
lost time to exactly this.) Quickest confirmation is step 4's `/api/monitor` coverage figure.

#### Step 3 — Create the Prolific study

Fill from §9.1. Generate a **new completion code**, and mirror it into
`PROLIFIC_COMPLETION_CODE` in Vercel. The study id appears in the study URL once created — you
need it for step 4.

#### Step 4 — Point the app at the new run

Set **`CURRENT_STUDY_ID`** to run 4's Prolific study id in Vercel, then **redeploy** (env changes
do not take effect without one). Verify `/api/monitor`:

| field | expected |
|---|---|
| `run` | run 4's study id |
| `totals.judgments` | `0` |
| `coverage` | `{"0": 2500}` |

`coverage` reading `{"0": 2500}` is the single check that proves all three of: the new file
deployed, the study id is scoped correctly, and no stale judgments are being counted. If it shows
4,800 you are still serving run 3's pairs.

#### Step 5 — One `/?test=1` session on the **deployed** production alias

Do this **after deploy, before Publish**. Local `npm run dev` does not prove the production
build has the new pairs (run 2 already lost time to that). Open:

```
https://faceiq-rating.vercel.app/?test=1
```

Click **Start a test session**, walk a handful of screens, and confirm: faces load, no broken
images, and finishing returns the **new** completion code. Test judgments are tagged
`studyId = "test"` and are excluded from coverage and from every analysis, so this costs nothing.
Delete those rows afterwards per §7.6 step 5.

Also hit `/api/monitor` (step 4) — `coverage: {"0": 2500}` is the check that the new file is live.

#### Step 6 — Publish

**Do not wipe the database.** Runs 2 and 3 stay archived in place; `CURRENT_STUDY_ID` scopes the
queue, coverage and `/monitor`. Runs 2–3 are the permanent human-GT test set.

#### Step 7 — On completion, archive before analysing

Dump via `psql` rather than `/api/export` — the export route 500s at this scale (§8 notes). Then
run §9.5's commands, and **run `rating_calibration.py` before any BT refit consumes run 4**, since
these are the only pairs in the programme no ranking has been fitted on.

### 9.5 What to run when the data lands ✅ all run 2026-08-04

**Step 0, before anything else: pull the data.** `/api/export?table=judgments` 500s at this scale, so go
straight to Postgres. Run 4 came out of the same database as runs 2–3 — `CURRENT_STUDY_ID` scopes the app
but not the tables, so **every query must filter on `studyId`**, and there were 4 leftover `test` sessions
in there from the pre-launch check:

```bash
cd ../faceiq-rating && set -a && . ./.env && set +a
mkdir -p ../faceiq-preference-ml/labels/panel-run-4-random/results

psql "$DATABASE_URL" -At -o ../faceiq-preference-ml/labels/panel-run-4-random/results/judgments.jsonl \
  -c "select row_to_json(t) from (
        select j.id as \"judgmentId\", j.\"sessionId\", s.\"prolificPid\", s.\"studyId\",
               j.\"pairId\", j.choice, j.\"responseTimeMs\", j.\"presentationOrder\",
               j.\"isGold\", j.comment,
               to_char(j.\"createdAt\" at time zone 'UTC', 'YYYY-MM-DD\"T\"HH24:MI:SS.MS\"Z\"') as \"createdAt\"
        from \"Judgment\" j join \"Session\" s on s.id = j.\"sessionId\"
        where s.\"studyId\" = '6a71110e4d5c6eba96c17d21'
        order by j.\"createdAt\") t;"

psql "$DATABASE_URL" -At -o ../faceiq-preference-ml/labels/panel-run-4-random/results/sessions.jsonl \
  -c "select row_to_json(t) from (
        select s.id as \"sessionId\", s.\"prolificPid\", s.\"studyId\", s.\"prolificSessionId\",
               s.\"assignedPairIds\", s.feedback,
               to_char(s.\"startedAt\" at time zone 'UTC', 'YYYY-MM-DD\"T\"HH24:MI:SS.MS\"Z\"') as \"startedAt\",
               case when s.\"completedAt\" is null then null
                    else to_char(s.\"completedAt\" at time zone 'UTC', 'YYYY-MM-DD\"T\"HH24:MI:SS.MS\"Z\"') end as \"completedAt\",
               (select count(*) from \"Judgment\" j where j.\"sessionId\" = s.id) as \"judgmentCount\"
        from \"Session\" s
        where s.\"studyId\" = '6a71110e4d5c6eba96c17d21'
        order by s.\"startedAt\") t;"
```

`Session` has **no `judgmentCount` column** — the subselect above synthesises it, because that is the field
the integrity check compares against the judgment line count. Verify three things before analysing: line
counts (31,237 / 303), `judgmentId` unique, and `sum(judgmentCount)` equal to the judgment count.

**`analyze_panel_run.py` needs a `golds.json` in the panel dir**, which a random draw does not generate —
run 4 carries the same 20 golds as runs 2–3, so copy the shipped file in first:

```bash
cp ../faceiq-rating/data/golds.json labels/panel-run-4-random/golds.json
```

Then, in this order (calibration **before** any refit):

```bash
# Rater QC, rejections, and per-band accuracy vs human ceiling. `stratum` is set to the
# band, so this emits the headroom table directly.
python scripts/analyze_panel_run.py \
    --results labels/panel-run-4-random/results \
    --panel-dir labels/panel-run-4-random \
    --out artifacts/panel-run-v4

# Population accuracy + the first out-of-sample calibration curve. --ratings must be the
# VLM-only refit: v4-panel absorbed runs 2-3, and run 4's pairs are fresh to neither if
# you refit first, so run this BEFORE any refit that consumes run 4.
python scripts/rating_calibration.py \
    --export data/exports/cmr1mr0m7000196d57zi3vcgn \
    --ratings artifacts/bt-refit-v2-qc/ratings.csv \
    --panel labels/panel-run-4-random/results labels/panel-run-4-random/sample-meta.json \
    --rejects artifacts/panel-run-v4/reject-pids.txt \
    --out artifacts/panel-run-v4/calibration.json

# Refreshed headroom with the new pairs folded in. Compare byPercentileGap against
# artifacts/label-information-v1/report-v2qc-bands.json: the 20-45 and 45-100 intervals
# should tighten from +/-2.5 to roughly +/-1.6.
python scripts/label_information.py --export data/exports/cmr1mr0m7000196d57zi3vcgn \
    --ratings artifacts/bt-refit-v2-qc/ratings.csv \
    --panel labels/panel-pilot/results labels/panel-pilot/sample-meta.json \
    --panel labels/panel-run-3/results labels/panel-run-3/sample-meta.json \
    --panel labels/panel-run-4-random/results labels/panel-run-4-random/sample-meta.json \
    --rejects artifacts/panel-run-v1/reject-pids.txt \
              artifacts/panel-run-v3/reject-pids.txt \
              artifacts/panel-run-v4/reject-pids.txt \
    --out artifacts/label-information-v2/report.json
```

Four more steps were added on the day, and belong in the sequence for any future run:

```bash
# Is the band structure real, or an artefact of how we drew pairs? Uniform pairs make this testable.
python scripts/gap_agreement_curve.py \
    --panel labels/panel-run-4-random/results labels/panel-run-4-random/sample-meta.json \
    --rejects artifacts/panel-run-v4/reject-pids.txt \
    --out artifacts/panel-run-v4/gap-curve.json

# What did this study buy for the ranking? (Expect ~zero on a uniform draw — that is the point.)
python scripts/panel_run_delta.py --export data/exports/cmr1mr0m7000196d57zi3vcgn \
    --prior labels/panel-pilot/results labels/panel-pilot/sample-meta.json \
    --prior labels/panel-run-3/results labels/panel-run-3/sample-meta.json \
    --new   labels/panel-run-4-random/results labels/panel-run-4-random/sample-meta.json \
    --rejects artifacts/panel-run-v1/reject-pids.txt artifacts/panel-run-v3/reject-pids.txt \
              artifacts/panel-run-v4/reject-pids.txt \
    --spend 968.58 --out artifacts/panel-run-delta-v4

# New ranking of record. --exclude-faces/--exclude-genders have NO defaults here: omit them and you
# silently rank all 2,999 faces instead of 2,866, and the gates still pass. Check metrics.json's
# qcExcludedFaces reads 133.
python scripts/refit_bt_panel.py --export data/exports/cmr1mr0m7000196d57zi3vcgn \
    --exclude-faces artifacts/face-qc-v1/exclude-faces.csv \
    --exclude-genders artifacts/face-qc-v1/gender-fixes.csv \
    --panel-results labels/panel-pilot/results labels/panel-run-3/results \
                    labels/panel-run-4-random/results \
    --panel-meta labels/panel-pilot/sample-meta.json labels/panel-run-3/sample-meta.json \
                 labels/panel-run-4-random/sample-meta.json \
    --rejects artifacts/panel-run-v1/reject-pids.txt artifacts/panel-run-v3/reject-pids.txt \
              artifacts/panel-run-v4/reject-pids.txt \
    --baseline artifacts/bt-refit-v2-qc/ratings.csv --out artifacts/bt-refit-v5-panel

# Population accuracy of the shipping model, which no earlier study could measure. Run it for every
# val_fraction 0.5 arm so compare_panel_evals.py can pair them on the identical 631-pair subset.
python scripts/eval_vs_panel.py --export data/exports/cmr1mr0m7000196d57zi3vcgn \
    --checkpoint checkpoints/train-v12-panel-soft/best.pt \
    --panel labels/panel-run-4-random/results labels/panel-run-4-random/sample-meta.json \
    --rejects artifacts/panel-run-v4/reject-pids.txt \
    --ratings artifacts/bt-refit-v2-qc/ratings.csv \
    --out artifacts/train-v12-panel-soft/panel-eval-run4.json \
    --dump-pairs artifacts/train-v12-panel-soft/panel-pairs-run4.csv
```

Note the run-4 pairs *were* the only pairs in the programme that no ranking had been fitted on. The
calibration test ran first, and `bt-refit-v5-panel` has now consumed them — **that property is spent.**
Re-testing calibration on a future ranking needs a fresh uniform draw (~700 pairs, ~$400, no
representative floor needed for a calibration curve alone).

**The decision this study makes:** if headroom at 20–45 and 45–100 comes back below ~2 points
with a tight interval, the remaining $15.3k is dead money and labelling stops at the 0–20 buy
zone. If it comes back above ~3 points, the wide bands are live and the buy zone was too narrow.

**Answer: −0.9 [−2.1, +0.4] and −1.3 [−2.2, −0.5]. Dead money, and the second interval is entirely
below zero — a purchased vote there is worse than the free label.** $14,367 cancelled; labelling stops at
the 0–20 buy zone with $7,463 left in it. See log §5.8 and `panel-study-playbook.md` §2a.

---

## 10. Run 5 — the buy zone (fill this next)

**This one is simple, and it is the first study bought purely to sharpen the ground truth.** Runs 2–3
improved the ranking, run 4 measured it. Run 5 finishes the part of the export where a human vote is
provably worth more than the free Gemini label: everything inside a **10-percentile-point** gap that
nobody has rated yet.

**Status: 🟡 DRAWN AND SHIPPED, not yet published (2026-08-04).** `labels/panel-run-5-buyzone/pairs.json`
= **5,020 pairs = 5,000 new + the same 20 golds** with their original left/right sides. Already copied to
`faceiq-rating/data/pairs.json`; `npx tsc --noEmit` clean and `npm run build` compiled, which executes
`lib/pairs.ts`'s boot assertion.

| band | pairs drawn | headroom over the free label |
|---|--:|--:|
| 0–2 | 886 | **+6.2** [+4.1, +8.3] |
| 2–5 | 1,484 | **+6.2** [+4.4, +7.9] |
| 5–10 | 2,630 | **+3.9** [+2.3, +5.6] |

Median percentile gap **5.3**, 2,775 faces touched at **3.6 new comparisons each**, only 279 faces
appearing once. Gender 2,500 / 2,500.

### 10.1 The fields that change from run 4

| Field | Run 5 value |
|---|---|
| Study name (participant-visible) | `Quick photo comparison task (~8 min)` |
| Internal study name | `panel run 5 — 5,000 buy-zone pairs (0–10 gap) × 6 votes` |
| Participants | **300** — still the representative-sample floor |
| How long will your study take | **9 mins** — 105 screens, unchanged |
| Reward | **$3.00** (= $20.00/hr at the stated 9 min) |
| Add to participant group | create **`faceiq-panel-run5`** |
| Block participants | exclude **`faceiq-panel-run2`**, **`faceiq-panel-run3`**, **`faceiq-panel-run4`**, and the soft-launch group |
| Completion code | **generate a new one** and mirror it into `PROLIFIC_COMPLETION_CODE` in Vercel |
| `CURRENT_STUDY_ID` | run 5's Prolific study id — `/api/monitor` `coverage` must read `{"0": 5000}` |
| `PAIRS_PER_SESSION` | **100** — leave it |

Everything else unchanged: **representative** US sample (Sex, Age, Ethnicity), desktop only, Decision
making, manually review, **no auto-reject**, once per participant.

| | |
|---|--:|
| Real pairs to cover | 5,000 |
| Real votes per session | 100 |
| Participants (Prolific floor) | 300 |
| Votes collected | 30,000 |
| **Votes per pair** | **6.00** |
| Reward × 300 | $900.00 |
| Prolific fee (42.86%) | $385.74 |
| **Total** | **≈ $1,286** ($0.0429/vote) |

### 10.2 Why 5,000 pairs at 0–10 rather than 2,642 at 0–5

The ask was the first two bands, which is **2,642 unbought pairs** (969 at 0–2, 1,673 at 2–5) —
drawn and kept at `labels/panel-run-5-close/` if you prefer it. It is the worse buy, **at identical
cost**, and the reason is the 300-participant floor:

| draw | pairs | votes/pair at 300 raters | new comparisons per face | faces seen once | cost |
|---|--:|--:|--:|--:|--:|
| 0–5 only | 2,642 | **11.4** | 2.2 | 808 | $1,286 |
| **0–10 (shipped)** | **5,000** | **6.00** | **3.6** | **279** | **$1,286** |

The floor fixes the spend at ~$1,286 whether we need the votes or not, so the only lever is how many
pairs to spread them across — and §1's measurement is unambiguous that **breadth beats depth for
improving the ranking**: at equal budget, 2,250 pairs × 6 votes beat 1,125 × 12 by 1.8 points. Eleven
votes on a near-tie is past the point of usefulness; six is the measured sweet spot. The 0–10 draw hits
**exactly 6.00**, covers a third validated band (+3.9 headroom), and nearly triples the number of faces
that get more than one new comparison — which is the mechanism the gain actually travels through.

After run 5 the buy zone has only **10–20** left: 7,247 pairs, ~$4,205 at 12 votes or ~$2,100 at 6.

### 10.3 What this run is and is not for

**It is for the ground truth, not for the comparator.** Log §5.8 measured the neural comparator
*already beating* the ranking under a 10-point gap — these are the pairs it is best at — so do not
expect run 5 to move the comparator much. What it moves is `bt-refit-vN`, which is the yardstick every
model, formula and EBM in the programme is scored against, and which is the reference set a production
score would be plotted onto. Buying it is buying a sharper ruler.

Set the expectation before launch, per §6 item 0: **the deliverable is a better ranking in the 0–10
band**, measured by `panel_run_delta.py` against the <1 point per $500 stop rule and by
`refit_bt_panel.py`'s held-out gain in the `topup` strata. Run 3, the closest comparable study, returned
1.24 points per $500.

### 10.4 Launch sequence

Identical to §9.4, with run 5's numbers:

1. ✅ **Draw and ship** — done. `scripts/select_topup_pairs.py --max-gap 10 --n 5000`, copied to
   `faceiq-rating/data/pairs.json`, `tsc` clean, `npm run build` compiled.
2. **Deploy** and confirm the *production alias* serves the new file, not a preview build.
3. **Create the Prolific study** from §10.1; generate a new completion code and mirror it into Vercel.
4. **Set `CURRENT_STUDY_ID`** to run 5's study id and **redeploy** (env changes need one).
   `/api/monitor` must read `coverage: {"0": 5000}` and `judgments: 0`. If it shows 2,500 you are still
   serving run 4.
5. **One `/?test=1` session** on `https://faceiq-rating.vercel.app`, then delete the test rows.
6. **Publish.** Do not wipe the database — `CURRENT_STUDY_ID` scopes everything, and runs 2–4 are the
   permanent human-GT test set.
7. **On completion**, follow §9.5's pull-and-analyse sequence with run 5's paths. Note that
   `rating_calibration.py` is **not** applicable this time: run 5's pairs are near-ties drawn on a
   ranking, so they cannot serve as an unbiased calibration set the way run 4's uniform draw did.

### 10.5 Before you publish — one gold to replace

Run 4 retired `cmr1mr5cw06y796d59gfjfwv7` (63% over 78 views, third independent failure), so the set is
effectively **19 sound golds**. The shipped `golds.json` still carries 20. Either drop it and ship 19, or
top up from run 4's unanimous non-gold pairs with `scripts/replace_golds.py`. Nineteen is workable —
raters see 5 each — but do not let it drift lower.
