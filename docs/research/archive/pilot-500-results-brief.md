# Pilot-500 results: two-rater labeling vs Gemini GT — and what to do next

*2026-07-23 · Dit + Alex, 500 pairs each, run `cmr1mr0m7000196d57zi3vcgn` ·
full tables in `artifacts/pilot-500/summary.md`*

## What we did

We both independently labeled the same 500 matchup pairs, stratified by
Bradley-Terry closeness (win probability p of the favorite): **close** (p < 0.60),
**mid** (0.60–0.80), **clear** (0.80–0.95), **very_clear** (≥ 0.95). The sample
deliberately oversamples hard pairs — 36% close vs 6.6% in the real corpus — so
headline numbers here look worse than corpus-wide reality. Neither of us saw
Gemini's labels, BT scores, or each other's votes.

## Headline numbers

| Agreement (decisive pairs) | overall | close | mid | clear | very_clear |
|---|---|---|---|---|---|
| **Dit vs Alex (human ceiling)** | **71.3%** | 68.5% | 71.1% | 71.2% | 80.7% |
| Gemini vs human consensus* | 62.3% | 51.6% | 59.4% | 69.0% | 84.8% |
| BT ranking vs human consensus | 70.8% | 53.3% | 73.3% | 85.7% | 84.8% |
| Comparator (v8) vs human consensus | 67.1% | 52.5% | 64.4% | 81.0% | 87.0% |

*Human consensus = the 353/500 pairs where we both decisively picked the same
winner. It's the closest thing to ground truth this pilot produces.

Weighted by the **real corpus mix** (6.6% close / 15.1% mid / 20.0% clear /
58.2% very_clear), agreement with human consensus is approximately:
**Gemini ~76%, BT ~81%, comparator ~80%.**

## Key findings

1. **Humans agree with each other only ~71% on this sample — and only ~80% even
   on the easiest pairs.** Attractiveness preference has real taste variance plus
   attention noise. Any single rater — human or VLM — is a noisy label source.
   This is the strongest argument for majority-vote panels over more solo labels.

2. **Gemini is at coin-flip on close pairs (51.6%) and weak on mid (59.4%).**
   This is not a prompting or curation problem — it mirrors us (we only agree 68.5%
   on close pairs ourselves). Close pairs are intrinsically low-signal; a VLM
   re-pass would re-roll the same dice.

3. **Gemini's medium-confidence labels are junk everywhere** (45–53% agreement
   with either of us across all buckets — chance level). Its high-confidence labels
   on clear/very_clear pairs are genuinely good (80–88% vs consensus, ≈ human
   ceiling). Confidence + closeness cleanly separates "keep" labels from "noise"
   labels.

4. **Aggregation already rescues a lot.** The BT ranking (built from ~35 noisy
   Gemini comparisons per face) agrees with human consensus at 70.8% — matching
   the human-vs-human rate and 8 points above raw Gemini. The per-pair noise
   partially cancels at the ranking level. The GT pipeline is not broken.

5. **The comparator tracks its training labels, as expected** (76% with BT,
   67% with humans). It cannot exceed the quality of what it's fed — better
   labels on hard pairs are the only route to human-aligned improvement there.

## What this means for the three options on the table

**Option A — re-run the VLM over all 52k with better stratification/curation: no.**
Finding 2 says the failure is intrinsic to close pairs, not fixable by curation.
We'd pay for a full re-pass to shuffle noise on ~22% of the corpus and reproduce
the ~78% we already have on the rest.

**Option B — scale VLM matchups to 100k+: no (not as-is).**
58% of the current corpus is very_clear pairs that every judge already gets right;
sampling more of the same adds little information. Model accuracy is 78.4% against
an effective label ceiling of ~80% (corpus-weighted VLM-vs-consensus) — the
bottleneck is label quality on hard pairs, not label quantity. More VLM data buys
more of what we already have enough of.

**Option C — panel pilot now: yes.** This pilot is precisely the evidence for it,
with one important refinement below.

## Recommendation: panel pilot → targeted hybrid relabel (not a flat 100k)

**Step 1 — panel pilot (~2–5k pairs × 5–10 raters), as in the existing panel plan.**
Stratify toward close/mid pairs. Two raters can't measure consensus — the pilot's
job is to establish: (a) majority-vote stability (does a 5-vote majority replicate
with a different 5?), (b) the true human ceiling per bucket, (c) Gemini bias by
demographic slice (our n=500 is too small to slice), and (d) rater QC mechanics.
Gate: majority-vote self-agreement on close pairs meaningfully above our 68.5%
two-person rate. If it isn't, close pairs are irreducibly subjective and we should
*stop paying anyone* to label them and treat them as soft ties.

**Step 2 — hybrid relabel of the existing 52k, not a 100k panel pass.**
The pilot data says labels split cleanly:

| Segment | pairs | current label quality | action |
|---|---|---|---|
| clear + very_clear, Gemini high-conf | 37,083 | ~80–88% ≈ human ceiling | **keep VLM labels** |
| close + mid (all confidences) | 11,391 | 50–63% | **panel-relabel, 5 votes each** |
| clear + very_clear, medium/low-conf | 3,929 | chance-level per pilot | panel or drop |

That's ~57–77k panel judgments (11.4–15.3k pairs × 5 votes) instead of 500k+
for a flat 100k×5 pass — roughly 7× cheaper — and every dollar lands on pairs
where labels are currently noise. Retrain the comparator on the blended labels; the variance head (v11) gives
us per-face uncertainty to sanity-check against panel controversy.

**Step 3 — if we then still want more data, generate new pairs adaptively:**
sample new matchups concentrated near the current BT margin (close/mid by
construction) and send them straight to the panel. 100k random new pairs would be
~58% very_clear — wasted budget regardless of who labels them.

## Caveats

- Two raters, both of us steeped in this project and using similar vocabularies —
  the panel may agree less (or differently). That's exactly what Step 1 measures.
- "Human consensus" here is n=353 pairs from 2 raters; it bounds Gemini's accuracy
  loosely on the hard buckets.
- Privacy/legal gate from the panel plan still applies before any vendor sees photos.
