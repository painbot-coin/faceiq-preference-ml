# Pilot-500: two-rater pairwise labeling

500 matchup pairs from GT run `cmr1mr0m7000196d57zi3vcgn`, sampled by Bradley-Terry
win-probability margin — p = sigmoid(|thetaA − thetaB|) from `bt-refit-v1`:

| stratum | target | definition |
|---------|--------|------------|
| close   | 175    | p < 0.60 |
| mid     | 125    | 0.60 <= p < 0.80 |
| clear   | 100    | 0.80 <= p < 0.95 |
| random  | 100    | drawn from all remaining eligible pairs |

Pairs already human-audited during the original GT run are excluded. Presentation
order and left/right placement are fixed by the sampling seed, so every rater sees
the identical sequence.

## Files

| file | committed | contents |
|------|-----------|----------|
| `pairs.json` | yes | blinded pair list: pairId, gender, left/right faceId + photo URL. No Gemini label, no BT theta, no bucket. |
| `votes-<rater>.jsonl` | yes | one line per decision: `{pairId, choice: left\|right\|tie, comment, ratedAt}`. One file per rater — never conflicts in git. |
| `sample-meta.json` | no (gitignored) | the answer key: stratum, bucket, win prob, thetas, Gemini outcome/confidence per pair. Regenerate anytime by rerunning the sampler with the same seed. |

## How to label

1. `git pull`
2. `source .venv/bin/activate && streamlit run app/dashboard.py --server.port 8502`
   and open the **Pilot-500 labeler** tab — or run the labeler alone:
   `streamlit run app/labeler.py`
3. Enter your name (e.g. `dit` or `alex`) — it becomes your votes filename.
4. Pick who is more attractive: click or use keys `1` (left), `2` (right),
   `T` (too close to call). Add one sentence on why — especially on close calls.
   Progress saves after every pair; quit and resume anytime. `Undo` reverts the
   last vote.
5. When done (or at any checkpoint):

   ```bash
   git add labels/pilot-500/votes-<you>.jsonl
   git commit -m "pilot-500: votes (<you>)"
   git push
   ```

Photos load from the local export (`data/exports/<runId>/images/`) when present,
otherwise from the public blob URLs — so labeling works with just the repo and an
internet connection. If you have the export zip (`<runId>-export.zip`), unzip it
into `data/exports/` for offline use and for running the analysis with the model.

**Don't peek** at `sample-meta.json`, the BT rankings tab, or the export matchup
files for these pairs until you've finished labeling — the point is independent
human judgments.

## Regenerate the sample (only if intentionally changing it)

```bash
python scripts/select_pilot_pairs.py --export data/exports/cmr1mr0m7000196d57zi3vcgn
```

Same seed (default 20260715) reproduces the identical `pairs.json`. Changing the
sample after votes exist invalidates them — don't.

## Analyze

After both votes files are pushed:

```bash
python scripts/analyze_pilot.py --export data/exports/cmr1mr0m7000196d57zi3vcgn \
    --checkpoint checkpoints/train-v8-arcface-e2e-long/best.pt   # optional model row
```

Writes `artifacts/pilot-500/`: `summary.md` (agreement matrices overall + per
bucket, BT calibration curve, Gemini-confidence breakdown), `agreement.json`,
`per_pair.csv`, and `disagreements.csv` (rater disagreements with both comments).
Results get written up in the research log (scoring-gt-research-log.md §5).
