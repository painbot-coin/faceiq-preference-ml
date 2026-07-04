---
name: bt-refit
description: Run and validate the Bradley-Terry ground-truth refit on a faceiq-labs pairwise export. Use when asked to refit BT, compute face rankings, check ranking stability, calibrate scores to /10, or validate against Labs overall_score.
---

# Bradley-Terry refit

Turns ~52k pairwise outcomes into a relative strength theta per face — the
authoritative GT ranking for the cohort. Methodology: `docs/research/scoring-gt-core.md`
§12/§14 and research log §5.1.

## Run

```bash
python scripts/run_bt.py --export data/exports/<runId> [--confidence high] [--out artifacts/bt-refit-vN]
```

Outputs to `artifacts/bt-refit-vN/`:
- `ratings.csv` — faceId, theta, percentile, scoreOutOf10, comparisonCount, gender, decileBin, labsOverallScore
- `metrics.json` — stability rho, connectivity, spearman vs Labs, config used

## Method

1. Load via `faceiq_pref.data.load_export()` (verifies manifest). Use `finalOutcome` only.
2. Same-gender graphs: fit **male and female separately** (queue is same-gender only;
   the combined graph is disconnected across genders by construction).
3. Ties: count as half-win for each side (0.5/0.5). Record the rule in metrics.json.
4. Fit: MLE via `choix.ilsr_pairwise` (or `opt_pairwise` fallback) with small
   regularization alpha for connectivity robustness.
5. Calibration: within each gender, percentile-rank theta then map through the
   pre-registered curve (core §12): p50 -> 5.0, p90 -> 7.0, p99 -> 8.0, p99.9 -> 8.5,
   interpolated monotonically (PCHIP), clamped to [0, 10].

## Acceptance gates (all must pass before trusting the ranking)

- [ ] Comparison graph **connected** within each gender
- [ ] **No face with < 15** resolved comparisons
- [ ] **Stability**: two independent refits on random 80% pair subsamples ->
      Spearman rho between rank lists **> 0.95** (per gender)
- [ ] Spearman vs Labs `overall_score` computed and logged (sanity signal, not a gate
      threshold — moderate positive correlation expected)

## After a successful refit

1. Record results in faceiq-labs `docs/research/scoring-gt-research-log.md` §5.1
   (status, metrics table, artifact path, decision).
2. If gates fail: check excluded faces, consider `--confidence high` filter (audit
   showed high-confidence ~87% accurate vs medium ~74%), or investigate disconnected
   components before re-running.
