# CLAUDE.md

Agent context for **faceiq-preference-ml** — the offline ML companion to the
`faceiq-labs` Next.js app (sibling folder `../faceiq-labs`).

## What this repo does

Consumes pairwise ground-truth exports from faceiq-labs and produces:

1. **Bradley-Terry ranking** — authoritative relative attractiveness score (theta) per
   face, calibrated to a /10 scale via a pre-registered percentile curve.
2. **Neural preference comparator** — siamese CNN trained on `(photoA, photoB, winner)`
   rows; validated by held-out pairwise accuracy and Kendall tau vs BT.

The labeling pipeline (cohort import, VLM batching, human audit, export) lives in
faceiq-labs. This repo **never** connects to the app database — it only reads file
exports under `data/exports/`.

## Data layout (produced by faceiq-labs `scripts/export-gt-run.ts`)

```
data/exports/<runId>/
├── manifest.json          # counts, sha256 per shard, label rules — ALWAYS verify first
├── matchups/
│   └── matchups-NNNNN.jsonl   # 10k rows per shard, ordered by pairIndex
├── faces.jsonl            # one face per line: faceId, gender, decileBin, labsOverallScore, imagePath
└── images/<faceId>.webp   # face photos (relative imagePath in faces.jsonl)
```

Current GT run: `cmr1mr0m7000196d57zi3vcgn` — **52,414** export-eligible pairs
(52,500 minus 86 audit-excluded), 3,000 faces, prompt `pairwise-v3-gt`,
model `gemini-2.5-flash`. Human audit: 84.9% agreement on a 750-pair seeded sample.

## Label rule (do not change without updating research log)

`finalOutcome` / `finalWinnerFaceId` per matchup row:
- **human label** (`humanOutcome`) when `humanLabeledAt` is set (~1,032 audited rows)
- else **VLM label** (`vlmOutcome`)

The exporter precomputes these fields; `src/faceiq_pref/data.py` re-derives and asserts
they match. Training and BT must consume `finalOutcome` only.

## Hard rules

- **Never commit** `data/`, `checkpoints/`, `artifacts/`, or any `.pt`/`.pth` file.
  Face photos must never enter git history.
- **Split train/val by face id**, never by random pair — a face appearing in both
  splits leaks identity information.
- **Never use Labs `overall_score` as a training label** — it is the legacy formula we
  are replacing. Stratification and post-hoc validation only.
- Always load exports through `faceiq_pref.data.load_export()` — it verifies manifest
  hashes/counts and fails loudly on corrupt or partial exports.
- BT acceptance gates (research log §5.1): comparison graph connected, no face with
  <15 resolved comparisons, 80% subsample Spearman rho > 0.95.
- **Never cut bands, strata or pair draws on a ranking that has seen the votes being analysed**
  (log §5.7 — doing it on `bt-refit-v4-panel` inverted the headroom table and would have sent $8.8k
  at the worst band). Band cuts use `bt-refit-v2-qc`, which predates every human vote.
- **Never quote an accuracy without its pair distribution** (log §5.8). The same ranking scores
  51.5% on near-ties and 81.5% on a uniform draw. Say which.
- Pass `--exclude-faces artifacts/face-qc-v1/exclude-faces.csv --exclude-genders
  artifacts/face-qc-v1/gender-fixes.csv` to `refit_bt_panel.py`. It has **no defaults**, and without
  them it silently ranks 2,999 faces instead of 2,866 while still passing every gate — check
  `qcExcludedFaces` in `metrics.json` reads 133.

## Commands

```bash
source .venv/bin/activate
python scripts/run_bt.py --export data/exports/cmr1mr0m7000196d57zi3vcgn   # BT + calibration
python scripts/train.py --config configs/train-v1.yaml                     # train comparator
streamlit run app/dashboard.py --server.port 8502                                            # results UI
pytest                                                                     # tests
```

Ranking of record is **`artifacts/bt-refit-v5-panel`** (all four panel runs: 10,280 pairs, 95,245
votes, 963 raters). `refit_bt_panel.py` produces it; `run_bt.py` is the VLM-labels-only path.

## Where results go

| Output | Location |
|--------|----------|
| BT refit (theta, /10, diagnostics) | `artifacts/bt-refit-vN/` (`ratings.csv`, `metrics.json`) |
| Training runs | `artifacts/train-vN/metrics.json` + `checkpoints/train-vN/*.pt` |
| Panel study analysis | `artifacts/panel-run-vN/` (rater QC, reject lists, per-band accuracy, calibration) |
| Human-grounded model eval | `artifacts/train-vN/panel-eval*.json` + `panel-pairs*.csv` — **not** `val_accuracy`, which keeps the VLM labels and cannot see the improvement |
| Decisions + summary numbers | `docs/research/scoring-gt-research-log.md` §5 — **this repo's copy is now canonical**, see below |

## Docs

**Start at `docs/research/README.md`** — current state, what is unknown, priority order.

⚠️ **The research log in *this* repo is the live record.** An older note said the canonical copy
lived in faceiq-labs; that is no longer true and following it loses work. As of 2026-08-04 the
faceiq-labs copy is **1,188 lines last touched 2026-07-25**, missing §5.5 (human panel, 4 runs,
95,245 votes), §5.6 (rating validation), §5.7 (label information) and §5.8 (population accuracy,
honest calibration, the spend verdict) — over 1,000 lines and every decision since. Write here. Sync
to faceiq-labs deliberately, as a publish step, not by assuming it is ahead.

**Current state in one line (2026-08-05):** the labelling programme is finished — $14,367 of planned
spend was cancelled on measurement, and the bottleneck is the neural comparator, which is 1.8–2.1
points *behind* the ranking it was distilled from on typical pairs. `train-v16` then added run 4's
wide-band votes and **changed nothing in the weak band**, confirming this is an extraction failure
rather than a data shortage. Next actions, none needing new labels: fix checkpoint selection (it uses
`val_accuracy`, scored against Gemini, which cannot see the improvement), try 224 px, and measure a
gap-routed ensemble.

**And as of 2026-08-05 the whole path is validated off-cohort, both genders** (log §5.10, §5.10a): on
70 female and 70 male faces from outside the cohort, the production placement reproduces one rater's
blind ordering **84.3%** / **76.4%** of the time against his own **97.1%** / **94.3%** ceilings, and
**accuracy rises monotonically with the placed gap in both sets** — that replication is the result,
because it is the band's central claim tested on faces no model has seen. Males are ~6 points harder
after controlling for the pair draw. Rules that came out of it: **never refit `T` on a single rater**
(it describes a *random* rater — two real panel raters agree with each other only **68.1%** of the
time, so one rater against himself at 97.1% is a far tighter target, and predicted 67.4% vs observed
84.3% is the rater, not a broken curve); **self-agreement is not the 74.9% panel ceiling** — one
rater vs himself and one rater vs the crowd are different metrics and must not be set side by side;
and the off-cohort **/10 as an absolute is still untested** because no hand ranges were collected.

Also 2026-08-05, and it closes a live hypothesis: **Labs' ~10-point deficit is the formula, not the
photos.** Restricting the panel comparison to pairs where both faces are clean per the §5.0 QC flags
widens Labs' gap to the human ceiling (−1.5 → −2.5) while the placement's holds (+8.7 → +8.9) — clean
photos help humans and models alike, so an upload quality gate recovers nothing. The one part still
untested is *perspective distortion*, which the QC schema has no category for
(`production-scoring-pipeline.md`, Labs composite).

- `docs/research/README.md` — **entry point**; live-doc index and priorities
- `docs/research/scoring-gt-research-log.md` — the lab notebook; **§5.8 (run 4) is the newest finding**
- `docs/research/programme-direction-review.md` — strategy: is the approach working, when to stop
  labelling, plan B, and the maths behind rating *bands*. Read before committing money
- `docs/research/scoring-gt-training.md` — this repo's charter and ML checklist
- `docs/research/scoring-gt-core.md` — §6-§8 BT, calibration, training methodology (reference)
- `docs/research/panel-study-playbook.md` — labelling policy; §2a is the buy/skip band table, §2c is
  the proof the bands are not circular
- `docs/research/prolific-soft-launch-form.md` — Prolific fill sheet, one section per run; §9.5 has
  the full pull-and-analyse command sequence
- `docs/research/panel-pilot-runbook.md` — study ops, draw → ship → deploy → monitor → archive
- `docs/research/panel-pilot-findings.md` — panel results, runs 1–4 (the run-4 addendum is the
  population-accuracy story)
- `docs/research/archive/` — superseded plans and closed sub-studies; read its README first
- `docs/ops/aws-gpu-training-setup.md` — EC2 GPU training (current; `scripts/sync_panel_to_gpu.sh`
  automates most of it)
- `.cursor/skills/bt-refit/` and `.cursor/skills/preference-training/` — step-by-step
  methodology for the two main jobs
