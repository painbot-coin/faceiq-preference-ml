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

## Commands

```bash
source .venv/bin/activate
python scripts/run_bt.py --export data/exports/cmr1mr0m7000196d57zi3vcgn   # BT + calibration
python scripts/train.py --config configs/train-v1.yaml                     # train comparator
streamlit run app/dashboard.py                                             # results UI
pytest                                                                     # tests
```

## Where results go

| Output | Location |
|--------|----------|
| BT refit (theta, /10, diagnostics) | `artifacts/bt-refit-vN/` (`ratings.csv`, `metrics.json`) |
| Training runs | `artifacts/train-vN/metrics.json` + `checkpoints/train-vN/*.pt` |
| Decisions + summary numbers | `docs/research/scoring-gt-research-log.md` §5 — **in faceiq-labs** (canonical); local copy is reference only |

## Reference docs (copied from faceiq-labs — canonical copies live there)

- `docs/research/scoring-gt-training.md` — this repo's charter
- `docs/research/scoring-gt-core.md` — §6-§8 BT, calibration, training methodology
- `docs/research/scoring-gt-research-log.md` — results so far; §5 is what we fill next
- `.cursor/skills/bt-refit/` and `.cursor/skills/preference-training/` — step-by-step
  methodology for the two main jobs
