# faceiq-preference-ml

Offline ML companion to [faceiq-labs](../faceiq-labs): turns ~52k VLM-labeled pairwise
face comparisons into an authoritative Bradley-Terry ranking, then trains a neural
pairwise preference comparator on the same matchups.

The labeling pipeline (import, VLM batch, human audit, export) lives in faceiq-labs.
This repo consumes its exports and never touches the app database.

## Pipeline

```
faceiq-labs export (JSONL + images)
        |
        v
1. BT refit ............ scripts/run_bt.py     -> artifacts/bt-refit-v1/
2. Calibration ......... percentile -> /10 curve (part of run_bt)
3. Validation .......... Spearman vs Labs overall_score (part of run_bt)
4. Train comparator .... scripts/train.py      -> checkpoints/
5. Inspect ............. streamlit run app/dashboard.py
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Getting the data

Run the export from the **faceiq-labs** repo (requires its `.env` DATABASE_URL):

```bash
cd ../faceiq-labs
npx tsx scripts/export-gt-run.ts \
  --run cmr1mr0m7000196d57zi3vcgn \
  --out ~/Developer/GitHub/faceiq-preference-ml/data/exports
```

This writes `data/exports/<runId>/` with `manifest.json`, `matchups/*.jsonl`,
`faces.jsonl`, and downloads face photos to `images/`. Re-running is resumable.

## Commands

| Command | Purpose |
|---------|---------|
| `python scripts/run_bt.py --export data/exports/<runId>` | BT refit + stability + calibration + Labs validation |
| `python scripts/train.py --config configs/train-v1.yaml` | Train pairwise comparator |
| `streamlit run app/dashboard.py` | Rankings / diagnostics / training dashboard |
| `pytest` | Unit tests |

## Ground rules

- **Never commit** anything under `data/`, `checkpoints/`, or `artifacts/` (gitignored).
- **Split by face id**, never by random pair — prevents leakage.
- Training label is the **matchup winner only** (human audit label when present, else
  VLM). Labs `overall_score` is for stratification/validation only, never a label.
- Research context: `docs/research/` (copied from faceiq-labs; faceiq-labs is canonical).
