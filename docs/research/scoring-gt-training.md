# Scoring GT — Model Training (Offline)

**Scope:** Train the **preference comparator** after GT labeling and Bradley–Terry refit. This step runs **outside** `/admin/pairwise` — the admin UI produces exports; training consumes them.

**Related:** [`scoring-gt-core.md`](./scoring-gt-core.md) §8 · [`scoring-gt-research-log.md`](./scoring-gt-research-log.md) §5.3 · [`scoring-gt-schema.md`](./scoring-gt-schema.md)

---

## 1. What runs where

| Step | Where | Tooling |
|------|-------|---------|
| Import cohort, VLM labeling, human review, **export API** | `/research/pairwise` | **faceiq-labs** (Next.js + Prisma) |
| **Bradley–Terry refit, calibration, stability** | **`faceiq-preference-ml`** | Python (`scipy` / `choix` / custom) |
| Export matchup JSON + face manifest | Admin `GET …/export` | → ML project `data/exports/` |
| **Train preference model** | **`faceiq-preference-ml`** | Python + PyTorch |
| **Primary evaluation** | Same ML project | Held-out human ratings |
| Deploy inference | TBD | Separate from training |
| *(Optional)* Persist BT θ to DB | faceiq-labs admin | Schema §3.3 — not built |

Admin answers: *“What are the labels and rankings?”*  
Training answers: *“Can a NN learn those preferences from pixels?”*

**Do not intermingle training code with the Next.js app.** The app repo stays labeling + export; training lives in its own folder/repo.

---

## 2. Recommended layout — separate project

Use a **sibling repo or folder**, not `faceiq-labs/src/`:

```text
~/Developer/GitHub/
├── faceiq-labs/                 ← this repo (admin, DB, export only)
└── faceiq-preference-ml/        ← new Python project (recommended)
    ├── README.md
    ├── pyproject.toml           or requirements.txt
    ├── src/
    │   ├── dataset.py           load export JSON + images
    │   ├── model.py             backbone + pairwise head
    │   └── train.py             training loop
    ├── notebooks/
    │   └── explore.ipynb        optional exploration
    ├── data/                    ← gitignored (exports copied here)
    │   ├── exports/
    │   └── images/              or symlinks to export images
    ├── artifacts/               ← gitignored (metrics, plots)
    └── checkpoints/             ← gitignored (.pt weights)
```

**Why separate instead of gitignore inside faceiq-labs?**

- No confusion with Next.js routes, Prisma, or prod deploy
- Python deps (torch, CUDA) don’t bloat the app `package.json`
- Data and weights never risk a commit to the app repo
- Open **both folders in one Cursor window** (File → Add Folder to Workspace) — agent works across both

**`.gitignore` in `faceiq-preference-ml` only:**

```gitignore
data/
checkpoints/
artifacts/
.venv/
__pycache__/
*.pt
*.pth
.DS_Store
```

Commit: code, configs, small `configs/train-v1.yaml`. Never commit: photos, exports, weights.

---

## 3. Epochs vs batches (training vocabulary)

| Term | Meaning |
|------|---------|
| **Batch (mini-batch)** | One forward/backward step on a subset of the dataset (e.g. 64 pairs) |
| **Epoch** | **One full pass** through the entire training set |

Example: 500 training pairs, batch size 64 → ~8 batches per epoch. **One epoch completes when all ~8 batches have been processed once.** Multiple epochs = repeat until validation accuracy plateaus.

This is unrelated to **VLM labeling batches** (`VlmPilotVlmBatch`) — those are API job groupings in the admin DB.

---

## 4. Jupyter, Colab, and Cursor

### What is `.ipynb`?

A **Jupyter notebook** — Python (usually) in cells with code, output, and markdown. Cursor opens and edits `.ipynb` files locally like any other file.

### Jupyter vs Google Colab

| Tool | What it is |
|------|------------|
| **Jupyter** | Software you run **locally** (or on a server). Notebooks live on your disk. |
| **Google Colab** | **Hosted Jupyter in the browser** by Google — free/cheap GPU, notebooks stored in Google Drive |

Colab is optional. You can do everything locally in Cursor with a `.ipynb` or plain `.py` files.

### Practical workflow

1. **Scaffold `faceiq-preference-ml`** — `train.py` + optional `explore.ipynb`
2. **Open in Cursor** — multi-root workspace: `faceiq-labs` + `faceiq-preference-ml`
3. **Smoke test locally** (CPU / Apple MPS) on 500 pairs
4. **Full GPU run** — local NVIDIA GPU, or upload notebook to Colab / RunPod if needed

Agent in Cursor writes Python in the ML repo; admin export comes from `faceiq-labs`.

---

## 5. Training inputs (from admin export)

Each training row (see core doc §8):

```text
(photo_A_path, photo_B_path, winner)   # winner ∈ {A, B, tie}
```

| Rule | Why |
|------|-----|
| Split by **face id**, not random pairs | Prevent leakage |
| Do **not** use Labs `overall_score` as label | Would reproduce old formula |
| Images | Read from `data/exports/` — never commit |

Target split: ~80% train / ~20% val matchups by face id.

---

## 6. What to log (granularity)

| Store in ML repo `artifacts/` | Store in faceiq-labs DB | Store in research log |
|------------------------------|-------------------------|------------------------|
| Loss curves, checkpoints, `config.yaml` | Labeling, VLM, BT (see schema doc) | Decisions + summary metrics |
| Val pairwise accuracy | Optional metric snapshots | Primary eval §6 |

---

## 7. End-to-end checklist

**Labeling + audit (faceiq-labs):**

- [x] Labeling complete — run `cmr1mr0m7000196d57zi3vcgn`, 52,500 VLM labels, `pairwise-v3-gt`
- [x] Human audit gate — 750-pair sample, 84.9% accuracy, 86 export exclusions
- [ ] **Export full run to disk** — paginate `GET /api/admin/pairwise/runs/cmr1mr0m7000196d57zi3vcgn/export?promptVersion=pairwise-v3-gt&offset=&limit=2000` until `total` exhausted; save under `~/research-data/scoring-gt/artifacts/gt-full-export/`

**ML repo (`faceiq-preference-ml` — create if missing):**

- [ ] Scaffold repo (`pyproject.toml`, `src/dataset.py`, `src/bt_refit.py`, `src/train.py`)
- [ ] Copy export JSON + symlink or copy face images to `data/exports/`
- [ ] Implement **finalize label** helper: human winner if audited, else VLM winner
- [ ] **BT refit** + stability check (80% subsample Spearman ρ > 0.95)
- [ ] **Calibration** — percentile → `/10` (core §12 anchor table)
- [ ] **Validate** — Spearman ρ vs Labs `overall_score` (log §5.2)
- [ ] Smoke train locally (CPU/MPS) on subset
- [ ] Full train on GPU; save checkpoint — log §5.3
- [ ] Primary evaluation on held-out human ratings (§6)

**Optional (faceiq-labs later):**

- [ ] Prisma §3.3 BT rating tables + admin dashboard
- [ ] Apply human overrides to all 1,032 labeled rows in export finalize script (already in export JSON)

---

## 8. When to consider SageMaker / cloud

Optional for v1. Use Colab / RunPod / local GPU first. SageMaker only if you need managed retraining pipelines later.

**Note:** Existing `sagemaker/front-model/` in faceiq-labs is for **landmark inference**, not preference training.

---

## 9. Open decisions (fill when starting)

| Decision | Choice |
|----------|--------|
| ML repo name / location | `faceiq-preference-ml` suggested |
| Backbone | |
| Image resolution | |
| Batch size / epochs | |
| Tie handling | |
