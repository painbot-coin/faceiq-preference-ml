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
- [x] **Export full run to disk** — done 2026-07-04 via `faceiq-labs/scripts/export-gt-run.ts` (idempotent, resumable) → `faceiq-preference-ml/data/exports/cmr1mr0m7000196d57zi3vcgn/` (manifest + 6 JSONL shards + 3,000 images, sha256-verified)

**ML repo (`faceiq-preference-ml` — create if missing):**

- [x] Scaffold repo (`pyproject.toml`, `src/faceiq_pref/{data,bt,calibrate,validate,model,train,eval}.py`)
- [x] Copy export JSON + symlink or copy face images to `data/exports/`
- [x] Implement **finalize label** helper: human winner if audited, else VLM winner (`data.py` re-derives and asserts vs exporter)
- [x] **BT refit** + stability check (80% subsample Spearman ρ > 0.95) — `bt-refit-v1`, all gates passed (log §5.1)
- [x] **Calibration** — percentile → `/10` (core §12 anchor table) — log §5.2
- [x] **Validate** — Spearman ρ vs Labs `overall_score` (log §5.2) — 0.745 F / 0.752 M
- [x] Smoke train locally (CPU/MPS) on subset
- [x] Full train on GPU; save checkpoint — log §5.3 (v1–v10; best `train-v8` at **78.4%**, EC2 g5.xlarge)
- [x] **Face photo QC** — 133 of 3,000 faces excluded → `bt-refit-v2-qc` (log §5.0)
- [x] **Human panel ground truth** — **4** Prolific studies, 963 raters in the fit, **95,245 votes**,
      **$4,155** (log §5.5, §5.8)
- [x] **Joint BT with per-source discrimination** — ranking of record **`bt-refit-v5-panel`** (log §5.1,
      §5.5, §5.8). `bt-refit-v2-qc` remains the only ranking allowed for band cuts and calibration,
      because it predates every human vote
- [x] **Human-grounded comparator eval** — `scripts/eval_vs_panel.py`; leak-free by construction
- [x] **Retrain on human targets** — `train-v12-panel-soft` 55.8%, `train-v13-panel-hard` **56.4%** vs the
      `train-v10` control 54.3%; ship refit `train-v14` (`val_fraction 0.2`); weighted arm `train-v15`
      (`panel_weight 3.0`) 56.55%. Hard target chosen; arms not separable (log §5.3.1).
      **Reversed 2026-08-04:** on a *population* sample the soft arm wins and separates (v12 67.67% vs
      the control 66.47%, +1.20 [+0.04, +2.35]) while the hard arm ties the control (log §5.8)
- [x] **Rating validation** — per-face uncertainty, /10 calibration, composites (log **§5.6**).
      Median per-face interval **0.91 /10 points**; no blend beats the comparator significantly
- [x] **Uniform random-pair study (run 4)** — 303 raters, 31,237 judgments, **$968.58**, closed
      2026-08-04 (log **§5.8**). Delivered population accuracy (**81.5%** ranking vs a **74.9%** human
      ceiling), a non-circular /10 calibration curve (`T = 1.920`), a proof that the percentile bands are
      not circular (monotone 53.2% → 83.6%), and a **negative** verdict on $14,367 of planned labelling
- [x] **`train-v16-panel-run4`** — run 4's votes in the loss on the soft target; panel coverage of the
      train split 14.7% → **19.4%**, leak-free eval set 1,928 → **2,559** pairs. **Result: a tie with
      `train-v12` (67.43% vs 67.67%, CI [−0.95, +1.46]) and still 2.09 pts behind the ranking, with the
      10–20 band unmoved at 54.7% against BT's 63.2%.** Aiming labels at the weak bands did not fix the
      weak bands, which confirms the deficit is extraction, not data (log §5.8)
- [x] **Checkpoint selection fixed (2026-08-04).** `train()` computes `panel_val_accuracy` every epoch —
      the share of real human votes agreeing with the model on val-split panel pairs — and selects
      `best.pt` on it above 200 such pairs. Matches `eval_vs_panel.py` to 0.02 pts; a smoke run caught the
      two metrics disagreeing (epoch 2 higher on `val_accuracy`, lower on humans). `val_accuracy` is still
      recorded, as the only figure comparable to pre-panel runs
- [ ] **← NOW: close the 10–45 band, without new labels.** Three arms running:
      `train-v17-panel-select` (v16 under the fixed selection), `train-v19-panel-variance` (the never-run
      variance head against panel targets — also the only route to a per-photo σ), and
      `train-v18-arcface-224` (resolution; **confounded** — ArcFace was pretrained at 112, so a null result
      cannot separate "resolution does not help" from "this backbone cannot use it")
- [ ] **Reference-set inference** — score a new face by comparing it against a fixed panel of cohort faces
      and reading off where it lands, rather than trusting the raw scalar. The comparator is *trained*
      pairwise and currently *read* as an absolute score; this is the likeliest explanation for uploads
      scoring oddly, and it needs no training
- [ ] **Gap-routed ensemble** — comparator under a 10-point gap, ranking above it. Measurable today on the
      631 leak-free run-4 pairs with no new data and no training run. §5.6 ruled out *global* blends;
      routing is a different operation and §5.8's band table is the first evidence for it
- [ ] **Test–retest stability** — needs a second photo view per person exported from Labs
      (`<sourceFaceId>_<view>.webp`); **no new labels**. Not possible from the current export, which is
      one `_front` photo per person (log §8.3). Runs in parallel — blocked on a Labs export, not on us.
      Also the hard gate on any "upload more photos, the band narrows" feature
      (`programme-direction-review.md` §5)
- [ ] **Gap-routed ensemble** — comparator under a 10-point gap, ranking above it. Measurable on the 631
      leak-free run-4 pairs with no new data. §5.6 ruled out *global* blends; routing is a different
      operation and run 4 is the first evidence for it
- [ ] Primary evaluation on held-out human ratings (§6) — partial in §6.4

> **Correction worth carrying (2026-08-01):** the 78.4% above is scored against export
> `finalOutcome`, i.e. Gemini's label on ~98% of rows. Against real human votes the same family of
> checkpoints reaches only **54.3%**, level with those Gemini labels and against a 59.1% ceiling.
> Cause: `PairDataset` read `final_outcome`, so no human vote ever entered a batch. Fixed via
> `panel_labels` in `TrainConfig` + `src/faceiq_pref/panel.py` (vote share as a soft target, train
> split only). **Quote 78.4% as "agreement with the VLM labeler", never as model quality.**

> **Second correction, the mirror image of the first (2026-08-04):** the 54.3%/56.4% figures are
> measured on pairs *chosen* to be near-ties, so they are the worst case rather than model quality
> either. On a uniform random draw the same checkpoints score **66.5–67.7%** against a **69.9%**
> vote-level ceiling. **Every accuracy in this repo must state (a) its pair distribution and (b) which
> of the two accuracy metrics it is** — near-tie versus uniform pairs moves a figure ~29 points, and
> majority-level versus vote-level moves it ~10 more. Log §5.8 has the 2 × 2; `scripts/accuracy_matrix.py`
> regenerates it.

**Optional (faceiq-labs later):**

- [ ] Prisma §3.3 BT rating tables + admin dashboard
- [ ] Apply human overrides to all 1,032 labeled rows in export finalize script (already in export JSON)
- [ ] **Production anchor panel** — manually assigned `/10` ladder and/or v2 cohort with thinner decile tails (research log §5.2 elite crowding, §5.4 Path A/B)

---

## 8. When to consider SageMaker / cloud

Optional for v1. Use Colab / RunPod / local GPU first. SageMaker only if you need managed retraining pipelines later.

**Note:** Existing `sagemaker/front-model/` in faceiq-labs is for **landmark inference**, not preference training.

---

## 9. Open decisions (fill when starting)

Settled by the v1–v10 sweep (log §5.3); recorded here so they are not relitigated.

| Decision | Choice |
|----------|--------|
| ML repo name / location | `faceiq-preference-ml` |
| Backbone | **ArcFace R50 (w600k), end-to-end** — beat ResNet-18 76.6%, ResNet-50 75.4%, DINOv2 probe 72.4% |
| Image resolution | **112px** (ArcFace native; 224 only for the ImageNet backbones) |
| Batch size / epochs | **32**, lr 1e-5, wd 1e-4; 8–16 epochs — best epoch is usually 5–8, so longer runs mostly cost time |
| Tie handling | Export ties (113 of 52,414) dropped, `skip_ties: true`. Human near-ties are **not** ties — they arrive as soft targets near 0.5 via `panel_labels` |
| Split | By **face id**, seed 42. `val_fraction` 0.2 to ship, **0.5 for panel-grounded experiments** (2,559 leak-free panel pairs with run 4 folded in, vs 447 at 0.2) |
| Label source | Export `finalOutcome`, overridden by panel **vote share** where we bought votes — train split only. **Soft (`panel_hard: false`)**, reversing the 2026-08-01 hard-target choice: on population pairs soft separates from the control and hard does not (log §5.8) |
| Headline metric | **Two numbers, never one**: near-tie accuracy *and* population accuracy, each with its ceiling, and each labelled majority-level or vote-level. Plus margin recovery and the per-face interval. Do not add a fifth metric to replace one that saturates (`programme-direction-review.md` §2) |
