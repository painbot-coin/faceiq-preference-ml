# FaceIQ Preference GT — Dev Team Brief (1 page)

**Goal:** replace the legacy formula-based attractiveness score with a ranking learned
from real pairwise preference judgments — then test whether a neural net can learn
those preferences from pixels alone.

---

## 1. Data gathering

- **3,000 curated faces** (1,500 M / 1,500 F), stratified from the parent pool.
- Pair queue built *before* labeling: each face in **~35 same-gender matchups** → 52,500 pairs.
- **VLM labeled every pair** (Gemini 2.5 Flash, locked prompt): A wins / B wins / tie.
- **Human audit** on a 750-pair sample: **84.9% agreement** (~15% override rate) — passed the ~85% gate.
- Final label rule: **human override when audited, else VLM**. 86 bad pairs excluded → **52,414 pairs exported**.
- Export = frozen files (JSONL + images + sha256 manifest). The ML repo **never touches the app DB**.

## 2. Bradley–Terry ranking (the ground truth)

- **Probabilistic model:** each face gets a latent strength θ; P(A beats B) grows with the gap θA − θB.
- The fit finds the set of θ's that **best explains all 52k outcomes at once** (maximum likelihood — unlike Elo, which updates game-by-game).
- **Transitive by construction:** fitted scores are scalars, so model preferences can't cycle (A>B>C>A impossible). Individual upsets in the data are fine — they're treated as probability, not contradiction.
- **Quality gates passed:** connected comparison graph · every face ≥15 comparisons · **stability ρ = 0.9925** (refit on 80% subsamples gives near-identical rankings).
- **Calibration:** percentile rank → pre-registered /10 curve (median = 5.0, top 10% = 7.0, top 1% = 8.0, cap 9.0 — a 3k sample can't resolve rarer tails).
- **Sanity vs legacy score: Spearman ρ ≈ 0.75** — strongly correlated (not random), yet clearly not duplicating the old formula. Promising: agreement at the extremes, new signal in the middle.

## 3. Neural comparator training

- Training rows: **(photo_A, photo_B, winner)** — always pairs.
- Train/val split **by face id** (a face never appears in both splits — prevents identity leakage).
- Architecture: **siamese pairwise** ("single-input"). Each photo is fed through the
  *same* network independently — one scalar score per face — then the scores are compared:

```
photo_A ──► PreferenceScorer (shared weights) ──► s(A) ─┐
                                                         ├──► s(A) − s(B) ──► P(A wins)
photo_B ──► PreferenceScorer (same weights)   ──► s(B) ─┘
```

- Loss: sigmoid over s(A) − s(B) vs the winner label, binary cross-entropy (**RankNet**).
- **Why not feed both photos in at once** (pair-input)? Joint models can learn cyclic,
  inconsistent preferences and give no per-face score. Scoring independently guarantees a
  coherent global ranking — higher score simply wins.
- Backbone: pretrained **ResNet-18** (swappable: ResNet-50, face-recognition backbones, ensembles later).
- Metrics: **held-out pairwise accuracy** (primary — % of unseen matchups predicted correctly)
  + **Kendall τ / Spearman ρ vs the BT ranking** (does the model's global ordering match GT?).

**Mental model:** train on pairs · score each image separately through shared weights ·
subtract · higher score wins.

## 4. Production inference (later)

The comparator outputs *relative* scores only. A new face is scored against a small
**reference panel of anchor faces** with known BT scores ("anchor ladder"); literature says
a modest panel (~5–10 anchors) beats comparing against everything. Panel spread also gives
free per-prediction confidence.

---

## Key numbers

| Metric | Value |
|---|---|
| VLM–human agreement (audit) | 84.9% |
| Exported pairs / faces | 52,414 / 3,000 |
| BT stability ρ (80% subsample) | 0.9925 |
| BT vs legacy score ρ | 0.745 (F) / 0.752 (M) |
| Faces scored | 2,999 (1 dropped: under-connected) |
