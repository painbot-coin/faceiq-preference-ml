# Meruzhan — next steps (comparator arm)

*Aligned to the team one-benchmark report. Yardstick: uniform run-4 pairs,
panel majority, leakage stratum `bothHeldOut`.*

## Your lane

**Neural comparator configs.** Success is only Goal A on the shared yardstick:

> Held-out faces, **uniform** pairs, did we pick the **panel majority** winner?

Human level on that draw: **73.2%**. Ranking control on bothHeldOut ≈ **85%**.

## Scoreboard (bothHeldOut majority, uniform)

| Run | bothHeldOut | n | vs v2-qc control | Verdict |
|---|---:|---:|---:|---|
| Labs formula | 71.7%* | — | — | baseline |
| `train-v14-panel-ship` | 78.4% | ~110 | — | prior ship |
| `train-v22-ship-0.2` | 80.2% | — | — | prior |
| `train-v23b-bt-distill-gpu` | 82.6% | 109 | 85.3% (−2.8) | prior best |
| `train-v24-arcface-r100` | 78.4% | 111 | 84.7% (−6.3) | **killed** |
| **`train-v25-panel-ft-v23b`** | **83.8%** | 111 | 85.6% (−1.8) | **current best** |
| `ensemble-v25-v1` | 77.8% | 108 | 86.1% (−8.3) | **killed** |
| `ensemble-v25-v4` | 81.7% | 109 | 85.3% (−3.7) | **killed** |
| `ensemble-v25-dino` | 78.6% | 112 | 84.8% (−6.2) | **killed** |
| `train-v26-margin-ft-v25` | 79.3% | 111 | 85.6% (−6.3) | **killed** |
| `train-v27-neartie-ft-v25` | 80.4% | 112 | 84.8% (−4.5) | **killed** |
| MediaPipe ratio late-fusion (v25 + Δratios) | 82.1% | 112 | — | **killed** (−1.8 vs score-sign) |

\*Labs pooled figure from the team report; not the same leakage stratum.

## Closed experiments

| Experiment | Result |
|---|---|
| More panel labels (`train-v22`) | Dead for production score |
| Gap-routing (predicted) | +0.0 bothHeldOut |
| BT soft-distill, head-only (v23 CPU) | 78.6% — no gain |
| BT soft-distill, **unfrozen** (v23b GPU) | 82.6% — ship vs v14 |
| Larger face backbone R100 (v24) | 78.4% — kill; more memorisation |
| Panel-only FT from v23b (v25) | **83.8%** — ship; +1.2 vs v23b (n=111, noisy) |
| Score ensemble v25+ResNet18 (v1) | 77.8% — kill; +8.9 in-sample advantage |
| Score ensemble v25+ResNet18-tuned (v4) | 81.7% — kill vs v25 |
| Score ensemble v25+DINOv2 (v3) | 78.6% — kill |
| Margin ranking FT from v25 (v26) | **79.3%** — kill; −4.5 vs v25 bothHeldOut |
| Near-tie BT-gap weighted BCE FT (v27) | **80.4%** — kill; −3.4 vs v25 bothHeldOut |
| Placement / readout ablations on v25 | **≤83.9%** — kill as accuracy lever (see below) |
| MediaPipe frontal-ratio late fusion on v25 | **82.1%** (n=112 score-sign) — kill; 0/8 control-right flips |
| Global Labs blend | Do not blend |

### v23b detail

Unfrozen BT soft-distill (vote-blind `bt-refit-v2-qc`), warm-start v14, best ep 6.

| Stratum | placed | v2-qc | margin |
|---|---:|---:|---:|
| both trained (1497) | 83.2% | 81.6% | +1.6% |
| one held out (757) | 81.1% | 82.4% | −1.3% |
| both held out (109) | **82.6%** | 85.3% | −2.8% |

### v24 detail (killed)

Same v14 labels, `arcface_r100` (antelopev2 glintr100). Best ep 9 on panel_val.
Pooled majority 82.3% looks fine; bothHeldOut collapses to **78.4%** with
in-sample advantage **+7.2%** [+0.6, +14.3] — memorisation, not a better scorer.
Keep the R100 loader in code; do not train another R100 schedule without a new idea.

## Closed: `train-v25-panel-ft-v23b`

Warm-start v23b → panel-only FT (5 ep, lr 5e-6). Best ep 4.

| Stratum | placed | v2-qc | margin |
|---|---:|---:|---:|
| both trained (1501) | 82.9% | 81.3% | +1.6% |
| one held out (756) | 81.2% | 82.4% | −1.2% |
| both held out (111) | **83.8%** | 85.6% | −1.8% |

Ship vs v23b on point estimate (+1.2). Gap to ranking control narrowed (−2.8 → −1.8).
CI on bothHeldOut margin still wide; treat as a small win, not a breakthrough.

## Closed: score-level ensembles (hurt bothHeldOut)

Z-score average of v25 with older non-ArcFace runs (same as historical `ensemble-v7-v1`
recipe). Helped τ in the VLM era; **hurts panel majority on bothHeldOut** now.

| Ensemble | bothHeldOut | vs v25 |
|---|---:|---|
| v25 alone | **83.8%** | — |
| v25 + train-v1 (ResNet18) | 77.8% | −6.0 |
| v25 + train-v4 (ResNet18 tuned) | 81.7% | −2.1 |
| v25 + train-v3 (DINOv2 probe) | 78.6% | −5.2 |

Do not ship. Helper: `scripts/score_ensemble_from_csvs.py`.

## Closed: `train-v26-margin-ft-v25`

Warm-start v25 → margin ranking loss only (5 ep, lr 5e-6, no distill). Best ep 3.

| Stratum | placed | v2-qc | margin |
|---|---:|---:|---:|
| both trained (1501) | 82.9% | 81.5% | +1.4% |
| one held out (756) | 81.7% | 82.3% | −0.5% |
| both held out (111) | **79.3%** | 85.6% | −6.3% |

**Kill.** bothHeldOut 79.3% ≤ v25 83.8% (−4.5). In-sample advantage +7.7% [+1.1, +14.6]; held-out collapsed while train strata stayed flat — margin loss did not help Goal A.

## Closed: `train-v27-neartie-ft-v25`

Warm-start v25 → BCE + near-tie BT-gap weights from vote-blind `bt-refit-v2-qc`
(eps=0.1, no distill). Best ep 2 on panel_val. Uniform run-4, panel majority.

| Stratum | placed | v2-qc | margin |
|---|---:|---:|---:|
| both trained (1503) | 82.3% | 81.6% | +0.7% |
| one held out (758) | 80.3% | 82.3% | −2.0% |
| both held out (112) | **80.4%** | 84.8% | −4.5% |

**Kill.** bothHeldOut 80.4% ≤ v25 83.8% (−3.4). In-sample advantage +5.2%.
Matches the audit caveat: the 8 control-right misses were not near-tie-concentrated,
so upweighting |Δθ| did not move Goal A.

## Closed: placement / readout ablations (v25 scores, eval-only)

`labs_composite_eval --leakage-split` picks winners by **placed `/10`** (hard sign vs
tier-stratified refs, default n=200) — not raw `modelScore` sign. Helper:
`scripts/placement_readout_ablation.py` →
`artifacts/train-v25-panel-ft-v23b/placement-readout-ablation.json`.

Uniform run-4, panel majority, bothHeldOut:

| Readout | bothHeldOut | n | vs place@200 | Notes |
|---|---:|---:|---:|---|
| **place hard refs=200 (ship)** | **83.8%** | 111 | — | production path |
| place hard refs=50 | 82.4% | 102 | −1.4 | more ties / coarser ladder |
| place hard refs=100 | 83.3% | 108 | −0.5 | |
| place hard refs=400 | 83.8% | 111 | +0.0 | identical to 200 |
| raw `modelScore` sign | 83.9% | 112 | +0.1 | 1 pair skipped as place-tie |
| z-score within gender → raw | 83.9% | 112 | +0.1 | noop (affine preserves signs) |
| soft-margin place T∈{0.5,1,2} @200 | 83.9% | 112 | +0.1 | same picks as raw |
| soft T=1 + z-score @200 | 83.9% | 112 | +0.1 | same |

**Kill as an accuracy lever.** Nothing clears a ≥+2 pt bar on n≈110; soft-margin and
raw only recover the single placement-tie pair (+0.1). Keep hard place @200 for the
product `/10` path (consult-a still stands); do not chase readout knobs for Goal A.

## Next
Still ~1.8 pts behind ranking control (v25 remains ship baseline).

1. **`train-v26-margin-ft-v25`** — **killed** (bothHeldOut 79.3% / n=111 vs v25 83.8%).
2. **Error audit (done)** — `artifacts/train-v25-panel-ft-v23b/bothHeldOut-error-audit.json`.
   18 model errors on uniform run-4 bothHeldOut — **8** control-right / model-wrong,
   **10** both-wrong. The 8 are **not** concentrated in small BT gaps (only 3/8 with
   |Δθ|<1; median ≈2.55; 4/8 with |Δθ|≥3).
3. **`train-v27-neartie-ft-v25`** — **killed** (bothHeldOut 80.4% / n=112 vs v25 83.8%;
   control 84.8%). Near-tie weighting confirmed live (30,599/33,449 train pairs) but
   did not help held-out majority.
4. **Placement / readout tightening** — **killed** as Goal A lever (table above). Ship
   readout unchanged: hard place, 200 tier-stratified refs.
5. **Harsh data-lab fusion — Option 2 slice (done 2026-08-12)** — miss × feature cross →
   `artifacts/train-v25-panel-ft-v23b/bothHeldOut-miss-feature-cross.{json,csv}` + brief
   [`harsh-data-lab-fusion-brief.md`](./harsh-data-lab-fusion-brief.md).
   Photo QC **not** enriched on the 8 control-right misses; ethnicity mismatch 6/8
   descriptive only — **do not** ship demographic late-fusion.
6. **Option 1 late-fusion — self-serve MediaPipe (done 2026-08-13)** — extracted 6
   frontal ratios for 2,998/3,000 cohort faces from export photos
   (`scripts/extract_frontal_ratios.py` → `artifacts/fusion-landmarks-v1/`). Logistic
   head on `[s_a−s_b, Δratios]`, train=5,942 panel pairs with **no** held-out faces,
   eval=uniform run-4 bothHeldOut: **82.1%** (n=112) vs score-sign **83.9%** / place
   **83.8%** — **KILL**. Flipped **0/8** control-right misses. Reproduce:
   `python scripts/late_fusion_landmarks.py --run train-v25-panel-ft-v23b`.

**Recommended next bet:** stop MediaPipe-ratio late fusion. Still want Labs
`frontLandmarks` (+ Labs-derived ratios / pose) for production parity and a second
schema check — but do **not** expect the same recipe to clear the yardstick unless
Labs geometry is materially richer (pose, perspective, anatomically curated ratios).
Next comparator lever is **not** another pixel-only loss FT; prefer a distinct path
(e.g. architecture / training objective rethink, or wait on Labs schema for a
targeted re-try) rather than fusion churn.

Do **not** re-open gap-routing, head-only distill, R100, ResNet/DINOv2 score ensembles,
margin-loss FT, near-tie BT-gap weighted BCE FT, placement/readout knobs,
ethnicity/QC-flag fusion, or MediaPipe-ratio late fusion for accuracy.

## Report line for the team

```
comparator train-v25-panel-ft-v23b: bothHeldOut majority = 83.8% (n=111, uniform run-4)
  vs ranking control 85.6% (−1.8); still ship baseline
train-v26-margin-ft-v25: 79.3% (n=111) vs control 85.6% (−6.3) — killed (≤83.8%)
train-v27-neartie-ft-v25: 80.4% (n=112) vs control 84.8% (−4.5) — killed (≤83.8%)
score ensembles v25+{v1,v4,dino}: 77.8% / 81.7% / 78.6% — all killed (hurt held-out)
placement/readout ablations on v25: raw/soft ≤83.9% (n=112); refs 50/100/400 ≤83.8%
  — killed as accuracy lever; keep hard place@200 for /10 product path
harsh fusion Option 2: miss×feature cross shipped (attrs/QC thin)
Option 1 self-serve MediaPipe ratios: bothHeldOut 82.1% (n=112) vs score-sign 83.9%
  — killed; 0/8 control-right flips; still want Labs frontLandmarks for prod parity
```
