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

Later (fusion bet, not a blocker): Harsh data lab / new feature signals.

Do **not** re-open gap-routing, head-only distill, R100, ResNet/DINOv2 score ensembles,
margin-loss FT, or near-tie BT-gap weighted BCE FT from v25.

## Report line for the team

```
comparator train-v25-panel-ft-v23b: bothHeldOut majority = 83.8% (n=111, uniform run-4)
  vs ranking control 85.6% (−1.8); still ship baseline
train-v26-margin-ft-v25: 79.3% (n=111) vs control 85.6% (−6.3) — killed (≤83.8%)
train-v27-neartie-ft-v25: 80.4% (n=112) vs control 84.8% (−4.5) — killed (≤83.8%)
score ensembles v25+{v1,v4,dino}: 77.8% / 81.7% / 78.6% — all killed (hurt held-out)
```
