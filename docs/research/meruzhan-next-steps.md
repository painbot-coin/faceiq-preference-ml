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

## Next

Still ~1.8 pts behind ranking control. Open bets that are not “more labels / bigger ArcFace /
blend with a weak ImageNet run”:

1. **Coordinate with Harsh** — data lab / any new feature signals before another GPU day.
2. **Lower distill weight warm-start** (v23b → w=0.3 panel+soft mix) only if Harsh agrees
   the GPU queue is free; otherwise pause comparator configs.

Do **not** re-open gap-routing, head-only distill, R100, or ResNet score ensembles.

## Report line for the team

```text
comparator train-v25-panel-ft-v23b: bothHeldOut majority = 83.8% (n=111, uniform run-4)
  vs ranking control 85.6% (−1.8); still ship baseline
score ensembles v25+{v1,v4,dino}: 77.8% / 81.7% / 78.6% — all killed (hurt held-out)
```
