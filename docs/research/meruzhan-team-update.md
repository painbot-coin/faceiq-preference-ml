# Comparator arm update — Meruzhan

**Date:** 2026-08-10  
**Owner:** Meruzhan (Harsh on parallel configs / data lab)  
**Yardstick (same for every arm):** uniform run-4 pairs → did we pick the **panel majority** winner on faces the model never trained on (`bothHeldOut`).

Benchmarks on that draw: coin flip 50% · human-vs-crowd 73.2% · Labs 71.7% · ranking control ≈85% · our target is the ranking.

---

## Headline

Current best comparator: **`train-v25-panel-ft-v23b` at 83.8%** bothHeldOut majority (n=111).

That is **+5.4 pts vs v14** (78.4%), **+1.2 vs the distill run (v23b)**, and **~1.8 pts behind** the vote-blind ranking control on the same stratum (85.6%). Still ahead of Labs and the human-vs-crowd line; still short of the ranking we distill from.

```text
comparator train-v25-panel-ft-v23b: bothHeldOut majority = 83.8% (n=111, uniform run-4)
  vs human 73.2% / Labs 71.7% / ranking control 85.6% (−1.8)
```

---

## What I ran (and the verdict)

| Run | Idea | bothHeldOut majority | Verdict |
|---|---|---:|---|
| v14 (prior ship) | Panel-hard ArcFace R50 | 78.4% | baseline |
| v22 | More / richer panel labels | ~flat vs humans | closed earlier — labels saturated |
| Gap-routing | Cmp near / ranking far | +0.0 predicted | **killed** (can't ship; uploads have no ranking row) |
| v23 CPU | BT soft-distill, **head-only** | 78.6% | no gain |
| **v23b GPU** | BT soft-distill, **unfrozen**, warm-start v14 | **82.6%** | **ship** — first real move |
| v24 | Bigger face backbone (ArcFace R100) | 78.4% | **killed** — pooled looked fine, held-out collapsed (memorisation) |
| **v25** | Warm-start v23b → short **panel-only** FT | **83.8%** | **current best** |
| Ensembles | Z-score avg of v25 + ResNet/DINOv2 | 77.8–81.7% | **killed** — all worse than v25 alone |

Teacher for distill was always vote-blind (`bt-refit-v2-qc`) so panel eval isn't circular.

---

## What this means

1. **Not everything is flat.** Unfrozen BT soft-distill (v23b) and a short panel fine-tune on top (v25) moved the honest metric. Head-only distill did not — the backbone had to move.
2. **Bigger face backbone ≠ better scorer.** R100 memorised train faces; bothHeldOut fell to v14 level. Keep the loader; don't burn another R100 schedule without a new idea.
3. **Diverse score ensembles that once helped Kendall τ now hurt majority.** Blending v25 with ImageNet/DINOv2 runs pulled held-out accuracy down. Do not ship.
4. **Gap to the ranking is smaller but real** (~1.8 pts on bothHeldOut). Extraction is still the bottleneck, not more of the same A/B labels.

---

## Ship recommendation (comparator arm)

- **Production candidate:** `checkpoints/train-v25-panel-ft-v23b/best.pt` (warm-start chain: v14 → v23b distill → v25 panel FT).
- **Do not blend** with Labs or with older ResNet/DINOv2 checkpoints for the majority metric.
- Bands/tiers stay a display layer on whatever gap the scorer already produces.

---

## Next (needs sync with Harsh / Ditmar)

Comparator configs are near the end of the free axes we listed. Before another GPU day:

1. Align with **Harsh** on data lab / any new signals worth training on.
2. Optional cheap residual: lower distill-weight warm-start (only if GPU queue is free).
3. Hand-ranging / absolute-scale check stays with Ditmar — ordering is what we measured; anchors are still open.

Detail scoreboard: `docs/research/meruzhan-next-steps.md`.
