---
name: preference-training
description: Train and evaluate the neural pairwise preference comparator on faceiq-labs matchup exports. Use when asked to train the preference model, tune training configs, evaluate pairwise accuracy, or compare model rankings against Bradley-Terry.
---

# Preference comparator training

Trains a siamese CNN that scores single face photos such that the higher-scored face
wins the pairwise comparison. Methodology: `docs/research/scoring-gt-core.md` §8 and
`docs/research/scoring-gt-training.md`.

## Run

```bash
python scripts/train.py --config configs/train-v1.yaml
```

Outputs: `checkpoints/<run-name>/best.pt` + `artifacts/<run-name>/metrics.json`
(per-epoch train loss, val loss, val pairwise accuracy).

## Non-negotiable rules

1. **Label = `finalOutcome` only** (human when audited, else VLM). Never Labs
   `overall_score`, never BT theta as the primary loss target.
2. **Split by face id, not by pair.** A pair goes to val only if BOTH faces are val
   faces; pairs mixing train/val faces are dropped. Target ~80/20 by face.
3. Ties: default is skip for v1 (log the count); alternative is soft target 0.5.
4. Same-gender pairs only (that is all the export contains) — the model learns
   within-gender preference.

## Architecture (v1 baseline)

- Backbone: torchvision `resnet18` or `resnet50` (config), pretrained, final FC ->
  scalar score `s(photo)`.
- Pairwise head: `P(A wins) = sigmoid(s(A) - s(B))`; binary cross-entropy on winner.
- Both photos go through the **same shared backbone** (siamese).
- Input: 224x224, standard ImageNet normalization; light augmentation (flip is OK —
  left/right presentation was already randomized at labeling time).

## Evaluation

- **Primary**: pairwise accuracy on held-out (face-id-split) matchups.
- **Secondary**: Kendall tau / Spearman between model-implied face ranking (mean score)
  and the BT theta ranking from `artifacts/bt-refit-vN/ratings.csv`.
- Context: single VLM labeler audited at ~85% vs human; a comparator in the low-to-mid
  80s on held-out pairs is near the label ceiling.

## After a run

Log summary (config, epochs, val accuracy, tau vs BT, checkpoint path) in faceiq-labs
`docs/research/scoring-gt-research-log.md` §5.3.
