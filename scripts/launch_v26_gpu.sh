#!/usr/bin/env bash
# Launch train-v26-margin-ft-v25 on the GPU box (nohup-safe).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

RUN=train-v26-margin-ft-v25
mkdir -p "artifacts/$RUN" "checkpoints/$RUN"

python - <<'PY'
import sys
sys.path.insert(0, "src")
from faceiq_pref.panel import load_panel_targets, load_rejects
from faceiq_pref.train import TrainConfig, pairwise_loss
import torch, torch.nn as nn
cfg = TrainConfig.from_yaml("configs/train-v26-margin-ft-v25.yaml")
assert cfg.loss_type == "margin", cfg.loss_type
assert cfg.run_name == "train-v26-margin-ft-v25"
P = [tuple(x) for x in cfg.panel_labels]
t = load_panel_targets(
    P, load_rejects(cfg.panel_rejects or []),
    min_votes=cfg.panel_min_votes, prior=cfg.panel_prior, hard=cfg.panel_hard,
)
print(f"preflight OK: targets={len(t)} loss={cfg.loss_type} margin={cfg.margin}")
bce = nn.BCEWithLogitsLoss(reduction="none")
loss = pairwise_loss(
    torch.tensor([2.0, -1.0, 0.5]),
    torch.tensor([1.0, 0.0, 1.0]),
    torch.ones(3),
    "margin", 1.0, bce,
)
print(f"margin smoke loss={float(loss):.4f}")
import os
assert os.path.isfile(cfg.init_checkpoint), cfg.init_checkpoint
print(f"init_checkpoint present: {cfg.init_checkpoint}")
PY

LOG="artifacts/$RUN/train.log"
echo "starting $RUN -> $LOG"
# Train, then evaluate + leakage yardstick so the kill vs 83.8% is ready when done.
setsid bash -c "
  cd '$PWD'
  source .venv/bin/activate
  set -x
  python scripts/train.py --config configs/$RUN.yaml
  python scripts/evaluate.py --checkpoint checkpoints/$RUN/best.pt \
      --ratings artifacts/bt-refit-v5-panel/ratings.csv
  python scripts/labs_composite_eval.py --run $RUN --leakage-split
" < /dev/null > "$LOG" 2>&1 &
echo "pid=$!"
echo "follow: tail -f $LOG"
