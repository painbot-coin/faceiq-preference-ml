#!/usr/bin/env bash
# Launch train-v27-neartie-ft-v25 on the GPU box (nohup-safe).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

RUN=train-v27-neartie-ft-v25
CFG="configs/${RUN}.yaml"
mkdir -p "artifacts/$RUN" "checkpoints/$RUN"

python - <<'PY'
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, "src")
from faceiq_pref.panel import load_panel_targets, load_rejects
from faceiq_pref.train import TrainConfig, load_bt_gap_weights

cfg = TrainConfig.from_yaml("configs/train-v27-neartie-ft-v25.yaml")
assert cfg.loss_type == "bce", cfg.loss_type
assert cfg.bt_gap_weighting is True
assert cfg.bt_distill_weight == 0.0, cfg.bt_distill_weight
assert cfg.run_name == "train-v27-neartie-ft-v25"
assert cfg.bt_ratings and os.path.isfile(cfg.bt_ratings), cfg.bt_ratings
assert os.path.isfile(cfg.init_checkpoint), cfg.init_checkpoint

P = [tuple(x) for x in cfg.panel_labels]
t = load_panel_targets(
    P,
    load_rejects(cfg.panel_rejects or []),
    min_votes=cfg.panel_min_votes,
    prior=cfg.panel_prior,
    hard=cfg.panel_hard,
)
print(f"preflight OK: targets={len(t)} loss={cfg.loss_type} gap_eps={cfg.bt_gap_eps}")

# Lightweight gap-weight smoke (no full export load): two synthetic pairs from ratings.
import pandas as pd

rank = pd.read_csv(cfg.bt_ratings)
ids = rank["faceId"].tolist()[:4]
assert len(ids) >= 4, "bt ratings too short"
fake = [
    SimpleNamespace(pair_index=1, face_a_id=ids[0], face_b_id=ids[1]),
    SimpleNamespace(pair_index=2, face_a_id=ids[2], face_b_id=ids[3]),
]
gw = load_bt_gap_weights(fake, cfg.bt_ratings, cfg.bt_gap_eps)
assert len(gw) == 2 and abs(sum(gw.values()) / 2 - 1.0) < 1e-6
print(f"gap weight smoke: { {k: round(v, 3) for k, v in gw.items()} }")
print(f"init_checkpoint present: {cfg.init_checkpoint}")
print(f"bt_ratings (vote-blind): {cfg.bt_ratings}")
PY

LOG="artifacts/$RUN/train.log"
echo "starting $RUN -> $LOG"
# Train, then evaluate + leakage yardstick so the kill vs 83.8% is ready when done.
setsid bash -c "
  cd '$PWD'
  source .venv/bin/activate
  set -x
  python scripts/train.py --config $CFG
  python scripts/evaluate.py --checkpoint checkpoints/$RUN/best.pt \
      --ratings artifacts/bt-refit-v5-panel/ratings.csv
  python scripts/labs_composite_eval.py --run $RUN --leakage-split
" < /dev/null > "$LOG" 2>&1 &
echo "pid=$!"
echo "follow: tail -f $LOG"
