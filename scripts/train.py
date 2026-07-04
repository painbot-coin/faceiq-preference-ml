#!/usr/bin/env python
"""Train the pairwise preference comparator.

Usage:
    python scripts/train.py --config configs/train-v1.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from faceiq_pref.data import load_export
from faceiq_pref.train import TrainConfig, train


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config (see configs/train-v1.yaml)")
    args = ap.parse_args()

    cfg = TrainConfig.from_yaml(args.config)
    export = load_export(cfg.export_dir)
    print(f"export OK: run {export.run_id} — training run {cfg.run_name!r}")

    metrics = train(export, cfg)
    print(f"\nbest val accuracy: {metrics['best_val_accuracy']:.4f}")
    print(f"metrics: artifacts/{cfg.run_name}/metrics.json")
    print(f"checkpoint: checkpoints/{cfg.run_name}/best.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
