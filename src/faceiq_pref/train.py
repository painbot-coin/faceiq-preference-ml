"""Training loop for the pairwise preference comparator.

Data flow: export dir -> face-id split -> PairDataset (loads photos lazily) ->
BCE-with-logits on winner -> checkpoints/<run>/best.pt + artifacts/<run>/metrics.json.

Set `panel_labels` in the config to replace the export's hard winner with a human vote share
on the pairs our Prolific panel judged. That applies to the train split only — see the note
in `train()` — so `val_accuracy` stays comparable across runs.

**Checkpoint selection follows the humans, not the VLM.** `val_accuracy` is scored against the
export's `finalOutcome`, which is a Gemini label on ~98% of rows, and research log §5.3.1 measured
that a run can gain 2 points against real people while moving `val_accuracy` by 0.4 — the metric
is structurally blind to the improvement these runs exist to produce. So when `panel_labels` is
set, each epoch also scores the *val* split's panel pairs against real votes (`panel_val_accuracy`)
and `best.pt` is chosen on that. `val_accuracy` is still computed and recorded, because it is the
only figure comparable to the pre-panel runs.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, fields
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from .backbones.arcface import is_arcface_backbone
from .data import Export, Face, Matchup, split_by_face_id
from .model import PairwiseModel, pick_device
from .panel import load_panel_targets, load_panel_votes, load_rejects

# Below this many leak-free val panel pairs the human-grounded metric is noisier than the
# differences it would be selecting on, so selection falls back to `val_accuracy`. At
# val_fraction 0.5 there are ~2,500; at 0.2 only ~450, which is thin but still usable.
MIN_PANEL_VAL_PAIRS = 200

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
# ArcFace w600k_r50: (pixel/255 - 0.5) / 0.5; backbone forward swaps RGB -> BGR.
ARCface_MEAN = [0.5, 0.5, 0.5]
ARCface_STD = [0.5, 0.5, 0.5]


def build_transforms(backbone: str, image_size: int, augment: bool) -> transforms.Compose:
    ops: list = [transforms.Resize((image_size, image_size))]
    if augment:
        ops.append(transforms.RandomHorizontalFlip())
    ops.append(transforms.ToTensor())
    if is_arcface_backbone(backbone):
        ops.append(transforms.Normalize(ARCface_MEAN, ARCface_STD))
    else:
        ops.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return transforms.Compose(ops)


@dataclass
class TrainConfig:
    export_dir: str
    run_name: str = "train-v1"
    backbone: str = "resnet18"
    image_size: int = 224
    batch_size: int = 64
    epochs: int = 10
    lr: float = 1e-4
    weight_decay: float = 1e-4
    val_fraction: float = 0.2
    split_seed: int = 42
    num_workers: int = 4
    skip_ties: bool = True
    augment: bool = True
    max_pairs: int | None = None  # subsample for smoke tests
    freeze_backbone: bool = False  # embedding probe: train only the head
    confidence_filter: str | None = None  # train only: high | medium | low (human labels kept)
    variance_head: bool = False  # UOL-style per-face Gaussian; probabilistic RankNet logit
    # Human panel labels, applied to the TRAIN split only so val accuracy stays comparable
    # to runs fitted on export labels. [[results_dir, sample_meta.json], ...]
    panel_labels: list[list[str]] | None = None
    panel_rejects: list[str] | None = None
    panel_min_votes: int = 4  # pairs with thinner coverage keep their export label
    panel_prior: float = 1.0  # Laplace count per side; stops 6-0 becoming a target of 1.0
    panel_hard: bool = False  # ablation: keep the crowd's winner, discard the margin
    # Loss multiplier on the ~13% of train pairs carrying a human target. The other 87% are
    # still Gemini labels, blind spot included, so at weight 1.0 the human corrections are a
    # minority vote against the very labels they exist to overrule.
    panel_weight: float = 1.0
    # BT soft-target distillation (consult brief §9b — never tried before). When
    # `bt_distill_weight` > 0 and `bt_ratings` is set, train targets become
    #   (1 - w) * base + w * sigmoid((θ_a - θ_b) / T)
    # where base is the panel vote share if present else the export hard winner.
    # Panel pairs always keep their human target (humans beat the teacher). Use a
    # vote-blind ranking (bt-refit-v2-qc) as teacher so panel eval is not circular.
    bt_ratings: str | None = None
    bt_distill_weight: float = 0.0
    bt_distill_temperature: float = 1.0
    # Warm-start from an existing checkpoint (fine-tune). Architecture must match.
    init_checkpoint: str | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        import yaml

        raw = yaml.safe_load(Path(path).read_text())
        return cls(**raw)

    @classmethod
    def from_saved(cls, raw: dict) -> "TrainConfig":
        """Rebuild from a checkpoint's stored config, tolerating field drift.

        `TrainConfig(**ckpt["config"])` raises on any key this class no longer has — so a
        checkpoint trained after a field was added cannot be loaded by older code at all,
        which is how the dashboard died on `variance_head`. Unknown keys are dropped with a
        warning instead. That is safe for load-bearing keys because dropping one that
        changes architecture (e.g. `variance_head`) makes `load_state_dict` fail loudly
        right afterwards; it is silence about *cosmetic* keys that we actually want.
        """
        import warnings

        known = {f.name for f in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            warnings.warn(
                f"checkpoint config has {len(unknown)} field(s) this build does not know "
                f"({', '.join(unknown)}); ignoring them. If the checkpoint is newer than "
                f"the code, pull the latest src/faceiq_pref before trusting results.",
                stacklevel=2,
            )
        return cls(**{k: v for k, v in raw.items() if k in known})


class PairDataset(Dataset):
    def __init__(
        self,
        matchups: list[Matchup],
        faces: dict[str, Face],
        export: Export,
        backbone: str,
        image_size: int,
        augment: bool,
        targets: dict[int, float] | None = None,
        target_weight: float = 1.0,
        heavy_indices: set[int] | None = None,
    ):
        self.rows = matchups
        self.faces = faces
        self.export = export
        self.tf = build_transforms(backbone, image_size, augment)
        self.targets = targets or {}
        self.target_weight = target_weight
        # Only these pair_index values get `target_weight` (panel pairs). BT-distill
        # retargets must not inherit the panel upweight.
        self.heavy_indices = heavy_indices or set()

    def __len__(self) -> int:
        return len(self.rows)

    def _load(self, face_id: str) -> torch.Tensor:
        path = self.export.image_path(self.faces[face_id])
        with Image.open(path) as img:
            return self.tf(img.convert("RGB"))

    def __getitem__(self, idx: int):
        m = self.rows[idx]
        # A human vote share, where the panel covered this pair, beats the export's winner:
        # it says which face and by how much. Everything else keeps the export label
        # (or a BT soft target written into `targets` by the distill path).
        label = self.targets.get(m.pair_index)
        weight = self.target_weight if m.pair_index in self.heavy_indices else 1.0
        if label is None:
            label = 1.0 if m.final_outcome == "A" else 0.0 if m.final_outcome == "B" else 0.5
        return (
            self._load(m.face_a_id),
            self._load(m.face_b_id),
            torch.tensor(float(label)),
            torch.tensor(float(weight)),
        )


def load_bt_soft_targets(
    rows: list[Matchup],
    ratings_csv: str | Path,
    temperature: float = 1.0,
) -> dict[int, float]:
    """P(A wins) under the BT teacher: sigmoid((θ_a - θ_b) / T)."""
    import math

    import pandas as pd

    if temperature <= 0:
        raise ValueError("bt_distill_temperature must be > 0")
    rank = pd.read_csv(ratings_csv)
    theta = dict(zip(rank["faceId"], rank["theta"]))
    out: dict[int, float] = {}
    for m in rows:
        ta, tb = theta.get(m.face_a_id), theta.get(m.face_b_id)
        if ta is None or tb is None:
            continue
        out[m.pair_index] = 1.0 / (1.0 + math.exp(-(ta - tb) / temperature))
    return out


def _apply_confidence_filter(rows: list[Matchup], floor: str) -> list[Matchup]:
    """Keep human-audited pairs; for VLM pairs require confidence >= floor."""
    order = {"low": 0, "medium": 1, "high": 2}
    min_level = order[floor]
    return [
        m
        for m in rows
        if m.human_labeled_at is not None
        or (m.confidence is not None and order.get(m.confidence, 0) >= min_level)
    ]


def _filter_rows(rows: list[Matchup], faces: dict[str, Face], export: Export, cfg: TrainConfig):
    kept = []
    for m in rows:
        if cfg.skip_ties and m.is_tie:
            continue
        pa = export.image_path(faces[m.face_a_id])
        pb = export.image_path(faces[m.face_b_id])
        if pa.exists() and pb.exists():
            kept.append(m)
    return kept


@torch.no_grad()
def evaluate(model: PairwiseModel, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss, correct, seen = 0.0, 0, 0
    # Validation is deliberately unweighted: val keeps export labels, and weighting it too
    # would make val_loss incomparable across runs for no benefit.
    for a, b, y, _w in loader:
        a, b, y = a.to(device), b.to(device), y.to(device)
        logits = model(a, b)
        total_loss += criterion(logits, y).item() * len(y)
        correct += ((logits > 0) == (y > 0.5)).sum().item()
        seen += len(y)
    return {"loss": total_loss / max(seen, 1), "accuracy": correct / max(seen, 1)}


@torch.no_grad()
def panel_agreement(
    model: PairwiseModel,
    loader: DataLoader,
    device: torch.device,
    rows: list[Matchup],
    votes: dict[int, tuple[float, float]],
) -> float:
    """Share of individual human votes agreeing with the model's pick, on val pairs.

    The same quantity `scripts/eval_vs_panel.py` reports, computed inline so it can drive
    checkpoint selection. Vote-weighted rather than pair-weighted on purpose: a pair the crowd
    split 7-5 should contribute less to the score than one it called 11-1, and weighting by
    ballots does that automatically.

    `loader` must be unshuffled over `rows` so batch order lines up with the vote table. The
    sign of the pairwise logit is the model's preference for face A under both the plain and
    the variance head, so no separate scoring pass is needed.
    """
    model.eval()
    agree = total = 0.0
    i = 0
    for a, b, _y, _w in loader:
        picks_a = (model(a.to(device), b.to(device)) > 0).tolist()
        for prefers_a in picks_a:
            wa, wb = votes[rows[i].pair_index]
            agree += wa if prefers_a else wb
            total += wa + wb
            i += 1
    return agree / total if total else float("nan")


def train(export: Export, cfg: TrainConfig) -> dict:
    device = pick_device()
    faces = export.faces()
    matchups = export.all_matchups()
    if cfg.max_pairs:
        matchups = matchups[: cfg.max_pairs]

    train_rows, val_rows = split_by_face_id(matchups, cfg.val_fraction, cfg.split_seed)
    if cfg.confidence_filter:
        before = len(train_rows)
        train_rows = _apply_confidence_filter(train_rows, cfg.confidence_filter)
        print(
            f"confidence filter >= {cfg.confidence_filter} (train only): "
            f"{before} -> {len(train_rows)} pairs"
        )
    train_rows = _filter_rows(train_rows, faces, export, cfg)
    val_rows = _filter_rows(val_rows, faces, export, cfg)

    # Panel / BT-distill targets go on the train split only. Rewriting val labels too would
    # move the yardstick and the model in the same experiment, leaving val accuracy
    # uninterpretable; `scripts/eval_vs_panel.py` / leakage-split majority is how a checkpoint
    # gets measured against humans.
    targets: dict[int, float] = {}
    panel_covered: set[int] = set()
    if cfg.panel_labels:
        panels = [(r, m) for r, m in cfg.panel_labels]
        targets = load_panel_targets(
            panels,
            load_rejects(cfg.panel_rejects),
            min_votes=cfg.panel_min_votes,
            prior=cfg.panel_prior,
            hard=cfg.panel_hard,
        )
        panel_covered = set(targets)
        hit = [m for m in train_rows if m.pair_index in targets]
        soft = sum(1 for m in hit if 0.35 < targets[m.pair_index] < 0.65)
        flipped = sum(
            1
            for m in hit
            if m.final_outcome in ("A", "B")
            and (targets[m.pair_index] > 0.5) != (m.final_outcome == "A")
        )
        print(
            f"panel targets: {len(targets):,} pairs have >= {cfg.panel_min_votes} votes; "
            f"{len(hit):,} of {len(train_rows):,} train pairs ({len(hit) / len(train_rows):.1%}) "
            f"take one"
        )
        print(
            f"  of those, {soft:,} land near 0.5 (the crowd was split) and "
            f"{flipped:,} disagree with the export winner"
        )
        val_hit = sum(1 for m in val_rows if m.pair_index in targets)
        print(f"  val split keeps export labels throughout ({val_hit:,} panel pairs untouched)")

    if cfg.bt_distill_weight > 0:
        if not cfg.bt_ratings:
            raise ValueError("bt_distill_weight > 0 requires bt_ratings")
        w = cfg.bt_distill_weight
        if not 0.0 < w <= 1.0:
            raise ValueError("bt_distill_weight must be in (0, 1]")
        bt_soft = load_bt_soft_targets(
            train_rows, cfg.bt_ratings, cfg.bt_distill_temperature
        )
        n_mixed = n_pure = n_skip = 0
        for m in train_rows:
            if m.pair_index in panel_covered:
                continue  # humans beat the teacher
            soft = bt_soft.get(m.pair_index)
            if soft is None:
                n_skip += 1
                continue
            if m.pair_index in targets:
                base = targets[m.pair_index]
            elif m.final_outcome == "A":
                base = 1.0
            elif m.final_outcome == "B":
                base = 0.0
            else:
                base = 0.5
            targets[m.pair_index] = (1.0 - w) * base + w * soft
            n_pure += int(w >= 1.0 - 1e-12)
            n_mixed += int(w < 1.0 - 1e-12)
        print(
            f"BT distill: teacher={cfg.bt_ratings} T={cfg.bt_distill_temperature} w={w} "
            f"-> {n_pure + n_mixed:,} train pairs retargeted "
            f"({n_pure:,} pure soft, {n_mixed:,} mixed), "
            f"{len(panel_covered):,} panel pairs kept human, {n_skip:,} missing θ"
        )

    # Human-grounded validation. These are val-split pairs, so no face here was trained on and
    # no vote here entered the loss — the same leak guard `eval_vs_panel.py` applies, just
    # computed every epoch so it can pick the checkpoint.
    panel_val_rows: list[Matchup] = []
    panel_votes: dict[int, tuple[float, float]] = {}
    if cfg.panel_labels:
        panel_votes = load_panel_votes(
            [(r, m) for r, m in cfg.panel_labels], load_rejects(cfg.panel_rejects)
        )
        panel_val_rows = [
            m for m in val_rows
            if sum(panel_votes.get(m.pair_index, (0.0, 0.0))) >= cfg.panel_min_votes
        ]
        n_votes = sum(sum(panel_votes[m.pair_index]) for m in panel_val_rows)
        selecting = len(panel_val_rows) >= MIN_PANEL_VAL_PAIRS
        print(
            f"  human-grounded val: {len(panel_val_rows):,} leak-free panel pairs "
            f"({n_votes:,.0f} votes) -> "
            + ("best.pt is selected on THIS, not val_accuracy"
               if selecting else
               f"too few (< {MIN_PANEL_VAL_PAIRS}); falling back to val_accuracy")
        )

    train_ds = PairDataset(
        train_rows, faces, export, cfg.backbone, cfg.image_size, cfg.augment,
        targets=targets, target_weight=cfg.panel_weight, heavy_indices=panel_covered,
    )
    val_ds = PairDataset(val_rows, faces, export, cfg.backbone, cfg.image_size, augment=False)
    train_dl = DataLoader(
        train_ds, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True
    )
    val_dl = DataLoader(val_ds, cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True)
    panel_dl = None
    if panel_val_rows:
        panel_ds = PairDataset(
            panel_val_rows, faces, export, cfg.backbone, cfg.image_size, augment=False
        )
        # shuffle=False is load-bearing: panel_agreement() walks batches in row order.
        panel_dl = DataLoader(
            panel_ds, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
            pin_memory=True,
        )

    model = PairwiseModel(cfg.backbone, variance_head=cfg.variance_head).to(device)
    if cfg.init_checkpoint:
        ckpt = torch.load(cfg.init_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(
            f"warm-start from {cfg.init_checkpoint} "
            f"(epoch {ckpt.get('epoch', '?')}, selected_on={ckpt.get('selected_on')})"
        )
    if cfg.freeze_backbone:
        for p in model.scorer.backbone.parameters():
            p.requires_grad = False
        model.scorer.backbone.eval()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    # reduction="none" so per-sample panel weights can be applied; the weighted mean below
    # keeps train_loss on the same scale as an unweighted run.
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    ckpt_dir = Path("checkpoints") / cfg.run_name
    art_dir = Path("artifacts") / cfg.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)

    # Which number decides `best.pt`. `val_accuracy` is scored against Gemini's labels, so on a
    # panel run it cannot see the improvement being trained for; prefer the human-grounded metric
    # whenever there are enough leak-free pairs to trust it.
    select_on = ("panel_val_accuracy"
                 if panel_dl is not None and len(panel_val_rows) >= MIN_PANEL_VAL_PAIRS
                 else "val_accuracy")

    history, best_score, best_epoch = [], -1.0, 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        if cfg.freeze_backbone:
            model.scorer.backbone.eval()
        running, seen = 0.0, 0
        for a, b, y, w in tqdm(train_dl, desc=f"epoch {epoch}/{cfg.epochs}"):
            a, b, y, w = a.to(device), b.to(device), y.to(device), w.to(device)
            optimizer.zero_grad()
            loss = (criterion(model(a, b), y) * w).sum() / w.sum()
            loss.backward()
            optimizer.step()
            running += loss.item() * len(y)
            seen += len(y)

        val = evaluate(model, val_dl, device)
        row = {
            "epoch": epoch,
            "train_loss": running / max(seen, 1),
            "val_loss": val["loss"],
            "val_accuracy": val["accuracy"],
            "time": time.time(),
        }
        if panel_dl is not None:
            row["panel_val_accuracy"] = panel_agreement(
                model, panel_dl, device, panel_val_rows, panel_votes
            )
        history.append(row)
        line = (f"epoch {epoch}: train_loss={row['train_loss']:.4f} "
                f"val_loss={val['loss']:.4f} val_acc={val['accuracy']:.4f}")
        if "panel_val_accuracy" in row:
            line += f" panel_val_acc={row['panel_val_accuracy']:.4f}"
        print(line + (f"  <- best on {select_on}" if row[select_on] > best_score else ""))

        if row[select_on] > best_score:
            best_score, best_epoch = row[select_on], epoch
            torch.save(
                {"model": model.state_dict(), "config": asdict(cfg), "epoch": epoch,
                 "selected_on": select_on, "selected_score": best_score},
                ckpt_dir / "best.pt",
            )

        metrics = {
            "config": asdict(cfg),
            "device": str(device),
            "train_pairs": len(train_rows),
            "val_pairs": len(val_rows),
            "panel_target_pairs": sum(1 for m in train_rows if m.pair_index in targets),
            "panel_val_pairs": len(panel_val_rows),
            "panel_weight": cfg.panel_weight,
            "selected_on": select_on,
            "selected_epoch": best_epoch,
            "best_val_accuracy": max(r["val_accuracy"] for r in history),
            "best_panel_val_accuracy": max(
                (r["panel_val_accuracy"] for r in history if "panel_val_accuracy" in r),
                default=None,
            ),
            "history": history,
        }
        (art_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"checkpoint: epoch {best_epoch}, selected on {select_on} = {best_score:.4f}")
    return metrics
