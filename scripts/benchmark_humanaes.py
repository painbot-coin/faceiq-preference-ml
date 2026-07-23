#!/usr/bin/env python
"""HumanAesExpert-1B (KlingTeam, MIT) as an external zero-shot cross-check.

Scores every export face with the VLM's fast regression path (`model.score`,
LM-head token probabilities mapped to a scalar), then reports the same metrics as
external_crosscheck.py / evaluate.py so numbers are directly comparable:
  - held-out pairwise accuracy on the shared val split (val_fraction 0.2, seed 42)
  - Kendall tau / Spearman rho vs BT theta, overall and per gender

This is a diagnostic external-validity check, NOT an upper bound: the model rates
whole-human-image aesthetics (12-dim standard incl. outfit/environment), not facial
attractiveness, and has never seen our distribution.

Per-face scores are cached incrementally to artifacts/external-humanaes-1b/
scores_cache.jsonl, so an interrupted run resumes where it left off.

Usage:
    python scripts/benchmark_humanaes.py \
        --export data/exports/cmr1mr0m7000196d57zi3vcgn \
        --ratings artifacts/bt-refit-v1/ratings.csv [--limit 20]

Requires: transformers==4.44.2 (per the model card), einops, timm, sentencepiece.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

from faceiq_pref.data import load_export, split_by_face_id
from faceiq_pref.eval import rank_agreement_vs_bt
from faceiq_pref.model import pick_device

MODEL_ID = "KlingTeam/HumanAesExpert-1B"
QUESTION = "<image>\nRate the aesthetics of this human picture."
OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "external-humanaes-1b"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ------------------------------------------------- image preprocessing (model card)


def build_transform(input_size: int) -> T.Compose:
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        },
        key=lambda x: x[0] * x[1],
    )
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_image(image_file: Path, input_size=448, max_num=12) -> torch.Tensor:
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    return torch.stack([transform(img) for img in images])


# ------------------------------------------------------------------------- scoring


def load_model(device: torch.device):
    from transformers import AutoModel, AutoTokenizer

    dtype = torch.float16 if device.type in ("cuda", "mps") else torch.float32
    model = (
        AutoModel.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            use_flash_attn=False,
            trust_remote_code=True,
        )
        .eval()
        .to(device)
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=False)
    return model, tokenizer, dtype


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export", required=True)
    ap.add_argument("--ratings", help="BT ratings.csv for rank agreement")
    ap.add_argument("--limit", type=int, help="score only the first N faces (smoke test)")
    ap.add_argument("--max-num", type=int, default=12, help="max dynamic tiles per image")
    args = ap.parse_args()

    export = load_export(args.export)
    print(f"export OK: run {export.run_id} — external model {MODEL_ID}")

    device = pick_device()
    model, tokenizer, dtype = load_model(device)
    print(f"model loaded on {device} ({dtype})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = OUT_DIR / "scores_cache.jsonl"
    scores: dict[str, float] = {}
    if cache_path.exists():
        with cache_path.open() as f:
            for line in f:
                row = json.loads(line)
                scores[row["faceId"]] = row["score"]
        print(f"resumed {len(scores)} cached scores from {cache_path}")

    faces = [f for f in export.faces().values() if export.image_path(f).exists()]
    if args.limit:
        faces = faces[: args.limit]
    todo = [f for f in faces if f.face_id not in scores]

    with cache_path.open("a") as cache:
        for face in tqdm(todo, desc="scoring faces"):
            pixel_values = (
                load_image(export.image_path(face), max_num=args.max_num).to(dtype).to(device)
            )
            s = model.score(tokenizer, pixel_values, QUESTION)
            s = float(s.item() if torch.is_tensor(s) else s)
            scores[face.face_id] = s
            cache.write(json.dumps({"faceId": face.face_id, "score": s}) + "\n")
            cache.flush()

    scores = {f.face_id: scores[f.face_id] for f in faces}

    # held-out pairwise accuracy, same protocol as evaluate.py / external_crosscheck.py
    _, val_rows = split_by_face_id(export.all_matchups(), 0.2, 42)
    correct = scored = 0
    for m in val_rows:
        if m.is_tie or m.face_a_id not in scores or m.face_b_id not in scores:
            continue
        correct += (scores[m.face_a_id] > scores[m.face_b_id]) == (m.final_outcome == "A")
        scored += 1

    results: dict = {
        "model": MODEL_ID,
        "license_note": "MIT — external zero-shot cross-check, whole-human aesthetics construct",
        "question": QUESTION,
        "n_faces_scored": len(scores),
        "val_pairs_scored": scored,
        "val_accuracy": correct / max(scored, 1),
    }
    print(f"held-out pairwise accuracy: {results['val_accuracy']:.4f} ({scored} pairs)")

    face_meta = export.faces()
    if args.ratings:
        results["rank_agreement_vs_bt"] = rank_agreement_vs_bt(scores, args.ratings)
        for gender in ("female", "male"):
            sub = {
                fid: s
                for fid, s in scores.items()
                if fid in face_meta and face_meta[fid].gender == gender
            }
            if len(sub) > 10:
                results[f"rank_agreement_vs_bt_{gender}"] = rank_agreement_vs_bt(
                    sub, args.ratings
                )
        agg = results["rank_agreement_vs_bt"]
        print(
            f"rank agreement vs BT: kendall_tau={agg['kendall_tau']:.4f} "
            f"spearman_rho={agg['spearman_rho']:.4f} ({agg['n_faces']} faces)"
        )

    bt_rows: dict[str, dict] = {}
    if args.ratings:
        with open(args.ratings) as f:
            bt_rows = {row["faceId"]: row for row in csv.DictReader(f)}
    with (OUT_DIR / "model_scores.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["faceId", "gender", "theta", "modelScore"])
        for fid, s in scores.items():
            bt = bt_rows.get(fid)
            face = face_meta.get(fid)
            gender = (bt["gender"] if bt else None) or (face.gender if face else "")
            if not gender:
                continue
            writer.writerow([fid, gender, bt["theta"] if bt else "", round(s, 6)])

    (OUT_DIR / "eval.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {OUT_DIR}/model_scores.csv and eval.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
