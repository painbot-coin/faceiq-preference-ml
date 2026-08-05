"""Remove repeat identities from a validation set, and catch cohort faces re-uploaded.

Two contamination routes survive the id-level exclusion list, and both need to compare
*faces* rather than trust ids:

**A cohort user re-uploading.** The exclusion list Harsh applied is per face id, and the
export carries no user id, so a person already in the 3,000-face cohort who uploads a new
photo today passes every id check. Their face is in BT, in the comparator's training set,
and possibly in the reference set. Scoring them as "unseen" is the single most inflating
thing that could happen to this study. Here we embed both sides and drop any match.

**The same face twice inside the set.** ~1.5 rows share an account. Deduping on the account
is wrong — 20 accounts carry both male and female photos, so people are scoring friends —
but the repeats are real and a pair of the same person is a coin flip with no right answer.

Identity comes from ArcFace R50 (`buffalo_l w600k`), which is a face *recognition* model and
so measures who someone is, not what they look like. The comparator would be the wrong tool:
it is trained to rank attractiveness, so it maps two different but similar-looking people to
near-identical scores.

**On the threshold.** The first version of this script calibrated the cut off the largest
similarity between two cohort faces, on the assumption that the cohort is 3,000 distinct
people. It is not — it contains ~360 people photographed more than once, including 35
groups of byte-identical files, so that "impostor maximum" was 1.000 and the resulting cut
of 1.05 could never fire. The cut is now a fixed 0.50, which sits in the empirical valley
between the impostor bulk (99.9th percentile 0.424) and the genuine-match mode, and which
matches the usual operating point for this model. Pairs in the grey zone below it are
printed rather than silently dropped, so a borderline call is a human's to make.

Usage:
    python scripts/audit_validation_identities.py --set set-1-female --set set-1-male
    python scripts/audit_validation_identities.py --set set-1-female --apply
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from faceiq_pref.backbones.arcface import load_arcface_r50
from faceiq_pref.preprocess import normalize_front_photo

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "artifacts" / ".cache"
SIZE = 112


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def embed(paths: list[Path], net: torch.nn.Module, dev: torch.device,
          normalize: bool, batch: int = 32) -> np.ndarray:
    """L2-normalised ArcFace embeddings, so a dot product is a cosine similarity."""
    out: list[np.ndarray] = []
    for i in range(0, len(paths), batch):
        tensors = []
        for p in paths[i: i + batch]:
            img = Image.open(p).convert("RGB")
            if normalize:
                img = normalize_front_photo(img) or img
            arr = np.asarray(img.resize((SIZE, SIZE)), dtype=np.float32) / 255.0
            tensors.append(torch.from_numpy(arr).permute(2, 0, 1))
        x = (torch.stack(tensors) - 0.5) / 0.5
        x = x[:, [2, 1, 0], :, :].to(dev)  # ArcFace wants BGR
        with torch.no_grad():
            e = net(x).float().cpu().numpy()
        out.append(e / np.linalg.norm(e, axis=1, keepdims=True))
        print(f"\r  embedded {min(i + batch, len(paths))}/{len(paths)}", end="", flush=True)
    print()
    return np.concatenate(out) if out else np.zeros((0, 512), dtype=np.float32)


def cohort_embeddings(net: torch.nn.Module, dev: torch.device) -> tuple[np.ndarray, list[str]]:
    cache = CACHE / "cohort-arcface.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return z["emb"], list(z["ids"])
    exp = next(p for p in sorted((ROOT / "data" / "exports").glob("*"))
               if (p / "faces.jsonl").exists())
    rows = [json.loads(line) for line in (exp / "faces.jsonl").read_text().splitlines()]
    paths, ids = [], []
    for r in rows:
        p = exp / r["imagePath"]
        if p.exists():
            paths.append(p)
            ids.append(r["sourceFaceId"])
    print(f"embedding {len(paths)} cohort faces (one-off, cached)")
    emb = embed(paths, net, dev, normalize=False)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, emb=emb, ids=np.array(ids))
    return emb, ids


def impostor_quantiles(emb: np.ndarray, seed: int = 0) -> tuple[float, float]:
    """(p99, p99.9) of cohort-cohort similarity — context for whether the cut is sane.

    Not a maximum: the cohort's own duplicates put genuine matches in this distribution, but
    they are far too few (~1.5k of 4.5M pairs) to move a 99.9th percentile.
    """
    rng = np.random.default_rng(seed)
    n = len(emb)
    i, j = rng.integers(0, n, 400_000), rng.integers(0, n, 400_000)
    keep = i != j
    sims = np.einsum("ij,ij->i", emb[i[keep]], emb[j[keep]])
    return float(np.percentile(sims, 99)), float(np.percentile(sims, 99.9))


def greedy_clusters(sim: np.ndarray, thr: float) -> list[int]:
    """Assign each row a cluster id; anything above `thr` joins an existing cluster."""
    label = [-1] * len(sim)
    nxt = 0
    for k in range(len(sim)):
        for prev in range(k):
            if sim[k, prev] >= thr:
                label[k] = label[prev]
                break
        if label[k] == -1:
            label[k] = nxt
            nxt += 1
    return label


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="sets", action="append", required=True)
    ap.add_argument("--threshold", type=float, default=0.50,
                    help="cosine similarity at or above which two photos are the same person")
    ap.add_argument("--grey", type=float, default=0.10,
                    help="print, but do not reject, matches this far below the threshold")
    ap.add_argument("--apply", action="store_true",
                    help="actually move rejects out of photos/ (default is a dry run)")
    a = ap.parse_args()

    dev = device()
    print(f"loading ArcFace R50 on {dev}")
    net = load_arcface_r50().to(dev).eval()

    cohort, cohort_ids = cohort_embeddings(net, dev)
    p99, p999 = impostor_quantiles(cohort)
    thr = a.threshold
    print(f"\nsame-person cut {thr:.2f}; for scale, two random cohort faces reach {p99:.3f} at "
          f"the 99th percentile and {p999:.3f} at the 99.9th\n")

    for name in a.sets:
        photos_dir = ROOT / "data" / "validation" / name / "photos"
        if not photos_dir.exists():
            print(f"{name}: no photos/ directory, skipping")
            continue
        paths = sorted(photos_dir.glob("*.webp"))
        print(f"{name}: {len(paths)} photos")
        emb = embed(paths, net, dev, normalize=True)

        cross = emb @ cohort.T
        best = cross.max(axis=1)
        leaked = [(paths[i].name, cohort_ids[int(cross[i].argmax())], float(best[i]))
                  for i in np.where(best >= thr)[0]]

        keep_idx = [i for i in range(len(paths)) if best[i] < thr]
        sub = emb[keep_idx]
        labels = greedy_clusters(sub @ sub.T, thr)
        seen: set[int] = set()
        reps, dupes = [], []
        for pos, lab in enumerate(labels):
            (reps if lab not in seen else dupes).append(keep_idx[pos])
            seen.add(lab)

        grey = [(paths[i].name, cohort_ids[int(cross[i].argmax())], float(best[i]))
                for i in np.where((best >= thr - a.grey) & (best < thr))[0]]

        print(f"  cohort re-uploads found : {len(leaked)}")
        for fn, cid, s in leaked[:5]:
            print(f"      {fn}  ~  cohort {cid}  (cos {s:.3f})")
        print(f"  repeat identities       : {len(dupes)}")
        print(f"  distinct faces kept     : {len(reps)}")
        if grey:
            print(f"  grey zone, kept but worth an eyeball ({thr - a.grey:.2f}-{thr:.2f}): "
                  f"{len(grey)}")
            for fn, cid, s in grey[:5]:
                print(f"      {fn}  ~  cohort {cid}  (cos {s:.3f})")

        report = {
            "set": name, "threshold": thr,
            "cohortImpostorP99": p99, "cohortImpostorP999": p999,
            "photos": len(paths), "cohortReuploads": [
                {"photo": f, "cohortSourceFaceId": c, "cosine": s} for f, c, s in leaked],
            "greyZone": [{"photo": f, "cohortSourceFaceId": c, "cosine": s}
                         for f, c, s in grey],
            "repeatIdentities": [paths[i].name for i in dupes],
            "kept": [paths[i].name for i in reps],
        }
        out = ROOT / "data" / "validation" / name / "identity-audit.json"
        out.write_text(json.dumps(report, indent=2))

        if a.apply:
            for reason, idxs in (("cohort-match", [paths.index(photos_dir / f)
                                                   for f, _, _ in leaked]),
                                 ("repeat-identity", dupes)):
                dest = ROOT / "data" / "validation" / name / "rejected" / reason
                dest.mkdir(parents=True, exist_ok=True)
                for i in idxs:
                    if paths[i].exists():
                        shutil.move(str(paths[i]), dest / paths[i].name)
            print(f"  -> moved rejects out; {len(list(photos_dir.glob('*.webp')))} remain")
            print(f"  next: python scripts/validate_placement.py make-pairs --set {name} "
                  f"--pairs 300 --repeat 0.1")
        else:
            print("  (dry run — rerun with --apply to move rejects out)")
        print(f"  report -> {out.relative_to(ROOT)}\n")


if __name__ == "__main__":
    main()
