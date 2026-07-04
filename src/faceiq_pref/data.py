"""Load and validate faceiq-labs GT exports.

Single entry point for reading an export directory produced by
`faceiq-labs/scripts/export-gt-run.ts`. Verifies manifest integrity (row counts +
sha256 per shard) before handing data to BT / training / dashboard, so every consumer
interprets the export identically.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

Outcome = Literal["A", "B", "tie"]

FINALIZE_RULE = "human label when humanLabeledAt is set, else VLM label"


@dataclass(frozen=True)
class Face:
    face_id: str
    source_face_id: str
    gender: str
    decile_bin: int | None
    labs_overall_score: float | None
    photo_url: str
    image_path: str  # relative to export dir, e.g. "images/<faceId>.webp"


@dataclass(frozen=True)
class Matchup:
    pair_index: int
    comparison_id: str
    face_a_id: str
    face_b_id: str
    gender: str
    face_a_decile: int | None
    face_b_decile: int | None
    vlm_outcome: Outcome | None
    vlm_winner_face_id: str | None
    confidence: str | None
    human_outcome: Outcome | None
    human_winner_face_id: str | None
    human_labeled_at: str | None
    is_human_override: bool
    final_outcome: Outcome
    final_winner_face_id: str | None

    @property
    def is_tie(self) -> bool:
        return self.final_outcome == "tie"


class ExportError(RuntimeError):
    """Export is missing, corrupt, or inconsistent with its manifest."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Export:
    """A verified faceiq-labs GT export."""

    def __init__(self, export_dir: Path, manifest: dict):
        self.dir = export_dir
        self.manifest = manifest
        self.run_id: str = manifest["runId"]

    # -- loading ------------------------------------------------------------

    def faces(self) -> dict[str, Face]:
        faces: dict[str, Face] = {}
        with (self.dir / "faces.jsonl").open() as f:
            for line in f:
                row = json.loads(line)
                face = Face(
                    face_id=row["faceId"],
                    source_face_id=row["sourceFaceId"],
                    gender=row["gender"],
                    decile_bin=row.get("decileBin"),
                    labs_overall_score=row.get("labsOverallScore"),
                    photo_url=row["photoUrl"],
                    image_path=row["imagePath"],
                )
                faces[face.face_id] = face
        expected = self.manifest["counts"]["faces"]
        if len(faces) != expected:
            raise ExportError(f"faces.jsonl has {len(faces)} rows, manifest says {expected}")
        return faces

    def matchups(self) -> Iterator[Matchup]:
        """Stream all matchups in pairIndex order."""
        for shard in self.manifest["matchupShards"]:
            path = self.dir / shard["path"]
            with path.open() as f:
                for line in f:
                    row = json.loads(line)
                    m = Matchup(
                        pair_index=row["pairIndex"],
                        comparison_id=row["comparisonId"],
                        face_a_id=row["faceAId"],
                        face_b_id=row["faceBId"],
                        gender=row["gender"],
                        face_a_decile=row.get("faceADecile"),
                        face_b_decile=row.get("faceBDecile"),
                        vlm_outcome=row.get("vlmOutcome"),
                        vlm_winner_face_id=row.get("vlmWinnerFaceId"),
                        confidence=row.get("confidence"),
                        human_outcome=row.get("humanOutcome"),
                        human_winner_face_id=row.get("humanWinnerFaceId"),
                        human_labeled_at=row.get("humanLabeledAt"),
                        is_human_override=row.get("isHumanOverride", False),
                        final_outcome=row["finalOutcome"],
                        final_winner_face_id=row.get("finalWinnerFaceId"),
                    )
                    _assert_finalize_rule(m)
                    yield m

    def all_matchups(self) -> list[Matchup]:
        rows = list(self.matchups())
        expected = self.manifest["counts"]["matchups"]
        if len(rows) != expected:
            raise ExportError(f"read {len(rows)} matchups, manifest says {expected}")
        return rows

    def image_path(self, face: Face) -> Path:
        return self.dir / face.image_path

    # -- integrity ----------------------------------------------------------

    def verify_shard_hashes(self) -> None:
        for shard in self.manifest["matchupShards"]:
            path = self.dir / shard["path"]
            if not path.exists():
                raise ExportError(f"missing shard {shard['path']}")
            actual = _sha256(path)
            if actual != shard["sha256"]:
                raise ExportError(
                    f"shard {shard['path']} hash mismatch: {actual} != {shard['sha256']}"
                )

    def image_coverage(self, faces: dict[str, Face] | None = None) -> tuple[int, list[str]]:
        """Return (present_count, missing_face_ids)."""
        faces = faces or self.faces()
        missing = [
            f.face_id
            for f in faces.values()
            if not (self.dir / f.image_path).exists()
            or (self.dir / f.image_path).stat().st_size == 0
        ]
        return len(faces) - len(missing), missing


def _assert_finalize_rule(m: Matchup) -> None:
    if m.human_labeled_at is not None:
        expected, expected_winner = m.human_outcome, m.human_winner_face_id
    else:
        expected, expected_winner = m.vlm_outcome, m.vlm_winner_face_id
    if m.final_outcome != expected or m.final_winner_face_id != expected_winner:
        raise ExportError(
            f"pair {m.pair_index}: finalOutcome inconsistent with finalize rule "
            f"({FINALIZE_RULE})"
        )


def load_export(export_dir: str | Path, verify_hashes: bool = True) -> Export:
    """Open an export directory and verify its manifest. Fails loudly on corruption."""
    export_dir = Path(export_dir).expanduser().resolve()
    manifest_path = export_dir / "manifest.json"
    if not manifest_path.exists():
        raise ExportError(
            f"no manifest.json in {export_dir} — export incomplete or wrong path"
        )
    manifest = json.loads(manifest_path.read_text())
    export = Export(export_dir, manifest)
    if verify_hashes:
        export.verify_shard_hashes()
    return export


# -- train/val split ---------------------------------------------------------


def split_by_face_id(
    matchups: list[Matchup],
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[Matchup], list[Matchup]]:
    """Split matchups by FACE id (leakage-safe), not by pair.

    A pair is val only if BOTH faces are val faces. Pairs mixing train and val
    faces are dropped (logged via returned counts implicitly).
    """
    face_ids = sorted({fid for m in matchups for fid in (m.face_a_id, m.face_b_id)})
    rng = random.Random(seed)
    rng.shuffle(face_ids)
    n_val = int(len(face_ids) * val_fraction)
    val_faces = set(face_ids[:n_val])

    train, val = [], []
    for m in matchups:
        a_val, b_val = m.face_a_id in val_faces, m.face_b_id in val_faces
        if not a_val and not b_val:
            train.append(m)
        elif a_val and b_val:
            val.append(m)
        # mixed pairs dropped
    return train, val
