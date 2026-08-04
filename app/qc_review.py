"""Human triage of the Stage 0 VLM face-QC flags.

Run:  streamlit run app/qc_review.py --server.port 8503

The VLM only *proposes* — a human confirms every flag before any face leaves the
cohort, because the expensive mistakes here are asymmetric: keeping a minor in a
paid rater study is unacceptable, while dropping a healthy face costs ~35 edges
out of 52k.

Decisions append to <qc>/decisions.jsonl (resumable, one line per face). When
you're done, "Write handoff files" emits the two lists faceiq-labs needs:

    exclude-faces.csv   faceId, labsReason   -> drop all their comparisons
    gender-fixes.csv    faceId, correctGender -> update VlmPilotFace.gender
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
QC_DIR = ROOT / "artifacts" / "face-qc-v1"
EXPORT_DIR = ROOT / "data" / "exports" / "cmr1mr0m7000196d57zi3vcgn"

# Flag reason -> the exclusion reason faceiq-labs understands
# (GT_EXPORT_EXCLUDE_REASONS: synthetic | quality | duplicate | other).
LABS_REASON = {
    "not_real_photo": "synthetic",
    "screenshot": "quality",
    "multiple_faces": "quality",
    "possible_minor": "other",
    "gender_mismatch": "other",
    "gender_ambiguous": "other",
}


@st.cache_data
def load_flags() -> list[dict]:
    with (QC_DIR / "flags.csv").open() as fh:
        return list(csv.DictReader(fh))


def load_decisions() -> dict[str, dict]:
    path = QC_DIR / "decisions.jsonl"
    if not path.exists():
        return {}
    decided: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            decided[row["faceId"]] = row
    return decided


def record(face_id: str, action: str, reason: str) -> None:
    with (QC_DIR / "decisions.jsonl").open("a") as fh:
        fh.write(json.dumps({"faceId": face_id, "action": action, "flagReason": reason}) + "\n")


def photo_source(face_id: str, url: str) -> str:
    local = EXPORT_DIR / "images" / f"{face_id}.webp"
    return str(local) if local.exists() else url


def write_handoff(decisions: dict[str, dict]) -> tuple[int, int]:
    excludes = [d for d in decisions.values() if d["action"] == "exclude"]
    fixes = [d for d in decisions.values() if d["action"].startswith("relabel:")]

    with (QC_DIR / "exclude-faces.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["faceId", "labsReason", "flagReason"])
        for d in excludes:
            w.writerow([d["faceId"], LABS_REASON.get(d["flagReason"], "other"), d["flagReason"]])

    with (QC_DIR / "gender-fixes.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["faceId", "correctGender"])
        for d in fixes:
            w.writerow([d["faceId"], d["action"].split(":", 1)[1]])

    return len(excludes), len(fixes)


def main() -> None:
    st.set_page_config(page_title="Face QC review", layout="wide")
    st.title("Stage 0 — face QC flag review")

    if not (QC_DIR / "flags.csv").exists():
        st.error(f"`{QC_DIR.relative_to(ROOT)}/flags.csv` not found — run scripts/qc_faces.py first.")
        return

    flags = load_flags()
    decisions = load_decisions()

    reasons = sorted({r for f in flags for r in f["reasons"].split(";")})
    with st.sidebar:
        st.metric("Reviewed", f"{len(decisions)} / {len(flags)}")
        chosen = st.selectbox("Flag group", reasons)
        per_page = st.selectbox("Faces per page", [6, 12, 24], index=1)
        hide_done = st.checkbox("Hide reviewed", value=True)
        st.divider()
        if st.button("Write handoff files", type="primary"):
            n_ex, n_fix = write_handoff(decisions)
            st.success(f"{n_ex} exclusions, {n_fix} gender fixes written to {QC_DIR.name}/")
        st.caption(
            "Guidance: `possible_minor` errs young on purpose — exclude only if "
            "you would not show this photo to a paid rater. Photo quality "
            "(lighting, crop, resolution) is **not** grounds for exclusion."
        )

    pool = [f for f in flags if chosen in f["reasons"].split(";")]
    if hide_done:
        pool = [f for f in pool if f["faceId"] not in decisions]

    st.caption(
        f"{chosen}: {len([f for f in flags if chosen in f['reasons'].split(';')])} flagged, "
        f"{len(pool)} shown"
    )
    if not pool:
        st.success(f"Nothing left to review in `{chosen}`.")
        return

    page = pool[:per_page]
    cols = st.columns(3, gap="medium")
    for i, face in enumerate(page):
        fid = face["faceId"]
        with cols[i % 3]:
            st.image(photo_source(fid, face["photoUrl"]), use_container_width=True)
            st.caption(
                f"label **{face['exportGender']}** · VLM **{face['apparentGender']}** "
                f"({face['genderConfidence']}) · age {face['apparentAgeBand']} · "
                f"{face['issues'] or 'no issues'}"
            )
            b1, b2, b3, b4 = st.columns(4)
            if b1.button("Keep", key=f"k-{fid}"):
                record(fid, "keep", chosen)
                st.rerun()
            if b2.button("Exclude", key=f"x-{fid}"):
                record(fid, "exclude", chosen)
                st.rerun()
            if b3.button("→M", key=f"m-{fid}", help="Relabel as male (keeps face, drops its edges)"):
                record(fid, "relabel:male", chosen)
                st.rerun()
            if b4.button("→F", key=f"f-{fid}", help="Relabel as female (keeps face, drops its edges)"):
                record(fid, "relabel:female", chosen)
                st.rerun()
            st.caption(f"`{fid}`")


if __name__ == "__main__":
    main()
