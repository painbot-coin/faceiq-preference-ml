"""Pilot-500 two-rater pairwise labeler UI.

Rendered as a tab inside app/dashboard.py (port 8502) and also runnable standalone
via app/labeler.py. Presents the blinded pairs from labels/pilot-500/pairs.json and
appends one JSONL line per decision to labels/pilot-500/votes-<rater>.jsonl.

Shows nothing about Gemini, BT, or sampling stratum. Photos come from the local
export (data/exports/<runId>/images/) when present, else from their public blob
URLs — so a rater needs either the data zip or an internet connection, not both.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[1]
PILOT_DIR = ROOT / "labels" / "pilot-500"
PAIRS_PATH = PILOT_DIR / "pairs.json"

CHOICE_LABELS = {
    "left": "Left wins (1)",
    "tie": "Too close to call (T)",
    "right": "Right wins (2)",
}


@st.cache_data
def load_pairs() -> dict:
    return json.loads(PAIRS_PATH.read_text())


def votes_path(rater: str) -> Path:
    return PILOT_DIR / f"votes-{rater}.jsonl"


def load_votes(rater: str) -> dict[str, dict]:
    """pairId -> vote row; later lines win if a pairId somehow repeats."""
    path = votes_path(rater)
    votes: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                votes[row["pairId"]] = row
    return votes


def append_vote(rater: str, pair_id: str, choice: str) -> None:
    comment = st.session_state.get(f"comment-{pair_id}", "").strip()
    row = {
        "pairId": pair_id,
        "choice": choice,
        "comment": comment,
        "ratedAt": datetime.now(timezone.utc).isoformat(),
    }
    with votes_path(rater).open("a") as f:
        f.write(json.dumps(row) + "\n")
    if not comment:
        st.toast("Recorded without a comment — one sentence why helps the analysis.")


def undo_last(rater: str) -> None:
    path = votes_path(rater)
    if not path.exists():
        return
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    if lines:
        path.write_text("".join(ln + "\n" for ln in lines[:-1]))


def photo_source(run_id: str, side: dict) -> str:
    """Prefer the local export image (offline, faster); fall back to the blob URL."""
    local = ROOT / "data" / "exports" / run_id / "images" / f"{side['faceId']}.webp"
    return str(local) if local.exists() else side["photoUrl"]


def keyboard_shortcuts() -> None:
    """1 = left, 2 = right, T = tie.

    Ignored while typing in an input, and only clicks visible buttons — so keys
    pressed while another dashboard tab is open don't record votes.
    """
    components.html(
        """
        <script>
        const doc = window.parent.document;
        if (!window.parent.__pilotLabelerKeys) {
            window.parent.__pilotLabelerKeys = true;
            doc.addEventListener('keydown', (e) => {
                const tag = (e.target.tagName || '').toLowerCase();
                if (tag === 'input' || tag === 'textarea' || e.target.isContentEditable) return;
                const marker = {'1': '(1)', '2': '(2)', 't': '(T)'}[e.key.toLowerCase()];
                if (!marker) return;
                for (const btn of doc.querySelectorAll('button')) {
                    if (btn.innerText.includes(marker) && btn.offsetParent !== null) {
                        btn.click();
                        break;
                    }
                }
            });
        }
        </script>
        """,
        height=0,
    )


def render_labeler() -> None:
    if not PAIRS_PATH.exists():
        st.error(
            f"`{PAIRS_PATH.relative_to(ROOT)}` not found — pull the repo, or generate it with "
            "`python scripts/select_pilot_pairs.py --export data/exports/<runId>`."
        )
        return

    doc = load_pairs()
    pairs: list[dict] = doc["pairs"]

    top_left, top_right = st.columns([1, 2])
    with top_left:
        raw_name = st.text_input(
            "Your name", value=st.session_state.get("rater", ""), key="rater-input"
        )
    rater = re.sub(r"[^a-z0-9-]", "", raw_name.strip().lower())
    if not rater:
        st.info("Enter your name to start (e.g. `dit` or `alex`).")
        return
    st.session_state["rater"] = rater

    votes = load_votes(rater)
    n_done = sum(1 for p in pairs if p["pairId"] in votes)
    with top_right:
        st.progress(n_done / len(pairs), text=f"{n_done} / {len(pairs)} labeled")
        st.caption(
            f"Votes file: `labels/pilot-500/votes-{rater}.jsonl` — saves after every pair; "
            "quit and resume anytime. Shortcuts: `1` left · `2` right · `T` too close."
        )

    current = next((p for p in pairs if p["pairId"] not in votes), None)

    if current is None:
        st.success(
            f"All {len(pairs)} pairs labeled — thank you! Commit and push your votes file:\n\n"
            f"```bash\ngit add labels/pilot-500/votes-{rater}.jsonl\n"
            f'git commit -m "pilot-500: votes ({rater})"\ngit push\n```'
        )
        return

    pair_id = current["pairId"]
    st.subheader(f"Pair {n_done + 1} of {len(pairs)} — who is more attractive?")

    col_left, col_right = st.columns(2, gap="large")
    with col_left:
        st.image(photo_source(doc["runId"], current["left"]), use_container_width=True)
    with col_right:
        st.image(photo_source(doc["runId"], current["right"]), use_container_width=True)

    st.text_input(
        "Why? (one sentence)",
        key=f"comment-{pair_id}",
        placeholder="e.g. stronger jawline and clearer skin",
    )

    btn_left, btn_tie, btn_right, btn_undo = st.columns([3, 3, 3, 1])
    btn_left.button(
        CHOICE_LABELS["left"],
        use_container_width=True,
        type="primary",
        on_click=append_vote,
        args=(rater, pair_id, "left"),
    )
    btn_tie.button(
        CHOICE_LABELS["tie"],
        use_container_width=True,
        on_click=append_vote,
        args=(rater, pair_id, "tie"),
    )
    btn_right.button(
        CHOICE_LABELS["right"],
        use_container_width=True,
        type="primary",
        on_click=append_vote,
        args=(rater, pair_id, "right"),
    )
    if n_done:
        btn_undo.button("Undo", use_container_width=True, on_click=undo_last, args=(rater,))

    keyboard_shortcuts()
