"""Human review of the panel gold pairs (attention checks).

Run:  streamlit run app/gold_review.py --server.port 8504

Golds are the one place in the study where we punish a rater for an answer, so
every one has to be defensible on sight. Each card shows the pair as the rater
will see it, with the expected winner marked. Accept it only if the answer is
obvious to you within a second or two; reject anything you'd have to think about.

Decisions append to artifacts/panel-golds-v1/decisions.jsonl (resumable). "Apply
to golds.json" rewrites labels/panel-pilot/golds.json with the accepted set only.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
PANEL_DIR = ROOT / "labels" / "panel-pilot"
DECISIONS = ROOT / "artifacts" / "panel-golds-v1" / "decisions.jsonl"
EXPORT_DIR = ROOT / "data" / "exports" / "cmr1mr0m7000196d57zi3vcgn"
MIN_GOLDS = 20


def load_review() -> list[dict]:
    """Read fresh every rerun.

    Not cached on purpose: a long-lived Streamlit process cached this once and
    kept serving gold candidates from a superseded draw, including a pair that
    had since been promoted to a worked example. Re-reading a 26 KB file is
    cheaper than that class of mistake.
    """
    review = json.loads((PANEL_DIR / "golds-review.json").read_text())["golds"]
    drawn = drawn_pair_ids()
    return [g for g in review if g["pairId"] in drawn]


def drawn_pair_ids() -> set[str]:
    return {p["pairId"] for p in json.loads((PANEL_DIR / "pairs.json").read_text())["pairs"]}


def load_decisions() -> dict[str, str]:
    if not DECISIONS.exists():
        return {}
    out: dict[str, str] = {}
    for line in DECISIONS.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["pairId"]] = row["decision"]
    return out


def record(pair_id: str, decision: str) -> None:
    DECISIONS.parent.mkdir(parents=True, exist_ok=True)
    with DECISIONS.open("a") as fh:
        fh.write(
            json.dumps(
                {
                    "pairId": pair_id,
                    "decision": decision,
                    "decidedAt": datetime.now(timezone.utc).isoformat(),
                }
            )
            + "\n"
        )


def photo_source(face_id: str, url: str) -> str:
    local = EXPORT_DIR / "images" / f"{face_id}.webp"
    return str(local) if local.exists() else url


def apply_to_golds(review: list[dict], decisions: dict[str, str]) -> tuple[int, int]:
    """Keep everything not explicitly rejected, so an unfinished pass is safe.

    Golds are intersected with the current draw: the app throws at boot on a gold
    that isn't in pairs.json, and that is not a failure we want to discover on a
    deploy.
    """
    drawn = drawn_pair_ids()
    kept = {
        g["pairId"]: g["expected"]
        for g in review
        if decisions.get(g["pairId"]) != "reject" and g["pairId"] in drawn
    }
    path = PANEL_DIR / "golds.json"
    doc = json.loads(path.read_text())
    doc["expected"] = kept
    doc["goldCount"] = len(kept)
    doc["reviewedAt"] = datetime.now(timezone.utc).isoformat()
    doc["rejectedInReview"] = sorted(p for p, d in decisions.items() if d == "reject")
    path.write_text(json.dumps(doc, indent=1) + "\n")
    return len(kept), len(review) - len(kept)


def main() -> None:
    st.set_page_config(page_title="Gold pair review", layout="wide")
    st.title("Panel golds — review before shipping")

    if not (PANEL_DIR / "golds-review.json").exists():
        st.error("`labels/panel-pilot/golds-review.json` not found — run scripts/make_golds.py.")
        return

    review = load_review()
    decisions = load_decisions()
    accepted = [g for g in review if decisions.get(g["pairId"]) != "reject"]

    with st.sidebar:
        # Count only decisions for candidates still in the draw, otherwise a
        # superseded pair inflates this past the total.
        reviewed = sum(1 for g in review if g["pairId"] in decisions)
        st.metric("Reviewed", f"{reviewed} / {len(review)}")
        st.metric("Would ship", len(accepted))
        if len(accepted) < MIN_GOLDS:
            st.warning(f"Below {MIN_GOLDS} golds — raters see ~5 each; the pool gets thin.")
        hide_done = st.checkbox("Hide reviewed", value=True)
        per_page = st.selectbox("Pairs per page", [3, 5, 10], index=1)
        st.divider()
        if st.button("Apply to golds.json", type="primary"):
            n_keep, n_drop = apply_to_golds(review, decisions)
            st.success(f"golds.json now has {n_keep} golds ({n_drop} rejected)")
        st.caption(
            "Accept only if the marked side is obviously the more attractive face "
            "at a glance. Reject if it's arguable, if either photo is bad, or if "
            "you disagree. Rejecting costs nothing — a bad gold costs a real "
            "rater their payment."
        )

    pool = [g for g in review if not (hide_done and g["pairId"] in decisions)]
    if not pool:
        st.success("Every gold has been reviewed. Hit **Apply to golds.json** in the sidebar.")
        return

    for g in pool[:per_page]:
        pid = g["pairId"]
        st.divider()
        head, ctrl = st.columns([3, 1])
        with head:
            st.markdown(
                f"**{g['gender']}** · expected **{g['expected'].upper()}** · "
                f"win prob {g['winProb']:.3f} · theta gap {g['thetaGap']:.2f} · "
                f"VLM {g['vlmConfidence']}"
            )
        with ctrl:
            b1, b2 = st.columns(2)
            if b1.button("Accept", key=f"a-{pid}", type="primary"):
                record(pid, "accept")
                st.rerun()
            if b2.button("Reject", key=f"r-{pid}"):
                record(pid, "reject")
                st.rerun()

        left, right = st.columns(2, gap="medium")
        for side, col in (("left", left), ("right", right)):
            with col:
                face = g[side]
                st.image(photo_source(face["faceId"], face["photoUrl"]), width="stretch")
                if g["expected"] == side:
                    st.success(f"expected winner — {side}")
                else:
                    st.caption(f"{side}")

        if g["comments"]:
            with st.expander("pilot-500 rater reasoning"):
                for rater, comment in g["comments"].items():
                    st.markdown(f"**{rater}** — {comment}")
        prior = decisions.get(pid)
        if prior:
            st.caption(f"previously **{prior}ed** · `{pid}`")
        else:
            st.caption(f"`{pid}`")


if __name__ == "__main__":
    main()
