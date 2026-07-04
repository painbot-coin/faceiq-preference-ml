"""Streamlit dashboard: export health, BT rankings, diagnostics, training runs.

Run:  streamlit run app/dashboard.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from faceiq_pref.calibrate import ANCHORS  # noqa: E402
from faceiq_pref.data import ExportError, load_export  # noqa: E402

st.set_page_config(page_title="FaceIQ Preference ML", layout="wide")
st.title("FaceIQ Preference ML")


# ---------------------------------------------------------------- helpers


def find_exports() -> list[Path]:
    base = ROOT / "data" / "exports"
    if not base.exists():
        return []
    return sorted(p.parent for p in base.glob("*/manifest.json"))


def find_bt_refits() -> list[Path]:
    base = ROOT / "artifacts"
    if not base.exists():
        return []
    return sorted(p.parent for p in base.glob("*/ratings.csv"))


def find_training_runs() -> list[Path]:
    base = ROOT / "artifacts"
    if not base.exists():
        return []
    return sorted(
        p.parent for p in base.glob("*/metrics.json") if "history" in p.read_text()[:2000]
    )


tab_health, tab_rankings, tab_diag, tab_training = st.tabs(
    ["Export health", "BT rankings", "Refit diagnostics", "Training runs"]
)


# ---------------------------------------------------------------- export health

with tab_health:
    exports = find_exports()
    if not exports:
        st.info("No exports found. Run the export script from faceiq-labs first — see README.")
    else:
        export_dir = st.selectbox("Export", exports, format_func=lambda p: p.name)
        verify = st.checkbox("Verify shard hashes (slower)", value=False)
        try:
            export = load_export(export_dir, verify_hashes=verify)
            m = export.manifest
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Matchups", f"{m['counts']['matchups']:,}")
            c2.metric("Faces", f"{m['counts']['faces']:,}")
            c3.metric("Excluded (audit)", m["counts"].get("excluded", "n/a"))
            c4.metric("Shards", len(m["matchupShards"]))
            st.caption(
                f"Run `{m['runId']}` · exported {m.get('exportedAt', '?')} · "
                f"label rule: {m.get('labelRule', '?')}"
            )
            if verify:
                st.success("Shard hashes verified.")

            with st.spinner("Checking image coverage..."):
                faces = export.faces()
                present, missing = export.image_coverage(faces)
            if missing:
                st.error(f"{len(missing)} faces missing images (of {len(faces)}).")
                st.dataframe(pd.DataFrame({"faceId": missing[:200]}))
            else:
                st.success(f"All {present:,} face images present.")
        except ExportError as e:
            st.error(f"Export failed validation: {e}")


# ---------------------------------------------------------------- BT rankings

with tab_rankings:
    refits = find_bt_refits()
    if not refits:
        st.info("No BT refits yet. Run: `python scripts/run_bt.py --export data/exports/<runId>`")
    else:
        refit_dir = st.selectbox("Refit", refits, format_func=lambda p: p.name, key="refit_rank")
        df = pd.read_csv(refit_dir / "ratings.csv")

        col1, col2 = st.columns(2)
        gender = col1.selectbox("Gender", ["all"] + sorted(df["gender"].unique().tolist()))
        top_n = col2.slider("Show top N", 10, 200, 50)

        view = df if gender == "all" else df[df["gender"] == gender]
        view = view.sort_values("theta", ascending=False).head(top_n).reset_index(drop=True)

        show_photos = st.checkbox("Show photos", value=True)
        if show_photos:
            exports = find_exports()
            export_for_photos = exports[0] if exports else None
            cols_per_row = 5
            for start in range(0, len(view), cols_per_row):
                cols = st.columns(cols_per_row)
                for col, (_, row) in zip(cols, view.iloc[start : start + cols_per_row].iterrows()):
                    img = (export_for_photos / row["imagePath"]) if export_for_photos else None
                    with col:
                        if img is not None and img.exists():
                            st.image(str(img), use_container_width=True)
                        st.caption(
                            f"**{row['scoreOutOf10']:.2f}** /10 · θ {row['theta']:.2f} · "
                            f"D{int(row['decileBin']) if pd.notna(row['decileBin']) else '?'} · "
                            f"Labs {row['labsOverallScore'] if pd.notna(row['labsOverallScore']) else '—'}"
                        )
        st.dataframe(
            view[
                ["faceId", "gender", "theta", "percentile", "scoreOutOf10",
                 "comparisonCount", "decileBin", "labsOverallScore"]
            ],
            use_container_width=True,
        )


# ---------------------------------------------------------------- diagnostics

with tab_diag:
    refits = find_bt_refits()
    if not refits:
        st.info("No BT refits yet.")
    else:
        refit_dir = st.selectbox("Refit", refits, format_func=lambda p: p.name, key="refit_diag")
        metrics = json.loads((refit_dir / "metrics.json").read_text())
        df = pd.read_csv(refit_dir / "ratings.csv")

        st.subheader("Gates")
        ok = metrics.get("allGatesPassed")
        (st.success if ok else st.error)(f"All gates passed: {ok}")
        for gender, g in metrics.get("genders", {}).items():
            with st.expander(f"{gender} — details", expanded=not ok):
                st.json(g)

        st.subheader("Comparisons per face")
        st.plotly_chart(
            px.histogram(df, x="comparisonCount", color="gender", nbins=40),
            use_container_width=True,
        )

        st.subheader("BT score vs Labs overall_score")
        labs = df.dropna(subset=["labsOverallScore"])
        if len(labs):
            st.plotly_chart(
                px.scatter(
                    labs, x="labsOverallScore", y="scoreOutOf10", color="gender",
                    opacity=0.4, hover_data=["faceId"]
                ),
                use_container_width=True,
            )

        st.subheader("Calibration curve (pre-registered anchors)")
        from faceiq_pref.calibrate import score_out_of_10

        xs = np.linspace(0, 1, 500)
        curve = pd.DataFrame({"percentile": xs, "scoreOutOf10": [score_out_of_10(x) for x in xs]})
        fig = px.line(curve, x="percentile", y="scoreOutOf10")
        fig.add_scatter(
            x=[p for p, _ in ANCHORS], y=[s for _, s in ANCHORS],
            mode="markers", name="anchors", marker={"size": 10},
        )
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------- training

with tab_training:
    runs = find_training_runs()
    if not runs:
        st.info("No training runs yet. Run: `python scripts/train.py --config configs/train-v1.yaml`")
    else:
        run_dir = st.selectbox("Run", runs, format_func=lambda p: p.name)
        metrics = json.loads((run_dir / "metrics.json").read_text())
        hist = pd.DataFrame(metrics["history"])

        c1, c2, c3 = st.columns(3)
        c1.metric("Best val accuracy", f"{metrics['best_val_accuracy']:.4f}")
        c2.metric("Train pairs", f"{metrics['train_pairs']:,}")
        c3.metric("Val pairs", f"{metrics['val_pairs']:,}")

        st.plotly_chart(
            px.line(hist, x="epoch", y=["train_loss", "val_loss"], title="Loss"),
            use_container_width=True,
        )
        st.plotly_chart(
            px.line(hist, x="epoch", y="val_accuracy", title="Val pairwise accuracy"),
            use_container_width=True,
        )
        with st.expander("Config"):
            st.json(metrics["config"])
