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


def export_for_run(run_id: str | None) -> Path | None:
    """Export dir whose manifest matches run_id; falls back to the first export found."""
    exports = find_exports()
    for p in exports:
        try:
            if json.loads((p / "manifest.json").read_text()).get("runId") == run_id:
                return p
        except (OSError, json.JSONDecodeError):
            continue
    return exports[0] if exports else None


tab_overview, tab_health, tab_rankings, tab_diag, tab_training = st.tabs(
    ["Runs overview", "Export health", "BT rankings", "Refit diagnostics", "Training runs"]
)


# ---------------------------------------------------------------- runs overview

with tab_overview:
    st.subheader("BT refits")
    refits = find_bt_refits()
    if not refits:
        st.info("No BT refits yet. Run: `python scripts/run_bt.py --export data/exports/<runId>`")
    else:
        rows = []
        for refit_dir in refits:
            m = json.loads((refit_dir / "metrics.json").read_text())
            for gender, g in m.get("genders", {}).items():
                rows.append(
                    {
                        "refit": refit_dir.name,
                        "gender": gender,
                        "faces": g.get("faces"),
                        "stabilityRho": g.get("stabilityRho"),
                        "spearmanVsLabs": g.get("spearmanVsLabs"),
                        "gatesPassed": all(g.get("gates", {}).values()),
                        "allGatesPassed": m.get("allGatesPassed"),
                        "calibration": m.get("calibrationVariant", "v1"),
                        "matchupsUsed": m.get("matchupsUsed"),
                        "confidenceFilter": m.get("confidenceFilter") or "none",
                        "runId": m.get("runId"),
                        "refitAt": m.get("refitAt"),
                    }
                )
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.subheader("Training runs")
    runs = find_training_runs()
    if not runs:
        st.info("No training runs yet. Run: `python scripts/train.py --config configs/train-v1.yaml`")
    else:
        rows = []
        for run_dir in runs:
            m = json.loads((run_dir / "metrics.json").read_text())
            cfg = m.get("config", {})
            rows.append(
                {
                    "run": run_dir.name,
                    "bestValAccuracy": m.get("best_val_accuracy"),
                    "epochsRun": len(m.get("history", [])),
                    "trainPairs": m.get("train_pairs"),
                    "valPairs": m.get("val_pairs"),
                    "backbone": cfg.get("backbone"),
                    "batchSize": cfg.get("batch_size"),
                    "lr": cfg.get("lr"),
                    "device": m.get("device"),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


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

def photo_grid(view: pd.DataFrame, export_dir: Path | None, cols_per_row: int = 5) -> None:
    for start in range(0, len(view), cols_per_row):
        cols = st.columns(cols_per_row)
        for col, (_, row) in zip(cols, view.iloc[start : start + cols_per_row].iterrows()):
            img = (export_dir / row["imagePath"]) if export_dir else None
            with col:
                if img is not None and img.exists():
                    st.image(str(img), use_container_width=True)
                st.caption(
                    f"**{row['scoreOutOf10']:.2f}** /10 · θ {row['theta']:.2f} · "
                    f"pct {row['percentile']:.3f} · "
                    f"D{int(row['decileBin']) if pd.notna(row['decileBin']) else '?'} · "
                    f"Labs {row['labsOverallScore'] if pd.notna(row['labsOverallScore']) else '—'}"
                )


with tab_rankings:
    refits = find_bt_refits()
    if not refits:
        st.info("No BT refits yet. Run: `python scripts/run_bt.py --export data/exports/<runId>`")
    else:
        refit_dir = st.selectbox("Refit", refits, format_func=lambda p: p.name, key="refit_rank")
        df = pd.read_csv(refit_dir / "ratings.csv")
        refit_metrics = json.loads((refit_dir / "metrics.json").read_text())
        export_for_photos = export_for_run(refit_metrics.get("runId"))
        if export_for_photos is None:
            st.warning("No export folder found — photos unavailable.")

        col1, col2, col3 = st.columns(3)
        gender = col1.selectbox("Gender", ["all"] + sorted(df["gender"].unique().tolist()))
        section = col2.selectbox("Section", ["top", "around median", "bottom"])
        n = col3.slider("Show N", 10, 200, 50)

        view = df if gender == "all" else df[df["gender"] == gender]
        view = view.sort_values("theta", ascending=False).reset_index(drop=True)
        if section == "top":
            view = view.head(n)
        elif section == "bottom":
            view = view.tail(n)
        else:
            mid = len(view) // 2
            view = view.iloc[max(0, mid - n // 2) : mid + n - n // 2]
        view = view.reset_index(drop=True)

        show_photos = st.checkbox("Show photos", value=True)
        if show_photos:
            photo_grid(view, export_for_photos)
        st.dataframe(
            view[
                ["faceId", "gender", "theta", "percentile", "scoreOutOf10",
                 "comparisonCount", "decileBin", "labsOverallScore"]
            ],
            use_container_width=True,
        )

        st.subheader("Face lookup")
        query = st.text_input("faceId (full or prefix)", key="face_lookup")
        if query:
            hits = df[df["faceId"].str.startswith(query.strip())]
            if hits.empty:
                st.warning("No face matches that id in this refit.")
            else:
                if len(hits) > 1:
                    st.caption(f"{len(hits)} matches — showing up to 10.")
                photo_grid(hits.head(10), export_for_photos)
                st.dataframe(
                    hits[
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

        st.subheader("Calibration curve")
        from faceiq_pref.calibrate import make_scorer

        anchors = [tuple(a) for a in metrics.get("calibrationAnchors", ANCHORS)]
        variant = metrics.get("calibrationVariant", "v1 (pre-registered)")
        st.caption(f"Anchor set: {variant}" + (
            f" · recalibrated from {metrics['recalibratedFrom']}"
            if metrics.get("recalibratedFrom") else ""
        ))
        scorer = make_scorer(anchors)
        xs = np.linspace(0, 1, 500)
        curve = pd.DataFrame({"percentile": xs, "scoreOutOf10": [scorer(x) for x in xs]})
        fig = px.line(curve, x="percentile", y="scoreOutOf10")
        fig.add_scatter(
            x=[p for p, _ in anchors], y=[s for _, s in anchors],
            mode="markers", name="anchors", marker={"size": 10},
        )
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------- training

with tab_training:
    runs = find_training_runs()
    if not runs:
        st.info("No training runs yet. Run: `python scripts/train.py --config configs/train-v1.yaml`")
    else:
        # ---- cross-run comparison --------------------------------------
        st.subheader("All runs — comparison")
        all_hist, summary_rows = [], []
        for r in runs:
            m = json.loads((r / "metrics.json").read_text())
            h = pd.DataFrame(m["history"])
            h["run"] = r.name
            all_hist.append(h)
            cfg = m.get("config", {})
            summary_rows.append(
                {
                    "run": r.name,
                    "bestValAccuracy": m.get("best_val_accuracy"),
                    "finalValLoss": h["val_loss"].iloc[-1] if len(h) else None,
                    "epochs": len(h),
                    "trainPairs": m.get("train_pairs"),
                    "valPairs": m.get("val_pairs"),
                    "backbone": cfg.get("backbone"),
                    "lr": cfg.get("lr"),
                    "batchSize": cfg.get("batch_size"),
                    "device": m.get("device"),
                }
            )
        hist_all = pd.concat(all_hist, ignore_index=True)

        c1, c2 = st.columns(2)
        c1.plotly_chart(
            px.line(hist_all, x="epoch", y="val_accuracy", color="run",
                    title="Val pairwise accuracy — all runs", markers=True),
            use_container_width=True,
        )
        c2.plotly_chart(
            px.line(hist_all, x="epoch", y="val_loss", color="run",
                    title="Val loss — all runs", markers=True),
            use_container_width=True,
        )
        summary_df = pd.DataFrame(summary_rows)
        c3, c4 = st.columns([1, 2])
        c3.plotly_chart(
            px.bar(summary_df, x="run", y="bestValAccuracy", title="Best val accuracy per run"),
            use_container_width=True,
        )
        with c4:
            st.caption("Run summary (0.5 accuracy = coin flip; BT-consistent labels top out below 1.0 due to genuine preference noise)")
            st.dataframe(summary_df, use_container_width=True)

        st.divider()

        # ---- single-run detail -----------------------------------------
        st.subheader("Run detail")
        run_dir = st.selectbox("Run", runs, format_func=lambda p: p.name, key="train_detail")
        metrics = json.loads((run_dir / "metrics.json").read_text())
        hist = pd.DataFrame(metrics["history"])
        best_epoch = int(hist.loc[hist["val_accuracy"].idxmax(), "epoch"]) if len(hist) else None

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Best val accuracy", f"{metrics['best_val_accuracy']:.4f}")
        c2.metric("Best epoch", best_epoch if best_epoch is not None else "—")
        c3.metric("Train pairs", f"{metrics['train_pairs']:,}")
        c4.metric("Val pairs", f"{metrics['val_pairs']:,}")
        c5.metric("Backbone", metrics.get("config", {}).get("backbone", "?"))

        c1, c2 = st.columns(2)
        fig_loss = px.line(hist, x="epoch", y=["train_loss", "val_loss"],
                           title="Loss (train vs val)", markers=True)
        if best_epoch is not None:
            fig_loss.add_vline(x=best_epoch, line_dash="dash", annotation_text="best")
        c1.plotly_chart(fig_loss, use_container_width=True)

        fig_acc = px.line(hist, x="epoch", y="val_accuracy",
                          title="Val pairwise accuracy", markers=True)
        fig_acc.add_hline(y=0.5, line_dash="dot", annotation_text="chance (0.5)")
        if best_epoch is not None:
            fig_acc.add_vline(x=best_epoch, line_dash="dash", annotation_text="best")
        c2.plotly_chart(fig_acc, use_container_width=True)

        if "time" in hist.columns and len(hist) > 1:
            durations = hist["time"].diff().dropna()
            st.plotly_chart(
                px.bar(x=hist["epoch"].iloc[1:], y=durations / 60,
                       labels={"x": "epoch", "y": "minutes"}, title="Epoch duration"),
                use_container_width=True,
            )

        with st.expander("Config"):
            st.json(metrics["config"])

        # ---- post-training evaluation ----------------------------------
        st.subheader("Post-training evaluation")
        eval_path = run_dir / "eval.json"
        scores_path = run_dir / "model_scores.csv"

        if eval_path.exists():
            ev = json.loads(eval_path.read_text())
            agree = ev.get("rank_agreement_vs_bt", {})
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Held-out accuracy", f"{ev.get('val_accuracy', float('nan')):.4f}")
            c2.metric("Kendall τ vs BT", f"{agree['kendall_tau']:.4f}" if agree else "—")
            c3.metric("Spearman ρ vs BT", f"{agree['spearman_rho']:.4f}" if agree else "—")
            c4.metric("Val pairs scored", f"{ev.get('val_pairs_scored', 0):,}")
        else:
            st.caption(
                "No eval artifacts yet. Generate with: "
                f"`python scripts/evaluate.py --checkpoint checkpoints/{run_dir.name}/best.pt "
                "--ratings artifacts/bt-refit-v1/ratings.csv`"
            )

        if scores_path.exists():
            sc = pd.read_csv(scores_path)
            c1, c2 = st.columns(2)
            c1.plotly_chart(
                px.scatter(sc, x="theta", y="modelScore",
                           color="gender" if "gender" in sc.columns else None,
                           opacity=0.4, hover_data=["faceId"],
                           title="Model score vs BT theta (per face)"),
                use_container_width=True,
            )
            c2.plotly_chart(
                px.histogram(sc, x="modelScore",
                             color="gender" if "gender" in sc.columns else None,
                             nbins=50, title="Model score distribution"),
                use_container_width=True,
            )
