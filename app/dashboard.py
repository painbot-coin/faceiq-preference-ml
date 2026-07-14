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


(tab_overview, tab_health, tab_rankings, tab_diag, tab_training, tab_inspect, tab_infer,
 tab_anchor) = st.tabs(
    ["Runs overview", "Export health", "BT rankings", "Refit diagnostics", "Training runs",
     "Model inspection", "Inference", "Anchor panel"]
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


# ---------------------------------------------------------------- shared: model + export caches


def find_score_runs() -> list[Path]:
    """Artifact dirs that have per-face model scores (training runs + ensembles)."""
    base = ROOT / "artifacts"
    if not base.exists():
        return []
    return sorted(p.parent for p in base.glob("*/model_scores.csv"))


def find_checkpoints() -> list[Path]:
    base = ROOT / "checkpoints"
    if not base.exists():
        return []
    return sorted(base.glob("*/best.pt"))


@st.cache_data(show_spinner="Loading face image paths...")
def load_face_image_paths(export_dir: str) -> dict[str, str]:
    export = load_export(export_dir, verify_hashes=False)
    return {f.face_id: str(export.image_path(f)) for f in export.faces().values()}


@st.cache_data(show_spinner="Loading val pairs (parses full export once)...")
def load_val_pairs_df(export_dir: str, val_fraction: float, split_seed: int) -> pd.DataFrame:
    from faceiq_pref.data import split_by_face_id

    export = load_export(export_dir, verify_hashes=False)
    _, val_rows = split_by_face_id(export.all_matchups(), val_fraction, split_seed)
    return pd.DataFrame(
        {
            "faceA": m.face_a_id,
            "faceB": m.face_b_id,
            "gender": m.gender,
            "finalOutcome": m.final_outcome,
            "confidence": m.confidence,
            "humanLabeled": m.human_labeled_at is not None,
        }
        for m in val_rows
    )


@st.cache_resource(show_spinner="Loading model checkpoint...")
def load_scorer_cached(ckpt_path: str):
    import torch

    from faceiq_pref.model import PreferenceScorer
    from faceiq_pref.train import TrainConfig, build_transforms

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = TrainConfig(**ckpt["config"])
    scorer = PreferenceScorer(cfg.backbone, pretrained=False)
    scorer.load_state_dict({k.removeprefix("scorer."): v for k, v in ckpt["model"].items()})
    scorer.eval()
    tf = build_transforms(cfg.backbone, cfg.image_size, augment=False)
    return scorer, cfg, tf


@st.cache_resource(show_spinner="Loading MediaPipe face landmarker...")
def normalizer_available() -> bool:
    try:
        from faceiq_pref.preprocess import _get_landmarker

        _get_landmarker()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------- model inspection

with tab_inspect:
    score_runs = find_score_runs()
    if not score_runs:
        st.info("No runs with model_scores.csv yet. Run scripts/evaluate.py with --ratings first.")
    else:
        run_dir = st.selectbox(
            "Run", score_runs, format_func=lambda p: p.name, key="inspect_run",
            index=len(score_runs) - 1,
        )
        sc = pd.read_csv(run_dir / "model_scores.csv")

        # export dir: from run config when available, else first export on disk
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            run_cfg = json.loads(metrics_path.read_text()).get("config", {})
        else:
            run_cfg = {}  # ensembles have no metrics.json; all runs share the split defaults
        exports = find_exports()
        export_dir = run_cfg.get("export_dir") or (str(exports[0]) if exports else None)
        img_paths = load_face_image_paths(export_dir) if export_dir else {}

        # per-gender ranks (1 = best)
        sc["modelRank"] = sc.groupby("gender")["modelScore"].rank(ascending=False)
        sc["btRank"] = sc.groupby("gender")["theta"].rank(ascending=False)
        sc["rankGap"] = (sc["modelRank"] - sc["btRank"]).abs()

        def face_card(col, face_id: str, caption: str) -> None:
            with col:
                p = img_paths.get(face_id)
                if p and Path(p).exists():
                    st.image(p, use_container_width=True)
                st.caption(caption)

        # ---- 1. biggest model-vs-BT disagreements -----------------------
        st.subheader("Biggest model vs BT rank disagreements")
        st.caption(
            "Faces the model ranks very differently from Bradley-Terry, shown **in context**: "
            "the faces each ranking considers its peers. If the face visually belongs with its "
            "BT peers, the model is wrong; if it belongs with its model peers, BT is likely "
            "noisy there. (rank 1 = most attractive within gender)"
        )
        c1, c2 = st.columns(2)
        gender_d = c1.selectbox("Gender", sorted(sc["gender"].unique()), key="inspect_gender")
        n_d = c2.slider("Show N", 3, 15, 5, key="inspect_n")

        g_view = sc[sc["gender"] == gender_d]
        by_bt = g_view.sort_values("btRank").reset_index(drop=True)
        by_model = g_view.sort_values("modelRank").reset_index(drop=True)
        n_gender = len(g_view)

        def rank_peers(df_sorted: pd.DataFrame, face_id: str, k: int = 4) -> pd.DataFrame:
            """k faces ranked immediately around face_id (excluding it)."""
            pos = int(df_sorted.index[df_sorted["faceId"] == face_id][0])
            lo = max(0, pos - k // 2 - (0 if pos + k // 2 < len(df_sorted) else k // 2))
            window = df_sorted.iloc[lo : lo + k + 1]
            return window[window["faceId"] != face_id].head(k)

        worst = (
            g_view.sort_values("rankGap", ascending=False).head(n_d).reset_index(drop=True)
        )
        for _, row in worst.iterrows():
            c_target, c_ctx = st.columns([1, 4])
            face_card(
                c_target, row["faceId"],
                f"**BT #{int(row['btRank'])}** vs **model #{int(row['modelRank'])}** "
                f"of {n_gender} · gap {int(row['rankGap'])}",
            )
            with c_ctx:
                st.caption(f"BT says its peers are (around #{int(row['btRank'])}):")
                peers = rank_peers(by_bt, row["faceId"])
                cols = st.columns(len(peers))
                for col, (_, p) in zip(cols, peers.iterrows()):
                    face_card(col, p["faceId"], f"BT #{int(p['btRank'])}")
                st.caption(f"Model says its peers are (around #{int(row['modelRank'])}):")
                peers = rank_peers(by_model, row["faceId"])
                cols = st.columns(len(peers))
                for col, (_, p) in zip(cols, peers.iterrows()):
                    face_card(col, p["faceId"], f"model #{int(p['modelRank'])}")
            st.divider()

        # ---- 2. eye test: top / bottom by model score --------------------
        st.subheader("Eye test — model's top and bottom")
        section = st.radio("Section", ["top 20", "bottom 20"], horizontal=True, key="inspect_sec")
        view = sc[sc["gender"] == gender_d].sort_values("modelScore", ascending=False)
        view = view.head(20) if section == "top 20" else view.tail(20).iloc[::-1]
        view = view.reset_index(drop=True)
        for start in range(0, len(view), 5):
            cols = st.columns(5)
            for col, (_, row) in zip(cols, view.iloc[start : start + 5].iterrows()):
                face_card(
                    col, row["faceId"],
                    f"model #{int(row['modelRank'])} · BT #{int(row['btRank'])} · "
                    f"θ {row['theta']:.2f}",
                )

        # ---- 3. confidently wrong val pairs ------------------------------
        st.subheader("Confidently wrong val pairs")
        st.caption(
            "Val pairs where the model's score margin was largest but the label disagreed. "
            "Siamese scoring means margin = model's pairwise logit exactly. "
            "Label errors show up here as pairs where you agree with the model."
        )
        if export_dir is None:
            st.warning("No export found — cannot rebuild the val split.")
        else:
            vp = load_val_pairs_df(
                export_dir,
                float(run_cfg.get("val_fraction", 0.2)),
                int(run_cfg.get("split_seed", 42)),
            )
            score_map = dict(zip(sc["faceId"], sc["modelScore"]))
            vp = vp[vp["finalOutcome"].isin(["A", "B"])].copy()
            vp = vp[vp["faceA"].isin(score_map) & vp["faceB"].isin(score_map)]
            vp["margin"] = vp["faceA"].map(score_map) - vp["faceB"].map(score_map)
            vp["predicted"] = np.where(vp["margin"] > 0, "A", "B")
            vp["correct"] = vp["predicted"] == vp["finalOutcome"]
            acc = vp["correct"].mean()
            st.caption(f"Val accuracy from per-face scores: {acc:.4f} ({len(vp)} pairs)")

            wrong = vp[~vp["correct"]].copy()
            wrong["absMargin"] = wrong["margin"].abs()
            wrong = wrong.sort_values("absMargin", ascending=False).head(15)
            for _, row in wrong.iterrows():
                ca, cb, ci = st.columns([2, 2, 3])
                face_card(ca, row["faceA"], "A")
                face_card(cb, row["faceB"], "B")
                with ci:
                    st.markdown(
                        f"**Label says {row['finalOutcome']} wins** · model picked "
                        f"{row['predicted']} (margin {row['absMargin']:.2f})\n\n"
                        f"confidence: `{row['confidence'] or '—'}` · "
                        f"human-labeled: {'yes' if row['humanLabeled'] else 'no'}"
                    )
                st.divider()


# ---------------------------------------------------------------- inference

with tab_infer:
    st.caption(
        "Score new, unseen photos with a trained checkpoint and see where they land in the "
        "ranked cohort. Sanity-check tool — not the production anchor-ladder."
    )
    ckpts = find_checkpoints()
    if not ckpts:
        st.info("No checkpoints found under checkpoints/*/best.pt.")
    else:
        ckpt_path = st.selectbox(
            "Checkpoint", ckpts, format_func=lambda p: p.parent.name,
            index=len(ckpts) - 1, key="infer_ckpt",
        )

        # cohort context: model_scores.csv of the same run when available
        score_runs = find_score_runs()
        default_ctx = next(
            (i for i, p in enumerate(score_runs) if p.name == ckpt_path.parent.name),
            len(score_runs) - 1 if score_runs else 0,
        )
        ctx_dir = st.selectbox(
            "Cohort context (per-face scores of this run)", score_runs,
            format_func=lambda p: p.name, index=default_ctx, key="infer_ctx",
        ) if score_runs else None
        if ctx_dir is not None and ctx_dir.name != ckpt_path.parent.name:
            st.warning(
                "Context run differs from the checkpoint — percentiles are only meaningful "
                "when both come from the same run."
            )

        gender_ctx = st.selectbox("Compare against", ["female", "male"], key="infer_gender")
        normalize = st.checkbox(
            "Normalize like faceiq-labs (MediaPipe eye-level crop, same pipeline that produced "
            "the cohort training photos — recommended)",
            value=True,
        )

        uploads = st.file_uploader(
            "Photos", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True,
        )
        if uploads and ctx_dir is not None:
            import torch
            from PIL import Image

            scorer, cfg, tf = load_scorer_cached(str(ckpt_path))
            use_norm = normalize and normalizer_available()
            if normalize and not use_norm:
                st.warning("mediapipe not available (`uv pip install mediapipe`) — scoring uncropped images.")

            sc = pd.read_csv(ctx_dir / "model_scores.csv")
            cohort = sc[sc["gender"] == gender_ctx].sort_values("modelScore").reset_index(drop=True)

            # join BT /10 + image paths for the ladder display
            refits = find_bt_refits()
            ratings = pd.read_csv(refits[0] / "ratings.csv") if refits else None
            exports = find_exports()
            img_paths = load_face_image_paths(str(exports[0])) if exports else {}

            for up in uploads:
                img = Image.open(up).convert("RGB")
                shown = img
                if use_norm:
                    from faceiq_pref.preprocess import normalize_front_photo

                    cropped = normalize_front_photo(img)
                    if cropped is None:
                        st.error(f"{up.name}: no face detected — scoring full image.")
                    else:
                        shown = cropped

                with torch.no_grad():
                    s = scorer(tf(shown).unsqueeze(0)).item()

                pct = float((cohort["modelScore"] < s).mean())
                c_img, c_res = st.columns([1, 3])
                with c_img:
                    st.image(shown, use_container_width=True)
                    st.caption(up.name)
                with c_res:
                    st.metric(
                        "Cohort percentile",
                        f"{pct:.1%}",
                        help=f"model score {s:.3f} vs {len(cohort)} {gender_ctx} cohort faces",
                    )
                    if ratings is not None:
                        # approximate /10 from the 5 nearest cohort faces by model score
                        joined = cohort.merge(
                            ratings[["faceId", "scoreOutOf10"]], on="faceId", how="inner"
                        )
                        joined["dist"] = (joined["modelScore"] - s).abs()
                        nearest = joined.nsmallest(5, "dist")
                        st.metric(
                            "Approx. /10 (5 nearest cohort faces)",
                            f"{nearest['scoreOutOf10'].mean():.2f}",
                        )

                    # ladder: 3 cohort faces just below and above
                    pos = int((cohort["modelScore"] < s).sum())
                    lo = cohort.iloc[max(0, pos - 3) : pos]
                    hi = cohort.iloc[pos : pos + 3]
                    ladder = pd.concat([lo, hi])
                    st.caption("Nearest cohort faces (below → above)")
                    cols = st.columns(max(len(ladder), 1))
                    for col, (_, row) in zip(cols, ladder.iterrows()):
                        with col:
                            p = img_paths.get(row["faceId"])
                            if p and Path(p).exists():
                                st.image(p, use_container_width=True)
                            st.caption(f"score {row['modelScore']:.2f}")
                st.divider()


# ---------------------------------------------------------------- anchor panel

with tab_anchor:
    from faceiq_pref.anchors import (
        add_anchor,
        anchor_image_path,
        load_anchors,
        panel_violations,
        place_on_ladder,
        remove_anchor,
    )

    st.caption(
        "Reference-set inference (§5.4 Path B). Curate anchor faces with **product-assigned** "
        "/10 scores; test photos are placed on that ladder instead of the research cohort's "
        "percentiles. Anchors are normalized with the faceiq-labs crop and stored locally "
        "under `data/anchors/` (git-ignored)."
    )

    # ---- 1. build the panel ------------------------------------------
    st.subheader("Build panel")
    if not normalizer_available():
        st.error("mediapipe not available (`uv pip install mediapipe`) — cannot add anchors.")
    else:
        with st.form("add_anchor", clear_on_submit=True):
            c1, c2, c3 = st.columns([3, 1, 1])
            anchor_up = c1.file_uploader(
                "Anchor photo", type=["jpg", "jpeg", "png", "webp"], key="anchor_up"
            )
            anchor_gender = c2.selectbox("Gender", ["female", "male"], key="anchor_gender")
            anchor_score = c3.number_input(
                "Product /10", min_value=1.0, max_value=10.0, value=6.0, step=0.5
            )
            anchor_label = st.text_input("Note (optional, e.g. who this is)", key="anchor_note")
            if st.form_submit_button("Normalize & save anchor") and anchor_up is not None:
                from PIL import Image

                saved = add_anchor(
                    Image.open(anchor_up).convert("RGB"),
                    anchor_gender, anchor_score, anchor_label,
                )
                if saved is None:
                    st.error("No face detected in that photo — anchor not saved.")
                else:
                    st.success(f"Saved anchor {saved.anchor_id} at {anchor_score:g}/10.")

    anchors = load_anchors()
    panel_gender = st.selectbox("Panel", ["female", "male"], key="panel_gender")
    panel = sorted(
        (a for a in anchors if a.gender == panel_gender),
        key=lambda a: a.product_score,
    )

    if not panel:
        st.info(
            f"No {panel_gender} anchors yet. Aim for ~8–10 spanning the range "
            "(e.g. 2, 3, 4, 5, 6, 7, 8, 9) — spacing below ~1.0 sits inside model noise."
        )
    else:
        # score the panel once per checkpoint (cached); shows validation inline
        ckpts_a = find_checkpoints()
        ckpt_anchor = st.selectbox(
            "Checkpoint", ckpts_a, format_func=lambda p: p.parent.name,
            index=len(ckpts_a) - 1 if ckpts_a else 0, key="anchor_ckpt",
        ) if ckpts_a else None

        @st.cache_data(show_spinner="Scoring anchors (cached per checkpoint)...")
        def score_anchors(ckpt: str, anchor_ids: tuple, _paths: tuple) -> dict[str, float]:
            import torch
            from PIL import Image

            scorer, cfg, tf = load_scorer_cached(ckpt)
            out = {}
            for aid, p in zip(anchor_ids, _paths):
                with Image.open(p) as img, torch.no_grad():
                    out[aid] = scorer(tf(img.convert("RGB")).unsqueeze(0)).item()
            return out

        anchor_scores: dict[str, float] = {}
        if ckpt_anchor is not None:
            anchor_scores = score_anchors(
                str(ckpt_anchor),
                tuple(a.anchor_id for a in panel),
                tuple(str(anchor_image_path(a)) for a in panel),
            )
            scored_panel = [(a, anchor_scores[a.anchor_id]) for a in panel]
            problems = panel_violations(scored_panel)
            if problems:
                st.error(
                    "**Panel ordering violations** — the model disagrees with your labels here; "
                    "placements near these rungs are unreliable:\n\n- " + "\n- ".join(problems)
                )
            else:
                st.success(
                    "Panel order validated: the model ranks all anchors in the same order as "
                    "your assigned scores."
                )

        cols = st.columns(min(len(panel), 10))
        for col, a in zip(cols * ((len(panel) // 10) + 1), panel):
            with col:
                p = anchor_image_path(a)
                if p.exists():
                    st.image(str(p), use_container_width=True)
                ms = f" · model {anchor_scores[a.anchor_id]:.2f}" if a.anchor_id in anchor_scores else ""
                st.caption(f"**{a.product_score:g}/10**{ms}" + (f" · {a.label}" if a.label else ""))
                if st.button("remove", key=f"rm_{a.anchor_id}"):
                    remove_anchor(a.anchor_id)
                    st.rerun()

        # ---- 2. cohort smoke test -------------------------------------
        if len(panel) >= 3 and ckpt_anchor is not None:
            with st.expander("Cohort smoke test — how does this ladder behave at scale?"):
                st.caption(
                    "Places every cohort face of this gender on the ladder using the "
                    "checkpoint's precomputed per-face scores. Validates ladder behavior "
                    "in bulk: band widths, inconsistent placements, coverage, and implied "
                    "/10 vs the research BT /10."
                )
                ctx_csv = ROOT / "artifacts" / ckpt_anchor.parent.name / "model_scores.csv"
                if not ctx_csv.exists():
                    st.warning(
                        f"No model_scores.csv for run {ckpt_anchor.parent.name} — run "
                        "scripts/evaluate.py with --ratings first."
                    )
                elif st.button("Run smoke test", key="smoke_run"):
                    cohort_sc = pd.read_csv(ctx_csv)
                    cohort_sc = cohort_sc[cohort_sc["gender"] == panel_gender]
                    scored_panel = [(a, anchor_scores[a.anchor_id]) for a in panel]

                    placements = []
                    for _, row in cohort_sc.iterrows():
                        r = place_on_ladder(row["modelScore"], scored_panel)
                        placements.append(
                            {
                                "faceId": row["faceId"],
                                "theta": row["theta"],
                                "implied": r["implied_score"],
                                "bandWidth": r["band"][1] - r["band"][0],
                                "inconsistent": r["inconsistent"],
                                "outside": r["position"] != "within the panel",
                            }
                        )
                    pl = pd.DataFrame(placements)

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Faces placed", f"{len(pl):,}")
                    c2.metric("Median band width", f"{pl['bandWidth'].median():.2f}")
                    c3.metric("Inconsistent", f"{pl['inconsistent'].mean():.1%}")
                    c4.metric("Outside panel", f"{pl['outside'].mean():.1%}")

                    cc1, cc2 = st.columns(2)
                    cc1.plotly_chart(
                        px.histogram(pl, x="implied", nbins=40,
                                     title="Implied product /10 distribution"),
                        use_container_width=True,
                    )
                    cc2.plotly_chart(
                        px.scatter(pl, x="theta", y="implied", opacity=0.3,
                                   hover_data=["faceId"],
                                   title="Implied /10 vs BT theta"),
                        use_container_width=True,
                    )
                    st.caption(
                        "Reading guide: band width ≈ placement uncertainty in /10 points "
                        "(median under ~1.5 is workable). Inconsistent or outside-panel "
                        "rates above a few percent mean the ladder needs better anchors "
                        "at the affected rungs. The scatter should rise monotonically; "
                        "flat plateaus mean several BT tiers collapse onto one rung."
                    )

        # ---- 3. place test photos on the ladder ----------------------
        st.subheader("Test photos against the ladder")
        if len(panel) < 3:
            st.info("Add at least 3 anchors to this panel to run placements.")
        elif ckpt_anchor is None:
            st.info("No checkpoint available.")
        else:
            test_ups = st.file_uploader(
                "Test photos", type=["jpg", "jpeg", "png", "webp"],
                accept_multiple_files=True, key="anchor_test",
            )
            if test_ups:
                import torch
                from PIL import Image

                from faceiq_pref.preprocess import normalize_front_photo

                scorer, cfg, tf = load_scorer_cached(str(ckpt_anchor))
                scored_panel = [(a, anchor_scores[a.anchor_id]) for a in panel]

                for up in test_ups:
                    img = Image.open(up).convert("RGB")
                    norm = normalize_front_photo(img)
                    if norm is None:
                        st.error(f"{up.name}: no face detected — skipped.")
                        continue
                    with torch.no_grad():
                        u = scorer(tf(norm).unsqueeze(0)).item()

                    res = place_on_ladder(u, scored_panel)
                    lo, hi = res["band"]

                    c_img, c_res = st.columns([1, 3])
                    with c_img:
                        st.image(norm, use_container_width=True)
                        st.caption(up.name)
                    with c_res:
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Implied product score", f"{res['implied_score']:.2f} /10")
                        m2.metric(
                            "Confidence band",
                            f"{lo:.1f} – {hi:.1f}" if lo != hi else "tight",
                            help="Anchors with ambiguous win probability (25–75%) span this range",
                        )
                        m3.metric("Beats / loses", f"{res['n_beaten']} / {res['n_lost']}")
                        if res["position"] != "within the panel":
                            st.warning(f"Placement is {res['position']} — extend the ladder.")
                        if res["inconsistent"]:
                            st.error(
                                "Inconsistent placement: beats an anchor labeled higher than one "
                                "it loses to (panel ordering defect near this score)."
                            )
                        ladder_cols = st.columns(len(res["rows"]))
                        for col, r in zip(ladder_cols, res["rows"]):
                            with col:
                                p = anchor_image_path(r["anchor"])
                                if p.exists():
                                    st.image(str(p), use_container_width=True)
                                verdict = "beats" if r["beats"] else "loses"
                                st.caption(
                                    f"**{r['anchor'].product_score:g}** · {verdict} "
                                    f"({r['p_win']:.0%})"
                                )
                    st.divider()
