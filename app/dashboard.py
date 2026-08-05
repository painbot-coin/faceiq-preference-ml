"""Streamlit dashboard: export health, BT rankings, diagnostics, training runs.

Run:  streamlit run app/dashboard.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from faceiq_pref.calibrate import ANCHORS, make_scorer, score_out_of_10  # noqa: E402
from faceiq_pref.composite import (  # noqa: E402
    METHODS,
    blend,
    discover_sources,
    evaluate_blend,
    normalise,
)
from faceiq_pref.data import ExportError, load_export  # noqa: E402
from faceiq_pref.panel import load_panel_votes  # noqa: E402

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


def run_order(name: str) -> tuple[int, int, str]:
    """Sort key putting the newest training run last.

    Plain alphabetical sort ranks `train-v9` above `train-v14`, so selectors defaulting to the
    last entry silently pick an old checkpoint. Sort on the version NUMBER, and keep non-run
    names (ensembles, external baselines) after the numbered runs.
    """
    m = re.match(r"train-v(\d+)", name)
    return (0, int(m.group(1)), name) if m else (1, 0, name)


def find_training_runs() -> list[Path]:
    base = ROOT / "artifacts"
    if not base.exists():
        return []
    return sorted(
        (p.parent for p in base.glob("*/metrics.json") if "history" in p.read_text()[:2000]),
        key=lambda p: run_order(p.name),
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


def gender_choices(series: pd.Series) -> list[str]:
    """Sorted gender labels, ignoring NaN / empty / non-string values."""
    vals = [v for v in series.dropna().unique() if isinstance(v, str) and v.strip()]
    return sorted(vals)


(tab_overview, tab_health, tab_rankings, tab_diag, tab_panel, tab_bands, tab_spend,
 tab_training, tab_inspect, tab_gallery, tab_composite, tab_infer,
 tab_labeler) = st.tabs(
    ["Runs overview", "Export health", "BT rankings", "Refit diagnostics", "Human panel",
     "Score bands", "Label spend", "Training runs", "Model inspection", "Model gallery",
     "Composite", "Inference", "Pilot-500 labeler"]
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
        gender = col1.selectbox("Gender", ["all"] + gender_choices(df["gender"]))
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


# ---------------------------------------------------------------- human panel

PANEL_VALUE = ROOT / "artifacts" / "panel-value-v1"
STRATA_ORDER = [
    "close", "mid", "lowconf", "overlap", "clear",
    "topup-0-10",
    # Run 4's strata are percentile-gap bands, so they must sort numerically rather than
    # alphabetically — "random-2-5" sorts before "random-10-20" as a string.
    "random-0-2", "random-2-5", "random-5-10", "random-10-20", "random-20-45", "random-45-100",
]

# Each study ships its own draw, so its results only make sense next to the
# sample-meta.json it was drawn from. Newest last.
PANEL_RUNS = {
    "run 2 — 3,000 pairs × 12 votes": {
        "artifacts": ROOT / "artifacts" / "panel-run-v1",
        "meta": ROOT / "labels" / "panel-pilot" / "sample-meta.json",
        "refit": "bt-refit-v3-panel",
        "delta": None,
        "spend": 1900,
        "raters": 360,
    },
    "run 3 — 4,800 hard pairs × 6 votes": {
        "artifacts": ROOT / "artifacts" / "panel-run-v3",
        "meta": ROOT / "labels" / "panel-run-3" / "sample-meta.json",
        "refit": "bt-refit-v4-panel",
        "delta": ROOT / "artifacts" / "panel-run-delta-v3" / "run-delta.json",
        "spend": 1286,
        "raters": 300,
    },
    "run 4 — 2,500 uniform random pairs × 12 votes": {
        "artifacts": ROOT / "artifacts" / "panel-run-v4",
        "meta": ROOT / "labels" / "panel-run-4-random" / "sample-meta.json",
        "refit": "bt-refit-v5-panel",
        "delta": ROOT / "artifacts" / "panel-run-delta-v4" / "run-delta.json",
        "spend": 969,
        "raters": 303,
    },
}


def strata_order(df: pd.DataFrame) -> list[str]:
    """Canonical strata first, then anything a later draw introduced."""
    present = list(dict.fromkeys(df["stratum"].dropna()))
    return [s for s in STRATA_ORDER if s in present] + \
           [s for s in present if s not in STRATA_ORDER]


@st.cache_data(show_spinner="Loading panel run...")
def load_panel_labels(run_dir: str, meta: str) -> pd.DataFrame:
    """Per-pair panel results joined to the answer key (strata, VLM call, thetas)."""
    df = pd.read_csv(Path(run_dir) / "majority-labels.csv")
    meta_path = Path(meta)
    if meta_path.exists():
        meta = pd.DataFrame(json.loads(meta_path.read_text())["pairs"])
        df = df.merge(
            meta[["pairId", "faceAId", "faceBId", "faceAOnLeft", "thetaA", "thetaB"]],
            on="pairId", how="left",
        )
    df["decided"] = df["left"] + df["right"]
    df["favShare"] = df[["left", "right"]].max(axis=1) / df["decided"]
    # votes for face A regardless of the side it was displayed on
    df["sharaA"] = np.where(df["faceAOnLeft"], df["left"], df["right"]) / df["decided"]
    return df


def agreement_curve(df: pd.DataFrame, ratings: pd.DataFrame, label: str) -> pd.DataFrame:
    """How often humans picked the face a given ranking rates higher, by gap size."""
    pct = dict(zip(ratings["faceId"], ratings["percentile"]))
    ten = dict(zip(ratings["faceId"], ratings["scoreOutOf10"]))
    rows = []
    for _, r in df.iterrows():
        pa, pb = pct.get(r["faceAId"]), pct.get(r["faceBId"])
        if pa is None or pb is None or not r["decided"]:
            continue
        agree = r["sharaA"] if pa > pb else 1 - r["sharaA"]
        rows.append({"gap": abs(pa - pb) * 100,
                     "tenGap": abs(ten[r["faceAId"]] - ten[r["faceBId"]]),
                     "agree": agree, "votes": r["decided"]})
    cur = pd.DataFrame(rows)
    if cur.empty:
        return cur
    bands = [(0, 5), (5, 10), (10, 20), (20, 30), (30, 45), (45, 60), (60, 80), (80, 101)]
    out = []
    for lo, hi in bands:
        sel = cur[(cur["gap"] >= lo) & (cur["gap"] < hi)]
        if sel.empty:
            continue
        out.append({"band": f"{lo}–{min(hi, 100)}",
                    "gapMid": (lo + min(hi, 100)) / 2,
                    "tenGap": sel["tenGap"].mean(),
                    "pairs": len(sel),
                    "humansAgree": sel["agree"].mean(),
                    "source": label})
    return pd.DataFrame(out)


with tab_panel:
    available = {k: v for k, v in PANEL_RUNS.items()
                 if (v["artifacts"] / "majority-labels.csv").exists()}
    if not available:
        st.info(
            "No panel run found. Run `python scripts/analyze_panel_run.py` after a "
            "Prolific study, then `python scripts/refit_bt_panel.py`."
        )
    else:
        run_name = st.selectbox("Study", list(available), index=len(available) - 1,
                                key="panel_run_pick")
        run = available[run_name]
        PANEL_RUN = run["artifacts"]
        labels = load_panel_labels(str(PANEL_RUN), str(run["meta"]))
        order = strata_order(labels)
        n_votes = int(labels["votes"].sum())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pairs judged", f"{len(labels):,}")
        c2.metric("Human votes", f"{n_votes:,}")
        c3.metric("Votes per pair", f"{n_votes/len(labels):.1f}")
        c4.metric("Dead splits", f"{(labels['majority'].isna()).mean():.1%}",
                  help="Pairs where the panel split exactly evenly — no majority label. "
                       "Rises when a study buys fewer votes per pair.")

        # ---- headline for a top-up study: what did the last cheque buy? ----
        if run["delta"] and run["delta"].exists():
            d = json.loads(run["delta"].read_text())
            st.subheader("Was this study worth the money?")
            st.caption(
                f"Scored on {d['heldOutPairs']:,} of this study's pairs held out "
                "**whole** — not one of their votes entered any fit. A model can only "
                "do better on them by having learned better face scores, which is the "
                "only way the *next* study could help pairs we have not bought. "
                f"Averaged over {d['seeds']} splits."
            )
            lad = pd.DataFrame(d["ladder"])
            names = {"VLM only": "machine labels only",
                     "+ prior runs": "+ earlier studies",
                     "+ this run": "+ this study"}
            lad["model"] = lad["model"].map(names).fillna(lad["model"])
            rt_path = PANEL_RUN / "raters.csv"
            ceiling = None
            if rt_path.exists():
                loo = pd.read_csv(rt_path)["looAgree"].dropna()
                ceiling = float(loo.mean())

            k1, k2, k3, k4 = st.columns(4)
            base, final = lad.iloc[0], lad.iloc[-1]
            k1.metric("Machine labels alone", f"{base['accuracyMean']:.1%}",
                      help="50% is a coin flip. On pairs this close the VLM is blind.")
            k2.metric("After this study", f"{final['accuracyMean']:.1%}",
                      delta=f"{final['deltaVsPrevious']:+.1%} vs before it")
            if ceiling:
                got = final["accuracyMean"] - base["accuracyMean"]
                k3.metric("Of the reachable gap",
                          f"{got / (ceiling - base['accuracyMean']):.0%}",
                          help=f"Ceiling is {ceiling:.1%}: how often one rater agrees "
                               "with the rest on the same pair. No ranking can beat it.")
            k4.metric("Cost per point", f"${d['dollarsPerPoint']:,.0f}",
                      help=f"${d['spend']:,.0f} bought "
                           f"{final['deltaVsPrevious'] * 100:.2f} points")

            fig = px.bar(lad, x="model", y="accuracyMean", error_y="accuracySd",
                         color="model", color_discrete_sequence=["#9aa0a6", "#8ab4f8",
                                                                "#1a73e8"],
                         labels={"accuracyMean": "predicts a vote on a pair it never saw",
                                 "model": ""})
            fig.add_hline(y=0.5, line_dash="dot", annotation_text="coin flip")
            if ceiling:
                fig.add_hline(y=ceiling, line_dash="dash", line_color="#e8710a",
                              annotation_text="ceiling — raters disagree above this")
            fig.update_yaxes(tickformat=".0%",
                             range=[0.48, (ceiling or 0.65) + 0.02])
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            dose = pd.DataFrame(d["doseCurve"])
            st.caption(
                "**Are we running out of road?** Same test, feeding a growing share of "
                "this study's pairs into the fit. If the curve bent over at the right "
                "edge, the next study would buy less than this one did."
            )
            fig = px.line(dose, x="votes", y="accuracyMean", error_y="accuracySd",
                          markers=True, hover_data=["pairs", "fraction"],
                          labels={"votes": "human votes in the fit",
                                  "accuracyMean": "accuracy on withheld pairs"})
            fig.update_yaxes(tickformat=".1%")
            st.plotly_chart(fig, use_container_width=True)
            steps = [(b["votes"] - a["votes"],
                      (b["accuracyMean"] - a["accuracyMean"]) * 100)
                     for a, b in zip(d["doseCurve"], d["doseCurve"][1:])]
            st.dataframe(pd.DataFrame(
                [{"votes added": v, "points gained": p,
                  "points per 1,000 votes": p / v * 1000} for v, p in steps]
            ).style.format({"votes added": "{:,.0f}", "points gained": "{:+.2f}",
                            "points per 1,000 votes": "{:.3f}"}),
                use_container_width=True)
            st.divider()

        # ---- headline: the one chart to send someone ----------------------
        cap_path = ROOT / "artifacts" / "cohort-capacity-v1" / "capacity.json"
        v3_path = ROOT / "artifacts" / run["refit"] / "metrics.json"
        if cap_path.exists() and v3_path.exists():
            cap = json.loads(cap_path.read_text())["headroom"]
            ho = json.loads(v3_path.read_text()).get("heldOut", {})
            by = ho.get("byStratum", {})
            rows = []
            for s in order:
                if s not in by or s not in cap:
                    continue
                rows.append({"stratum": s,
                             "before (machine labels only)": by[s]["vlmOnly"],
                             "after (+ human votes)": by[s]["joint"],
                             "ceiling (best any ranking can do)": cap[s]["ceiling"]})
            hl = pd.DataFrame(rows)
            if not hl.empty:
                st.subheader("Headline — did paying for human labels work?")
                close = hl[hl["stratum"] == "close"]
                c1, c2, c3, c4 = st.columns(4)
                if len(close):
                    r = close.iloc[0]
                    avail = r["ceiling (best any ranking can do)"] - r["before (machine labels only)"]
                    got = r["after (+ human votes)"] - r["before (machine labels only)"]
                    c1.metric("Hard pairs — before", f"{r['before (machine labels only)']:.1%}",
                              help="Machine labels alone. 50% is a coin flip.")
                    c2.metric("Hard pairs — after", f"{r['after (+ human votes)']:.1%}",
                              delta=f"{got:+.1%}")
                    c3.metric("Of the achievable gain", f"{got/avail:.0%}",
                              help="The ceiling is under 100% because raters disagree with "
                                   "each other on the same pair. This is the share of the "
                                   "reachable gap we closed.")
                c4.metric("Spent", f"${run['spend']:,}",
                          help=f"{run['raters']} raters, {n_votes:,} votes")

                melted = hl.melt(id_vars="stratum",
                                 value_vars=["before (machine labels only)",
                                             "after (+ human votes)"],
                                 var_name="ranking", value_name="accuracy")
                fig = px.bar(melted, x="stratum", y="accuracy", color="ranking",
                             barmode="group", category_orders={"stratum": order},
                             color_discrete_sequence=["#9aa0a6", "#4c8bf5"],
                             title="Predicting a withheld rater's actual choice")
                fig.add_scatter(
                    x=hl["stratum"], y=hl["ceiling (best any ranking can do)"],
                    mode="markers", name="ceiling (raters disagree above this)",
                    marker={"symbol": "line-ew-open", "size": 34, "line": {"width": 3},
                            "color": "#e8710a"},
                )
                fig.add_hline(y=0.5, line_dash="dot", annotation_text="coin flip")
                fig.update_yaxes(tickformat=".0%", range=[0.45, 0.82],
                                 title="share of individual votes predicted")
                fig.update_layout(legend={"orientation": "h", "y": -0.2})
                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    "**How to read it:** grey is the ranking built from machine labels "
                    "alone, blue is the same ranking after absorbing the panel's votes, "
                    "orange is the most any ranking could ever score (raters contradict "
                    "each other above that line). On `close` pairs the machine ranking was "
                    "at a coin flip — it genuinely could not tell which face people prefer — "
                    "and human votes closed a large share of the reachable gap. On `clear` "
                    "pairs there was nothing to fix, which is why we no longer buy labels "
                    "there. Every number is measured on raters whose votes were withheld "
                    "from the fit."
                )
                st.divider()

        # ---- the two accuracies, which are not interchangeable -------------
        acc_matrix = ROOT / "artifacts" / "panel-run-v4" / "accuracy-matrix.json"
        if acc_matrix.exists():
            with st.expander(
                "⚠️ There are two 'accuracies' and mixing them invents a 10-point gap — read this "
                "before quoting any number"
            ):
                st.markdown(
                    "- **vs the majority label** — did the predictor pick the face that *more* "
                    "raters picked? A pair split 9–3 is one clean win, so 100% is reachable.\n"
                    "- **vs individual votes** — what share of the raw ballots agree with the "
                    "pick? On that same 9–3 pair a **perfect** predictor scores only 75%, because "
                    "a quarter of the crowd disagreed with itself.\n\n"
                    "Each needs its own ceiling. `analyze_panel_run.py` reports the first; "
                    "`eval_vs_panel.py` and `panel_run_delta.py` report the second."
                )
                am = json.loads(acc_matrix.read_text())
                rows = []
                for dist, block in am.items():
                    for pred, r in block["predictors"].items():
                        rows.append({"pair distribution": dist, "predictor": pred,
                                     "vs majority label": r["vsMajority"],
                                     "vs individual votes": r["vsVotes"]})
                st.dataframe(pd.DataFrame(rows).style.format({
                    "vs majority label": "{:.1%}", "vs individual votes": "{:.1%}"}),
                    use_container_width=True, hide_index=True)
                st.caption(
                    "The same ranking is **~8 points better than a person** at naming the crowd's "
                    "choice and **level with a person** at predicting one person's ballot. Both "
                    "are true and they answer different questions. Note also how far every number "
                    "moves between the two pair distributions — that is why an accuracy without a "
                    "stated pair distribution is not a number. Regenerate with "
                    "`scripts/accuracy_matrix.py`."
                )
            st.divider()

        st.subheader("Does our ranking predict what people actually choose?")
        st.caption(
            "Each point is a band of pairs grouped by how far apart the ranking puts the two "
            "faces. The y-axis is how often real raters picked the face the ranking prefers. "
            "A flat line at 50% would mean the ranking is noise; rising to 90%+ means it is "
            "measuring real preference. Where the curve is flat at the **left** is our "
            "resolution limit — differences that small are invisible to people."
        )
        refits = find_bt_refits()
        chosen = st.multiselect(
            "Rankings to compare", [p.name for p in refits],
            default=[n for n in ("bt-refit-v2-qc", run["refit"])
                     if n in [p.name for p in refits]] or [refits[-1].name],
            key="panel_refits",
        )
        curves = []
        for name in chosen:
            ratings = pd.read_csv(ROOT / "artifacts" / name / "ratings.csv")
            c = agreement_curve(labels, ratings, name)
            if not c.empty:
                curves.append(c)
        if curves:
            cur = pd.concat(curves, ignore_index=True)
            fig = px.line(cur, x="gapMid", y="humansAgree", color="source", markers=True,
                          hover_data=["band", "pairs", "tenGap"],
                          labels={"gapMid": "gap between the faces (percentile points)",
                                  "humansAgree": "humans pick the higher-rated face"})
            fig.add_hline(y=0.5, line_dash="dot", annotation_text="coin flip")
            fig.add_hline(y=0.75, line_dash="dot", annotation_text="3 in 4 agree")
            fig.update_yaxes(tickformat=".0%", range=[0.4, 1.0])
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "⚠️ In-sample for any ranking that was fit on these votes — every panel "
                "refit at or after the study selected above. The non-circular versions "
                "are the held-out tests on this page."
            )
            st.dataframe(
                cur.pivot(index="band", columns="source", values="humansAgree")
                   .style.format("{:.1%}"),
                use_container_width=True,
            )

        st.divider()
        st.subheader("Did the human votes actually improve the ranking?")
        st.caption(
            "Honest test: half the raters are withheld, the ranking is refit on the other "
            "half, and both versions are scored on the withheld half. No pair's own votes "
            "help predict itself."
        )
        refit_metrics = ROOT / "artifacts" / run["refit"] / "metrics.json"
        if not refit_metrics.exists():
            st.info("Run `python scripts/refit_bt_panel.py` to produce this comparison.")
        else:
            m = json.loads(refit_metrics.read_text())
            ho = m.get("heldOut", {})
            by = ho.get("byStratum", {})
            # The baseline is whatever --baseline pointed at. Once that is itself a panel
            # refit it was fit on every rater, test half included, so it is only a fair
            # comparison on strata it never saw.
            base_dir = Path(m.get("vsBaseline", {}).get("baseline", "")).parent.name
            base_col = "machine labels only" if base_dir in ("", "bt-refit-v2-qc") \
                else f"before ({base_dir})"
            contaminated = base_dir not in ("", "bt-refit-v2-qc")
            if by:
                rows = [{"stratum": s,
                         base_col: by[s]["vlmOnly"],
                         "+ human votes": by[s]["joint"],
                         "gain": by[s]["joint"] - by[s]["vlmOnly"],
                         "votes scored": by[s]["votes"]}
                        for s in order if s in by]
                rows.append({"stratum": "ALL",
                             base_col: ho["vlmOnlyAccuracy"],
                             "+ human votes": ho["jointAccuracy"],
                             "gain": ho["delta"],
                             "votes scored": ho["votesEvaluated"]})
                hd = pd.DataFrame(rows)
                if contaminated:
                    st.warning(
                        f"The baseline here is **{base_dir}**, which was fit on *all* "
                        "raters — including the withheld half — for the pairs it owned. "
                        "It therefore looks artificially strong on earlier studies' "
                        "strata. Only rows for pairs it never saw are a fair fight; the "
                        "clean comparison is the held-out-pairs test at the top."
                    )
                melted = hd[hd["stratum"] != "ALL"].melt(
                    id_vars="stratum", value_vars=[base_col, "+ human votes"],
                    var_name="ranking", value_name="accuracy")
                fig = px.bar(melted, x="stratum", y="accuracy", color="ranking",
                             barmode="group",
                             category_orders={"stratum": order},
                             labels={"accuracy": "predicts a withheld rater's vote"})
                fig.add_hline(y=0.5, line_dash="dot", annotation_text="coin flip")
                fig.update_yaxes(tickformat=".0%", range=[0.45, 0.8])
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(
                    hd.style.format({base_col: "{:.1%}", "+ human votes": "{:.1%}",
                                     "gain": "{:+.1%}", "votes scored": "{:,.0f}"}),
                    use_container_width=True,
                )
                if not contaminated:
                    st.caption(
                        "`close` is the headline: the VLM-only ranking sat at **chance** "
                        "there, so it had no idea which face people prefer. `clear` "
                        "barely moves because there was nothing to fix."
                    )

        st.divider()
        st.subheader("Is the disagreement on hard pairs real, or is it noise?")
        st.caption(
            "`max share` is the winning side's share of the votes. It is ≥50% by construction, "
            "so it is plotted against what pure coin-flipping would produce given the same "
            "vote counts. Bars above the dashed line are real consensus."
        )
        rng = np.random.default_rng(0)
        rows = []
        for s in order:
            sel = labels[(labels["stratum"] == s) & (labels["decided"] >= 6)]
            if sel.empty:
                continue
            null = []
            for n in sel["decided"].astype(int):
                k = rng.binomial(n, 0.5, size=40)
                null.append(np.mean(np.maximum(k, n - k) / n))
            rows.append({"stratum": s, "pairs": len(sel),
                         "observed": sel["favShare"].mean(),
                         "coin flip": float(np.mean(null)),
                         "≥75% agreement": float((sel["favShare"] >= 0.75).mean())})
        cons = pd.DataFrame(rows)
        if not cons.empty:
            fig = px.bar(cons.melt(id_vars="stratum", value_vars=["observed", "coin flip"],
                                   var_name="", value_name="max share"),
                         x="stratum", y="max share", color="", barmode="group",
                         category_orders={"stratum": order})
            fig.update_yaxes(tickformat=".0%", range=[0.5, 0.9])
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(
                cons.style.format({"observed": "{:.1%}", "coin flip": "{:.1%}",
                                   "≥75% agreement": "{:.1%}"}),
                use_container_width=True,
            )

        st.divider()
        st.subheader("Breadth or depth? — how to shape the next buy")
        st.caption(
            "Measured on run 2, which is the only study with enough votes per pair to "
            "vary depth. It is what set the 6-votes-on-many-pairs shape of later runs."
        )
        if not (PANEL_VALUE / "value-curve.json").exists():
            st.info(
                "Run `python scripts/panel_value_curve.py --export data/exports/<runId>` "
                "to measure whether more spend keeps paying."
            )
        else:
            vc = json.loads((PANEL_VALUE / "value-curve.json").read_text())
            cur = pd.DataFrame(vc["curve"])
            cur["mode"] = np.where(cur["label"].str.startswith("cap"), "depth (more votes/pair)",
                                   "breadth (more pairs)")
            st.caption(
                f"Measured on {vc['heldOutPairs']} pairs whose votes were **never used in any "
                f"fit** ({vc['heldOutClosePairs']} of them hard pairs), so the gain has to "
                "travel through better face scores — the same mechanism a bigger buy relies "
                "on. A curve that flattens means the next dollar buys nothing."
            )
            metric = st.radio("Accuracy on", ["hard pairs only", "all held-out pairs"],
                              horizontal=True, key="panel_value_metric")
            ycol = "heldOutClose" if metric == "hard pairs only" else "heldOutAccuracy"
            fig = px.line(cur.sort_values("spendEquivalent"), x="spendEquivalent", y=ycol,
                          color="mode", markers=True, hover_data=["label", "pairsWithVotes"],
                          labels={"spendEquivalent": "equivalent spend ($)",
                                  ycol: "accuracy on unseen pairs"})
            fig.add_hline(y=0.5, line_dash="dot", annotation_text="coin flip")
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

            eq_path = PANEL_VALUE / "equal-budget.json"
            if eq_path.exists():
                eq = pd.DataFrame(json.loads(eq_path.read_text())["configs"])
                eq["config"] = (eq["pairs"].astype(str) + " pairs × "
                                + eq["votesPerPair"].astype(str) + " votes")
                st.caption(
                    "**Same money, spent three ways.** This is the design decision: covering "
                    "more pairs shallowly beats covering fewer pairs deeply."
                )
                fig = px.bar(eq, x="config", y="heldOutCloseMean", error_y="heldOutCloseSd",
                             labels={"heldOutCloseMean": "accuracy on unseen hard pairs"})
                fig.update_yaxes(tickformat=".0%", range=[0.45, 0.6])
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(
                    eq[["config", "spendEquivalent", "heldOutCloseMean", "heldOutCloseSd",
                        "heldOutAccuracyMean"]].style.format(
                        {"spendEquivalent": "${:,.0f}", "heldOutCloseMean": "{:.2%}",
                         "heldOutCloseSd": "±{:.2%}", "heldOutAccuracyMean": "{:.2%}"}),
                    use_container_width=True,
                )

        st.divider()
        st.subheader("Rater quality")
        raters_path = PANEL_RUN / "raters.csv"
        if raters_path.exists():
            rt = pd.read_csv(raters_path)
            c1, c2 = st.columns(2)
            if "looAgree" in rt.columns:
                fig = px.histogram(rt.dropna(subset=["looAgree"]), x="looAgree", nbins=30,
                                   labels={"looAgree": "agreement with the rest of the panel"})
                fig.add_vline(x=0.5, line_dash="dot", annotation_text="chance")
                fig.update_xaxes(tickformat=".0%")
                c1.plotly_chart(fig, use_container_width=True)
            gold_col = next((c for c in ("goldPassRate", "goldRate", "golds") if c in rt.columns),
                            None)
            if gold_col:
                c2.plotly_chart(
                    px.histogram(rt, x=gold_col, nbins=20,
                                 labels={gold_col: "attention checks passed"}),
                    use_container_width=True,
                )
            st.caption(
                "Attention checks alone are a poor quality signal: with 5 golds each, "
                "honest raters fail two by luck often enough to fill a reject list. "
                "Agreement with the rest of the panel is 20× the evidence, so a rater "
                "is only dropped when independent signals agree — and only *rejected* "
                "on Prolific when the evidence is behavioural (see "
                "`prolific-rejections.txt`, which is the shorter list)."
            )
            with st.expander("Per-rater table"):
                st.dataframe(rt, use_container_width=True)


# ---------------------------------------------------------------- score bands

BAND_CAL = ROOT / "artifacts" / "panel-run-v4" / "band-calibration.json"
OOS_CAL = ROOT / "artifacts" / "panel-run-v4" / "calibration.json"
GAP_CURVE = ROOT / "artifacts" / "panel-run-v4" / "gap-curve.json"


def _sigmoid_agree(gap: float, t: float) -> float:
    return 1.0 / (1.0 + np.exp(-gap / t))


with tab_bands:
    st.subheader("What number do we actually put in front of a user?")
    st.caption(
        "A point score like **6.43** asserts an ordering that most people would not agree with. "
        "This tab is the measured version of that claim, and the arithmetic for turning it into "
        "a band. Everything here comes from **run 4** — 2,500 pairs drawn uniformly at random, "
        "the only pairs in the programme no ranking had been fitted on when they were scored. "
        "Regenerate with `scripts/band_calibration.py`."
    )

    if not BAND_CAL.exists():
        st.info(
            "Run `python scripts/band_calibration.py --ratings artifacts/bt-refit-v2-qc/"
            "ratings.csv --panel labels/panel-run-4-random/results "
            "labels/panel-run-4-random/sample-meta.json --rejects "
            "artifacts/panel-run-v4/reject-pids.txt --out "
            "artifacts/panel-run-v4/band-calibration.json`"
        )
    else:
        bc = json.loads(BAND_CAL.read_text())
        T = bc["temperatureOnTenScale"]

        st.markdown(
            "##### Three different things get called a band, and only one of them is a "
            "property of the world"
        )
        st.dataframe(pd.DataFrame([
            {"band": "Disagreement",
             "what it is": "people genuinely differ about the same two faces",
             "does it shrink?": "No — never. It is a property of the population",
             "status": f"measured: T = {T:.3f} (this tab)"},
            {"band": "Estimation",
             "what it is": "how well we know THIS face's score from its comparisons",
             "does it shrink?": "Yes, as 1/sqrt(comparisons)",
             "status": "measured: median 95% interval 0.91 /10 pts (bt_uncertainty.py)"},
            {"band": "Photo",
             "what it is": "how much a score moves between two photos of one person",
             "does it shrink?": "Yes, as 1/sqrt(photos) — down to a floor",
             "status": "UNMEASURED — needs a second-view export"},
        ]), use_container_width=True, hide_index=True)
        st.caption(
            "Only the **photo** band is what a 'upload more photos and your band narrows' "
            "feature can reduce, and it is the one we have never measured: all 3,000 cohort "
            "rows are a single front view of 3,000 distinct people. The **disagreement** band "
            "is the one to show users, because it is true today and does not move."
        )
        st.divider()

        # ---- the measured curve ------------------------------------------
        st.markdown("##### The measured curve: what a /10 gap buys in agreement")
        gaps = np.linspace(0, 5, 200)
        curve = pd.DataFrame({"gap": gaps, "agree": [_sigmoid_agree(g, T) for g in gaps]})
        fig = px.line(curve, x="gap", y="agree",
                      labels={"gap": "difference in /10 score",
                              "agree": "share of people who agree with that ordering"})
        if OOS_CAL.exists():
            oc = json.loads(OOS_CAL.read_text())["rankings"]
            first = next(iter(oc.values()))
            mids = {"0-0.25": 0.125, "0.25-0.5": 0.375, "0.5-1": 0.75,
                    "1-1.5": 1.25, "1.5-2": 1.75, "2+": 3.0}
            obs = pd.DataFrame([
                {"gap": mids.get(r["gap"], np.nan), "agree": r["higherScoreWinShare"],
                 "pairs": r["pairs"], "band": r["gap"],
                 "lo": r["ci"][0], "hi": r["ci"][1]}
                for r in first["byScoreGap"]
            ]).dropna(subset=["gap"])
            fig.add_scatter(
                x=obs["gap"], y=obs["agree"], mode="markers", name="observed (run 4)",
                marker={"size": 11, "color": "#e8710a"},
                error_y={"type": "data", "symmetric": False,
                         "array": obs["hi"] - obs["agree"],
                         "arrayminus": obs["agree"] - obs["lo"]},
                customdata=obs[["band", "pairs"]],
                hovertemplate="%{customdata[0]} pts · %{customdata[1]} pairs · %{y:.1%}",
            )
        fig.add_hline(y=0.5, line_dash="dot", annotation_text="coin flip")
        fig.update_yaxes(tickformat=".0%", range=[0.45, 1.0])
        fig.update_layout(legend={"orientation": "h", "y": -0.25})
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"Orange points are the raw observed win rates with 95% intervals; the line is the "
            f"fitted `P(higher score wins) = sigmoid(gap / T)` with **T = {T:.3f}**. "
            "**T is the only parameter, and it is a scale in /10 points**: it is the gap at "
            "which agreement reaches 73.1%, so a small T would mean the scale is sharp and a "
            f"large T means it is blunt. Ours is {T:.2f}, which is blunt — that is the finding, "
            "not a fitting problem."
        )

        levels = pd.DataFrame(bc["bandForAgreement"])
        st.dataframe(
            levels.rename(columns={
                "agreement": "if we want this many people to agree",
                "gapPoints": "the two scores must differ by (/10 pts)",
                "shareOfCohortWithinBandOfMedian": "share of the cohort inside that band",
            }).style.format({"if we want this many people to agree": "{:.0%}",
                             "the two scores must differ by (/10 pts)": "{:.2f}"}),
            use_container_width=True, hide_index=True)
        st.divider()

        # ---- the product surface ----------------------------------------
        st.markdown("##### What the user sees — pick a score and an honesty level")
        c1, c2 = st.columns([1, 2])
        score = c1.slider("Model's point score", 1.0, 9.0, 6.4, 0.1, key="band_score")
        conf = c1.select_slider(
            "How many people should agree with the band's edges?",
            options=[0.55, 0.60, 2 / 3, 0.75, 0.80],
            value=2 / 3, format_func=lambda v: f"{v:.0%}", key="band_conf")
        gap = T * np.log(conf / (1 - conf))          # gap at which `conf` of people agree
        half = gap / 2                               # so two non-overlapping bands differ by gap
        lo, hi = max(1.0, score - half), min(10.0, score + half)
        c2.metric("Band to display", f"{lo:.1f} – {hi:.1f}",
                  help="Half-width is T·logit(agreement)/2, so two people whose bands do not "
                       "overlap differ by the full resolvable gap.")
        from faceiq_pref.placement import agreement_vs_tier_below, tier_of

        _tier, _tlo, _thi = tier_of(score)
        _below = agreement_vs_tier_below(score)
        _tier_line = (
            f"About **{_below[1]:.0%} of people** would place you above a typical **Tier "
            f"{_below[0]}** face."
            if _below else "This is the lowest tier, so there is no tier below to compare to."
        )
        c2.markdown(
            f"> **You score {score:.1f} — Tier {_tier} of 7** ({_tlo:.1f}–{_thi:.1f}).  \n"
            f"> {_tier_line}  \n"
            f"> Anyone whose band overlaps **{lo:.1f}–{hi:.1f}** is closer to you than the scale "
            f"can resolve — we would not claim an ordering between you."
        )
        c2.caption(
            "**Why this is phrased against a tier and not a number.** The mechanical version of "
            f"the sentence is *\"{conf:.0%} of people would place you above someone scoring "
            f"{max(1.0, score - gap):.1f}\"* — arithmetically correct and bad copy, because it "
            "reads as though some raters think this face **is** that number. They do not: that "
            "number is a *different, lower-scoring face*, one resolvable gap down. Naming the tier "
            "below carries the same information and cannot be misread that way."
        )
        c2.caption(
            "**Why half the gap:** two bands stop overlapping exactly when the two scores differ "
            f"by the full {gap:.2f} points, which is the distance at which {conf:.0%} of people "
            "agree. So non-overlapping bands mean a real ordering and overlapping bands mean a "
            "call we cannot make — the same convention as a confidence interval, but the width "
            "comes from how much *people* disagree rather than from our sample size. Note what "
            "this does **not** need: no panel judges the user. The band is a property of the "
            "*scale*, fitted once offline on 2,500 pairs and 29,351 votes; at inference the model "
            "emits one score and the band is arithmetic on top of it."
        )
        st.divider()

        # ---- multi-photo consolidation ----------------------------------
        st.markdown("##### If a user uploads several photos, how much should the band narrow?")
        st.caption(
            "The intuition is right and the obvious rule is wrong. **Intersecting** the "
            "intervals from each photo is not how independent estimates combine: it is "
            "overconfident when the photos agree and returns an **empty** band when they "
            "disagree. The correct operation is inverse-variance weighting."
        )
        st.latex(r"\hat\theta=\frac{\sum_i \theta_i/\sigma_i^2}{\sum_i 1/\sigma_i^2}"
                 r"\qquad\hat\sigma^2=\frac{1}{\sum_i 1/\sigma_i^2}"
                 r"\qquad\Rightarrow\qquad"
                 r"\hat\sigma^2=\sigma^2\frac{1+(n-1)\rho}{n}")
        p1, p2 = st.columns(2)
        sigma_photo = p1.slider(
            "σ_photo — per-photo score noise (/10 pts). UNMEASURED, this is the whole point",
            0.0, 1.0, 0.45, 0.05, key="band_sigma")
        rho = p1.slider(
            "ρ — correlation between two photos' errors (same face, so > 0)",
            0.0, 0.9, 0.4, 0.05, key="band_rho")
        ns = np.arange(1, 11)
        shrunk = sigma_photo * np.sqrt((1 + (ns - 1) * rho) / ns)
        total = np.sqrt(shrunk ** 2 + (T * np.log(2) / 2) ** 2)
        fig = px.line(pd.DataFrame({"photos": ns, "photo term only": shrunk,
                                    "photo + disagreement": total}),
                      x="photos", y=["photo term only", "photo + disagreement"], markers=True,
                      labels={"value": "band half-width (/10 pts)", "photos": "photos uploaded",
                              "variable": ""})
        fig.add_hline(y=sigma_photo * np.sqrt(rho), line_dash="dash", line_color="#e8710a",
                      annotation_text="floor: σ·√ρ — more photos cannot beat this")
        fig.update_layout(legend={"orientation": "h", "y": -0.25})
        p2.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"At these settings, going from 1 photo to 3 narrows the photo term by "
            f"**{1 - shrunk[2] / shrunk[0]:.0%}** and the *displayed* band by only "
            f"**{1 - total[2] / total[0]:.0%}**, because the disagreement term "
            f"(±{T * np.log(2) / 2:.2f} at 2-in-3) does not move at all. The dashed floor is "
            "σ·√ρ — the person's irreducible score uncertainty. **Measuring σ_photo and ρ is "
            "exactly what the test–retest export gives us, and until then the feature has an "
            "unknown asymptote.** Slide σ_photo to 0.1 to see the case where the feature is "
            "not worth building."
        )

        if GAP_CURVE.exists():
            gc = json.loads(GAP_CURVE.read_text())
            with st.expander("Is the scale real, or an artefact of how we drew pairs?"):
                st.caption(
                    "The percentile bands are cut on a ranking built from **machine** labels, so "
                    "the fair objection is that this is circular. Run 4 tests it: pairs drawn "
                    "uniformly, bands recorded rather than imposed, scored against votes the "
                    "ranking has never seen. A circular partition cannot produce a monotone "
                    "out-of-sample curve — and this one is monotone in all six bands."
                )
                gb = pd.DataFrame(gc["bands"])
                null = gc.get("coinFlipNull", {})
                fig = px.bar(gb, x="band", y="btAgrees", hover_data=["pairs"],
                             labels={"btAgrees": "ranking agrees with the popular vote",
                                     "band": "percentile gap between the two faces"})
                fig.add_scatter(x=gb["band"], y=gb["majorityShare"], mode="lines+markers",
                                name="crowd majority share (how united the crowd was)",
                                line={"color": "#e8710a"})
                if null.get("majorityShare"):
                    fig.add_hline(y=null["majorityShare"], line_dash="dash",
                                  line_color="#e8710a",
                                  annotation_text="majority share if every pair were a coin flip")
                fig.add_hline(y=0.5, line_dash="dot", annotation_text="ranking at chance")
                fig.update_yaxes(tickformat=".0%", range=[0.45, 1.0])
                fig.update_layout(legend={"orientation": "h", "y": -0.3})
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(
                    f"`spearman(percentile gap, crowd decisiveness)` = "
                    f"**{gc['spearmanGapDecisiveness']:+.3f}** with no binning. "
                    "**Bars** are the ranking's accuracy and rise monotonically. The **orange "
                    "line** is a different quantity — how united the crowd was, regardless of "
                    "who it favoured — and it is *flat* below a 20-point gap, well above its own "
                    "coin-flip floor. So closeness in the ranking predicts the ranking's own "
                    "error, but below 20 points it does not predict how divided people are."
                )
        st.divider()
        st.caption(
            "Written up in `docs/research/programme-direction-review.md` §5 and research log "
            "§5.8. **Do not build interval intersection.**"
        )


# ---------------------------------------------------------------- label spend

LABEL_INFO = ROOT / "artifacts" / "label-information-v2" / "report.json"
LABEL_INFO_R4 = ROOT / "artifacts" / "label-information-v2" / "report-run4only.json"


with tab_spend:
    st.subheader("Where is a human vote worth buying?")
    st.caption(
        "**Headroom** is the column that decides money: how often one rater agrees with the "
        "crowd, minus how often the free Gemini label does. Where it is positive, a purchased "
        "vote adds information the machine label does not have. Where it is zero we would be "
        "buying a duplicate — and where it is *negative*, the purchased vote is worse than the "
        "free one. Regenerate with `scripts/label_information.py`."
    )
    if not LABEL_INFO.exists():
        st.info("Run `scripts/label_information.py --out artifacts/label-information-v2/report.json`")
    else:
        def headroom_bands(path: Path) -> pd.DataFrame:
            """byPercentileGap joined to byBandCost, with headroom converted to /100 points."""
            rep = json.loads(path.read_text())
            df = pd.DataFrame(rep["byPercentileGap"])
            cost = {r["band"]: r for r in rep.get("byBandCost", [])}
            df["remaining"] = df["band"].map(lambda b: cost.get(b, {}).get("remaining"))
            df["cost"] = df["band"].map(lambda b: cost.get(b, {}).get("cost"))
            for c in ("headroom", "headroomLo", "headroomHi"):
                df[c] = df[c] * 100
            # Colour on the interval, not the point estimate: an interval straddling zero is a
            # different decision from one entirely below it.
            df["call"] = np.where(df["headroomLo"] > 0, "buy",
                                  np.where(df["headroomHi"] < 0, "skip — worse than free",
                                           "do not buy"))
            return df

        bands = headroom_bands(LABEL_INFO)
        buy = bands[bands["call"] == "buy"]

        k1, k2, k3 = st.columns(3)
        k1.metric("Validated spend remaining", f"${buy['cost'].sum():,.0f}",
                  help=f"{buy['remaining'].sum():,.0f} unbought pairs inside a 20-point gap, "
                       "where the whole interval clears zero")
        k2.metric("Cancelled on measurement",
                  f"${bands.loc[bands['call'] != 'buy', 'cost'].sum():,.0f}",
                  help="Bands whose headroom interval does not clear zero. Not 'deferred' — "
                       "at 45-100 the whole interval is below zero.")
        k3.metric("Full coverage would cost", f"${bands['cost'].sum():,.0f}",
                  help="And most of it would buy duplicates of a free label.")

        fig = px.bar(bands, x="band", y="headroom", hover_data=["pairs", "cost"],
                     labels={"headroom": "points a human vote adds over the free label",
                             "band": "percentile gap between the two faces", "call": ""},
                     color="call",
                     color_discrete_map={"buy": "#1a73e8", "do not buy": "#9aa0a6",
                                         "skip — worse than free": "#d93025"},
                     error_y=bands["headroomHi"] - bands["headroom"],
                     error_y_minus=bands["headroom"] - bands["headroomLo"])
        fig.add_hline(y=0, line_dash="dash")
        fig.update_layout(legend={"orientation": "h", "y": -0.25})
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Bars with the whole interval above zero are validated spend. **45–100 is the one "
            "to notice: the entire interval sits below zero**, so at a large score gap a single "
            "purchased human vote agrees with the crowd *less* often than the free machine "
            "label does. That is a hard stop, not a low priority."
        )

        cols = ["band", "pairs", "vlmAgreement", "crowdCeiling", "headroom", "headroomLo",
                "headroomHi", "remaining", "cost", "call"]
        renames = {"band": "percentile gap", "pairs": "pairs we own",
                   "vlmAgreement": "free label agrees", "crowdCeiling": "one rater agrees",
                   "headroom": "headroom (pts)", "headroomLo": "lo", "headroomHi": "hi",
                   "remaining": "pairs unbought", "cost": "cost to finish", "call": "verdict"}
        fmt = {"free label agrees": "{:.1%}", "one rater agrees": "{:.1%}",
               "headroom (pts)": "{:+.1f}", "lo": "{:+.1f}", "hi": "{:+.1f}",
               "pairs unbought": "{:,.0f}", "cost to finish": "${:,.0f}"}
        st.dataframe(bands[cols].rename(columns=renames).style.format(fmt),
                     use_container_width=True, hide_index=True)

        if LABEL_INFO_R4.exists():
            with st.expander("The same table on run 4's uniform pairs only (unbiased, smaller)"):
                st.caption(
                    "Runs 2–3 bought near-ties on purpose, so pooled bands are enriched for "
                    "pairs the machine found easy *within* their band, which flatters it. Run 4 "
                    "drew uniformly. Both agree on the boundary, which is why it is a decision "
                    "and not a hunch."
                )
                r4 = headroom_bands(LABEL_INFO_R4)
                keep = [c for c in cols if c not in ("remaining", "cost")]
                st.dataframe(r4[keep].rename(columns=renames).style.format(fmt),
                             use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("##### Two other methods that have to agree before we act on this")
        deltas = []
        for name, p, spend in (
            ("run 3 — 4,800 near-tie pairs",
             ROOT / "artifacts" / "panel-run-delta-v3" / "run-delta.json", 1286),
            ("run 4 — 2,500 uniform pairs",
             ROOT / "artifacts" / "panel-run-delta-v4" / "run-delta.json", 969),
        ):
            if p.exists():
                d = json.loads(p.read_text())
                gain = d["ladder"][-1]["deltaVsPrevious"] * 100
                deltas.append({"study": name, "spend": spend, "points gained": gain,
                               "points per $500": gain / spend * 500,
                               "$ per point": d["dollarsPerPoint"]})
        if deltas:
            st.dataframe(pd.DataFrame(deltas).style.format({
                "spend": "${:,.0f}", "points gained": "{:+.2f}",
                "points per $500": "{:.3f}", "$ per point": "${:,.0f}"}),
                use_container_width=True, hide_index=True)
            st.caption(
                "Same test, same month, same rate per vote — and a **27× difference** purely "
                "from which pairs were bought. The standing stop rule is 1 point per $500, so "
                "near-ties pass comfortably and uniform pairs fail by a factor of 20. **Apply "
                "the stop rule per pair type, never globally**, or these average into a number "
                "that describes neither."
            )
        v5 = ROOT / "artifacts" / "bt-refit-v5-panel" / "metrics.json"
        if v5.exists():
            by = json.loads(v5.read_text()).get("heldOut", {}).get("byStratum", {})
            rows = [{"band": s.replace("random-", ""), "before": v["vlmOnly"],
                     "after (+ human votes)": v["joint"],
                     "gain": (v["joint"] - v["vlmOnly"]) * 100}
                    for s, v in by.items() if s.startswith("random-")]
            if rows:
                order4 = ["0-2", "2-5", "5-10", "10-20", "20-45", "45-100"]
                rows.sort(key=lambda r: order4.index(r["band"]) if r["band"] in order4 else 99)
                st.caption(
                    "**Third method, completely different computation:** fit the ranking on half "
                    "the raters, predict the other half's withheld votes, and see where the "
                    "human money actually landed. It decays to *exactly zero* at 45–100."
                )
                fig = px.bar(pd.DataFrame(rows), x="band", y="gain",
                             labels={"gain": "points the panel added, out of sample",
                                     "band": "percentile gap"})
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
        run_dir = st.selectbox("Run", runs, format_func=lambda p: p.name, key="train_detail",
                               index=len(runs) - 1)
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
    return sorted((p.parent for p in base.glob("*/model_scores.csv")),
                  key=lambda p: run_order(p.name))


MODEL_LABELS: dict[str, str] = {
    "external-scut": "SCUT (ResNet-18)",
    "external-mebeauty": "MEBeauty (ResNet-18)",
    "train-v8-arcface-e2e-long": "FaceIQ v8 (ArcFace)",
    "train-v7-arcface-e2e": "FaceIQ v7 (ArcFace)",
    "train-v1": "FaceIQ v1 (ResNet-18)",
    "ensemble-v7-v1": "Ensemble v7+v1",
    # Panel-trained arms. v14 is the ship candidate: same label recipe as v13, refit at
    # val_fraction 0.2. v12/v13/v15 are measurement runs at 0.5 — comparable to each other,
    # trained on 13k pairs rather than 33k, so don't read their absolute scores as shippable.
    "train-v10-arcface-val50": "FaceIQ v10 (ArcFace, VLM labels, val 50%)",
    "train-v12-panel-soft": "FaceIQ v12 (panel vote share, val 50%)",
    "train-v13-panel-hard": "FaceIQ v13 (panel majority, val 50%)",
    "train-v14-panel-ship": "FaceIQ v14 (panel majority, ship split) ★",
    "train-v15-panel-weighted": "FaceIQ v15 (panel majority ×3 weight, val 50%)",
}


def model_label(run_name: str) -> str:
    return MODEL_LABELS.get(run_name, run_name)


@st.cache_data(show_spinner="Merging model scores...")
def load_merged_model_scores(run_names: tuple[str, ...]) -> pd.DataFrame:
    """Join per-face scores from multiple artifact dirs on faceId."""
    if not run_names:
        return pd.DataFrame()

    base: pd.DataFrame | None = None
    for name in run_names:
        path = ROOT / "artifacts" / name / "model_scores.csv"
        if not path.exists():
            continue
        sc = pd.read_csv(path)[["faceId", "modelScore"]].rename(
            columns={"modelScore": f"score__{name}"}
        )
        if base is None:
            sc_meta = pd.read_csv(path)
            keep = ["faceId", "gender", "theta"]
            base = sc_meta[keep].merge(sc, on="faceId", how="inner")
        else:
            base = base.merge(sc, on="faceId", how="inner")

    if base is None or base.empty:
        return pd.DataFrame()

    refits = find_bt_refits()
    if refits:
        ratings = pd.read_csv(refits[0] / "ratings.csv")
        extra = ratings[["faceId", "scoreOutOf10", "percentile", "labsOverallScore"]]
        base = base.merge(extra, on="faceId", how="left", suffixes=("", "_dup"))
        base = base.drop(columns=[c for c in base.columns if c.endswith("_dup")], errors="ignore")

    spread_path = ROOT / "artifacts" / "ensemble-spread-v1" / "spread.csv"
    if spread_path.exists():
        spread = pd.read_csv(spread_path)[["faceId", "spread", "meanScore"]]
        base = base.merge(spread, on="faceId", how="left")

    for name in run_names:
        col = f"score__{name}"
        if col not in base.columns:
            continue
        base[f"pct__{name}"] = base.groupby("gender")[col].rank(pct=True) * 100

    return base


def find_checkpoints() -> list[Path]:
    base = ROOT / "checkpoints"
    if not base.exists():
        return []
    return sorted(base.glob("*/best.pt"), key=lambda p: run_order(p.parent.name))


def find_ensemble_presets() -> dict[str, list[Path]]:
    """Artifact dirs whose eval.json lists 2+ member checkpoints."""
    presets: dict[str, list[Path]] = {}
    for eval_path in sorted((ROOT / "artifacts").glob("*/eval.json")):
        data = json.loads(eval_path.read_text())
        ckpts = data.get("checkpoints")
        if not ckpts or len(ckpts) < 2:
            continue
        presets[eval_path.parent.name] = [ROOT / c for c in ckpts]
    return presets


def zscore_raw(raw: float, cohort: pd.Series) -> float:
    mu = float(cohort.mean())
    sd = float(cohort.std(ddof=0)) or 1.0
    return (raw - mu) / sd


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
    cfg = TrainConfig.from_saved(ckpt["config"])
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

        # This whole tab compares model rank against BT rank, so a face the ranking never
        # scored has nothing to compare. That happens whenever the model scored all 3,000
        # export faces but was joined to a QC'd ranking (2,866), leaving theta empty.
        unranked = int(sc["theta"].isna().sum()) if "theta" in sc.columns else 0
        if unranked:
            sc = sc.dropna(subset=["theta"]).copy()
            st.caption(
                f"{unranked} scored face(s) are absent from this run's reference ranking "
                f"(QC exclusions) and are left out of the comparisons below."
            )

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
        gender_d = c1.selectbox("Gender", gender_choices(sc["gender"]), key="inspect_gender")
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


# ---------------------------------------------------------------- model gallery (visual cross-check)

with tab_gallery:
    st.caption(
        "Browse cohort faces by model score. Filter on a percentile range within gender "
        "(comparable across models on different scales), toggle which scores appear on "
        "each card, and eyeball whether the ranking looks right. SCUT / MEBeauty are "
        "research-only cross-checks — not production models."
    )

    score_runs = find_score_runs()
    if not score_runs:
        st.info("No model_scores.csv files yet. Run evaluate.py or external_crosscheck.py first.")
    else:
        all_names = [p.name for p in score_runs]
        default_names = [
            n for n in ("external-scut", "external-mebeauty", "train-v8-arcface-e2e-long")
            if n in all_names
        ] or all_names[-3:]

        exports = find_exports()
        export_dir = st.selectbox(
            "Export (photos)",
            exports,
            format_func=lambda p: p.name,
            key="gallery_export",
        ) if exports else None
        img_paths = (
            load_face_image_paths(str(export_dir)) if export_dir is not None else {}
        )

        c_models, c_gender = st.columns([3, 1])
        selected = c_models.multiselect(
            "Models to show on cards",
            all_names,
            default=default_names,
            format_func=model_label,
            key="gallery_models",
        )
        gender_g = c_gender.selectbox(
            "Gender", ["female", "male"], key="gallery_gender",
        )

        merged = load_merged_model_scores(tuple(selected))
        if merged.empty:
            st.warning("No overlapping faces across the selected models.")
        else:
            view = merged[merged["gender"] == gender_g].copy()
            filter_choices = [n for n in selected if f"pct__{n}" in view.columns]
            if not filter_choices:
                st.stop()

            c_filt, c_sort, c_grid = st.columns([2, 2, 1])
            filter_model = c_filt.selectbox(
                "Filter by model (percentile within gender)",
                filter_choices,
                format_func=model_label,
                key="gallery_filter_model",
            )
            pct_col = f"pct__{filter_model}"
            lo, hi = c_filt.slider(
                "Percentile range",
                0.0, 100.0, (0.0, 100.0), 1.0,
                key="gallery_pct_range",
            )
            sort_opts = {
                f"score__{filter_model}": f"{model_label(filter_model)} score (high first)",
                "theta": "BT theta (high first)",
                "scoreOutOf10": "BT /10 (high first)",
            }
            if "spread" in view.columns:
                sort_opts["spread"] = "Model disagreement (high first)"
            for n in selected:
                sort_opts[f"score__{n}"] = f"{model_label(n)} raw score"
            sort_col = c_sort.selectbox(
                "Sort by", list(sort_opts.keys()),
                format_func=lambda k: sort_opts[k],
                key="gallery_sort",
            )
            cols_per_row = c_grid.selectbox("Columns", [3, 4, 5, 6], index=1, key="gallery_cols")
            n_show = c_grid.slider("Show N faces", 5, 60, 20, 5, key="gallery_n")

            view = view[(view[pct_col] >= lo) & (view[pct_col] <= hi)]
            view = view.sort_values(sort_col, ascending=False).head(n_show).reset_index(drop=True)

            st.caption(
                f"Showing **{len(view)}** {gender_g} faces "
                f"({model_label(filter_model)} percentile {lo:.0f}–{hi:.0f}%) · "
                f"{len(merged[merged['gender'] == gender_g]):,} total in cohort"
            )

            if filter_choices:
                hist = merged[merged["gender"] == gender_g]
                st.plotly_chart(
                    px.histogram(
                        hist, x=f"pct__{filter_model}",
                        nbins=40,
                        title=f"{model_label(filter_model)} percentile distribution ({gender_g})",
                        labels={f"pct__{filter_model}": "percentile"},
                    ),
                    use_container_width=True,
                )

            for start in range(0, len(view), cols_per_row):
                cols = st.columns(cols_per_row)
                for col, (_, row) in zip(cols, view.iloc[start : start + cols_per_row].iterrows()):
                    with col:
                        fid = row["faceId"]
                        p = img_paths.get(fid)
                        if p and Path(p).exists():
                            st.image(p, use_container_width=True)
                        else:
                            st.caption("(image missing)")
                        lines = [
                            f"**BT** θ {row['theta']:.2f} · /10 {row.get('scoreOutOf10', float('nan')):.2f}"
                            if pd.notna(row.get("scoreOutOf10"))
                            else f"**BT** θ {row['theta']:.2f}",
                        ]
                        for name in selected:
                            sc = row.get(f"score__{name}")
                            pct = row.get(f"pct__{name}")
                            if pd.notna(sc):
                                lines.append(
                                    f"**{model_label(name)}** {sc:.2f} "
                                    f"({pct:.0f}th pct)"
                                    if pd.notna(pct)
                                    else f"**{model_label(name)}** {sc:.2f}"
                                )
                        if pd.notna(row.get("spread")):
                            lines.append(f"spread {row['spread']:.3f}")
                        st.markdown("  \n".join(lines))
                        with st.expander("faceId"):
                            st.code(fid)


# ---------------------------------------------------------------- composite (blend, measured)


def find_panel_runs() -> list[tuple[str, str]]:
    """(results dir, sample-meta.json) for every archived panel study on disk."""
    out = []
    for meta in sorted((ROOT / "labels").glob("*/sample-meta.json")):
        results = meta.parent / "results"
        if (results / "judgments.jsonl").exists():
            out.append((str(results), str(meta)))
    return out


def find_reject_lists() -> list[str]:
    return [str(p) for p in sorted((ROOT / "artifacts").glob("panel-run-*/reject-pids.txt"))]


@st.cache_data(show_spinner="Indexing export and discovering rating sources...")
def load_composite_context(export_dir: str):
    export = load_export(export_dir, verify_hashes=False)
    matchups = export.all_matchups()
    faces = export.faces()
    sources = discover_sources(ROOT, faces, matchups)
    dropped: set[str] = set()
    for p in (ROOT / "artifacts/face-qc-v1/exclude-faces.csv",
              ROOT / "artifacts/face-qc-v1/gender-fixes.csv"):
        if p.exists():
            with open(p, newline="") as fh:
                dropped.update(r["faceId"] for r in csv.DictReader(fh))
    by_index = {m.pair_index: m for m in matchups
                if m.face_a_id not in dropped and m.face_b_id not in dropped}
    gender_of = {fid: f.gender for fid, f in faces.items()}
    return sources, by_index, gender_of


@st.cache_data(show_spinner="Pooling human votes...")
def load_votes_cached(panels: tuple[tuple[str, str], ...], rejects: tuple[str, ...]):
    reject_ids: set[str] = set()
    for p in rejects:
        if Path(p).exists():
            reject_ids |= set(Path(p).read_text().split())
    return load_panel_votes([list(p) for p in panels], reject_ids)


@st.cache_data(show_spinner="Scoring the blend against human votes...")
def evaluate_blend_cached(export_dir: str, names: tuple[str, ...], weights: tuple[float, ...],
                          method: str, panels: tuple, rejects: tuple, ratings: str):
    sources, by_index, gender_of = load_composite_context(export_dir)
    chosen = [s for n in names for s in sources if s.name == n]
    votes = load_votes_cached(panels, rejects)
    with open(ratings, newline="") as fh:
        pct = {r["faceId"]: float(r["percentile"]) for r in csv.DictReader(fh)}
    rep = evaluate_blend(chosen, list(weights), gender_of, votes, by_index, pct,
                         method=method)
    return rep, pct


@st.cache_data(show_spinner="Loading per-face uncertainty...")
def load_uncertainty(refit_dir: str) -> pd.DataFrame:
    p = Path(refit_dir) / "uncertainty.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


with tab_composite:
    st.caption(
        "Blend several rating sources into one score — and **measure** the blend against "
        "real human votes rather than assuming an average must be better. Averaging only "
        "helps when the components' errors are independent, which is not a safe assumption "
        "here: the network trained on VLM labels, BT was fit on the same labels, and Labs "
        "is the formula being replaced. A blend can be worse than its best ingredient."
    )

    exports_c = find_exports()
    panels_c = find_panel_runs()
    if not exports_c:
        st.info("No exports on disk.")
    elif not panels_c:
        st.info("No archived panel studies under labels/*/results — nothing to score against.")
    else:
        export_c = st.selectbox("Export", exports_c, format_func=lambda p: p.name,
                                key="comp_export")
        sources_all, _, _ = load_composite_context(str(export_c))
        refits_c = [p for p in find_bt_refits()]
        ratings_c = st.selectbox(
            "Reference ranking (defines percentile-gap bands and the /10 column)",
            refits_c, format_func=lambda p: p.name,
            index=len(refits_c) - 1 if refits_c else 0, key="comp_ratings",
        )

        names_all = [s.name for s in sources_all]
        default_sel = [n for n in ("train-v13-panel-hard", "bt-refit-v2-qc") if n in names_all]
        chosen_names = st.multiselect(
            "Sources to blend", names_all, default=default_sel or names_all[:1],
            format_func=lambda n: next(
                (f"{s.label}  [{s.kind}]" for s in sources_all if s.name == n), n),
            key="comp_sources",
        )
        if not chosen_names:
            st.info("Pick at least one source.")
            st.stop()

        c_m, c_w = st.columns([1, 3])
        method_c = c_m.radio("Normalisation", list(METHODS), key="comp_method",
                             help="Both are computed within gender: male and female faces "
                                  "were never compared, so their scales share no origin.")
        weights_c = []
        wcols = c_w.columns(min(len(chosen_names), 4))
        for k, n in enumerate(chosen_names):
            weights_c.append(wcols[k % len(wcols)].number_input(
                f"weight — {n[:22]}", 0.0, 10.0, 1.0, 0.25, key=f"comp_w_{n}"))

        chosen_srcs = [s for n in chosen_names for s in sources_all if s.name == n]
        circ = [s for s in chosen_srcs if s.circular]
        no_leak_free = [s for s in chosen_srcs if s.val_faces is not None and not s.val_faces]
        if circ:
            st.warning(
                "**Circular source selected.** "
                + "; ".join(f"`{s.name}` {s.caveat}" for s in circ)
                + ". Its accuracy below is not evidence, and the blend inherits the problem."
            )
        if no_leak_free:
            st.error(
                "**No leak-free pairs available.** "
                + "; ".join(f"`{s.name}`: {s.caveat}" for s in no_leak_free)
            )

        rep, pct_ref = evaluate_blend_cached(
            str(export_c), tuple(chosen_names), tuple(weights_c), method_c,
            tuple(panels_c), tuple(find_reject_lists()), str(ratings_c / "ratings.csv"),
        )

        if not rep.keys:
            st.error(
                "No panel pair has both faces inside every selected component's validation "
                "split, so nothing can be scored without leakage. Drop the narrowest-split "
                "component (a `val_fraction` of 0.2 leaves only 600 faces) and retry."
            )
        else:
            st.divider()
            st.subheader("Measured against human votes")
            st.caption(
                f"**{len(rep.keys):,}** leak-free panel pairs · "
                f"**{rep.votes_scored:,.0f}** individual human votes · accuracy is the share "
                f"of those votes agreeing with each predictor's pick. Pairs are restricted to "
                f"the intersection of the components' validation splits."
            )

            tbl = pd.DataFrame([
                {"predictor": r["predictor"],
                 "agreement": r["accuracy"],
                 "vs ceiling": r["accuracy"] - rep.ceiling,
                 "votes": r["votes"],
                 "circular": "yes" if r["circular"] else ""}
                for r in sorted(rep.rows, key=lambda r: -r["accuracy"])
            ])
            m1, m2, m3 = st.columns(3)
            blend_acc = next(r["accuracy"] for r in rep.rows if r["name"] == "BLEND")
            m1.metric("Blend", f"{blend_acc:.2%}")
            m2.metric("Crowd ceiling (another rater)", f"{rep.ceiling:.2%}")
            m3.metric("Chance", "50.00%")
            st.dataframe(
                tbl.style.format({"agreement": "{:.2%}", "vs ceiling": "{:+.2%}",
                                  "votes": "{:,.0f}"}),
                use_container_width=True, hide_index=True,
            )
            st.caption(
                "The ceiling is how often the crowd majority predicts one of its own members "
                "(leave-one-out). It is the practical maximum on these pairs — they were "
                "selected to be close, so people genuinely disagree on them."
            )

            if rep.versus_best:
                v = rep.versus_best
                verdict = ("**beats** it" if v["lo"] > 0 else
                           "**loses to** it" if v["hi"] < 0 else
                           "is **indistinguishable** from it")
                msg = (
                    f"Blend minus its best honest component (`{rep.best_component}`): "
                    f"**{v['delta']:+.2%}**, 95% CI [{v['lo']:+.2%}, {v['hi']:+.2%}] "
                    f"from a paired bootstrap over {v['pairs']:,} pairs — the blend {verdict}."
                )
                (st.success if v["lo"] > 0 else st.info)(msg)
                st.caption(
                    "Paired, because both predictors are scored on the same pairs: comparing "
                    "two independent confidence intervals would overstate the uncertainty of "
                    "their difference. Resampling is over pairs, not votes — the ~12 votes on "
                    "one pair are not independent observations."
                )

            band_rows = []
            for name, bands in rep.bands.items():
                label = next((r["predictor"] for r in rep.rows if r["name"] == name), name)
                for b in bands:
                    band_rows.append({"band": b["band"], "predictor": label,
                                      "accuracy": b["accuracy"] * 100, "pairs": b["pairs"]})
            if band_rows:
                bdf = pd.DataFrame(band_rows)
                fig = px.bar(
                    bdf, x="band", y="accuracy", color="predictor", barmode="group",
                    title="Agreement with human votes, by percentile gap between the two faces",
                    labels={"accuracy": "agrees with human votes (%)",
                            "band": "percentile gap"},
                    hover_data=["pairs"],
                )
                fig.add_hline(y=rep.ceiling * 100, line_dash="dot",
                              annotation_text=f"crowd ceiling {rep.ceiling:.1%}")
                fig.add_hline(y=50, line_dash="dash", annotation_text="chance")
                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    "Ordering is near chance for the closest pairs and near-perfect for the "
                    "widest. That shape is the product story: the rating orders people who "
                    "differ, and cannot resolve people who do not."
                )

            _, _, gender_of_c = load_composite_context(str(export_c))
            blended_c = blend(chosen_srcs, list(weights_c), gender_of_c, method_c)
            # Percentile-average -> the same /10 ladder BT uses, so the blend's score is
            # directly comparable to the ranking of record rather than a new unit.
            blend_pct_c = normalise(blended_c, gender_of_c, "percentile")
            blend10_c = {f: score_out_of_10(p / 100.0) for f, p in blend_pct_c.items()}

            st.divider()
            st.subheader("The blend's /10 scores")
            b10 = pd.DataFrame({
                "faceId": list(blend10_c),
                "gender": [gender_of_c.get(f, "?") for f in blend10_c],
                "blendScoreOutOf10": [blend10_c[f] for f in blend10_c],
            })
            q = b10["blendScoreOutOf10"]
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Median", f"{q.median():.2f}")
            s2.metric("90th pct", f"{q.quantile(0.9):.2f}")
            s3.metric("99th pct", f"{q.quantile(0.99):.2f}")
            s4.metric("Max", f"{q.max():.2f}")
            st.info(
                "**The /10 range is set by the calibration anchors, not by the model.** "
                "Percentile-averaging maps onto the same ladder as BT (50th = 5.0, "
                "90th = 7.0, 99th = 8.0), so every blend has an *identical* /10 "
                "distribution — changing sources reshuffles **who** gets a 7.0, it cannot "
                "make the best face score 9.5. Only re-anchoring can do that, and that is "
                "the deferred anchor-ladder decision, not a modelling one."
            )
            st.plotly_chart(
                px.histogram(b10, x="blendScoreOutOf10", color="gender", nbins=45,
                             barmode="overlay", title="Blend /10 distribution",
                             labels={"blendScoreOutOf10": "/10"}),
                use_container_width=True,
            )

            with st.expander("Rank agreement vs BT (secondary diagnostic)"):
                st.caption(
                    "Correlation with BT says how much a blend *moves* the ranking, not "
                    "whether it moves it in the right direction. Agreement with human votes "
                    "above is the decisive number; this is here to show scale of change."
                )
                bt_theta = next((s.scores for s in sources_all
                                 if s.name == ratings_c.name), {})
                common_c = sorted(set(blended_c) & set(bt_theta))
                if common_c:
                    rho_c, _ = spearmanr([blended_c[f] for f in common_c],
                                         [bt_theta[f] for f in common_c])
                    st.metric(f"Spearman rho vs {ratings_c.name}", f"{rho_c:.4f}",
                              help=f"{len(common_c):,} faces in common")

            caveat_rows = [{"source": s.label, "kind": s.kind,
                            "leak-free faces": ("all (never trained on our cohort)"
                                                if s.val_faces is None
                                                else f"{len(s.val_faces):,}"),
                            "caveat": s.caveat or "—"}
                           for s in chosen_srcs]
            with st.expander("Source caveats", expanded=bool(circ)):
                st.dataframe(pd.DataFrame(caveat_rows), use_container_width=True,
                             hide_index=True)

            # ---- visual validation -------------------------------------------
            st.divider()
            st.subheader("Visual validation")
            unc = load_uncertainty(str(ratings_c))
            if unc.empty:
                st.info(
                    f"No `uncertainty.csv` in `{ratings_c.name}`. Generate per-face "
                    "confidence intervals with `python scripts/bt_uncertainty.py "
                    f"--ratings {ratings_c}/ratings.csv --out {ratings_c}`."
                )
            else:
                img_paths_c = load_face_image_paths(str(export_c))
                gender_c = st.selectbox("Gender", gender_choices(unc["gender"]),
                                        key="comp_gender")
                gview = unc[unc["gender"] == gender_c].copy()

                mode_c = st.radio(
                    "View", ["Ranked by score band", "Most suspicious"],
                    horizontal=True, key="comp_visual_mode",
                    help="'Most suspicious' surfaces the faces worth your eyes: the ones "
                         "the sources disagree about, or that the data cannot place.",
                )

                gview["blendScoreOutOf10"] = gview["faceId"].map(blend10_c)
                gview["blendPct"] = gview["faceId"].map(blend_pct_c)

                if mode_c == "Ranked by score band":
                    st.caption(
                        "Faces in one /10 band, each card showing its interval. If two "
                        "faces' intervals overlap, the data does not order them — that is "
                        "the honest reading, not a bug."
                    )
                    which10 = st.radio(
                        "Score to band and sort by", ["Blend /10", "BT /10"],
                        horizontal=True, key="comp_which10",
                    )
                    col10 = ("blendScoreOutOf10" if which10 == "Blend /10"
                             else "scoreOutOf10")
                    lo_b, hi_b = st.slider("Score band (/10)", 1.0, 10.0, (7.0, 10.0), 0.25,
                                           key="comp_band")
                    sel_v = gview.dropna(subset=[col10])
                    sel_v = sel_v[(sel_v[col10] >= lo_b) & (sel_v[col10] <= hi_b)]
                    sel_v = sel_v.sort_values(col10, ascending=False)
                else:
                    st.caption(
                        "Ranked by how much the sources disagree about a face, or how little "
                        "the data pins it down. These are the faces to check by eye."
                    )
                    susp_by = st.selectbox(
                        "Rank by",
                        ["BT vs blend disagreement", "BT vs Labs disagreement",
                         "Widest interval", "Not identified (undefeated)"],
                        key="comp_susp",
                    )
                    sel_v = gview.copy()
                    sel_v["btPct"] = sel_v["percentile"] * 100
                    labs_src = next((s for s in sources_all if s.kind == "labs"), None)
                    if labs_src:
                        sel_v["labsPct"] = sel_v["faceId"].map(
                            normalise(labs_src.scores, gender_of_c, "percentile"))
                    if susp_by == "BT vs blend disagreement":
                        sel_v["gap"] = (sel_v["blendPct"] - sel_v["btPct"]).abs()
                        sel_v = sel_v.dropna(subset=["gap"]).sort_values("gap", ascending=False)
                    elif susp_by == "BT vs Labs disagreement":
                        if "labsPct" not in sel_v.columns:
                            st.warning("No Labs scores in this export.")
                            sel_v = sel_v.head(0)
                        else:
                            sel_v["gap"] = (sel_v["labsPct"] - sel_v["btPct"]).abs()
                            sel_v = sel_v.dropna(subset=["gap"]).sort_values(
                                "gap", ascending=False)
                    elif susp_by == "Widest interval":
                        wcol = ("resamplePctWidth" if "resamplePctWidth" in sel_v.columns
                                else "relabelPctWidth")
                        sel_v = sel_v.sort_values(wcol, ascending=False)
                    else:
                        sel_v = sel_v[sel_v.get("notIdentified", 0) == 1].sort_values(
                            "theta", ascending=False)

                cc1, cc2 = st.columns(2)
                ncols_c = cc1.selectbox("Columns", [3, 4, 5, 6], index=1, key="comp_cols")
                nshow_c = cc2.slider("Show N faces", 4, 48, 12, 4, key="comp_n")
                sel_v = sel_v.head(nshow_c).reset_index(drop=True)
                st.caption(f"Showing **{len(sel_v)}** of {len(gview):,} {gender_c} faces.")

                for start in range(0, len(sel_v), ncols_c):
                    cols_v = st.columns(ncols_c)
                    chunk = sel_v.iloc[start:start + ncols_c]
                    for col_v, (_, r) in zip(cols_v, chunk.iterrows()):
                        with col_v:
                            p = img_paths_c.get(r["faceId"])
                            if p and Path(p).exists():
                                st.image(p, use_container_width=True)
                            else:
                                st.caption("(image missing)")
                            lines = []
                            if pd.notna(r.get("blendScoreOutOf10")):
                                lines.append(f"**blend {r['blendScoreOutOf10']:.2f}/10** · "
                                             f"{r['blendPct']:.0f}th pct")
                            lines.append(f"BT {r['scoreOutOf10']:.2f}/10 · "
                                         f"{r['percentile'] * 100:.1f}th pct")
                            if pd.notna(r.get("resampleScoreLo")):
                                lines.append(
                                    f"BT CI {r['resampleScoreLo']:.2f}–"
                                    f"{r['resampleScoreHi']:.2f} "
                                    f"({r['resamplePctWidth']:.0f} pct pts)")
                            lines.append(
                                f"{int(r['wins'])}W-{int(r['losses'])}L-{int(r['ties'])}T · "
                                f"{r['effectiveComparisons']:.1f} eff. comparisons")
                            if pd.notna(r.get("labsPct")):
                                lines.append(f"Labs {r['labsPct']:.0f}th pct "
                                             f"(d{int(r['decileBin'])})")
                            if isinstance(r.get("flags"), str) and r["flags"]:
                                lines.append(f":orange[{r['flags']}]")
                            st.markdown("  \n".join(lines))
                            with st.expander("faceId"):
                                st.code(r["faceId"])


# ---------------------------------------------------------------- inference

with tab_infer:
    st.caption(
        "Score new, unseen photos with a trained checkpoint (or ensemble) and see where they "
        "land in the ranked cohort. Sanity-check tool — not the production anchor-ladder."
    )
    ckpts = find_checkpoints()
    ensembles = find_ensemble_presets()
    if not ckpts:
        st.info("No checkpoints found under checkpoints/*/best.pt.")
    else:
        infer_mode = st.radio(
            "Scorer", ["Single checkpoint", "Ensemble"],
            horizontal=True, key="infer_mode",
        )
        score_runs = find_score_runs()
        member_z_cohorts: list[pd.Series] = []

        if infer_mode == "Single checkpoint":
            # Default to the checkpoint measured best at PLACEMENT, not the newest one. The v12-v21
            # arms all ran at val_fraction 0.5 to hold out panel pairs, so they trained on 13k pairs
            # against v14's 33k; scored on the same held-out faces v14 places at 0.34/0.28 median
            # /10 error vs 0.44-0.47. See scripts/placement_by_checkpoint.py and pipeline doc §3d.
            shippable = ("train-v14-panel-ship", "train-v22-ship-0.2")
            default_ckpt = next(
                (i for i, p in enumerate(ckpts) if p.parent.name == shippable[0]),
                len(ckpts) - 1,
            )
            ckpt_path = st.selectbox(
                "Checkpoint", ckpts, format_func=lambda p: p.parent.name,
                index=default_ckpt, key="infer_ckpt",
            )
            if ckpt_path.parent.name not in shippable:
                st.caption(
                    f"⚠️ Use **{shippable[0]}** or **{shippable[1]}** — they are a dead tie with "
                    f"each other (81.7% vs human votes, +0.00 [−0.96, +0.96]) and both far ahead "
                    f"of everything else on placement: 0.34 median /10 error and 67–69% tier-exact "
                    f"versus 0.44–0.47 and 55–57%. The other arms ran at `val_fraction 0.5` to hold "
                    f"out panel pairs, so they trained on 13k pairs instead of 33k — controlled "
                    f"label-recipe comparisons, never production candidates. "
                    f"`scripts/placement_by_checkpoint.py --common-val`."
                )
            member_ckpts: list[Path] = [ckpt_path]
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
        else:
            if not ensembles:
                st.info(
                    "No ensemble presets found. Run `scripts/ensemble_eval.py` first "
                    "(writes artifacts/<name>/eval.json with a checkpoints list)."
                )
                st.stop()
            ens_names = sorted(ensembles)
            ens_name = st.selectbox(
                "Ensemble", ens_names, format_func=model_label,
                index=ens_names.index("ensemble-v7-v1") if "ensemble-v7-v1" in ens_names else 0,
                key="infer_ensemble",
            )
            member_ckpts = ensembles[ens_name]
            st.caption(
                "Members: "
                + ", ".join(model_label(p.parent.name) for p in member_ckpts)
            )
            ctx_dir = ROOT / "artifacts" / ens_name
            if not (ctx_dir / "model_scores.csv").exists():
                st.error(f"Missing {ctx_dir / 'model_scores.csv'} — re-run ensemble_eval.py.")
                st.stop()
            missing = [
                p.parent.name for p in member_ckpts
                if not (ROOT / "artifacts" / p.parent.name / "model_scores.csv").exists()
            ]
            if missing:
                st.error(
                    "Missing per-face scores for ensemble members: "
                    + ", ".join(missing)
                    + ". Run scripts/evaluate.py --ratings on each first."
                )
                st.stop()
            member_z_cohorts = [
                pd.read_csv(ROOT / "artifacts" / p.parent.name / "model_scores.csv")["modelScore"]
                for p in member_ckpts
            ]

        gender_ctx = st.selectbox("Compare against", ["female", "male"], key="infer_gender")
        normalize = st.checkbox(
            "Normalize like faceiq-labs (MediaPipe eye-level crop, same pipeline that produced "
            "the cohort training photos — recommended)",
            value=True,
        )

        # ---- reference set ----------------------------------------------------------
        # Built before the uploader so it can be inspected on its own. The comparator was trained
        # pairwise, so the honest read is "which cohort faces of known theta does this face beat",
        # not "what number did the head emit". See src/faceiq_pref/placement.py and log §5.8.
        ref_ids: list[str] = []
        if ctx_dir is not None:
            from faceiq_pref.placement import (
                ANCHORS_TOP10,
                DEFAULT_AGREEMENT,
                agreement_vs_tier_below,
                place,
                stratified_reference_ids,
                tier_edges,
                tier_of,
            )

            to_ten_top10 = make_scorer(ANCHORS_TOP10)
            sc = pd.read_csv(ctx_dir / "model_scores.csv")
            cohort = sc[sc["gender"] == gender_ctx].sort_values("modelScore").reset_index(drop=True)
            refits = find_bt_refits()
            exports = find_exports()
            img_paths = load_face_image_paths(str(exports[0])) if exports else {}

            rank_names = [p.name for p in refits]
            pl_cols = st.columns([2, 1, 1])
            rank_pick = pl_cols[0].selectbox(
                "Reference ranking (supplies each reference face's θ)", rank_names,
                index=rank_names.index("bt-refit-v5-panel")
                if "bt-refit-v5-panel" in rank_names else len(rank_names) - 1,
                key="infer_rank",
            )
            n_refs = pl_cols[1].selectbox(
                "Reference faces", [50, 200, "all"], index=1, key="infer_nrefs",
                help="200 is the measured operating point — past it only the standard error "
                     "shrinks, and it is already ~5x below the disagreement band.",
            )
            agreement = pl_cols[2].select_slider(
                "Band = this many people agree", options=[0.55, 0.60, DEFAULT_AGREEMENT, 0.75],
                value=DEFAULT_AGREEMENT, format_func=lambda v: f"{v:.0%}", key="infer_agree",
            )

            # Same file the placement uses, deliberately. Reading the alphabetically-first refit
            # here (bt-refit-v1) made the k-NN cross-check differ from the placed score by a third,
            # invisible reason on top of the estimator and the ladder.
            rank_df = pd.read_csv(ROOT / "artifacts" / rank_pick / "ratings.csv")
            ratings = rank_df if "scoreOutOf10" in rank_df.columns else None
            rank_g = rank_df[rank_df["gender"] == gender_ctx]
            theta_of = dict(zip(rank_g["faceId"], rank_g["theta"]))
            pct_of = dict(zip(rank_g["faceId"], rank_g["percentile"]))
            score_of = dict(zip(cohort["faceId"], cohort["modelScore"]))
            population_thetas = list(theta_of.values())

            usable = {f: p for f, p in pct_of.items() if f in score_of}
            ref_ids = (list(usable) if n_refs == "all"
                       else stratified_reference_ids(usable, n=int(n_refs)))
            ref_thetas = [theta_of[f] for f in ref_ids]
            ref_scores = [score_of[f] for f in ref_ids]
            st.caption(
                f"{len(ref_ids):,} reference faces from **{rank_pick}** ({gender_ctx}), θ known; "
                f"percentile is read off all {len(population_thetas):,} cohort faces, because a "
                f"tier-stratified reference set is deliberately *not* population-representative. "
                f"Tier edges: " + ", ".join(f"{e:.1f}" for e in tier_edges())
            )

            # ---- post-hoc second opinion: blend the Labs deterministic score in ----------
            # Applied AFTER the MLE, on the /10 scale, so it never touches the reference-set fit.
            # Off by default and it should stay off: scripts/labs_composite_eval.py shows the
            # blend improving agreement with BT while flat-to-worse against real human votes.
            labs_w = 1.0
            labs_ladder: list[float] = []
            labs_of: dict[str, float] = {}
            comp_json = ctx_dir / "labs-composite.json"
            with st.expander("Second opinion: blend in the Labs deterministic score"):
                st.markdown(
                    "Post-hoc by construction — the reference-set MLE runs first and untouched, "
                    "then `w · placed + (1−w) · labs`, with the Labs score rank-normalised onto "
                    "our /10 through the same anchor curve (the two scales are not otherwise "
                    "comparable). Geometric weighting was measured too and lands within 0.003 of "
                    "arithmetic, because both scores sit in a narrow range."
                )
                if comp_json.exists():
                    cj = json.loads(comp_json.read_text())
                    st.error(
                        f"**Measured verdict: do not blend.** The two sources make almost "
                        f"perfectly uncorrelated errors (r = {cj['errorCorrelation']:+.3f}), which "
                        f"is normally the condition for a blend to help — and against **BT /10** "
                        f"it does, clearly and significantly. Against **real human votes** it is "
                        f"flat to slightly worse at every weight. BT is the proxy; the panel is "
                        f"the target. A change that moves the proxy and not the target is fitting "
                        f"the proxy's error, so this stays off."
                    )
                    st.dataframe(pd.DataFrame([
                        {"w on comparator": f"{r['w']:.0%}",
                         "BT median err": round(b["median"], 3),
                         "BT tier exact": f"{b['tierExact']:.1%}",
                         "vs panel majority": f"{r['vsMajority']:.1%}",
                         "vs panel votes": f"{r['vsVotes']:.1%}"}
                        for b, r in zip(cj["vsBT"], cj["vsPanel"])
                    ]), hide_index=True, use_container_width=True)
                else:
                    st.caption(
                        f"No measurement for this run yet — "
                        f"`python scripts/labs_composite_eval.py --run {ctx_dir.name}`"
                    )
                labs_w = st.slider(
                    "Weight on the comparator (1.00 = placement only)", 0.5, 1.0, 1.0, 0.05,
                    key="infer_labs_w",
                    help="Left of 1.00 shows the composite next to the placement-only read for "
                         "every uploaded face. Kept adjustable because the EBM will slot into "
                         "exactly this position with a better second opinion.",
                )
                if labs_w < 1.0:
                    exports_l = find_exports()
                    if exports_l:
                        fj = Path(exports_l[0]) / "faces.jsonl"
                        rowsl = [json.loads(ln) for ln in fj.read_text().splitlines()]
                        labs_of = {r["faceId"]: r["labsOverallScore"] for r in rowsl
                                   if r.get("labsOverallScore") is not None}
                        labs_ladder = sorted(labs_of[f] for f in usable if f in labs_of)
                        st.caption(
                            f"Labs ladder built from {len(labs_ladder):,} {gender_ctx} cohort "
                            f"faces. Uploads have no Labs score, so enter one per photo below to "
                            f"see the composite."
                        )

            with st.expander(f"Browse the whole {gender_ctx} reference set ({len(ref_ids)} faces)"):
                st.caption(
                    "The reference set is per-gender: a face is only ever compared against "
                    "references of its own gender, so there are two independent sets and this "
                    "shows one at a time. Faces are laid out in **θ order**, so reading left to "
                    "right down the page should look like increasing attractiveness — where it "
                    "does not, that is either real ranking noise or a pair the scale genuinely "
                    "cannot resolve, and the tier headings tell you which."
                )
                to_ten_ref = to_ten_top10
                ref_tbl = pd.DataFrame({
                    "faceId": ref_ids,
                    "theta": [theta_of[f] for f in ref_ids],
                    "percentile": [pct_of[f] for f in ref_ids],
                    "modelScore": [score_of[f] for f in ref_ids],
                })
                ref_tbl["scoreOutOf10"] = ref_tbl["percentile"].map(to_ten_ref)
                ref_tbl["tier"] = ref_tbl["scoreOutOf10"].map(lambda v: tier_of(v)[0])
                ref_tbl = ref_tbl.sort_values("theta").reset_index(drop=True)

                # How many references per tier, and how thin the cohort itself is up there. This is
                # the table that answers "do we have references at the top" -- tier stratification
                # fills to availability, so the top tiers are capped by the cohort, not by quota.
                pop_tier = pd.Series(
                    [tier_of(to_ten_ref(p))[0] for p in pct_of.values()]
                ).value_counts()
                counts = pd.DataFrame({
                    "tier": range(1, 8),
                    "/10 range": [f"{tier_edges()[i]:.1f}–{tier_edges()[i + 1]:.1f}"
                                  for i in range(7)],
                    "references": [int((ref_tbl["tier"] == t).sum()) for t in range(1, 8)],
                    f"{gender_ctx} cohort": [int(pop_tier.get(t, 0)) for t in range(1, 8)],
                })
                counts["share of references"] = counts["references"] / max(1, len(ref_tbl))
                st.dataframe(
                    counts.style.format({"share of references": "{:.1%}"}),
                    use_container_width=True, hide_index=True,
                )
                st.caption(
                    "If a top tier shows fewer references than its quota, the cohort has run out "
                    "of faces there — not a sampling bug. That thinness is exactly why a user in "
                    "the top 1% can beat every reference and come back unbounded."
                )

                if not img_paths:
                    st.info("No export found, so no face images to show — table only.")
                else:
                    show_tier = st.multiselect(
                        "Tiers to show", list(range(1, 8)),
                        default=list(range(1, 8)), key="ref_tiers",
                    )
                    per_row = 8
                    for t in sorted(show_tier):
                        rows = ref_tbl[ref_tbl["tier"] == t]
                        if rows.empty:
                            continue
                        st.markdown(
                            f"**Tier {t}** ({tier_edges()[t - 1]:.1f}–{tier_edges()[t]:.1f}) — "
                            f"{len(rows)} references"
                        )
                        for start in range(0, len(rows), per_row):
                            chunk = rows.iloc[start:start + per_row]
                            for col, (_, r) in zip(st.columns(per_row), chunk.iterrows()):
                                p = img_paths.get(r["faceId"])
                                if p and Path(p).exists():
                                    col.image(str(p), use_container_width=True)
                                col.caption(
                                    f"{r['scoreOutOf10']:.1f} · p{r['percentile'] * 100:.0f} · "
                                    f"θ{r['theta']:+.2f}"
                                )

        uploads = st.file_uploader(
            "Photos", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True,
        )
        if uploads and ctx_dir is not None:
            import torch
            from PIL import Image

            use_norm = normalize and normalizer_available()
            if normalize and not use_norm:
                st.warning("mediapipe not available (`uv pip install mediapipe`) — scoring uncropped images.")

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

                if infer_mode == "Ensemble":
                    z_parts: list[float] = []
                    member_raw: list[tuple[str, float, float]] = []
                    for ckpt, z_cohort in zip(member_ckpts, member_z_cohorts):
                        scorer, _cfg, tf = load_scorer_cached(str(ckpt))
                        with torch.no_grad():
                            raw = scorer(tf(shown).unsqueeze(0)).item()
                        z = zscore_raw(raw, z_cohort)
                        z_parts.append(z)
                        member_raw.append((ckpt.parent.name, raw, z))
                    s = sum(z_parts) / len(z_parts)
                else:
                    scorer, _cfg, tf = load_scorer_cached(str(member_ckpts[0]))
                    with torch.no_grad():
                        s = scorer(tf(shown).unsqueeze(0)).item()
                    member_raw = []

                pct = float((cohort["modelScore"] < s).mean())
                c_img, c_res = st.columns([1, 3])
                with c_img:
                    st.image(shown, use_container_width=True)
                    st.caption(up.name)
                with c_res:
                    score_label = "Ensemble z-score" if infer_mode == "Ensemble" else "Model score"
                    st.metric(
                        "Cohort percentile",
                        f"{pct:.1%}",
                        help=f"{score_label} {s:.3f} vs {len(cohort)} {gender_ctx} cohort faces",
                    )
                    if ratings is not None:
                        # approximate /10 from the 5 nearest cohort faces by model score
                        joined = cohort.merge(
                            ratings[["faceId", "scoreOutOf10"]], on="faceId", how="inner"
                        )
                        joined["dist"] = (joined["modelScore"] - s).abs()
                        nearest = joined.nsmallest(5, "dist")
                        st.metric(
                            "Cross-check /10 (5 nearest cohort faces)",
                            f"{nearest['scoreOutOf10'].mean():.2f}",
                            help=f"Sanity check only, and it will not match the placed /10 below. "
                                 f"It averages the pre-registered scoreOutOf10 of the 5 "
                                 f"{rank_pick} faces with the closest comparator score — a "
                                 f"different estimator (k-NN, not the reference-set MLE) on a "
                                 f"different anchor ladder (pre-registered top≈9, not top=10). A "
                                 f"gap of a few tenths is expected; more than ~1 point means the "
                                 f"placement is fighting its own k-NN neighbourhood and is worth "
                                 f"looking at.",
                        )

                    # ---- the production read: band + tier from reference-set placement ----
                    if ref_ids:
                        pl = place(s, ref_scores, ref_thetas,
                                   population_thetas=population_thetas, agreement=agreement)
                        m1, m2, m3 = st.columns(3)
                        m1.metric(f"Tier {pl.tier} of 7",
                                  f"{pl.tier_low:.1f} – {pl.tier_high:.1f}",
                                  help="Tier width 1.29 /10 against a measured resolution limit of "
                                       "1.33, so differences inside a tier are below what people "
                                       "agree on.")
                        m2.metric("Band", f"{pl.band_low:.1f} – {pl.band_high:.1f}",
                                  help=f"±{pl.band_half:.2f} = quadrature of the disagreement "
                                       f"width and this face's own se(θ)={pl.se:.2f}")
                        m3.metric("Point /10 (placed)", f"{pl.score_ten:.2f}",
                                  help=f"θ {pl.theta:+.2f}, percentile {pl.percentile:.1%}, "
                                       f"beat {pl.wins} of {pl.references} references. Reference-set "
                                       f"MLE on the top-10 anchor ladder — not the same estimator "
                                       f"or ladder as the k-NN figure above, so they differ.")
                        below = agreement_vs_tier_below(pl.score_ten)
                        tail = (f"About **{below[1]:.0%} of people** would place you above a "
                                f"typical **Tier {below[0]}** face."
                                if below else
                                "This is the lowest tier, so there is no tier below to compare to.")
                        st.markdown(
                            f"> **{pl.score_ten:.1f} — Tier {pl.tier} of 7.** {tail}  \n"
                            f"> You place above **{pl.percentile:.0%}** of the {gender_ctx} cohort."
                        )
                        if not pl.bounded:
                            st.warning(
                                "This face beat (or lost to) **every** reference, so the "
                                "likelihood has no finite maximum — the score is capped at the "
                                "extreme reference plus a margin and is a lower/upper bound, not "
                                "an estimate. Expected for roughly the top 1%; use more references "
                                "or read it as 'at least this'."
                            )

                        # ---- side by side with the Labs composite, when asked for ----
                        if labs_w < 1.0 and labs_ladder:
                            raw = st.number_input(
                                "Labs overall_score for this photo",
                                min_value=0.0, max_value=10.0, value=0.0, step=0.05,
                                key=f"labs_{up.name}",
                                help="From the faceiq-labs app. Leave at 0 to skip — an upload "
                                     "has no Labs score until that pipeline runs on it.",
                            )
                            if raw > 0:
                                lt = to_ten_top10(
                                    float(np.searchsorted(labs_ladder, raw)) / len(labs_ladder)
                                )
                                blended = labs_w * pl.score_ten + (1 - labs_w) * lt
                                b1, b2, b3 = st.columns(3)
                                b1.metric("Placement only", f"{pl.score_ten:.2f}",
                                          help=f"Tier {pl.tier}")
                                b2.metric("Labs, on our ladder", f"{lt:.2f}",
                                          help=f"raw {raw:.2f} → rank {np.searchsorted(labs_ladder, raw) / len(labs_ladder):.1%} "
                                               f"of the {gender_ctx} cohort → /10")
                                b3.metric(
                                    f"Composite ({labs_w:.0%} / {1 - labs_w:.0%})",
                                    f"{blended:.2f}", delta=f"{blended - pl.score_ten:+.2f}",
                                    help=f"Tier {tier_of(blended)[0]}. Shown for comparison; the "
                                         f"measured verdict is to ship the placement-only number.",
                                )
                    if member_raw:
                        with st.expander("Per-member scores"):
                            for name, raw, z in member_raw:
                                st.write(f"**{model_label(name)}** — raw {raw:.3f}, z {z:.3f}")

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

    # ---- validate on faces from outside the cohort --------------------------------
    # Every accuracy number in this repo is measured against BT theta fitted on the same 3,000
    # faces the comparator trained on. That is honest about new *comparisons* and silent about new
    # *faces from a different source*, which is all production ever sees. This is the only surface
    # that tests the latter, so its verdict outranks anything on the Training runs tab.
    st.subheader("Validate on unseen faces")
    val_sets = sorted(p.name for p in (ROOT / "data" / "validation").glob("*")
                      if (p / "pairs.json").exists())
    if not val_sets:
        st.info(
            "**No validation set yet.** Every accuracy number in this repo is measured against BT θ "
            "fitted on the same 3,000 faces the comparator trained on. That is honest about new "
            "*comparisons* and silent about new **faces from a different source**, which is all "
            "production ever sees. This is the only surface that tests the latter.\n\n"
            "**What you do here:** judge ~300 pairs of unseen faces by clicking the more attractive "
            "one, without seeing any score. That produces an independent human ordering. The report "
            "then asks whether the system reproduces it. You are not grading faces and not checking "
            "whether scores look right."
        )
        st.markdown(
            "| step | where |\n|---|---|\n"
            "| 1. Put 50–100 photos, one clear front-facing face each, from **anywhere but "
            "faceiq-labs**, in `data/validation/<name>/photos/` | filesystem |\n"
            "| 2. `make-pairs` builds the queue and secretly re-asks ~10% flipped | terminal |\n"
            "| 3. Judge them here (~20 min) | this tab |\n"
            "| 4. `score` runs the photos through the production path, never seeing your clicks "
            "| terminal |\n"
            "| 5. `report` prints the three verdicts | terminal |\n"
        )
        st.code(
            "python scripts/validate_placement.py make-pairs --set set-1 --pairs 300 --repeat 0.1",
            language="bash",
        )
        with st.expander("What the three tests tell you, and why they are reported separately"):
            st.markdown(
                "They fail for different reasons and have **different fixes** — one combined number "
                "is how you spend a month on the model to fix an arithmetic problem.\n\n"
                "**1. Ordering.** Does the placed score reproduce your pairwise choices? Scored as a "
                "share of *your own* repeat-pair consistency, not against 100% — if you only agree "
                "with yourself 80% of the time, 80% is the target. A failure means the comparator "
                "does not generalise off-cohort, and the fix is model work. This would be the first "
                "evidence that more labels are not the answer.\n\n"
                "**2. Band.** The band claims pairs closer than ~1.33 /10 are near coin-flips and "
                "pairs further apart are reliable. This checks both, and whether the predicted "
                "agreement matches what you actually did. A failure means `T = 1.920` was fitted on "
                "cohort pairs and does not transfer — the fix is refitting `T` on this data, which "
                "is cheap. **This has never been checked anywhere.**\n\n"
                "**3. Calibration.** Optional, and needs you to also type a *range* (\"6 to 7\") per "
                "photo. It splits the error into a **shift** (everything placed 1.2 too low → the "
                "anchor ladder is wrong, free to fix) and a **spread** (each face wrong in a "
                "different direction → the model is wrong, expensive to fix). Those two demand "
                "opposite responses, which is exactly why they must not be averaged together."
            )
    else:
        vs = st.selectbox("Validation set", val_sets, key="val_set")
        vdir = ROOT / "data" / "validation" / vs
        queue = json.loads((vdir / "pairs.json").read_text())["pairs"]
        jfile = vdir / "judgments.jsonl"
        done = [json.loads(ln) for ln in jfile.read_text().splitlines() if ln] \
            if jfile.exists() else []

        st.caption(
            f"**{len(done)} of {len(queue)} judged.** You are judging *pairs*, not assigning "
            "scores, for the same reason the panel does: an ordering needs no shared scale, so "
            "your judgements can be compared to the model's without either of you having to agree "
            "on what a 7 is. Some pairs repeat with the sides flipped — that is deliberate, and "
            "your agreement with yourself on those is the ceiling the model is scored against."
        )
        st.progress(min(1.0, len(done) / max(1, len(queue))))

        if len(done) >= len(queue):
            st.success("Queue complete. Score and report:")
            st.code(
                f"python scripts/validate_placement.py score --set {vs} \\\n"
                f"  --checkpoint {member_ckpts[0].relative_to(ROOT)} \\\n"
                f"  --context artifacts/{ctx_dir.name if ctx_dir else '<run>'}\n"
                f"python scripts/validate_placement.py report --set {vs}",
                language="bash",
            )
        else:
            item = queue[len(done)]
            pa, pb = vdir / "photos" / item["a"], vdir / "photos" / item["b"]
            jc = st.columns([1, 1])
            for col, path in ((jc[0], pa), (jc[1], pb)):
                with col:
                    if path.exists():
                        st.image(str(path), use_container_width=True)
                    if st.button("More attractive", key=f"vote_{len(done)}_{path.name}",
                                 use_container_width=True):
                        with jfile.open("a") as fh:
                            fh.write(json.dumps({
                                "a": item["a"], "b": item["b"], "winner": path.name,
                                "index": len(done),
                            }) + "\n")
                        st.rerun()
            if st.button("Skip / can't tell", key=f"skip_{len(done)}"):
                with jfile.open("a") as fh:
                    fh.write(json.dumps({
                        "a": item["a"], "b": item["b"], "winner": None, "index": len(done),
                    }) + "\n")
                st.rerun()

        rep = vdir / "report.json"
        if rep.exists():
            with st.expander("Last report", expanded=True):
                st.json(json.loads(rep.read_text()))


# ---------------------------------------------------------------- pilot-500 labeler

with tab_labeler:
    from labeler_ui import render_labeler

    render_labeler()
