#!/usr/bin/env python
"""Late-fusion logistic head: v25 score delta + MediaPipe frontal Δratios.

Train only on panel pairs whose faces are outside the face-id val split
(bothHeldOut). Eval: uniform run-4, panel majority, bothHeldOut.

Kill if fusion ≤ 83.8% (n=111 place path) or clearly no lift vs score-sign 83.9%.

Usage:
    python scripts/late_fusion_landmarks.py --run train-v25-panel-ft-v23b
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from accuracy_matrix import load_votes  # noqa: E402
from extract_frontal_ratios import RATIO_NAMES  # noqa: E402
from faceiq_pref.data import load_export, split_by_face_id  # noqa: E402

# Panel sources for fusion-head training (eval stays run-4 bothHeldOut).
TRAIN_PANELS = [
    ("labels/panel-run-4-random", "artifacts/panel-run-v4/reject-pids.txt"),
    ("labels/panel-run-3", "artifacts/panel-run-v3/reject-pids.txt"),
    ("labels/panel-pilot", "artifacts/panel-run-v1/reject-pids.txt"),
    ("labels/panel-pilot/soft-launch", "artifacts/panel-run-v1/reject-pids.txt"),
]

BASELINE_PLACE = 0.838  # n=111 place@200
BASELINE_SCORE_SIGN = 0.839  # n=112 score-sign audit


def _load_rejects(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()}


def _pair_features(
    fa: str,
    fb: str,
    model: dict[str, float],
    ratios: dict[str, dict[str, float]],
    ratio_names: list[str],
) -> np.ndarray | None:
    if fa not in model or fb not in model or fa not in ratios or fb not in ratios:
        return None
    ra, rb = ratios[fa], ratios[fb]
    feats = [model[fa] - model[fb]]
    for name in ratio_names:
        if name not in ra or name not in rb:
            return None
        feats.append(ra[name] - rb[name])
    return np.asarray(feats, dtype=np.float64)


def _collect_pairs(
    panel_dir: Path,
    rejects: set[str],
    model: dict[str, float],
    ratios: dict[str, dict[str, float]],
    ratio_names: list[str],
    face_filter,
) -> list[dict]:
    results = panel_dir / "results"
    meta_path = panel_dir / "sample-meta.json"
    if not results.exists() or not meta_path.exists():
        return []
    votes, meta = load_votes(results, meta_path, rejects, True)
    out: list[dict] = []
    for pid, (va, vb) in votes.items():
        m = meta.get(pid)
        if m is None or va == vb:
            continue
        fa, fb = m["faceAId"], m["faceBId"]
        if not face_filter(fa, fb):
            continue
        x = _pair_features(fa, fb, model, ratios, ratio_names)
        if x is None:
            continue
        sa, sb = model[fa], model[fb]
        if sa == sb:
            continue
        maj_a = va > vb
        out.append({
            "pairId": pid,
            "faceAId": fa,
            "faceBId": fb,
            "panel": str(panel_dir.relative_to(ROOT)),
            "y": int(maj_a),
            "x": x,
            "scoreSignPickA": bool(sa > sb),
            "votesA": va,
            "votesB": vb,
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="train-v25-panel-ft-v23b")
    ap.add_argument("--ratios", default="artifacts/fusion-landmarks-v1/frontal-ratios.csv")
    ap.add_argument("--control-ranking", default="artifacts/bt-refit-v2-qc")
    ap.add_argument("--eval-panel", default="labels/panel-run-4-random")
    ap.add_argument("--eval-rejects", default="artifacts/panel-run-v4/reject-pids.txt")
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--out", default="artifacts/fusion-landmarks-v1")
    a = ap.parse_args()

    art = ROOT / "artifacts" / a.run
    scores_path = art / "model_scores.csv"
    ratios_path = ROOT / a.ratios
    if not scores_path.exists():
        raise SystemExit(f"missing {scores_path}")
    if not ratios_path.exists():
        raise SystemExit(f"missing {ratios_path} — run extract_frontal_ratios.py first")

    export = next(
        p
        for p in sorted((ROOT / "data" / "exports").glob("*"))
        if (p / "faces.jsonl").exists()
    )
    scores = pd.read_csv(scores_path)
    model = dict(zip(scores["faceId"], scores["modelScore"]))
    ratio_df = pd.read_csv(ratios_path)
    ratio_names = [c for c in RATIO_NAMES if c in ratio_df.columns]
    ratios = {
        r["faceId"]: {n: float(r[n]) for n in ratio_names}
        for r in ratio_df.to_dict(orient="records")
    }

    ctrl = pd.read_csv(ROOT / a.control_ranking / "ratings.csv")
    control = dict(zip(ctrl["faceId"], ctrl["theta"]))

    _, val_rows = split_by_face_id(
        load_export(export).all_matchups(), a.val_fraction, a.split_seed
    )
    held_out = {f for m in val_rows for f in (m.face_a_id, m.face_b_id)}

    train_rows: list[dict] = []
    for panel_rel, rej_rel in TRAIN_PANELS:
        train_rows.extend(
            _collect_pairs(
                ROOT / panel_rel,
                _load_rejects(ROOT / rej_rel),
                model,
                ratios,
                ratio_names,
                face_filter=lambda fa, fb: fa not in held_out and fb not in held_out,
            )
        )
    # Dedup by pairId (soft-launch ⊂ pilot)
    seen: set[str] = set()
    deduped: list[dict] = []
    for r in train_rows:
        if r["pairId"] in seen:
            continue
        seen.add(r["pairId"])
        deduped.append(r)
    train_rows = deduped

    eval_rows = _collect_pairs(
        ROOT / a.eval_panel,
        _load_rejects(ROOT / a.eval_rejects),
        model,
        ratios,
        ratio_names,
        face_filter=lambda fa, fb: fa in held_out and fb in held_out,
    )

    if len(train_rows) < 30:
        raise SystemExit(f"too few train pairs after face-id filter: {len(train_rows)}")
    if not eval_rows:
        raise SystemExit("no bothHeldOut eval pairs with scores+ratios")

    X_train = np.stack([r["x"] for r in train_rows])
    y_train = np.asarray([r["y"] for r in train_rows], dtype=np.int64)
    X_eval = np.stack([r["x"] for r in eval_rows])
    y_eval = np.asarray([r["y"] for r in eval_rows], dtype=np.int64)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_eval_s = scaler.transform(X_eval)

    clf = LogisticRegression(
        C=a.C,
        max_iter=2000,
        solver="lbfgs",
        random_state=42,
    )
    clf.fit(X_train_s, y_train)

    # score-sign alone: first feature before scaling
    score_sign_pred = (X_eval[:, 0] > 0).astype(np.int64)
    fusion_pred = clf.predict(X_eval_s)
    # Also score-delta-only logistic (ablation)
    clf_score = LogisticRegression(C=a.C, max_iter=2000, solver="lbfgs", random_state=42)
    clf_score.fit(X_train_s[:, :1], y_train)
    score_logit_pred = clf_score.predict(X_eval_s[:, :1])

    def acc(pred: np.ndarray) -> float:
        return float((pred == y_eval).mean())

    score_sign_acc = acc(score_sign_pred)
    fusion_acc = acc(fusion_pred)
    score_logit_acc = acc(score_logit_pred)
    n_eval = len(eval_rows)

    # Control-right miss flips (from v25 error audit)
    audit_path = art / "bothHeldOut-error-audit.json"
    miss_flips: list[dict] = []
    n_ctrl_right = 0
    n_ctrl_right_flipped = 0
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        by_pid = {r["pairId"]: r for r in eval_rows}
        for e in audit["errors"]:
            if e.get("bucket") != "modelWrong_controlRight":
                continue
            n_ctrl_right += 1
            row = by_pid.get(e["pairId"])
            if row is None:
                miss_flips.append({
                    "pairId": e["pairId"],
                    "status": "missing_from_eval",
                })
                continue
            idx = eval_rows.index(row)
            fused_ok = bool(fusion_pred[idx] == y_eval[idx])
            if fused_ok:
                n_ctrl_right_flipped += 1
            miss_flips.append({
                "pairId": e["pairId"],
                "faceAId": e["faceAId"],
                "faceBId": e["faceBId"],
                "panelMajority": e["panelMajority"],
                "scoreSignPick": "A" if row["scoreSignPickA"] else "B",
                "fusionPick": "A" if fusion_pred[idx] == 1 else "B",
                "fusionCorrect": fused_ok,
                "btGap": e.get("btGap"),
            })

    # Coefficients (original feature order after scaler)
    coef = {
        name: float(c)
        for name, c in zip(
            ["score_delta", *ratio_names],
            clf.coef_.ravel() / scaler.scale_,
            strict=True,
        )
    }

    lift_vs_score_sign = fusion_acc - score_sign_acc
    # Kill rule: ≤ place baseline or clearly no lift vs score-sign
    kill = (fusion_acc <= BASELINE_PLACE + 1e-12) or (lift_vs_score_sign <= 0.005 + 1e-12)
    verdict = "KILL" if kill else "SHIP_CANDIDATE"

    summary = {
        "run": a.run,
        "yardstick": "uniform run-4, panel majority, bothHeldOut",
        "pairDistribution": "uniform run-4",
        "metric": "panel majority",
        "leakageStratum": "bothHeldOut",
        "landmarkSource": "mediapipe_face_landmarker_478 via extract_frontal_ratios.py",
        "ratiosPath": a.ratios,
        "ratioNames": ratio_names,
        "trainPanels": [p for p, _ in TRAIN_PANELS],
        "trainPairs": len(train_rows),
        "evalPairs": n_eval,
        "scoreSignAccuracy": score_sign_acc,
        "scoreLogitOnlyAccuracy": score_logit_acc,
        "fusionAccuracy": fusion_acc,
        "baselinePlace200": BASELINE_PLACE,
        "baselineScoreSign": BASELINE_SCORE_SIGN,
        "liftVsScoreSign": lift_vs_score_sign,
        "liftVsPlace200": fusion_acc - BASELINE_PLACE,
        "controlRightMisses": n_ctrl_right,
        "controlRightMissesFlipped": n_ctrl_right_flipped,
        "coefficientsUnscaled": coef,
        "verdict": verdict,
        "killRule": (
            "KILL if fusion ≤ 83.8% (place@200 n=111) or lift vs score-sign ≤ 0.5 pt"
        ),
        "note": (
            "Logistic late fusion on [s_a-s_b, Δratios]. Face-id split: train pairs "
            "exclude any held-out face; eval is bothHeldOut only. Vote-blind control "
            "ranking not used as features."
        ),
    }

    out_dir = ROOT / a.out
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "late-fusion-eval.json"
    payload = {"summary": summary, "controlRightMissFlips": miss_flips}
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Per-eval-pair CSV
    rows_csv = []
    for i, r in enumerate(eval_rows):
        fa, fb = r["faceAId"], r["faceBId"]
        ca = control.get(fa)
        cb = control.get(fb)
        rows_csv.append({
            "pairId": r["pairId"],
            "faceAId": fa,
            "faceBId": fb,
            "panelMajority": "A" if r["y"] else "B",
            "scoreSignPick": "A" if r["scoreSignPickA"] else "B",
            "fusionPick": "A" if fusion_pred[i] == 1 else "B",
            "scoreSignCorrect": bool(score_sign_pred[i] == y_eval[i]),
            "fusionCorrect": bool(fusion_pred[i] == y_eval[i]),
            "controlPick": (
                "A" if ca is not None and cb is not None and ca > cb
                else ("B" if ca is not None and cb is not None and ca < cb else "tie")
            ),
            "scoreDelta": float(X_eval[i, 0]),
        })
    pd.DataFrame(rows_csv).to_csv(out_dir / "late-fusion-eval.csv", index=False)

    print(f"late fusion vs {a.run} (uniform run-4, bothHeldOut, panel majority)")
    print(f"  train pairs (no held-out faces): {len(train_rows)}")
    print(f"  eval n={n_eval}")
    print(f"  score-sign alone:     {score_sign_acc:.1%}")
    print(f"  score-logit only:     {score_logit_acc:.1%}")
    print(f"  fusion (score+dratio): {fusion_acc:.1%}")
    print(f"  vs place@200 83.8%:   {fusion_acc - BASELINE_PLACE:+.1%}")
    print(f"  vs score-sign:        {lift_vs_score_sign:+.1%}")
    print(f"  control-right misses flipped: {n_ctrl_right_flipped}/{n_ctrl_right}")
    print(f"  verdict: {verdict}")
    print(f"-> {out_json.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
