# Harsh data-lab fusion brief (comparator arm)

*2026-08-12 (Option 2). Amended 2026-08-13 after self-serve Option 1. Yardstick:
uniform run-4, panel majority, bothHeldOut. Ship baseline `train-v25-panel-ft-v23b`
**83.8%** (n=111 place@200); score-sign audit stratum n=112 / 83.9%.*

## Decision

**Option 2 executed** (2026-08-12) — error-driven feature brief + offline cross.

**Option 1 executed without waiting on Labs** (2026-08-13) — MediaPipe Face
Landmarker on cohort export photos → 6 frontal ratios → logistic late fusion with
v25 `(s_a − s_b)`. **Killed:** bothHeldOut **82.1%** (n=112, uniform run-4) vs
score-sign **83.9%** / place@200 **83.8%**. Flipped **0/8** control-right misses.

Do **not** train ethnicity- or QC-flag late fusion as an accuracy bet. Do **not**
re-run the same MediaPipe-ratio recipe hoping for a different draw.

We still want Labs `frontLandmarks` (+ Labs-derived ratios / pose) for **production
parity** and a schema that matches the inference path — not because this prototype
cleared the yardstick.

## Inventory (what exists today)

| Asset | Path | Fields | Fusion-ready? |
|---|---|---|---|
| Face attributes | `labels/face-attributes.json` | `ethnicities`, `prodGender` | No — demographic only |
| Face QC v1 | `artifacts/face-qc-v1/` | ethnicity, Fitzpatrick, age band, issues, excludes | Partial — QC / strata only |
| Self-serve MediaPipe ratios | `artifacts/fusion-landmarks-v1/frontal-ratios.csv` | 6 scale-invariant ratios, 2,998 faces | **Tried — killed** |
| Error audit | `artifacts/train-v25-panel-ft-v23b/bothHeldOut-error-audit.json` | 8 control-right / model-wrong | Done |
| Miss × feature cross | `artifacts/train-v25-panel-ft-v23b/bothHeldOut-miss-feature-cross.{json,csv}` | rates + per-miss table | Done |
| Late-fusion eval | `artifacts/fusion-landmarks-v1/late-fusion-eval.json` | score-sign vs fusion | **KILL 82.1%** |

Research already named the Labs path: `Face.frontLandmarks` /
`mediapipeLandmarks` on every upload; `export-gt-run.ts` omits them (log §5.7
follow-on). Self-serve extraction unblocked the offline test; exporter parity is
still the production ask.

## Measurable cross (score-sign bothHeldOut, n=112)

| Signal | all bothHeldOut | model-correct | control-right misses (n=8) |
|---|---:|---:|---:|
| photoIssueEither | 57.1% | 58.5% | 62.5% |
| qcIssueEither | 63.4% | 66.0% | 62.5% |
| ethnicityMismatch | 49.1% | 45.7% | **75.0%** (6/8) |
| btGap &lt; 1 | 9.8% | 8.5% | 37.5% (3/8) |
| btGap ≥ 3 | 67.9% | 69.1% | 50.0% (4/8) |
| qcExcludedEither | 0% | 0% | 0% |

Read carefully:

- **Photo QC is not enriched** on the 8 misses vs model-correct — uploading a
  QC-flag fusion head would chase noise.
- **Ethnicity mismatch looks enriched (6/8)** but n=8 is tiny; do not ship a
  demographic fusion lever.
- **BT gaps on the 8 are mixed** — already killed near-tie reweighting.
- **0/8** involve QC-excluded faces.

Reproduce:

```bash
python scripts/bothheldout_miss_feature_cross.py --run train-v25-panel-ft-v23b
```

## Option 1 result (self-serve MediaPipe)

| Item | Value |
|---|---|
| Landmark source | MediaPipe Face Landmarker 478 (`faceiq_pref.preprocess`), export `images/*.webp` |
| Features | `[s_a−s_b, Δeye_span/face_w, Δnose/face_h, Δjaw/face_w, Δmouth/face_w, Δmidface/face_h, Δcanthal_tilt]` |
| Train | 5,942 panel pairs, **both faces outside** face-id val split |
| Eval | uniform run-4, panel majority, bothHeldOut **n=112** |
| Score-sign alone | **83.9%** |
| Fusion | **82.1%** (−1.8 vs score-sign; −1.7 vs place 83.8%) |
| Control-right flips | **0/8** |
| Verdict | **KILL** |

```bash
python scripts/extract_frontal_ratios.py --out artifacts/fusion-landmarks-v1
python scripts/late_fusion_landmarks.py --run train-v25-panel-ft-v23b
```

## What Harsh should still export (production parity)

1. **`frontLandmarks` for the 3,000 cohort faces** — freeze schema (point count,
   ordering, coordinate space, pose-normalised or not). Join key `faceId`.
2. **Derived frontal ratios** Labs already computes (canthal tilt, facial thirds,
   jaw/width, eye spacing, nose/mouth) — may differ from our 6 MediaPipe proxies.
3. **Pose / framing scalars** — yaw, pitch, roll, face-fill, plus
   `perspectiveSuspect` if available (still the untested QC hole).
4. **Do not prioritize ethnicity for fusion.**

Minimum viable handoff unchanged: one JSONL/CSV keyed by `faceId`. A re-try of
late fusion is only justified if Labs geometry is **materially richer** than the
MediaPipe ratio table already killed (especially pose / perspective / curated
anatomical ratios) — not a re-export of the same six proxies.

## Ship / kill on this slice

| Action | Result |
|---|---|
| Ethnicity/QC late-fusion prototype | **Not run** — wrong signal class |
| Miss × feature cross + Option 2 brief | **Shipped** |
| MediaPipe-ratio late fusion (Option 1) | **KILL** — 82.1% (n=112) |
| Same-pixel loss / readout churn | Still closed |

## Exact next action

Harsh: still add `frontLandmarks` (+ Labs ratio/pose dict) to the GT export when
convenient for production parity. Meruzhan: do **not** idle on that for the next
accuracy bet; MediaPipe-ratio fusion is closed. Pick a non-fusion comparator lever
or a Labs-geometry re-try only if the export adds pose/perspective beyond what we
already extracted.
