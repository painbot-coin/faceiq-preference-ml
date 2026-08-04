#!/usr/bin/env bash
# Ship the panel training arms to the GPU box and run them.
#
# `git pull` on the box is NOT enough, for two separate reasons, and both are easy to lose an
# hour to:
#
#   1. The vote data is deliberately gitignored. `labels/*/results/judgments.jsonl` carries
#      Prolific participant IDs and `sample-meta.json` is the answer key (thetaA/thetaB,
#      vlmOutcome, gold candidacy). Neither may ever enter git history, so neither can arrive
#      by pull — they have to be copied.
#   2. `artifacts/` is gitignored wholesale, which includes the reject lists the fit needs to
#      drop excluded raters.
#
# So this copies exactly the files a panel training run consumes, verifies the box already has
# the photo export, and then runs both label arms plus the human-grounded eval.
#
# Usage:
#   scripts/sync_panel_to_gpu.sh 16.148.117.99            # sync, then train both arms
#   scripts/sync_panel_to_gpu.sh 16.148.117.99 --sync-only
#
# If SSH times out (rather than being refused), it is almost always the security group: the
# inbound rule was pinned to "My IP" when the box was created and your home IP has changed
# since. Re-add your current IP, shown at the end of a failed run.

set -euo pipefail

HOST="${1:?usage: $0 <public-ip> [--sync-only]}"
MODE="${2:-run}"
KEY="${FACEIQ_GPU_KEY:-$HOME/.ssh/faceiq-gpu.pem}"
USER="${FACEIQ_GPU_USER:-ubuntu}"
REMOTE="${FACEIQ_GPU_DIR:-faceiq-preference-ml}"
SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20)

cd "$(dirname "$0")/.."

if ! "${SSH[@]}" -o BatchMode=yes "$USER@$HOST" true 2>/dev/null; then
  echo "cannot reach $USER@$HOST:22 with $KEY" >&2
  echo >&2
  echo "A timeout means packets are being dropped — check the security group's inbound SSH" >&2
  echo "rule. Your current public IP is:" >&2
  curl -s --max-time 10 https://checkip.amazonaws.com >&2 || echo "  (could not determine)" >&2
  exit 1
fi

# The box is not a git checkout, so nothing there tracks local edits. Copying only the couple
# of files a change touched leaves the rest of the package at whatever version last landed,
# and the mismatch surfaces as a confusing TypeError deep in the run: an older model.py was
# once paired with a newer train.py that passed variance_head. Sync the whole package instead.
CODE=(src/faceiq_pref scripts configs)

# Which arms to train, in order. Each name must be both the config basename and the
# `run_name` inside it, which is the convention every panel config follows.
#   FACEIQ_ARMS="train-v13-panel-hard" scripts/sync_panel_to_gpu.sh <ip>
read -r -a ARMS <<< "${FACEIQ_ARMS:-train-v16-panel-run4}"

# Which studies to ship is **derived from the arms' configs**, not listed here. It used to be a
# hardcoded list of runs 2 and 3, which silently shipped the wrong data the moment an arm added
# a study: training would run for 35 minutes on two thirds of its labels and report a number
# nobody could reproduce. `panel_labels` in the config is the single source of truth.
PANEL_ARGS=$(python3 - "${ARMS[@]}" <<'PY'
import sys, yaml
panels, rejects = [], []
for arm in sys.argv[1:]:
    cfg = yaml.safe_load(open(f"configs/{arm}.yaml"))
    for results, meta in cfg.get("panel_labels") or []:
        if [results, meta] not in panels:
            panels.append([results, meta])
    for r in cfg.get("panel_rejects") or []:
        if r not in rejects:
            rejects.append(r)
# Line 1: the files to copy. Line 2: the --panel/--rejects flags eval_vs_panel.py needs.
files = [m for _, m in panels] + [f"{r}/judgments.jsonl" for r, _ in panels] + rejects
print(" ".join(files))
print(" ".join(f"--panel {r} {m}" for r, m in panels)
      + (" --rejects " + " ".join(rejects) if rejects else ""))
PY
)
read -r PANEL_FILES <<< "$(printf '%s\n' "$PANEL_ARGS" | sed -n 1p)"
EVAL_PANEL_FLAGS=$(printf '%s\n' "$PANEL_ARGS" | sed -n 2p)

# Data the run consumes that git will not carry: the vote data and reject lists above (derived),
# plus three files eval_vs_panel.py needs — the ranking that defines the percentile-gap bands,
# and the QC lists that keep excluded faces out of the eval. v2-qc is deliberately the gap
# source: it never saw a human vote, so a band cut on it cannot be circular.
read -r -a FILES <<< "$PANEL_FILES"
FILES+=(
  artifacts/bt-refit-v2-qc/ratings.csv
  artifacts/face-qc-v1/exclude-faces.csv
  artifacts/face-qc-v1/gender-fixes.csv
)

for f in "${FILES[@]}"; do
  [[ -f "$f" ]] || { echo "missing locally: $f" >&2; exit 1; }
done
for a in "${ARMS[@]}"; do
  [[ -f "configs/$a.yaml" ]] || { echo "no config for arm: configs/$a.yaml" >&2; exit 1; }
done

echo "==> creating directories on $HOST"
"${SSH[@]}" "$USER@$HOST" "cd '$REMOTE' && mkdir -p $(printf '%q ' $(printf '%s\n' "${FILES[@]}" | xargs -n1 dirname | sort -u))"

echo "==> syncing code (${CODE[*]})"
for d in "${CODE[@]}"; do
  rsync -a --exclude '__pycache__' --exclude '*.pyc' \
    -e "ssh -i $KEY -o StrictHostKeyChecking=accept-new" \
    --out-format='    %n' "$d/" "$USER@$HOST:$REMOTE/$d/"
done

echo "==> copying ${#FILES[@]} data files (vote data + reject lists are gitignored by design)"
for f in "${FILES[@]}"; do
  scp -q -i "$KEY" -o StrictHostKeyChecking=accept-new "$f" "$USER@$HOST:$REMOTE/$f"
  printf '    %s\n' "$f"
done

echo "==> checking the box has the photo export and a GPU"
"${SSH[@]}" "$USER@$HOST" "cd '$REMOTE' && \
  test -f data/exports/cmr1mr0m7000196d57zi3vcgn/manifest.json \
    || { echo 'MISSING data/exports/ — this is a fresh volume; copy the export first' >&2; exit 1; }; \
  echo -n '    images: '; ls data/exports/cmr1mr0m7000196d57zi3vcgn/images | wc -l; \
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/    gpu: /'"

# Fail fast on the label plumbing before committing ~45 min of GPU: if the vote files did not
# arrive intact, the box builds far fewer targets than we have locally and the run reports a
# number nobody can reproduce. Compare the box's count against this machine's rather than a
# literal — the literal (7,780) went stale the moment run 4 was added.
echo "==> verifying panel targets built on the box"
COUNT_TARGETS="import sys; sys.path.insert(0,'src')
from faceiq_pref.panel import load_panel_targets, load_rejects
import yaml
cfg=yaml.safe_load(open('configs/${ARMS[0]}.yaml'))
P=[tuple(x) for x in cfg['panel_labels']]
R=load_rejects(cfg.get('panel_rejects') or [])
t=load_panel_targets(P,R,min_votes=cfg.get('panel_min_votes',4),prior=cfg.get('panel_prior',1.0))
print(len(t), sum(1 for v in t.values() if 0.35<v<0.65))"
LOCAL_TARGETS=$(source .venv/bin/activate && python -c "$COUNT_TARGETS")
REMOTE_TARGETS=$("${SSH[@]}" "$USER@$HOST" "cd '$REMOTE' && source .venv/bin/activate && python -c \"$COUNT_TARGETS\"")
if [[ "$LOCAL_TARGETS" != "$REMOTE_TARGETS" ]]; then
  echo "    panel targets differ — local '$LOCAL_TARGETS' vs box '$REMOTE_TARGETS'" >&2
  echo "    the vote files did not copy intact; do not start a run on this" >&2
  exit 1
fi
echo "    ${REMOTE_TARGETS% *} soft targets, ${REMOTE_TARGETS#* } near 0.5, box matches local  OK"

# Build the model each config actually asks for. Cheap, and it catches a stale package on the
# box before the run has spent any GPU time getting to the same failure.
echo "==> building the configured model"
"${SSH[@]}" "$USER@$HOST" "cd '$REMOTE' && source .venv/bin/activate && ARMS='${ARMS[*]}' python -c \"
import os, sys, yaml; sys.path.insert(0,'src')
from faceiq_pref.model import PairwiseModel
for a in os.environ['ARMS'].split():
    cfg=yaml.safe_load(open(f'configs/{a}.yaml'))
    assert cfg['run_name']==a, f\\\"{a}.yaml has run_name {cfg['run_name']!r}\\\"
    PairwiseModel(cfg['backbone'], variance_head=cfg.get('variance_head', False))
    print(f\\\"    {a}: {cfg['backbone']}, val_fraction {cfg['val_fraction']}, \\\"
          f\\\"panel_weight {cfg.get('panel_weight', 1.0)} ok\\\")
\""

if [[ "$MODE" == "--sync-only" ]]; then
  echo "==> sync only, stopping here"
  exit 0
fi

# Getting ssh to actually RETURN here is fiddlier than it looks. The job must be a single
# simple command with all three descriptors redirected, so no wrapper subshell is left holding
# the SSH channel. `cd X && source Y && nohup job > log &` does NOT achieve that: the `&`
# backgrounds the whole `&&` chain, and the subshell running it keeps ssh's stdout open until
# the job exits — so ssh hangs for the full training run even though the job is detached and
# would survive. Hence: everything inside the `bash -c`, `setsid` to leave the session, and
# stdin/stdout/stderr all redirected on the one command.
# Timestamped so a new launch cannot erase the log of the run we are still writing up.
LOG="panel-train-$(date +%Y%m%d-%H%M%S).log"
echo "==> training ${#ARMS[@]} arm(s): ${ARMS[*]}"
echo "    (~35 min per arm at val_fraction 0.5, ~90 min at 0.2; runs under nohup, safe to"
echo "     disconnect)"
"${SSH[@]}" -n "$USER@$HOST" "setsid bash -c '
    cd '$REMOTE'
    source .venv/bin/activate
    set -x
    for run in ${ARMS[*]}; do
      python scripts/train.py --config configs/\$run.yaml
    done
    for run in ${ARMS[*]}; do
      python scripts/eval_vs_panel.py \
        --export data/exports/cmr1mr0m7000196d57zi3vcgn \
        --checkpoint checkpoints/\$run/best.pt \
        $EVAL_PANEL_FLAGS \
        --ratings artifacts/bt-refit-v2-qc/ratings.csv \
        --out artifacts/\$run/panel-eval.json \
        --dump-pairs artifacts/\$run/panel-pairs.csv
    done
    # The population number, on run 4 alone. This is the one that matters now: log §5.8 measured
    # the comparator 1.8-3.1 pts behind the ranking on typical pairs, and the pooled eval above
    # cannot show that because it is dominated by near-ties.
    for run in ${ARMS[*]}; do
      python scripts/eval_vs_panel.py \
        --export data/exports/cmr1mr0m7000196d57zi3vcgn \
        --checkpoint checkpoints/\$run/best.pt \
        --panel labels/panel-run-4-random/results labels/panel-run-4-random/sample-meta.json \
        --rejects artifacts/panel-run-v4/reject-pids.txt \
        --ratings artifacts/bt-refit-v2-qc/ratings.csv \
        --out artifacts/\$run/panel-eval-run4.json \
        --dump-pairs artifacts/\$run/panel-pairs-run4.csv
    done
  ' < /dev/null > '$REMOTE/$LOG' 2>&1 &
  echo '    started; follow with:'"
echo "    ssh -i $KEY $USER@$HOST 'tail -f $REMOTE/$LOG'"
