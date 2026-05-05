#!/bin/bash
# True S4ND (HiPPO-LegS DPLR) sweep on Pathfinder and 15_dist (PathTracker).
# Mirrors sweep_s4nd_convs5.sh layout — OOM-retry, resumable progress log.
#
# S4ND-LegS (`s4nd_full`): separable per-axis S4 with HiPPO-LegS init and
# rank-1 DPLR low-rank correction (canonical S4ND from Nguyen et al., 2022).
#   embed_dim ∈ {32, 64}, d_state ∈ {16, 64}, bidirectional=True
#
# Pathfinder: seq_len=1 (purely spatial; same convention as `s4nd`).
# 15_dist (Girik PathTracker, 32 frames): seq_len=32.
#
# Total: 2 datasets × 4 configs = 8 runs.
#
# Usage:
#   bash scripts/sweep_s4nd_full.sh 2>&1 | tee -a sweep_s4nd_full_$(hostname).out

set -uo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
PROGRESS_LOG="sweep_s4nd_full_progress.log"
touch "$PROGRESS_LOG"

COMPLETED=0
SKIPPED=0
FAILED=0

# ── GPU cleanup ──────────────────────────────────────────────────────────────
kill_gpu_orphans() {
    local pids
    pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u)
    if [ -n "$pids" ]; then
        echo "  Cleaning up GPU processes: $pids"
        for p in $pids; do kill "$p" 2>/dev/null; done
        sleep 3
        for p in $pids; do kill -9 "$p" 2>/dev/null; done
        sleep 2
    fi
}

# ── Run one config with OOM auto-retry ───────────────────────────────────────
run_with_retry() {
    local NAME=$1 START_BS=$2
    shift 2

    local BS=$START_BS
    local MIN_BS=8
    local ATTEMPT=0
    local LOGFILE="/tmp/sweep_run_${NAME}.log"

    while [ "$BS" -ge "$MIN_BS" ]; do
        ATTEMPT=$((ATTEMPT + 1))
        echo "  Attempt $ATTEMPT: batch_size=$BS"

        python main.py "$@" \
            --batch_size ${BS} --run_name ${NAME} \
            2>&1 | tee "$LOGFILE"

        local EXIT_CODE=${PIPESTATUS[0]}

        if [ $EXIT_CODE -eq 0 ]; then
            echo "  SUCCESS: $NAME (bs=$BS) at $(date)"
            return 0
        fi

        if grep -qiE "out of memory|RESOURCE_EXHAUSTED|oom|cuda error.*memory" "$LOGFILE" 2>/dev/null; then
            echo "  OOM at batch_size=$BS -- halving and retrying..."
            BS=$((BS / 2))
            kill_gpu_orphans
        else
            echo "  NON-OOM FAILURE (exit=$EXIT_CODE) -- not retrying"
            kill_gpu_orphans
            return 1
        fi
    done

    echo "  GAVE UP: $NAME -- OOM even at batch_size=$MIN_BS"
    return 1
}

# ── Helper: dispatch a named run ─────────────────────────────────────────────
dispatch_run() {
    local NAME=$1 START_BS=$2
    shift 2

    if grep -q "^${NAME}" "$PROGRESS_LOG"; then
        echo "[${RUN_IDX}/${TOTAL}] SKIP: $NAME"
        SKIPPED=$((SKIPPED + 1))
        return
    fi

    echo "${NAME} RUNNING $(hostname) $(date +%s)" >> "$PROGRESS_LOG"

    echo ""
    echo "========================================"
    echo "[${RUN_IDX}/${TOTAL}] $NAME"
    echo "  $*"
    echo "  Started: $(date)"
    echo "========================================"

    kill_gpu_orphans

    if run_with_retry "$NAME" "$START_BS" "$@"; then
        sed "s/^${NAME} RUNNING.*/${NAME} DONE/" "$PROGRESS_LOG" > /tmp/sweep_prog_tmp.$$ && mv /tmp/sweep_prog_tmp.$$ "$PROGRESS_LOG"
        COMPLETED=$((COMPLETED + 1))
    else
        sed "s/^${NAME} RUNNING.*/${NAME} FAILED/" "$PROGRESS_LOG" > /tmp/sweep_prog_tmp.$$ && mv /tmp/sweep_prog_tmp.$$ "$PROGRESS_LOG"
        FAILED=$((FAILED + 1))
    fi
}

# ── Shared run flags ─────────────────────────────────────────────────────────
COMMON_BASE="--arch simple --stem_mode pathtracker --depth 1 \
  --pool_type max --pos_embed mrope --stem_norm_order post \
  --drop_path_rate 0.0 --epochs 120 --lr 1e-3 --weight_decay 1e-4 --grad_clip 1.0 \
  --norm_type global_layer --checkpointer simple --frame_readout last"

TOTAL=8
RUN_IDX=0

# Skip a dataset block when its TFRecord dir is missing or empty (scratch retention etc).
have_data() {
    [ -d "$1" ] && [ -n "$(ls -A "$1" 2>/dev/null)" ]
}

# ── Pathfinder ───────────────────────────────────────────────────────────────
PF_TFRECORD="/oscar/scratch/dlinsley/pathfinder_tfrecords_128"
PF_PROJECT="CSSM_pathfinder"
PF_EXTRA="--dataset pathfinder --pathfinder_difficulty 14 --image_size 128 \
  --tfrecord_dir ${PF_TFRECORD} --project ${PF_PROJECT}"

if have_data "$PF_TFRECORD"; then
  for EMBED in 32 64; do
    for STATE in 16 64; do
      RUN_IDX=$((RUN_IDX + 1))
      NAME="pf_s4nd_full_d1_e${EMBED}_n${STATE}"
      dispatch_run "$NAME" 128 \
        $COMMON_BASE --cssm s4nd_full --embed_dim ${EMBED} --s4nd_d_state ${STATE} \
        --seq_len 1 --max_seq_len 1 $PF_EXTRA
    done
  done
else
  echo "SKIPPING Pathfinder block: ${PF_TFRECORD} missing or empty"
fi

# ── 15_dist (Girik PathTracker, 32 frames) ───────────────────────────────────
D15_TFRECORD="/oscar/scratch/dlinsley/15_dist"
D15_PROJECT="CSSM_15dist"
D15_EXTRA="--dataset girik --image_size 32 \
  --tfrecord_dir ${D15_TFRECORD} --project ${D15_PROJECT}"

if have_data "$D15_TFRECORD"; then
  for EMBED in 32 64; do
    for STATE in 16 64; do
      RUN_IDX=$((RUN_IDX + 1))
      NAME="15_s4nd_full_d1_e${EMBED}_n${STATE}"
      dispatch_run "$NAME" 128 \
        $COMMON_BASE --cssm s4nd_full --embed_dim ${EMBED} --s4nd_d_state ${STATE} \
        --seq_len 32 --max_seq_len 32 $D15_EXTRA
    done
  done
else
  echo "SKIPPING 15_dist block: ${D15_TFRECORD} missing or empty (scratch retention?)"
fi

echo ""
echo "========================================"
echo "S4ND-LegS (true S4ND) sweep complete at $(date)"
echo "  Total: $TOTAL  Completed: $COMPLETED  Skipped: $SKIPPED  Failed: $FAILED"
echo "========================================"
