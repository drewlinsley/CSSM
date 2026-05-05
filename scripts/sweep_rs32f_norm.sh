#!/bin/bash
# Norm + readout sweep: GDN-InT e32 dk4, kernel_size=7
# on PathTracker Restyled 32f
#
# norm_type ∈ {layer, batch, instance, temporal_layer, global_layer, global_instance}
# readout: (frame_readout=last, pool_type=max) vs (frame_readout=all, pool_type=mean)
#
# Total: 12 runs
#
# Usage:
#   bash scripts/sweep_rs32f_norm.sh 2>&1 | tee -a sweep_rs32f_norm_$(hostname).out

set -uo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
PROGRESS_LOG="sweep_rs32f_norm_progress.log"
touch "$PROGRESS_LOG"

TFRECORD_DIR="/oscar/scratch/dlinsley/pathtracker_restyled_32f_tfrecords"
PROJECT="CSSM_pathtracker_restyled_32f"

COMPLETED=0
SKIPPED=0
FAILED=0

# ── GPU cleanup ──────────────────────────────────────────────────────────────
kill_gpu_orphans() {
    local pids
    pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u)
    if [ -n "$pids" ]; then
        echo "  Cleaning up GPU processes: $pids"
        for p in $pids; do
            kill "$p" 2>/dev/null
        done
        sleep 3
        for p in $pids; do
            kill -9 "$p" 2>/dev/null
        done
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
            echo "  OOM at batch_size=$BS — halving and retrying..."
            BS=$((BS / 2))
            kill_gpu_orphans
        else
            echo "  NON-OOM FAILURE (exit=$EXIT_CODE) — not retrying"
            kill_gpu_orphans
            return 1
        fi
    done

    echo "  GAVE UP: $NAME — OOM even at batch_size=$MIN_BS"
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

# ── Common args ──────────────────────────────────────────────────────────────
COMMON="--arch simple --cssm gdn_int --dataset pathtracker \
    --stem_mode pathtracker --embed_dim 32 --gate_type factored \
    --kernel_size 7 --pos_embed mrope \
    --seq_len 32 --max_seq_len 32 --use_complex32 \
    --drop_path_rate 0.0 \
    --epochs 120 --lr 1e-3 --weight_decay 1e-4 --grad_clip 1.0 \
    --checkpointer simple --stem_norm_order post \
    --project ${PROJECT} --tfrecord_dir ${TFRECORD_DIR} \
    --depth 1 --delta_key_dim 4"

# ── Build run list ───────────────────────────────────────────────────────────
RUN_IDX=0
TOTAL=12

for NORM in layer batch instance temporal_layer global_layer global_instance; do
  NORM_SHORT=$(echo "$NORM" | sed 's/temporal_layer/tlayer/;s/global_layer/glayer/;s/global_instance/ginst/')

  # Readout 1: frame_readout=last, pool_type=max
  RUN_IDX=$((RUN_IDX + 1))
  NAME="rs32f_norm_${NORM_SHORT}_last_max"

  dispatch_run "$NAME" 128 \
      ${COMMON} --norm_type ${NORM} \
      --frame_readout last --pool_type max

  # Readout 2: frame_readout=all, pool_type=mean
  RUN_IDX=$((RUN_IDX + 1))
  NAME="rs32f_norm_${NORM_SHORT}_all_mean"

  dispatch_run "$NAME" 128 \
      ${COMMON} --norm_type ${NORM} \
      --frame_readout all --pool_type mean

done

echo ""
echo "========================================"
echo "Sweep complete at $(date)"
echo "  Total: $TOTAL  Completed: $COMPLETED  Skipped: $SKIPPED  Failed: $FAILED"
echo "========================================"
