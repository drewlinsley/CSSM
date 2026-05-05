#!/bin/bash
# GDN norm comparison on PathTracker (new dataset)
# Grid: norm_type × delta_key_dim = 4 × 2 = 8 runs, 20 epochs each
# Config: GDN depth=1, embed_dim=32, qkv_conv_size=1, lr=1e-3
#
# Usage:
#   nohup bash scripts/sweep_gdn_norm.sh > sweep_gdn_norm_$(hostname).out 2>&1 &

set -uo pipefail

PROGRESS_LOG="sweep_gdn_norm_progress.log"
touch "$PROGRESS_LOG"

TFRECORD_DIR="/cifs/data/tserre_lrs/projects/projects/prj_video_datasets/pathtracker_equal_large_tfrecords"
PROJECT="CSSM_pathtracker"

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
    local NORM=$1 DK=$2 NAME=$3 START_BS=$4

    local BS=$START_BS
    local MIN_BS=8
    local ATTEMPT=0
    local LOGFILE="/tmp/sweep_run_${NAME}.log"

    while [ "$BS" -ge "$MIN_BS" ]; do
        ATTEMPT=$((ATTEMPT + 1))
        echo "  Attempt $ATTEMPT: batch_size=$BS"

        python main.py --arch simple --cssm gdn --dataset pathtracker \
            --stem_mode pathtracker --embed_dim 32 --gate_type factored \
            --kernel_size 11 --pool_type max --pos_embed mrope \
            --max_seq_len 64 --batch_size ${BS} --epochs 20 --lr 1e-3 \
            --weight_decay 1e-4 --grad_clip 1.0 --checkpointer simple \
            --stem_norm_order post --project ${PROJECT} \
            --tfrecord_dir ${TFRECORD_DIR} \
            --depth 1 --delta_key_dim ${DK} --norm_type ${NORM} \
            --run_name ${NAME} \
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

# ── Build run list ───────────────────────────────────────────────────────────
RUN_IDX=0
TOTAL=8

for NORM in layer batch temporal_layer instance; do
  for DK in 2 3; do
    RUN_IDX=$((RUN_IDX + 1))
    NAME="gdn_norm_${NORM}_dk${DK}"

    # Skip if completed or claimed
    if grep -q "^${NAME}" "$PROGRESS_LOG"; then
      echo "[${RUN_IDX}/${TOTAL}] SKIP: $NAME"
      SKIPPED=$((SKIPPED + 1))
      continue
    fi

    echo "${NAME} RUNNING $(hostname) $(date +%s)" >> "$PROGRESS_LOG"

    echo ""
    echo "========================================"
    echo "[${RUN_IDX}/${TOTAL}] $NAME"
    echo "  norm_type=$NORM delta_key_dim=$DK"
    echo "  Started: $(date)"
    echo "========================================"

    kill_gpu_orphans

    if run_with_retry "$NORM" "$DK" "$NAME" 128; then
      sed "s/^${NAME} RUNNING.*/${NAME} DONE/" "$PROGRESS_LOG" > /tmp/sweep_prog_tmp.$$ && mv /tmp/sweep_prog_tmp.$$ "$PROGRESS_LOG"
      COMPLETED=$((COMPLETED + 1))
    else
      sed "s/^${NAME} RUNNING.*/${NAME} FAILED/" "$PROGRESS_LOG" > /tmp/sweep_prog_tmp.$$ && mv /tmp/sweep_prog_tmp.$$ "$PROGRESS_LOG"
      FAILED=$((FAILED + 1))
    fi

  done
done

echo ""
echo "========================================"
echo "GDN norm sweep complete at $(date)"
echo "  Total: $TOTAL  Completed: $COMPLETED  Skipped: $SKIPPED  Failed: $FAILED"
echo "========================================"
