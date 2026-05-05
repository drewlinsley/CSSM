#!/bin/bash
# Re-run the 15_dist (Girik PathTracker) half of the s4nd_full sweep using
# the persistent CIFS data location (/oscar/scratch/dlinsley/15_dist was purged
# by scratch retention).

set -uo pipefail

PROGRESS_LOG="sweep_s4nd_full_15dist_progress.log"
touch "$PROGRESS_LOG"

COMPLETED=0
SKIPPED=0
FAILED=0

kill_gpu_orphans() {
    local pids
    pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u)
    if [ -n "$pids" ]; then
        for p in $pids; do kill "$p" 2>/dev/null; done
        sleep 3
        for p in $pids; do kill -9 "$p" 2>/dev/null; done
        sleep 2
    fi
}

run_with_retry() {
    local NAME=$1 START_BS=$2
    shift 2
    local BS=$START_BS
    local MIN_BS=8
    local LOGFILE="/tmp/sweep_run_${NAME}.log"
    while [ "$BS" -ge "$MIN_BS" ]; do
        echo "  Attempt: batch_size=$BS"
        python main.py "$@" --batch_size ${BS} --run_name ${NAME} 2>&1 | tee "$LOGFILE"
        local EC=${PIPESTATUS[0]}
        if [ $EC -eq 0 ]; then
            echo "  SUCCESS: $NAME (bs=$BS) at $(date)"
            return 0
        fi
        if grep -qiE "out of memory|RESOURCE_EXHAUSTED|oom|cuda error.*memory" "$LOGFILE" 2>/dev/null; then
            BS=$((BS / 2)); kill_gpu_orphans
        else
            echo "  NON-OOM FAILURE (exit=$EC) -- not retrying"
            kill_gpu_orphans
            return 1
        fi
    done
    echo "  GAVE UP: $NAME"
    return 1
}

dispatch_run() {
    local NAME=$1 START_BS=$2
    shift 2
    if grep -q "^${NAME}" "$PROGRESS_LOG"; then
        echo "[${RUN_IDX}/${TOTAL}] SKIP: $NAME"
        SKIPPED=$((SKIPPED + 1)); return
    fi
    echo "${NAME} RUNNING $(hostname) $(date +%s)" >> "$PROGRESS_LOG"
    echo ""
    echo "========================================"
    echo "[${RUN_IDX}/${TOTAL}] $NAME"
    echo "  Started: $(date)"
    echo "========================================"
    kill_gpu_orphans
    if run_with_retry "$NAME" "$START_BS" "$@"; then
        sed "s/^${NAME} RUNNING.*/${NAME} DONE/" "$PROGRESS_LOG" > /tmp/sw15_$$.tmp && mv /tmp/sw15_$$.tmp "$PROGRESS_LOG"
        COMPLETED=$((COMPLETED + 1))
    else
        sed "s/^${NAME} RUNNING.*/${NAME} FAILED/" "$PROGRESS_LOG" > /tmp/sw15_$$.tmp && mv /tmp/sw15_$$.tmp "$PROGRESS_LOG"
        FAILED=$((FAILED + 1))
    fi
}

COMMON_BASE="--arch simple --stem_mode pathtracker --depth 1 \
  --pool_type max --pos_embed mrope --stem_norm_order post \
  --drop_path_rate 0.0 --epochs 120 --lr 1e-3 --weight_decay 1e-4 --grad_clip 1.0 \
  --norm_type global_layer --checkpointer simple --frame_readout last"

D15_TFRECORD="/oscar/scratch/dlinsley/15_dist"
D15_PROJECT="CSSM_15dist"
D15_EXTRA="--dataset girik --image_size 32 \
  --tfrecord_dir ${D15_TFRECORD} --project ${D15_PROJECT}"

TOTAL=4
RUN_IDX=0

for EMBED in 32 64; do
  for STATE in 16 64; do
    RUN_IDX=$((RUN_IDX + 1))
    NAME="15_s4nd_full_d1_e${EMBED}_n${STATE}"
    dispatch_run "$NAME" 128 \
      $COMMON_BASE --cssm s4nd_full --embed_dim ${EMBED} --s4nd_d_state ${STATE} \
      --seq_len 32 --max_seq_len 32 $D15_EXTRA
  done
done

echo ""
echo "========================================"
echo "S4ND-LegS 15_dist sweep complete at $(date)"
echo "  Total: $TOTAL  Completed: $COMPLETED  Skipped: $SKIPPED  Failed: $FAILED"
echo "========================================"
