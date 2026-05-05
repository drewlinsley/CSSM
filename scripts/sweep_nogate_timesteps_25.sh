#!/bin/bash
# NoGate CSSM 7×7 timestep sweep on Pathfinder CL-25
#
# Tests whether the spectral kernel alone (no B/C/Delta gates) needs
# temporal recurrence to work. At T=8 it gets 86.1% on CL-25.
# Does it need all 8 steps or can fewer suffice?
#
# 10 runs: T=1..10, kernel_size=7, embed_dim=64, depth=1
#
# Usage:
#   bash scripts/sweep_nogate_timesteps.sh 2>&1 | tee -a sweep_nogate_ts_$(hostname).out

set -uo pipefail

PROGRESS_LOG="sweep_nogate_ts_progress.log"
touch "$PROGRESS_LOG"

PROJECT="CSSM_pathfinder"
TFRECORD_DIR="/oscar/scratch/dlinsley/pathfinder_tfrecords_128"
GDN_NORM="global_layer"

export WANDB_DIR="/oscar/scratch/dlinsley/wandb"
mkdir -p "$WANDB_DIR"

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
    local BS=$START_BS MIN_BS=8 ATTEMPT=0 LOGFILE="/tmp/sweep_run_${NAME}.log"
    while [ "$BS" -ge "$MIN_BS" ]; do
        ATTEMPT=$((ATTEMPT + 1))
        echo "  Attempt $ATTEMPT: batch_size=$BS"
        python main.py "$@" --batch_size ${BS} --run_name ${NAME} 2>&1 | tee "$LOGFILE"
        local EXIT_CODE=${PIPESTATUS[0]}
        if [ $EXIT_CODE -eq 0 ]; then
            echo "  SUCCESS: $NAME (bs=$BS) at $(date)"
            return 0
        fi
        if grep -qiE "out of memory|RESOURCE_EXHAUSTED|oom|cuda error.*memory" "$LOGFILE" 2>/dev/null; then
            echo "  OOM at batch_size=$BS — halving..."
            BS=$((BS / 2))
            kill_gpu_orphans
        else
            echo "  NON-OOM FAILURE (exit=$EXIT_CODE)"
            kill_gpu_orphans
            return 1
        fi
    done
    echo "  GAVE UP: $NAME — OOM even at batch_size=$MIN_BS"
    return 1
}

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

COMMON="--dataset pathfinder --pathfinder_difficulty 25 --image_size 128 --stem_layers 1 \
    --embed_dim 64 --pool_type max --pos_embed mrope --norm_type ${GDN_NORM} \
    --drop_path_rate 0.0 --epochs 120 --lr 1e-3 --weight_decay 1e-4 --grad_clip 1.0 \
    --checkpointer simple --stem_norm_order post \
    --project ${PROJECT} --tfrecord_dir ${TFRECORD_DIR} --depth 1"

RUN_IDX=0
TOTAL=3

for T in 25 50 100; do
    RUN_IDX=$((RUN_IDX + 1))
    dispatch_run "pf_nogate_ks7_t${T}_d1_e64" 128 \
        --arch simple --cssm no_gate --seq_len ${T} \
        --kernel_size 7 --short_conv_spatial_size 0 ${COMMON}
done

echo ""
echo "========================================"
echo "Sweep complete at $(date)"
echo "  Total: $TOTAL  Completed: $COMPLETED  Skipped: $SKIPPED  Failed: $FAILED"
echo "========================================"
