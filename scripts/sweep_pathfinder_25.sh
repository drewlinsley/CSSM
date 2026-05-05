#!/bin/bash
# Pathfinder CL-25 sweep — CSSM and GDN with 1 and 10 timesteps
#
# Priority order:
#   1. CSSM 1×1 T=1  ← MONITOR THIS. If it works, that's the key finding.
#   2. CSSM 1×1 T=10
#   3. CSSM 11×11 T=1
#   4. CSSM 11×11 T=10
#   5. GDN 1×1 T=1
#   6. GDN 1×1 T=10
#   7. GDN 11×11 T=1
#   8. GDN 11×11 T=10
#
# All use factored gates, embed_dim=64, depth=1, 128px images.
# New W&B project: CSSM_pathfinder_25
#
# Usage:
#   bash scripts/sweep_pathfinder_25.sh 2>&1 | tee -a sweep_pf25_$(hostname).out

set -uo pipefail

PROGRESS_LOG="sweep_pf25_progress.log"
touch "$PROGRESS_LOG"

PROJECT="CSSM_pathfinder_25"
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
        echo "  Cleaning up GPU processes: $pids"
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

# ── Common args ──────────────────────────────────────────────────────────────
COMMON="--dataset pathfinder --pathfinder_difficulty 25 --image_size 128 --stem_layers 1 \
    --embed_dim 64 --pool_type max --pos_embed mrope --norm_type ${GDN_NORM} \
    --drop_path_rate 0.0 --epochs 120 --lr 1e-3 --weight_decay 1e-4 --grad_clip 1.0 \
    --checkpointer simple --stem_norm_order post \
    --project ${PROJECT} --tfrecord_dir ${TFRECORD_DIR} --depth 1"

RUN_IDX=0
TOTAL=10

# ── (1) CSSM 1×1 — T=1 and T=10 ────────────────────────────────────────────
RUN_IDX=$((RUN_IDX + 1))
dispatch_run "pf25_cssm_1x1_t1" 128 \
    --arch simple --cssm gated --gate_type factored --seq_len 1 \
    --kernel_size 1 --short_conv_spatial_size 0 ${COMMON}

RUN_IDX=$((RUN_IDX + 1))
dispatch_run "pf25_cssm_1x1_t10" 128 \
    --arch simple --cssm gated --gate_type factored --seq_len 10 \
    --kernel_size 1 --short_conv_spatial_size 0 ${COMMON}

# ── (2) CSSM 11×11 — T=1 and T=10 ──────────────────────────────────────────
RUN_IDX=$((RUN_IDX + 1))
dispatch_run "pf25_cssm_11x11_t1" 128 \
    --arch simple --cssm gated --gate_type factored --seq_len 1 \
    --kernel_size 11 --short_conv_spatial_size 0 ${COMMON}

RUN_IDX=$((RUN_IDX + 1))
dispatch_run "pf25_cssm_11x11_t10" 128 \
    --arch simple --cssm gated --gate_type factored --seq_len 10 \
    --kernel_size 11 --short_conv_spatial_size 0 ${COMMON}

# ── (3) GDN 1×1 — T=1 and T=10 ─────────────────────────────────────────────
RUN_IDX=$((RUN_IDX + 1))
dispatch_run "pf25_gdn_1x1_t1" 128 \
    --arch simple --cssm gdn --gate_type factored --seq_len 1 \
    --kernel_size 1 --qkv_conv_size 1 --delta_key_dim 2 ${COMMON}

RUN_IDX=$((RUN_IDX + 1))
dispatch_run "pf25_gdn_1x1_t10" 128 \
    --arch simple --cssm gdn --gate_type factored --seq_len 10 \
    --kernel_size 1 --qkv_conv_size 1 --delta_key_dim 2 ${COMMON}

# ── (3) GDN 11×11 — T=1 and T=10 ───────────────────────────────────────────
RUN_IDX=$((RUN_IDX + 1))
dispatch_run "pf25_gdn_11x11_t1" 128 \
    --arch simple --cssm gdn --gate_type factored --seq_len 1 \
    --kernel_size 11 --qkv_conv_size 5 --delta_key_dim 2 ${COMMON}

RUN_IDX=$((RUN_IDX + 1))
dispatch_run "pf25_gdn_11x11_t10" 128 \
    --arch simple --cssm gdn --gate_type factored --seq_len 10 \
    --kernel_size 11 --qkv_conv_size 5 --delta_key_dim 2 ${COMMON}

# ── (4) NoGate CSSM 1×1 — T=1 ──────────────────────────────────────────────
# Same Mamba architecture but NO B/C/Delta gates. Tests whether input-dependent
# frequency modulation is necessary, or if the fixed spectral kernel + SiLU
# nonlinearities + in_proj/out_proj channel mixing suffice.
RUN_IDX=$((RUN_IDX + 1))
dispatch_run "pf25_nogate_1x1_t1" 128 \
    --arch simple --cssm no_gate --seq_len 1 \
    --kernel_size 1 --short_conv_spatial_size 0 ${COMMON}

# NoGate 11×11 T=1 for comparison
RUN_IDX=$((RUN_IDX + 1))
dispatch_run "pf25_nogate_11x11_t1" 128 \
    --arch simple --cssm no_gate --seq_len 1 \
    --kernel_size 11 --short_conv_spatial_size 0 ${COMMON}

echo ""
echo "========================================"
echo "Sweep complete at $(date)"
echo "  Total: $TOTAL  Completed: $COMPLETED  Skipped: $SKIPPED  Failed: $FAILED"
echo "========================================"
