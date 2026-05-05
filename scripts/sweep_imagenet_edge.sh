#!/bin/bash
# ImageNet-1k edge model comparison sweep
#
# Compares SHViT (attention) vs CSSM-SHViT (spectral Mamba) on ImageNet-1k.
# CSSM replaces single-head attention in stages 2-3 with GatedCSSM.
#
# Models (6 runs):
#   SHViT-S1 baseline (attention, ~6M params)
#   CSSM-SHViT-S1 1x1 (kernel_size=1, no spatial conv — pure FFT spatial mixing)
#   CSSM-SHViT-S1 5x5 (kernel_size=5, spatial short conv)
#   SHViT-S4 baseline (attention, ~22M params)
#   CSSM-SHViT-S4 1x1
#   CSSM-SHViT-S4 5x5
#
# Reference (published SHViT paper numbers):
#   SHViT-S1: 79.4% top-1
#   SHViT-S4: 83.4% top-1
#
# Training: TIMM recipe (AdamW, RandAugment, Mixup/CutMix, label smoothing, EMA)
# 300 epochs, 224x224, cosine LR schedule
#
# Usage:
#   bash scripts/sweep_imagenet_edge.sh 2>&1 | tee -a sweep_imagenet_$(hostname).out

set -uo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
PROGRESS_LOG="sweep_imagenet_edge_progress.log"
touch "$PROGRESS_LOG"

export WANDB_DIR="/oscar/scratch/dlinsley/wandb"
mkdir -p "$WANDB_DIR"

DATA_DIR="/gpfs/data/shared/imagenet/ILSVRC2012"
TFRECORD_DIR="/oscar/scratch/dlinsley/imagenet_tfrecords"
PROJECT="cssm-imagenet-edge"

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
    local MIN_BS=4
    local ATTEMPT=0
    local LOGFILE="/tmp/sweep_run_${NAME}.log"

    while [ "$BS" -ge "$MIN_BS" ]; do
        ATTEMPT=$((ATTEMPT + 1))
        echo "  Attempt $ATTEMPT: batch_size=$BS"

        python src/training/train_imagenet.py "$@" \
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

# ── Common training args (TIMM recipe) ───────────────────────────────────────
COMMON="--timm_recipe --epochs 300 --image_size 224 \
    --lr 1e-3 --weight_decay 0.05 --drop_path_rate 0.1 --grad_clip 1.0 \
    --data_loader tfrecord --tfrecord_dir ${TFRECORD_DIR} \
    --project ${PROJECT} --save_every 10 --eval_every 1"

# ── Build run list ───────────────────────────────────────────────────────────
RUN_IDX=0
TOTAL=6

# ── SHViT-S1 baseline (attention, ~6M params) ────────────────────────────────
RUN_IDX=$((RUN_IDX + 1))
dispatch_run "shvit_s1_baseline" 512 \
    --model shvit --model_size s1 ${COMMON}

# ── CSSM-SHViT-S1 1x1 (pure FFT spatial mixing) ─────────────────────────────
RUN_IDX=$((RUN_IDX + 1))
dispatch_run "cssm_shvit_s1_1x1" 512 \
    --model cssm_shvit --model_size s1 ${COMMON} \
    --kernel_size 1 --short_conv_spatial_size 0 --num_timesteps 8

# ── CSSM-SHViT-S1 5x5 (spectral kernel + spatial short conv) ─────────────────
RUN_IDX=$((RUN_IDX + 1))
dispatch_run "cssm_shvit_s1_5x5" 512 \
    --model cssm_shvit --model_size s1 ${COMMON} \
    --kernel_size 5 --short_conv_spatial_size 3 --num_timesteps 8

# ── SHViT-S4 baseline (attention, ~22M params) ───────────────────────────────
RUN_IDX=$((RUN_IDX + 1))
dispatch_run "shvit_s4_baseline" 512 \
    --model shvit --model_size s4 ${COMMON}

# ── CSSM-SHViT-S4 1x1 ───────────────────────────────────────────────────────
RUN_IDX=$((RUN_IDX + 1))
dispatch_run "cssm_shvit_s4_1x1" 512 \
    --model cssm_shvit --model_size s4 ${COMMON} \
    --kernel_size 1 --short_conv_spatial_size 0 --num_timesteps 8

# ── CSSM-SHViT-S4 5x5 ───────────────────────────────────────────────────────
RUN_IDX=$((RUN_IDX + 1))
dispatch_run "cssm_shvit_s4_5x5" 512 \
    --model cssm_shvit --model_size s4 ${COMMON} \
    --kernel_size 5 --short_conv_spatial_size 3 --num_timesteps 8

echo ""
echo "========================================"
echo "Sweep complete at $(date)"
echo "  Total: $TOTAL  Completed: $COMPLETED  Skipped: $SKIPPED  Failed: $FAILED"
echo "========================================"
