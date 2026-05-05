#!/bin/bash
# Sweep on Girik PathTracker (64-frame 128×128 RGB videos)
#
# GDN-InT grid (6 runs): gdn_int, depth=1
#   embed_dim ∈ {32,64}, delta_key_dim ∈ {1,2,4}
#
# GDN grid (4 runs): gdn, depth=1
#   embed_dim ∈ {32,64}, qkv_conv_size ∈ {1,5}, delta_key_dim=4
#
# CSSM / Spectral Mamba (2 runs): gated, depth=1
#   embed_dim ∈ {32,64}
#
# Transformer (2 runs): spatiotemporal_attn, depth=1
#   embed_dim ∈ {32,64}
#
# Total: 14 runs
#
# Data: 128×128 RGB, 64 frames. stem_layers=1 (2x downsample → 64×64 latent).
# All models use frame_readout=all, seq_len=64 (all frames).
#
# Usage:
#   bash scripts/sweep_girik.sh 2>&1 | tee -a sweep_girik_$(hostname).out

set -uo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
PROGRESS_LOG="sweep_girik_progress.log"
touch "$PROGRESS_LOG"

TFRECORD_DIR="/oscar/scratch/dlinsley/15_dist"
PROJECT="CSSM_15dist"
GDN_NORM="global_layer"

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
# 32×32 native, 32 frames. No resize needed.
# pathtracker stem (1×1 conv, no downsample) → 32×32 latent.
# frame_readout=last, pool_type=max (read last frame, max pool over H×W).
COMMON="--dataset girik --tfrecord_dir ${TFRECORD_DIR} \
    --seq_len 32 --max_seq_len 32 --image_size 32 \
    --stem_mode pathtracker \
    --frame_readout last --pool_type max \
    --pos_embed mrope --stem_norm_order post \
    --drop_path_rate 0.0 --lr 1e-3 --weight_decay 1e-4 --grad_clip 1.0 \
    --checkpointer simple --project ${PROJECT}"

# ── Build run list ───────────────────────────────────────────────────────────
RUN_IDX=0
TOTAL=14

# ── Transformers (depth=1): dim ∈ {32,64} — 2 runs ──────────────────────────
for DIM in 32 64; do
  RUN_IDX=$((RUN_IDX + 1))
  if [ "$DIM" = "32" ]; then HEADS=2; else HEADS=4; fi
  NAME="15dist_st_d1_e${DIM}"

  dispatch_run "$NAME" 128 \
      --arch simple --cssm spatiotemporal_attn ${COMMON} \
      --embed_dim ${DIM} --num_heads ${HEADS} \
      --mlp_ratio 4.0 --norm_type layer --act_type gelu \
      --inter_mlp_ratio 4.0 \
      --epochs 60 --depth 1
done

# ── GDN (depth=1): dim × qkv — 4 runs ──────────────────────────────────────
for DIM in 32 64; do
  for CONV in 1 5; do
    RUN_IDX=$((RUN_IDX + 1))
    NAME="15dist_gdn_d1_e${DIM}_qkv${CONV}_dk4"

    dispatch_run "$NAME" 128 \
        --arch simple --cssm gdn ${COMMON} \
        --embed_dim ${DIM} --gate_type factored \
        --kernel_size 11 --norm_type ${GDN_NORM} \
        --epochs 120 --depth 1 --qkv_conv_size ${CONV} --delta_key_dim 4
  done
done

# ── GDN-InT (depth=1): dim × dk — 6 runs ────────────────────────────────────
for DIM in 32 64; do
  for DK in 1 2 4; do
    RUN_IDX=$((RUN_IDX + 1))
    NAME="15dist_gdnint_d1_e${DIM}_dk${DK}"

    dispatch_run "$NAME" 128 \
        --arch simple --cssm gdn_int ${COMMON} \
        --embed_dim ${DIM} --gate_type factored \
        --kernel_size 11 --norm_type ${GDN_NORM} --use_complex32 \
        --epochs 120 --depth 1 --delta_key_dim ${DK}
  done
done

# ── CSSM / Spectral Mamba (depth=1): dim ∈ {32,64} — 2 runs ─────────────────
for DIM in 32 64; do
  RUN_IDX=$((RUN_IDX + 1))
  NAME="15dist_cssm_d1_e${DIM}"

  dispatch_run "$NAME" 128 \
      --arch simple --cssm gated --gate_type factored ${COMMON} \
      --embed_dim ${DIM} \
      --kernel_size 11 --norm_type ${GDN_NORM} \
      --epochs 120 --depth 1
done

echo ""
echo "========================================"
echo "Sweep complete at $(date)"
echo "  Total: $TOTAL  Completed: $COMPLETED  Skipped: $SKIPPED  Failed: $FAILED"
echo "========================================"
