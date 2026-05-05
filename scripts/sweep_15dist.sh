#!/bin/bash
# Full model sweep on 15_dist PathTracker (32×32, 32 frames)
#
# Spatiotemporal Transformer grid (8 runs):
#   depth ∈ {1,3}, dim ∈ {32,64}, drop_path ∈ {0.0, 0.1 for d3}
#   + lr sweep: depth=1, dim ∈ {32,64}, lr=3e-4 (2 runs)
#   + head sweep: depth=1, dim=32 h∈{1,4}, dim=64 h∈{1,2,8} (5 runs)
#
# GDN no-log grid (12 runs): gdn --use_complex32
#   dim ∈ {32,64}, qkv ∈ {1,5}, dk ∈ {1,2,4}
#
# GDN log grid (12 runs): gdn
#   dim ∈ {32,64}, qkv ∈ {1,5}, dk ∈ {1,2,4}
#
# GDN-InT spectral (6 runs): dim ∈ {32,64}, dk ∈ {1,2,4}
# GDN-InT elem (4 runs): dim ∈ {32,64}, dk ∈ {2,4}
# GDN-InT qk (4 runs): dim ∈ {32,64}, dk ∈ {2,4}
#
# CSSM / Spectral Mamba (2 runs): dim ∈ {32,64}
# CSSM full-kernel (2 runs): dim ∈ {32,64}, kernel_size=32
#
# Mamba-2 Seq (4 runs): dim ∈ {32,64}, state_dim ∈ {16,64}
# GDN Seq (4 runs): dim ∈ {32,64}, dk ∈ {2,4}
# ConvSSM (4 runs): dim ∈ {32,64}, ks ∈ {3,5}
#
# Total: 67 runs
#
# Usage:
#   bash scripts/sweep_15dist.sh 2>&1 | tee -a sweep_15dist_$(hostname).out

set -uo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
PROGRESS_LOG="sweep_15dist_progress.log"
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
# 32×32 native, 32 frames. pathtracker stem (1×1, no downsample).
# frame_readout=last, pool_type=max.
COMMON="--dataset girik --tfrecord_dir ${TFRECORD_DIR} \
    --seq_len 32 --max_seq_len 32 \
    --stem_mode pathtracker \
    --frame_readout last --pool_type max \
    --pos_embed mrope --stem_norm_order post \
    --drop_path_rate 0.0 --lr 1e-3 --weight_decay 1e-4 --grad_clip 1.0 \
    --checkpointer simple --project ${PROJECT}"

# ── Build run list ───────────────────────────────────────────────────────────
RUN_IDX=0
TOTAL=77

# ============================================================================
# Spatiotemporal Transformers
# ============================================================================

# depth × dim grid (6 runs)
for DEPTH in 1 3; do
  for DIM in 32 64; do
    if [ "$DEPTH" -eq 1 ]; then
      DROP_PATHS="0.0"
    else
      DROP_PATHS="0.0 0.1"
    fi

    for DROP_PATH in $DROP_PATHS; do
      RUN_IDX=$((RUN_IDX + 1))
      DP_SHORT=$(echo "$DROP_PATH" | sed 's/\.//g')
      if [ "$DIM" = "32" ]; then HEADS=2; else HEADS=4; fi
      NAME="15dist_st_d${DEPTH}_e${DIM}_dp${DP_SHORT}"

      dispatch_run "$NAME" 128 \
          --arch simple --cssm spatiotemporal_attn ${COMMON} \
          --embed_dim ${DIM} --num_heads ${HEADS} \
          --mlp_ratio 4.0 --norm_type layer --act_type gelu \
          --inter_mlp_ratio 4.0 \
          --epochs 60 --depth ${DEPTH} --drop_path_rate ${DROP_PATH}
    done
  done
done

# lr=3e-4 (depth=1, dim ∈ {32,64}) — 2 runs
for DIM in 32 64; do
  RUN_IDX=$((RUN_IDX + 1))
  if [ "$DIM" = "32" ]; then HEADS=2; else HEADS=4; fi
  NAME="15dist_st_d1_e${DIM}_lr3e4"

  dispatch_run "$NAME" 128 \
      --arch simple --cssm spatiotemporal_attn ${COMMON} \
      --embed_dim ${DIM} --num_heads ${HEADS} \
      --mlp_ratio 4.0 --norm_type layer --act_type gelu \
      --inter_mlp_ratio 4.0 \
      --epochs 60 --depth 1 --lr 3e-4
done

# Head sweep (depth=1, lr=1e-3) — 5 runs
for DIM in 32 64; do
  if [ "$DIM" = "32" ]; then HEAD_LIST="1 4"; else HEAD_LIST="1 2 8"; fi

  for HEADS in $HEAD_LIST; do
    RUN_IDX=$((RUN_IDX + 1))
    NAME="15dist_st_d1_e${DIM}_h${HEADS}"

    dispatch_run "$NAME" 128 \
        --arch simple --cssm spatiotemporal_attn ${COMMON} \
        --embed_dim ${DIM} --num_heads ${HEADS} \
        --mlp_ratio 4.0 --norm_type layer --act_type gelu \
        --inter_mlp_ratio 4.0 \
        --epochs 60 --depth 1
  done
done

# ============================================================================
# GDN no-log (--use_complex32) — 12 runs
# ============================================================================
for DIM in 32 64; do
  for CONV in 1 5; do
    for DK in 1 2 4; do
      RUN_IDX=$((RUN_IDX + 1))
      NAME="15dist_gdn_nolog_d1_e${DIM}_qkv${CONV}_dk${DK}"

      dispatch_run "$NAME" 128 \
          --arch simple --cssm gdn ${COMMON} \
          --embed_dim ${DIM} --gate_type factored \
          --kernel_size 11 --norm_type ${GDN_NORM} --use_complex32 \
          --epochs 120 --depth 1 --qkv_conv_size ${CONV} --delta_key_dim ${DK}
    done
  done
done

# ============================================================================
# GDN log-space — 12 runs
# ============================================================================
for DIM in 32 64; do
  for CONV in 1 5; do
    for DK in 1 2 4; do
      RUN_IDX=$((RUN_IDX + 1))
      NAME="15dist_gdn_d1_e${DIM}_qkv${CONV}_dk${DK}"

      dispatch_run "$NAME" 128 \
          --arch simple --cssm gdn ${COMMON} \
          --embed_dim ${DIM} --gate_type factored \
          --kernel_size 11 --norm_type ${GDN_NORM} \
          --epochs 120 --depth 1 --qkv_conv_size ${CONV} --delta_key_dim ${DK}
    done
  done
done

# ============================================================================
# GDN-InT spectral — 6 runs
# ============================================================================
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

# ============================================================================
# GDN-InT elem — 4 runs
# ============================================================================
for DIM in 32 64; do
  for DK in 2 4; do
    RUN_IDX=$((RUN_IDX + 1))
    NAME="15dist_gdnint_elem_d1_e${DIM}_dk${DK}"

    dispatch_run "$NAME" 128 \
        --arch simple --cssm gdn_int_elem ${COMMON} \
        --embed_dim ${DIM} --gate_type factored \
        --kernel_size 11 --norm_type ${GDN_NORM} --use_complex32 \
        --epochs 120 --depth 1 --delta_key_dim ${DK}
  done
done

# ============================================================================
# GDN-InT qk — 4 runs
# ============================================================================
for DIM in 32 64; do
  for DK in 2 4; do
    RUN_IDX=$((RUN_IDX + 1))
    NAME="15dist_gdnint_qk_d1_e${DIM}_dk${DK}"

    dispatch_run "$NAME" 128 \
        --arch simple --cssm gdn_int_qk ${COMMON} \
        --embed_dim ${DIM} --gate_type factored \
        --kernel_size 11 --norm_type ${GDN_NORM} --use_complex32 \
        --epochs 120 --depth 1 --delta_key_dim ${DK}
  done
done

# ============================================================================
# CSSM / Spectral Mamba — 2 runs
# ============================================================================
for DIM in 32 64; do
  RUN_IDX=$((RUN_IDX + 1))
  NAME="15dist_cssm_d1_e${DIM}"

  dispatch_run "$NAME" 128 \
      --arch simple --cssm gated --gate_type factored ${COMMON} \
      --embed_dim ${DIM} \
      --kernel_size 11 --norm_type ${GDN_NORM} \
      --epochs 120 --depth 1
done

# ============================================================================
# CSSM 1x1 kernel — 2 runs (per-channel scalar decay, no spatial mixing)
# ============================================================================
for DIM in 32 64; do
  RUN_IDX=$((RUN_IDX + 1))
  NAME="15dist_cssm_1x1_d1_e${DIM}"

  dispatch_run "$NAME" 128 \
      --arch simple --cssm gated --gate_type factored ${COMMON} \
      --embed_dim ${DIM} \
      --kernel_size 1 --short_conv_spatial_size 0 \
      --norm_type ${GDN_NORM} \
      --epochs 120 --depth 1
done

# ============================================================================
# CSSM full-kernel — 2 runs (kernel_size=32 = full latent)
# ============================================================================
for DIM in 32 64; do
  RUN_IDX=$((RUN_IDX + 1))
  NAME="15dist_cssm_full_d1_e${DIM}"

  dispatch_run "$NAME" 128 \
      --arch simple --cssm gated --gate_type factored ${COMMON} \
      --embed_dim ${DIM} \
      --kernel_size 32 --norm_type ${GDN_NORM} \
      --epochs 120 --depth 1
done

# ============================================================================
# Mamba-2 Seq — 4 runs
# ============================================================================
for DIM in 32 64; do
  for SD in 16 64; do
    RUN_IDX=$((RUN_IDX + 1))
    NAME="15dist_m2seq_d1_e${DIM}_n${SD}"

    dispatch_run "$NAME" 64 \
        --arch simple --cssm mamba2_seq ${COMMON} \
        --embed_dim ${DIM} --state_dim ${SD} \
        --norm_type ${GDN_NORM} \
        --epochs 120 --depth 1
  done
done

# ============================================================================
# GDN Seq — 4 runs
# ============================================================================
for DIM in 32 64; do
  for DK in 2 4; do
    RUN_IDX=$((RUN_IDX + 1))
    NAME="15dist_gdnseq_d1_e${DIM}_dk${DK}"

    dispatch_run "$NAME" 64 \
        --arch simple --cssm gdn_seq ${COMMON} \
        --embed_dim ${DIM} --delta_key_dim ${DK} \
        --norm_type ${GDN_NORM} \
        --epochs 120 --depth 1
  done
done

# ============================================================================
# ConvSSM — 6 runs (ks=1 is per-channel scalar decay, ks=3/5 have spatial mixing)
# ============================================================================
for DIM in 32 64; do
  for KS in 1 3 5; do
    RUN_IDX=$((RUN_IDX + 1))
    NAME="15dist_convssm_d1_e${DIM}_ks${KS}"

    dispatch_run "$NAME" 128 \
        --arch simple --cssm conv_ssm ${COMMON} \
        --embed_dim ${DIM} --kernel_size ${KS} \
        --norm_type ${GDN_NORM} \
        --epochs 120 --depth 1
  done
done

# ============================================================================
# No-FFT CSSM — 2 runs (pixel-domain scan, no Fourier transform)
# ============================================================================
for DIM in 32 64; do
  RUN_IDX=$((RUN_IDX + 1))
  NAME="15dist_nofft_d1_e${DIM}"

  dispatch_run "$NAME" 128 \
      --arch simple --cssm no_fft ${COMMON} \
      --embed_dim ${DIM} \
      --short_conv_spatial_size 0 \
      --norm_type ${GDN_NORM} \
      --epochs 120 --depth 1
done

# ============================================================================
# No-FFT CSSM factored gates — 2 runs
# ============================================================================
for DIM in 32 64; do
  RUN_IDX=$((RUN_IDX + 1))
  NAME="15dist_nofft_fac_d1_e${DIM}"

  dispatch_run "$NAME" 128 \
      --arch simple --cssm no_fft --gate_type factored ${COMMON} \
      --embed_dim ${DIM} \
      --short_conv_spatial_size 0 \
      --norm_type ${GDN_NORM} \
      --epochs 120 --depth 1
done

# ============================================================================
# No-Gate CSSM: kernel_size × dim — 4 runs
# FFT + spectral kernel ONLY, no per-frequency B/C/Delta gates.
# ============================================================================
for DIM in 32 64; do
  for KS in 1 11; do
    RUN_IDX=$((RUN_IDX + 1))
    NAME="15dist_nogate_ks${KS}_d1_e${DIM}"

    dispatch_run "$NAME" 128 \
        --arch simple --cssm no_gate ${COMMON} \
        --embed_dim ${DIM} \
        --kernel_size ${KS} --short_conv_spatial_size 0 \
        --norm_type ${GDN_NORM} \
        --epochs 120 --depth 1
  done
done

echo ""
echo "========================================"
echo "Sweep complete at $(date)"
echo "  Total: $TOTAL  Completed: $COMPLETED  Skipped: $SKIPPED  Failed: $FAILED"
echo "========================================"
