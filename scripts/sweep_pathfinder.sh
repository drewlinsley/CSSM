#!/bin/bash
# Transformer + GDN sweep on Pathfinder 14 (128px)
#
# GDN no-log (8 runs): gdn --use_complex32, depth=1
#   embed_dim in {32,64}, qkv_conv_size in {1,5}, delta_key_dim in {2,4}
#
# GDN grid (8 runs): gdn, lr=1e-3, gate_type=factored, depth=1 only
#   embed_dim in {32,64}, qkv_conv_size in {1,5}, delta_key_dim in {2,4}
#
# Transformer grid (6 runs): spatiotemporal_attn, lr=1e-3, inter_mlp_ratio=4.0
#   depth=1: dim in {32,64}, drop_path=0.0 -- 2 runs
#   depth=3: dim in {32,64}, drop_path in {0.0,0.1} -- 4 runs
#
# GDN-InT spectral grid (6 runs): gdn_int, depth=1 only
#   embed_dim in {32,64}, delta_key_dim in {1,2,4}
#
# GDN-InT elem grid (4 runs): gdn_int_elem, depth=1 only
#   embed_dim in {32,64}, delta_key_dim in {2,4}
#
# GDN-InT qk grid (4 runs): gdn_int_qk, depth=1 only
#   embed_dim in {32,64}, delta_key_dim in {2,4}
#
# CSSM (Spectral Mamba) grid (2 runs): gated, depth=1
#   embed_dim in {32,64}
#
# Mamba-2 Seq grid (4 runs): mamba2_seq, depth=1, seq_len=1
#   embed_dim in {32,64}, state_dim in {16,64}
#
# GDN Seq grid (4 runs): gdn_seq, depth=1, seq_len=1
#   embed_dim in {32,64}, delta_key_dim in {2,4}
#
# ConvSSM grid (4 runs): conv_ssm, depth=1, seq_len=8
#   embed_dim in {32,64}, kernel_size in {3,5}
#
# Total: 52 runs (transformers 60 epochs, all GDN/CSSM/seq variants 120 epochs)
#
# Features:
#   - Auto-retries with halved batch size on OOM (down to min 8)
#   - Kills orphan GPU processes between runs to prevent NCCL corruption
#   - Resumable via progress log -- re-run after machine reset
#
# Usage:
#   bash scripts/sweep_pathfinder.sh 2>&1 | tee -a sweep_pathfinder_$(hostname).out

set -uo pipefail

# -- Configuration -----------------------------------------------------------
PROGRESS_LOG="sweep_pathfinder_progress.log"
touch "$PROGRESS_LOG"

TFRECORD_DIR="/oscar/scratch/dlinsley/pathfinder_tfrecords_128/"
PROJECT="CSSM_pathfinder"

# GDN norm type
GDN_NORM="global_layer"

COMPLETED=0
SKIPPED=0
FAILED=0

# -- GPU cleanup -------------------------------------------------------------
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

# -- Run one config with OOM auto-retry -------------------------------------
# Usage: run_with_retry NAME START_BS [extra python args...]
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

# -- Helper: dispatch a named run -------------------------------------------
dispatch_run() {
    local NAME=$1 START_BS=$2
    shift 2

    # Skip if completed or claimed
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

# -- Pathfinder-specific common args -----------------------------------------
# Static 128px images, repeated seq_len=8 times. Default stem (4x downsample).
PF_COMMON="--dataset pathfinder --pathfinder_difficulty 14 --image_size 128 --seq_len 8 --stem_layers 1"
# For seq models (mamba2_seq, gdn_seq) that flatten to 1D: use seq_len=1 (single image)
PF_SEQ="--dataset pathfinder --pathfinder_difficulty 14 --image_size 128 --seq_len 1 --stem_layers 1"

# -- Build run list ----------------------------------------------------------
RUN_IDX=0
TOTAL=94

# ============================================================================
# PRIORITY: GDN no-log (--use_complex32) -- 12 runs
# ============================================================================
for DIM in 32 64; do
  for CONV in 1 5; do
    for DK in 1 2 4; do
      RUN_IDX=$((RUN_IDX + 1))
      NAME="pf_gdn_nolog_d1_e${DIM}_qkv${CONV}_dk${DK}"

      dispatch_run "$NAME" 128 \
          --arch simple --cssm gdn ${PF_COMMON} \
          --embed_dim ${DIM} --gate_type factored \
          --kernel_size 11 --pool_type max --pos_embed mrope \
          --norm_type ${GDN_NORM} --use_complex32 \
          --drop_path_rate 0.0 \
          --epochs 120 --lr 1e-3 --weight_decay 1e-4 --grad_clip 1.0 \
          --checkpointer simple --stem_norm_order post \
          --project ${PROJECT} --tfrecord_dir ${TFRECORD_DIR} \
          --depth 1 --qkv_conv_size ${CONV} --delta_key_dim ${DK}
    done
  done
done

# ============================================================================
# GDN (depth=1, log-space) -- 12 runs
# ============================================================================
for DIM in 32 64; do
  for CONV in 1 5; do
    for DK in 1 2 4; do
      RUN_IDX=$((RUN_IDX + 1))
      NAME="pf_gdn_d1_e${DIM}_qkv${CONV}_dk${DK}"

      dispatch_run "$NAME" 128 \
          --arch simple --cssm gdn ${PF_COMMON} \
          --embed_dim ${DIM} --gate_type factored \
          --kernel_size 11 --pool_type max --pos_embed mrope \
          --norm_type ${GDN_NORM} \
          --drop_path_rate 0.0 \
          --epochs 120 --lr 1e-3 --weight_decay 1e-4 --grad_clip 1.0 \
          --checkpointer simple --stem_norm_order post \
          --project ${PROJECT} --tfrecord_dir ${TFRECORD_DIR} \
          --depth 1 --qkv_conv_size ${CONV} --delta_key_dim ${DK}
    done
  done
done

# ============================================================================
# Transformers -- spatial-only attention (Pathfinder is static images)
# depth=1, dim x lr x heads sweep = 14 runs
#   e32: heads in {1,2,4}, lr in {1e-3,3e-4} = 6 runs
#   e64: heads in {1,2,4,8}, lr in {1e-3,3e-4} = 8 runs
# ============================================================================
for DIM in 32 64; do
  if [ "$DIM" = "32" ]; then HEAD_LIST="1 2 4"; else HEAD_LIST="1 2 4 8"; fi

  for LR in 1e-3 3e-4; do
    LR_SHORT=$(echo "$LR" | sed 's/[-.]//g')

    for HEADS in $HEAD_LIST; do
      RUN_IDX=$((RUN_IDX + 1))
      NAME="pf_sa_d1_e${DIM}_lr${LR_SHORT}_h${HEADS}"

      dispatch_run "$NAME" 128 \
          --arch simple --cssm spatial_attn ${PF_COMMON} \
          --embed_dim ${DIM} --num_heads ${HEADS} \
          --mlp_ratio 4.0 --drop_path_rate 0.0 --norm_type layer --pool_type max \
          --frame_readout all --pos_embed mrope --act_type gelu \
          --inter_mlp_ratio 4.0 \
          --epochs 60 --weight_decay 1e-4 --grad_clip 1.0 \
          --checkpointer simple --stem_norm_order post \
          --project ${PROJECT} --tfrecord_dir ${TFRECORD_DIR} \
          --depth 1 --lr ${LR}
    done
  done
done

# ============================================================================
# CSSM / Spectral Mamba (depth=1): dim in {32,64} -- 2 runs
# ============================================================================
for DIM in 32 64; do
  RUN_IDX=$((RUN_IDX + 1))
  NAME="pf_cssm_d1_e${DIM}"

  dispatch_run "$NAME" 128 \
      --arch simple --cssm gated --gate_type factored ${PF_COMMON} \
      --embed_dim ${DIM} \
      --kernel_size 11 --pool_type max --pos_embed mrope \
      --norm_type ${GDN_NORM} \
      --drop_path_rate 0.0 \
      --epochs 120 --lr 1e-3 --weight_decay 1e-4 --grad_clip 1.0 \
      --checkpointer simple --stem_norm_order post \
      --project ${PROJECT} --tfrecord_dir ${TFRECORD_DIR} \
      --depth 1
done

# ============================================================================
# CSSM / Spectral Mamba 1x1 kernel (depth=1): dim in {32,64} -- 2 runs
# kernel_size=1 = per-channel scalar decay (no spatial mixing in transition)
# ============================================================================
for DIM in 32 64; do
  RUN_IDX=$((RUN_IDX + 1))
  NAME="pf_cssm_1x1_d1_e${DIM}"

  dispatch_run "$NAME" 128 \
      --arch simple --cssm gated --gate_type factored ${PF_COMMON} \
      --embed_dim ${DIM} \
      --kernel_size 1 --short_conv_spatial_size 0 \
      --pool_type max --pos_embed mrope \
      --norm_type ${GDN_NORM} \
      --drop_path_rate 0.0 \
      --epochs 120 --lr 1e-3 --weight_decay 1e-4 --grad_clip 1.0 \
      --checkpointer simple --stem_norm_order post \
      --project ${PROJECT} --tfrecord_dir ${TFRECORD_DIR} \
      --depth 1
done

# ============================================================================
# CSSM / Spectral Mamba full-kernel (depth=1): dim in {32,64} -- 2 runs
# kernel_size=64 = full latent (128px, stem_layers=1, 2x pool → 64×64)
# ============================================================================
for DIM in 32 64; do
  RUN_IDX=$((RUN_IDX + 1))
  NAME="pf_cssm_full_d1_e${DIM}"

  dispatch_run "$NAME" 128 \
      --arch simple --cssm gated --gate_type factored ${PF_COMMON} \
      --embed_dim ${DIM} \
      --kernel_size 64 --pool_type max --pos_embed mrope \
      --norm_type ${GDN_NORM} \
      --drop_path_rate 0.0 \
      --epochs 120 --lr 1e-3 --weight_decay 1e-4 --grad_clip 1.0 \
      --checkpointer simple --stem_norm_order post \
      --project ${PROJECT} --tfrecord_dir ${TFRECORD_DIR} \
      --depth 1
done

# ============================================================================
# Mamba-2 Seq (depth=1): dim × state_dim -- 4 runs
# Pure 1D sequence model on flattened tokens. seq_len=1 for Pathfinder.
# ============================================================================
for DIM in 32 64; do
  for SD in 16 64; do
    RUN_IDX=$((RUN_IDX + 1))
    NAME="pf_m2seq_d1_e${DIM}_n${SD}"

    dispatch_run "$NAME" 128 \
        --arch simple --cssm mamba2_seq ${PF_SEQ} \
        --embed_dim ${DIM} --state_dim ${SD} \
        --pool_type max --pos_embed mrope \
        --norm_type ${GDN_NORM} \
        --drop_path_rate 0.0 \
        --epochs 120 --lr 1e-3 --weight_decay 1e-4 --grad_clip 1.0 \
        --checkpointer simple --stem_norm_order post \
        --project ${PROJECT} --tfrecord_dir ${TFRECORD_DIR} \
        --depth 1
  done
done

# ============================================================================
# GDN Seq (depth=1): dim × dk -- 4 runs
# Pure 1D delta rule on flattened tokens. seq_len=1 for Pathfinder.
# ============================================================================
for DIM in 32 64; do
  for DK in 2 4; do
    RUN_IDX=$((RUN_IDX + 1))
    NAME="pf_gdnseq_d1_e${DIM}_dk${DK}"

    dispatch_run "$NAME" 128 \
        --arch simple --cssm gdn_seq ${PF_SEQ} \
        --embed_dim ${DIM} --delta_key_dim ${DK} \
        --pool_type max --pos_embed mrope \
        --norm_type ${GDN_NORM} \
        --drop_path_rate 0.0 \
        --epochs 120 --lr 1e-3 --weight_decay 1e-4 --grad_clip 1.0 \
        --checkpointer simple --stem_norm_order post \
        --project ${PROJECT} --tfrecord_dir ${TFRECORD_DIR} \
        --depth 1
  done
done

# ============================================================================
# ConvSSM (depth=1): dim × kernel_size -- 4 runs
# Spatiotemporal: spatial conv + temporal scan. seq_len=8 (repeat image).
# ============================================================================
for DIM in 32 64; do
  for KS in 3 5; do
    RUN_IDX=$((RUN_IDX + 1))
    NAME="pf_convssm_d1_e${DIM}_ks${KS}"

    dispatch_run "$NAME" 128 \
        --arch simple --cssm conv_ssm ${PF_COMMON} \
        --embed_dim ${DIM} --kernel_size ${KS} \
        --pool_type max --pos_embed mrope \
        --norm_type ${GDN_NORM} \
        --drop_path_rate 0.0 \
        --epochs 120 --lr 1e-3 --weight_decay 1e-4 --grad_clip 1.0 \
        --checkpointer simple --stem_norm_order post \
        --project ${PROJECT} --tfrecord_dir ${TFRECORD_DIR} \
        --depth 1
  done
done

# ============================================================================
# Frame readout=all ablation: CSSM + best GDNs -- 10 runs
# Tests whether pooling ALL timestep outputs helps vs just the last frame.
# On static images (repeated T=8), last vs all shouldn't differ much,
# but this confirms the readout isn't a bottleneck.
# ============================================================================

# CSSM readout=all: dim in {32,64} -- 2 runs
for DIM in 32 64; do
  RUN_IDX=$((RUN_IDX + 1))
  NAME="pf_cssm_all_d1_e${DIM}"

  dispatch_run "$NAME" 128 \
      --arch simple --cssm gated --gate_type factored ${PF_COMMON} \
      --embed_dim ${DIM} \
      --kernel_size 11 --pool_type max --pos_embed mrope \
      --frame_readout all --norm_type ${GDN_NORM} \
      --drop_path_rate 0.0 \
      --epochs 120 --lr 1e-3 --weight_decay 1e-4 --grad_clip 1.0 \
      --checkpointer simple --stem_norm_order post \
      --project ${PROJECT} --tfrecord_dir ${TFRECORD_DIR} \
      --depth 1
done

# CSSM full-kernel readout=all: dim in {32,64} -- 2 runs
for DIM in 32 64; do
  RUN_IDX=$((RUN_IDX + 1))
  NAME="pf_cssm_full_all_d1_e${DIM}"

  dispatch_run "$NAME" 128 \
      --arch simple --cssm gated --gate_type factored ${PF_COMMON} \
      --embed_dim ${DIM} \
      --kernel_size 64 --pool_type max --pos_embed mrope \
      --frame_readout all --norm_type ${GDN_NORM} \
      --drop_path_rate 0.0 \
      --epochs 120 --lr 1e-3 --weight_decay 1e-4 --grad_clip 1.0 \
      --checkpointer simple --stem_norm_order post \
      --project ${PROJECT} --tfrecord_dir ${TFRECORD_DIR} \
      --depth 1
done

# Best GDNs readout=all: top configs from sweep -- 6 runs
# e64: qkv5 x dk{1,2,4} (top 3 overall)
for DK in 1 2 4; do
  RUN_IDX=$((RUN_IDX + 1))
  NAME="pf_gdn_all_d1_e64_qkv5_dk${DK}"

  dispatch_run "$NAME" 128 \
      --arch simple --cssm gdn ${PF_COMMON} \
      --embed_dim 64 --gate_type factored \
      --kernel_size 11 --pool_type max --pos_embed mrope \
      --frame_readout all --norm_type ${GDN_NORM} \
      --drop_path_rate 0.0 \
      --epochs 120 --lr 1e-3 --weight_decay 1e-4 --grad_clip 1.0 \
      --checkpointer simple --stem_norm_order post \
      --project ${PROJECT} --tfrecord_dir ${TFRECORD_DIR} \
      --depth 1 --qkv_conv_size 5 --delta_key_dim ${DK}
done

# e32: qkv5 x dk{1,2,4} (best e32 configs)
for DK in 1 2 4; do
  RUN_IDX=$((RUN_IDX + 1))
  NAME="pf_gdn_all_d1_e32_qkv5_dk${DK}"

  dispatch_run "$NAME" 128 \
      --arch simple --cssm gdn ${PF_COMMON} \
      --embed_dim 32 --gate_type factored \
      --kernel_size 11 --pool_type max --pos_embed mrope \
      --frame_readout all --norm_type ${GDN_NORM} \
      --drop_path_rate 0.0 \
      --epochs 120 --lr 1e-3 --weight_decay 1e-4 --grad_clip 1.0 \
      --checkpointer simple --stem_norm_order post \
      --project ${PROJECT} --tfrecord_dir ${TFRECORD_DIR} \
      --depth 1 --qkv_conv_size 5 --delta_key_dim ${DK}
done

# ============================================================================
# No-FFT CSSM (depth=1): dim in {32,64} -- 2 runs
# Same as Spectral Mamba but NO Fourier transform. Pure pixel-domain scan.
# Tests whether FFT provides spatial structure beyond per-pixel gating.
# ============================================================================
for DIM in 32 64; do
  RUN_IDX=$((RUN_IDX + 1))
  NAME="pf_nofft_d1_e${DIM}"

  dispatch_run "$NAME" 128 \
      --arch simple --cssm no_fft ${PF_COMMON} \
      --embed_dim ${DIM} \
      --short_conv_spatial_size 0 \
      --pool_type max --pos_embed mrope \
      --norm_type ${GDN_NORM} \
      --drop_path_rate 0.0 \
      --epochs 120 --lr 1e-3 --weight_decay 1e-4 --grad_clip 1.0 \
      --checkpointer simple --stem_norm_order post \
      --project ${PROJECT} --tfrecord_dir ${TFRECORD_DIR} \
      --depth 1
done

# ============================================================================
# No-FFT CSSM factored gates (depth=1): dim in {32,64} -- 2 runs
# ============================================================================
for DIM in 32 64; do
  RUN_IDX=$((RUN_IDX + 1))
  NAME="pf_nofft_fac_d1_e${DIM}"

  dispatch_run "$NAME" 128 \
      --arch simple --cssm no_fft --gate_type factored ${PF_COMMON} \
      --embed_dim ${DIM} \
      --short_conv_spatial_size 0 \
      --pool_type max --pos_embed mrope \
      --norm_type ${GDN_NORM} \
      --drop_path_rate 0.0 \
      --epochs 120 --lr 1e-3 --weight_decay 1e-4 --grad_clip 1.0 \
      --checkpointer simple --stem_norm_order post \
      --project ${PROJECT} --tfrecord_dir ${TFRECORD_DIR} \
      --depth 1
done

# ============================================================================
# No-Gate CSSM (depth=1): kernel_size × dim -- 4 runs
# FFT + spectral kernel ONLY, no per-frequency B/C/Delta gates.
# Isolates the spatial mixing power of the spectral convolution alone.
# ============================================================================
for DIM in 32 64; do
  for KS in 1 11; do
    RUN_IDX=$((RUN_IDX + 1))
    NAME="pf_nogate_ks${KS}_d1_e${DIM}"

    dispatch_run "$NAME" 128 \
        --arch simple --cssm no_gate ${PF_COMMON} \
        --embed_dim ${DIM} \
        --kernel_size ${KS} --short_conv_spatial_size 0 \
        --pool_type max --pos_embed mrope \
        --norm_type ${GDN_NORM} \
        --drop_path_rate 0.0 \
        --epochs 120 --lr 1e-3 --weight_decay 1e-4 --grad_clip 1.0 \
        --checkpointer simple --stem_norm_order post \
        --project ${PROJECT} --tfrecord_dir ${TFRECORD_DIR} \
        --depth 1
  done
done

# ============================================================================
# T=1 CONTROLS: CSSM and GDN with seq_len=1 (single timestep, no temporal scan)
# Tests whether temporal recurrence is necessary or if pure spatial mixing suffices.
# With T=1: y = C · B · x (simple spectral filter, no resonance amplification)
# With T=8: y = C · (Σ Aᵏ) · B · x (IIR filter with frequency-selective resonance)
# ============================================================================
PF_T1="--dataset pathfinder --pathfinder_difficulty 14 --image_size 128 --seq_len 1 --stem_layers 1"

# CSSM 1×1 factored timestep sweep: T in {1..10}, dim=64 -- 10 runs
# Now uses ACTUAL factored gates (Dense(C→H) × Dense(C→W_freq)) = 34K params vs 427K dense
for T in 1 2 3 4 5 6 7 8 9 10; do
  RUN_IDX=$((RUN_IDX + 1))
  dispatch_run "pf_cssm_1x1_fac_t${T}_d1_e64" 128 \
      --arch simple --cssm gated --gate_type factored \
      --dataset pathfinder --pathfinder_difficulty 14 --image_size 128 --seq_len ${T} --stem_layers 1 \
      --embed_dim 64 \
      --kernel_size 1 --short_conv_spatial_size 0 \
      --pool_type max --pos_embed mrope \
      --norm_type ${GDN_NORM} \
      --drop_path_rate 0.0 \
      --epochs 120 --lr 1e-3 --weight_decay 1e-4 --grad_clip 1.0 \
      --checkpointer simple --stem_norm_order post \
      --project ${PROJECT} --tfrecord_dir ${TFRECORD_DIR} \
      --depth 1
done

# NoFFT factored timestep sweep: T in {1..10}, dim=64 -- 10 runs
# Same architecture as CSSM but NO FFT. Per-pixel gates, no spatial mixing.
for T in 1 2 3 4 5 6 7 8 9 10; do
  RUN_IDX=$((RUN_IDX + 1))
  dispatch_run "pf_nofft_fac_t${T}_d1_e64" 128 \
      --arch simple --cssm no_fft --gate_type factored \
      --dataset pathfinder --pathfinder_difficulty 14 --image_size 128 --seq_len ${T} --stem_layers 1 \
      --embed_dim 64 \
      --short_conv_spatial_size 0 \
      --pool_type max --pos_embed mrope \
      --norm_type ${GDN_NORM} \
      --drop_path_rate 0.0 \
      --epochs 120 --lr 1e-3 --weight_decay 1e-4 --grad_clip 1.0 \
      --checkpointer simple --stem_norm_order post \
      --project ${PROJECT} --tfrecord_dir ${TFRECORD_DIR} \
      --depth 1
done

echo ""
echo "========================================"
echo "Sweep complete at $(date)"
echo "  Total: $TOTAL  Completed: $COMPLETED  Skipped: $SKIPPED  Failed: $FAILED"
echo "========================================"
