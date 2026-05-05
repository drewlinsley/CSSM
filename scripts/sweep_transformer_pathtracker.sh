#!/bin/bash
# Transformer + GDN sweep on PathTracker (v3: flash attention + GDN models)
#
# Transformer grid (6 runs): spatiotemporal_attn, lr=1e-3, inter_mlp_ratio=4.0
#   depth=1: dim ∈ {32,64}, drop_path=0.0 → 2 runs
#   depth=3: dim ∈ {32,64}, drop_path ∈ {0.0,0.1} → 4 runs
#
# GDN grid (8 runs): gdn, lr=1e-3, gate_type=factored, depth=1 only
#   embed_dim ∈ {32,64}, qkv_conv_size ∈ {1,5}, delta_key_dim ∈ {2,4}
#
# GDN-InT spectral grid (6 runs): gdn_int, depth=1 only
#   embed_dim ∈ {32,64}, delta_key_dim ∈ {1,2,4}
#
# GDN-InT elem grid (4 runs): gdn_int_elem, depth=1 only
#   embed_dim ∈ {32,64}, delta_key_dim ∈ {2,4}
#
# GDN-InT qk grid (4 runs): gdn_int_qk, depth=1 only
#   embed_dim ∈ {32,64}, delta_key_dim ∈ {2,4}
#
# CSSM / Spectral Mamba (2 runs): gated, depth=1
#   embed_dim ∈ {32,64}
#
# Mamba-2 Seq (4 runs): mamba2_seq, dim ∈ {32,64}, state_dim ∈ {16,64}
# GDN Seq (4 runs): gdn_seq, dim ∈ {32,64}, dk ∈ {2,4}
# ConvSSM (4 runs): conv_ssm, dim ∈ {32,64}, ks ∈ {3,5}
#
# Total: 55 runs
# Ordering: depth ascending, transformers before GDN within each depth level
#
# Features:
#   - Auto-retries with halved batch size on OOM (down to min 8)
#   - Kills orphan GPU processes between runs to prevent NCCL corruption
#   - Resumable via progress log — re-run after machine reset
#
# Usage:
#   bash scripts/sweep_transformer_pathtracker.sh 2>&1 | tee -a sweep_transformer_$(hostname).out

set -uo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
PROGRESS_LOG="sweep_transformer_progress.log"
touch "$PROGRESS_LOG"

TFRECORD_DIR="/cifs/data/tserre_lrs/projects/projects/prj_video_datasets/pathtracker_equal_large_tfrecords"
PROJECT="CSSM_pathtracker"

# GDN norm type — global_layer chosen (reduce T,H,W,C → stats per B)
# Norm sweep: temporal_layer 0.6386, layer 0.6360, global_layer 0.6357, global_instance 0.6348, batch 0.5990, instance 0.5001
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

# ── Build run list (depth-ascending, transformers before GDN) ────────────────
RUN_IDX=0
TOTAL=57

for DEPTH in 1 3; do

  # ── Transformer runs ─────────────────────────────────────────────────────
  for DIM in 32 64; do
    if [ "$DEPTH" -eq 1 ]; then
      DROP_PATHS="0.0"
    else
      DROP_PATHS="0.0 0.1"
    fi

    for DROP_PATH in $DROP_PATHS; do
      RUN_IDX=$((RUN_IDX + 1))

      DP_SHORT=$(echo "$DROP_PATH" | sed 's/\.//g')
      NAME="st_d${DEPTH}_e${DIM}_dp${DP_SHORT}_mlp40"

      if [ "$DIM" = "32" ]; then HEADS=2; else HEADS=4; fi

      dispatch_run "$NAME" 128 \
          --arch simple --cssm spatiotemporal_attn --dataset pathtracker \
          --stem_mode pathtracker --embed_dim ${DIM} --num_heads ${HEADS} \
          --mlp_ratio 4.0 --drop_path_rate ${DROP_PATH} --norm_type layer --pool_type max \
          --frame_readout all --pos_embed mrope --seq_len 64 --max_seq_len 64 --act_type gelu \
          --inter_mlp_ratio 4.0 \
          --epochs 60 --weight_decay 1e-4 --grad_clip 1.0 \
          --checkpointer simple --stem_norm_order post \
          --project ${PROJECT} --tfrecord_dir ${TFRECORD_DIR} \
          --depth ${DEPTH} --lr 1e-3
    done
  done

  # ── GDN runs (depth=1 only) ───────────────────────────────────────────────
  if [ "$DEPTH" -eq 1 ]; then
    for DIM in 32 64; do
      for CONV in 1 5; do
        for DK in 1 2 4; do
          RUN_IDX=$((RUN_IDX + 1))

          NAME="gdn_d${DEPTH}_e${DIM}_qkv${CONV}_dk${DK}"

          dispatch_run "$NAME" 128 \
              --arch simple --cssm gdn --dataset pathtracker \
              --stem_mode pathtracker --embed_dim ${DIM} --gate_type factored \
              --kernel_size 11 --pool_type max --pos_embed mrope \
              --seq_len 64 --max_seq_len 64 --norm_type ${GDN_NORM} \
              --drop_path_rate 0.0 \
              --epochs 60 --lr 1e-3 --weight_decay 1e-4 --grad_clip 1.0 \
              --checkpointer simple --stem_norm_order post \
              --project ${PROJECT} --tfrecord_dir ${TFRECORD_DIR} \
              --depth 1 --qkv_conv_size ${CONV} --delta_key_dim ${DK}
        done
      done
    done
  fi

done

# ── Spatial-only attention ablation (depth=1): dim in {32,64} -- 2 runs ──────
for DIM in 32 64; do
  RUN_IDX=$((RUN_IDX + 1))
  if [ "$DIM" = "32" ]; then HEADS=2; else HEADS=4; fi
  NAME="sa_d1_e${DIM}"

  dispatch_run "$NAME" 128 \
      --arch simple --cssm spatial_attn --dataset pathtracker \
      --stem_mode pathtracker --embed_dim ${DIM} --num_heads ${HEADS} \
      --mlp_ratio 4.0 --drop_path_rate 0.0 --norm_type layer --pool_type max \
      --frame_readout all --pos_embed mrope --seq_len 64 --max_seq_len 64 --act_type gelu \
      --inter_mlp_ratio 4.0 \
      --epochs 60 --weight_decay 1e-4 --grad_clip 1.0 \
      --checkpointer simple --stem_norm_order post \
      --project ${PROJECT} --tfrecord_dir ${TFRECORD_DIR} \
      --depth 1 --lr 1e-3
done

# ── Spatiotemporal lr=3e-4 (depth=1): dim in {32,64} -- 2 runs ──────────────
for DIM in 32 64; do
  RUN_IDX=$((RUN_IDX + 1))
  if [ "$DIM" = "32" ]; then HEADS=2; else HEADS=4; fi
  NAME="st_d1_e${DIM}_lr3e4"

  dispatch_run "$NAME" 128 \
      --arch simple --cssm spatiotemporal_attn --dataset pathtracker \
      --stem_mode pathtracker --embed_dim ${DIM} --num_heads ${HEADS} \
      --mlp_ratio 4.0 --drop_path_rate 0.0 --norm_type layer --pool_type max \
      --frame_readout all --pos_embed mrope --seq_len 64 --max_seq_len 64 --act_type gelu \
      --inter_mlp_ratio 4.0 \
      --epochs 60 --weight_decay 1e-4 --grad_clip 1.0 \
      --checkpointer simple --stem_norm_order post \
      --project ${PROJECT} --tfrecord_dir ${TFRECORD_DIR} \
      --depth 1 --lr 3e-4
done

# ── Spatiotemporal head sweep (depth=1, lr=1e-3): heads in {1,4,8} -- 6 runs ─
# e32 default=2, add {1,4}; e64 default=4, add {1,2,8}
for DIM in 32 64; do
  if [ "$DIM" = "32" ]; then HEAD_LIST="1 4"; else HEAD_LIST="1 2 8"; fi

  for HEADS in $HEAD_LIST; do
    RUN_IDX=$((RUN_IDX + 1))
    NAME="st_d1_e${DIM}_h${HEADS}"

    dispatch_run "$NAME" 128 \
        --arch simple --cssm spatiotemporal_attn --dataset pathtracker \
        --stem_mode pathtracker --embed_dim ${DIM} --num_heads ${HEADS} \
        --mlp_ratio 4.0 --drop_path_rate 0.0 --norm_type layer --pool_type max \
        --frame_readout all --pos_embed mrope --seq_len 64 --max_seq_len 64 --act_type gelu \
        --inter_mlp_ratio 4.0 \
        --epochs 60 --weight_decay 1e-4 --grad_clip 1.0 \
        --checkpointer simple --stem_norm_order post \
        --project ${PROJECT} --tfrecord_dir ${TFRECORD_DIR} \
        --depth 1 --lr 1e-3
  done
done

# ── GDN-InT runs (depth=1 only): dk ∈ {1,2,4}, dim ∈ {32,64} = 6 runs ──────
for DIM in 32 64; do
  for DK in 1 2 4; do
    RUN_IDX=$((RUN_IDX + 1))

    NAME="gdnint_d1_e${DIM}_dk${DK}"

    dispatch_run "$NAME" 128 \
        --arch simple --cssm gdn_int --dataset pathtracker \
        --stem_mode pathtracker --embed_dim ${DIM} --gate_type factored \
        --kernel_size 11 --pool_type max --pos_embed mrope \
        --seq_len 64 --max_seq_len 64 --norm_type ${GDN_NORM} --use_complex32 \
        --drop_path_rate 0.0 \
        --epochs 120 --lr 1e-3 --weight_decay 1e-4 --grad_clip 1.0 \
        --checkpointer simple --stem_norm_order post \
        --project ${PROJECT} --tfrecord_dir ${TFRECORD_DIR} \
        --depth 1 --delta_key_dim ${DK}
  done
done

# ── GDN-InT elem (depth=1): dk ∈ {2,4}, dim ∈ {32,64} = 4 runs ─────────────
for DIM in 32 64; do
  for DK in 2 4; do
    RUN_IDX=$((RUN_IDX + 1))
    NAME="gdnint_elem_d1_e${DIM}_dk${DK}"

    dispatch_run "$NAME" 128 \
        --arch simple --cssm gdn_int_elem --dataset pathtracker \
        --stem_mode pathtracker --embed_dim ${DIM} --gate_type factored \
        --kernel_size 11 --pool_type max --pos_embed mrope \
        --seq_len 64 --max_seq_len 64 --norm_type ${GDN_NORM} --use_complex32 \
        --drop_path_rate 0.0 \
        --epochs 120 --lr 1e-3 --weight_decay 1e-4 --grad_clip 1.0 \
        --checkpointer simple --stem_norm_order post \
        --project ${PROJECT} --tfrecord_dir ${TFRECORD_DIR} \
        --depth 1 --delta_key_dim ${DK}
  done
done

# ── GDN-InT qk (depth=1): dk ∈ {2,4}, dim ∈ {32,64} = 4 runs ──────────────
for DIM in 32 64; do
  for DK in 2 4; do
    RUN_IDX=$((RUN_IDX + 1))
    NAME="gdnint_qk_d1_e${DIM}_dk${DK}"

    dispatch_run "$NAME" 128 \
        --arch simple --cssm gdn_int_qk --dataset pathtracker \
        --stem_mode pathtracker --embed_dim ${DIM} --gate_type factored \
        --kernel_size 11 --pool_type max --pos_embed mrope \
        --seq_len 64 --max_seq_len 64 --norm_type ${GDN_NORM} --use_complex32 \
        --drop_path_rate 0.0 \
        --epochs 120 --lr 1e-3 --weight_decay 1e-4 --grad_clip 1.0 \
        --checkpointer simple --stem_norm_order post \
        --project ${PROJECT} --tfrecord_dir ${TFRECORD_DIR} \
        --depth 1 --delta_key_dim ${DK}
  done
done

# ── CSSM / Spectral Mamba (depth=1): dim ∈ {32,64} — 2 runs ─────────────────
for DIM in 32 64; do
  RUN_IDX=$((RUN_IDX + 1))
  NAME="cssm_d1_e${DIM}"

  dispatch_run "$NAME" 128 \
      --arch simple --cssm gated --dataset pathtracker \
      --stem_mode pathtracker --embed_dim ${DIM} \
      --kernel_size 11 --pool_type max --pos_embed mrope \
      --seq_len 64 --max_seq_len 64 --norm_type ${GDN_NORM} \
      --drop_path_rate 0.0 \
      --epochs 120 --lr 1e-3 --weight_decay 1e-4 --grad_clip 1.0 \
      --checkpointer simple --stem_norm_order post \
      --project ${PROJECT} --tfrecord_dir ${TFRECORD_DIR} \
      --depth 1
done

# ── CSSM / Spectral Mamba full-kernel (depth=1): dim ∈ {32,64} — 2 runs ──────
# kernel_size=32 = full latent (pathtracker stem, no downsample, 32×32)
for DIM in 32 64; do
  RUN_IDX=$((RUN_IDX + 1))
  NAME="cssm_full_d1_e${DIM}"

  dispatch_run "$NAME" 128 \
      --arch simple --cssm gated --dataset pathtracker \
      --stem_mode pathtracker --embed_dim ${DIM} \
      --kernel_size 32 --pool_type max --pos_embed mrope \
      --seq_len 64 --max_seq_len 64 --norm_type ${GDN_NORM} \
      --drop_path_rate 0.0 \
      --epochs 120 --lr 1e-3 --weight_decay 1e-4 --grad_clip 1.0 \
      --checkpointer simple --stem_norm_order post \
      --project ${PROJECT} --tfrecord_dir ${TFRECORD_DIR} \
      --depth 1
done

# ── Mamba-2 Seq (depth=1): dim × state_dim — 4 runs ─────────────────────────
# Pure 1D sequence on flattened T*H*W tokens. Long sequences (64×32×32=65536).
for DIM in 32 64; do
  for SD in 16 64; do
    RUN_IDX=$((RUN_IDX + 1))
    NAME="m2seq_d1_e${DIM}_n${SD}"

    dispatch_run "$NAME" 64 \
        --arch simple --cssm mamba2_seq --dataset pathtracker \
        --stem_mode pathtracker --embed_dim ${DIM} --state_dim ${SD} \
        --pool_type max --pos_embed mrope \
        --seq_len 64 --max_seq_len 64 --norm_type ${GDN_NORM} \
        --drop_path_rate 0.0 \
        --epochs 120 --lr 1e-3 --weight_decay 1e-4 --grad_clip 1.0 \
        --checkpointer simple --stem_norm_order post \
        --project ${PROJECT} --tfrecord_dir ${TFRECORD_DIR} \
        --depth 1
  done
done

# ── GDN Seq (depth=1): dim × dk — 4 runs ────────────────────────────────────
for DIM in 32 64; do
  for DK in 2 4; do
    RUN_IDX=$((RUN_IDX + 1))
    NAME="gdnseq_d1_e${DIM}_dk${DK}"

    dispatch_run "$NAME" 64 \
        --arch simple --cssm gdn_seq --dataset pathtracker \
        --stem_mode pathtracker --embed_dim ${DIM} --delta_key_dim ${DK} \
        --pool_type max --pos_embed mrope \
        --seq_len 64 --max_seq_len 64 --norm_type ${GDN_NORM} \
        --drop_path_rate 0.0 \
        --epochs 120 --lr 1e-3 --weight_decay 1e-4 --grad_clip 1.0 \
        --checkpointer simple --stem_norm_order post \
        --project ${PROJECT} --tfrecord_dir ${TFRECORD_DIR} \
        --depth 1
  done
done

# ── ConvSSM (depth=1): dim × kernel_size — 4 runs ───────────────────────────
for DIM in 32 64; do
  for KS in 3 5; do
    RUN_IDX=$((RUN_IDX + 1))
    NAME="convssm_d1_e${DIM}_ks${KS}"

    dispatch_run "$NAME" 128 \
        --arch simple --cssm conv_ssm --dataset pathtracker \
        --stem_mode pathtracker --embed_dim ${DIM} --kernel_size ${KS} \
        --pool_type max --pos_embed mrope \
        --seq_len 64 --max_seq_len 64 --norm_type ${GDN_NORM} \
        --drop_path_rate 0.0 \
        --epochs 120 --lr 1e-3 --weight_decay 1e-4 --grad_clip 1.0 \
        --checkpointer simple --stem_norm_order post \
        --project ${PROJECT} --tfrecord_dir ${TFRECORD_DIR} \
        --depth 1
  done
done

echo ""
echo "========================================"
echo "Sweep complete at $(date)"
echo "  Total: $TOTAL  Completed: $COMPLETED  Skipped: $SKIPPED  Failed: $FAILED"
echo "========================================"
