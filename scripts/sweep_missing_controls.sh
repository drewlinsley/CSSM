#!/bin/bash
# Fill missing control models across datasets
#
# Part 1: Spatial Transformer on 15-dist PathTracker (8 runs)
#   - Matches pathfinder spatial_attn configs but adapted for pathtracker
#   - depth ∈ {1,3}, dim ∈ {32,64}, head sweep, lr sweep
#
# Part 2: SCSSM (no_gate) on 15-dist PathTracker (10 runs)
#   - kernel_size ∈ {1,7,11}, dim ∈ {32,64}
#   - Plus timestep variations: T=1,4,8 at ks=7 dim=64
#
# Part 3: SCSSM (no_gate) timestep sweep T=1-10 on PF-14 (10 runs)
#   - ks=7, dim=64 (same as existing nogate_timesteps but fresh runs)
#   - Also ks=11 dim=64 T=1-10 (10 runs)
#
# Total: 38 runs
#
# Usage:
#   bash scripts/sweep_missing_controls.sh 2>&1 | tee -a sweep_missing_$(hostname).out

set -uo pipefail

PROGRESS_LOG="sweep_missing_controls_progress.log"
touch "$PROGRESS_LOG"

export WANDB_DIR="/oscar/scratch/dlinsley/wandb"
mkdir -p "$WANDB_DIR"

GDN_NORM="global_layer"

COMPLETED=0
SKIPPED=0
FAILED=0

# ── GPU cleanup ──────────────────────────────────────────────────────────────
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

# ── Common args ──────────────────────────────────────────────────────────────

# 15-dist PathTracker: 32x32 native, 32 frames
COMMON_15DIST="--dataset girik --tfrecord_dir /oscar/scratch/dlinsley/15_dist \
    --seq_len 32 --max_seq_len 32 \
    --stem_mode pathtracker \
    --frame_readout last --pool_type max \
    --pos_embed mrope --stem_norm_order post \
    --drop_path_rate 0.0 --lr 1e-3 --weight_decay 1e-4 --grad_clip 1.0 \
    --checkpointer simple --project CSSM_15dist"

# Pathfinder CL-14: 128x128, repeated image
COMMON_PF14="--dataset pathfinder --pathfinder_difficulty 14 --image_size 128 \
    --stem_layers 1 --embed_dim 64 --pool_type max --pos_embed mrope \
    --norm_type ${GDN_NORM} --drop_path_rate 0.0 \
    --epochs 120 --lr 1e-3 --weight_decay 1e-4 --grad_clip 1.0 \
    --checkpointer simple --stem_norm_order post \
    --project CSSM_pathfinder --tfrecord_dir /oscar/scratch/dlinsley/pathfinder_tfrecords_128 \
    --depth 1"

RUN_IDX=0
TOTAL=38

# ============================================================================
# PART 1: Spatial Transformer on 15-dist (0 runs currently → 8 new runs)
# ============================================================================
echo "──── Part 1: Spatial Transformer on 15-dist ────"

# depth × dim grid (4 runs)
for DEPTH in 1 3; do
  for DIM in 32 64; do
    RUN_IDX=$((RUN_IDX + 1))
    if [ "$DIM" = "32" ]; then HEADS=2; else HEADS=4; fi
    NAME="15dist_sa_d${DEPTH}_e${DIM}"

    dispatch_run "$NAME" 128 \
        --arch simple --cssm spatial_attn ${COMMON_15DIST} \
        --embed_dim ${DIM} --num_heads ${HEADS} \
        --mlp_ratio 4.0 --norm_type layer --act_type gelu \
        --inter_mlp_ratio 4.0 \
        --epochs 60 --depth ${DEPTH}
  done
done

# lr=3e-4 (depth=1) — 2 runs
for DIM in 32 64; do
  RUN_IDX=$((RUN_IDX + 1))
  if [ "$DIM" = "32" ]; then HEADS=2; else HEADS=4; fi
  NAME="15dist_sa_d1_e${DIM}_lr3e4"

  dispatch_run "$NAME" 128 \
      --arch simple --cssm spatial_attn ${COMMON_15DIST} \
      --embed_dim ${DIM} --num_heads ${HEADS} \
      --mlp_ratio 4.0 --norm_type layer --act_type gelu \
      --inter_mlp_ratio 4.0 \
      --epochs 60 --depth 1 --lr 3e-4
done

# head sweep (depth=1, dim=64) — 2 runs
for HEADS in 1 8; do
  RUN_IDX=$((RUN_IDX + 1))
  NAME="15dist_sa_d1_e64_h${HEADS}"

  dispatch_run "$NAME" 128 \
      --arch simple --cssm spatial_attn ${COMMON_15DIST} \
      --embed_dim 64 --num_heads ${HEADS} \
      --mlp_ratio 4.0 --norm_type layer --act_type gelu \
      --inter_mlp_ratio 4.0 \
      --epochs 60 --depth 1
done

# ============================================================================
# PART 2: SCSSM (no_gate) on 15-dist (2 runs currently → 10 new runs)
# ============================================================================
echo "──── Part 2: SCSSM (no_gate) on 15-dist ────"

# kernel_size × dim grid (6 runs: ks ∈ {1,7,11}, dim ∈ {32,64})
for DIM in 32 64; do
  for KS in 1 7 11; do
    RUN_IDX=$((RUN_IDX + 1))
    NAME="15dist_nogate_ks${KS}_d1_e${DIM}_v2"

    dispatch_run "$NAME" 128 \
        --arch simple --cssm no_gate ${COMMON_15DIST} \
        --embed_dim ${DIM} \
        --kernel_size ${KS} --short_conv_spatial_size 0 \
        --norm_type ${GDN_NORM} \
        --seq_len 32 --epochs 120 --depth 1
  done
done

# timestep variations at ks=7, dim=64: T=1,4,8,16 (4 runs)
for T in 1 4 8 16; do
  RUN_IDX=$((RUN_IDX + 1))
  NAME="15dist_nogate_ks7_t${T}_d1_e64"

  dispatch_run "$NAME" 128 \
      --arch simple --cssm no_gate ${COMMON_15DIST} \
      --embed_dim 64 \
      --kernel_size 7 --short_conv_spatial_size 0 \
      --norm_type ${GDN_NORM} \
      --seq_len ${T} --max_seq_len ${T} \
      --epochs 120 --depth 1
done

# ============================================================================
# PART 3: SCSSM (no_gate) timestep sweep T=1-10 on PF-14 (10 runs each)
#   ks=11, dim=64 (complements existing ks=7 sweep)
# ============================================================================
echo "──── Part 3: SCSSM (no_gate) T=1-10 ks=11 on PF-14 ────"

for T in 1 2 3 4 5 6 7 8 9 10; do
  RUN_IDX=$((RUN_IDX + 1))
  NAME="pf_nogate_ks11_t${T}_d1_e64"

  dispatch_run "$NAME" 128 \
      --arch simple --cssm no_gate --seq_len ${T} \
      --kernel_size 11 --short_conv_spatial_size 0 ${COMMON_PF14}
done

echo ""
echo "========================================"
echo "Sweep complete at $(date)"
echo "  Total: $TOTAL  Completed: $COMPLETED  Skipped: $SKIPPED  Failed: $FAILED"
echo "========================================"
