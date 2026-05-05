#!/bin/bash
# Pipeline: parallel-copy Ego4D MP4s from cifs to /oscar/scratch, build a local
# manifest, then launch the T=8 k=3 SSL training job. Restart-safe: if the copy
# was partial, the xargs loop skips already-present files.
#
# Artifacts:
#   /oscar/scratch/dlinsley/ego4d_v2_down_scaled/     — destination video dir
#   /oscar/scratch/dlinsley/ego4d_all_local.json      — manifest pointing at it
#   /tmp/ego4d_pipeline.status                        — per-stage status log
#   /tmp/ego4d_copy.log                               — xargs per-file errors
#   /tmp/ego4d_training.log                           — training stdout

set -u -o pipefail

SRC=/cifs/data/tserre_lrs/projects/projects/prj_ego4d/ego4d_data/v2/down_scaled
DST=/oscar/scratch/dlinsley/ego4d_v2_down_scaled
MANIFEST=/oscar/scratch/dlinsley/ego4d_all_local.json
STATUS=/tmp/ego4d_pipeline.status
COPYLOG=/tmp/ego4d_copy.log
PYTHON=/users/dlinsley/envs/cssm_uv/bin/python

mkdir -p "$DST"
: > "$COPYLOG"
echo "START copy $(date -Iseconds)" > "$STATUS"

# =====================================================================
# 1. Parallel copy (12 concurrent rsync workers)
# =====================================================================
EXPECTED=$(ls "$SRC"/*.mp4 2>/dev/null | wc -l)
echo "  expected=$EXPECTED files" >> "$STATUS"

# Use rsync per-file with --ignore-existing so reruns are cheap. xargs -P 12
# gives a good balance between CIFS concurrency and NFS write pressure on
# /oscar/scratch. `-quiet` so a 16k-file list doesn't flood the log.
ls "$SRC"/*.mp4 | xargs -P 12 -I {} bash -c '
  dst="'"$DST"'/$(basename "$1")"
  if [ ! -s "$dst" ]; then
    if ! rsync -a --quiet "$1" "$dst" 2>> "'"$COPYLOG"'"; then
      echo "FAIL $1" >> "'"$COPYLOG"'"
    fi
  fi
' _ {}

COPIED=$(ls "$DST"/*.mp4 2>/dev/null | wc -l)
echo "END copy $(date -Iseconds) copied=$COPIED" >> "$STATUS"

if [ "$COPIED" -lt "$EXPECTED" ]; then
    echo "  INCOMPLETE copy: $COPIED < $EXPECTED" >> "$STATUS"
    # Allow up to 50 missing files (transient cifs hiccups); bail if worse
    MISSING=$((EXPECTED - COPIED))
    if [ "$MISSING" -gt 50 ]; then
        echo "  FATAL: $MISSING files missing, aborting" >> "$STATUS"
        exit 1
    fi
    echo "  continuing with $COPIED/$EXPECTED (tolerable gap)" >> "$STATUS"
fi

# =====================================================================
# 2. Build local manifest
# =====================================================================
echo "BUILD manifest $(date -Iseconds)" >> "$STATUS"
$PYTHON -c "
import os, json
src = '$DST'
paths = sorted(os.path.join(src, f) for f in os.listdir(src) if f.endswith('.mp4'))
videos = [{'path': p, 'video_id': os.path.splitext(os.path.basename(p))[0]} for p in paths]
with open('$MANIFEST', 'w') as f:
    json.dump(videos, f)
print(f'Wrote {len(videos)} entries to $MANIFEST')
" >> "$STATUS" 2>&1

# =====================================================================
# 3. Launch SSL training
# =====================================================================
echo "START training $(date -Iseconds)" >> "$STATUS"
cd /files22_lrsresearch/CLPS_Serre_Lab/projects/prj_video_imagenet/CSSM

exec $PYTHON src/training/train_imagenet.py \
    --model cssm_shvit --model_size s1 \
    --cssm_type gdn --delta_key_dim 2 --output_norm rms \
    --gate_type_cssm factored --block_norm global_layer \
    --epochs 50 --warmup_epochs 2 --image_size 224 \
    --batch_size 64 --lr 1e-4 \
    --weight_decay 0.05 --drop_path_rate 0.1 --grad_clip 1.0 \
    --kernel_size 3 --num_timesteps 8 \
    --rope_mode spatiotemporal \
    --gate_proj_bias_init 0.0 --pos_conv --head_pool mean \
    --ssl_temporal_loss \
    --ssl_pair_mode successive \
    --ssl_loss_weight 1.0 --ssl_sigreg_weight 25.0 \
    --ssl_proj_dim 64 --ssl_num_slices 1024 --ssl_num_points 17 \
    --probe_lr 1e-3 \
    --data_loader ego4d \
    --ego4d_video_dir "$DST" \
    --ego4d_manifest "$MANIFEST" \
    --ego4d_frame_stride 4 \
    --tfrecord_dir /oscar/scratch/dlinsley/imagenet_tfrecords \
    --checkpoint_dir /oscar/scratch/dlinsley/imagenet_checkpoints \
    --resume /oscar/scratch/dlinsley/imagenet_checkpoints/cssm_shvit_s1_gdn_dk2_k3x3_t8_timm_repro/epoch_140 \
    --precision bf16 \
    --project cssm-imagenet-edge \
    --run_name cssm_gdn_T8_k3_lejepa_succ_sig25 \
    --num_workers 16 \
    --save_every 5 --eval_every 999 \
    2>&1 | tee /tmp/ego4d_lejepa_training.log
