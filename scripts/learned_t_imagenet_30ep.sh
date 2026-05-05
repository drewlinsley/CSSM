#!/bin/bash
# 30-epoch supervised ImageNet run matching v5cfg config except rope_mode=learned_t.
# Purpose: test whether a learned (T, C) zero-init temporal embedding lets the model
# discover per-t gate variation that spatiotemporal RoPE cannot surface.
#
# Compare against v5cfg baseline checkpoint at matching epochs (epoch_10, 20, 30_ema)
# and re-run the gate-variation diagnostic (per-t stdev of alpha_t / beta_t).

set -u -o pipefail

PYTHON=/users/dlinsley/envs/cssm_uv/bin/python

cd /files22_lrsresearch/CLPS_Serre_Lab/projects/prj_video_imagenet/CSSM

export CUDA_VISIBLE_DEVICES=1

exec $PYTHON src/training/train_imagenet.py \
    --model cssm_shvit --model_size s1 \
    --cssm_type gdn --delta_key_dim 2 --output_norm rms \
    --gate_type_cssm factored --block_norm global_layer \
    --timm_recipe --augmentation 3aug \
    --epochs 30 --warmup_epochs 3 --image_size 224 \
    --batch_size 256 --lr 1e-3 \
    --weight_decay 0.05 --drop_path_rate 0.1 --grad_clip 1.0 \
    --kernel_size 3 --num_timesteps 8 \
    --rope_mode learned_t \
    --gate_proj_bias_init 0.0 --pos_conv --head_pool mean \
    --data_loader tfrecord \
    --tf_parallel_reads 4 --tf_prefetch_batches 2 --tf_threadpool_size 4 \
    --tfrecord_dir /oscar/scratch/dlinsley/imagenet_tfrecords \
    --checkpoint_dir /oscar/scratch/dlinsley/imagenet_checkpoints \
    --project cssm-imagenet-edge \
    --run_name cssm_gdn_T8_k3_v5cfg_learned_t \
    --save_every 5 --eval_every 1 \
    2>&1 | tee /tmp/learned_t_30ep.log
