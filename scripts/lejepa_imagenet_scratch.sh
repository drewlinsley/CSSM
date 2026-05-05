#!/bin/bash
# From-scratch recurrent-LeJEPA / SFA training on ImageNet with the TIMM recipe.
# No --resume — random init. Matches the tog9av97 supervised recipe (AdamW, lr=1e-3,
# 3-Augment, mixup, cutmix, label_smoothing, EMA 0.9998, random_erase) but swaps the
# supervised loss for recurrent-LeJEPA: pair_mode=successive + SIGReg (λ=25).
#
# Artifacts:
#   /oscar/scratch/dlinsley/imagenet_checkpoints/cssm_gdn_T8_k3_lejepa_succ_imagenet_scratch/
#   /tmp/lejepa_imagenet_scratch.log

set -u -o pipefail

PYTHON=/users/dlinsley/envs/cssm_uv/bin/python

cd /files22_lrsresearch/CLPS_Serre_Lab/projects/prj_video_imagenet/CSSM

exec $PYTHON src/training/train_imagenet.py \
    --model cssm_shvit --model_size s1 \
    --cssm_type gdn --delta_key_dim 2 --output_norm rms \
    --gate_type_cssm factored --block_norm global_layer \
    --kernel_size 3 --num_timesteps 8 \
    --rope_mode spatiotemporal \
    --gate_proj_bias_init 0.0 --pos_conv --head_pool mean \
    --timm_recipe \
    --epochs 100 --warmup_epochs 5 --image_size 224 \
    --batch_size 256 \
    --drop_path_rate 0.1 --grad_clip 1.0 \
    --precision bf16 \
    --ssl_temporal_loss \
    --ssl_pair_mode successive \
    --ssl_loss_weight 1.0 --ssl_sigreg_weight 25.0 \
    --ssl_proj_dim 64 --ssl_num_slices 1024 --ssl_num_points 17 \
    --probe_lr 1e-3 \
    --data_loader tfrecord \
    --tf_parallel_reads 4 --tf_prefetch_batches 2 --tf_threadpool_size 4 \
    --tfrecord_dir /oscar/scratch/dlinsley/imagenet_tfrecords \
    --checkpoint_dir /oscar/scratch/dlinsley/imagenet_checkpoints \
    --project cssm-imagenet-edge \
    --run_name cssm_gdn_T8_k3_lejepa_succ_imagenet_scratch \
    --save_every 5 --eval_every 999 \
    2>&1 | tee /tmp/lejepa_imagenet_scratch.log
