#!/bin/bash
# Sweep machine 1: 12 of 24 combinations (224px, stem_layers=2)
# stem_norm_order × learned_init × norm_type × pool_type

BASE="CUDA_VISIBLE_DEVICES=0 python main.py --arch simple --cssm add_kqv --dataset pathfinder \
    --pathfinder_difficulty 14 --tfrecord_dir /oscar/scratch/dlinsley/pathfinder_tfrecords/ \
    --gate_type factored --embed_dim 32 --seq_len 8 --batch_size 256 --lr 3e-4 --epochs 20 \
    --no_wandb --use_complex32 --stem_layers 2 --pos_embed separate --kernel_size 11"

run_experiment() {
    local sno=$1    # stem_norm_order
    local li=$2     # learned_init (0 or 1)
    local nt=$3     # norm_type
    local pt=$4     # pool_type

    local li_flag=""
    local li_tag="noinit"
    if [ "$li" = "1" ]; then
        li_flag="--learned_init"
        li_tag="init"
    fi

    local name="sweep224_${sno}_${li_tag}_${nt}_${pt}"
    echo "========================================"
    echo "Running: $name"
    echo "========================================"

    eval $BASE --stem_norm_order $sno $li_flag --norm_type $nt --pool_type $pt --run_name $name
}

# Machine 1 gets: stem_norm_order=pre (all 12)
for sno in pre; do
    for li in 0 1; do
        for nt in layer batch instance; do
            for pt in mean max; do
                run_experiment $sno $li $nt $pt
            done
        done
    done
done

echo "Machine 1 (224) sweep complete."
