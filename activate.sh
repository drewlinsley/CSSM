#!/bin/bash
# Quick activate script for CSSM environment
# Usage: source activate.sh

ENV_PATH="$HOME/envs/cssm_uv"

# Activate environment
source "$ENV_PATH/bin/activate"

# Set CUDA library path
export LD_LIBRARY_PATH=/oscar/rt/9.6/25/spack/x86_64_v4/cuda-12.6.0-wefccjmiz6tr2sj6lgcxj6ihwvcnqkuo/lib64:$LD_LIBRARY_PATH

# Skip CUDA constraints check for newer GPUs
export JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1

# XLA performance flags for Blackwell (B200). Must be exported BEFORE jax
# is imported. Kept to a conservative set — triton_gemm/autotune/PGLE have
# caused extremely long compiles and CUPTI errors on this system; enable them
# at your own risk by editing this file.
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true \
--xla_gpu_enable_pipelined_all_reduce=true \
--xla_gpu_enable_command_buffer=FUSION,CUSTOM_CALL"

echo "CSSM environment activated (JAX + CUDA 12.6, XLA latency-hiding flags on)"
