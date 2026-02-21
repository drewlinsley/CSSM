# Multi-GPU Training Guide

This guide covers multi-GPU setup and troubleshooting for CSSM training.

## Automatic Detection

Multi-GPU training is automatic when multiple GPUs are detected:

```bash
python main.py --dataset pathfinder ...
# Output: "Devices: 4 (['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'])"
# Output: "Multi-GPU training enabled: batch_size=256, per_device=64"
```

## NCCL Testing

Before training, the system tests NCCL collective operations:

```
Testing NCCL collective operations...
NCCL test PASSED: pmean result shape=(4, 10), value=1.0000
```

If this fails:
- NCCL library may be misconfigured
- GPUs may not support direct communication

## Forcing Multi-GPU

If NCCL test fails but you want to proceed:

```bash
--force_multi_gpu
```

This bypasses the test (use at your own risk).

## Batch Size Handling

- Batch size is **total** batch size
- Automatically split across GPUs
- Adjusted if not evenly divisible

Example with 4 GPUs:
```bash
--batch_size 256  # 64 per GPU
```

## Common Issues

### 1. NCCL Timeout

**Symptom:** Training hangs during first batch

**Solutions:**
- Set environment variable: `export NCCL_P2P_DISABLE=1`
- Try: `export NCCL_IB_DISABLE=1`
- Reduce batch size

### 2. OOM on Some GPUs

**Symptom:** CUDA OOM error on GPU 1+ but not GPU 0

**Solutions:**
- Reduce batch size
- Check no other processes using GPUs: `nvidia-smi`

### 3. TensorFlow GPU Conflict

**Symptom:** JAX NCCL fails when TF also tries to use GPU

**Note:** The code already handles this by setting:
```python
tf.config.set_visible_devices([], 'GPU')
```

### 4. Slow Training

**Symptom:** Multi-GPU slower than expected

**Check:**
- GPU utilization: `nvidia-smi dmon -s u`
- Should see >90% utilization on all GPUs
- If not, data loading may be bottleneck

**Solutions:**
- Increase `--num_workers`
- Increase `--prefetch_batches`
- Use TFRecord data loading

## Checkpointing with Multi-GPU

State is replicated across GPUs. When saving:
- Only first GPU's copy is saved
- Use `--checkpointer simple` for NFS filesystems

## Environment Variables

Useful NCCL environment variables:

```bash
# Disable peer-to-peer (if communication issues)
export NCCL_P2P_DISABLE=1

# Disable InfiniBand (if IB issues)
export NCCL_IB_DISABLE=1

# Debug NCCL issues
export NCCL_DEBUG=INFO

# Set specific GPU order
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

## Verifying Multi-GPU

Check that all GPUs are being used:

1. **During training:** Watch `nvidia-smi` - all GPUs should show activity
2. **In logs:** "Multi-GPU training enabled" message
3. **Timing:** Multi-GPU should be faster than single GPU

## Single-GPU Fallback

If multi-GPU fails, training automatically falls back to single GPU:

```
Multi-GPU may not work. Falling back to single GPU...
```

To explicitly use single GPU:
```bash
export CUDA_VISIBLE_DEVICES=0
```
