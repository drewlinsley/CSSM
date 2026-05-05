#!/usr/bin/env python
"""
Multi-GPU diagnostic script for JAX/NCCL.
Run with: NCCL_DEBUG=WARN python diagnose_multigpu.py
"""

import os
import subprocess
import sys

def run_cmd(cmd):
    """Run shell command and return output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        return result.stdout.strip()
    except Exception as e:
        return f"Error: {e}"

print("=" * 60)
print("MULTI-GPU DIAGNOSTIC")
print("=" * 60)

# 1. GPU Hardware
print("\n[1] GPU Hardware (nvidia-smi)")
print("-" * 40)
print(run_cmd("nvidia-smi -L"))

# 2. CUDA/Driver versions
print("\n[2] CUDA/Driver Versions")
print("-" * 40)
print(run_cmd("nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1"))
print(run_cmd("nvcc --version 2>/dev/null | grep release || echo 'nvcc not in PATH'"))

# 3. NCCL version
print("\n[3] NCCL Info")
print("-" * 40)
nccl_info = run_cmd("python -c \"import jax; from jax._src.lib import xla_extension; print('JAX version:', jax.__version__)\" 2>/dev/null")
print(nccl_info)
print("NCCL_DEBUG:", os.environ.get("NCCL_DEBUG", "not set"))

# 4. JAX devices
print("\n[4] JAX Devices")
print("-" * 40)
import jax
devices = jax.devices()
print(f"Number of devices: {len(devices)}")
for i, d in enumerate(devices):
    print(f"  {i}: {d}")

# 5. Environment variables that affect NCCL
print("\n[5] Relevant Environment Variables")
print("-" * 40)
env_vars = [
    "CUDA_VISIBLE_DEVICES",
    "NCCL_DEBUG",
    "NCCL_P2P_DISABLE",
    "NCCL_SHM_DISABLE",
    "NCCL_IB_DISABLE",
    "NCCL_SOCKET_IFNAME",
    "XLA_FLAGS",
]
for var in env_vars:
    val = os.environ.get(var, "not set")
    print(f"  {var}: {val}")

# 6. Simple pmap test
print("\n[6] NCCL Communication Test")
print("-" * 40)
import jax.numpy as jnp

if len(devices) < 2:
    print("SKIP: Only 1 device, multi-GPU not applicable")
else:
    print(f"Testing pmean across {len(devices)} devices...")

    try:
        # Create test data
        data = jnp.ones((len(devices), 4), dtype=jnp.float32)

        # Define pmap function
        @jax.pmap
        def test_pmean(x):
            return jax.lax.pmean(x, axis_name='batch')

        # Monkey-patch axis_name
        test_pmean = jax.pmap(lambda x: jax.lax.pmean(x, 'batch'), axis_name='batch')

        # Run and block
        result = test_pmean(data)
        jax.block_until_ready(result)

        print(f"SUCCESS! pmean result[0] = {result[0]}")
        print("Multi-GPU should work.")

    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        print("\nTroubleshooting suggestions:")
        print("  1. Try: export NCCL_P2P_DISABLE=1")
        print("  2. Try: export NCCL_SHM_DISABLE=1")
        print("  3. Try: export NCCL_IB_DISABLE=1")
        print("  4. Check CUDA/NCCL version compatibility")
        print("  5. Run with NCCL_DEBUG=INFO for more details")

# 7. GPU topology
print("\n[7] GPU Topology")
print("-" * 40)
topo = run_cmd("nvidia-smi topo -m 2>/dev/null || echo 'Topology info not available'")
print(topo[:2000] if len(topo) > 2000 else topo)

print("\n" + "=" * 60)
print("END DIAGNOSTIC")
print("=" * 60)
