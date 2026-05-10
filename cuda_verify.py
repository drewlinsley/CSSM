import jax
import jax.numpy as jnp

from src.models.cuda_scan import cuda_complex_scan
from src.models.math import cssm_scalar_scan_op
from src.models.goom import to_goom, from_goom

# ssm = load(name='ssm', sources=['ssm_binding.cpp', 'ssm.cpp', 'ssm.cu', 'ssm_fwd_back.cpp', 'ssm_fwd.cu', 'ssm_bwd.cu'], verbose=True)

B_size, T, D = 2, 4, 3
M = 4

# Compare CUDA scan vs JAX native scan on same input
key = jax.random.PRNGKey(758493)  # Random seed is explicit in JAX

A_c = jax.random.uniform(key, shape=(B_size, M+1, D))  # small test input
U_c = jax.random.uniform(key, shape=(B_size, M+1, D))

# CUDA
h_cuda = cuda_complex_scan(A_c, U_c)

# JAX
# _, h_jax = jax.lax.associative_scan(_scan_op, (A_c, U_c), axis=1)
A_log = to_goom(A_c)
U_log = to_goom(U_c)
_, X_log = jax.lax.associative_scan(
    cssm_scalar_scan_op, (A_log, U_log), axis=1
)
h_jax = from_goom(X_log)

print(jnp.max(jnp.abs(h_cuda - h_jax)))