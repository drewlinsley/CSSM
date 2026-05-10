/*
 * Complex parallel scan for the recurrence:
 *     h[t] = A[t] * h[t-1] + U[t]
 *
 * where A, U, h are all complex-valued (split into real/imag float arrays).
 *
 * CSSM convention: U = B_gate * x is pre-combined before the scan,
 * so the kernel takes 2 complex inputs (A, U) instead of 3 (A, B, x).
 *
 * JAX FFI entry point: receives (stream, buffers, opaque, opaque_len)
 * from XLA custom call. Aggregate workspace is allocated internally.
 *
 * Build:
 *   nvcc -shared -o libssm_scan.so ssm_fwd_large_complex.cu \
 *        -I$CUDA_HOME/include --compiler-options '-fPIC'
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef CHUNK_SIZE
#define CHUNK_SIZE 1024
#endif

// ============================================================================
// Descriptor struct — must match the kwargs packed by jax_ffi.ffi_call
// ============================================================================
struct ScanDescriptor {
    long B_size;
    long T;
    long D;
    long M;
    long num_chunks;
    int  num_levels;
};

// ============================================================================
// Kernel: Blelloch scan (per-chunk local scan + aggregate extraction)
// ============================================================================
__global__ void __launch_bounds__(CHUNK_SIZE) parallel_scan_complex_kernel_fwd(
    const float *A_re, const float *A_im,
    const float *U_re, const float *U_im,
    float *Aseq_re, float *Aseq_im,
    float *useq_re, float *useq_im,
    float *Aagg_re, float *Aagg_im,
    float *uagg_re, float *uagg_im,
    long B_size, long T, long D,
    long M, long num_chunks, int level)
{
    __shared__ float s_A_re[CHUNK_SIZE], s_A_im[CHUNK_SIZE];
    __shared__ float s_u_re[CHUNK_SIZE], s_u_im[CHUNK_SIZE];

    int dstep, li, ri, tmp_idx, chunk_last_local, agg_idx;
    int chunk = blockIdx.z, d = blockIdx.x, b = blockIdx.y,
        t_local = threadIdx.x, t_global = chunk * CHUNK_SIZE + t_local;

    float Ar_re, Ar_im, Al_re, Al_im;
    float ur_re, ur_im;

    // ---- Load into shared memory ----
    if (level == 0)
    {
        if (t_global < T)
        {
            tmp_idx = b * T * D + t_global * D + d;
            s_A_re[t_local] = A_re[tmp_idx];
            s_A_im[t_local] = A_im[tmp_idx];
            s_u_re[t_local] = U_re[tmp_idx];
            s_u_im[t_local] = U_im[tmp_idx];
        }
        else if (t_global < M)
        {
            s_A_re[t_local] = 1.0f;
            s_A_im[t_local] = 0.0f;
            s_u_re[t_local] = 0.0f;
            s_u_im[t_local] = 0.0f;
        }
    }
    else
    {
        if (t_global < M)
        {
            tmp_idx = b * M * D + t_global * D + d;
            s_A_re[t_local] = Aseq_re[tmp_idx];
            s_A_im[t_local] = Aseq_im[tmp_idx];
            s_u_re[t_local] = useq_re[tmp_idx];
            s_u_im[t_local] = useq_im[tmp_idx];
        }
    }
    __syncthreads();

    // ---- Up-sweep (reduce) ----
    for (dstep = 1; dstep < CHUNK_SIZE; dstep *= 2)
    {
        if (t_local % (2 * dstep) == 0)
        {
            li = t_local + dstep - 1;
            ri = t_local + (2 * dstep) - 1;

            if (ri < CHUNK_SIZE)
            {
                Ar_re = s_A_re[ri];
                Ar_im = s_A_im[ri];
                Al_re = s_A_re[li];
                Al_im = s_A_im[li];

                // A[ri] = A[li] * A[ri]
                s_A_re[ri] = Al_re * Ar_re - Al_im * Ar_im;
                s_A_im[ri] = Al_re * Ar_im + Al_im * Ar_re;

                // u[ri] = A_orig[ri] * u[li] + u[ri]
                s_u_re[ri] = (Ar_re * s_u_re[li] - Ar_im * s_u_im[li]) + s_u_re[ri];
                s_u_im[ri] = (Ar_re * s_u_im[li] + Ar_im * s_u_re[li]) + s_u_im[ri];
            }
        }
        __syncthreads();
    }
    __syncthreads();

    // ---- Save chunk aggregates ----
    chunk_last_local = CHUNK_SIZE - 1;
    if (t_local == 0)
    {
        agg_idx = b * num_chunks * D + chunk * D + d;
        Aagg_re[agg_idx] = s_A_re[chunk_last_local];
        Aagg_im[agg_idx] = s_A_im[chunk_last_local];
        uagg_re[agg_idx] = s_u_re[chunk_last_local];
        uagg_im[agg_idx] = s_u_im[chunk_last_local];
    }
    __syncthreads();

    // ---- Reset last position to identity ----
    if (t_local == 0)
    {
        s_A_re[chunk_last_local] = 1.0f;
        s_A_im[chunk_last_local] = 0.0f;
        s_u_re[chunk_last_local] = 0.0f;
        s_u_im[chunk_last_local] = 0.0f;
    }
    __syncthreads();

    // ---- Down-sweep (distribute) ----
    for (dstep = CHUNK_SIZE / 2; dstep >= 1; dstep /= 2)
    {
        if (t_local % (2 * dstep) == 0)
        {
            li = t_local + dstep - 1;
            ri = t_local + (2 * dstep) - 1;
            if (ri < CHUNK_SIZE)
            {
                Al_re = s_A_re[li];
                Al_im = s_A_im[li];
                Ar_re = s_A_re[ri];
                Ar_im = s_A_im[ri];

                // Swap: A[li] = A[ri]
                s_A_re[li] = Ar_re;
                s_A_im[li] = Ar_im;

                // A[ri] = Al * Ar
                s_A_re[ri] = Al_re * Ar_re - Al_im * Ar_im;
                s_A_im[ri] = Al_re * Ar_im + Al_im * Ar_re;

                // Save u[ri]
                ur_re = s_u_re[ri];
                ur_im = s_u_im[ri];

                // u[ri] = Al * u[ri] + u[li]
                s_u_re[ri] = (Al_re * ur_re - Al_im * ur_im) + s_u_re[li];
                s_u_im[ri] = (Al_re * ur_im + Al_im * ur_re) + s_u_im[li];

                // u[li] = saved u[ri]
                s_u_re[li] = ur_re;
                s_u_im[li] = ur_im;
            }
        }
        __syncthreads();
    }
    __syncthreads();

    // ---- Store back to global memory ----
    if (t_global < M)
    {
        tmp_idx = b * M * D + t_global * D + d;
        Aseq_re[tmp_idx] = s_A_re[t_local];
        Aseq_im[tmp_idx] = s_A_im[t_local];
        useq_re[tmp_idx] = s_u_re[t_local];
        useq_im[tmp_idx] = s_u_im[t_local];
    }
    __syncthreads();
}


// ============================================================================
// Kernel: prefix propagation across chunks
// ============================================================================
__global__ void __launch_bounds__(CHUNK_SIZE) apply_prefixes_complex(
    float *Aseq_re, float *Aseq_im,
    float *useq_re, float *useq_im,
    float *Aagg_re, float *Aagg_im,
    float *uagg_re, float *uagg_im,
    long B_size, long D, long M, long num_chunks)
{
    int chunk = blockIdx.z, d = blockIdx.x, b = blockIdx.y,
        t_local = threadIdx.x,
        t_global = chunk * CHUNK_SIZE + t_local,
        prefix_idx, idx;
    float pA_re, pA_im, pu_re, pu_im;
    float lA_re, lA_im, lu_re, lu_im;

    if (chunk == 0 || t_global >= M)
        return;

    prefix_idx = b * num_chunks * D + chunk * D + d;
    pA_re = Aagg_re[prefix_idx];
    pA_im = Aagg_im[prefix_idx];
    pu_re = uagg_re[prefix_idx];
    pu_im = uagg_im[prefix_idx];

    idx = b * M * D + t_global * D + d;
    lA_re = Aseq_re[idx];
    lA_im = Aseq_im[idx];
    lu_re = useq_re[idx];
    lu_im = useq_im[idx];

    // A[t] = local_A * prefix_A
    Aseq_re[idx] = lA_re * pA_re - lA_im * pA_im;
    Aseq_im[idx] = lA_re * pA_im + lA_im * pA_re;

    // u[t] = local_A * prefix_u + local_u
    useq_re[idx] = (lA_re * pu_re - lA_im * pu_im) + lu_re;
    useq_im[idx] = (lA_re * pu_im + lA_im * pu_re) + lu_im;
}


// ============================================================================
// Kernel: convert exclusive scan to inclusive and write to output
// ============================================================================
// Replaces both the cudaMemcpyAsync (useq → h) AND the Python-side fixup.
// Computes: h_inclusive[t] = A[t] * h_exclusive[t] + U[t]
// Reads from useq (B, M, D), writes to h (B, T, D).
//
__global__ void finalize_inclusive_complex(
    const float *A_re, const float *A_im,       // original inputs (B, T, D)
    const float *U_re, const float *U_im,       // original inputs (B, T, D)
    const float *excl_re, const float *excl_im,  // exclusive scan in useq (B, M, D)
    float *h_re, float *h_im,                    // inclusive output (B, T, D)
    long B_size, long T, long D, long M)
{
    int d = blockIdx.x;
    int b = blockIdx.y;
    int t = blockIdx.z * blockDim.x + threadIdx.x;

    if (d >= D || b >= B_size || t >= T)
        return;

    long in_idx  = b * T * D + t * D + d;  // index into (B, T, D)
    long exc_idx = b * M * D + t * D + d;  // index into (B, M, D)

    // h[t] = A[t] * excl[t] + U[t]  (complex multiply-add)
    float a_re = A_re[in_idx],    a_im = A_im[in_idx];
    float e_re = excl_re[exc_idx], e_im = excl_im[exc_idx];
    float u_re = U_re[in_idx],    u_im = U_im[in_idx];

    h_re[in_idx] = (a_re * e_re - a_im * e_im) + u_re;
    h_im[in_idx] = (a_re * e_im + a_im * e_re) + u_im;
}
//
// TODO(library): Replace this static cache with JAX-managed workspace buffers
// (declare as extra ffi_call outputs) before releasing as a library. The static
// approach works for research (single model size per process) but:
//   - Never freed (minor leak, cleaned up at process exit)
//   - Not thread-safe
//   - Breaks if two different model sizes run in the same process
// See: https://jax.readthedocs.io/en/latest/ffi.html for the proper approach.
//
// ============================================================================

#define MAX_LEVELS 32

static struct {
    float *Aseq_re, *Aseq_im, *useq_re, *useq_im;
    float *Aagg_re[MAX_LEVELS], *Aagg_im[MAX_LEVELS];
    float *uagg_re[MAX_LEVELS], *uagg_im[MAX_LEVELS];
    float **Aagg_re_h, **Aagg_im_h, **uagg_re_h, **uagg_im_h;
    long cached_B, cached_M, cached_D;
    int  cached_num_levels;
    bool initialized;
} g_workspace = { .initialized = false };

static void workspace_ensure(long B_size, long M, long D, int num_levels)
{
    if (g_workspace.initialized &&
        g_workspace.cached_B == B_size &&
        g_workspace.cached_M == M &&
        g_workspace.cached_D == D &&
        g_workspace.cached_num_levels == num_levels)
    {
        return;  // Already allocated with matching dimensions
    }

    // Free old buffers if dimensions changed
    if (g_workspace.initialized)
    {
        cudaFree(g_workspace.Aseq_re);
        cudaFree(g_workspace.Aseq_im);
        cudaFree(g_workspace.useq_re);
        cudaFree(g_workspace.useq_im);
        for (int lev = 0; lev < g_workspace.cached_num_levels; lev++)
        {
            cudaFree(g_workspace.Aagg_re[lev]);
            cudaFree(g_workspace.Aagg_im[lev]);
            cudaFree(g_workspace.uagg_re[lev]);
            cudaFree(g_workspace.uagg_im[lev]);
        }
    }

    // Allocate seq buffers: (B, M, D)
    size_t seq_bytes = B_size * M * D * sizeof(float);
    cudaMalloc(&g_workspace.Aseq_re, seq_bytes);
    cudaMalloc(&g_workspace.Aseq_im, seq_bytes);
    cudaMalloc(&g_workspace.useq_re, seq_bytes);
    cudaMalloc(&g_workspace.useq_im, seq_bytes);

    // Allocate aggregate buffers per level
    int n = M;
    for (int lev = 0; lev < num_levels; lev++)
    {
        int nc = (n + CHUNK_SIZE - 1) / CHUNK_SIZE;
        size_t agg_bytes = B_size * nc * D * sizeof(float);
        cudaMalloc(&g_workspace.Aagg_re[lev], agg_bytes);
        cudaMalloc(&g_workspace.Aagg_im[lev], agg_bytes);
        cudaMalloc(&g_workspace.uagg_re[lev], agg_bytes);
        cudaMalloc(&g_workspace.uagg_im[lev], agg_bytes);
        n = nc;
    }

    g_workspace.cached_B = B_size;
    g_workspace.cached_M = M;
    g_workspace.cached_D = D;
    g_workspace.cached_num_levels = num_levels;
    g_workspace.initialized = true;
}


// ============================================================================
// Host launcher (framework-agnostic, takes stream explicitly)
// ============================================================================
static void launch_complex_scan(
    cudaStream_t stream,
    const float *A_re, const float *A_im,
    const float *U_re, const float *U_im,
    float *h_re, float *h_im,
    long B_size, long T, long D, long M,
    long num_chunks, int num_levels)
{
    if (T == 1)
    {
        cudaMemcpyAsync(h_re, U_re, B_size * T * D * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(h_im, U_im, B_size * T * D * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
        return;
    }

    // Ensure workspace is allocated (no-op after first call with same dims)
    workspace_ensure(B_size, M, D, num_levels);

    float *Aseq_re = g_workspace.Aseq_re;
    float *Aseq_im = g_workspace.Aseq_im;
    float *useq_re = g_workspace.useq_re;
    float *useq_im = g_workspace.useq_im;

    // --- Phase 1/2: local Blelloch scan per chunk + save aggregates ---
    float *Aseq_ptr_re = Aseq_re, *Aseq_ptr_im = Aseq_im;
    float *useq_ptr_re = useq_re, *useq_ptr_im = useq_im;
    int level_sizes[MAX_LEVELS];
    int n = M;

    for (int level = 0; level < num_levels; level++)
    {
        int nc = (n + CHUNK_SIZE - 1) / CHUNK_SIZE;
        dim3 grid(D, B_size, nc);
        dim3 block(CHUNK_SIZE);

        parallel_scan_complex_kernel_fwd<<<grid, block, 0, stream>>>(
            A_re, A_im, U_re, U_im,
            Aseq_ptr_re, Aseq_ptr_im,
            useq_ptr_re, useq_ptr_im,
            g_workspace.Aagg_re[level], g_workspace.Aagg_im[level],
            g_workspace.uagg_re[level], g_workspace.uagg_im[level],
            B_size, T, D, n, nc, level);

        Aseq_ptr_re = g_workspace.Aagg_re[level];
        Aseq_ptr_im = g_workspace.Aagg_im[level];
        useq_ptr_re = g_workspace.uagg_re[level];
        useq_ptr_im = g_workspace.uagg_im[level];
        level_sizes[level] = n;
        n = nc;
    }

    // --- Phase 3: propagate prefixes back down ---
    for (int level = num_levels - 1; level >= 0; level--)
    {
        Aseq_ptr_re = (level == 0) ? Aseq_re : g_workspace.Aagg_re[level - 1];
        Aseq_ptr_im = (level == 0) ? Aseq_im : g_workspace.Aagg_im[level - 1];
        useq_ptr_re = (level == 0) ? useq_re : g_workspace.uagg_re[level - 1];
        useq_ptr_im = (level == 0) ? useq_im : g_workspace.uagg_im[level - 1];
        int nc = (level_sizes[level] + CHUNK_SIZE - 1) / CHUNK_SIZE;

        dim3 grid(D, B_size, nc);
        dim3 block(CHUNK_SIZE);

        apply_prefixes_complex<<<grid, block, 0, stream>>>(
            Aseq_ptr_re, Aseq_ptr_im,
            useq_ptr_re, useq_ptr_im,
            g_workspace.Aagg_re[level], g_workspace.Aagg_im[level],
            g_workspace.uagg_re[level], g_workspace.uagg_im[level],
            B_size, D, level_sizes[level], nc);
    }

    // --- Phase 4: convert exclusive→inclusive and write to output ---
    {
        int t_blocks = (T + 255) / 256;
        dim3 grid(D, B_size, t_blocks);
        dim3 block(256);

        finalize_inclusive_complex<<<grid, block, 0, stream>>>(
            A_re, A_im, U_re, U_im,
            useq_re, useq_im,
            h_re, h_im,
            B_size, T, D, M);
    }
}


// ============================================================================
// JAX FFI entry point
// ============================================================================
extern "C" {

void cuda_scan_complex_fwd(
    cudaStream_t stream,
    void **buffers,
    const char *opaque,
    size_t opaque_len)
{
    const ScanDescriptor *desc = (const ScanDescriptor *)opaque;

    // Inputs (4 buffers)
    const float *A_re = (const float *)buffers[0];
    const float *A_im = (const float *)buffers[1];
    const float *U_re = (const float *)buffers[2];
    const float *U_im = (const float *)buffers[3];

    // Outputs (2 buffers, pre-allocated by JAX)
    float *h_re = (float *)buffers[4];
    float *h_im = (float *)buffers[5];

    launch_complex_scan(
        stream,
        A_re, A_im, U_re, U_im,
        h_re, h_im,
        desc->B_size, desc->T, desc->D,
        desc->M, desc->num_chunks, desc->num_levels);
}

}  // extern "C"
