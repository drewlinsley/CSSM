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
struct ScanDescriptor
{
    long B_size;
    long T;
    long D;
    long M;
    long num_chunks;
    int num_levels;
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
        // Just copy input U to output h
        cudaMemcpyAsync(h_re, U_re, B_size * T * D * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(h_im, U_im, B_size * T * D * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
        return;
    }

    // --- Allocate workspace ---
    // Aseq/useq: (B, M, D) — 4 buffers (re/im for A and u)
    size_t seq_bytes = B_size * M * D * sizeof(float);
    float *Aseq_re, *Aseq_im, *useq_re, *useq_im;
    cudaMalloc(&Aseq_re, seq_bytes);
    cudaMalloc(&Aseq_im, seq_bytes);
    cudaMalloc(&useq_re, seq_bytes);
    cudaMalloc(&useq_im, seq_bytes);

    // Aggregate buffers at each level: 4 arrays per level
    float **Aagg_re_h = (float **)malloc(num_levels * sizeof(float *));
    float **Aagg_im_h = (float **)malloc(num_levels * sizeof(float *));
    float **uagg_re_h = (float **)malloc(num_levels * sizeof(float *));
    float **uagg_im_h = (float **)malloc(num_levels * sizeof(float *));

    int n = M;
    for (int lev = 0; lev < num_levels; lev++)
    {
        int nc = (n + CHUNK_SIZE - 1) / CHUNK_SIZE;
        size_t agg_bytes = B_size * nc * D * sizeof(float);
        cudaMalloc(&Aagg_re_h[lev], agg_bytes);
        cudaMalloc(&Aagg_im_h[lev], agg_bytes);
        cudaMalloc(&uagg_re_h[lev], agg_bytes);
        cudaMalloc(&uagg_im_h[lev], agg_bytes);
        n = nc;
    }

    // --- Phase 1/2: local Blelloch scan per chunk + save aggregates ---
    float *Aseq_ptr_re = Aseq_re, *Aseq_ptr_im = Aseq_im;
    float *useq_ptr_re = useq_re, *useq_ptr_im = useq_im;
    int level_sizes[32];
    n = M;

    for (int level = 0; level < num_levels; level++)
    {
        int nc = (n + CHUNK_SIZE - 1) / CHUNK_SIZE;
        dim3 grid(D, B_size, nc);
        dim3 block(CHUNK_SIZE);

        parallel_scan_complex_kernel_fwd<<<grid, block, 0, stream>>>(
            A_re, A_im, U_re, U_im,
            Aseq_ptr_re, Aseq_ptr_im,
            useq_ptr_re, useq_ptr_im,
            Aagg_re_h[level], Aagg_im_h[level],
            uagg_re_h[level], uagg_im_h[level],
            B_size, T, D, n, nc, level);

        Aseq_ptr_re = Aagg_re_h[level];
        Aseq_ptr_im = Aagg_im_h[level];
        useq_ptr_re = uagg_re_h[level];
        useq_ptr_im = uagg_im_h[level];
        level_sizes[level] = n;
        n = nc;
    }

    // --- Phase 3: propagate prefixes back down ---
    for (int level = num_levels - 1; level >= 0; level--)
    {
        Aseq_ptr_re = (level == 0) ? Aseq_re : Aagg_re_h[level - 1];
        Aseq_ptr_im = (level == 0) ? Aseq_im : Aagg_im_h[level - 1];
        useq_ptr_re = (level == 0) ? useq_re : uagg_re_h[level - 1];
        useq_ptr_im = (level == 0) ? useq_im : uagg_im_h[level - 1];
        int nc = (level_sizes[level] + CHUNK_SIZE - 1) / CHUNK_SIZE;

        dim3 grid(D, B_size, nc);
        dim3 block(CHUNK_SIZE);

        apply_prefixes_complex<<<grid, block, 0, stream>>>(
            Aseq_ptr_re, Aseq_ptr_im,
            useq_ptr_re, useq_ptr_im,
            Aagg_re_h[level], Aagg_im_h[level],
            uagg_re_h[level], uagg_im_h[level],
            B_size, D, level_sizes[level], nc);
    }

    // --- Copy results: useq[:, :T, :] -> h (output) ---
    if (M == T)
    {
        cudaMemcpyAsync(h_re, useq_re, B_size * T * D * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(h_im, useq_im, B_size * T * D * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
    }
    else
    {
        // M > T: copy only the first T timesteps per batch
        for (long b = 0; b < B_size; b++)
        {
            cudaMemcpyAsync(
                h_re + b * T * D,
                useq_re + b * M * D,
                T * D * sizeof(float),
                cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(
                h_im + b * T * D,
                useq_im + b * M * D,
                T * D * sizeof(float),
                cudaMemcpyDeviceToDevice, stream);
        }
    }

    // --- Free workspace ---
    for (int lev = 0; lev < num_levels; lev++)
    {
        cudaFree(Aagg_re_h[lev]);
        cudaFree(Aagg_im_h[lev]);
        cudaFree(uagg_re_h[lev]);
        cudaFree(uagg_im_h[lev]);
    }
    free(Aagg_re_h);
    free(Aagg_im_h);
    free(uagg_re_h);
    free(uagg_im_h);

    cudaFree(Aseq_re);
    cudaFree(Aseq_im);
    cudaFree(useq_re);
    cudaFree(useq_im);
}

// ============================================================================
// JAX FFI entry point
// ============================================================================
extern "C"
{

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

} // extern "C"
