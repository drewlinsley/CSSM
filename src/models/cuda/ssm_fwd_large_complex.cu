#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <ATen/cuda/CUDAContext.h>
#include "ssm_common.h"

/*
 * Complex parallel scan for the recurrence:
 *     h[t] = A[t] * h[t-1] + U[t]
 *
 * where A, U, h are all complex-valued (split into real/imag float arrays).
 *
 * This is the CSSM convention: U = B_gate * x is pre-combined before the scan,
 * so the kernel only takes 2 complex inputs (A, U) instead of 3 (A, B, x).
 *
 * Complex multiply:  (a + bi)(c + di) = (ac - bd) + (ad + bc)i
 * Complex mul-add:   A * u + v  = (A_re*u_re - A_im*u_im + v_re)
 *                                + (A_re*u_im + A_im*u_re + v_im)i
 *
 * All inputs/outputs are split into separate real and imaginary float arrays
 * to match the linear_split convention used in the CSSM codebase and to
 * ensure coalesced memory access.
 */

__global__ void __launch_bounds__(CHUNK_SIZE) parallel_scan_complex_kernel_fwd(
    const float *A_re, const float *A_im,   // (B_size, T, D) - per-timestep decay
    const float *U_re, const float *U_im,   // (B_size, T, D) - pre-gated input
    float *Aseq_re, float *Aseq_im,         // (B_size, M, D) - accumulated A products
    float *useq_re, float *useq_im,         // (B_size, M, D) - accumulated hidden states
    float *Aagg_re, float *Aagg_im,         // (B_size, num_chunks, D) - chunk aggregates for A
    float *uagg_re, float *uagg_im,         // (B_size, num_chunks, D) - chunk aggregates for u
    long B_size, long T, long D,
    long M, long num_chunks, int level)
{
    __shared__ float s_A_re[CHUNK_SIZE], s_A_im[CHUNK_SIZE];
    __shared__ float s_u_re[CHUNK_SIZE], s_u_im[CHUNK_SIZE];

    int dstep, li, ri, tmp_idx, chunk_last_local, agg_idx;
    int chunk = blockIdx.z, d = blockIdx.x, b = blockIdx.y,
        t_local = threadIdx.x, t_global = chunk * CHUNK_SIZE + t_local;

    // Temporaries for complex arithmetic
    float Ar_re, Ar_im, Al_re, Al_im;
    float ur_re, ur_im;
    float new_A_re, new_A_im, new_u_re, new_u_im;

    // =========================================================================
    // Load data into shared memory
    // =========================================================================
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
            // Identity for A (1 + 0i), zero for u
            s_A_re[t_local] = 1.0f;
            s_A_im[t_local] = 0.0f;
            s_u_re[t_local] = 0.0f;
            s_u_im[t_local] = 0.0f;
        }
    }
    else
    {
        // Deeper levels — data already in Aseq/useq from previous level
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

    // =========================================================================
    // Up-sweep (reduce)
    // =========================================================================
    for (dstep = 1; dstep < CHUNK_SIZE; dstep *= 2)
    {
        if (t_local % (2 * dstep) == 0)
        {
            li = t_local + dstep - 1;
            ri = t_local + (2 * dstep) - 1;

            if (ri < CHUNK_SIZE)
            {
                // A[ri] = A[li] * A[ri]  (complex multiply)
                Ar_re = s_A_re[ri];
                Ar_im = s_A_im[ri];
                Al_re = s_A_re[li];
                Al_im = s_A_im[li];

                s_A_re[ri] = Al_re * Ar_re - Al_im * Ar_im;
                s_A_im[ri] = Al_re * Ar_im + Al_im * Ar_re;

                // u[ri] = A[ri] * u[li] + u[ri]  (complex mul-add)
                // Note: using original A[ri] (before overwrite above)
                s_u_re[ri] = (Ar_re * s_u_re[li] - Ar_im * s_u_im[li]) + s_u_re[ri];
                s_u_im[ri] = (Ar_re * s_u_im[li] + Ar_im * s_u_re[li]) + s_u_im[ri];
            }
        }
        __syncthreads();
    }
    __syncthreads();

    // =========================================================================
    // Save chunk aggregates
    // =========================================================================
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

    // =========================================================================
    // Reset final position to identity
    // =========================================================================
    if (t_local == 0)
    {
        s_A_re[chunk_last_local] = 1.0f;  // complex identity = 1 + 0i
        s_A_im[chunk_last_local] = 0.0f;
        s_u_re[chunk_last_local] = 0.0f;
        s_u_im[chunk_last_local] = 0.0f;
    }
    __syncthreads();

    // =========================================================================
    // Down-sweep (distribute)
    // =========================================================================
    for (dstep = CHUNK_SIZE / 2; dstep >= 1; dstep /= 2)
    {
        if (t_local % (2 * dstep) == 0)
        {
            li = t_local + dstep - 1;
            ri = t_local + (2 * dstep) - 1;
            if (ri < CHUNK_SIZE)
            {
                // Save left values
                Al_re = s_A_re[li];
                Al_im = s_A_im[li];
                Ar_re = s_A_re[ri];
                Ar_im = s_A_im[ri];

                // Swap: A[li] = A[ri]
                s_A_re[li] = Ar_re;
                s_A_im[li] = Ar_im;

                // A[ri] = A[li] * A[ri]  (complex multiply, using saved Al)
                s_A_re[ri] = Al_re * Ar_re - Al_im * Ar_im;
                s_A_im[ri] = Al_re * Ar_im + Al_im * Ar_re;

                // Save u[ri] before overwrite
                ur_re = s_u_re[ri];
                ur_im = s_u_im[ri];

                // u[ri] = A[li] * u[ri] + u[li]  (complex mul-add)
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

    // =========================================================================
    // Store back to global memory
    // =========================================================================
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

    // Read this chunk's prefix from the scanned aggregates
    prefix_idx = b * num_chunks * D + chunk * D + d;
    pA_re = Aagg_re[prefix_idx];
    pA_im = Aagg_im[prefix_idx];
    pu_re = uagg_re[prefix_idx];
    pu_im = uagg_im[prefix_idx];

    // Read local values
    idx = b * M * D + t_global * D + d;
    lA_re = Aseq_re[idx];
    lA_im = Aseq_im[idx];
    lu_re = useq_re[idx];
    lu_im = useq_im[idx];

    // Apply: A[t] = local_A * prefix_A  (complex multiply)
    Aseq_re[idx] = lA_re * pA_re - lA_im * pA_im;
    Aseq_im[idx] = lA_re * pA_im + lA_im * pA_re;

    // Apply: u[t] = local_A * prefix_u + local_u  (complex mul-add)
    useq_re[idx] = (lA_re * pu_re - lA_im * pu_im) + lu_re;
    useq_im[idx] = (lA_re * pu_im + lA_im * pu_re) + lu_im;
}


extern "C"
{

    void parallel_scan_complex_fwd_large_cuda(
        const float *A_re, const float *A_im,
        const float *U_re, const float *U_im,
        float *Aseq_re, float *Aseq_im,
        float *useq_re, float *useq_im,
        float **Aagg_re, float **Aagg_im,
        float **uagg_re, float **uagg_im,
        long B_size, long T, long D, long M,
        long num_chunks, int num_levels)
    {
        if (T == 1)
            return;

        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        int level, num_chunks_local, n = M, level_sizes[32];
        float *Aseq_ptr_re, *Aseq_ptr_im, *useq_ptr_re, *useq_ptr_im;
        Aseq_ptr_re = Aseq_re;
        Aseq_ptr_im = Aseq_im;
        useq_ptr_re = useq_re;
        useq_ptr_im = useq_im;

        // Phase 1/2: local Blelloch scan per chunk + save aggregates
        for (level = 0; level < num_levels; level++)
        {
            num_chunks_local = (n + CHUNK_SIZE - 1) / CHUNK_SIZE;

            dim3 grid(D, B_size, num_chunks_local);
            dim3 block(CHUNK_SIZE);

            parallel_scan_complex_kernel_fwd<<<grid, block, 0, stream>>>(
                A_re, A_im, U_re, U_im,
                Aseq_ptr_re, Aseq_ptr_im,
                useq_ptr_re, useq_ptr_im,
                Aagg_re[level], Aagg_im[level],
                uagg_re[level], uagg_im[level],
                B_size, T, D, n, num_chunks_local, level);

            // Next level operates on the aggregates
            Aseq_ptr_re = Aagg_re[level];
            Aseq_ptr_im = Aagg_im[level];
            useq_ptr_re = uagg_re[level];
            useq_ptr_im = uagg_im[level];
            level_sizes[level] = n;
            n = num_chunks_local;
        }

        // Phase 3: propagate prefixes back down
        for (level = num_levels - 1; level >= 0; level--)
        {
            Aseq_ptr_re = (level == 0) ? Aseq_re : Aagg_re[level - 1];
            Aseq_ptr_im = (level == 0) ? Aseq_im : Aagg_im[level - 1];
            useq_ptr_re = (level == 0) ? useq_re : uagg_re[level - 1];
            useq_ptr_im = (level == 0) ? useq_im : uagg_im[level - 1];
            num_chunks_local = (level_sizes[level] + CHUNK_SIZE - 1) / CHUNK_SIZE;

            dim3 grid(D, B_size, num_chunks_local);
            dim3 block(CHUNK_SIZE);

            apply_prefixes_complex<<<grid, block, 0, stream>>>(
                Aseq_ptr_re, Aseq_ptr_im,
                useq_ptr_re, useq_ptr_im,
                Aagg_re[level], Aagg_im[level],
                uagg_re[level], uagg_im[level],
                B_size, D, level_sizes[level], num_chunks_local);
        }
    }
}
