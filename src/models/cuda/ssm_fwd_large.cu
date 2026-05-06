#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "ssm_common.h"

struct ScanDescriptor
{
    long B_size;
    long T;
    long D;
    long M;
    long num_chunks;
    int num_levels;
};

__global__ void __launch_bounds__(CHUNK_SIZE) parallel_scan_SSM_diag_kernel_fwd(
    cudaStream_t stream,
    void **buffers,     // array of input/output buffer pointers
    const char *opaque, // packed static attributes
    size_t opaque_len)
{
    // A          : (d) - diagonal matrix represented as a vector
    // B          : (d) - diagonal matrix represented as a vector
    // x          : (B_size * T * D) - input, flattened to vector
    // A_seq      : (B_size, m, d) - accumulated A values at each time step
    // u_seq      : (B_size, m, d) - accumulated hidden states
    // A_last     : (D) - temporary hold due to Blelloch overwrite
    // u_last     : (B_size, D) - temporary hold due to Blelloch overwrite
    // A_agg      : (B_size, num_chunks, D) - holding aggregated values per chunk for A_seq
    // u_agg      : (B_size, num_chunks, D) - holding aggregated values per chunk for u_seq
    // B_size     : batch size
    // T          : time dimension length
    // D          : diagonal size
    // M          : smallest ^2 size greater than T
    // num_chunks : number of chunks

    __shared__ float s_A[CHUNK_SIZE];
    __shared__ float s_u[CHUNK_SIZE];

    int dstep, li, ri, tmp_idx, chunk_last_local, agg_idx;
    int chunk = blockIdx.z, d = blockIdx.x, b = blockIdx.y,
        t_local = threadIdx.x, t_global = chunk * CHUNK_SIZE + t_local;
    float A_r, A_l, u_r, shifted_val;

    // filling in values (if first level), loading global memory
    if (level == 0)
    {
        if (t_global < T)
        {
            s_A[t_local] = A[d];
            s_u[t_local] = x[b * T * D + t_global * D + d] * B[d];
        }
        else if (t_global < M)
        {
            s_A[t_local] = 1.0f;
            s_u[t_local] = 0.0f;
        }
    }
    else
    {
        // Deeper levels — data already in A_seq/u_seq from previous level
        if (t_global < M)
        {
            tmp_idx = b * M * D + t_global * D + d;
            s_A[t_local] = A_seq[tmp_idx];
            s_u[t_local] = u_seq[tmp_idx];
        }
    }
    __syncthreads();

    // up-sweep
    for (dstep = 1; dstep < CHUNK_SIZE; dstep *= 2)
    {
        // need to use t_local here, not t_global, to make sure that the
        // scan is always relative to the beginning of the block
        if (t_local % (2 * dstep) == 0)
        {
            li = t_local + dstep - 1;
            ri = t_local + (2 * dstep) - 1;

            if (ri < CHUNK_SIZE)
            {
                A_r = s_A[ri];
                s_A[ri] = s_A[li] * A_r;
                s_u[ri] = A_r * s_u[li] + s_u[ri];
            }
        }
        __syncthreads();
    }
    __syncthreads();

    // save of final time-step for agg values
    chunk_last_local = CHUNK_SIZE - 1;
    if (t_local == 0)
    {
        agg_idx = b * num_chunks * D + chunk * D + d;
        A_agg[agg_idx] = s_A[chunk_last_local];
        u_agg[agg_idx] = s_u[chunk_last_local];

        if ((level == 0) && (chunk == num_chunks - 1))
        {
            tmp_idx = b * D + d;
            u_last[tmp_idx] = s_u[chunk_last_local];

            if (b == 0)
            {
                A_last[d] = s_A[chunk_last_local];
            }
        }
    }
    __syncthreads();

    // reset final time-steps
    if (t_local == 0)
    {
        s_A[chunk_last_local] = 1.0f;
        s_u[chunk_last_local] = 0.0f;
    }
    __syncthreads();

    // down-sweep
    for (dstep = CHUNK_SIZE / 2; dstep >= 1; dstep /= 2)
    {
        if (t_local % (2 * dstep) == 0)
        {
            li = t_local + dstep - 1;
            ri = t_local + (2 * dstep) - 1;
            if (ri < CHUNK_SIZE)
            {
                A_l = s_A[li];
                A_r = s_A[ri];
                s_A[li] = A_r;
                s_A[ri] = A_l * A_r;

                u_r = s_u[ri];
                s_u[ri] = A_l * s_u[ri] + s_u[li];
                s_u[li] = u_r;
            }
        }
        __syncthreads();
    }
    __syncthreads();

    // store shared memory in gloval memory
    if (t_global < M)
    {
        tmp_idx = b * M * D + t_global * D + d;
        A_seq[tmp_idx] = s_A[t_local];
        u_seq[tmp_idx] = s_u[t_local];
    }
    __syncthreads();

    // // shift every timestep one level down
    // if (t < M - 1)
    // {
    //     shifted_val = u_seq[b * M * D + (t + 1) * D + d];
    // }
    // __syncthreads();
    // if (t < M - 1)
    // {
    //     u_seq[b * M * D + t * D + d] = shifted_val;
    // }
    // __syncthreads();

    // // copy from u_last
    // u_idx = b * M * D + (M - 1) * D + d;
    // tmp_idx = b * D + d;
    // u_seq[u_idx] = u_last[tmp_idx];
}

__global__ void __launch_bounds__(CHUNK_SIZE) apply_prefixes(
    float *A_seq, float *u_seq,
    float *A_agg, float *u_agg,
    long B_size, long D, long M, long num_chunks)
{
    int chunk = blockIdx.z, d = blockIdx.x, b = blockIdx.y,
        t_local = threadIdx.x,
        t_global = chunk * CHUNK_SIZE + t_local,
        prefix_idx, idx;
    float prefix_A, prefix_u, local_A, local_u;

    if (chunk == 0 || t_global >= M)
        return; // chunk 0 has no prefix, nothing to add

    // read this chunk's prefix from the scanned aggregates
    prefix_idx = b * num_chunks * D + chunk * D + d;
    prefix_A = A_agg[prefix_idx];
    prefix_u = u_agg[prefix_idx];

    // apply: A[t] = prefix_A * A[t]
    //        u[t] = prefix_A * prefix_u + u[t]
    idx = b * M * D + t_global * D + d;
    local_A = A_seq[idx];
    local_u = u_seq[idx];
    A_seq[idx] = local_A * prefix_A;
    u_seq[idx] = local_A * prefix_u + local_u;
    // A_seq[idx] = prefix_A * local_A;
    // u_seq[idx] = prefix_A * local_u + prefix_u;
}

extern "C"
{

    void parallel_scan_SSM_diag_fwd_large_cuda(
        float *A, float *B, float *x,
        float *A_seq, float *u_seq,
        float *A_last, float *u_last,
        float **A_agg, float **u_agg,
        long B_size, long T, long D, long M,
        long num_chunks, int num_levels)
    {
        // A          : (d) - diagonal matrix represented as a vector
        // B          : (d) - diagonal matrix represented as a vector
        // x          : (B_size * T * D) - input, flattened to vector
        // A_seq      : (m, d) - accumulated A values at each time step
        // u_seq      : (B_size, m, d) - accumulated hidden states
        // A_last     : (D) - temporary hold due to Blelloch overwrite
        // u_last     : (B_size, D) - temporary hold due to Blelloch overwrite
        // A_agg      : (B_size, num_chunks, D) - holding aggregated values per chunk for A_seq at each level
        // u_agg      : (B_size, num_chunks, D) - holding aggregated values per chunk for u_seq at each level
        // B_size     : batch size
        // T          : time dimension length
        // D          : diagonal size
        // M          : smallest ^2 size greater than T
        // num_chunks : number of chunks
        // num_levels : number of levels

        if (T == 1)
            return;

        // int num_chunks = (M + CHUNK_SIZE - 1) / CHUNK_SIZE;

        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        // dim3 grid(B_size, D, num_chunks);
        // dim3 block(CHUNK_SIZE); // num_threads // up to 1024, depends on T // setting to 1024 for now
        int level, num_chunks_local, n = M, level_sizes[32];
        float *A_agg_ptr, *u_agg_ptr;
        A_agg_ptr = A_seq;
        u_agg_ptr = u_seq;

        // Phase 1/2: aggregating results for A and u at each chunk level
        for (level = 0; level < num_levels; level++)
        {
            num_chunks_local = (n + CHUNK_SIZE - 1) / CHUNK_SIZE;

            dim3 grid(D, B_size, num_chunks_local);
            dim3 block(CHUNK_SIZE);

            parallel_scan_SSM_diag_kernel_fwd<<<grid, block, 0, stream>>>(
                A, B, x, A_agg_ptr, u_agg_ptr, A_last, u_last,
                A_agg[level], u_agg[level],
                B_size, T, D, n, num_chunks_local, level);

            // next level operates on the aggregates
            A_agg_ptr = A_agg[level];
            u_agg_ptr = u_agg[level];
            level_sizes[level] = n;
            n = num_chunks_local;
        }

        // Phase 3: applying prefixes to the block-level values
        for (level = num_levels - 1; level >= 0; level--)
        {
            A_agg_ptr = (level == 0) ? A_seq : A_agg[level - 1];
            u_agg_ptr = (level == 0) ? u_seq : u_agg[level - 1];
            num_chunks_local = (level_sizes[level] + CHUNK_SIZE - 1) / CHUNK_SIZE;

            dim3 grid(D, B_size, num_chunks_local);
            dim3 block(CHUNK_SIZE);

            apply_prefixes<<<grid, block, 0, stream>>>(
                A_agg_ptr, u_agg_ptr,
                A_agg[level], u_agg[level],
                B_size, D, level_sizes[level], num_chunks_local);
        }
    }
}