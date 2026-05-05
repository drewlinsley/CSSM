# Spatiotemporal Gated DeltaNet: Implementation Specification

## What You Are Building

A neural network layer for video that performs **global spatiotemporal attention** in linear time. It processes tensors of shape `[B, T, H, W, D]` (batch, time, height, width, channels) and lets every spatial position attend to every other position across all past timesteps — without the quadratic cost of standard attention.

The core idea has three parts:

1. **Gated Delta Networks (GDN)** — a recurrent update rule for maintaining an associative key-value memory over time, taken from the ICLR 2025 paper "Gated Delta Networks: Improving Mamba2 with Delta Rule" by Yang, Kautz, Hatamizadeh.
2. **2D Convolutions** — standard spatial feature extraction applied independently per frame.
3. **Fourier decomposition** — converting 2D spatial convolutions into independent per-frequency operations via the FFT, so the temporal recurrence can run in parallel across all spatial frequencies.

The result: each frame's spatial features are convolved, transformed to Fourier space, and fed into independent Gated DeltaNet recurrences per frequency bin. This gives global spatial interaction (through the Fourier basis) combined with temporal memory (through the recurrence), all at O(T·H·W·D·d) cost instead of O((T·H·W)²·D) for full attention.

---

## Background: The Three Recurrences

All three recurrences operate on a **state matrix** S that acts as an associative memory. You write key-value pairs into it and read from it with queries.

### Reading (same for all three):

```
o_t = S_t · q_t
```

S is a matrix, q is a vector, o is a vector. This retrieves the value associated with query q from the memory.

### Recurrence 1: Standard Gated DeltaNet (1D sequences, as in the paper)

```
S_t = α_t · S_{t-1} · (I - β_t · k_t · k_tᵀ) + β_t · v_t · k_tᵀ
```

**Dimensions and types:**

| Symbol | Shape | Type | Description |
|--------|-------|------|-------------|
| `S_t` | `[d_v, d_k]` | real | Hidden state matrix (the associative memory) |
| `k_t` | `[d_k]` | real | Key vector — "what this input represents" for storage |
| `v_t` | `[d_v]` | real | Value vector — "what to store" at this key |
| `q_t` | `[d_k]` | real | Query vector — "what to look up" (used only for readout) |
| `α_t` | scalar | real, (0,1) | Decay/forget gate — how much of the whole state to keep |
| `β_t` | scalar | real, (0,1) | Write strength — how aggressively to overwrite at key k_t |
| `I` | `[d_k, d_k]` | real | Identity matrix |
| `k_t · k_tᵀ` | `[d_k, d_k]` | real | Outer product — rank-1 projection onto k_t direction |
| `v_t · k_tᵀ` | `[d_v, d_k]` | real | Outer product — the new key-value pair to write |

**What each part does:**

- `α_t · S_{t-1}`: Decay the entire state by scalar α. Setting α→0 wipes memory; α→1 keeps everything.
- `(I - β_t · k_t · k_tᵀ)`: A Householder-like reflection that erases the component of S along the k_t direction. This removes the OLD value associated with key k_t.
- `β_t · v_t · k_tᵀ`: Write the NEW value v_t at key k_t.

Combined: decay everything, erase the old value at this key, write the new value. Two knobs: α controls bulk forgetting, β controls surgical replacement.

### Recurrence 2: Convolutional Gated DeltaNet (per spatial position)

```
S_t(x) = α_t(x) · S_{t-1}(x) · (I - β_t(x) · k_t(x) · k_t(x)ᵀ) + β_t(x) · v_t(x) · k_t(x)ᵀ
```

where the keys, queries, values are produced by 2D convolution:

```
k_t(x) = (w_k * z_t)(x) = Σ_y w_k(x - y) · z_t(y)
```

**New notation:**

| Symbol | Shape | Type | Description |
|--------|-------|------|-------------|
| `x` | `[2]` (int) | — | Spatial position (pixel coordinate) on the 2D grid |
| `S_t(x)` | `[d_v, d_k]` | real | State matrix **at position x** — every pixel has its own memory |
| `z_t(x)` | `[D]` | real | Raw input feature map at position x, time t |
| `w_k` | `[K, K, D_in, D_out]` | real | 2D conv kernel (e.g. 3×3) |
| `w_k * z_t` | convolution | — | Standard 2D convolution — mixes local spatial neighborhood |
| `α_t(x), β_t(x)` | scalar per position | real, (0,1) | Gating, now spatially varying |

**What changed:** The recurrence is structurally identical to Recurrence 1, just applied independently at each spatial position. Spatial mixing only happens through the conv that produces k, q, v — the states at different positions never directly interact.

### Recurrence 3: Fourier Convolutional Gated DeltaNet (the target)

Apply the 2D Discrete Fourier Transform over space. By the convolution theorem, convolution in spatial domain becomes pointwise multiplication in frequency domain. The recurrence becomes:

```
Ŝ_t(ω) = α_t(ω) · Ŝ_{t-1}(ω) · (I - β_t(ω) · k̂_t(ω) · k̂_t(ω)†) + β_t(ω) · v̂_t(ω) · k̂_t(ω)†
```

**New/changed notation:**

| Symbol | Shape | Type | Description |
|--------|-------|------|-------------|
| `ω = (ω₁, ω₂)` | `[2]` (int) | — | 2D spatial frequency index (replaces spatial position x) |
| `N_ω` | scalar | — | Number of frequency bins = H × (W//2 + 1) using Hermitian symmetry |
| `ˆ` (hat) | — | — | Denotes 2D Discrete Fourier Transform of the quantity |
| `Ŝ_t(ω)` | `[d_v, d_k]` | **complex** | State matrix at frequency ω |
| `k̂_t(ω)` | `[d_k]` | **complex** | Key at frequency ω = ŵ_k(ω) · ẑ_t(ω) (conv became pointwise multiply) |
| `†` | — | — | Conjugate transpose (replaces ᵀ because vectors are now complex) |
| `k̂ · k̂†` | `[d_k, d_k]` | complex | Hermitian outer product (rank-1, complex) |
| `α_t(ω), β_t(ω)` | scalar per freq | **real**, (0,1) | Per-frequency gating (must remain real for stability) |

**What changed from Recurrence 2:**

- Position x became frequency ω
- Real became complex
- Transpose ᵀ became conjugate transpose †
- The structure is otherwise **identical**

**Why this is better:** In Recurrence 2, each spatial position has an isolated state. In Recurrence 3, each frequency bin captures a **global spatial pattern** — low frequencies track large-scale structure (object positions, scene layout), high frequencies track fine details (textures, edges). The inverse FFT at the output reconstructs spatial information from all frequencies, giving every pixel access to global context.

---

## Architecture

### Full Layer Forward Pass

```
Input: x ∈ ℝ[B, T, H, W, D]

═══ Step 1: Per-frame 2D Conv projections ═══

  Treat T as a batch dimension. Apply 2D convolutions over (H, W):

    Reshape: [B, T, H, W, D] → [B·T, D, H, W]  (channels-first for conv)

    Conv2D (3×3, groups=D → depthwise) + Conv2D (1×1 → pointwise):
      q_raw = Conv1x1_q(DepthwiseConv3x3(x))    → [B·T, n_h·d_k, H, W]
      k_raw = Conv1x1_k(DepthwiseConv3x3(x))    → [B·T, n_h·d_k, H, W]
      v_raw = Conv1x1_v(DepthwiseConv3x3(x))    → [B·T, n_h·d_v, H, W]

    Activations:
      q = L2Norm(SiLU(q_raw), dim=d_k)           per-head L2 normalization
      k = L2Norm(SiLU(k_raw), dim=d_k)           per-head L2 normalization
      v = SiLU(v_raw)

    Output gate (for final output modulation):
      gate = SiLU(Linear(x))                      → [B, T, H, W, D]

    Reshape: [B·T, n_h, d, H, W] → [B, T, H, W, n_h, d]

═══ Step 2: Compute α, β per frequency ═══

  Pool spatial dims to get a global frame descriptor:
    x_pool = mean(x, dims=[H, W])                → [B, T, D]

  Project to per-frequency scalars:
    α = sigmoid(Linear_α(x_pool))                → [B, T, n_h, N_ω]   real, in (0,1)
    β = sigmoid(Linear_β(x_pool))                → [B, T, n_h, N_ω]   real, in (0,1)

  N_ω = H × (W // 2 + 1)  due to Hermitian symmetry of real-input FFT.
  The Linear layers have output dimension n_h × N_ω.

═══ Step 3: Batched 2D rFFT ═══

  Apply torch.fft.rfft2 (or jnp.fft.rfft2) over the (H, W) dimensions:

    q̂ = rfft2(q, dims=(-3, -2))                 → [B, T, n_h, H, W//2+1, d_k]  complex
    k̂ = rfft2(k, dims=(-3, -2))                 → same
    v̂ = rfft2(v, dims=(-3, -2))                 → [B, T, n_h, H, W//2+1, d_v]  complex

  Hermitian symmetry means we only compute/store half the frequency bins along W.

═══ Step 4: Reshape for GDN kernel ═══

  Merge spatial frequency bins into the head dimension:

    q̂: [B, T, n_h, H, W//2+1, d_k]
      → [B, n_h · H · (W//2+1), T, d_k]
      =  [B, H_eff, T, d_k]

  where H_eff = n_h × N_ω is the "virtual head" count.

  Similarly reshape k̂, v̂, α, β.

  α, β: [B, T, n_h, N_ω] → [B, H_eff, T]    (one scalar per virtual head per timestep)

═══ Step 5: Chunkwise Gated DeltaNet over T ═══

  THIS IS THE CORE RECURRENCE — detailed in the next section.

  Input:  q̂, k̂, v̂ ∈ ℂ[B, H_eff, T, d]
          α, β ∈ ℝ[B, H_eff, T]
  Output: ô ∈ ℂ[B, H_eff, T, d_v]

  Processes T timesteps using the chunkwise parallel algorithm.
  Parallel over B × H_eff. Sequential (chunked) over T.

═══ Step 6: Inverse FFT ═══

  Reshape back:
    ô: [B, H_eff, T, d_v] → [B, T, n_h, H, W//2+1, d_v]

  Apply irfft2 over spatial frequency dims:
    o = irfft2(ô, dims=(-3, -2), s=(H, W))     → [B, T, H, W, n_h, d_v]  real

  Merge heads: → [B, T, H, W, n_h·d_v] = [B, T, H, W, D]

═══ Step 7: Output ═══

  o_final = OutProjection(Norm(o) ⊙ gate)       → [B, T, H, W, D]

  Norm = GroupNorm or RMSNorm over the channel dimension.
  ⊙ = elementwise multiply with the gate from Step 1.
  OutProjection = Linear layer, D → D.
```

---

## The Chunkwise Parallel Algorithm (Step 5 in detail)

This section explains how to parallelize the sequential recurrence over T for efficient training on GPUs.

### Why parallelization is needed

The recurrence `Ŝ_t = f(Ŝ_{t-1}, k̂_t, v̂_t, α_t, β_t)` is inherently sequential — each step depends on the previous state. Running it step-by-step wastes GPU resources because each step involves only a tiny matmul (d×d), while the GPU has thousands of cores idle.

### The chunkwise strategy

Split the T timesteps into chunks of size C. Within each chunk, express all outputs in terms of:
- The chunk's initial state (inherited from previous chunk)
- The chunk's local keys, queries, values

The intra-chunk computation becomes large matmuls (C×d) that fully utilize the GPU. The only sequential part is passing the state from chunk to chunk.

### Notation for the chunkwise algorithm

| Symbol | Meaning |
|--------|---------|
| `C` | Chunk size (e.g., 32 or 64) |
| `[t]` subscript | Chunk index (t = 0, 1, ..., T/C - 1) |
| `r` superscript | Position within a chunk (r = 1, ..., C) |
| `K̂_[t]` | `∈ ℂ[C, d_k]` — all C keys in chunk t, stacked as rows |
| `Q̂_[t]` | `∈ ℂ[C, d_k]` — all C queries in chunk t |
| `V̂_[t]` | `∈ ℂ[C, d_v]` — all C values in chunk t |
| `α_[t]` | `∈ ℝ[C]` — decay scalars for all positions in chunk t |
| `β_[t]` | `∈ ℝ[C]` — write strengths for all positions in chunk t |
| `Ŝ_[t]` | `∈ ℂ[d_v, d_k]` — state at the START of chunk t |
| `γʳ_[t]` | `= Π_{i=1}^{r} α^i_[t]` — cumulative decay product within chunk |
| `γ_C` | `= γ^C_[t]` — total decay over the entire chunk |
| `Γ_[t]` | `∈ ℝ[C, C]` — decay ratio matrix: Γ_ij = γ_i / γ_j for i ≥ j, else 0 |

### The algorithm, step by step

For each chunk t, in parallel over all B × H_eff virtual heads:

```
PHASE A: Intra-chunk computation (all in GPU SRAM)
──────────────────────────────────────────────────

  A1. Load K̂, V̂, Q̂, α, β for this chunk from HBM into SRAM.

  A2. Cumulative decay products:
        γ[r] = α[1] · α[2] · ... · α[r]      for r = 1..C
      This is a tiny prefix product (scan) over C scalars.

  A3. Build decay ratio matrix:
        Γ[i,j] = γ[i] / γ[j]     if i ≥ j
        Γ[i,j] = 0                if i < j     (causal mask)
      Shape: [C, C], real-valued.

  A4. Gram matrix (decay-weighted key self-similarity):
        G = Γ ⊙ (K̂ · K̂†)
      Shape: [C, C], complex. This is a matmul K̂ @ K̂.conj().T then
      elementwise multiply with Γ.

  A5. WY forward substitution (triangular solve):
      Build the matrix T that represents the cumulative product of
      Householder reflections. This is done row by row:

        T[0,:] = β[0] · e_0                    (first row is trivial)

        For r = 1, ..., C-1:
          T[r,:] = β[r] · (e_r - Σ_{i<r} T[i,:] · G[i,r])

      Each row depends on all previous rows — this is sequential over C.
      But C is small (32-64) and everything is in SRAM, so it's fast.

      Shape of T: [C, C], complex.

  A6. Compute the WY-transformed values and keys:
        Ug = T @ V̂           shape: [C, d_v]   (matmul in SRAM)
        W  = T @ K̂           shape: [C, d_k]   (matmul in SRAM)


PHASE B: State interaction (incorporate chunk initial state)
────────────────────────────────────────────────────────────

  B1. Load chunk initial state Ŝ_[t] from HBM.   shape: [d_v, d_k]

  B2. Decay keys and queries for inter-chunk interaction:
        Q̃[r] = γ[r] · Q̂[r]                     (multiply each query by its cumulative decay)
        K̃[r] = (γ_C / γ[r]) · K̂[r]              (multiply each key by decay to end of chunk)
        W̃[r] = γ[r] · W[r]                       (same for WY-transformed keys)

  B3. Compute correction (difference from what the initial state alone would give):
        correction = Ug - W̃ @ Ŝ_[t]†             shape: [C, d_v]

  B4. Compute output for this chunk:
        O_inter = Q̃ @ Ŝ_[t]†                     shape: [C, d_v]   (contribution from initial state)
        O_intra = (Q̂ @ K̂† ⊙ Γ_causal) @ correction   shape: [C, d_v]   (contribution from within chunk)
        O_[t] = O_inter + O_intra

  B5. Update state for next chunk:
        Ŝ_[t+1] = γ_C · Ŝ_[t] + correction† @ K̃    shape: [d_v, d_k]

  B6. Store O_[t] to HBM. Pass Ŝ_[t+1] to next chunk iteration.
```

### Inter-chunk sequencing

The chunks must be processed in order because each chunk needs the previous chunk's final state (step B5 → B1 of next chunk).

Two implementation strategies:

**Strategy 1: Sequential loop (recommended for video where T/C is small)**
```python
S = zeros(B, H_eff, d_v, d_k)  # initial state
for t in range(T // C):
    O_chunk, S = process_chunk(S, K_chunks[t], Q_chunks[t], V_chunks[t], ...)
```

**Strategy 2: Parallel prefix scan (for very long sequences, T/C > 16)**
The state update `Ŝ_[t+1] = γ_C · Ŝ_[t] + ΔŜ` is a linear recurrence, so it admits an associative scan:
```python
# Associative operator:
(γ_a, ΔS_a) ⊕ (γ_b, ΔS_b) = (γ_a · γ_b, γ_b · ΔS_a + ΔS_b)
# This gives O(log(T/C)) depth instead of O(T/C)
```
For video with T=128-256 and C=32, T/C = 4-8 chunks. Sequential loop is fine.

---

## Scans in the Algorithm

There are exactly three scans. Two are tiny and inside the main kernel. One is the outer temporal recurrence.

### Scan 1: Cumulative product of α (step A2)

```
γ[r] = α[1] · α[2] · ... · α[r]     for r = 1, ..., C
```

- **Size**: C = 32 scalars
- **Type**: Prefix product (associative scan with multiplication)
- **Implementation**: Simple for-loop in Triton, or `jnp.cumprod` / `jax.lax.associative_scan(multiply, ...)` in JAX
- **Cost**: Negligible (31 scalar multiplies)

### Scan 2: Forward substitution for WY representation (step A5)

```
T[r] = β[r] · (e_r - Σ_{i<r} T[i] · G[i,r])     for r = 0, ..., C-1
```

- **Size**: C = 32 steps, each doing a dot product of length ≤ C
- **Type**: Sequential (each row depends on all previous — NOT an associative scan)
- **Implementation**: for-loop in Triton or `jax.lax.scan` in JAX
- **Cost**: O(C²) = ~1024 multiply-adds per virtual head. Negligible.
- **Cannot be parallelized**: This is inherently sequential, but C is chosen small enough that it doesn't matter.

### Scan 3: Inter-chunk state passing (the outer loop)

```
Ŝ_[t+1] = γ_C · Ŝ_[t] + ΔŜ_[t]     for t = 0, ..., T/C - 1
```

- **Size**: T/C = 4-8 steps for typical video
- **Type**: Linear recurrence (can be associative scan, but not worth it for 4-8 steps)
- **Implementation**: for-loop in persistent Triton kernel, or `jax.lax.scan` in JAX
- **Cost**: T/C matrix-scalar-multiplies and matrix-adds of size [d_v, d_k]
- **Could be parallel scan**: Yes, using the associative operator above, but only worthwhile for T/C > ~16

---

## Kernel Implementation Plan

### Overview: What to implement and how

| Operation | Implementation | Why |
|-----------|---------------|-----|
| 2D Conv projections | cuDNN / `torch.nn.Conv2d` / `jax.lax.conv` | Library-optimized, no custom kernel needed |
| SiLU + L2 Norm | `torch.compile` fused kernel or Triton | Simple pointwise ops, fusion avoids extra memory round-trips |
| α, β projection | `torch.compile` / `jax.jit` | Tiny — just linear + sigmoid on pooled features |
| Batched rFFT2 | `torch.fft.rfft2` / `jnp.fft.rfft2` | Library-optimized FFT |
| **Chunkwise GDN** | **Custom Triton or Pallas kernel** | Core of the model — needs custom kernel for performance |
| Batched irFFT2 | `torch.fft.irfft2` / `jnp.fft.irfft2` | Library-optimized FFT |
| Output norm + gate + proj | `torch.compile` fused kernel or Triton | Simple pointwise ops |

### The GDN Kernel (Triton)

This is the one custom kernel you need. Here is the design:

**Launch grid**: `[B × H_eff]` programs. Each program handles one virtual head across all chunks.

**Each program does:**
1. Initialize state S = zeros(d_v, d_k) in registers
2. Loop over chunks t = 0, ..., T/C - 1:
   a. Load chunk data (K̂, Q̂, V̂, α, β) from HBM into SRAM
   b. Run Phase A (intra-chunk) entirely in SRAM
   c. Run Phase B (state interaction) using S from registers
   d. Update S in registers
   e. Store output chunk to HBM
3. Store final state S to HBM (for backward pass checkpointing)

**Key property**: The state S stays in registers across all chunks — zero HBM traffic for state between chunks. This is why a persistent kernel beats a scan of separate kernel launches.

**Complex arithmetic**: Store everything as pairs of real tensors (real part, imaginary part). All matmuls become 4 real matmuls:
```
(A_re + i·A_im)(B_re + i·B_im) = (A_re·B_re - A_im·B_im) + i·(A_re·B_im + A_im·B_re)
```
This avoids complex-number support issues in Triton and keeps Tensor Cores in their optimized real-valued mode.

### The GDN Kernel (JAX / Pallas)

In JAX, use `jax.lax.scan` for the outer chunk loop and the inner forward substitution. XLA will compile and optimize the matmuls.

```python
def process_chunk(S, chunk_data):
    """Process one chunk. S is the carry (recurrent state)."""
    K, Q, V, alpha, beta = chunk_data
    # ... Phase A and Phase B computations ...
    return S_new, O_chunk

init_state = jnp.zeros([B, H_eff, d_v, d_k], dtype=jnp.complex64)
final_state, all_outputs = jax.lax.scan(process_chunk, init_state, chunked_inputs)
```

**Advantage of JAX**: `jax.lax.scan` is auto-differentiable. You get the backward pass for free, including gradient checkpointing at chunk boundaries (via `jax.checkpoint`). In Triton, you must write the backward kernel manually.

---

## Recommended Design Parameters

### Sizing guidelines

The key constraint is state memory: H_eff × d_v × d_k × bytes_per_element.

| Parameter | Notation | Recommended | Notes |
|-----------|----------|-------------|-------|
| Spatial resolution | H, W | Full (e.g. 64×64) | No downsampling — this is the point |
| Model dimension | D | 512 | Total channels |
| Num heads | n_h | 16 | Many heads × small head dim |
| Head dim (key) | d_k | 32 | Determines Gram matrix size — 32 is good for Tensor Cores |
| Head dim (value) | d_v | 32 | Can differ from d_k (D = n_h × d_v) |
| Freq bins | N_ω | H × (W//2+1) | Hermitian half-spectrum (e.g. 64×33 = 2112) |
| Virtual heads | H_eff | n_h × N_ω | e.g. 16 × 2112 = 33,792 |
| Chunk size | C | 32 | Should fit intra-chunk data in SRAM |
| Sequence length | T | 128-512 | Video frames |

### Memory budget (per sample)

```
State:           H_eff × d_v × d_k × 8 bytes (complex64 as 2×float32)
                 = 33,792 × 32 × 32 × 8 ≈ 276 MB

One chunk's KQV:  3 × H_eff × C × d × 8 bytes
                  = 3 × 33,792 × 32 × 32 × 8 ≈ 789 MB

With activation checkpointing, you only hold one chunk live at a time.
Total working memory ≈ 276 + 789 ≈ 1.1 GB per sample.
```

### Compute budget (per layer)

```
Chunkwise GDN:   ~7 TFLOP per chunk × T/C chunks ≈ 28 TFLOP
FFT/IFFT:        O(T·H·W·D·log(HW)) ≈ 0.5 TFLOP
Conv projections: O(T·H·W·D·K²) ≈ 2.4 TFLOP (for 3×3 depthwise + 1×1)

Total per layer:  ~31 TFLOP
H100 throughput:  ~990 TFLOPS (FP16)
Time per layer:   ~31 ms (compute-bound estimate; memory bandwidth may dominate)
```

---

## α, β Design: Per-Frequency Gating

The decay (α) and write strength (β) must be real scalars in (0,1) per frequency bin. They should NOT be FFT'd from spatial-domain values (that would make them complex).

**Recommended approach**: Compute from spatially-pooled features.

```python
# Pool over spatial dims to get a global frame descriptor
x_pooled = x.mean(dim=[H, W])                    # [B, T, D]

# Project to per-frequency, per-head scalars
alpha = sigmoid(Linear(x_pooled))                 # [B, T, n_h × N_ω]
beta  = sigmoid(Linear(x_pooled))                 # [B, T, n_h × N_ω]

# Reshape to match virtual head layout
alpha = alpha.reshape(B, T, n_h, N_ω)             # → merge into H_eff later
beta  = beta.reshape(B, T, n_h, N_ω)
```

**Inductive bias**: Different frequency bands should forget at different rates. Low frequencies (large-scale scene structure) persist → higher α. High frequencies (textures, details) change fast → lower α. The Linear projection can learn this from data, or you can add a learnable per-frequency bias:

```python
alpha_base = sigmoid(nn.Parameter(torch.zeros(n_h, N_ω)))   # learned per-freq prior
alpha_mod = sigmoid(Linear(x_pooled))                        # data-dependent modulation
alpha = alpha_base * alpha_mod                                # combined
```

---

## Backward Pass

### JAX: Automatic

```python
@jax.checkpoint   # saves only chunk boundary states, recomputes intermediates
def layer_forward(params, x):
    ...
    final_state, outputs = jax.lax.scan(process_chunk, init_state, chunks)
    ...

grad_fn = jax.grad(loss_fn)   # automatic differentiation through the scan
```

### PyTorch/Triton: Manual

You must implement a custom backward kernel that:

1. Loads the saved chunk boundary states (from the forward pass)
2. Recomputes intra-chunk intermediates (Ug, W, correction) for each chunk
3. Propagates gradients backward through chunks (reverse scan)
4. Accumulates gradients for K, Q, V, α, β

The GDN paper's existing backward kernel handles the WY representation gradients. Your extension adds complex arithmetic but the structure is the same.

**Recommendation**: Start in JAX for correctness (free autodiff), then port to Triton for production speed.

---

## File Structure

```
spatiotemporal_gdn/
├── model.py                 # Top-level layer: conv → FFT → GDN → IFFT → output
├── gdn_recurrence.py        # Chunkwise GDN algorithm (Python reference implementation)
├── gdn_kernel.py            # Triton kernel for the chunkwise GDN (or Pallas for JAX)
├── projections.py           # Conv2D projections for q, k, v, α, β, gate
├── utils.py                 # FFT/IFFT wrappers, Hermitian symmetry handling
└── config.py                # Hyperparameters: n_h, d_k, d_v, C, etc.
```

---

## Summary of Key Equations

**The recurrence** (one line — this is what you're implementing):

```
Ŝ_t(ω) = α_t(ω) · Ŝ_{t-1}(ω) · (I - β_t(ω) · k̂_t(ω) · k̂_t(ω)†) + β_t(ω) · v̂_t(ω) · k̂_t(ω)†
```

**The readout** (how to get output from state):

```
ô_t(ω) = Ŝ_t(ω) · q̂_t(ω)
```

**Back to spatial domain**:

```
o_t(x) = IFFT2D[ ô_t(ω) ]
```

**Where keys/values come from** (2D conv, then FFT):

```
k_t(x) = L2Norm(SiLU(Conv2D(x_t)))     (spatial domain)
k̂_t(ω) = FFT2D[ k_t(x) ]               (frequency domain)
```

**Training parallelization**: Chunk the time dimension into groups of C. Within each chunk, use the WY representation to convert the sequential Householder products into matmuls. Pass state between chunks sequentially (or via parallel scan for long T).
