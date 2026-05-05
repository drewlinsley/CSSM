# CSSM Architecture Deep Dive

This document explains the internal architecture of CSSM models.

## Core Concept: Spectral State Space

CSSM operates in the **spectral (frequency) domain** using FFT:

1. **Input** is transformed to frequency domain via FFT
2. **State dynamics** operate on spectral coefficients
3. **Output** is transformed back via inverse FFT

This makes spatial convolution become pointwise multiplication in frequency domain, enabling efficient parallel computation.

## Log-Space Computation (GOOM)

To ensure numerical stability and enable parallel scans, CSSM uses log-space computation:

- **GOOM** (Generalized Order of Magnitude): Represents numbers as (log_magnitude, phase)
- Multiplication becomes addition in log-space
- Enables associative scan for O(log T) parallel computation

## Associative Scan

Traditional RNN: O(T) sequential
```
h_1 = f(h_0, x_1)
h_2 = f(h_1, x_2)
h_3 = f(h_2, x_3)
...
```

CSSM Associative Scan: O(log T) parallel
```
# Level 1: Pairwise combine
(h_1, h_2), (h_3, h_4), ...

# Level 2: Combine pairs
(h_1..4), (h_5..8), ...

# Level log(T): Final result
```

This requires the combine operation to be **associative**:
```
(a ∘ b) ∘ c = a ∘ (b ∘ c)
```

## HGRUBilinearCSSM Architecture

```
Input: (B, T, H, W, C)
    │
    ▼
┌─────────────────────────────────┐
│  Position Encoding (RoPE)       │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  FFT to Spectral Domain         │
│  (B, T, C, H, W_freq)           │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Input-Dependent Gates          │
│  • Decay gates (α, δ, ε)        │
│  • Coupling gates (μ, γ)        │
│  • I/O gates (B, C)             │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  3x3 State Dynamics             │
│  [X]   [decay_x  -μ_I  -α_I] [X]   [U_X]
│  [Y] = [μ_E     decay_y α_E] [Y] + [U_Y]
│  [Z]   [γ        δ       ε ] [Z]   [U_Z]
│                                 │
│  Spatial convolution via        │
│  spectral multiplication        │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Associative Scan (O(log T))    │
│  GOOM log-space computation     │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Output Gating                  │
│  Concat [X, Y, Z] + Project     │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Inverse FFT to Spatial         │
└─────────────────────────────────┘
    │
    ▼
Output: (B, T, H, W, C)
```

## SimpleCSSM Full Architecture

```
Input: (B, T, H, W, 3) or (B, H, W, 3)
    │
    ▼
┌─────────────────────────────────┐
│  Temporal Broadcast (if image)  │
│  Repeat to (B, T, H, W, 3)      │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Stem: Conv Block 1             │
│  Conv(3→embed_dim) + Act + Norm │
│  MaxPool 2x2                    │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Stem: Conv Block 2             │
│  Conv + Act + Norm + MaxPool    │
│  Resolution: H/4, W/4           │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Position Embeddings            │
│  (spatiotemporal RoPE default)  │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  CSSM Block(s) × depth          │
│  With residual connection       │
│  x = x + CSSM(x)                │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Frame Selection                │
│  'last': x[:, -1]               │
│  'all': keep temporal           │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Readout                        │
│  Norm → Act → Pool → Norm       │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Classification Head            │
│  Dense(num_classes)             │
└─────────────────────────────────┘
    │
    ▼
Output: (B, num_classes)
```

## Position Encoding Options

### VideoRoPE (spatiotemporal)
- Low frequencies for temporal (smooth time evolution)
- High frequencies for spatial (fine position detail)
- Rotary encoding preserves relative positions

### Spatial-Only
- Only encodes (h, w) position
- Time is implicit in recurrence
- Better for length generalization

### Sinusoidal Temporal
- Classic Transformer positional encoding
- Natural extrapolation to longer sequences

## CSSM-ViT Architecture

For larger-scale training (ImageNet):

```
Input: (B, T, H, W, 3)
    │
    ▼
┌─────────────────────────────────┐
│  Stem (Patch or Conv)           │
│  Patches: patch_size × patch_size│
│  → (B, T, H/patch, W/patch, D)  │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Pre-Norm CSSM Blocks × depth   │
│  ┌─────────────────────────┐    │
│  │ Norm → CSSM → +residual │    │
│  │ Norm → MLP → +residual  │    │
│  └─────────────────────────┘    │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Global Pool + Norm + Head      │
└─────────────────────────────────┘
```

## Parameter Efficiency

Compared to attention (O(n²) memory):

| Component | Attention | CSSM |
|-----------|-----------|------|
| Spatial | O(HW × HW) | O(HW × kernel²) |
| Temporal | O(T²) | O(T) via scan |
| Memory | O(T × HW)² | O(T × HW × k²) |

CSSM scales linearly with sequence length due to the associative scan.
