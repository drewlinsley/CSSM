# CSSM Model Variants

This document describes the available CSSM (Cepstral State Space Model) variants and their characteristics.

## Overview

CSSM models process video sequences using state space dynamics in the spectral (frequency) domain. They replace attention mechanisms with recurrent dynamics that operate on spatial frequency representations.

## Available Variants

### HGRUBilinearCSSM (Primary)

**CLI flag:** `--cssm hgru_bi`

The primary CSSM variant, inspired by the hGRU (horizontal Gated Recurrent Unit) from neuroscience.

**Key Features:**
- 3-state system: X (excitatory), Y (inhibitory), Z (interaction channel)
- Separate excitation and inhibition kernels (K_E and K_I)
- Z channel learns to approximate bilinear X*Y dynamics
- Maintains associativity for O(log T) parallel scan

**Dynamics:**
```
X_t = decay_x·X - μ_I·K_I·Y - α_I·K_I·Z + U_X
Y_t = μ_E·K_E·X + decay_y·Y + α_E·K_E·Z + U_Y
Z_t = γ·X + δ·Y + ε·Z + U_Z
```

**Best For:** Most tasks, especially those requiring lateral interactions (Pathfinder, cABC).

---

### GatedCSSM

**CLI flag:** `--cssm gated`

Mamba-style gated CSSM with input-dependent integration.

**Key Features:**
- Full Mamba formulation in log-spectral domain
- Input-dependent decay (A_bar), input projection (B_bar), and output projection (C)
- Supports both depthwise and dense (LMME) channel mixing

**Dynamics:**
```
h_t = A_bar * h_{t-1} + B_bar * u_t
y_t = C * h_t
```

**Best For:** Simpler tasks, faster training, fewer parameters.

---

### TransformerCSSM

**CLI flag:** `--cssm transformer` or `--cssm kqv`

Minimal transformer-inspired CSSM with Q, K, A states.

**Key Features:**
- Q (Query), K (Key), A (Attention/Accumulator) state channels
- Additive dynamics using GOOM (Generalized Order of Magnitude)
- A accumulates Q-K correlation over time (iteratively growing attention)
- Single spatial kernel (simplified from E/I)

**Dynamics:**
```
Q_t = decay_Q·Q + w·K·K + α·K·A + U_Q
K_t = w·K·Q + decay_K·K + U_K
A_t = γ·Q + γ·K + decay_A·A + U_A
```

**Best For:** Tasks benefiting from attention-like dynamics, interpretability studies.

---

### MultiplicativeTransformerCSSM

**CLI flag:** `--cssm mult_transformer`

Multiplicative Q/K/A dynamics in log-spectral space.

**Key Features:**
- Purely multiplicative dynamics become LINEAR in log space
- No GOOM/LSE complexity - uses regular matrix multiplication
- LayerNorm in log-space prevents overflow

**Log-space Dynamics:**
```
q_t = u_Q + k_s + a_qq·q_{t-1} + a_qk·k_{t-1} + a_qa·a_{t-1}
k_t = u_K + k_s + a_kq·q_{t-1} + a_kk·k_{t-1}
a_t = u_A + γ·q_{t-1} + γ·k_{t-1} + a_aa·a_{t-1}
```

**Options:**
- `--no_goom`: Disable GOOM (assumes positive inputs, faster but less flexible)

**Best For:** Experimental, when multiplicative dynamics are desired.

---

## Ablation Variants (Testing Only)

These variants exist for ablation studies and are not recommended for production use:

### LinearCSSM
Vanilla spectral convolution without log-space. Uses sequential O(T) scan instead of parallel O(log T).

### LinearOpponentCSSM
2x2 opponent dynamics without log-space. Sequential scan.

---

## Architecture Options

### SimpleCSSM Architecture
The SimpleCSSM architecture (`--arch simple`) is recommended for smaller tasks:

```
Conv -> act -> norm -> maxpool
Conv -> act -> norm -> maxpool
+ Position Embeddings
CSSM block(s)
Frame selection -> Norm -> act -> pool -> head
```

### CSSM-ViT Architecture
The CSSM-ViT architecture (`--arch vit`) is for larger-scale training:

- Patch or conv stem
- Pre-norm transformer-style blocks
- CSSM replaces attention
- Global pooling + classifier head

---

## Common Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--kernel_size` | Spatial kernel size for E/I convolutions | 11 |
| `--block_size` | Channel mixing block size (1=depthwise) | 1 |
| `--readout_state` | Which state(s) to use for output | 'xyz' |
| `--pre_output_act` | Activation before output projection | 'none' |

---

## Choosing a Variant

| Task | Recommended | Reason |
|------|-------------|--------|
| Pathfinder | `hgru_bi` | Lateral E/I interactions critical |
| cABC | `hgru_bi` | Contour integration benefits from E/I |
| ImageNet | `hgru_bi` or `gated` | Both work well |
| Fast training | `gated` | Fewer parameters |
| Interpretability | `transformer` | Explicit Q/K/A |
