# CSSM Architecture Progression: From Vanilla to hGRU-Style

## Overview

CSSM (Cepstral State Space Model) operates in the **spectral domain** using FFT, combined with **temporal recurrence** via parallel associative scans. All variants share this spectral processing but differ in their temporal dynamics.

**Key Innovation**: All operations happen in log-space (GOOM = Generalized Order of Magnitude) for numerical stability, enabling O(log T) parallel scans instead of O(T) sequential RNNs.

---

## 1. Vanilla CSSM (StandardCSSM)

The simplest form: a scalar linear recurrence in spectral domain.

### Equations

**State Update** (per-channel, per-frequency):
```
H_t = K · H_{t-1} + U_t
```

Where:
- `H_t` ∈ ℂ^(C×H×W_freq) — Hidden state at time t (complex-valued spectral representation)
- `K` ∈ ℂ^(C×H×W_freq) — Learnable spectral kernel (per-channel, per-frequency)
- `U_t` = FFT(x_t) — Input in spectral domain

### Log-Space (GOOM) Formulation

Since multiplication in linear space becomes addition in log-space:
```
log(H_t) = log(K · H_{t-1} + U_t)
         = LSE(log(K) + log(H_{t-1}), log(U_t))
```

Where LSE = Log-Sum-Exp (log-space addition).

### Associative Scan Operator

```python
def cssm_scalar_scan_op(carry_i, carry_j):
    k_i, u_i = carry_i  # (kernel_log, state_log)
    k_j, u_j = carry_j

    k_new = k_j + k_i                    # log(K_j · K_i)
    u_new = log_add_exp(k_j + u_i, u_j)  # log(K_j·u_i + u_j)

    return k_new, u_new
```

### Properties

| Property | Value |
|----------|-------|
| State dimension | Scalar (per channel) |
| Complexity | O(C) per step, O(log T) parallel |
| Cross-channel | ✗ Independent channels |
| Gating | ✗ None |
| Nonlinearity | ✗ Linear only |

### Limitations

- Each channel is independent (no cross-channel interaction)
- Purely linear dynamics — no nonlinear gating
- Fixed kernel (not input-dependent)

---

## 2. Gated CSSM (GatedCSSM)

Adds **Mamba-style input-dependent gating** to the scalar recurrence.

### Equations

**State Update** with input-dependent gates:
```
H_t = A_bar · H_{t-1} + B_bar · U_t
Y_t = C · H_t
```

Where:
- `A_bar = K · exp(-Δ)` — Input-dependent decay (Δ is gated)
- `B_bar` — Input-dependent input projection
- `C` — Input-dependent output projection

### Input-Dependent Gates (Mamba-Style)

All gates are learned projections from spatial context:
```python
ctx = x.mean(axis=(2, 3))  # Spatial pooling: (B, T, C)

# Per-channel, per-frequency gates
Δ = softplus(Dense(ctx))   # Decay rate (controls memory)
B_gate = sigmoid(Dense(ctx))  # Input gating
C_gate = sigmoid(Dense(ctx))  # Output gating
```

### Gate Effect on Dynamics

```
# Decay gating (Δ controls how fast state decays)
K_gated = K · exp(-Δ)  # Larger Δ → faster decay → shorter memory

# Input gating
U_modulated = U_hat · B_gate  # Controls what enters state

# Output gating
Y_hat = H_hat · C_gate  # Controls what exits state
```

### Properties

| Property | Value |
|----------|-------|
| State dimension | Scalar (per channel) |
| Complexity | O(C) per step, O(log T) parallel |
| Cross-channel | ✗ Independent channels |
| Gating | ✓ Input-dependent (Δ, B, C) |
| Nonlinearity | ✗ Linear dynamics, nonlinear gates |

### Advance Over Vanilla

- Input-dependent gating allows selective memory
- Δ gate enables variable-length temporal integration
- B/C gates act like attention over time

---

## 3. Gated Opponent CSSM (GatedOpponentCSSM)

Adds **biologically-inspired 2×2 coupled oscillator** dynamics with separate excitation (X) and inhibition (Y) pathways.

### Biological Motivation

Inspired by **center-surround receptive fields** in visual cortex:
- **X pathway**: Excitation (ON cells) — responds to stimulus
- **Y pathway**: Inhibition (OFF cells) — suppresses over time
- Coupled via opponent dynamics: X excites Y, Y inhibits X

### Equations

**2×2 Coupled State Update**:
```
[X_t]   [α·decay    -μ·K_I ] [X_{t-1}]   [U_t]
[Y_t] = [γ·K_E      δ·decay] [Y_{t-1}] + [0  ]
```

Expanded:
```
X_t = α·decay·X_{t-1} − μ·K_I·Y_{t-1} + U_t
Y_t = γ·K_E·X_{t-1} + δ·decay·Y_{t-1}
```

Where:
- `α, δ` — Self-decay gates (diagonal, input-dependent)
- `μ` — Inhibition strength: Y → X coupling (negative, off-diagonal)
- `γ` — Excitation strength: X → Y coupling (positive, off-diagonal)
- `K_E` — Excitatory spatial kernel (learned, in spectral domain)
- `K_I` — Inhibitory spatial kernel (learned, in spectral domain)
- `decay` ∈ (0.1, 0.99) — Learnable per-channel base decay

### Transition Matrix Structure

```
      ┌─────────────────────────────────────┐
      │  α·decay      │    -μ·K_I          │
      │  (X self)     │    (Y inhibits X)  │
A =   ├───────────────┼────────────────────┤
      │  γ·K_E        │    δ·decay         │
      │  (X excites Y)│    (Y self)        │
      └─────────────────────────────────────┘
```

### Input-Dependent Gates

All 6 gates are input-dependent via learned projections:
```python
ctx = x.mean(axis=(2, 3))  # Spatial pooling: (B, T, C)

# Decay gates (diagonal)
alpha = sigmoid(Dense(ctx))  # X self-decay
delta = sigmoid(Dense(ctx))  # Y self-decay

# Coupling gates (off-diagonal)
mu = sigmoid(Dense(ctx))     # Y→X inhibition strength
gamma = sigmoid(Dense(ctx))  # X→Y excitation strength

# I/O gates
B_gate = sigmoid(Dense(ctx))  # Input gating
C_gate = sigmoid(Dense(ctx))  # Output gating
```

### Log-Space 2×2 Matrix Scan

The 2×2 transition matrix is processed in GOOM:
```python
def cssm_matrix_scan_op(carry_i, carry_j):
    K_i, u_i = carry_i  # K: (..., 2, 2), u: (..., 2)
    K_j, u_j = carry_j

    K_new = log_matmul_2x2(K_j, K_i)      # Matrix multiplication in log-space
    Ku_i = log_matvec_2x2(K_j, u_i)       # Matrix-vector in log-space
    u_new = log_add_exp(Ku_i, u_j)        # Element-wise LSE

    return K_new, u_new
```

### Properties

| Property | Value |
|----------|-------|
| State dimension | 2D vector [X, Y] per channel |
| Complexity | O(C) per step (2×2 is constant), O(log T) parallel |
| Cross-channel | ✓ E↔I coupling via K_E, K_I |
| Gating | ✓ Full input-dependent (α, δ, μ, γ, B, C) |
| Nonlinearity | ✗ Linear coupling only |

### Advance Over Gated CSSM

- Cross-pathway E↔I interaction (like visual cortex)
- Separate excitation and inhibition kernels
- Center-surround receptive field dynamics

### Limitation

**LINEAR coupling only** — the coupling terms (μ·K_I·Y and γ·K_E·X) are linear in the state. Real neurons have nonlinear gain control.

---

## 4. hGRU-Style CSSM (HGRUStyleCSSM)

Adds **bilinear X·Y terms** for nonlinear gain control, matching the horizontal Gated Recurrent Unit (hGRU) from Linsley et al.

### Biological Motivation

Based on **horizontal GRU (hGRU)** which models:
- **Horizontal connections** in visual cortex (long-range spatial integration)
- **Nonlinear gain control**: The strength of inhibition/excitation depends on BOTH X and Y states

Key insight: In real neurons, the gating signal `(α·X + μ)` is **state-dependent**, creating multiplicative (bilinear) interactions.

### Original hGRU Equations (Linsley et al.)

```
X_t = U − (α·X + μ)·(K_I ⊛ Y) + decay·X
Y_t = (α·Y + μ)·(K_E ⊛ X) + decay·Y
```

The term **(α·X + μ)** is the **gain control signal** — it modulates the coupling strength based on the current state.

### Expanded Form

Distributing the gain control term:
```
X_t = decay·X − μ·K_I·Y − α·K_I·X·Y + U
                └──────┘   └───────┘
                 linear    bilinear

Y_t = decay·Y + μ·K_E·X + α·K_E·X·Y
                └──────┘   └───────┘
                 linear    bilinear
```

The **bilinear X·Y terms** (`α·K_I·X·Y` and `α·K_E·X·Y`) are what distinguish hGRU from linear opponent dynamics.

### Log-Space Bilinear Terms

**Key insight: In log-space, products become sums!**
```
log(X · Y) = log(X) + log(Y)
```

This means bilinear terms are **compatible with associative scans**:
```python
log_X = u_i[..., 0]  # Log of X state
log_Y = u_i[..., 1]  # Log of Y state
log_XY = log_X + log_Y  # Bilinear product is just addition in log-space!
```

### Full State Update Structure

```
Linear 2×2 Matrix (same as GatedOpponentCSSM):
┌───────────────────────────────────────┐
│  decay_x        │    -μ·K_I          │
│  (X self)       │    (Y inhibits X)  │
├─────────────────┼────────────────────┤
│  μ·K_E          │    decay_y         │
│  (X excites Y)  │    (Y self)        │
└───────────────────────────────────────┘

Plus Bilinear Corrections:
X_t += −α·K_I·X·Y    (additional inhibition, quadratic in state)
Y_t += +α·K_E·X·Y    (additional excitation, quadratic in state)
```

### Associative Scan Operator with Bilinear Terms

```python
def cssm_hgru_scan_op(carry_i, carry_j, K_inhib_bilinear, K_excit_bilinear):
    K_i, u_i = carry_i  # K: (..., 2, 2), u: (..., 2)
    K_j, u_j = carry_j

    # === LINEAR PART (same as GatedOpponentCSSM) ===
    K_new = log_matmul_2x2(K_j, K_i)
    Ku_i = log_matvec_2x2(K_j, u_i)

    # === BILINEAR PART: X * Y ===
    log_X = u_i[..., 0]
    log_Y = u_i[..., 1]
    log_XY = log_X + log_Y  # Product becomes sum in log-space!

    # Bilinear inhibition: −α·K_I·X·Y (negative, phase π)
    bilinear_inhib = K_inhib_bilinear + log_XY
    u_new_X = log_add_exp(
        log_add_exp(Ku_i[..., 0], u_j[..., 0]),
        bilinear_inhib
    )

    # Bilinear excitation: +α·K_E·X·Y (positive, phase 0)
    bilinear_excit = K_excit_bilinear + log_XY
    u_new_Y = log_add_exp(
        log_add_exp(Ku_i[..., 1], u_j[..., 1]),
        bilinear_excit
    )

    return K_new, jnp.stack([u_new_X, u_new_Y], axis=-1)
```

### Constraint for Associativity

For the associative scan to work correctly, the bilinear coefficients (`K_inhib_bilinear`, `K_excit_bilinear`) must be **time-independent**:

```python
# α is learnable but CONSTANT across time (not input-dependent)
alpha = self.param('alpha_bilinear', init, (C, H, W_freq))  # Shape: (C, H, W_freq)

# Bilinear coefficients (no batch or time dimensions)
K_inhib_bilinear = to_goom(-alpha * K_I)  # Phase π encodes negative sign
K_excit_bilinear = to_goom(+alpha * K_E)  # Phase 0 for positive
```

**Why?** The associative scan combines partial results from different time ranges. If the bilinear coefficient varied with time, the combination would be incorrect.

### Properties

| Property | Value |
|----------|-------|
| State dimension | 2D vector [X, Y] per channel |
| Complexity | O(C) per step, O(log T) parallel |
| Cross-channel | ✓ E↔I coupling via K_E, K_I |
| Gating | ✓ Mixed (μ, B, C input-dependent; α fixed) |
| Nonlinearity | ✓ Bilinear X·Y terms |

### Advance Over Gated Opponent CSSM

- **Bilinear X·Y dynamics** (like hGRU's gain control)
- Stronger nonlinearity enables more complex temporal patterns
- Better suited for tasks requiring nonlinear signal integration

### Trade-off

The bilinear coupling strength `α` must be **time-independent** (learnable but constant) to maintain associativity. The linear coupling `μ` can still be input-dependent.

---

## Comparison Table

| Feature | Vanilla | Gated | Opponent | hGRU-Style |
|---------|---------|-------|----------|------------|
| **State** | Scalar | Scalar | 2D [X,Y] | 2D [X,Y] |
| **Dynamics** | Linear | Linear | Linear coupled | Linear + Bilinear |
| **Gates** | None | Δ, B, C | α, δ, μ, γ, B, C | μ, B, C (+ fixed α) |
| **Input-dep gates** | 0 | 3 | 6 | 3 + learnable α |
| **E↔I coupling** | ✗ | ✗ | ✓ Linear | ✓ Linear + X·Y |
| **Bilinear X·Y** | ✗ | ✗ | ✗ | ✓ |
| **Parallel scan** | O(log T) | O(log T) | O(log T) | O(log T) |
| **Bio analog** | — | Mamba/S6 | Center-surround | hGRU |

---

## GOOM: Generalized Order of Magnitude

All variants use GOOM for numerical stability in deep temporal scans.

### Representation

A complex number `z = r · e^(iθ)` is stored as:
```
GOOM(z) = log(r) + i·θ
```

Where:
- Real part = log-magnitude
- Imaginary part = phase

### Conversion Functions

```python
def to_goom(z):
    """Convert complex tensor to GOOM (log-space)."""
    magnitude = jnp.abs(z) + 1e-12  # Avoid log(0)
    phase = jnp.angle(z)
    return jnp.log(magnitude) + 1j * phase

def from_goom(z_log):
    """Convert GOOM back to complex tensor."""
    log_mag = z_log.real
    phase = z_log.imag
    return jnp.exp(log_mag) * jnp.exp(1j * phase)
```

### Operations in GOOM

| Linear Operation | GOOM Equivalent |
|-----------------|-----------------|
| `a · b` (multiply) | `log(a) + log(b)` (add) |
| `a + b` (add) | `LSE(log(a), log(b))` (log-sum-exp) |
| `A @ v` (matmul) | log-matmul (LSE over inner dim) |

### Why GOOM?

1. **Prevents underflow**: For T > 100, products like `0.9^100 ≈ 10^-5` underflow. In log-space: `100 · log(0.9) ≈ -4.6`
2. **Stable gradients**: Log transforms compress dynamic range
3. **Natural for spectral**: Magnitude-phase is standard for Fourier analysis

---

## Processing Pipeline (All Variants)

```
Input: x ∈ ℝ^(B, T, H, W, C)
         │
    ┌────▼────┐
    │ RoPE    │  (Optional rotary positional encoding)
    └────┬────┘
         │
    ┌────▼────┐
    │ 2D FFT  │  U_hat = FFT2(x) ∈ ℂ^(B, T, C, H, W_freq)
    └────┬────┘
         │
    ┌────▼────────────┐
    │ Compute Gates   │  From spatial context: ctx = mean(x, spatial)
    │ (variant-dep)   │  Gates = sigmoid/softplus(Dense(ctx))
    └────┬────────────┘
         │
    ┌────▼────────────┐
    │ Input Gating    │  U_modulated = U_hat ⊙ B_gate
    └────┬────────────┘
         │
    ┌────▼────────────────┐
    │ GOOM Conversion     │  U_log = to_goom(U_modulated)
    │ (Log-space)         │  K_log = to_goom(K_mat)
    └────┬────────────────┘
         │
    ┌────▼─────────────────────┐
    │ Parallel Associative     │  O(log T) complexity
    │ Scan (jax.lax.assoc_scan)│  Uses variant-specific operator
    └────┬─────────────────────┘
         │
    ┌────▼────────────────┐
    │ GOOM → Complex      │  H_hat = from_goom(State_log)
    └────┬────────────────┘
         │
    ┌────▼────────────┐
    │ Output Gating   │  H_gated = H_hat ⊙ C_gate
    └────┬────────────┘
         │
    ┌────▼────┐
    │ IFFT2   │  y = IFFT2(H_gated) ∈ ℝ^(B, T, H, W, C)
    └────┬────┘
         │
         ▼
Output: y ∈ ℝ^(B, T, H, W, C)
```

---

## Usage Examples

```bash
# 1. Vanilla CSSM (simplest, for baseline)
python main.py --arch vit --cssm standard --dataset pathfinder

# 2. Gated CSSM (Mamba-style, single channel)
python main.py --arch vit --cssm gated --dataset pathfinder

# 3. Gated Opponent CSSM (2x2 E/I dynamics)
python main.py --arch vit --cssm opponent --dataset pathfinder

# 4. hGRU-Style CSSM (bilinear X·Y dynamics)
python main.py --arch vit --cssm hgru --dataset pathfinder
```

---

## References

1. **Mamba/S6**: Gu & Dao (2023) - Input-dependent state space models
2. **hGRU**: Linsley et al. (2018) - "Learning long-range spatial dependencies with horizontal gated recurrent units"
3. **GOOM**: Log-space computation for stable neural dynamics
4. **Associative Scan**: Blelloch (1990) - Parallel prefix algorithms

---

## File Locations

| Component | File |
|-----------|------|
| StandardCSSM | `src/models/cssm.py:180` |
| GatedCSSM | `src/models/cssm.py:297` |
| GatedOpponentCSSM | `src/models/cssm.py:730` |
| HGRUStyleCSSM | `src/models/cssm.py:1873` |
| GOOM primitives | `src/models/goom.py` |
| Scan operators | `src/models/math.py` |
| Log-space operations | `src/models/operations.py` |
