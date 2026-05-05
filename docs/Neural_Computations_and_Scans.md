# Canonical Neural Computations and Associative Scans

## Overview

This document maps canonical neural computations from computational neuroscience to what can (and cannot) be implemented using parallel associative scans. The goal is to understand the computational expressivity of scan-based architectures like CSSM.

**Key constraint:** Associative scans require an operator ⊕ where `(a ⊕ b) ⊕ c = a ⊕ (b ⊕ c)`. This fundamentally limits us to **linear** operations on state, though we can have **input-dependent** (but state-independent) modulation of those linear operations.

---

## The 15 Canonical Neural Computations

### 1. Linear Filtering / Convolution
**What it is:** Weighted sum of inputs over space/time. The basis of receptive fields.
```
y = sum_i(w_i * x_i) = conv(x, w)
```

**In scans:** ✅ **FULLY IMPLEMENTABLE**
- FFT converts convolution to multiplication
- In log-space (GOOM): multiplication → addition
- Kernel accumulation over time = growing receptive field

```python
# In CSSM: K_E, K_I are conv kernels
# Transition matrix multiplication IS convolution in spectral domain
K_new = log_matmul(K_j, K_i)  # Convolves kernels together
```

---

### 2. Divisive Normalization
**What it is:** Response divided by pool of activity. Ubiquitous in cortex.
```
y = x / (σ² + sum_i(w_i * x_i²))
```

**In scans:** ❌ **NOT IMPLEMENTABLE**
- Requires division by a function of state
- Division is not associative: (a/b)/c ≠ a/(b/c)
- The denominator depends on ALL inputs, not just accumulated state

**Approximations:**
1. **Chunked normalization:** Scan in chunks, normalize between chunks
2. **Post-scan normalization:** Apply LayerNorm/BatchNorm after scan output
3. **Learned "soft" normalization:** Train gates to approximate normalization effect

**Why it matters:** Normalization provides contrast invariance, prevents saturation, and implements "explaining away" in probabilistic terms.

---

### 3. Thresholding / Rectification
**What it is:** Nonlinear activation function applied per-timestep.
```
y = max(0, x)  or  y = sigmoid(x)  or  y = tanh(x)
```

**In scans:** ❌ **NOT IMPLEMENTABLE (per-timestep)**
- Nonlinearity at each timestep breaks associativity
- `f(f(a ⊕ b) ⊕ c) ≠ f(a ⊕ f(b ⊕ c))`

**Approximations:**
1. **Chunked scan:** Process T/k timesteps, apply nonlinearity, continue
   - O(k * log(T/k)) instead of O(log T)
2. **Input nonlinearity:** Apply nonlinearity to INPUT before scan (doesn't help state)
3. **Output nonlinearity:** Apply after scan (misses per-timestep effects)
4. **Soft thresholding via decay:** Very small decay ≈ forgetting weak signals

**Why it matters:** Thresholding provides noise rejection, creates sparse codes, enables decision-making.

---

### 4. Multiplicative Gating / Gain Modulation
**What it is:** One signal multiplicatively modulates another.
```
y = x * g(context)  # State-independent gating: OK
y = x * g(state)    # State-dependent gating: NOT OK
```

**In scans:** ⚠️ **PARTIALLY IMPLEMENTABLE**

✅ **Input-dependent gating:** Gates computed from INPUT, not state
```python
# CSSM does this:
ctx = x.mean(axis=(2,3))  # Context from INPUT
gate = sigmoid(Dense(ctx))
A_xy = gate * K_I  # Gate modulates transition, not state
```

❌ **State-dependent gating (X*Y):** Breaks associativity
```python
# This is what hGRU does - we CAN'T do this:
y = x * sigmoid(state)  # State appears inside nonlinearity
```

**Approximations:**
1. **Z interaction channel (hgru_bi):** Third state tracks X+Y, feeds back
2. **Input-derived "pseudo-state":** Use lagged input as proxy for state
3. **Chunked with state readout:** Read state between chunks for gating

**Why it matters:** Gain modulation is how attention works, how context modulates processing, and how learning signals affect representations.

---

### 5. Winner-Take-All / Competition
**What it is:** Mutual inhibition leads to sparse, competitive activation.
```
y_i = x_i / sum_j(exp(x_j))  # Softmax
y_i = x_i * (x_i == max(x))   # Hard WTA
```

**In scans:** ❌ **NOT IMPLEMENTABLE**
- Requires comparison across all units at each timestep
- Softmax denominator is a nonlinear function of state
- Hard WTA requires argmax, which is discontinuous

**Approximations:**
1. **Strong mutual inhibition:** Large negative off-diagonal → partial competition
2. **Post-scan softmax:** Apply softmax to final output only
3. **Iterative refinement:** Multiple scan passes with increasing inhibition
4. **Temperature annealing:** Gradually strengthen inhibition over training

**Why it matters:** WTA creates sparse codes, makes decisions, implements attention.

---

### 6. Temporal Integration / Evidence Accumulation
**What it is:** Accumulating information over time toward a decision.
```
x_t = x_{t-1} + input_t  # Perfect integration
x_t = λ*x_{t-1} + input_t  # Leaky integration
```

**In scans:** ✅ **FULLY IMPLEMENTABLE**
- This IS what associative scan does!
- Decay parameter λ controls integration timescale
- Different frequencies can have different decay rates

```python
# CSSM decay gates
decay = 0.1 + 0.89 * sigmoid(Dense(ctx))  # [0.1, 0.99]
# Low decay = fast forgetting, high decay = long integration
```

---

### 7. Adaptation / Habituation
**What it is:** Responses decrease with repeated/sustained stimulation.
```
y_t = x_t - α * running_average(x)
adaptation_state_t = β * adaptation_state_{t-1} + (1-β) * x_t
```

**In scans:** ✅ **IMPLEMENTABLE via opponent channels**
- Y channel can track running average of X
- Subtraction via negative coupling: X_new = X - μ*K_I*Y

```python
# hGRU dynamics naturally implement this:
X_t = decay*X - μ_I*K_I*Y + U  # Y inhibits X
Y_t = μ_E*K_E*X + decay*Y      # Y tracks X
# If Y tracks smoothed X, then -μ_I*K_I*Y ≈ -μ_I*K_I*smooth(X) = adaptation
```

---

### 8. Predictive Coding / Error Computation
**What it is:** Computing difference between prediction and observation.
```
error = observation - prediction
prediction = f(higher_level_state)
```

**In scans:** ⚠️ **PARTIALLY IMPLEMENTABLE**
- Linear prediction: ✅ Y can predict X, error = X - Y
- Nonlinear prediction: ❌ Requires nonlinearity

```python
# Linear predictive coding in 2x2:
# X = sensory input
# Y = prediction (from previous X via K_E)
# Error shows up in X - Y interaction

# Could set up:
X_t = U - K_I*Y  # X gets input minus prediction (Y)
Y_t = K_E*X      # Y predicts next X
```

**Why it matters:** Predictive coding may be the brain's fundamental algorithm for learning and inference.

---

### 9. Oscillations / Rhythm Generation
**What it is:** Self-sustaining periodic activity patterns.
```
For oscillation, need eigenvalues with |λ| ≈ 1 and Im(λ) ≠ 0
Or explicit phase variable: θ_t = θ_{t-1} + ω
```

**In scans:** ✅ **IMPLEMENTABLE**
- Complex eigenvalues in transition matrix → oscillation
- E/I opponent dynamics naturally create oscillations
- GOOM handles complex numbers natively

```python
# 2x2 opponent matrix eigenvalues:
# [α   -β]
# [γ    δ]
# If αδ - βγ < 0 and trace small, get complex eigenvalues → oscillation
```

**Why it matters:** Neural oscillations coordinate information flow, implement attention, and support memory.

---

### 10. Coincidence Detection
**What it is:** Detecting when multiple inputs arrive simultaneously.
```
y = f(x1 * x2)  # Product detects coincidence
y = f(x1 + x2) with threshold  # Sum + threshold approximates AND
```

**In scans:** ❌ **NOT DIRECTLY IMPLEMENTABLE**
- True coincidence detection requires X*Y (multiplicative)
- This is the bilinear problem again

**Approximations:**
1. **Z channel:** Tracks X+Y, large Z indicates both active
2. **Threshold approximation:** If X+Y > θ, both probably active
3. **Learned detection:** Train system to detect via linear combination

**Why it matters:** Coincidence detection underlies spike-timing dependent plasticity, binding, and feature conjunction.

---

### 11. Sequence Detection
**What it is:** Detecting specific ordered patterns of activation.
```
Detect: A → B → C (in that order)
```

**In scans:** ⚠️ **PARTIALLY IMPLEMENTABLE**
- Can learn transition patterns via accumulated state
- But can't do precise "A then B then C" detection

**How it partially works:**
```python
# State accumulates information about past inputs
# Transition matrix encodes expected sequences
# Deviation from expected = high "error" state

# Example: If we expect A→B→C:
# After A: state encodes "saw A"
# After B: state encodes "saw A then B"
# After C: state encodes "complete sequence"
```

**Limitations:**
- Can't do arbitrary sequence lengths
- Can't do conditional sequences (if A then expect B, else expect C)
- Variable-length patterns are hard

---

### 12. Associative Learning / Hebbian Plasticity
**What it is:** "Neurons that fire together wire together"
```
Δw_ij = η * x_i * x_j  # Hebbian
Δw_ij = η * x_i * (x_j - w_ij * x_i)  # Oja's rule (normalized)
```

**In scans:** ❌ **NOT IMPLEMENTABLE WITHIN SCAN**
- Requires X*Y products for weight updates
- Weight updates are inherently nonlinear operations

**Where it CAN happen:**
- In the outer training loop (backprop)
- Scan computes forward pass, gradients compute Hebbian-like updates
- But not online within a single forward pass

**Why it matters:** Hebbian learning is how biological neural networks learn without backprop.

---

### 13. Working Memory / Persistent Activity
**What it is:** Maintaining information over delays without input.
```
x_t = x_{t-1} (perfect maintenance)
x_t = λ*x_{t-1}, λ≈1 (leaky maintenance)
```

**In scans:** ✅ **IMPLEMENTABLE**
- Decay ≈ 1 maintains state
- Different frequencies can have different persistence
- Input gates can "write" to memory, output gates can "read"

```python
# CSSM working memory:
decay = 0.99  # High decay = long memory
# State persists: X_t ≈ 0.99^t * X_0 + accumulated_inputs
```

**Enhancement via attention:** Selective gating of what enters/leaves memory

---

### 14. Feedback Amplification / Recurrent Excitation
**What it is:** Recurrent connections amplify weak but consistent signals.
```
x_t = x_{t-1} + α*W*x_{t-1} + input
If eigenvalue > 1: amplification (unstable without saturation)
```

**In scans:** ✅ **IMPLEMENTABLE (with stability constraints)**
- Off-diagonal excitation amplifies cross-channel
- Must keep spectral radius < 1 for stability
- CSSM's `_stable_spectral_magnitude` enforces this

```python
# In opponent CSSM:
A_yx = μ_E * K_E  # X excites Y
# If K_E is strong, Y amplifies X signal
```

**The catch:** True amplification needs eigenvalue > 1, which is unstable. We approximate via near-unity eigenvalues + many timesteps.

---

### 15. Surround Suppression / Center-Surround
**What it is:** Excitation from center, inhibition from surround.
```
y = center_weight * x_center - surround_weight * x_surround
  = DoG(x)  # Difference of Gaussians
```

**In scans:** ✅ **FULLY IMPLEMENTABLE**
- K_E: Center excitation kernel (narrow Gaussian)
- K_I: Surround inhibition kernel (wide Gaussian)
- Their interaction creates center-surround receptive fields

```python
# CSSM kernels:
k_exc = narrow_gaussian  # Center
k_inh = wide_gaussian    # Surround
# X_t = ... + K_E*signal - K_I*signal
# = (K_E - K_I) * signal = center-surround
```

---

## Summary Table: Neural Computations in CSSM Models

| Computation | Standard | Opponent | hgru (2x2) | hgru_bi (3x3) | Notes |
|-------------|----------|----------|------------|---------------|-------|
| **Linear Filtering** | ✅ | ✅ | ✅ | ✅ | FFT convolution |
| **Divisive Normalization** | ❌ | ❌ | ❌ | ❌ | Post-scan only |
| **Thresholding** | ❌ | ❌ | ❌ | ❌ | Chunked or post-scan |
| **Input-Dependent Gating** | ❌ | ✅ | ✅ | ✅ | Gates from context |
| **State-Dependent Gating (X*Y)** | ❌ | ❌ | ❌ | ~⚠️ | Z approximates |
| **Winner-Take-All** | ❌ | ~⚠️ | ~⚠️ | ~⚠️ | Strong inhibition helps |
| **Temporal Integration** | ✅ | ✅ | ✅ | ✅ | Core scan operation |
| **Adaptation** | ❌ | ✅ | ✅ | ✅ | Y tracks & inhibits |
| **Predictive Coding (linear)** | ❌ | ✅ | ✅ | ✅ | Y predicts X |
| **Oscillations** | ❌ | ✅ | ✅ | ✅ | Complex eigenvalues |
| **Coincidence Detection** | ❌ | ❌ | ❌ | ~⚠️ | Z tracks correlation |
| **Sequence Detection** | ~⚠️ | ~⚠️ | ~⚠️ | ~⚠️ | Partial via state |
| **Hebbian Learning** | ❌ | ❌ | ❌ | ❌ | Outer loop only |
| **Working Memory** | ✅ | ✅ | ✅ | ✅ | High decay |
| **Feedback Amplification** | ❌ | ✅ | ✅ | ✅ | Off-diagonal terms |
| **Center-Surround** | ❌ | ✅ | ✅ | ✅ | K_E - K_I |

Legend:
- ✅ = Fully implementable
- ~⚠️ = Partial/approximate
- ❌ = Not implementable in scan

---

## The Fundamental Barrier: Nonlinearity

The core limitation is that associative scans can only compute **linear recurrences** (with input-dependent coefficients). Any computation requiring:

1. **State × State products** (bilinear)
2. **Division by state** (normalization)
3. **Nonlinear functions of state** (thresholding)
4. **Comparisons between state elements** (WTA, argmax)

...breaks associativity and cannot be parallelized in the standard scan framework.

### Why This Matters for Neuroscience

The brain uses **all** of these computations, often in combination. A single cortical circuit might:
1. Convolve input with receptive field (✅ scan can do)
2. Apply divisive normalization (❌ scan can't do)
3. Threshold with ReLU-like activation (❌ scan can't do)
4. Compete via lateral inhibition (❌ scan can't do well)
5. Modulate by attention (⚠️ partial)

This suggests that **pure scan-based models are fundamentally limited** compared to biological neural circuits.

---

## Strategies for Expanding Expressivity

### 1. Chunked Scans with Inter-Chunk Nonlinearity
```python
for chunk in chunks:
    state = scan(chunk)      # O(log chunk_size) parallel
    state = nonlinearity(state)  # Per-chunk nonlinearity
    state = normalize(state)     # Per-chunk normalization
# Total: O(num_chunks * log(chunk_size))
```

**Trade-off:** Loses full parallelism but gains nonlinearity.

### 2. Multi-Pass Refinement
```python
for iteration in range(K):
    state = scan(input, state)  # Parallel scan
    state = nonlinear_refinement(state)  # Global nonlinearity
# Iterative refinement like diffusion models or equilibrium networks
```

### 3. Hybrid Architectures
```python
# Parallel scan for temporal aggregation
temporal_features = scan(input)

# Attention or MLP for nonlinear processing
output = attention(temporal_features)  # O(T²) but powerful
```

### 4. Learned "Soft" Approximations
Train the linear system to approximate nonlinear computations:
- Z channel approximates X*Y
- Strong inhibition approximates WTA
- Decay scheduling approximates thresholding

---

## Future Directions

### 1. Scan-Compatible Normalization
Can we design a normalization that IS associative?
```python
# Idea: Track running mean and variance as state
# But division still breaks associativity...
```

### 2. Product Spaces
What if state lives in a product/tensor space where "multiplication" is redefined?
```python
# In log-space: a * b = exp(log(a) + log(b))
# Can we find a space where normalization becomes additive?
```

### 3. Approximate Nonlinearity via Polynomials
Taylor expand nonlinearities and track polynomial terms:
```python
# sigmoid(x) ≈ 0.5 + 0.25*x - 0.02*x³ + ...
# Track x, x², x³... as state (Carleman linearization)
```

### 4. Neural Architecture Search for Scan Operators
Learn the optimal scan operator structure:
- How many state channels?
- What coupling patterns?
- What approximations to nonlinearity work best?

---

## Conclusion

Associative scans are powerful for **linear temporal dynamics** but fundamentally limited for **nonlinear neural computations**. The brain's computational repertoire exceeds what's achievable in pure scan form.

However, CSSM and related architectures can still capture many important computations:
- Spatiotemporal filtering (convolution)
- Evidence accumulation (integration)
- Adaptation and habituation
- Oscillatory dynamics
- Center-surround processing
- Working memory
- Feedback amplification

The key insight for building better models: **combine scans with strategic nonlinearities** rather than trying to do everything in a single parallel operation.

The hgru_bi (3x3) model pushes the boundary by adding an interaction channel that can learn to approximate some multiplicative effects, but true divisive normalization and hard competition remain out of reach for pure scan architectures.

---

## References

1. Carandini & Heeger (2012). "Normalization as a canonical neural computation." Nature Reviews Neuroscience.
2. Douglas & Martin (2004). "Neuronal circuits of the neocortex." Annual Review of Neuroscience.
3. Linsley et al. (2018). "Learning long-range spatial dependencies with horizontal gated recurrent units." NeurIPS.
4. Blelloch (1990). "Prefix sums and their applications." Synthesis of Parallel Algorithms.
5. Gu et al. (2022). "Efficiently modeling long sequences with structured state spaces." ICLR.
