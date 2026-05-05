# Spectral Gating as Implicit Global Spatial Convolution

## Summary

The CSSM (Cepstral State Space Model) operates in the Fourier domain: inputs are FFT'd, a scalar recurrence runs per frequency bin, and outputs are iFFT'd back. We discovered that even with a **1×1 spatial kernel** (no spatial mixing in the transition matrix), the model achieves 83–84% accuracy on PathTracker and Pathfinder — nearly matching the 11×11 kernel variant. Removing the FFT entirely (NoFFT baseline) drops performance to chance (50%).

This document explains why: the FFT transforms per-frequency pointwise gating into **implicit global spatial convolutions**, providing full-image spatial mixing at every timestep without any explicit convolutional kernel.

## The CSSM Recurrence

The GatedCSSM (Spectral Mamba) processes 5D video tensors `(B, T, H, W, C)`. At each timestep, the recurrence is:

$$S_t = A \cdot S_{t-1} + \Delta \cdot B \cdot X_t$$
$$Y_t = C \cdot S_t$$

where all operations happen in the **spectral domain** after FFT.

### Step-by-step

1. **FFT the input** per channel:
$$\hat{X}_t(c, f) = \text{FFT}(x_t(c, \cdot, \cdot))$$

2. **Compute input-dependent gates** from spatially-pooled context:
$$\text{ctx}_t = \frac{1}{HW} \sum_{h,w} x_t(\cdot, h, w) \quad \in \mathbb{R}^C$$
$$\hat{B}(f) = W_B \cdot \text{ctx}_t \quad \in \mathbb{R}^{H \times W_\text{freq}}$$
$$\hat{C}(f) = W_C \cdot \text{ctx}_t \quad \in \mathbb{R}^{H \times W_\text{freq}}$$
$$\hat{\Delta}(f) = \text{softplus}(W_\Delta \cdot \text{ctx}_t) \quad \in \mathbb{R}^{H \times W_\text{freq}}$$

3. **Modulate input** in frequency domain:
$$\hat{U}_t(c, f) = \hat{B}(f) \cdot \hat{X}_t(c, f)$$

4. **Spectral kernel** (from learned spatial kernel, FFT'd):
$$\hat{K}(f) = \text{stable\_magnitude}(\text{FFT}(\text{kernel}))$$

5. **Scalar scan** per frequency bin:
$$\hat{S}_t(c, f) = \hat{K}(f) \cdot e^{-\hat{\Delta}(f)} \cdot \hat{S}_{t-1}(c, f) + \hat{\Delta}(f) \cdot \hat{U}_t(c, f)$$

6. **Output gating** and iFFT:
$$\hat{Y}_t(c, f) = \hat{C}(f) \cdot \hat{S}_t(c, f)$$
$$y_t(c, \cdot, \cdot) = \text{iFFT}(\hat{Y}_t(c, \cdot))$$

## The Convolution Theorem

The convolution theorem states:

$$\text{iFFT}(\hat{F} \cdot \hat{G}) = f \circledast g$$

where $\circledast$ denotes circular convolution and $f = \text{iFFT}(\hat{F})$, $g = \text{iFFT}(\hat{G})$.

**Pointwise multiplication in frequency domain IS circular convolution in spatial domain.**

## What the 1×1 Kernel CSSM Actually Computes

### The kernel (transition matrix)

With `kernel_size=1`, the learned spatial kernel is a single scalar per channel. Padding to H×W and FFT'ing gives a **constant** across all frequency bins:

$$\hat{K}(f) = k_0 \quad \forall f$$

This means the transition `K_hat(f) * S_hat(t-1, f)` is just uniform scalar decay — no spatial mixing. By the convolution theorem, multiplying all frequencies by the same constant is equivalent to multiplying the spatial signal by the same constant. **The kernel contributes zero spatial structure.**

### The B gate (input modulation)

The Dense projection `W_B: R^C → R^{H×W_freq}` outputs a **different scalar for each frequency bin**. When this multiplies the FFT of the input:

$$\hat{U}_t(c, f) = \hat{B}(f) \cdot \hat{X}_t(c, f)$$

By the convolution theorem, this is equivalent to:

$$u_t(c, h, w) = b(h, w) \circledast x_t(c, h, w)$$

where $b = \text{iFFT}(\hat{B})$ is a **global spatial convolution kernel of size H×W**.

The Dense layer doesn't just scale frequencies — it implicitly parameterizes a full-image circular convolution. Every output pixel depends on every input pixel.

### The C gate (output modulation)

Similarly:

$$y_t(c, h, w) = c(h, w) \circledast S_t(c, h, w)$$

where $c = \text{iFFT}(\hat{C})$ is another global spatial convolution kernel.

### The Delta gate (decay modulation)

The per-frequency Delta gate modulates the decay rate at each frequency, which in spatial domain corresponds to frequency-selective temporal smoothing.

## The Full Picture

Even with a 1×1 kernel, the CSSM's forward pass in spatial domain is:

$$u_t = (b_t \circledast x_t) \cdot \delta_t$$
$$S_t = a \cdot S_{t-1} + u_t$$
$$y_t = c_t \circledast S_t$$

where:
- $b_t = \text{iFFT}(\hat{B}_t)$ is a **learned global spatial filter** (H×W) applied to the input
- $c_t = \text{iFFT}(\hat{C}_t)$ is a **learned global spatial filter** (H×W) applied to the state
- $a$ is uniform scalar decay (from the 1×1 kernel)
- $\delta_t$ is per-frequency decay modulation (from the Delta gate)
- Both $b_t$ and $c_t$ are **input-dependent** (conditioned on the spatial mean of the current frame)

**Each timestep applies two global spatial convolutions**, providing full-image spatial mixing without any explicit convolutional kernel.

## Comparison: With FFT vs Without FFT

| Component | With FFT (1×1 CSSM) | Without FFT (NoFFT) |
|-----------|---------------------|---------------------|
| B gate | $b \circledast x$ — **global circular convolution** | $B \odot x$ — pointwise scaling |
| C gate | $c \circledast S$ — **global circular convolution** | $C \odot S$ — pointwise scaling |
| Delta gate | Per-frequency decay modulation | Per-pixel decay modulation |
| Spatial mixing | **Global** — every pixel interacts at every step | **None** — pixels evolve independently |
| Receptive field | Full image (H×W) at every timestep | Single pixel (1×1) at every timestep |

### Why the kernel size barely matters

The spatial kernel $\hat{K}(f)$ controls whether different frequencies decay at different rates in the transition. With `kernel_size=11`, low and high frequencies can have different decay rates, allowing the model to preferentially retain certain spatial scales. With `kernel_size=1`, all frequencies decay equally.

But the B and C gates already provide **per-frequency modulation** of the input and output. The model can compensate for uniform decay by adjusting how much of each frequency it writes into the state (via B) and reads out (via C). The kernel provides marginal additional expressivity on top of the already-global B/C gates.

## Experimental Evidence

### PathTracker (15-distractor, 32×32, 32 frames)

| Model | Spatial Structure | Accuracy |
|-------|------------------|----------|
| GDN (ks=11, spectral) | 11×11 spectral kernel + per-freq gates | **84.6%** |
| CSSM 1×1 (with FFT) | No spatial kernel, per-freq gates only | **83.4%** |
| CSSM ks=11 (with FFT) | 11×11 spectral kernel + per-freq gates | 81.7% |
| NoFFT (no spectral transform) | Per-pixel gates only, no spatial mixing | **50.5%** (chance) |
| Transformer (spatiotemporal) | Self-attention (global) | 50.4% (chance) |

### Pathfinder (128×128, difficulty 14)

| Model | Spatial Structure | Accuracy |
|-------|------------------|----------|
| GDN (ks=11, spectral) | 11×11 spectral kernel + matrix delta rule | **90.8%** |
| CSSM ks=11 (with FFT) | 11×11 spectral kernel + per-freq gates | 87.3% |
| CSSM 1×1 (with FFT) | No spatial kernel, per-freq gates only | **84.2%** (still training) |
| CSSM ks=64 full (with FFT) | Full-image spectral kernel + per-freq gates | 87.2% |

## Implications

1. **The FFT is the key ingredient**, not the spatial kernel. It transforms per-frequency gating into global spatial convolution. Removing the FFT (NoFFT) destroys spatial mixing and the model fails completely.

2. **The spatial kernel provides diminishing returns** on top of FFT-based gating. The B/C/Delta gates already provide per-frequency (= global spatial) modulation. The kernel adds frequency-selective decay, which helps marginally (~3% on Pathfinder) but is not essential.

3. **The model learns implicit spatial filters** through the Dense projections to frequency space. These filters are input-dependent (conditioned on the spatial mean of each frame) and change at every timestep.

4. **This is fundamentally different from standard convolution**: a 1×1 conv in spatial domain is pointwise and provides no spatial mixing. A 1×1 kernel in a spectral RNN still has global spatial mixing through the FFT-domain gates. The "1×1" label is misleading — the model has full-image receptive field at every step.

## Connection to Signal Processing

This analysis connects to classical results in signal processing:

- **Wiener filter**: The optimal linear filter for signal estimation is diagonal in the frequency domain. The CSSM's per-frequency gating learns a data-dependent Wiener-like filter at each timestep.

- **Spectral methods in PDEs**: Solving PDEs spectrally (FFT, operate per-frequency, iFFT) is equivalent to using global basis functions. The CSSM's recurrence is analogous to a spectral method for temporal dynamics with learned, input-dependent coefficients.

- **Global convolution via FFT**: The FFT+pointwise+iFFT pattern is the standard O(N log N) algorithm for computing convolutions. The CSSM uses this pattern not for efficiency, but as an architectural choice that provides global spatial mixing as a side effect of spectral-domain computation.
