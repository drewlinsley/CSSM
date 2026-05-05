# CSSM Explained: From Basics to Implementation

A complete guide written for someone with high school math background.

---

## Table of Contents
1. [FFT Basics: What is a Fourier Transform?](#1-fft-basics)
2. [Convolution: Image Space vs Spectral Domain](#2-convolution)
3. [Depthwise vs Regular Convolution with FFT](#3-depthwise-vs-regular)
4. [The Opponent Circuit: Step-by-Step](#4-opponent-circuit)
5. [GOOM: Handling Negative Numbers in Log-Space](#5-goom)
6. [LMME: Channel Mixing in Blocks](#6-lmme)
7. [Why O(C³) for Full Channel Mixing](#7-complexity)

---

## 1. FFT Basics: What is a Fourier Transform? <a name="1-fft-basics"></a>

### The Core Idea

**Any signal can be broken down into a sum of sine waves.**

Think of a musical chord: when you play C-E-G together, you hear one sound, but it's actually three frequencies combined. FFT separates them back out.

```
Original Signal          FFT           Frequency Components

   /\    /\                           Low freq:  ~~~~
  /  \  /  \    ──────────►           Mid freq:  ∿∿∿∿
 /    \/    \                         High freq: ⋀⋀⋀⋀
```

### For Images (2D FFT)

Images have patterns too! FFT finds them:

```
Image Domain                    Frequency Domain
┌─────────────┐                ┌─────────────┐
│ ░░▓▓░░▓▓░░ │                │      ●      │  ← Low freq (center) = smooth areas
│ ░░▓▓░░▓▓░░ │    rfft2()     │    ● ● ●    │  ← Mid freq = gradual changes
│ ░░▓▓░░▓▓░░ │  ──────────►   │  ●   ●   ●  │  ← High freq (edges) = sharp edges
│ ░░▓▓░░▓▓░░ │                │    ● ● ●    │
└─────────────┘                └─────────────┘
  (H × W real)                  (H × W_freq complex)
                                W_freq = W//2 + 1
```

### Why W_freq = W//2 + 1?

For **real** images (not complex), the FFT has a symmetry: negative frequencies mirror positive ones. So `rfft2` only returns half (+1 for the center):

```
Full FFT (fft2):     Real FFT (rfft2):
W frequencies        W//2 + 1 frequencies

[-4 -3 -2 -1 0 1 2 3 4]  →  [0 1 2 3 4]
     redundant              kept (saves memory!)
```

### Complex Numbers in FFT

Each frequency bin is a **complex number** with two parts:

```
Complex number: z = a + bi

         Imaginary
              │
              │    ● z = a + bi
              │   /│
              │  / │
     magnitude│ /  │ b (imaginary part)
              │/θ  │
    ──────────┼────┴───── Real
              │    a

magnitude = √(a² + b²)   ← "how strong" this frequency is
phase = arctan(b/a)      ← "where" in the cycle it starts
```

### Code Example

```python
import jax.numpy as jnp

# Input image: (H, W) = (8, 8)
image = jnp.ones((8, 8))

# 2D FFT for real input
freq = jnp.fft.rfft2(image)
# Output: (H, W_freq) = (8, 5) complex numbers
#         W_freq = 8//2 + 1 = 5

print(f"Input shape:  {image.shape}")      # (8, 8)
print(f"Output shape: {freq.shape}")       # (8, 5)
print(f"Output dtype: {freq.dtype}")       # complex64

# Inverse FFT goes back to image
image_back = jnp.fft.irfft2(freq, s=(8, 8))
# Output: (8, 8) real - same as input!
```

---

## 2. Convolution: Image Space vs Spectral Domain <a name="2-convolution"></a>

### Image-Space Convolution (The Slow Way)

Slide a kernel over the image, multiply-and-sum at each position:

```
Image (5×5)              Kernel (3×3)           Output (5×5)
┌─────────────────┐      ┌───────────┐         ┌─────────────────┐
│ 1  2  3  4  5   │      │ 1  0 -1   │         │ ?  ?  ?  ?  ?   │
│ 6  7  8  9  10  │  ⊛   │ 2  0 -2   │    =    │ ?  ?  ?  ?  ?   │
│ 11 12 13 14 15  │      │ 1  0 -1   │         │ ?  ?  ?  ?  ?   │
│ 16 17 18 19 20  │      └───────────┘         │ ?  ?  ?  ?  ?   │
│ 21 22 23 24 25  │                            │ ?  ?  ?  ?  ?   │
└─────────────────┘                            └─────────────────┘

For each output pixel:
  - Place kernel centered on that pixel
  - Multiply kernel × image patch element-wise
  - Sum all products

Complexity: O(H × W × K × K) where K = kernel size
For 224×224 image with 11×11 kernel: 224 × 224 × 11 × 11 = 6 million operations!
```

### Spectral Convolution (The Fast Way)

**Key insight**: Convolution in image space = Multiplication in frequency space!

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                   │
│   Image ⊛ Kernel  =  IFFT( FFT(Image) × FFT(Kernel) )            │
│                                                                   │
│   "Slide and sum"    "Just multiply element-wise!"               │
│   O(H × W × K²)       O(H × W × log(H×W))                        │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

Step by step:

```
Step 1: FFT both image and kernel
────────────────────────────────

Image (H×W)          Kernel (K×K)
    │                     │
    │ pad to (H×W)        │ pad to (H×W)  ← zero-pad kernel to image size
    │                     │
    ▼                     ▼
Image (H×W)          Kernel (H×W)
    │                     │
    │ rfft2               │ rfft2
    ▼                     ▼
I_hat (H×W_freq)     K_hat (H×W_freq)    ← both complex


Step 2: Multiply element-wise (not matrix multiply!)
───────────────────────────────────────────────────

I_hat (H×W_freq)  ×  K_hat (H×W_freq)  =  O_hat (H×W_freq)
     ●                    ●                     ●
     ●          ×         ●          =          ●
     ●                    ●                     ●

Each frequency bin multiplied independently:
  O_hat[i,j] = I_hat[i,j] × K_hat[i,j]


Step 3: Inverse FFT to get result
─────────────────────────────────

O_hat (H×W_freq)
    │
    │ irfft2
    ▼
Output (H×W)  ← same as image ⊛ kernel!
```

### Code Example

```python
def fft_conv2d(image, kernel):
    """
    Convolution via FFT.

    Args:
        image: (H, W) input image
        kernel: (K, K) convolution kernel

    Returns:
        (H, W) convolved output
    """
    H, W = image.shape
    K = kernel.shape[0]

    # Pad kernel to image size (center it)
    pad_h = (H - K) // 2
    pad_w = (W - K) // 2
    kernel_padded = jnp.pad(kernel, ((pad_h, H-K-pad_h), (pad_w, W-K-pad_w)))
    # kernel_padded: (H, W)

    # FFT both
    I_hat = jnp.fft.rfft2(image)          # (H, W_freq) complex
    K_hat = jnp.fft.rfft2(kernel_padded)  # (H, W_freq) complex

    # Multiply (element-wise, NOT matrix multiply)
    O_hat = I_hat * K_hat                 # (H, W_freq) complex

    # Inverse FFT
    output = jnp.fft.irfft2(O_hat, s=(H, W))  # (H, W) real

    return output
```

### Tensor Sizes Summary

```
                    Image Space              Frequency Space
                    ───────────              ───────────────
Input image:        (H, W) real        →    (H, W_freq) complex
Kernel:             (K, K) real        →    (H, W_freq) complex (after padding)
After multiply:          N/A           →    (H, W_freq) complex
Output:             (H, W) real        ←    (H, W_freq) complex

where W_freq = W // 2 + 1
```

---

## 3. Depthwise vs Regular Convolution with FFT <a name="3-depthwise-vs-regular"></a>

### What's the Difference?

```
REGULAR CONV: Each output channel mixes ALL input channels
─────────────────────────────────────────────────────────

Input: (C_in, H, W)     Kernel: (C_out, C_in, K, K)     Output: (C_out, H, W)

For output channel c_out:
    out[c_out] = Σ (input[c_in] ⊛ kernel[c_out, c_in])
                c_in

    "Sum convolutions across ALL input channels"

Example: C_in=3 (RGB), C_out=64
    out[0] = in[R] ⊛ k[0,R] + in[G] ⊛ k[0,G] + in[B] ⊛ k[0,B]
    out[1] = in[R] ⊛ k[1,R] + in[G] ⊛ k[1,G] + in[B] ⊛ k[1,B]
    ...

Total kernel params: C_out × C_in × K × K = 64 × 3 × 3 × 3 = 1,728


DEPTHWISE CONV: Each channel processed INDEPENDENTLY
────────────────────────────────────────────────────

Input: (C, H, W)        Kernel: (C, K, K)              Output: (C, H, W)

For channel c:
    out[c] = input[c] ⊛ kernel[c]

    "Each channel has its OWN kernel, no mixing"

Example: C=64
    out[0] = in[0] ⊛ k[0]
    out[1] = in[1] ⊛ k[1]
    ...

Total kernel params: C × K × K = 64 × 3 × 3 = 576 (much fewer!)
```

### Visual Comparison

```
REGULAR CONV (mixes channels):

Input           Kernels              Output
┌───┐          ┌───┬───┬───┐        ┌───┐
│ R │          │k00│k01│k02│───────►│ 0 │  = R⊛k00 + G⊛k01 + B⊛k02
├───┤          ├───┼───┼───┤        ├───┤
│ G │    ⊛     │k10│k11│k12│───────►│ 1 │  = R⊛k10 + G⊛k11 + B⊛k12
├───┤          └───┴───┴───┘        └───┘
│ B │
└───┘
(3,H,W)        (2,3,K,K)            (2,H,W)


DEPTHWISE CONV (independent channels):

Input           Kernels              Output
┌───┐          ┌───┐                ┌───┐
│ R │    ⊛     │kR │ ──────────────►│ R'│  = R ⊛ kR
├───┤          ├───┤                ├───┤
│ G │    ⊛     │kG │ ──────────────►│ G'│  = G ⊛ kG
├───┤          ├───┤                ├───┤
│ B │    ⊛     │kB │ ──────────────►│ B'│  = B ⊛ kB
└───┘          └───┘                └───┘
(3,H,W)        (3,K,K)              (3,H,W)
```

### CSSM Uses Depthwise (in Frequency Domain)

```python
# CSSM Spectral Convolution (depthwise)
# Input:  (B, T, C, H, W) real
# Output: (B, T, C, H, W) real

def cssm_spectral_conv(x, kernel):
    """
    x:      (B, T, C, H, W) input features
    kernel: (C, K, K) one kernel PER channel (depthwise)
    """
    B, T, C, H, W = x.shape
    W_freq = W // 2 + 1

    # FFT input (last 2 dims = spatial)
    X_hat = jnp.fft.rfft2(x, axes=(-2, -1))
    # X_hat: (B, T, C, H, W_freq) complex

    # FFT kernels (pad to H×W first)
    K_hat = jnp.fft.rfft2(pad_kernels(kernel, H, W), axes=(-2, -1))
    # K_hat: (C, H, W_freq) complex

    # Multiply - broadcasts over B, T
    # Each channel c: X_hat[..., c, :, :] × K_hat[c, :, :]
    Y_hat = X_hat * K_hat[None, None, :, :, :]
    # Y_hat: (B, T, C, H, W_freq) complex

    # Inverse FFT
    y = jnp.fft.irfft2(Y_hat, s=(H, W), axes=(-2, -1))
    # y: (B, T, C, H, W) real

    return y
```

### How to Simulate Regular Conv with FFT?

To mix channels, you'd need a **separate sum** over input channels:

```python
def fft_regular_conv(x, kernels):
    """
    x:       (C_in, H, W) input
    kernels: (C_out, C_in, K, K) one kernel per (out, in) pair
    """
    C_in, H, W = x.shape
    C_out = kernels.shape[0]

    # FFT input
    X_hat = jnp.fft.rfft2(x, axes=(-2, -1))  # (C_in, H, W_freq)

    # FFT all kernels
    K_hat = jnp.fft.rfft2(pad_kernels(kernels, H, W), axes=(-2, -1))
    # K_hat: (C_out, C_in, H, W_freq)

    # For each output channel, sum over input channels
    Y_hat = jnp.zeros((C_out, H, W // 2 + 1), dtype=jnp.complex64)
    for c_out in range(C_out):
        for c_in in range(C_in):
            Y_hat = Y_hat.at[c_out].add(X_hat[c_in] * K_hat[c_out, c_in])

    # Or vectorized:
    # Y_hat = jnp.einsum('ihw,oihw->ohw', X_hat, K_hat)
    # Y_hat: (C_out, H, W_freq)

    return jnp.fft.irfft2(Y_hat, s=(H, W), axes=(-2, -1))
```

**Key insight**: Regular conv needs O(C_out × C_in) FFT multiplications. Depthwise needs only O(C).

---

## 4. The Opponent Circuit: Step-by-Step <a name="4-opponent-circuit"></a>

### What Is It?

The opponent circuit models **excitation and inhibition** - like how your visual system works:

```
         Excitation (X)              Inhibition (Y)
         "Enhance signal"            "Suppress noise"

              ┌─────┐                    ┌─────┐
    Input ───►│  X  │◄───── inhibits ────│  Y  │
              └──┬──┘                    └──┬──┘
                 │                          ▲
                 │                          │
                 └────────── excites ───────┘

X gets SUPPRESSED by Y (center-surround)
Y gets EXCITED by X (builds up inhibition over time)
```

### The Math (2×2 System)

At each timestep t, we update both X and Y:

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌     ┐   ┌              ┐ ┌       ┐   ┌   ┐              │
│  │ X_t │   │  α    -K_I·μ │ │ X_t-1 │   │ U │              │
│  │     │ = │              │ │       │ + │   │              │
│  │ Y_t │   │ K_E·γ    δ   │ │ Y_t-1 │   │ 0 │              │
│  └     ┘   └              ┘ └       ┘   └   ┘              │
│                                                             │
│  state      transition      prev        input               │
│  (2×1)       matrix        state       (2×1)               │
│              (2×2)         (2×1)                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Where:
  X_t = excitation at time t
  Y_t = inhibition at time t
  U   = input (only feeds X, not Y)

  α   = X self-decay (0 to 1, how much X remembers itself)
  δ   = Y self-decay (0 to 1, how much Y remembers itself)
  μ   = Y→X inhibition strength (Y suppresses X)
  γ   = X→Y excitation strength (X builds up Y)

  K_E = excitatory spatial kernel (spreads X's influence)
  K_I = inhibitory spatial kernel (spreads Y's influence)
```

### Written Out Explicitly

```
X_t = α · X_{t-1}   -   K_I ⊛ (μ · Y_{t-1})   +   U
      ─────────────     ───────────────────       ───
      X remembers       Y inhibits X via          new input
      itself            surround kernel

Y_t = K_E ⊛ (γ · X_{t-1})   +   δ · Y_{t-1}
      ───────────────────       ─────────────
      X excites Y via           Y remembers
      center kernel             itself
```

### Why "Opponent"?

X and Y **oppose** each other:
- X tries to respond to input
- Y builds up and suppresses X
- Creates **center-surround** receptive fields (like V1 neurons!)

```
Time 0:  Input arrives
         X = U (responds to input)
         Y = 0 (no inhibition yet)

Time 1:  Inhibition builds
         X = α·X_0 - K_I·μ·Y_0 + U = less response (Y starts inhibiting)
         Y = K_E·γ·X_0 (Y grows based on X)

Time 2+: Equilibrium
         X settles to value where excitation = inhibition
         Strong sustained inputs win, weak/noisy inputs get suppressed
```

### Sequential Code (No Log-Space, Easy to Understand)

```python
def opponent_circuit_sequential(x, K_E, K_I, alpha, delta, mu, gamma, num_steps):
    """
    Sequential opponent circuit - easy to understand version.

    Args:
        x: (B, H, W, C) input image/features
        K_E: (C, K, K) excitatory kernels (one per channel)
        K_I: (C, K, K) inhibitory kernels (one per channel)
        alpha: (C,) X decay per channel
        delta: (C,) Y decay per channel
        mu: (C,) Y→X inhibition strength
        gamma: (C,) X→Y excitation strength
        num_steps: int, number of recurrent steps (T)

    Returns:
        X: (B, T, H, W, C) excitation states over time
        Y: (B, T, H, W, C) inhibition states over time
    """
    B, H, W, C = x.shape

    # Initialize states to zero
    X_prev = jnp.zeros((B, H, W, C))  # Excitation
    Y_prev = jnp.zeros((B, H, W, C))  # Inhibition

    # Input U is just x, broadcast to each timestep
    U = x  # (B, H, W, C)

    X_all = []
    Y_all = []

    for t in range(num_steps):
        # ═══════════════════════════════════════════════════════════
        # UPDATE X (excitation)
        # X_t = α · X_{t-1} - K_I ⊛ (μ · Y_{t-1}) + U
        # ═══════════════════════════════════════════════════════════

        # Term 1: X remembers itself (element-wise multiply)
        X_memory = alpha[None, None, None, :] * X_prev
        # Shape: (1,1,1,C) * (B,H,W,C) → (B,H,W,C)

        # Term 2: Y inhibits X via spatial kernel K_I
        Y_scaled = mu[None, None, None, :] * Y_prev  # (B,H,W,C)

        # Depthwise convolution in frequency domain
        Y_inhibition = spectral_conv_depthwise(Y_scaled, K_I)  # (B,H,W,C)

        # Term 3: New input
        # U is just x

        # Combine: X = memory - inhibition + input
        X_new = X_memory - Y_inhibition + U
        #        (B,H,W,C)  (B,H,W,C)    (B,H,W,C)

        # ═══════════════════════════════════════════════════════════
        # UPDATE Y (inhibition)
        # Y_t = K_E ⊛ (γ · X_{t-1}) + δ · Y_{t-1}
        # ═══════════════════════════════════════════════════════════

        # Term 1: X excites Y via spatial kernel K_E
        X_scaled = gamma[None, None, None, :] * X_prev  # (B,H,W,C)
        X_excitation = spectral_conv_depthwise(X_scaled, K_E)  # (B,H,W,C)

        # Term 2: Y remembers itself
        Y_memory = delta[None, None, None, :] * Y_prev  # (B,H,W,C)

        # Combine: Y = excitation + memory
        Y_new = X_excitation + Y_memory

        # ═══════════════════════════════════════════════════════════
        # Store and update
        # ═══════════════════════════════════════════════════════════
        X_all.append(X_new)
        Y_all.append(Y_new)
        X_prev = X_new
        Y_prev = Y_new

    # Stack over time dimension
    X_out = jnp.stack(X_all, axis=1)  # (B, T, H, W, C)
    Y_out = jnp.stack(Y_all, axis=1)  # (B, T, H, W, C)

    return X_out, Y_out


def spectral_conv_depthwise(x, kernel):
    """
    Depthwise convolution via FFT.

    Args:
        x: (B, H, W, C) input
        kernel: (C, K, K) kernels, one per channel

    Returns:
        (B, H, W, C) convolved output
    """
    B, H, W, C = x.shape
    K = kernel.shape[1]
    W_freq = W // 2 + 1

    # Transpose to (B, C, H, W) for easier FFT
    x_t = x.transpose(0, 3, 1, 2)  # (B, C, H, W)

    # FFT input
    X_hat = jnp.fft.rfft2(x_t, axes=(-2, -1))  # (B, C, H, W_freq) complex

    # Pad and FFT kernels
    kernel_padded = pad_to_size(kernel, H, W)  # (C, H, W)
    K_hat = jnp.fft.rfft2(kernel_padded, axes=(-2, -1))  # (C, H, W_freq) complex

    # Multiply element-wise (depthwise - no cross-channel!)
    Y_hat = X_hat * K_hat[None, :, :, :]  # (B, C, H, W_freq)
    #       (B,C,H,W_f) * (1,C,H,W_f) broadcast

    # Inverse FFT
    y_t = jnp.fft.irfft2(Y_hat, s=(H, W), axes=(-2, -1))  # (B, C, H, W)

    # Transpose back
    return y_t.transpose(0, 2, 3, 1)  # (B, H, W, C)
```

### Tensor Size Tracking

```
Input: x (B, H, W, C)
       │
       ▼
Initialize:
  X_prev: (B, H, W, C) zeros
  Y_prev: (B, H, W, C) zeros
       │
       ▼
For each timestep t = 0, 1, ..., T-1:
  │
  ├─► X_memory = alpha * X_prev
  │   alpha: (C,) broadcast to (1,1,1,C)
  │   Result: (B, H, W, C)
  │
  ├─► Y_scaled = mu * Y_prev
  │   mu: (C,) broadcast to (1,1,1,C)
  │   Result: (B, H, W, C)
  │
  ├─► Y_inhibition = spectral_conv(Y_scaled, K_I)
  │   │
  │   ├─► FFT: (B,H,W,C) → (B,C,H,W_freq) complex
  │   ├─► K_I FFT: (C,K,K) → (C,H,W_freq) complex
  │   ├─► Multiply: (B,C,H,W_freq) * (C,H,W_freq) → (B,C,H,W_freq)
  │   └─► IFFT: (B,C,H,W_freq) → (B,H,W,C)
  │
  ├─► X_new = X_memory - Y_inhibition + U
  │   All (B, H, W, C), element-wise ops
  │
  ├─► (similar for Y_new)
  │
  └─► Store X_new, Y_new
       │
       ▼
Output:
  X: (B, T, H, W, C) - stack of all X_new
  Y: (B, T, H, W, C) - stack of all Y_new
```

### 2×2 vs 3×3: What's the Difference?

**2×2 Opponent (GatedOpponentCSSM)** - what we actually use:
```
State: [X, Y]  (excitation, inhibition)

┌     ┐   ┌              ┐ ┌       ┐   ┌   ┐
│ X_t │   │  α    -K_I·μ │ │ X_t-1 │   │ U │
│     │ = │              │ │       │ + │   │
│ Y_t │   │ K_E·γ    δ   │ │ Y_t-1 │   │ 0 │
└     ┘   └              ┘ └       ┘   └   ┘
```

**3×3 Bilinear (BilinearOpponentCSSM)** - experimental, adds X² inhibition:
```
State: [X, Y, Z] where Z = X_{t-1} (delayed copy of X)

┌     ┐   ┌                  ┐ ┌       ┐   ┌   ┐
│ X_t │   │  α    -K_I·μ   0 │ │ X_t-1 │   │ U │
│ Y_t │ = │ K_E·γ    δ     0 │ │ Y_t-1 │ + │ 0 │  + bilinear term
│ Z_t │   │  1       0     0 │ │ Z_t-1 │   │ 0 │
└     ┘   └                  ┘ └       ┘   └   ┘

Bilinear term: X_t gets extra inhibition proportional to X·Z = X·X_{t-1} ≈ X²
This mimics hGRU's α·X² divisive normalization.
```

---

## 5. GOOM: Handling Negative Numbers in Log-Space <a name="5-goom"></a>

### Why Log-Space?

When you multiply many small numbers, you get **underflow** (too small for float32):

```
Problem: 0.9^100 = 2.66 × 10⁻⁵ → might become 0.0 in float32!

Solution: Work in LOG space
  log(0.9^100) = 100 × log(0.9) = -4.6

  No underflow! Then convert back: exp(-4.6) = 0.01
```

### The GOOM Representation

**GOOM** = Generalized Order Of Magnitude

```
Normal number:  z = magnitude × sign
                z = |z| × (±1)

GOOM number:    z = (log_magnitude, sign_bit)
                  = (log|z|, s)    where s ∈ {0, 1}

                s=0 means positive
                s=1 means negative
```

### Converting To and From GOOM

```
TO GOOM (encoding):
────────────────────

z = 5.0  (positive)
  → log_mag = log(5.0) = 1.609
  → sign = 0 (positive)
  → GOOM: (1.609, 0)

z = -3.0  (negative)
  → log_mag = log(|-3.0|) = log(3.0) = 1.099
  → sign = 1 (negative)
  → GOOM: (1.099, 1)

z = 0.001  (small positive)
  → log_mag = log(0.001) = -6.9
  → sign = 0
  → GOOM: (-6.9, 0)  ← No underflow!


FROM GOOM (decoding):
─────────────────────

GOOM (1.609, 0)
  → magnitude = exp(1.609) = 5.0
  → sign = +1 (since bit is 0)
  → z = +5.0

GOOM (1.099, 1)
  → magnitude = exp(1.099) = 3.0
  → sign = -1 (since bit is 1)
  → z = -3.0
```

### Complex Numbers in GOOM

Complex numbers have magnitude AND phase:

```
Complex: z = |z| × e^(i·θ)

where:
  |z| = magnitude = √(real² + imag²)
  θ   = phase = atan2(imag, real)

GOOM for complex:
  log_mag = log(|z|)
  phase = θ    (keeps track of angle, including sign via π)
```

### How Subtraction Becomes Addition (The Magic!)

In normal space, subtraction is hard in log-space. But we use a **phase trick**:

```
SUBTRACTION VIA PHASE:
──────────────────────

a - b = a + (-b)
      = a + (b × e^(iπ))     ← multiply by e^(iπ) = -1
      = a + (b with phase shifted by π)

In GOOM:
  a:     (log|a|, θ_a)
  b:     (log|b|, θ_b)
  -b:    (log|b|, θ_b + π)   ← just add π to phase!

Then a + (-b) becomes log-space addition!
```

### Log-Space Addition: LogSumExp

Adding in log-space uses the "log-sum-exp" trick:

```
We want: log(e^a + e^b)

Naive:   log(exp(a) + exp(b))  ← can overflow!

Stable:  max(a,b) + log(1 + exp(-|a-b|))

         = max(a,b) + softplus(-|a-b|)

This is numerically stable because:
  - We first factor out the larger term
  - The exp() is always of a negative number (no overflow)
```

### Code for GOOM Operations

```python
def to_goom(z):
    """
    Convert complex number to GOOM representation.

    Args:
        z: complex array

    Returns:
        (log_magnitude, phase) tuple
    """
    magnitude = jnp.abs(z)              # |z|
    phase = jnp.angle(z)                # atan2(imag, real)
    log_mag = jnp.log(magnitude + 1e-10)  # log|z|, epsilon for stability
    return log_mag, phase


def from_goom(log_mag, phase):
    """
    Convert GOOM back to complex number.

    Args:
        log_mag: log of magnitude
        phase: angle in radians

    Returns:
        complex array
    """
    magnitude = jnp.exp(log_mag)        # |z| = exp(log|z|)
    z = magnitude * jnp.exp(1j * phase)  # z = |z| × e^(iθ)
    return z


def log_add_exp_complex(log_a, phase_a, log_b, phase_b):
    """
    Compute log(exp(a) + exp(b)) in GOOM.

    This handles the phase (sign) properly!
    """
    # Convert to complex, add, convert back
    a = from_goom(log_a, phase_a)
    b = from_goom(log_b, phase_b)
    c = a + b
    return to_goom(c)


def log_multiply_complex(log_a, phase_a, log_b, phase_b):
    """
    Compute log(a × b) in GOOM.

    Multiplication is EASY in log-space!
    """
    # |a × b| = |a| × |b|
    # log|a × b| = log|a| + log|b|
    log_c = log_a + log_b

    # angle(a × b) = angle(a) + angle(b)
    phase_c = phase_a + phase_b

    return log_c, phase_c
```

### The Stop-Gradient Trick

When training, we need gradients. But `jnp.angle()` has discontinuities. Fix:

```python
def safe_angle_with_stopgrad(z):
    """
    Compute angle with smooth gradients.

    The trick: use atan2 for forward pass, but compute gradients
    through a different (smooth) path.
    """
    # Forward: use standard angle
    angle_forward = jnp.angle(z)

    # Backward: use smooth approximation
    # angle ≈ imag(z) / |z| for small angles (first-order Taylor)
    angle_smooth = jnp.imag(z) / (jnp.abs(z) + 1e-10)

    # stop_gradient makes forward use angle_forward,
    # but gradients flow through angle_smooth
    return jax.lax.stop_gradient(angle_forward - angle_smooth) + angle_smooth
```

### Visual Summary: GOOM Pipeline

```
Input (complex)              GOOM Domain                    Output (complex)

z = 3 + 4i                   log_mag = log(5) = 1.61       z' = 5 × e^(0.93i)
|z| = 5                      phase = atan2(4,3) = 0.93        = 3 + 4i
θ = 0.93 rad
       │                            │                             ▲
       │ to_goom()                  │ log-space ops               │ from_goom()
       ▼                            ▼                             │
┌──────────────┐            ┌──────────────┐              ┌──────────────┐
│ log_mag=1.61 │            │ Multiply:    │              │ log_mag=1.61 │
│ phase=0.93   │───────────►│ log_a+log_b  │─────────────►│ phase=0.93   │
└──────────────┘            │ Add phases   │              └──────────────┘
                            │              │
                            │ Add:         │
                            │ LogSumExp    │
                            │ (complex)    │
                            └──────────────┘
```

---

## 6. LMME: Channel Mixing in Blocks <a name="6-lmme"></a>

### The Problem

Depthwise processing = no channel mixing. But we WANT channels to talk!

```
Depthwise (no mixing):        We want (with mixing):

Channel 0: ───────────►       Channel 0: ──┬──┬──►
Channel 1: ───────────►       Channel 1: ──┼──┼──►  (channels interact)
Channel 2: ───────────►       Channel 2: ──┴──┴──►
```

### Full Channel Mixing = O(C²) Problem

If every channel affects every other channel:

```
C channels, each affects all C others = C × C = C² parameters

For C = 384 (ViT-Small): 384² = 147,456 parameters per frequency!
And we have H × W_freq frequencies...

Total: 147,456 × H × W_freq = BILLIONS of parameters!
```

### LMME Solution: Block-Diagonal Mixing

Split channels into blocks, only mix WITHIN blocks:

```
FULL MIXING (C×C):                BLOCK MIXING (blocks of size B):

┌─────────────────┐               ┌───┬───┬───┬───┐
│ ● ● ● ● ● ● ● ● │               │ ● │   │   │   │
│ ● ● ● ● ● ● ● ● │               │ ● │   │   │   │
│ ● ● ● ● ● ● ● ● │               ├───┼───┼───┼───┤
│ ● ● ● ● ● ● ● ● │               │   │ ● │   │   │
│ ● ● ● ● ● ● ● ● │               │   │ ● │   │   │
│ ● ● ● ● ● ● ● ● │               ├───┼───┼───┼───┤
│ ● ● ● ● ● ● ● ● │               │   │   │ ● │   │
│ ● ● ● ● ● ● ● ● │               │   │   │ ● │   │
└─────────────────┘               ├───┼───┼───┼───┤
                                  │   │   │   │ ● │
C² = 64 params                    │   │   │   │ ● │
                                  └───┴───┴───┴───┘

                                  4 blocks × (C/4)² = C²/4 params
                                  = 16 params (4× fewer!)
```

### LMME in Code

```python
def lmme_channel_mixing(x, block_size=32):
    """
    Linear Matrix Mixing in Embedding space.

    Args:
        x: (B, T, H, W, C) input, C must be divisible by block_size
        block_size: channels per block

    Returns:
        (B, T, H, W, C) with channel mixing within blocks
    """
    B, T, H, W, C = x.shape
    num_blocks = C // block_size

    # Reshape to expose blocks
    x_blocks = x.reshape(B, T, H, W, num_blocks, block_size)
    # Shape: (B, T, H, W, num_blocks, block_size)
    #                      └────────────────────┘
    #                      last 2 dims = block structure

    # Mixing matrix: one (block_size × block_size) per block
    # But shared across spatial positions!
    W_mix = param('W_mix', shape=(num_blocks, block_size, block_size))
    # Shape: (num_blocks, block_size, block_size)
    #         └─ one matrix per block

    # Apply mixing within each block
    # For block b: output[..., b, :] = input[..., b, :] @ W_mix[b]
    x_mixed = jnp.einsum('bthwnc,ncd->bthwnd', x_blocks, W_mix)
    # 'bthwnc' = (B, T, H, W, num_blocks, block_size)
    # 'ncd'    = (num_blocks, block_size, block_size)
    # 'bthwnd' = (B, T, H, W, num_blocks, block_size)
    #
    # The 'n' index (block) matches, 'c' contracts, 'd' is output

    # Reshape back
    x_out = x_mixed.reshape(B, T, H, W, C)

    return x_out
```

### Tensor Size Tracking for LMME

```
Input: x (B, T, H, W, C)
       │
       │ Example: (2, 8, 56, 56, 384)
       │          B=2, T=8, H=W=56, C=384
       │          block_size = 32
       │          num_blocks = 384 // 32 = 12
       ▼
Reshape to blocks:
       │
       ▼
x_blocks (B, T, H, W, num_blocks, block_size)
         (2, 8, 56, 56, 12, 32)
       │
       │ W_mix: (num_blocks, block_size, block_size)
       │        (12, 32, 32)
       │        = 12 separate 32×32 matrices
       │        = 12 × 32 × 32 = 12,288 params
       │
       │ Compare to full: C×C = 384×384 = 147,456 params
       │ Savings: 12× fewer!
       │
       ▼
einsum: 'bthwnc,ncd->bthwnd'
       │
       │ For each of the 12 blocks:
       │   input:  (2, 8, 56, 56, 32)  ← the 'c' dimension
       │   weight: (32, 32)            ← 'cd'
       │   output: (2, 8, 56, 56, 32)  ← the 'd' dimension
       │
       ▼
x_mixed (B, T, H, W, num_blocks, block_size)
        (2, 8, 56, 56, 12, 32)
       │
       │ Reshape back
       ▼
Output: (B, T, H, W, C)
        (2, 8, 56, 56, 384)
```

### Where Aggregation Occurs

```
WITHIN each block:
──────────────────

Block b contains channels [b*32, b*32+1, ..., b*32+31]

Before LMME:
  Channel b*32:    independent
  Channel b*32+1:  independent
  ...
  Channel b*32+31: independent

After LMME:
  Channel b*32:    weighted sum of ALL 32 channels in block b
  Channel b*32+1:  weighted sum of ALL 32 channels in block b
  ...

  out[c] = Σ W_mix[c,d] × in[d]   for c,d in same block
           d


ACROSS blocks:
──────────────

Block 0 (ch 0-31):   ────────►  Block 0 output (ch 0-31)
                                     │
                                     │ NO mixing between blocks!
                                     │
Block 1 (ch 32-63):  ────────►  Block 1 output (ch 32-63)
```

### Full CSSM with LMME (GatedCSSM)

```python
class GatedCSSM(nn.Module):
    channels: int = 384
    block_size: int = 32  # LMME block size

    def __call__(self, x):
        B, T, H, W, C = x.shape
        num_blocks = C // self.block_size

        # ═══════════════════════════════════════════════════════════
        # STEP 1: Reshape input to blocks
        # ═══════════════════════════════════════════════════════════
        x_blocks = x.reshape(B, T, H, W, num_blocks, self.block_size)
        # (B, T, H, W, num_blocks, block_size)

        # ═══════════════════════════════════════════════════════════
        # STEP 2: FFT (spatial → frequency domain)
        # ═══════════════════════════════════════════════════════════
        # Transpose for FFT: put spatial dims last
        x_t = x_blocks.transpose(0, 1, 4, 5, 2, 3)
        # (B, T, num_blocks, block_size, H, W)

        X_hat = jnp.fft.rfft2(x_t, axes=(-2, -1))
        # (B, T, num_blocks, block_size, H, W_freq) complex

        # ═══════════════════════════════════════════════════════════
        # STEP 3: Build LMME transition matrices
        # ═══════════════════════════════════════════════════════════
        # Transition matrix K: (num_blocks, block_size, block_size)
        # Applied at each frequency (H × W_freq) and timestep (T)

        K_raw = self.param('K', shape=(num_blocks, H, W_freq,
                                       self.block_size, self.block_size))
        # This is a (block_size × block_size) matrix at EACH frequency!

        # Apply spectral normalization for stability
        K = normalize_spectral(K_raw)

        # ═══════════════════════════════════════════════════════════
        # STEP 4: Temporal scan with matrix recurrence
        # ═══════════════════════════════════════════════════════════
        # For each block and frequency:
        #   state_t = K @ state_{t-1} + input_t
        #
        # This is where LMME mixing happens!
        # The matrix K mixes channels WITHIN each block

        def scan_fn(carry, x_t):
            # carry: (num_blocks, block_size, H, W_freq)
            # x_t:   (num_blocks, block_size, H, W_freq)

            # Matrix multiply K @ carry for each block/frequency
            # K:     (num_blocks, H, W_freq, block_size, block_size)
            # carry: (num_blocks, block_size, H, W_freq)

            # Rearrange for matmul
            carry_r = carry.transpose(0, 2, 3, 1)
            # (num_blocks, H, W_freq, block_size)

            K_carry = jnp.einsum('nhwcd,nhwd->nhwc', K, carry_r)
            # 'nhwcd' @ 'nhwd' -> 'nhwc'
            # n=num_blocks, h=H, w=W_freq, c,d=block_size

            K_carry = K_carry.transpose(0, 3, 1, 2)
            # (num_blocks, block_size, H, W_freq)

            new_carry = K_carry + x_t
            return new_carry, new_carry

        init_carry = jnp.zeros((B, num_blocks, self.block_size, H, W_freq),
                               dtype=jnp.complex64)

        _, Y_hat = jax.lax.scan(scan_fn, init_carry,
                                X_hat.transpose(1, 0, 2, 3, 4, 5))
        # Y_hat: (T, B, num_blocks, block_size, H, W_freq)

        Y_hat = Y_hat.transpose(1, 0, 2, 3, 4, 5)
        # (B, T, num_blocks, block_size, H, W_freq)

        # ═══════════════════════════════════════════════════════════
        # STEP 5: Inverse FFT (frequency → spatial domain)
        # ═══════════════════════════════════════════════════════════
        y_t = jnp.fft.irfft2(Y_hat, s=(H, W), axes=(-2, -1))
        # (B, T, num_blocks, block_size, H, W)

        # Transpose back
        y_blocks = y_t.transpose(0, 1, 4, 5, 2, 3)
        # (B, T, H, W, num_blocks, block_size)

        # Reshape to original channel dim
        y = y_blocks.reshape(B, T, H, W, C)
        # (B, T, H, W, C)

        return y
```

---

## 7. Why O(C³) for Full Channel Mixing <a name="7-complexity"></a>

### The Question

> "This creates a C×C matrix at every spatial frequency - the scan operator would be O(C³) per step"

### Understanding FFT Frequencies

Each frequency bin represents a **pattern** in the image:

```
Frequency (0,0): DC component = average brightness
Frequency (1,0): Horizontal stripes, 1 cycle
Frequency (0,1): Vertical stripes, 1 cycle
Frequency (2,3): Diagonal pattern, 2 horizontal + 3 vertical cycles
...

Total frequencies: H × W_freq = H × (W//2 + 1)

For 56×56 feature map: 56 × 29 = 1,624 frequencies
```

### Each Channel at Each Frequency

After FFT, we have:

```
X_hat: (B, T, C, H, W_freq) complex

This means:
  - B batches
  - T timesteps
  - C channels
  - H × W_freq frequency bins

At frequency (h, w):
  X_hat[:, :, :, h, w] has shape (B, T, C)

  This is a C-dimensional vector!
  Each of the C channels has a complex value at this frequency.
```

### Full Channel Mixing at Each Frequency

If we want channels to mix:

```
At frequency (h, w):

  input:  (B, T, C)
  output: (B, T, C)

  output = input @ W[h, w]

  where W[h, w] is a (C × C) mixing matrix

Matrix multiply cost: O(C² × C) = O(C³) for batch operations

Actually O(B × T × C²) per frequency, but the C² matrices
require C³ operations to apply to C-dimensional vectors naively.
```

### Why This Explodes

```
Full mixing at each frequency:

  Parameters: H × W_freq × C × C
            = 56 × 29 × 384 × 384
            = 239 BILLION parameters!

  Operations per step: H × W_freq × O(C²)
                     = 1,624 × 147,456
                     = 240 million ops per timestep!
```

### LMME Fixes This

```
Block mixing at each frequency:

  Parameters: H × W_freq × num_blocks × block_size × block_size
            = 56 × 29 × 12 × 32 × 32
            = 20 million parameters (12,000× fewer!)

  Operations per step: H × W_freq × num_blocks × O(block_size²)
                     = 1,624 × 12 × 1,024
                     = 20 million ops (12× fewer!)
```

### Depthwise: Even Simpler

```
No channel mixing (current CSSM):

  Each channel has its own scalar decay at each frequency.

  Parameters: H × W_freq × C (just one scalar per channel-frequency)
            = 56 × 29 × 384
            = 624,000 parameters

  Operations: H × W_freq × C × O(1)
            = 624,000 ops (384× fewer than LMME!)
```

### Summary Table

| Approach | Parameters | Ops/Step | Channel Mixing |
|----------|------------|----------|----------------|
| Full C×C | O(H×W×C²) | O(H×W×C³) | All ↔ All |
| LMME | O(H×W×(C/B)×B²) | O(H×W×C×B) | Within blocks |
| Depthwise | O(H×W×C) | O(H×W×C) | None |

Where B = block_size.

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────────┐
│                     CSSM QUICK REFERENCE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  FFT: Image (H,W) → Freq (H, W//2+1) complex                   │
│       Convolution = element-wise multiply in freq domain        │
│                                                                 │
│  DEPTHWISE: Each channel has its own kernel, no mixing          │
│             out[c] = in[c] ⊛ kernel[c]                         │
│                                                                 │
│  OPPONENT: X (excitation) and Y (inhibition) interact           │
│            X_t = α·X - K_I·μ·Y + U                             │
│            Y_t = K_E·γ·X + δ·Y                                 │
│            2×2 state matrix at each frequency                   │
│                                                                 │
│  GOOM: Log-space with phase for sign                           │
│        multiply → add logs                                      │
│        subtract → add π to phase, then add                     │
│        add → LogSumExp                                          │
│                                                                 │
│  LMME: Block-diagonal channel mixing                           │
│        Split C channels into C/B blocks of size B              │
│        Mix within blocks: O(B²) instead of O(C²)               │
│                                                                 │
│  LINEAR: No log-space, uses *, -, + directly                   │
│          Sequential O(T), used as ablation control             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```
