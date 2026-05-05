# Canonical Neural Computations and Parallel Scan Architectures: A Framework for Scalable Computational Neuroscience

## Abstract

Computational neuroscience has produced a rich taxonomy of canonical neural computations—from divisive normalization to predictive coding—each formalized in recurrent neural network models that capture specific aspects of cortical processing. However, these models face a fundamental scaling barrier: their sequential nature precludes parallelization, limiting both the temporal extent of simulations and the feasibility of fitting models to large-scale neural recordings. This document presents a systematic framework for implementing canonical neural computations using parallel associative scans, which achieve O(log T) time complexity through the parallel prefix algorithm. We analyze which computations admit exact scan implementations, which require approximation, and what architectural choices enable each. The resulting framework offers two advantages for computational neuroscience: (1) scalability to the long timescales and large datasets needed to observe emergent neural phenomena, and (2) linear dynamics that are inherently more interpretable than nonlinear recurrent networks, enabling principled analysis of learned circuit structure.

---

## 1. Introduction

### 1.1 The Modeling Challenge in Computational Neuroscience

Understanding how neural circuits implement computation remains one of the central challenges in neuroscience. Over the past three decades, researchers have identified a set of "canonical" computations that appear repeatedly across brain areas and species (Carandini & Heeger, 2012; Miller, 2016). These include operations like divisive normalization, gain modulation, winner-take-all competition, and temporal integration—building blocks that combine to produce the remarkable computational capabilities of biological neural networks.

To formalize these computations, the field has developed a variety of recurrent neural network (RNN) models. The horizontal gated recurrent unit (hGRU) captures contour integration in visual cortex through excitatory-inhibitory interactions (Linsley et al., 2018). Predictive coding networks implement hierarchical Bayesian inference through recurrent error correction (Rao & Ballard, 1999). Normalization models explain contrast invariance and attention through divisive pooling (Reynolds & Heeger, 2009). Each model provides insight into specific neural phenomena, and collectively they constitute a computational vocabulary for describing cortical processing.

### 1.2 The Scaling Problem

Despite their explanatory power, these models share a critical limitation: they are inherently sequential. The recurrence relation

```
h_t = f(W_h · h_{t-1} + W_x · x_t)
```

requires computing h_{t-1} before h_t, precluding parallelization across time. This sequential bottleneck has several consequences:

1. **Computational cost**: Simulating T timesteps requires O(T) sequential operations, regardless of available parallel hardware.

2. **Training difficulty**: Backpropagation through time over long sequences leads to vanishing or exploding gradients (Bengio et al., 1994; Pascanu et al., 2013).

3. **Limited temporal extent**: Most published models simulate only hundreds to thousands of timesteps, far short of the millions of timesteps in typical neural recordings.

4. **Dataset constraints**: Fitting models to large-scale neural data (e.g., Neuropixels recordings with thousands of neurons over hours) becomes computationally prohibitive.

These limitations matter because many neural phenomena of interest—learning, adaptation, memory consolidation, oscillatory dynamics—unfold over timescales that exceed current modeling capabilities. If we wish to observe emergent phenomena in neural models, or to infer circuit structure by fitting models to experimental data, we need architectures that scale.

### 1.3 Parallel Scans: A Path Forward

The parallel scan (or prefix sum) algorithm, introduced by Blelloch (1990), offers a solution. For any associative binary operator ⊕, the sequence of cumulative applications

```
y_0 = x_0
y_1 = x_0 ⊕ x_1
y_2 = x_0 ⊕ x_1 ⊕ x_2
...
y_T = x_0 ⊕ x_1 ⊕ ... ⊕ x_T
```

can be computed in O(log T) parallel time using O(T) processors. This represents an exponential improvement over sequential computation.

The key constraint is associativity: (a ⊕ b) ⊕ c = a ⊕ (b ⊕ c). Linear recurrences satisfy this constraint. If we write the recurrence as

```
h_t = A_t · h_{t-1} + b_t
```

then the binary operator (A_t, b_t) ⊕ (A_{t-1}, b_{t-1}) = (A_t · A_{t-1}, A_t · b_{t-1} + b_t) is associative, and the entire sequence {h_1, h_2, ..., h_T} can be computed in O(log T) parallel steps (Martin & Cundy, 2018; Gu et al., 2022).

### 1.4 The Interpretability Advantage

Beyond scalability, scan-based architectures offer a second advantage: interpretability. Because the temporal dynamics are linear (within each layer), we can apply the full apparatus of linear systems theory:

- **Spectral analysis**: The eigenvalues of the transition matrix directly encode timescales and oscillation frequencies.
- **Impulse response**: The system's response to a single input is fully characterized by the matrix exponential.
- **Superposition**: Responses to complex inputs decompose into responses to components.
- **Stability analysis**: Boundedness is guaranteed when all eigenvalues have magnitude less than one.

This stands in contrast to nonlinear RNNs, where dynamics can exhibit chaos, bifurcations, and other phenomena that resist analytical characterization. While nonlinearity provides computational power, it also obscures mechanism. A linear system that explains neural data is inherently more interpretable than a nonlinear system that fits equally well.

### 1.5 Document Overview

This document proceeds as follows. Section 2 reviews the canonical neural computations identified in the literature. Section 3 surveys RNN models developed to explain specific neural phenomena. Section 4 presents the mathematical framework for parallel scans. Section 5 systematically analyzes which computations can be implemented in scans, and how. Section 6 presents specific architectures for modeling neural phenomena. Section 7 discusses implications for computational neuroscience. Section 8 addresses limitations and future directions.

---

## 2. Canonical Neural Computations

Carandini and Heeger (2012) proposed that a small set of canonical computations underlies the diversity of neural processing. This section reviews these computations, their computational roles, and their neural implementations.

### 2.1 Linear Filtering

**Definition**: Weighted summation of inputs over space and/or time.

```
y(t) = ∫ w(τ) · x(t - τ) dτ
```

**Neural implementation**: The dendritic tree performs spatial summation of synaptic inputs. Temporal filtering arises from membrane time constants and synaptic dynamics.

**Computational role**: Extracts features at specific spatial frequencies, orientations, and temporal scales. The foundation of receptive field structure.

**Key references**: Hubel & Wiesel (1962); Adelson & Bergen (1985); Simoncelli & Heeger (1998).

### 2.2 Divisive Normalization

**Definition**: Response divided by a measure of local activity.

```
y = x / (σ² + Σ_i w_i · x_i^n)
```

**Neural implementation**: Shunting inhibition, where inhibitory conductances divide rather than subtract. GABAergic interneurons provide the inhibitory drive.

**Computational role**: Contrast gain control, attention, multisensory integration, efficient coding. Creates invariance to stimulus intensity while preserving relative differences.

**Key references**: Heeger (1992); Carandini & Heeger (2012); Schwartz & Simoncelli (2001).

### 2.3 Thresholding and Rectification

**Definition**: Nonlinear input-output function with a threshold.

```
y = max(0, x - θ)  [ReLU-like]
y = x^n / (σ^n + x^n)  [Naka-Rushton]
```

**Neural implementation**: Spike threshold in neurons. Subthreshold inputs produce no output; suprathreshold inputs produce graded or all-or-none responses.

**Computational role**: Noise rejection, sparse coding, decision making. Creates qualitative distinctions from quantitative differences.

**Key references**: Naka & Rushton (1966); Laughlin (1981); Priebe & Ferster (2008).

### 2.4 Multiplicative Gain Modulation

**Definition**: One signal multiplicatively scales another.

```
y = g(context) · f(input)
```

**Neural implementation**: Neuromodulatory inputs (dopamine, acetylcholine, norepinephrine) alter neuronal gain. Attentional signals modulate sensory responses multiplicatively.

**Computational role**: Attention, coordinate transformations, context-dependent processing. Allows flexible routing of information.

**Key references**: Salinas & Thier (2000); Treue & Martínez-Trujillo (1999); McAdams & Maunsell (1999).

### 2.5 Winner-Take-All and Competitive Selection

**Definition**: Mutual inhibition leading to selection of strongest input.

```
y_i = x_i · [x_i = max_j(x_j)]  [hard WTA]
y_i = exp(x_i) / Σ_j exp(x_j)  [soft WTA / softmax]
```

**Neural implementation**: Lateral inhibition in cortical circuits. Recurrent excitation amplifies the winner while inhibition suppresses losers.

**Computational role**: Decision making, attention allocation, categorical perception. Converts graded evidence into discrete choices.

**Key references**: Hopfield (1982); Usher & McClelland (2001); Wang (2002).

### 2.6 Temporal Integration and Evidence Accumulation

**Definition**: Accumulating input over time toward a decision bound.

```
h_t = λ · h_{t-1} + x_t  [leaky integration]
```

**Neural implementation**: Persistent activity in prefrontal and parietal cortex. Recurrent excitation maintains accumulated evidence.

**Computational role**: Perceptual decisions, working memory, motor planning. Implements drift-diffusion and accumulator models.

**Key references**: Gold & Shadlen (2007); Brody & Hanks (2016); Mante et al. (2013).

### 2.7 Adaptation and Habituation

**Definition**: Response reduction with repeated or sustained stimulation.

```
y_t = x_t - α · running_mean(x)
```

**Neural implementation**: Short-term synaptic depression, slow afterhyperpolarization, intrinsic conductances. Multiple timescales from milliseconds to minutes.

**Computational role**: Novelty detection, efficient coding, contrast enhancement. Removes predictable components of input.

**Key references**: Tsodyks & Markram (1997); Fairhall et al. (2001); Weber et al. (2019).

### 2.8 Predictive Coding and Error Computation

**Definition**: Computing the difference between predicted and observed input.

```
error = observation - prediction
prediction = f(higher_level_representation)
```

**Neural implementation**: Hierarchical cortical circuits where feedback provides predictions and feedforward carries errors. Distinct cell types for error vs. prediction.

**Computational role**: Hierarchical Bayesian inference, efficient coding, learning. The brain as a prediction machine.

**Key references**: Rao & Ballard (1999); Friston (2005); Keller & Mrsic-Flogel (2018).

### 2.9 Neural Oscillations

**Definition**: Periodic fluctuations in neural activity.

```
dθ/dt = ω + K · sin(θ_external - θ)  [phase oscillator]
```

**Neural implementation**: Interplay of excitation and inhibition creates rhythms. Different frequency bands (theta, alpha, gamma) arise from different circuit mechanisms.

**Computational role**: Temporal coordination, attention, memory encoding and retrieval. Oscillations as clocks and gates.

**Key references**: Buzsáki & Draguhn (2004); Fries (2015); Lisman & Jensen (2013).

### 2.10 Center-Surround Organization

**Definition**: Excitation from spatial center, inhibition from surround.

```
y = DoG(x) = G_center(x) - G_surround(x)
```

**Neural implementation**: Lateral inhibition mediated by horizontal connections and inhibitory interneurons. ON-center/OFF-surround and vice versa.

**Computational role**: Edge detection, contrast enhancement, redundancy reduction. Whitens natural image statistics.

**Key references**: Kuffler (1953); Rodieck & Stone (1965); Srinivasan et al. (1982).

### 2.11 Working Memory and Persistent Activity

**Definition**: Maintaining information over delays without external input.

```
h_t = h_{t-1}  [perfect integrator]
h_t = W · f(h_{t-1})  [attractor dynamics]
```

**Neural implementation**: Recurrent excitation in prefrontal cortex. Attractor networks with discrete or continuous attractors.

**Computational role**: Bridging temporal gaps, maintaining task context, mental manipulation.

**Key references**: Funahashi et al. (1989); Goldman-Rakic (1995); Compte et al. (2000).

### 2.12 Coincidence Detection

**Definition**: Detecting temporal coincidence of multiple inputs.

```
y = f(x_1 · x_2)  [product]
y = f(x_1 + x_2) with threshold  [sum + threshold]
```

**Neural implementation**: NMDA receptor requires both glutamate binding and depolarization. Dendritic nonlinearities amplify coincident inputs.

**Computational role**: Feature binding, spike-timing dependent plasticity, sound localization.

**Key references**: Magee & Johnston (1997); Bi & Poo (1998); Carr & Konishi (1990).

### 2.13 Sequence Detection

**Definition**: Detecting specific temporal patterns of activation.

```
y = 1 if input matches pattern (A → B → C)
```

**Neural implementation**: Synfire chains, time cells in hippocampus, sequential activity in motor cortex.

**Computational role**: Temporal pattern recognition, motor sequencing, episodic memory.

**Key references**: Abeles (1991); MacDonald et al. (2011); Pastalkova et al. (2008).

### 2.14 Associative/Hebbian Learning

**Definition**: Strengthening connections between co-active neurons.

```
Δw_ij = η · x_i · x_j  [basic Hebbian]
Δw_ij = η · x_i · (x_j - w_ij · x_i)  [Oja's rule]
```

**Neural implementation**: Long-term potentiation and depression. NMDA-dependent coincidence detection triggers plasticity.

**Computational role**: Learning, memory formation, development of receptive fields.

**Key references**: Hebb (1949); Bliss & Lømo (1973); Bi & Poo (1998).

### 2.15 Feedback Amplification

**Definition**: Recurrent excitation amplifies weak signals.

```
y = x + α · W · y  [linear amplification]
```

**Neural implementation**: Recurrent connections within cortical layers. Feedback from higher areas.

**Computational role**: Signal amplification, filling-in, figure-ground segregation.

**Key references**: Douglas et al. (1995); Lamme & Roelfsema (2000); Liang et al. (2017).

---

## 3. Recurrent Neural Network Models of Neural Phenomena

This section reviews influential RNN models that formalize specific canonical computations. For each, we note the sequential nature that limits scalability.

### 3.1 The Horizontal Gated Recurrent Unit (hGRU)

**Phenomenon modeled**: Contour integration, figure-ground segregation, texture segmentation.

**Model**: Linsley et al. (2018) proposed the hGRU, which implements excitatory-inhibitory interactions through horizontal connections:

```
G_t = σ(W_g * H_{t-1} + U_g * X)  [gate]
C_t = tanh(W_c * (H_{t-1} ⊙ G_t) + U_c * X)  [candidate]
H_t = (1 - G_t) ⊙ H_{t-1} + G_t ⊙ C_t  [update]
```

The model includes multiplicative interactions (H_{t-1} ⊙ G_t) that enable the network to selectively propagate activity along contours while suppressing noise.

**Key result**: The hGRU matches human performance on challenging contour integration tasks (PathFinder) where feedforward networks fail.

**Scaling limitation**: The recurrence requires O(T) sequential steps. Training on long sequences (T > 100) becomes difficult due to gradient issues.

### 3.2 Predictive Coding Networks

**Phenomenon modeled**: Hierarchical inference, efficient coding, extra-classical receptive field effects.

**Model**: Rao & Ballard (1999) formalized predictive coding as a recurrent network where each level predicts the level below:

```
r_t^(l) = r_{t-1}^(l) + ε · (U^(l) · e_t^(l-1) - e_t^(l))  [representation update]
e_t^(l) = x^(l) - W^(l) · r_t^(l+1)  [error computation]
```

Feedback (W^(l) · r_t^(l+1)) provides top-down predictions; feedforward (U^(l) · e_t^(l-1)) carries bottom-up errors.

**Key result**: Predictive coding explains end-stopping, surround suppression, and other extra-classical receptive field phenomena as consequences of hierarchical prediction.

**Scaling limitation**: Multiple interacting levels, each with recurrent dynamics. Sequential computation at each level and across levels.

### 3.3 Normalization Models of Attention

**Phenomenon modeled**: Attentional modulation, response saturation, cross-orientation suppression.

**Model**: Reynolds & Heeger (2009) proposed the normalization model of attention:

```
R = A · E · S / (σ² + A · S)
```

where E is stimulus drive, S is suppressive drive, and A is attention signal.

**Key result**: A single model explains how attention affects contrast response functions, size tuning, and feature-based effects.

**Scaling limitation**: The division operation is inherently nonlinear and applied at each timestep. Sequential updates when combined with temporal dynamics.

### 3.4 Attractor Networks for Working Memory

**Phenomenon modeled**: Persistent activity, categorical representations, decision making.

**Model**: Compte et al. (2000) developed a spiking network model of spatial working memory:

```
τ dh_i/dt = -h_i + Σ_j W_ij · f(h_j) + I_i + noise
```

with structured connectivity W_ij that creates a continuous attractor ("bump").

**Key result**: The model produces persistent activity that encodes remembered locations, with drift patterns matching experimental observations.

**Scaling limitation**: Dynamics involve nonlinear firing rate function f(·). Must be simulated with small time steps for numerical stability.

### 3.5 Wilson-Cowan Oscillator Models

**Phenomenon modeled**: Neural oscillations, pattern formation, traveling waves.

**Model**: Wilson & Cowan (1972) introduced coupled excitatory-inhibitory population equations:

```
τ_E dE/dt = -E + f(w_EE · E - w_EI · I + h_E)
τ_I dI/dt = -I + f(w_IE · E - w_II · I + h_I)
```

**Key result**: Depending on parameters, the system produces damped oscillations, limit cycles, or chaotic dynamics—matching the diversity of neural rhythms.

**Scaling limitation**: Nonlinear dynamics require careful numerical integration. Chaotic regimes prevent parallelization.

### 3.6 Short-Term Plasticity Models

**Phenomenon modeled**: Synaptic adaptation, temporal filtering, gain control.

**Model**: Tsodyks & Markram (1997) formalized short-term depression and facilitation:

```
dx/dt = (1 - x)/τ_D - u · x · δ(t - t_spike)  [depression]
du/dt = (U - u)/τ_F + U · (1 - u) · δ(t - t_spike)  [facilitation]
```

**Key result**: Short-term plasticity acts as a temporal filter, implementing high-pass (depression) or low-pass (facilitation) filtering of spike trains.

**Scaling limitation**: State variables (x, u) must be updated at each spike. Sequential processing of spike trains.

### 3.7 Drift-Diffusion Models

**Phenomenon modeled**: Perceptual decision making, reaction times, error rates.

**Model**: The drift-diffusion model (Ratcliff, 1978; Gold & Shadlen, 2007):

```
dX = μ dt + σ dW  [evidence accumulation]
decision when X crosses +θ or -θ
```

**Key result**: DDM quantitatively predicts reaction time distributions, accuracy, and speed-accuracy tradeoffs across many tasks.

**Scaling limitation**: Stochastic simulation requires sequential sampling. Fitting to data involves many simulated trials.

---

## 4. Mathematical Framework for Parallel Scans

This section presents the mathematical foundations of associative scans, establishing when and how they can replace sequential recurrence.

### 4.1 The Parallel Prefix Algorithm

Given a sequence of elements x_0, x_1, ..., x_{T-1} and an associative binary operator ⊕, the parallel prefix (scan) algorithm computes all prefix sums:

```
y_i = x_0 ⊕ x_1 ⊕ ... ⊕ x_i  for i = 0, 1, ..., T-1
```

**Associativity requirement**: (a ⊕ b) ⊕ c = a ⊕ (b ⊕ c)

The algorithm proceeds in two phases (Blelloch, 1990):

**Up-sweep (reduce)**: Compute pairwise combinations bottom-up, building a tree of partial results.

**Down-sweep**: Propagate results back down the tree to compute all prefixes.

**Complexity**: O(T) work, O(log T) parallel depth.

### 4.2 Linear Recurrences as Scans

Consider the first-order linear recurrence:

```
h_t = A_t · h_{t-1} + b_t
```

Define the binary operator on pairs (A, b):

```
(A_j, b_j) ⊕ (A_i, b_i) = (A_j · A_i, A_j · b_i + b_j)
```

**Theorem**: This operator is associative.

*Proof*:
```
[(A_k, b_k) ⊕ (A_j, b_j)] ⊕ (A_i, b_i)
= (A_k · A_j, A_k · b_j + b_k) ⊕ (A_i, b_i)
= (A_k · A_j · A_i, A_k · A_j · b_i + A_k · b_j + b_k)

(A_k, b_k) ⊕ [(A_j, b_j) ⊕ (A_i, b_i)]
= (A_k, b_k) ⊕ (A_j · A_i, A_j · b_i + b_j)
= (A_k · A_j · A_i, A_k · (A_j · b_i + b_j) + b_k)
= (A_k · A_j · A_i, A_k · A_j · b_i + A_k · b_j + b_k)
```

Both expressions are equal. □

**Corollary**: Any linear RNN can be computed in O(log T) parallel time.

### 4.3 Numerical Stability via Log-Space Computation

Direct computation of matrix products over long sequences leads to numerical overflow or underflow. The solution is to work in log-space.

**GOOM representation** (Generalized Order of Magnitude): Represent complex number z = r · e^{iφ} as:

```
log(z) = log(r) + i·φ
```

In this representation:
- Multiplication becomes addition: log(a · b) = log(a) + log(b)
- Addition becomes log-sum-exp: log(a + b) = log_add_exp(log(a), log(b))

The log_add_exp operation is computed stably as:

```
log_add_exp(x, y) = max(x, y) + log(1 + exp(-|x - y|))
```

**Result**: Matrix operations in log-space are numerically stable for arbitrary sequence lengths.

### 4.4 The Constraint: Linearity

The critical constraint is that the recurrence must be linear in the state h. Nonlinear recurrences:

```
h_t = f(A_t · h_{t-1} + b_t)  [nonlinear activation]
h_t = h_{t-1} · g(h_{t-1})  [state-dependent gating]
h_t = h_{t-1} / ||h_{t-1}||  [normalization]
```

violate associativity and cannot be directly parallelized.

However, the coefficients A_t and b_t can depend arbitrarily on the input x_t. This enables input-dependent dynamics while maintaining scannability.

---

## 5. Implementing Canonical Computations with Scans

This section systematically analyzes each canonical computation, determining whether it admits a scan implementation and, if not, what approximations are possible.

### 5.1 Directly Implementable Computations

#### 5.1.1 Linear Filtering

**Implementation**: Exact.

In the frequency domain, convolution becomes pointwise multiplication. In log-space, this becomes addition:

```
Y(ω) = H(ω) · X(ω)  →  log Y(ω) = log H(ω) + log X(ω)
```

The scan accumulates filter responses over time, with the transition matrix encoding the filter kernel.

**Architecture**: Single scan layer with FFT-domain transition matrices.

#### 5.1.2 Temporal Integration

**Implementation**: Exact.

The leaky integrator h_t = λ · h_{t-1} + x_t is a linear recurrence. The decay parameter λ controls the integration timescale.

**Architecture**: Scan with scalar or diagonal transition matrix. Different channels can have different timescales.

#### 5.1.3 Adaptation

**Implementation**: Exact via opponent channels.

With two channels [X, Y]:
```
X_t = decay · X_{t-1} - μ · K · Y_{t-1} + input
Y_t = decay · Y_{t-1} + ν · K · X_{t-1}
```

Y tracks a smoothed version of X. The term -μ · K · Y implements adaptation: current response minus recent history.

**Architecture**: 2×2 transition matrix with negative off-diagonal coupling.

#### 5.1.4 Oscillations

**Implementation**: Exact via complex eigenvalues.

A 2×2 transition matrix with complex eigenvalues produces oscillatory dynamics:
```
[cos(θ)  -sin(θ)]  eigenvalues: e^{±iθ}
[sin(θ)   cos(θ)]
```

Different frequencies arise from different θ values. Decay (|eigenvalue| < 1) prevents unbounded growth.

**Architecture**: 2×2 scan with appropriate eigenvalue structure. The E-I opponent matrix naturally produces complex eigenvalues.

#### 5.1.5 Center-Surround Organization

**Implementation**: Exact via opponent kernels.

With excitatory kernel K_E (narrow) and inhibitory kernel K_I (wide):
```
response = K_E * input - K_I * input = (K_E - K_I) * input
```

This is the difference-of-Gaussians (DoG) receptive field.

**Architecture**: Scan with K_E in excitatory pathway, K_I in inhibitory pathway.

#### 5.1.6 Working Memory

**Implementation**: Exact via high decay.

With decay λ ≈ 1, state persists indefinitely:
```
h_t ≈ h_{t-1}  when λ → 1
```

Input and output gates control what enters and exits memory.

**Architecture**: Scan with decay gates near unity. Input-dependent gating determines write/read operations.

#### 5.1.7 Feedback Amplification

**Implementation**: Exact via off-diagonal excitation.

Positive off-diagonal terms create mutual excitation:
```
[λ    α]
[α    λ]
```

Activity in one channel amplifies the other. Stability requires |eigenvalues| < 1.

**Architecture**: 2×2 scan with positive coupling. Spectral radius constraint enforced by sigmoid squashing.

#### 5.1.8 Linear Predictive Coding

**Implementation**: Exact for linear predictions.

Channel Y predicts channel X:
```
X_t = X_{t-1} - prediction_error = X_{t-1} - (X_{t-1} - Y_{t-1})
Y_t = f(X_{t-1})  [linear predictor]
```

The error X - Y propagates forward; the prediction Y comes from past X.

**Architecture**: 2×2 scan where Y tracks filtered X, and X receives -Y (error = input - prediction).

### 5.2 Computations Requiring Approximation

#### 5.2.1 Divisive Normalization

**Problem**: Division by state violates associativity.

**Approximation strategies**:

1. **Inter-layer normalization**: Apply LayerNorm between scan layers. This provides divisive normalization at layer boundaries, approximating continuous normalization.

2. **Auxiliary state for denominator**: Track the normalization denominator as a separate state channel:
   ```
   S_t = λ · S_{t-1} + x_t²
   ```
   Then apply division after the scan: y = X / sqrt(S).

3. **Soft normalization via decay**: Train decay to be inversely related to input magnitude. High activity → faster decay → implicit gain control.

**Architecture**: Scan → LayerNorm, or Scan([X, S]) → X / sqrt(S) post-processing.

#### 5.2.2 Thresholding and Rectification

**Problem**: Nonlinear activation at each timestep violates associativity.

**Approximation strategies**:

1. **Inter-layer nonlinearity**: Apply GELU, ReLU, or sigmoid between scan layers. Each layer is linear; nonlinearity separates layers.

2. **Soft thresholding via gating**: Input-dependent gates act as soft thresholds:
   ```
   gate = σ(W · input)  [near 0 for weak inputs]
   effective_input = gate · input  [weak inputs suppressed]
   ```

3. **Magnitude-dependent decay**: Fast decay for weak signals, slow decay for strong:
   ```
   decay = σ(W · |input|)
   ```
   Weak signals die out quickly, approximating thresholding over time.

**Architecture**: Scan → GELU → Scan, or input gating within scan.

#### 5.2.3 Multiplicative Gating (State × State)

**Problem**: h_t = h_{t-1} · g(h_{t-1}) requires state-dependent gating, violating associativity.

**Approximation strategies**:

1. **Z interaction channel**: Add a third state channel that tracks interaction:
   ```
   Z_t = γ · X_{t-1} + δ · Y_{t-1} + ε · Z_{t-1}
   ```
   Z accumulates a linear combination of X and Y. When Z feeds back into X and Y, it provides an approximation to X · Y gating.

2. **Inter-layer multiplication**: Compute X and Y in separate branches, multiply between layers:
   ```
   X_out = Scan_X(input)
   Y_out = Scan_Y(input)
   XY = X_out * Y_out  [between layers]
   output = Scan_XY(XY)
   ```

3. **Input-derived proxy**: Use input features as proxy for state:
   ```
   gate = σ(W · input)  [not state-dependent]
   ```
   This captures input-dependent gating but not true state-dependent gating.

**Architecture**: 3×3 scan (hgru_bi), or multi-branch with inter-layer multiplication.

#### 5.2.4 Winner-Take-All

**Problem**: Competitive selection requires comparison across units, which is nonlinear.

**Approximation strategies**:

1. **Strong mutual inhibition**: Large negative off-diagonal coupling creates competition:
   ```
   [-large  decay]
   [decay  -large]
   ```
   The more active channel suppresses the other. Soft, not hard, competition.

2. **Inter-layer softmax**: Apply softmax between scan layers:
   ```
   Scan → Softmax → Scan
   ```

3. **Iterative refinement with annealing**: Multiple passes with decreasing softmax temperature:
   ```
   for i in range(K):
       x = Scan(x)
       x = Softmax(x / T)
       T = T * 0.9
   ```
   Temperature annealing sharpens competition toward hard WTA.

**Architecture**: Scan with strong inhibition, or [Scan → Softmax] × K.

#### 5.2.5 Coincidence Detection

**Problem**: Detecting X · Y at specific times requires state multiplication.

**Approximation strategies**:

1. **Z channel**: As above, Z tracks X + Y history. High Z indicates both active (approximately).

2. **Sum + threshold**: X + Y > θ implies both are active (for positive, bounded X and Y).

3. **Inter-layer product**: Compute X · Y between scan layers.

**Architecture**: 3×3 scan or multi-layer with inter-layer multiplication.

#### 5.2.6 Nonlinear Predictive Coding

**Problem**: Nonlinear prediction functions violate associativity.

**Approximation strategies**:

1. **Linear prediction + nonlinear error processing**: Prediction is linear (scannable); error passes through nonlinearity between layers.

2. **Multi-layer hierarchy**: Each layer provides linear prediction for the layer below; nonlinearity at layer boundaries.

**Architecture**: Stack of [Scan → GELU] layers with skip connections for error propagation.

### 5.3 Fundamentally Non-Scannable Computations

#### 5.3.1 Hebbian Learning (Online)

**Problem**: Weight updates require state × state products computed online.

**Status**: Cannot be implemented within a scan. Learning must occur in an outer loop (backpropagation), not within the forward pass.

**Alternative**: The scan can compute quantities useful for Hebbian learning (e.g., activity correlations), which are then used for weight updates in the outer loop.

#### 5.3.2 Arbitrary Sequence Detection

**Problem**: Detecting arbitrary patterns (A → B → C in exact order) requires state-dependent branching.

**Status**: Linear systems cannot implement arbitrary finite automata. Only "soft" sequence detection via accumulated state.

**Alternative**: Deep stacks of scan layers can learn approximate sequence detectors for specific patterns encountered in training.

---

## 6. Architectures for Neural Phenomena

This section presents specific scan-based architectures for modeling the neural phenomena discussed in Section 3.

### 6.1 Contour Integration (hGRU → hgru / hgru_bi)

**Original model**: The hGRU uses multiplicative gating for selective propagation along contours.

**Scan architecture (hgru)**: 2×2 linear opponent dynamics.

```
[X_t]   [decay_x    -μ_I · K_I] [X_{t-1}]   [U_X]
[Y_t] = [μ_E · K_E   decay_y  ] [Y_{t-1}] + [U_Y]
```

- X: Excitatory pathway carrying contour evidence
- Y: Inhibitory pathway providing surround suppression
- K_E: Narrow kernel for local excitation (contour linking)
- K_I: Wide kernel for surround inhibition (noise suppression)
- Input-dependent gates (μ_I, μ_E, decay) allow context-dependent dynamics

**Enhanced architecture (hgru_bi)**: 3×3 with Z interaction channel.

```
[X_t]   [decay_x    -μ_I · K_I   -α_I · K_I] [X_{t-1}]   [U_X]
[Y_t] = [μ_E · K_E   decay_y     +α_E · K_E] [Y_{t-1}] + [U_Y]
[Z_t]   [γ          δ            ε         ] [Z_{t-1}]   [U_Z]
```

Z learns to track X-Y correlation, approximating the multiplicative gating in the original hGRU.

**Comparison to original**:
- Original hGRU: O(T) sequential, explicit X·Y gating
- Scan hgru: O(log T) parallel, linear dynamics
- Scan hgru_bi: O(log T) parallel, Z approximates X·Y

### 6.2 Predictive Coding

**Original model**: Hierarchical network with prediction and error units.

**Scan architecture**: Multi-layer scan with skip connections.

```
Layer l:
  prediction = Scan_down(representation_l+1)
  error = input_l - prediction
  representation = Scan_up(error)
  error → Layer l+1
```

Each layer is a linear scan. Nonlinearity (GELU) applied to errors between layers. Skip connections propagate errors up the hierarchy.

**Key insight**: In the scan formulation, prediction is a linear filter of higher-layer representations. This captures the core computational structure while enabling parallelization.

### 6.3 Normalization and Attention

**Original model**: Divisive normalization of attention.

**Scan architecture**: Scan with LayerNorm and gating.

```
features = Scan(input)              # Temporal filtering
features = LayerNorm(features)      # Divisive normalization
attention = Softmax(Scan_attn(ctx)) # Attention weights
output = attention · features       # Attended output
```

LayerNorm provides divisive normalization. Attention weights computed from a separate scan pathway.

### 6.4 Working Memory and Persistent Activity

**Original model**: Attractor network with nonlinear dynamics.

**Scan architecture**: High-decay scan with gated read/write.

```
# Write gate: what to remember
write_gate = σ(W_w · input)

# Read gate: what to output
read_gate = σ(W_r · input)

# Memory state with high decay
memory_t = 0.99 · memory_{t-1} + write_gate · input

# Output gated read
output = read_gate · memory
```

High decay (0.99) enables long-term maintenance. Gates control information flow.

### 6.5 Oscillations and Rhythms

**Original model**: Wilson-Cowan with nonlinear dynamics.

**Scan architecture**: 2×2 opponent with complex eigenvalues.

```
[E_t]   [a   -b] [E_{t-1}]   [input]
[I_t] = [c    d] [I_{t-1}] + [0    ]
```

Choose a, b, c, d such that eigenvalues are e^{-γ ± iω}:
- γ controls decay (stability)
- ω controls oscillation frequency

Different frequency bands (theta, gamma, etc.) use different ω values.

### 6.6 Adaptation and Gain Control

**Original model**: Short-term synaptic plasticity.

**Scan architecture**: Opponent dynamics with asymmetric timescales.

```
# Fast pathway (signal)
X_t = fast_decay · X_{t-1} - K · Y_{t-1} + input

# Slow pathway (adaptation state)
Y_t = slow_decay · Y_{t-1} + K · X_{t-1}
```

Y accumulates activity over a slow timescale. X is reduced by Y, implementing adaptation.

The ratio of timescales (fast_decay / slow_decay) determines adaptation dynamics.

---

## 7. Implications for Computational Neuroscience

### 7.1 Scaling to Biological Timescales

Biological neural processes unfold over timescales ranging from milliseconds (synaptic transmission) to hours (memory consolidation). Traditional RNN models are limited to thousands of timesteps; scan-based models can process millions.

**Example**: A 30-minute neural recording at 1 kHz sampling contains 1.8 million timesteps. Sequential RNN simulation would require ~1.8 million forward steps. A scan with O(log T) ≈ 21 parallel steps can process this in seconds on modern hardware.

This enables:
- Fitting models to entire experimental sessions
- Simulating slow processes (learning, consolidation)
- Observing emergent phenomena that require long timescales

### 7.2 Inferring Circuit Structure from Data

A major goal of computational neuroscience is to infer the structure of neural circuits from experimental recordings. Scan-based models offer advantages for this task:

1. **Linear dynamics are identifiable**: The eigenvalues and eigenvectors of the transition matrix can be directly estimated from data using spectral methods.

2. **Spatial kernels are interpretable**: K_E and K_I correspond to excitatory and inhibitory connectivity, with clear spatial structure.

3. **Gates reveal context-dependence**: Input-dependent gates show how processing varies with stimulus or behavioral state.

4. **No local optima from chaos**: Linear systems don't exhibit chaotic dynamics, making optimization more reliable.

### 7.3 Emergent Phenomena

Complex systems often exhibit emergent phenomena—patterns that arise from simple rules but weren't explicitly programmed. Observing emergence requires:

1. **Scale**: Enough units and timesteps for patterns to develop
2. **Appropriate dynamics**: The right building blocks for the phenomenon
3. **Observation**: Ability to analyze what emerges

Scan architectures provide (1) through parallelization and (2) through implementing canonical computations. The linear structure aids (3) by enabling spectral analysis of emergent dynamics.

**Potential emergent phenomena**:
- Spontaneous oscillation patterns from E-I interactions
- Attractor dynamics from recurrent excitation
- Traveling waves from spatial-temporal kernels
- Critical dynamics near instability boundaries

### 7.4 The Interpretability Advantage

Nonlinear RNNs are notoriously difficult to interpret. The trained weights interact in complex ways, and small changes can qualitatively alter dynamics.

Scan-based models inherit the interpretability of linear systems:

**Eigenvalue analysis**: The eigenvalues of the transition matrix directly encode:
- Timescales (inverse of real part)
- Oscillation frequencies (imaginary part)
- Stability (magnitude < 1 required)

**Impulse response**: The response to a brief input is the matrix exponential, which can be computed and visualized.

**Superposition**: Responses to complex stimuli decompose into responses to simple components.

**Perturbation analysis**: Small changes to parameters produce proportionally small changes to dynamics.

This makes it possible to understand *why* a trained model behaves as it does, not just *that* it fits the data.

---

## 8. Limitations and Future Directions

### 8.1 The Nonlinearity Gap

The fundamental limitation of scan-based models is the restriction to linear dynamics. Biological neural circuits are replete with nonlinearities: spike thresholds, saturating synapses, divisive normalization, multiplicative gating.

While inter-layer nonlinearities partially address this, the approximation may be insufficient for some phenomena. Future work should:

1. Quantify the approximation error for specific nonlinear computations
2. Develop better approximation strategies
3. Identify which phenomena fundamentally require per-timestep nonlinearity

### 8.2 Chunked Scans for True Nonlinearity

A promising direction is chunked scanning: process the sequence in chunks using parallel scans, apply nonlinearity between chunks.

```
for chunk in chunks(sequence, size=T/K):
    state = parallel_scan(state, chunk)  # O(log(T/K))
    state = nonlinearity(state)          # Per-chunk nonlinearity
```

This trades full parallelism (O(log T)) for O(K · log(T/K)), but gains true nonlinearity every T/K timesteps.

The optimal chunk size balances parallelism against approximation quality.

### 8.3 Learning Scan Structure

Current architectures use hand-designed scan operators (2×2, 3×3 matrices). Future work could learn the optimal structure:

- How many channels? (2, 3, more?)
- What connectivity pattern? (which elements are nonzero?)
- What constraints? (sparsity, symmetry, stability?)

Neural architecture search for scan operators could discover structures suited to specific phenomena.

### 8.4 Multi-Scale Dynamics

Biological neural circuits exhibit dynamics over multiple timescales simultaneously. Current scan architectures use a single transition matrix per layer.

Extensions could include:
- Parallel scans at different timescales, merged across layers
- Hierarchical scans (scan of scans) for multi-scale structure
- Timescale-specific pathways with different decay constants

### 8.5 Connections to Other Frameworks

Scan-based models connect to several other frameworks in computational neuroscience and machine learning:

- **State space models** (Gu et al., 2022): S4 and Mamba use similar linear recurrence structure
- **Neural ODEs** (Chen et al., 2018): Continuous-time limit of discrete scans
- **Dynamical systems theory**: Linear scans are amenable to full dynamical analysis
- **Control theory**: Scan architectures as controllable linear systems

Deeper exploration of these connections could yield new insights and techniques.

---

## 9. Conclusion

The parallel associative scan provides a computational framework for implementing neural computations at scale. While not all canonical computations admit exact scan implementations, approximation strategies and multi-layer architectures can recover much of the expressivity of traditional recurrent neural networks.

The advantages for computational neuroscience are twofold:

1. **Scalability**: O(log T) parallel complexity enables simulation and fitting at biological timescales, opening the door to emergent phenomena and large-scale data.

2. **Interpretability**: Linear dynamics within layers enable principled analysis using spectral methods, impulse responses, and perturbation analysis.

The framework presented here—mapping canonical neural computations to scan-based architectures—provides a foundation for the next generation of neural models: models that can match the scale of modern neural recordings while remaining interpretable enough to yield genuine insight into neural computation.

---

## References

Abeles, M. (1991). Corticonics: Neural circuits of the cerebral cortex. Cambridge University Press.

Adelson, E. H., & Bergen, J. R. (1985). Spatiotemporal energy models for the perception of motion. Journal of the Optical Society of America A, 2(2), 284-299.

Bengio, Y., Simard, P., & Frasconi, P. (1994). Learning long-term dependencies with gradient descent is difficult. IEEE Transactions on Neural Networks, 5(2), 157-166.

Bi, G. Q., & Poo, M. M. (1998). Synaptic modifications in cultured hippocampal neurons: dependence on spike timing, synaptic strength, and postsynaptic cell type. Journal of Neuroscience, 18(24), 10464-10472.

Blelloch, G. E. (1990). Prefix sums and their applications. Technical Report CMU-CS-90-190, Carnegie Mellon University.

Bliss, T. V., & Lømo, T. (1973). Long-lasting potentiation of synaptic transmission in the dentate area of the anaesthetized rabbit following stimulation of the perforant path. Journal of Physiology, 232(2), 331-356.

Brody, C. D., & Hanks, T. D. (2016). Neural underpinnings of the evidence accumulator. Current Opinion in Neurobiology, 37, 149-157.

Buzsáki, G., & Draguhn, A. (2004). Neuronal oscillations in cortical networks. Science, 304(5679), 1926-1929.

Carandini, M., & Heeger, D. J. (2012). Normalization as a canonical neural computation. Nature Reviews Neuroscience, 13(1), 51-62.

Carr, C. E., & Konishi, M. (1990). A circuit for detection of interaural time differences in the brain stem of the barn owl. Journal of Neuroscience, 10(10), 3227-3246.

Chen, R. T., Rubanova, Y., Bettencourt, J., & Duvenaud, D. K. (2018). Neural ordinary differential equations. Advances in Neural Information Processing Systems, 31.

Compte, A., Brunel, N., Goldman-Rakic, P. S., & Wang, X. J. (2000). Synaptic mechanisms and network dynamics underlying spatial working memory in a cortical network model. Cerebral Cortex, 10(9), 910-923.

Douglas, R. J., Koch, C., Mahowald, M., Martin, K. A., & Suarez, H. H. (1995). Recurrent excitation in neocortical circuits. Science, 269(5226), 981-985.

Fairhall, A. L., Lewen, G. D., Bialek, W., & de Ruyter van Steveninck, R. R. (2001). Efficiency and ambiguity in an adaptive neural code. Nature, 412(6849), 787-792.

Fries, P. (2015). Rhythms for cognition: communication through coherence. Neuron, 88(1), 220-235.

Friston, K. (2005). A theory of cortical responses. Philosophical Transactions of the Royal Society B: Biological Sciences, 360(1456), 815-836.

Funahashi, S., Bruce, C. J., & Goldman-Rakic, P. S. (1989). Mnemonic coding of visual space in the monkey's dorsolateral prefrontal cortex. Journal of Neurophysiology, 61(2), 331-349.

Gold, J. I., & Shadlen, M. N. (2007). The neural basis of decision making. Annual Review of Neuroscience, 30, 535-574.

Goldman-Rakic, P. S. (1995). Cellular basis of working memory. Neuron, 14(3), 477-485.

Gu, A., Goel, K., & Ré, C. (2022). Efficiently modeling long sequences with structured state spaces. International Conference on Learning Representations.

Hebb, D. O. (1949). The organization of behavior: A neuropsychological theory. Wiley.

Heeger, D. J. (1992). Normalization of cell responses in cat striate cortex. Visual Neuroscience, 9(2), 181-197.

Hopfield, J. J. (1982). Neural networks and physical systems with emergent collective computational abilities. Proceedings of the National Academy of Sciences, 79(8), 2554-2558.

Hubel, D. H., & Wiesel, T. N. (1962). Receptive fields, binocular interaction and functional architecture in the cat's visual cortex. Journal of Physiology, 160(1), 106-154.

Keller, G. B., & Mrsic-Flogel, T. D. (2018). Predictive processing: a canonical cortical computation. Neuron, 100(2), 424-435.

Kuffler, S. W. (1953). Discharge patterns and functional organization of mammalian retina. Journal of Neurophysiology, 16(1), 37-68.

Lamme, V. A., & Roelfsema, P. R. (2000). The distinct modes of vision offered by feedforward and recurrent processing. Trends in Neurosciences, 23(11), 571-579.

Laughlin, S. (1981). A simple coding procedure enhances a neuron's information capacity. Zeitschrift für Naturforschung C, 36(9-10), 910-912.

Liang, H., Gong, X., Chen, M., Yan, Y., Li, W., & Gilbert, C. D. (2017). Interactions between feedback and lateral connections in the primary visual cortex. Proceedings of the National Academy of Sciences, 114(32), 8637-8642.

Linsley, D., Kim, J., Veerabadran, V., Windolf, C., & Serre, T. (2018). Learning long-range spatial dependencies with horizontal gated recurrent units. Advances in Neural Information Processing Systems, 31.

Lisman, J. E., & Jensen, O. (2013). The theta-gamma neural code. Neuron, 77(6), 1002-1016.

MacDonald, C. J., Lepage, K. Q., Eden, U. T., & Eichenbaum, H. (2011). Hippocampal "time cells" bridge the gap in memory for discontiguous events. Neuron, 71(4), 737-749.

Magee, J. C., & Johnston, D. (1997). A synaptically controlled, associative signal for Hebbian plasticity in hippocampal neurons. Science, 275(5297), 209-213.

Mante, V., Sussillo, D., Shenoy, K. V., & Newsome, W. T. (2013). Context-dependent computation by recurrent dynamics in prefrontal cortex. Nature, 503(7474), 78-84.

Martin, E., & Cundy, C. (2018). Parallelizing linear recurrent neural nets over sequence length. International Conference on Learning Representations.

McAdams, C. J., & Maunsell, J. H. (1999). Effects of attention on orientation-tuning functions of single neurons in macaque cortical area V4. Journal of Neuroscience, 19(1), 431-441.

Miller, K. D. (2016). Canonical computations of cerebral cortex. Current Opinion in Neurobiology, 37, 75-84.

Naka, K. I., & Rushton, W. A. (1966). S-potentials from colour units in the retina of fish (Cyprinidae). Journal of Physiology, 185(3), 536-555.

Pascanu, R., Mikolov, T., & Bengio, Y. (2013). On the difficulty of training recurrent neural networks. International Conference on Machine Learning, 1310-1318.

Pastalkova, E., Itskov, V., Amarasingham, A., & Buzsáki, G. (2008). Internally generated cell assembly sequences in the rat hippocampus. Science, 321(5894), 1322-1327.

Priebe, N. J., & Ferster, D. (2008). Inhibition, spike threshold, and stimulus selectivity in primary visual cortex. Neuron, 57(4), 482-497.

Rao, R. P., & Ballard, D. H. (1999). Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects. Nature Neuroscience, 2(1), 79-87.

Ratcliff, R. (1978). A theory of memory retrieval. Psychological Review, 85(2), 59-108.

Reynolds, J. H., & Heeger, D. J. (2009). The normalization model of attention. Neuron, 61(2), 168-185.

Rodieck, R. W., & Stone, J. (1965). Analysis of receptive fields of cat retinal ganglion cells. Journal of Neurophysiology, 28(5), 832-849.

Salinas, E., & Thier, P. (2000). Gain modulation: a major computational principle of the central nervous system. Neuron, 27(1), 15-21.

Schwartz, O., & Simoncelli, E. P. (2001). Natural signal statistics and sensory gain control. Nature Neuroscience, 4(8), 819-825.

Simoncelli, E. P., & Heeger, D. J. (1998). A model of neuronal responses in visual area MT. Vision Research, 38(5), 743-761.

Srinivasan, M. V., Laughlin, S. B., & Dubs, A. (1982). Predictive coding: a fresh view of inhibition in the retina. Proceedings of the Royal Society of London B, 216(1205), 427-459.

Treue, S., & Martínez-Trujillo, J. C. (1999). Feature-based attention influences motion processing gain in macaque visual cortex. Nature, 399(6736), 575-579.

Tsodyks, M. V., & Markram, H. (1997). The neural code between neocortical pyramidal neurons depends on neurotransmitter release probability. Proceedings of the National Academy of Sciences, 94(2), 719-723.

Usher, M., & McClelland, J. L. (2001). The time course of perceptual choice: the leaky, competing accumulator model. Psychological Review, 108(3), 550-592.

Wang, X. J. (2002). Probabilistic decision making by slow reverberation in cortical circuits. Neuron, 36(5), 955-968.

Weber, A. I., Krishnamurthy, K., & Fairhall, A. L. (2019). Coding principles in adaptation. Annual Review of Vision Science, 5, 427-449.

Wilson, H. R., & Cowan, J. D. (1972). Excitatory and inhibitory interactions in localized populations of model neurons. Biophysical Journal, 12(1), 1-24.
