# Methods

All experiments are implemented in JAX/Flax in this repository
(`main.py`, `src/models/simple_cssm.py`, `src/models/cssm.py`) and reproduced
end-to-end by the sweep scripts under `scripts/`. We refer to the
no-gate spectral CSSM (`src/models/cssm.py:6101`, registry key `'no_gate'`)
as **sCSSM**, and to the PathTracker–15-distractor benchmark (Linsley
et al.; `--dataset girik`) as **PathTracker** for brevity.

## 1. Architecture

### 1.1 sCSSM (NoGateCSSM)

The sCSSM block (a) embeds the input frame stack in a learned channel
basis with a small convolutional stem, (b) maps each frame to the
spatial Fourier domain, (c) runs a per-(channel, spatial-frequency)
linear recurrence in parallel along time, and (d) maps back to the
spatial domain, with a single linear out-projection. The full per-block
computation is:

1. Spatial real-FFT: `Û_t = rfft2(x_t) ∈ ℂ^{C×H×(W/2+1)}` for each frame t.
2. Spectral kernel: a learned real kernel `k ∈ ℝ^{C×K×K}` (default
   `K = 11`) is zero-padded to (H, W) and transformed to the spectral
   domain, `K̂ = rfft2(pad_{HW}(k))`.
3. Stability squash: `K̂ ← K̂ · ρ / (1 + |K̂|)` with `ρ = 0.999`
   (`_stable_spectral_magnitude`, `cssm.py:29`). This preserves phase
   and bounds the per-frequency magnitude strictly below 1, guaranteeing
   geometric decay of the linear recurrence at every spatial frequency.
4. Per-frequency scalar recurrence
   `Ŝ_t = K̂ ⊙ Ŝ_{t-1} + Û_t`,
   computed with `jax.lax.associative_scan` along the temporal axis
   using the closed-form combiner
   `(K_i, U_i) ∘ (K_j, U_j) = (K_j K_i, K_j U_i + U_j)`. Because every
   array involved is per-frequency scalar in C, the carry shape never
   grows during scan composition, giving total work O(T·C·H·(W/2+1))
   in `complex64`.
5. Inverse FFT and out-projection: `S_t = irfft2(Ŝ_t)`, followed by a
   single `Dense(C)` (`out_proj`) that produces a residual added to
   the block input.

Two algorithmic facts about this design are validated by the
`benchmarks/bench_image_timing.py` Mechanism Validation panel
(H=W=C=32, T up to 1024): replacing the spatial FFT with a real-space
circular conv (a strictly mathematically equivalent recurrence) is
~67× slower as a sequential `lax.scan` and ~460× slower as a
parallel-scan with image-padded kernels (`NoGateRealSpaceSeqCSSM`,
`NoGateRealSpaceParallelCSSM` in `cssm.py`). The pre-padded variant
is an upper bound on the cost of a true growing-kernel parallel scan;
the sequential variant is the fair non-spectral baseline.

### 1.2 SimpleCSSM wrapper

All path-domain runs use the `SimpleCSSM` wrapper
(`src/models/simple_cssm.py`). The forward pass is:

- **Stem.** For PathTracker and Pathfinder we use `--stem_mode pathtracker`
  (`simple_cssm.py:258`), which is a single 1×1 `Conv(embed_dim)` — no
  spatial downsampling, since both tasks demand pixel-resolution
  contour reasoning. ImageNet uses the default stem (N×{3×3 Conv → norm
  → activation → 2×2 max-pool}, `--stem_layers 2`), giving 4×
  spatial downsampling before the CSSM blocks. `stem_norm_order=post`
  throughout.
- **Position embeddings.** Multi-axis RoPE (`--pos_embed mrope`) on
  the (T, H, W) axes, as defined in `simple_cssm.py`.
- **CSSM blocks.** `--depth 1` for all main results. Block output is
  added residually to the input. No inter-block MLP at depth=1.
- **Readout.** `--frame_readout last` selects the final temporal frame;
  `--pool_type max` then performs spatial max-pooling to a single
  embedding; a final `LayerNorm` and `Dense(num_classes)` produce
  logits. Per-frame readout (`frame_readout='all'`, spatiotemporal max
  pool) is used for the timestep ablations as a control.
- **Norm.** `--norm_type global_layer` (LayerNorm computed jointly
  over T, H, W, C — i.e., one (μ, σ) per sample) is the default for
  the path-domain experiments.

Baselines are accessed via `--cssm` and registered in
`CSSM_REGISTRY` (`simple_cssm.py:22-60`); the variants used in this
paper are `no_gate` (sCSSM), `gdn`, `gdn_seq`, `mamba2_seq`,
`conv_ssm`, `s4nd`, `s4nd_full`, `convs5`, `transformer`
(spatiotemporal_attn) and `spatial_attn`. The s4nd / s4nd_full /
convs5 ports are described in `cssm.py` and were added so that the
PathTracker/Pathfinder comparisons also include faithful state-space
baselines.

## 2. Training

We use a single training entry point (`main.py`) for all path-domain
experiments. ImageNet-1k uses `src/training/train_imagenet.py`, which
implements the same optimizer/schedule but adds the TIMM augmentation
recipe.

### 2.1 Optimizer and schedule

- AdamW (`optax.adamw`, `main.py:265`) with weight decay applied to
  all weights except norms and biases via the standard optax mask.
- Linear warmup followed by cosine decay
  (`optax.warmup_cosine_decay_schedule`, `main.py:246-249`). The
  warmup length is `min(500, total_steps − 1)` (`main.py:1154`); the
  cosine end value is 1% of the peak learning rate.
- Global gradient norm clipping at 1.0 (`--grad_clip 1.0`).
- Numerical safety wrapper `optax.apply_if_finite(..., max_consecutive_errors=5)`
  (`main.py:270`), which skips updates with non-finite gradients.

### 2.2 Loss

Softmax cross-entropy (`optax.softmax_cross_entropy`, `main.py:404`)
on one-hot targets. Label smoothing is 0 for path-domain tasks
(binary classification) and 0.1 for ImageNet under the TIMM recipe.

### 2.3 Per-task hyperparameters

All values below are taken verbatim from the sweep scripts.

| | PathTracker (`sweep_15dist.sh`) | Pathfinder-14 (`sweep_pathfinder.sh`) | Pathfinder-25 (`sweep_pathfinder_25.sh`) | ImageNet-1k (`sweep_imagenet_edge.sh`) |
|---|---|---|---|---|
| Image size | 32×32 | 128×128 | 128×128 | 224×224 |
| Frames T | 32 | 8 (static image repeated) | {1, 10} | 8 |
| Epochs | 60 (transformers) / 120 (all SSMs) | 60 (transformers) / 120 (all SSMs) | 120 | 300 |
| Batch size | 128 (64 for `gdn_seq`/`mamba2_seq`) | same | 128 | 512 (auto-halved on OOM) |
| Peak LR | 1e-3 | 1e-3 | 1e-3 | 1e-3 |
| Weight decay | 1e-4 | 1e-4 | 1e-4 | 5e-2 |
| Grad clip | 1.0 | 1.0 | 1.0 | 1.0 |
| Drop-path | 0.0 (0.1 for depth-3 transformers) | same | 0.0 | 0.1 |
| Label smoothing | 0.0 | 0.0 | 0.0 | 0.1 |
| Stem | `pathtracker` (1×1 Conv) | `pathtracker` (1×1 Conv) | `pathtracker` (1×1 Conv) | default (2-layer 3×3 + pool) |
| Position embed | mrope | mrope | mrope | learned (TIMM) |
| Norm | global\_layer | global\_layer | global\_layer | global\_layer |
| Pool | max | max | max | mean |
| Readout | last frame | last frame | last frame | mean (head\_pool) |
| Augmentation | none (TFRecord native) | none | none | RandAugment-3aug, Mixup α=0.8, CutMix α=1.0, RandomErasing p=0.25 |
| EMA | – | – | – | 0.9998 (TIMM) |

The sweep scripts wrap each run in `run_with_retry`, which catches
OOM exit codes and halves the batch size down to a 4–8 floor before
giving up; this is the only source of cross-machine batch-size
variation in our results.

### 2.4 Datasets

- **PathTracker (15-distractor variant)** — `--dataset girik`
  (`main.py:995`), TFRecords under
  `/oscar/scratch/dlinsley/15_dist`, project `CSSM_15dist`. Binary
  classification: connected vs. disconnected start/end dot. Native
  resolution 32×32, 32 frames per video. Loaded via
  `get_girik_tfrecord_loader` (`main.py:1366`).
- **PathTracker–restyled** — `/oscar/scratch/dlinsley/pathtracker_restyled_32f_tfrecords`,
  project `CSSM_pathtracker_restyled_32f`. Same task, restyled
  stimuli; we sweep the same model grid via
  `sweep_pathtracker_restyled_32f.sh` (57 runs).
- **Pathfinder-14 / Pathfinder-25** — `--dataset pathfinder`,
  `--pathfinder_difficulty {14, 25}` (`main.py:890, 970`), TFRecords
  under `/oscar/scratch/dlinsley/pathfinder_tfrecords_128`, project
  `CSSM_pathfinder`. Binary connected-contour task, 128×128 pixels,
  static image. We repeat the static frame T times to drive the
  temporal recurrence (`--seq_len T`); transformers and seq-style
  baselines (`mamba2_seq`, `gdn_seq`) use `--seq_len 1` because they
  flatten spatial tokens directly.
- **ImageNet-1k** — TFRecords under
  `/oscar/scratch/dlinsley/imagenet_tfrecords`. Standard 1000-class
  classification. The TFRecord loader uses 4 parallel readers and
  prefetches 2 batches (`--tf_parallel_reads 4 --tf_prefetch_batches 2`).

### 2.5 Compute environment

All training runs were performed on a single Brown CCV node with
**2× NVIDIA B200 GPUs** (180 GB HBM3e each, driver 575.57.08).
Sweeps are sequential — at most one run per GPU at a time — because
the OOM-retry harness (`scripts/sweep_*.sh`) needs an empty GPU to
attempt a halved batch size cleanly. JAX compilation cache is shared
under `~/.cache/jax_compilation_cache`. We use single-GPU training
for all runs except ImageNet, which uses pmap across both B200s.

## 3. Experiment grid and compute budget

For each task, every architecture is swept over a small grid of
`(embed_dim, kernel_size, state_dim, depth)` factors specified in the
header comment of the corresponding `scripts/sweep_*.sh` script.
Total run counts are taken directly from the `TOTAL=…` line of each
script. Wall-clock per run is read from the
`Started:`/`SUCCESS:` timestamps in the captured `sweep_*.out`
logs.

| Sweep | Script | Runs | Median run time (B200) | GPU-hours |
|---|---|---:|---:|---:|
| PathTracker (15-dist, 32 frames) | `sweep_15dist.sh` | 77 | ~25 min | ~32 |
| PathTracker–restyled (32 frames) | `sweep_pathtracker_restyled_32f.sh` | 57 | ~25 min | ~24 |
| Pathfinder-14 (128 px, T=8) | `sweep_pathfinder.sh` | 94 | ~80 min (sCSSM 39 min, GDN ~3.5 h, NoFFT ~50 min) | ~125 |
| Pathfinder-25 (128 px) | `sweep_pathfinder_25.sh` | 10 | ~100 min | ~17 |
| Timestep ablation T=1…20 (PF-14) | `sweep_nogate_timesteps.sh` | 20 | ~25 min/T (T=1: 22 min, T=6: 225 min) | ~88 |
| Timestep ablation T={25, 50, 100} (PF-25) | `sweep_nogate_timesteps_25.sh` | 3 | ~25 min/T | ~73 |
| S4ND-full + ConvS5 controls | `sweep_s4nd_full.sh` + `sweep_s4nd_full_15dist.sh` + `sweep_s4nd_convs5.sh` | 28 | ~60 min | ~28 |
| ImageNet learned-t supervised (30 ep) | `learned_t_imagenet_30ep.sh` | 1 | ~5 h | ~5 |
| ImageNet LeJEPA (100 ep, recurrent SSL) | `lejepa_imagenet_scratch.sh` | 1 | ~16 h | ~16 |
| ImageNet edge sweep (300 ep, 6 models) | `sweep_imagenet_edge.sh` | 6 | ~40 h (2× B200, pmap) | ~240 (effective 480 single-GPU) |
| **Total (this paper's main results)** | | **~297** | | **≈ 650 GPU-h ≈ 27 B200-days** |

These numbers are within ~10% of the wall-clock observed in the
captured sweep logs (`sweep_pathfinder_gpu4004.out`,
`sweep_15dist_gpu4004.out`, `sweep_s4nd_convs5_gpu4007.out`); the
spread within a sweep is dominated by the model family
(GDN/Mamba2-seq variants run 3–5× longer than sCSSM at the same
embed/depth because their per-step cost grows with `delta_key_dim`
and `state_dim`). Compile-time overhead is amortized by the JAX
compilation cache and contributes <2 minutes per run after the first
within a sweep.

## 4. Evaluation

For all sweeps we report **best validation accuracy across the run**,
selected by `--save_best_only` (`main.py`), evaluated every epoch on
the test/val split of each dataset (`main.py:1511-1585` for
Pathfinder, `1579-1585` for PathTracker which uses the test split
because it has no held-out val split). Metrics are pushed to W&B
projects `CSSM_15dist`, `CSSM_pathfinder`,
`CSSM_pathtracker_restyled_32f`, and `cssm-imagenet-edge`; figures in
the paper are produced from W&B exports by the plotting scripts in
`scripts/plot_*.py`. Variability across seeds is dominated by the
embed-dim/state-dim grid we sweep, which we report directly in the
results figures rather than running additional seed replicas.

## 5. Reproducing the results

Every result in the paper can be reproduced with the corresponding
sweep script:

```bash
# PathTracker (15-distractor)
bash scripts/sweep_15dist.sh          2>&1 | tee -a sweep_15dist_$(hostname).out

# Pathfinder-14
bash scripts/sweep_pathfinder.sh      2>&1 | tee -a sweep_pathfinder_$(hostname).out

# Pathfinder-25 + timestep ablation
bash scripts/sweep_pathfinder_25.sh   2>&1 | tee -a sweep_pathfinder_25_$(hostname).out
bash scripts/sweep_nogate_timesteps.sh    2>&1 | tee -a sweep_nogate_ts_$(hostname).out
bash scripts/sweep_nogate_timesteps_25.sh 2>&1 | tee -a sweep_nogate_ts_25_$(hostname).out

# ImageNet
bash scripts/sweep_imagenet_edge.sh   2>&1 | tee -a sweep_imagenet_$(hostname).out
bash scripts/learned_t_imagenet_30ep.sh
bash scripts/lejepa_imagenet_scratch.sh
```

Each sweep maintains a per-script progress log (`sweep_*_progress.log`)
and is fully resumable: re-running the script after a node failure
will skip completed runs and retry only the failed/incomplete ones.
