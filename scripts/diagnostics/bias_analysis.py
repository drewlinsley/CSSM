"""
Pathfinder CL-25 Dataset Bias Analysis
=======================================
Checks whether frequency-domain statistics or simple pixel statistics
can discriminate between label=0 and label=1 without spatial reasoning.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import numpy as np
from scipy import stats
from scipy.ndimage import uniform_filter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ─── 1. Load TFRecords ────────────────────────────────────────────────────────

DATA_DIR = "/oscar/scratch/dlinsley/pathfinder_tfrecords_128/difficulty_25/val/"
tfrecord_files = sorted(tf.io.gfile.glob(os.path.join(DATA_DIR, "*.tfrecord")))
print(f"Found {len(tfrecord_files)} TFRecord files")

feature_spec = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64),
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'channels': tf.io.FixedLenFeature([], tf.int64),
}

images_list = []
labels_list = []

for f in tfrecord_files:
    dataset = tf.data.TFRecordDataset(f)
    for raw_record in dataset:
        example = tf.io.parse_single_example(raw_record, feature_spec)
        h = example['height'].numpy()
        w = example['width'].numpy()
        c = example['channels'].numpy()
        img = tf.io.decode_raw(example['image'], tf.float32).numpy().reshape(h, w, c)
        label = example['label'].numpy()
        images_list.append(img)
        labels_list.append(label)
    print(f"  Loaded {f}, total so far: {len(images_list)}")

images = np.array(images_list)
labels = np.array(labels_list)
print(f"\nTotal images: {len(images)}, shape: {images[0].shape}")
print(f"Label distribution: 0={np.sum(labels==0)}, 1={np.sum(labels==1)}")
print(f"Image value range: [{images.min():.3f}, {images.max():.3f}]")
print(f"Image mean: {images.mean():.4f}, std: {images.std():.4f}")

# ─── 2. Check ordering bias ──────────────────────────────────────────────────

print("\n" + "="*70)
print("CHECK: Label ordering in TFRecords")
print("="*70)
# Show first 50 and last 50 labels
print(f"First 50 labels: {labels[:50].tolist()}")
print(f"Last 50 labels:  {labels[-50:].tolist()}")
# Check runs
changes = np.sum(np.diff(labels) != 0)
print(f"Number of label changes in sequence: {changes}")
print(f"If random, expect ~{len(labels)//2} changes. Actual: {changes}")
if changes < len(labels) // 4:
    print("WARNING: Labels are NOT well-shuffled!")
else:
    print("Labels appear reasonably shuffled.")

# ─── 3. Frequency-domain analysis ────────────────────────────────────────────

print("\n" + "="*70)
print("FREQUENCY-DOMAIN ANALYSIS")
print("="*70)

def compute_fft_features(img):
    """Compute FFT-based features for a single image (H, W, C)."""
    # Average across channels for grayscale-like analysis
    gray = img.mean(axis=-1)

    # 2D FFT
    fft2 = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft2)
    power_spectrum = np.abs(fft_shift) ** 2

    # Radial power spectrum
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    R = np.sqrt((X - cx)**2 + (Y - cy)**2).astype(int)
    max_r = min(cy, cx)
    radial_power = np.zeros(max_r)
    for r in range(max_r):
        mask = R == r
        if mask.any():
            radial_power[r] = power_spectrum[mask].mean()

    # Low vs high frequency power
    freq_threshold = int(0.10 * max_r)  # 10% of max freq
    low_freq_power = radial_power[:freq_threshold].sum()
    high_freq_power = radial_power[freq_threshold:].sum()
    total_power = radial_power.sum()
    low_frac = low_freq_power / (total_power + 1e-10)
    high_frac = high_freq_power / (total_power + 1e-10)

    return {
        'power_spectrum': power_spectrum,
        'radial_power': radial_power,
        'low_freq_power': low_freq_power,
        'high_freq_power': high_freq_power,
        'low_frac': low_frac,
        'high_frac': high_frac,
        'total_power': total_power,
        'fft_magnitude': np.abs(fft_shift),
    }

# Compute FFT features for all images
print("Computing FFT features for all images...")
fft_results = [compute_fft_features(img) for img in images]

# Separate by class
idx0 = labels == 0
idx1 = labels == 1

# Compare radial power spectra
radial_0 = np.array([r['radial_power'] for r in np.array(fft_results)[idx0]])
radial_1 = np.array([r['radial_power'] for r in np.array(fft_results)[idx1]])

print(f"\nRadial power spectrum shape: {radial_0.shape}")
print("\nPer-frequency-bin t-tests (class 0 vs class 1):")
print(f"{'Bin':>4} {'Mean_0':>12} {'Mean_1':>12} {'t-stat':>10} {'p-value':>12} {'Sig':>5}")
print("-" * 60)
n_bins = radial_0.shape[1]
significant_bins = 0
for i in range(n_bins):
    t_stat, p_val = stats.ttest_ind(radial_0[:, i], radial_1[:, i])
    sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else ""))
    if p_val < 0.05:
        significant_bins += 1
    if i < 15 or p_val < 0.01:  # Print first 15 bins and any significant ones
        print(f"{i:4d} {radial_0[:, i].mean():12.2f} {radial_1[:, i].mean():12.2f} {t_stat:10.3f} {p_val:12.6f} {sig:>5}")

print(f"\nSignificant bins (p<0.05): {significant_bins}/{n_bins} = {significant_bins/n_bins*100:.1f}%")
print(f"Expected by chance at 5%: {n_bins * 0.05:.1f} bins")

# Compare low/high frequency fractions
low_frac_0 = np.array([r['low_frac'] for r in np.array(fft_results)[idx0]])
low_frac_1 = np.array([r['low_frac'] for r in np.array(fft_results)[idx1]])
high_frac_0 = np.array([r['high_frac'] for r in np.array(fft_results)[idx0]])
high_frac_1 = np.array([r['high_frac'] for r in np.array(fft_results)[idx1]])
total_power_0 = np.array([r['total_power'] for r in np.array(fft_results)[idx0]])
total_power_1 = np.array([r['total_power'] for r in np.array(fft_results)[idx1]])

print("\n--- Aggregate frequency statistics ---")
for name, arr0, arr1 in [
    ("Low freq fraction (<10%)", low_frac_0, low_frac_1),
    ("High freq fraction (>10%)", high_frac_0, high_frac_1),
    ("Total power", total_power_0, total_power_1),
]:
    t_stat, p_val = stats.ttest_ind(arr0, arr1)
    d = (arr0.mean() - arr1.mean()) / np.sqrt((arr0.std()**2 + arr1.std()**2) / 2)  # Cohen's d
    print(f"{name}:")
    print(f"  Class 0: {arr0.mean():.6f} +/- {arr0.std():.6f}")
    print(f"  Class 1: {arr1.mean():.6f} +/- {arr1.std():.6f}")
    print(f"  t={t_stat:.3f}, p={p_val:.6f}, Cohen's d={d:.4f}")

# ─── 4. Spatial autocorrelation ──────────────────────────────────────────────

print("\n" + "="*70)
print("SPATIAL AUTOCORRELATION")
print("="*70)

def spatial_autocorr(img, max_lag=20):
    """Compute spatial autocorrelation at various lags."""
    gray = img.mean(axis=-1)
    gray_centered = gray - gray.mean()
    var = gray_centered.var()
    if var < 1e-10:
        return np.zeros(max_lag)

    autocorrs = []
    for lag in range(1, max_lag + 1):
        # Horizontal autocorrelation
        h_corr = np.mean(gray_centered[:, :-lag] * gray_centered[:, lag:]) / var
        # Vertical autocorrelation
        v_corr = np.mean(gray_centered[:-lag, :] * gray_centered[lag:, :]) / var
        autocorrs.append((h_corr + v_corr) / 2)
    return np.array(autocorrs)

print("Computing spatial autocorrelation...")
autocorrs = np.array([spatial_autocorr(img) for img in images])
autocorr_0 = autocorrs[idx0]
autocorr_1 = autocorrs[idx1]

print(f"\n{'Lag':>4} {'Mean_0':>10} {'Mean_1':>10} {'t-stat':>10} {'p-value':>12} {'Cohen_d':>10}")
print("-" * 60)
for lag in range(autocorrs.shape[1]):
    t_stat, p_val = stats.ttest_ind(autocorr_0[:, lag], autocorr_1[:, lag])
    d = (autocorr_0[:, lag].mean() - autocorr_1[:, lag].mean()) / np.sqrt(
        (autocorr_0[:, lag].std()**2 + autocorr_1[:, lag].std()**2) / 2)
    sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else ""))
    print(f"{lag+1:4d} {autocorr_0[:, lag].mean():10.6f} {autocorr_1[:, lag].mean():10.6f} "
          f"{t_stat:10.3f} {p_val:12.6f} {d:10.4f} {sig}")

# ─── 5. Simple statistics per image ─────────────────────────────────────────

print("\n" + "="*70)
print("SIMPLE IMAGE STATISTICS (per class)")
print("="*70)

for ch_name, ch_idx in [("All channels", None), ("Channel 0", 0), ("Channel 1", 1), ("Channel 2", 2)]:
    if ch_idx is None:
        vals_0 = images[idx0].reshape(idx0.sum(), -1)
        vals_1 = images[idx1].reshape(idx1.sum(), -1)
    else:
        vals_0 = images[idx0, :, :, ch_idx].reshape(idx0.sum(), -1)
        vals_1 = images[idx1, :, :, ch_idx].reshape(idx1.sum(), -1)

    means_0 = vals_0.mean(axis=1)
    means_1 = vals_1.mean(axis=1)
    stds_0 = vals_0.std(axis=1)
    stds_1 = vals_1.std(axis=1)

    t_mean, p_mean = stats.ttest_ind(means_0, means_1)
    t_std, p_std = stats.ttest_ind(stds_0, stds_1)

    print(f"\n{ch_name}:")
    print(f"  Mean - Class 0: {means_0.mean():.6f}+/-{means_0.std():.6f}, "
          f"Class 1: {means_1.mean():.6f}+/-{means_1.std():.6f}, "
          f"t={t_mean:.3f}, p={p_mean:.6f}")
    print(f"  Std  - Class 0: {stds_0.mean():.6f}+/-{stds_0.std():.6f}, "
          f"Class 1: {stds_1.mean():.6f}+/-{stds_1.std():.6f}, "
          f"t={t_std:.3f}, p={p_std:.6f}")

# ─── 6. Linear classifiers ──────────────────────────────────────────────────

print("\n" + "="*70)
print("LINEAR CLASSIFIER EXPERIMENTS")
print("="*70)

def eval_linear_classifier(X, y, name, max_iter=5000):
    """Train logistic regression, report accuracy."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Try multiple regularization strengths
    best_acc = 0
    best_C = None
    for C in [0.001, 0.01, 0.1, 1.0, 10.0]:
        clf = LogisticRegression(C=C, max_iter=max_iter, solver='lbfgs', random_state=42)
        clf.fit(X_train_s, y_train)
        train_acc = accuracy_score(y_train, clf.predict(X_train_s))
        test_acc = accuracy_score(y_test, clf.predict(X_test_s))
        if test_acc > best_acc:
            best_acc = test_acc
            best_C = C
            best_train_acc = train_acc

    print(f"\n{name}:")
    print(f"  Feature dim: {X.shape[1]}")
    print(f"  Best C={best_C}: Train acc={best_train_acc:.4f}, Test acc={best_acc:.4f}")
    print(f"  Chance level: {max(y.mean(), 1-y.mean()):.4f}")
    return best_acc

# (a) Flattened raw pixels
print("\n--- (a) Raw pixel classifier ---")
X_pixels = images.reshape(len(images), -1)
acc_pixels = eval_linear_classifier(X_pixels, labels, "Raw Pixels")

# (b) 2D FFT magnitude spectrum (flattened)
print("\n--- (b) FFT magnitude spectrum classifier ---")
X_fft = np.array([np.abs(np.fft.fftshift(np.fft.fft2(img.mean(axis=-1)))).flatten()
                   for img in images])
acc_fft = eval_linear_classifier(X_fft, labels, "FFT Magnitude (full)")

# (c) Radial power spectrum
print("\n--- (c) Radial power spectrum classifier ---")
X_radial = np.array([r['radial_power'] for r in fft_results])
acc_radial = eval_linear_classifier(X_radial, labels, "Radial Power Spectrum")

# (d) Channel moments (mean, std, skew, kurtosis per channel)
print("\n--- (d) Channel moments classifier ---")
moments_list = []
for img in images:
    feats = []
    for ch in range(img.shape[-1]):
        ch_data = img[:, :, ch].flatten()
        feats.extend([
            ch_data.mean(),
            ch_data.std(),
            float(stats.skew(ch_data)),
            float(stats.kurtosis(ch_data)),
        ])
    moments_list.append(feats)
X_moments = np.array(moments_list)
acc_moments = eval_linear_classifier(X_moments, labels, "Channel Moments (mean/std/skew/kurt)")

# (e) BONUS: Combined low-dimensional features
print("\n--- (e) Combined low-dim features ---")
X_combined = np.hstack([
    X_moments,
    X_radial,
    autocorrs,
])
acc_combined = eval_linear_classifier(X_combined, labels, "Moments + Radial + Autocorr")

# (f) BONUS: Per-channel FFT magnitude
print("\n--- (f) Per-channel FFT magnitude ---")
X_fft_ch = []
for img in images:
    feats = []
    for ch in range(img.shape[-1]):
        fft_ch = np.abs(np.fft.fftshift(np.fft.fft2(img[:, :, ch])))
        feats.append(fft_ch.flatten())
    X_fft_ch.append(np.concatenate(feats))
X_fft_ch = np.array(X_fft_ch)
acc_fft_ch = eval_linear_classifier(X_fft_ch, labels, "Per-channel FFT Magnitude")

# ─── 7. Summary ──────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("SUMMARY OF LINEAR CLASSIFIER RESULTS")
print("="*70)
print(f"{'Method':<45} {'Test Accuracy':>15}")
print("-" * 62)
print(f"{'Chance level':<45} {max(labels.mean(), 1-labels.mean()):>15.4f}")
print(f"{'(a) Raw pixels':<45} {acc_pixels:>15.4f}")
print(f"{'(b) FFT magnitude (full, grayscale)':<45} {acc_fft:>15.4f}")
print(f"{'(c) Radial power spectrum (1D, ~64 bins)':<45} {acc_radial:>15.4f}")
print(f"{'(d) Channel moments (12 features)':<45} {acc_moments:>15.4f}")
print(f"{'(e) Combined low-dim features':<45} {acc_combined:>15.4f}")
print(f"{'(f) Per-channel FFT magnitude':<45} {acc_fft_ch:>15.4f}")
print()
print("If any of these are substantially above chance (50%), there is a")
print("dataset bias exploitable without spatial reasoning.")
print("="*70)
