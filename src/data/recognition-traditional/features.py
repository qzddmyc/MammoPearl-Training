"""Hand-crafted feature extraction for mammography patches.

Extracts and concatenates:
  * GLCM texture features  (contrast, correlation, energy, entropy)
  * LBP histogram
  * Wavelet subband statistics
  * Gabor filter response statistics
  * Statistical moments  (mean, std, skewness, kurtosis, IQR)

All functions expect a float32 grayscale patch in [0, 1] of shape
(PATCH_SIZE, PATCH_SIZE) as returned by preprocessing.preprocess_patch().
"""

from __future__ import annotations

import warnings

import cv2
import numpy as np
import pywt
from scipy.stats import kurtosis, skew
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from tqdm import tqdm

from config import (
    GABOR_FREQUENCIES,
    GABOR_THETAS,
    GLCM_ANGLES,
    GLCM_DISTANCES,
    LBP_N_POINTS,
    LBP_RADIUS,
    WAVELET,
    WAVELET_LEVEL,
)


# ---------------------------------------------------------------------------
# GLCM features
# ---------------------------------------------------------------------------

def _glcm_features(patch_norm: np.ndarray) -> np.ndarray:
    """Compute GLCM-based texture descriptors.

    Quantise to 32 levels to reduce compute while retaining discriminability.
    """
    levels = 32
    img_q = (patch_norm * (levels - 1)).clip(0, levels - 1).astype(np.uint8)

    # graycomatrix expects integer pixel values
    glcm = graycomatrix(
        img_q,
        distances=GLCM_DISTANCES,
        angles=GLCM_ANGLES,
        levels=levels,
        symmetric=True,
        normed=True,
    )  # shape: (levels, levels, n_dist, n_angle)

    props = ("contrast", "correlation", "energy", "homogeneity")
    feats = []
    for prop in props:
        vals = graycoprops(glcm, prop)  # shape: (n_dist, n_angle)
        feats.extend(vals.ravel().tolist())

    # Entropy of the GLCM
    glcm_sum = glcm.sum(axis=(0, 1), keepdims=True)
    p = np.where(glcm_sum > 0, glcm / (glcm_sum + 1e-12), 0.0)
    entropy = -np.sum(p * np.log2(p + 1e-12), axis=(0, 1))  # (n_dist, n_angle)
    feats.extend(entropy.ravel().tolist())

    return np.array(feats, dtype=np.float32)


# ---------------------------------------------------------------------------
# LBP features
# ---------------------------------------------------------------------------

def _lbp_features(patch_norm: np.ndarray) -> np.ndarray:
    """Compute a normalised LBP histogram."""
    img_u8 = (patch_norm * 255).clip(0, 255).astype(np.uint8)
    lbp = local_binary_pattern(img_u8, LBP_N_POINTS, LBP_RADIUS, method="uniform")
    n_bins = LBP_N_POINTS + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist.astype(np.float32)


# ---------------------------------------------------------------------------
# Wavelet features
# ---------------------------------------------------------------------------

def _wavelet_features(patch_norm: np.ndarray) -> np.ndarray:
    """Compute statistics of wavelet high-frequency subbands."""
    coeffs = pywt.wavedec2(patch_norm, WAVELET, level=WAVELET_LEVEL)
    feats = []
    # Skip the approximation subband (index 0); use detail subbands
    for level_coeffs in coeffs[1:]:
        for subband in level_coeffs:
            arr = subband.ravel()
            feats.extend([
                float(np.mean(np.abs(arr))),
                float(np.std(arr)),
                float(np.sum(arr ** 2)),   # energy
            ])
    return np.array(feats, dtype=np.float32)


# ---------------------------------------------------------------------------
# Gabor features
# ---------------------------------------------------------------------------

def _gabor_features(patch_norm: np.ndarray) -> np.ndarray:
    """Apply Gabor filters and collect response statistics."""
    img_f32 = patch_norm.astype(np.float32)
    feats = []
    for freq in GABOR_FREQUENCIES:
        for theta in GABOR_THETAS:
            ksize = 31
            sigma = 3.0 / (2 * np.pi * freq + 1e-6)
            sigma = float(np.clip(sigma, 1.0, 10.0))
            kernel_real = cv2.getGaborKernel(
                (ksize, ksize), sigma, theta, 1.0 / freq, 0.5, 0, ktype=cv2.CV_32F
            )
            kernel_imag = cv2.getGaborKernel(
                (ksize, ksize), sigma, theta, 1.0 / freq, 0.5, np.pi / 2, ktype=cv2.CV_32F
            )
            resp_real = cv2.filter2D(img_f32, cv2.CV_32F, kernel_real)
            resp_imag = cv2.filter2D(img_f32, cv2.CV_32F, kernel_imag)
            magnitude = np.sqrt(resp_real ** 2 + resp_imag ** 2)
            feats.extend([
                float(np.mean(magnitude)),
                float(np.std(magnitude)),
            ])
    return np.array(feats, dtype=np.float32)


# ---------------------------------------------------------------------------
# Statistical features
# ---------------------------------------------------------------------------

def _statistical_features(patch_norm: np.ndarray) -> np.ndarray:
    """Compute first-order statistical moments of pixel intensities."""
    arr = patch_norm.ravel().astype(np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sk = float(skew(arr))
        ku = float(kurtosis(arr))
    q25, q75 = np.percentile(arr, [25, 75])
    return np.array(
        [float(arr.mean()), float(arr.std()), sk, ku, float(q75 - q25)],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Extra features for specific finding types
# ---------------------------------------------------------------------------

def _calcification_features(patch_norm: np.ndarray) -> np.ndarray:
    """Laplacian-based high-frequency features favourable for calcifications."""
    img_u8 = (patch_norm * 255).clip(0, 255).astype(np.uint8)
    lap = cv2.Laplacian(img_u8, cv2.CV_32F)
    arr = lap.ravel()
    return np.array(
        [float(np.mean(np.abs(arr))), float(np.std(arr)), float(np.max(np.abs(arr)))],
        dtype=np.float32,
    )


def _mass_shape_features(patch_norm: np.ndarray) -> np.ndarray:
    """Coarse shape features (roundness, compactness) for mass-type findings."""
    img_u8 = (patch_norm * 255).clip(0, 255).astype(np.uint8)
    _, binary = cv2.threshold(img_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(3, dtype=np.float32)
    cnt = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    perimeter = float(cv2.arcLength(cnt, True))
    circularity = (4 * np.pi * area / (perimeter ** 2 + 1e-6)) if perimeter > 0 else 0.0
    hull_area = float(cv2.contourArea(cv2.convexHull(cnt)))
    solidity = area / (hull_area + 1e-6)
    density = float(np.mean(img_u8[binary > 0])) / 255.0 if area > 0 else 0.0
    return np.array([circularity, solidity, density], dtype=np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_features(patch_norm: np.ndarray, extended: bool = False) -> np.ndarray:
    """Extract a concatenated feature vector from a preprocessed patch.

    Parameters
    ----------
    patch_norm:
        Float32 grayscale patch in [0, 1], shape (PATCH_SIZE, PATCH_SIZE).
    extended:
        If True, also append calcification and mass shape features
        (used for Stage-2 training).

    Returns
    -------
    np.ndarray of float32, 1-D feature vector.
    """
    parts = [
        _glcm_features(patch_norm),
        _lbp_features(patch_norm),
        _wavelet_features(patch_norm),
        _gabor_features(patch_norm),
        _statistical_features(patch_norm),
    ]
    if extended:
        parts.append(_calcification_features(patch_norm))
        parts.append(_mass_shape_features(patch_norm))

    return np.concatenate(parts).astype(np.float32)


def extract_features_batch(
    patches_norm: list[np.ndarray],
    extended: bool = False,
    verbose: bool = True,
) -> np.ndarray:
    """Extract features for a list of preprocessed patches.

    Returns an array of shape (N, D).
    """
    result = []
    desc = "[features] extract(ext)" if extended else "[features] extract"
    it = tqdm(patches_norm, desc=desc, unit="patch", disable=not verbose)
    for p in it:
        result.append(extract_features(p, extended=extended))
    return np.stack(result, axis=0)
