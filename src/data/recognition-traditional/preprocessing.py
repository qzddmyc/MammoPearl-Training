"""Image preprocessing utilities for the traditional classification pipeline.

Provides CLAHE enhancement, image normalisation, and Frangi blob filtering.
"""

from __future__ import annotations

import cv2
import numpy as np

from config import CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID, PATCH_SIZE


def to_gray(img: np.ndarray) -> np.ndarray:
    """Convert an image array to 8-bit grayscale regardless of input format."""
    if img.ndim == 2:
        gray = img
    elif img.ndim == 3 and img.shape[2] >= 3:
        gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gray


def apply_clahe(gray: np.ndarray) -> np.ndarray:
    """Apply CLAHE to a grayscale image to enhance local contrast."""
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID,
    )
    return clahe.apply(gray)


def normalize_image(gray: np.ndarray) -> np.ndarray:
    """Normalise pixel values to the [0, 1] float range."""
    arr = gray.astype(np.float32)
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    return arr


def frangi_filter(gray_norm: np.ndarray) -> np.ndarray:
    """Apply Frangi blob-enhancement filter.

    Uses a pure-numpy approximation of the Frangi filter (second-order
    Hessian eigenvalues) to enhance rounded blob-like structures (masses)
    and suppress linear backgrounds.  Returns a float32 array in [0, 1].

    Parameters
    ----------
    gray_norm:
        Float image in [0, 1].
    """
    # Multi-scale Hessian blob enhancement
    scales = [1.0, 2.0, 4.0]
    max_response = np.zeros_like(gray_norm, dtype=np.float32)

    img_u8 = (gray_norm * 255).clip(0, 255).astype(np.uint8)

    for sigma in scales:
        ksize = max(3, int(6 * sigma) | 1)  # odd kernel size
        blurred = cv2.GaussianBlur(img_u8, (ksize, ksize), sigma).astype(np.float32)

        # Compute second-order derivatives via Sobel
        dxx = cv2.Sobel(blurred, cv2.CV_32F, 2, 0, ksize=3)
        dyy = cv2.Sobel(blurred, cv2.CV_32F, 0, 2, ksize=3)
        dxy = cv2.Sobel(blurred, cv2.CV_32F, 1, 1, ksize=3)

        # Hessian eigenvalues (analytical formula for 2×2 matrix)
        trace = dxx + dyy
        det = dxx * dyy - dxy * dxy
        discriminant = np.sqrt(np.maximum(0.0, trace ** 2 - 4 * det))

        lam1 = 0.5 * (trace + discriminant)
        lam2 = 0.5 * (trace - discriminant)

        # Frangi blob measure: both eigenvalues negative ↔ bright blob
        rb = np.where(lam2 != 0, (lam1 / lam2) ** 2, 0.0)
        s2 = lam1 ** 2 + lam2 ** 2
        beta = 0.5
        c = 15.0

        vesselness = np.exp(-rb / (2 * beta ** 2)) * (1 - np.exp(-s2 / (2 * c ** 2)))
        vesselness = np.where(lam2 > 0, vesselness, 0.0).astype(np.float32)

        max_response = np.maximum(max_response, vesselness)

    # Normalise to [0, 1]
    mn, mx = max_response.min(), max_response.max()
    if mx > mn:
        max_response = (max_response - mn) / (mx - mn)
    return max_response


def preprocess_patch(patch: np.ndarray) -> np.ndarray:
    """Full preprocessing pipeline for a single image patch.

    1. Convert to grayscale.
    2. Resize to PATCH_SIZE × PATCH_SIZE.
    3. Apply CLAHE.
    4. Normalise to [0, 1].

    Returns float32 array of shape (PATCH_SIZE, PATCH_SIZE).
    """
    gray = to_gray(patch)
    gray = cv2.resize(gray, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_AREA)
    enhanced = apply_clahe(gray)
    normalised = normalize_image(enhanced)
    return normalised
