"""
Shared helpers for order_slices.py and reconstruct_4d.py.

Acquisition model (dynamic 2-D sagittal bSSFP with interleaved 2-D navigators):
  * N positions are swept repeatedly ("sweeps"); one position per ~450 ms slot
  * every slot holds two images: one data slice (at the swept position) and
    one navigator frame; they are told apart by TriggerTime
  * the navigator measures the breathing state at that moment, so data slices
    from different sweeps can be re-assembled into volumes (navigator sorting)
"""

from __future__ import annotations

import os
import re

import numpy as np


# =============================================================================
# DICOM reading
# =============================================================================
def file_number(name: str):
    m = re.search(r"\((\d+)\)", name) or re.search(r"(\d+)(?!.*\d)", name)
    return int(m.group(1)) if m else None


def dicom_time_to_s(value: str) -> float:
    """'130951.94' (HHMMSS.ff) -> seconds since midnight."""
    value = str(value).strip()
    return int(value[0:2]) * 3600 + int(value[2:4]) * 60 + float(value[4:])


def read_dicom(path: str, with_pixels: bool = True) -> dict:
    import pydicom

    ds = pydicom.dcmread(path, stop_before_pixels=not with_pixels, force=True)
    rec = {
        "file": os.path.basename(path),
        "acq_time": str(ds.get("AcquisitionTime", "")),
        "slice_location": round(float(ds.get("SliceLocation", "nan")), 3),
        "trigger": round(float(ds.get("TriggerTime", 0.0)), 1),
        "ipp": np.array([float(v) for v in ds.get("ImagePositionPatient", [0, 0, 0])]),
        "iop": np.array([float(v) for v in ds.get("ImageOrientationPatient", [1, 0, 0, 0, 1, 0])]),
        "pixel_spacing": [float(v) for v in ds.get("PixelSpacing", [1.0, 1.0])],
    }
    if with_pixels:
        slope = float(ds.get("RescaleSlope", 1.0))
        intercept = float(ds.get("RescaleIntercept", 0.0))
        rec["img"] = ds.pixel_array.astype(np.float32) * slope + intercept
    return rec


def read_folder(folder: str, with_pixels: bool = True) -> dict[int, dict]:
    records = {}
    for name in os.listdir(folder):
        n = file_number(name)
        if n is None or name.startswith("."):
            continue
        rec = read_dicom(os.path.join(folder, name), with_pixels)
        rec["instance"] = n
        records[n] = rec
    return records


# =============================================================================
# Image features
# =============================================================================
def block_downsample(img: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return img
    h, w = (img.shape[0] // factor) * factor, (img.shape[1] // factor) * factor
    return img[:h, :w].reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))


def zflat(images: np.ndarray, factor: int) -> np.ndarray:
    """[N,H,W] -> [N,d], each row z-scored (so a dot product / d = NCC)."""
    x = np.stack([block_downsample(i, factor) for i in images]).reshape(len(images), -1)
    return (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)


def pca_scores(x: np.ndarray, k: int):
    xc = x - x.mean(0)
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    return xc @ vt[:k].T, (s ** 2 / (s ** 2).sum())[:k]


def position_dependence(x: np.ndarray, positions: np.ndarray) -> float:
    """Fraction of the variance of the images' first principal component that
    is explained by slice position. ~0 for a navigator at one fixed plane (its
    main variation is breathing), ~1 for images that follow the swept position."""
    pc1 = pca_scores(x, 1)[0][:, 0]
    total = pc1.var()
    if total == 0:
        return 0.0
    within = np.mean([(pc1[positions == p] - pc1[positions == p].mean()).var() * (positions == p).sum()
                      for p in np.unique(positions)]) * len(np.unique(positions)) / len(pc1)
    return float(1.0 - within / total)


def same_vs_cross_position_ncc(x: np.ndarray, positions: np.ndarray,
                               n_pairs: int = 4000, seed: int = 0) -> tuple[float, float]:
    """Mean NCC between images at the SAME position (different sweeps) and at
    DIFFERENT positions. A navigator at a fixed plane has cross ~= same; a
    data slice (sweeping positions) has cross << same."""
    rng = np.random.default_rng(seed)
    d = x.shape[1]
    i, j = rng.integers(0, len(x), n_pairs), rng.integers(0, len(x), n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    ncc = np.einsum("nd,nd->n", x[i], x[j]) / d
    same = positions[i] == positions[j]
    return float(ncc[same].mean()), float(ncc[~same].mean())


# =============================================================================
# Breathing state
# =============================================================================
def vertical_shift(img: np.ndarray, ref: np.ndarray, weight: np.ndarray, max_shift: int) -> float:
    """Sub-pixel head-feet shift of img vs ref (weighted NCC + parabola).
    Positive = anatomy moved down the image (towards the feet for sagittal
    images with the usual orientation) = inhalation."""
    H = img.shape[0]
    scores = []
    for s in range(-max_shift, max_shift + 1):
        if s >= 0:
            a, b, w = img[s:], ref[:H - s], weight[:H - s]
        else:
            a, b, w = img[:H + s], ref[-s:], weight[-s:]
        wsum = w.sum() + 1e-8
        da = a - (a * w).sum() / wsum
        db = b - (b * w).sum() / wsum
        num = (w * da * db).sum()
        den = np.sqrt((w * da * da).sum() * (w * db * db).sum()) + 1e-8
        scores.append(num / den)
    scores = np.array(scores)
    k = int(np.argmax(scores))
    offset = 0.0
    if 0 < k < len(scores) - 1:
        y0, y1, y2 = scores[k - 1], scores[k], scores[k + 1]
        denom = y0 - 2 * y1 + y2
        offset = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
    return (k - max_shift) + offset


def fixed_navigator_state(navs: np.ndarray, factor: int, n_pcs: int):
    """Navigator at one fixed plane: PCA of the navigator images."""
    scores, explained = pca_scores(zflat(navs, factor), n_pcs)
    return scores, explained


def shift_navigator_state(navs: np.ndarray, positions: np.ndarray, factor: int,
                          max_shift_px: int, min_std_px: float):
    """Navigator NOT at a fixed plane: per-position head-feet shift vs that
    position's median image, z-scored per position. Returns [N] with NaN for
    positions without visible motion, plus the list of those positions."""
    small = np.stack([block_downsample(n, factor) for n in navs])
    max_shift = max(1, int(round(max_shift_px / factor)))
    raw = np.full(len(navs), np.nan)
    unreliable = []
    for p in np.unique(positions):
        idx = np.where(positions == p)[0]
        stack = small[idx]
        ref = np.median(stack, axis=0)
        weight = stack.std(axis=0)
        weight = weight / (weight.max() + 1e-8)
        shifts = np.array([vertical_shift(s, ref, weight, max_shift) for s in stack]) * factor
        if shifts.std() < min_std_px:
            unreliable.append(int(p))
            continue
        raw[idx] = (shifts - shifts.mean()) / shifts.std()
    return raw, unreliable


# =============================================================================
# Time-series helpers
# =============================================================================
def lag1_autocorr(series: np.ndarray) -> float:
    a, b = series[:-1], series[1:]
    ok = ~np.isnan(a) & ~np.isnan(b)
    if ok.sum() < 10:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def fill_nan_in_time(x: np.ndarray) -> np.ndarray:
    good = ~np.isnan(x)
    if good.all() or not good.any():
        return np.nan_to_num(x)
    t = np.arange(len(x))
    return np.interp(t, t[good], x[good])


def smooth(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    kernel = np.ones(k) / k
    pad = k // 2
    return np.convolve(np.pad(x, (pad, k - 1 - pad), mode="edge"), kernel, mode="valid")


def dominant_period(x: np.ndarray, dt: float) -> float:
    x = x - x.mean()
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), d=dt)
    band = (freqs > 0.05) & (freqs < 1.0)
    return float(1.0 / freqs[band][np.argmax(spec[band])]) if band.any() else float("nan")


def acquisition_orders(n_pos: int) -> dict[str, list[int]]:
    """Candidate orders of positions within one sweep (position = spatial index)."""
    even, odd = list(range(0, n_pos, 2)), list(range(1, n_pos, 2))
    return {
        "sequential": list(range(n_pos)),
        "interleaved_even_first": even + odd,
        "interleaved_odd_first": odd + even,
    }