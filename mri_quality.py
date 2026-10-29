"""
mri_structure_clarity.py - how clearly are anatomical structures visible?
Compares the average score of all images in one folder with another
(e.g. small vs large receive antenna).

python mri_structure_clarity.py --dataset large_antenna /data/large \
                                --dataset small_antenna /data/small --out_dir clarity

Inputs: DICOM (2-D, with or without extension) and NIfTI (2-D/3-D/4-D) files,
searched recursively. 3-D volumes are measured slice by slice (along
--slice_axis, default: the axis with the fewest voxels) and averaged.

Why these metrics
-----------------
"Clarity" means a structure stands out from the noise and has a sharp
border. Plain sharpness/contrast measures are fooled by noise (noise looks
like detail), so every metric here is either calibrated against the noise
level of the same image or measured on edges that clearly exceed it. All
of them are invariant to the scanner's intensity scaling. Internal
structures only: the body outline (air/skin edge) is excluded.

  edge_cnr              typical intensity step across internal edges, in units of
                        noise sigma: how many noise levels a border stands out.   higher = clearer
  visible_edge_density  length of (thinned, 1-pixel) borders per cm2 of body that
                        are stronger than what noise alone produces 99.5% of the
                        time (noise-calibrated by simulation): how much structure
                        can be told apart from noise. Thinning keeps blur from
                        inflating it.                                             higher = clearer
  edge_width_mm         10-90% rise distance across strong internal edges, i.e.
                        how blurred borders are (needs edges well above noise).   lower = sharper
  tissue_cnr            contrast between the two main tissue classes / noise.     higher = clearer
  cjv                   overlap of those two classes' intensities.                 lower = clearer
  snr                   mean body signal / noise (context: drives most of the above)

Noise sigma is estimated inside the body (Immerkaer's method), so blanked
backgrounds or parallel imaging do not break it.

Outputs (in --out_dir)
  per_image.csv, summary.csv (mean / std / median per folder),
  comparison.csv (+ printed table: % difference, Mann-Whitney p, Cliff's delta,
  which folder is clearer), boxplots.png, clarity_examples.png (a typical image
  of each folder with its visible edges highlighted).

For a fair comparison: same sequence parameters, similar anatomy/slice
positions; use --group_by parent (one folder per subject) so that slices of
one subject are not counted as independent.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import numpy as np
from scipy import ndimage, stats
from scipy.ndimage import gaussian_filter, map_coordinates

HIGHER_IS_BETTER = {
    "edge_cnr": True,
    "visible_edge_density": True,
    "edge_width_mm": False,
    "tissue_cnr": True,
    "cjv": False,
    "snr": True,
}
METRICS = list(HIGHER_IS_BETTER)

SMOOTH_PX = 1.0          # pre-smoothing for edge detection
PROFILE_HALF_PX = 6.0    # edge profiles sampled over +-6 px
EDGE_EXCLUDE_PX = 5      # erosion that removes the body outline


# =============================================================================
# Loading (DICOM / NIfTI)
# =============================================================================
def iter_images(path: str, frame_step: int = 1):
    """Yield (image_id, subject, array, spacing_mm)."""
    files = [path] if os.path.isfile(path) else sorted(
        os.path.join(root, f) for root, _, fs in os.walk(path) for f in fs if not f.startswith(".")
    )
    for f in files:
        subject = os.path.basename(os.path.dirname(f))
        low = f.lower()
        if low.endswith((".nii", ".nii.gz")):
            import nibabel as nib
            img = nib.load(f)
            data = np.asarray(img.dataobj, dtype=np.float32)
            zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
            if data.ndim == 4:
                for t in range(0, data.shape[3], frame_step):
                    yield f"{f}#t{t}", subject, data[..., t], zooms
            else:
                yield f, subject, np.squeeze(data), zooms[:np.squeeze(data).ndim]
            continue
        if low.endswith((".csv", ".txt", ".json", ".png", ".jpg", ".npy", ".pt", ".ply", ".py")):
            continue
        try:
            import pydicom
            ds = pydicom.dcmread(f, force=True)
            arr = ds.pixel_array.astype(np.float32)
        except Exception:
            continue
        arr = arr * float(ds.get("RescaleSlope", 1.0)) + float(ds.get("RescaleIntercept", 0.0))
        ps = [float(v) for v in ds.get("PixelSpacing", [1.0, 1.0])]
        spacing = tuple(ps) if arr.ndim == 2 else (float(ds.get("SliceThickness", 1.0)), *ps)
        yield f, subject, arr, spacing


def slices_of(arr: np.ndarray, spacing, slice_axis: int | None):
    """2-D images to measure: the image itself, or the slices of a volume."""
    if arr.ndim == 2:
        return [(arr, tuple(spacing[-2:]))]
    axis = int(np.argmin(arr.shape)) if slice_axis is None else slice_axis
    in_plane = tuple(s for i, s in enumerate(spacing) if i != axis) if len(spacing) == 3 else (1.0, 1.0)
    return [(np.take(arr, k, axis=axis), in_plane) for k in range(arr.shape[axis])]


# =============================================================================
# Masks / noise
# =============================================================================
def otsu(values: np.ndarray) -> float:
    hist, edges = np.histogram(values, bins=256)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers) / np.maximum(w0, 1)
    m1 = ((hist * centers).sum() - np.cumsum(hist * centers)) / np.maximum(w1, 1)
    return float(centers[np.argmax(w0 * w1 * (m0 - m1) ** 2)])


def body_mask(img: np.ndarray) -> np.ndarray:
    smooth = gaussian_filter(img, 1.0)
    fg = smooth > otsu(smooth)
    fg = ndimage.binary_opening(fg, iterations=1)
    labels, n = ndimage.label(fg)
    if n > 1:
        sizes = ndimage.sum(fg, labels, range(1, n + 1))
        fg = np.isin(labels, 1 + np.where(sizes >= 0.01 * fg.size)[0])
    return ndimage.binary_fill_holes(fg)


def immerkaer_sigma(img: np.ndarray, region: np.ndarray) -> float:
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    resp = np.abs(ndimage.convolve(img, kernel, mode="reflect"))[region]
    return float(np.sqrt(np.pi / 2) * resp.mean() / 6.0) if resp.size else float("nan")


# =============================================================================
# Clarity metrics for one 2-D image
# =============================================================================
def edge_profiles(img, gy, gx, gmag, candidates, max_edges=1500, rng=None):
    """Non-maximum-suppressed edge points + intensity profiles across them."""
    ys, xs = np.nonzero(candidates)
    if len(ys) == 0:
        return np.empty((0, 0)), np.empty(0), 0
    uy, ux = gy[ys, xs] / (gmag[ys, xs] + 1e-8), gx[ys, xs] / (gmag[ys, xs] + 1e-8)
    # keep local maxima of the gradient along its direction (thin edges)
    g_plus = map_coordinates(gmag, [ys + uy, xs + ux], order=1, mode="nearest")
    g_minus = map_coordinates(gmag, [ys - uy, xs - ux], order=1, mode="nearest")
    keep = (gmag[ys, xs] >= g_plus) & (gmag[ys, xs] >= g_minus)
    ys, xs, uy, ux = ys[keep], xs[keep], uy[keep], ux[keep]
    n_total = len(ys)
    if len(ys) > max_edges:
        sel = (rng or np.random.default_rng(0)).choice(len(ys), max_edges, replace=False)
        ys, xs, uy, ux = ys[sel], xs[sel], uy[sel], ux[sel]
    t = np.arange(-PROFILE_HALF_PX, PROFILE_HALF_PX + 1e-9, 0.25)
    py = ys[:, None] + uy[:, None] * t[None]
    px = xs[:, None] + ux[:, None] * t[None]
    prof = map_coordinates(img, [py.ravel(), px.ravel()], order=1, mode="nearest").reshape(len(ys), -1)
    return prof, t, n_total


def width_10_90(prof_norm: np.ndarray, t: np.ndarray) -> np.ndarray:
    """10-90% rise distance of normalised (0 -> 1) profiles, searched outward
    from the centre; NaN when a crossing is missing."""
    c = len(t) // 2
    widths = np.full(len(prof_norm), np.nan)
    for i, p in enumerate(prof_norm):
        left = np.where(p[:c + 1] <= 0.1)[0]
        right = np.where(p[c:] >= 0.9)[0]
        if len(left) and len(right):
            widths[i] = t[c + right[0]] - t[left[-1]]
    return widths


def clarity_metrics(img: np.ndarray, spacing_2d, rng=None) -> dict | None:
    img = np.nan_to_num(img.astype(np.float32))
    fg = body_mask(img)
    core = ndimage.binary_erosion(fg, iterations=EDGE_EXCLUDE_PX)     # internal structures only
    if core.sum() < 400:
        return None
    rng = rng or np.random.default_rng(0)
    sigma = immerkaer_sigma(img, ndimage.binary_erosion(fg, iterations=3))
    if not np.isfinite(sigma) or sigma <= 0:
        return None
    body = img[fg]
    out = {"noise_sigma": sigma, "snr": float(body.mean() / sigma)}

    # ---- noise reference: same filters applied to pure noise of this sigma -------
    noise = rng.normal(0.0, sigma, img.shape).astype(np.float32)
    ns = gaussian_filter(noise, SMOOTH_PX)
    ngy, ngx = np.gradient(ns)
    noise_gmag = np.hypot(ngy, ngx)
    g_thr = float(np.percentile(noise_gmag, 99.5))

    # ---- edges -------------------------------------------------------------------
    sm = gaussian_filter(img, SMOOTH_PX)
    gy, gx = np.gradient(sm)
    gmag = np.hypot(gy, gx)
    visible = core & (gmag > g_thr)
    px_mm = float(np.mean(spacing_2d))
    prof, t, n_edges = edge_profiles(img, gy, gx, gmag, visible, rng=rng)
    # false detections that pure noise of this sigma produces in the same area
    _, _, n_noise = edge_profiles(noise, ngy, ngx, noise_gmag, core & (noise_gmag > g_thr), rng=rng)
    # edge length (mm) per cm2 of internal body area, noise false positives removed
    out["visible_edge_density"] = float(max(n_edges - n_noise, 0) * px_mm
                                        / (core.sum() * px_mm ** 2 / 100.0))
    if len(prof) >= 20:
        lo = prof[:, t <= -3.0].mean(axis=1)
        hi = prof[:, t >= 3.0].mean(axis=1)
        step = hi - lo                                   # positive: profiles follow the gradient
        out["edge_cnr"] = float(np.median(step) / sigma)
        clear = step > 5 * sigma                         # widths only where the edge clearly beats noise
        if clear.sum() >= 20:
            pn = (prof[clear] - lo[clear, None]) / step[clear, None]
            w = width_10_90(pn, t)
            out["edge_width_mm"] = float(np.nanmedian(w) * px_mm) if np.isfinite(w).any() else float("nan")
    out["n_visible_edges"] = int(n_edges)

    # ---- two tissue classes inside the body ------------------------------------------
    inner = img[core]
    thr = otsu(inner)
    a, b = inner[inner > thr], inner[inner <= thr]
    if len(a) > 20 and len(b) > 20:
        diff = abs(a.mean() - b.mean())
        out["tissue_cnr"] = float(diff / sigma)
        out["cjv"] = float((a.std() + b.std()) / diff) if diff > 0 else float("nan")
    out["_visible_mask"] = visible           # for the example figure only
    return out


def measure_image(arr, spacing, slice_axis):
    """Average the per-slice metrics of one file (slices without body skipped)."""
    per_slice = []
    best = None
    for sl, sp in slices_of(arr, spacing, slice_axis):
        m = clarity_metrics(sl, sp)
        if m is None:
            continue
        per_slice.append(m)
        if best is None or m.get("snr", 0) > best[1].get("snr", 0):
            best = (sl, m)
    if not per_slice:
        return None, None
    avg = {k: float(np.nanmean([m[k] for m in per_slice if k in m]))
           for k in METRICS + ["noise_sigma"] if any(k in m for m in per_slice)}
    avg["n_slices"] = len(per_slice)
    return avg, best


# =============================================================================
# Aggregation
# =============================================================================
def cliffs_delta(a, b):
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    return float((sum((x > b).sum() for x in a) - sum((x < b).sum() for x in a)) / (len(a) * len(b)))


def values(rows, dataset, metric, group_by):
    vals = [(r["subject"], r.get(metric)) for r in rows if r["dataset"] == dataset]
    vals = [(s, v) for s, v in vals if v is not None and np.isfinite(v)]
    if group_by == "parent":
        per = defaultdict(list)
        for s, v in vals:
            per[s].append(v)
        return np.array([np.mean(v) for v in per.values()])
    return np.array([v for _, v in vals])


def main():
    ap = argparse.ArgumentParser(description="Structure clarity: folder vs folder")
    ap.add_argument("--dataset", nargs=2, action="append", required=True, metavar=("NAME", "PATH"),
                    help="Give it twice: reference first (e.g. large antenna), then the other")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--group_by", choices=["none", "parent"], default="none")
    ap.add_argument("--slice_axis", type=int, default=None,
                    help="3-D files: axis to slice along (default: the shortest axis)")
    ap.add_argument("--frame_step", type=int, default=1)
    ap.add_argument("--max_images", type=int, default=None)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    names = [n for n, _ in args.dataset]

    rows, examples = [], defaultdict(list)
    for name, path in args.dataset:
        count = 0
        for image_id, subject, arr, spacing in iter_images(path, args.frame_step):
            if arr.ndim not in (2, 3):
                continue
            m, best = measure_image(arr, spacing, args.slice_axis)
            if m is None:
                continue
            rows.append({"dataset": name, "subject": subject, "image": image_id, **m})
            examples[name].append((m.get("edge_cnr", np.nan), image_id, best))
            count += 1
            if count % 50 == 0:
                print(f"  [{name}] {count} images")
            if args.max_images and count >= args.max_images:
                break
        print(f"{name}: {count} images measured from {path}")
    if not rows:
        raise SystemExit("No measurable images found.")

    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(args.out_dir, "per_image.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    # ---- folder averages ------------------------------------------------------------
    summary = []
    print(f"\nAverage per folder ({'per subject' if args.group_by == 'parent' else 'per image'}):")
    print(f"  {'metric':22s} " + " ".join(f"{n[:18]:>22s}" for n in names))
    for m in METRICS:
        cells = []
        for n in names:
            v = values(rows, n, m, args.group_by)
            if len(v):
                summary.append({"dataset": n, "metric": m, "n": len(v), "mean": v.mean(),
                                "std": v.std(ddof=1) if len(v) > 1 else 0.0,
                                "median": float(np.median(v))})
                cells.append(f"{v.mean():10.4g} +- {v.std(ddof=1) if len(v) > 1 else 0:<8.3g}")
            else:
                cells.append(f"{'-':>22s}")
        print(f"  {m:22s} " + " ".join(f"{c:>22s}" for c in cells))
    with open(os.path.join(args.out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    # ---- comparison -------------------------------------------------------------------
    comparison = []
    if len(names) >= 2:
        ref, other = names[0], names[1]
        print(f"\n{other} vs {ref}")
        print(f"  {'metric':22s} {'difference':>11s} {'p (MWU)':>9s} {'Cliff d':>8s}  clearer in")
        tally = defaultdict(int)
        for m in METRICS:
            a, b = values(rows, ref, m, args.group_by), values(rows, other, m, args.group_by)
            if len(a) < 2 or len(b) < 2:
                continue
            diff = 100 * (b.mean() - a.mean()) / abs(a.mean()) if a.mean() else float("nan")
            p = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
            d = cliffs_delta(b, a)
            b_better = (b.mean() > a.mean()) == HIGHER_IS_BETTER[m]
            winner = (other if b_better else ref) if p < 0.05 else "no clear difference"
            if m != "snr":
                tally[winner] += 1
            comparison.append({"reference": ref, "other": other, "metric": m,
                               "mean_reference": a.mean(), "mean_other": b.mean(),
                               "difference_pct": diff, "p_mannwhitney": p,
                               "cliffs_delta": d, "clearer_in": winner})
            print(f"  {m:22s} {diff:+10.1f}% {p:9.3g} {d:+8.2f}  {winner}")
        print("\n  clarity metrics favouring each folder (p < 0.05, SNR excluded): "
              + ", ".join(f"{k}: {v}" for k, v in tally.items()))
        with open(os.path.join(args.out_dir, "comparison.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(comparison[0]))
            w.writeheader()
            w.writerows(comparison)

    # ---- figures -------------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        for ax, m in zip(axes.ravel(), METRICS):
            data = [values(rows, n, m, args.group_by) for n in names]
            try:
                ax.boxplot(data, tick_labels=names)
            except TypeError:
                ax.boxplot(data, labels=names)
            ax.set_title(f"{m} ({'higher' if HIGHER_IS_BETTER[m] else 'lower'} = clearer)", fontsize=9)
        for ax in axes.ravel()[len(METRICS):]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "boxplots.png"), dpi=110)
        plt.close(fig)

        # typical image of each folder (median edge CNR) with its visible edges
        fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 6), squeeze=False)
        for ax, n in zip(axes[0], names):
            ex = [e for e in examples[n] if np.isfinite(e[0])]
            if not ex:
                ax.axis("off")
                continue
            ex.sort(key=lambda e: e[0])
            score, image_id, (sl, m) = ex[len(ex) // 2]
            lo, hi = np.percentile(sl, [1, 99])
            ax.imshow(sl, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
            overlay = np.zeros((*sl.shape, 4))
            overlay[m["_visible_mask"]] = [1, 0.2, 0.2, 0.6]
            ax.imshow(overlay, interpolation="nearest")
            ax.set_title(f"{n}: {os.path.basename(image_id)}\n"
                         f"edge CNR {m.get('edge_cnr', np.nan):.1f} | visible edges "
                         f"{m.get('visible_edge_density', np.nan):.1f} mm/cm2 | SNR {m['snr']:.1f}",
                         fontsize=10)
            ax.axis("off")
        fig.suptitle("Typical image per folder (median edge CNR); red = edges distinguishable from noise")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "clarity_examples.png"), dpi=110)
        plt.close(fig)
    except ImportError:
        pass

    print(f"\nResults in {args.out_dir}: per_image.csv, summary.csv, comparison.csv, "
          f"boxplots.png, clarity_examples.png")


if __name__ == "__main__":
    main()