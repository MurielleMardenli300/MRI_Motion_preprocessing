"""
Step 2 - navigator-based 4D reconstruction.

python reconstruct_4d.py --mri_dir "/path/to/all_1188_files" --slots slots.csv \
    --out_dir /path/to/recon --name paired_01

For every slot t (one per ~450 ms) the scanner measured ONE data slice, at
position p_t. The volume for time t is completed with, at every other
position, the data slice (from another sweep) whose navigator shows the
closest breathing state, preferring the same breathing direction
(inhale vs exhale).

Breathing state (from slots.json, override with --state):
  fixed : navigator at one fixed plane -> PCA of the navigator images
  shift : navigator changes with position -> head-feet shift of each
          navigator vs the median image at its own position, z-scored per
          position, smoothed in time

Outputs (in <out_dir>/<name>/)
  native/frame_XXXX.nii.gz  full resolution, exact scanner geometry and raw
                            intensities -> opens correctly in 3D Slicer in all
                            three planes
  train/<prefix>XXXX.nii.gz (only with --train_size) resized + [0,1]-normalised
                            copies for the training pipeline; check axis order
                            and flips against one Nav_dataset volume
  reconstruction_log.csv, breathing_state.npy, breathing_state.png
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

from utils import (
    dominant_period,
    fill_nan_in_time,
    fixed_navigator_state,
    lag1_autocorr,
    read_dicom,
    shift_navigator_state,
    smooth,
    zflat,
)


# =============================================================================
# Core steps (no file I/O, testable)
# =============================================================================
def compute_state(navs, positions, method, n_pcs, factor, max_shift_px, min_std_px, smooth_k):
    if method == "fixed":
        state, explained = fixed_navigator_state(navs, factor, n_pcs)
        info = f"navigator PCA, explained variance {np.round(explained, 3).tolist()}"
        if smooth_k > 1:
            state = np.stack([smooth(state[:, i], smooth_k) for i in range(state.shape[1])], 1)
        return state, info
    raw, unreliable = shift_navigator_state(navs, positions, factor, max_shift_px, min_std_px)
    state = smooth(fill_nan_in_time(raw), smooth_k)[:, None]
    n_pos = len(np.unique(positions))
    info = (f"head-feet shift, {n_pos - len(unreliable)}/{n_pos} positions with visible motion"
            + (f" (none at {unreliable})" if unreliable else ""))
    return state, info


def breathing_direction(state1d: np.ndarray) -> np.ndarray:
    return np.where(np.gradient(state1d) >= 0, 1, -1)


def match_slices(state, direction, positions, use_direction=True):
    """chosen[t, p] = slot whose data slice fills position p at time t."""
    T = len(state)
    pos_list = np.unique(positions)
    chosen = np.zeros((T, len(pos_list)), dtype=int)
    dist = np.zeros((T, len(pos_list)), dtype=np.float32)
    cands = {p: np.where(positions == p)[0] for p in pos_list}
    for t in range(T):
        for j, p in enumerate(pos_list):
            if p == positions[t]:
                chosen[t, j] = t
                continue
            pool = cands[p]
            if use_direction:
                same = pool[direction[pool] == direction[t]]
                pool = same if len(same) else pool
            d = np.linalg.norm(state[pool] - state[t], axis=1)
            k = int(np.argmin(d))
            chosen[t, j], dist[t, j] = pool[k], d[k]
    return chosen, dist, pos_list


def leave_one_out_score(datas, state, direction, positions, factor, use_direction=True):
    """Hide each slot's own data slice, re-pick it from the other sweeps with
    the state rule, compare with the truth (NCC). 1 = ideal, 0 = random."""
    x = zflat(datas, factor)
    d = x.shape[1]
    chosen_v, random_v, best_v = [], [], []
    idx_all = np.arange(len(x))
    for t in range(len(x)):
        cand = idx_all[(positions == positions[t]) & (idx_all != t)]
        ncc = x[cand] @ x[t] / d
        pool = cand
        if use_direction:
            same = cand[direction[cand] == direction[t]]
            pool = same if len(same) else cand
        j = pool[np.argmin(np.linalg.norm(state[pool] - state[t], axis=1))]
        chosen_v.append(ncc[np.where(cand == j)[0][0]])
        random_v.append(ncc.mean())
        best_v.append(ncc.max())
    c, r, b = float(np.mean(chosen_v)), float(np.mean(random_v)), float(np.mean(best_v))
    return c, r, b, (c - r) / (b - r + 1e-8)


def native_geometry(ipp_by_pos: dict, iop: np.ndarray, pixel_spacing):
    """Spatial slice order + RAS affine for arrays shaped (cols, rows, slices),
    following the standard DICOM -> NIfTI convention."""
    row_dir, col_dir = iop[:3], iop[3:]
    normal = np.cross(row_dir, col_dir)
    order = sorted(ipp_by_pos, key=lambda p: float(ipp_by_pos[p] @ normal))
    first, last = ipp_by_pos[order[0]], ipp_by_pos[order[-1]]
    step = (last - first) / max(len(order) - 1, 1)
    lps = np.eye(4)
    lps[:3, 0] = row_dir * pixel_spacing[1]    # along a row  = increasing column index
    lps[:3, 1] = col_dir * pixel_spacing[0]    # down a column = increasing row index
    lps[:3, 2] = step
    lps[:3, 3] = first
    return order, np.diag([-1.0, -1.0, 1.0, 1.0]) @ lps   # LPS -> RAS


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mri_dir", required=True)
    ap.add_argument("--slots", required=True, help="slots.csv from order_slices.py")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--name", required=True, help="Output folder (= sequence name)")
    ap.add_argument("--state", choices=["auto", "fixed", "shift"], default="auto")
    ap.add_argument("--n_pcs", type=int, default=3)
    ap.add_argument("--smooth", type=int, default=None,
                    help="Temporal smoothing in slots (default: 1 fixed, 3 shift)")
    ap.add_argument("--no_direction", action="store_true")
    ap.add_argument("--downsample", type=int, default=4)
    ap.add_argument("--max_shift_px", type=int, default=24)
    ap.add_argument("--min_std_px", type=float, default=0.5)
    ap.add_argument("--frame_step", type=int, default=1, help="Write every N-th frame")
    ap.add_argument("--no_native", action="store_true", help="Skip the full-resolution volumes")
    ap.add_argument("--train_size", nargs=3, type=int, default=None,
                    help="Also write resized volumes, axes (left-right, AP, head-feet), e.g. 32 64 64")
    ap.add_argument("--train_flip", nargs="*", type=int, default=[],
                    help="Axes (0,1,2) of the training volume to flip")
    ap.add_argument("--prefix", default="im", help="Training file prefix (2 characters)")
    args = ap.parse_args()

    with open(args.slots) as f:
        rows = sorted(csv.DictReader(f), key=lambda r: int(r["slot_idx"]))
    summary_path = args.slots.rsplit(".", 1)[0] + ".json"
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
    method = args.state if args.state != "auto" else \
        ("fixed" if summary.get("navigator_fixed") else "shift")
    smooth_k = args.smooth if args.smooth is not None else (1 if method == "fixed" else 3)
    slot_dt = float(summary.get("slot_duration_s", 0.45))

    T = len(rows)
    positions = np.array([int(r["position"]) for r in rows])
    print(f"{T} slots, {len(np.unique(positions))} positions | state: {method} | smoothing {smooth_k}")

    navs, datas, data_recs = [], [], []
    for r in rows:
        navs.append(read_dicom(os.path.join(args.mri_dir, r["nav_file"]))["img"])
        rec = read_dicom(os.path.join(args.mri_dir, r["data_file"]))
        datas.append(rec["img"])
        data_recs.append(rec)
    navs, datas = np.stack(navs), np.stack(datas)

    # ---- breathing state + checks ---------------------------------------------------
    state, info = compute_state(navs, positions, method, args.n_pcs, args.downsample,
                                args.max_shift_px, args.min_std_px, smooth_k)
    direction = breathing_direction(state[:, 0])
    print(f"state: {info}")
    print(f"state check: lag-1 autocorrelation {lag1_autocorr(state[:, 0]):.2f} (want > 0.7), "
          f"dominant period {dominant_period(state[:, 0], slot_dt):.1f} s (breathing ~2.5-7 s)")

    use_dir = not args.no_direction
    c, r, b, score = leave_one_out_score(datas, state, direction, positions, args.downsample, use_dir)
    print(f"leave-one-out matching: NCC chosen {c:.3f} | random {r:.3f} | best {b:.3f} "
          f"-> score {score:.2f} (1 = ideal, 0 = random)")

    chosen, dist, pos_list = match_slices(state, direction, positions, use_dir)
    spread = np.linalg.norm(state - state.mean(0), axis=1).mean() + 1e-8
    rel = dist / spread
    print(f"matching distance (relative to state spread): median {np.median(rel):.2f}, "
          f"95th pct {np.percentile(rel, 95):.2f}")

    # ---- geometry --------------------------------------------------------------------
    ipp_by_pos = {}
    for rec, p in zip(data_recs, positions):
        ipp_by_pos.setdefault(int(p), rec["ipp"])
    spatial_order, affine = native_geometry(ipp_by_pos, data_recs[0]["iop"],
                                            data_recs[0]["pixel_spacing"])
    col_of = {int(p): j for j, p in enumerate(pos_list)}
    print(f"voxel size (mm): {np.round(np.linalg.norm(affine[:3, :3], axis=0), 3).tolist()}")

    seq_dir = os.path.join(args.out_dir, args.name)
    native_dir, train_dir = os.path.join(seq_dir, "native"), os.path.join(seq_dir, "train")
    if not args.no_native:
        os.makedirs(native_dir, exist_ok=True)
    if args.train_size:
        if len(args.prefix) != 2:
            raise SystemExit("--prefix must be 2 characters (loader reads name[2:-7])")
        os.makedirs(train_dir, exist_ok=True)
        lo, hi = np.percentile(datas, 0.5), np.percentile(datas, 99.5)

    import nibabel as nib
    from scipy.ndimage import zoom

    for t in range(0, T, args.frame_step):
        slices = [datas[chosen[t, col_of[p]]] for p in spatial_order]
        vol = np.stack([s.T for s in slices], axis=-1).astype(np.float32)   # (cols, rows, slices)
        if not args.no_native:
            nib.save(nib.Nifti1Image(vol, affine), os.path.join(native_dir, f"frame_{t:04d}.nii.gz"))
        if args.train_size:
            tv = vol.transpose(2, 0, 1)                                     # (LR, AP, HF)
            for ax in args.train_flip:
                tv = np.flip(tv, axis=ax)
            tv = zoom(tv, [o / s for o, s in zip(args.train_size, tv.shape)], order=1)
            tv = np.clip((tv - lo) / (hi - lo + 1e-8), 0, 1).astype(np.float32)
            sp = np.linalg.norm(affine[:3, :3], axis=0)[[2, 0, 1]] * \
                np.array(vol.transpose(2, 0, 1).shape) / np.array(args.train_size)
            nib.save(nib.Nifti1Image(tv, np.diag([*sp, 1.0])),
                     os.path.join(train_dir, f"{args.prefix}{t:04d}.nii.gz"))
        if t % 100 == 0:
            print(f"  frame {t}/{T}")

    # ---- log + QC ----------------------------------------------------------------------
    with open(os.path.join(seq_dir, "reconstruction_log.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "slot_idx", "own_position", "direction", "mean_rel_dist",
                    "max_rel_dist", "source_slot_per_position_in_space"])
        for t in range(T):
            w.writerow([t, rows[t]["slot_idx"], positions[t], direction[t],
                        f"{rel[t].mean():.3f}", f"{rel[t].max():.3f}",
                        ";".join(str(chosen[t, col_of[p]]) for p in spatial_order)])
    np.save(os.path.join(seq_dir, "breathing_state.npy"), state)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(np.arange(T) * slot_dt, state[:, 0], lw=0.8)
        ax.set_xlabel("time (s)")
        ax.set_ylabel(f"breathing state ({method})")
        fig.tight_layout()
        fig.savefig(os.path.join(seq_dir, "breathing_state.png"), dpi=120)
        plt.close(fig)
    except ImportError:
        pass
    print(f"done -> {seq_dir}")


if __name__ == "__main__":
    main()