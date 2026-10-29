"""
Step 1 - chronological slot table for a dynamic 2-D sagittal bSSFP series with
interleaved 2-D navigator frames.

python order_slices.py --mri_dir "/path/to/all_1188_files" --out slots.csv

What it works out from the files themselves
  1. sweeps   : distinct AcquisitionTime values (start of each sweep)
  2. positions: distinct SliceLocation values, indexed in space (0..N-1)
  3. slots    : (sweep, position) -> one image per TriggerTime group (2 groups)
  4. roles    : which TriggerTime group is the NAVIGATOR. Decided from image
                content: navigator frames at one fixed plane look alike across
                positions, data slices do not. Also reports whether the
                navigator really is at a fixed plane (needed for the simplest,
                most accurate state; otherwise reconstruct_4d.py falls back to
                a per-position head-feet shift).
  5. order    : order in which positions are acquired within a sweep
                (sequential / interleaved even-first / interleaved odd-first),
                chosen as the order that makes the navigator breathing signal
                smoothest in time. Forward vs backward cannot be told apart
                this way; use --reverse if needed.
Outputs slots.csv (chronological) and slots.json (summary, read by step 2).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict

import numpy as np

from utils import (
    acquisition_orders,
    dicom_time_to_s,
    fixed_navigator_state,
    lag1_autocorr,
    read_folder,
    position_dependence,
    same_vs_cross_position_ncc,
    shift_navigator_state,
    zflat,
)


def build_slots(records: dict[int, dict]):
    """Group images into (sweep, position) slots; validate the layout."""
    triggers = sorted({r["trigger"] for r in records.values()})
    if len(triggers) != 2:
        raise SystemExit(f"Expected 2 TriggerTime values (navigator + data), found {triggers}.")

    sweeps = sorted({r["acq_time"] for r in records.values()}, key=dicom_time_to_s)
    sweep_idx = {t: i for i, t in enumerate(sweeps)}

    # Spatial position index, numbered in the direction of the instance numbering.
    first_seen = list(dict.fromkeys(records[n]["slice_location"] for n in sorted(records)))
    locs = sorted(first_seen)
    if first_seen[0] == locs[-1]:
        locs = locs[::-1]
    pos_idx = {loc: i for i, loc in enumerate(locs)}

    slots = defaultdict(dict)
    for n in sorted(records):
        r = records[n]
        key = (sweep_idx[r["acq_time"]], pos_idx[r["slice_location"]])
        if r["trigger"] in slots[key]:
            raise SystemExit(f"Two images with trigger {r['trigger']} in sweep/position {key}: "
                             f"{slots[key][r['trigger']]['file']} and {r['file']}.")
        slots[key][r["trigger"]] = r
    missing = [k for k, v in slots.items() if len(v) != 2]
    if missing:
        raise SystemExit(f"{len(missing)} slots do not have both images, e.g. {missing[:5]}.")
    expected = len(sweeps) * len(locs)
    if len(slots) != expected:
        print(f"WARNING: {len(slots)} slots found, {expected} expected "
              f"({len(sweeps)} sweeps x {len(locs)} positions).")
    return slots, triggers, sweeps, locs


def detect_roles(slots, triggers, factor: int):
    """Navigator = trigger group whose main variation is NOT the slice position.

    position_dependence: share of the images' first principal component
    explained by position (~0: fixed-plane navigator, ~1: follows the sweep).
    """
    keys = sorted(slots)
    positions = np.array([k[1] for k in keys])
    stats = {}
    for trig in triggers:
        x = zflat(np.stack([slots[k][trig]["img"] for k in keys]), factor)
        same, cross = same_vs_cross_position_ncc(x, positions)
        stats[trig] = {"position_dependence": position_dependence(x, positions),
                       "ncc_same_position": same, "ncc_cross_position": cross}
    nav = min(triggers, key=lambda t: stats[t]["position_dependence"])
    return nav, stats


def order_test(slots, nav_trigger, n_sweeps, n_pos, fixed: bool, factor: int):
    """Smoothness of the navigator state under each candidate order."""
    keys = sorted(slots)
    positions = np.array([k[1] for k in keys])
    navs = np.stack([slots[k][nav_trigger]["img"] for k in keys])
    if fixed:
        state = fixed_navigator_state(navs, factor, 1)[0][:, 0]
    else:
        state, _ = shift_navigator_state(navs, positions, factor, 24, 0.5)
    lookup = {k: state[i] for i, k in enumerate(keys)}
    scores = {}
    for name, order in acquisition_orders(n_pos).items():
        series = np.array([lookup.get((s, p), np.nan) for s in range(n_sweeps) for p in order])
        scores[name] = lag1_autocorr(series)
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mri_dir", required=True)
    ap.add_argument("--out", default="slots.csv")
    ap.add_argument("--nav_trigger", type=float, default=None,
                    help="Force the navigator TriggerTime instead of detecting it")
    ap.add_argument("--order", default="auto",
                    choices=["auto", *acquisition_orders(3).keys()],
                    help="Within-sweep acquisition order (auto = smoothest navigator signal)")
    ap.add_argument("--reverse", action="store_true",
                    help="Positions are swept in the opposite spatial direction")
    ap.add_argument("--fixed_threshold", type=float, default=0.3,
                    help="position dependence below which the navigator counts as a fixed plane")
    ap.add_argument("--downsample", type=int, default=4)
    args = ap.parse_args()

    records = read_folder(args.mri_dir, with_pixels=True)
    print(f"{len(records)} images read")
    slots, triggers, sweeps, locs = build_slots(records)
    n_sweeps, n_pos = len(sweeps), len(locs)

    starts = [dicom_time_to_s(t) for t in sweeps]
    sweep_dur = float(np.median(np.diff(starts))) if n_sweeps > 1 else 0.0
    slot_dur = sweep_dur / n_pos
    print(f"{n_sweeps} sweeps x {n_pos} positions | sweep {sweep_dur:.2f} s | slot {slot_dur:.3f} s | "
          f"triggers {triggers} ms")
    gaps = np.abs(np.diff(locs))
    print(f"slice spacing {np.median(gaps):.2f} mm, from {locs[0]} to {locs[-1]}")

    # ---- roles ---------------------------------------------------------------
    nav, stats = detect_roles(slots, triggers, args.downsample)
    for trig in triggers:
        s = stats[trig]
        print(f"trigger {trig:g} ms: position dependence {s['position_dependence']:.2f} "
              f"(0 = fixed plane, 1 = follows the sweep) | NCC same position "
              f"{s['ncc_same_position']:.3f}, different positions {s['ncc_cross_position']:.3f}")
    if args.nav_trigger is not None:
        nav = float(args.nav_trigger)
        print(f"navigator forced to trigger {nav:g} ms")
    else:
        print(f"navigator detected: trigger {nav:g} ms")
    data = [t for t in triggers if t != nav][0]
    fixed = stats[nav]["position_dependence"] <= args.fixed_threshold
    print("navigator at a FIXED plane: " + ("yes -> PCA state" if fixed else
          "NO (it changes with position) -> per-position head-feet shift state"))
    if not fixed and args.nav_trigger is None:
        print("  (both groups follow the sweep, so which one is the navigator cannot be told "
              "from content; force it with --nav_trigger if you know)")

    # ---- acquisition order -------------------------------------------------------
    scores = order_test(slots, nav, n_sweeps, n_pos, fixed, args.downsample)
    print("acquisition order test (lag-1 autocorrelation of the navigator state):")
    for name, v in scores.items():
        print(f"  {name:24s} {v:+.2f}")
    order_name = max(scores, key=lambda k: np.nan_to_num(scores[k], nan=-9)) \
        if args.order == "auto" else args.order
    order = acquisition_orders(n_pos)[order_name]
    if args.reverse:
        order = [n_pos - 1 - p for p in order]
    print(f"using order: {order_name}{' (reversed)' if args.reverse else ''}")
    acq_rank = {p: i for i, p in enumerate(order)}

    # ---- chronological table -----------------------------------------------------
    t0 = starts[0]
    rows = []
    for (sweep, pos), images in slots.items():
        rank = acq_rank[pos]
        slot_idx = sweep * n_pos + rank
        slot_start = starts[sweep] - t0 + rank * slot_dur
        n_img, d_img = images[nav], images[data]
        rows.append({
            "slot_idx": slot_idx,
            "sweep": sweep,
            "acq_rank": rank,
            "position": pos,
            "slice_location": locs[pos],
            "slot_start_s": round(slot_start, 3),
            "nav_file": n_img["file"],
            "nav_time_s": round(slot_start + nav / 1000.0, 3),
            "data_file": d_img["file"],
            "data_time_s": round(slot_start + data / 1000.0, 3),
        })
    rows.sort(key=lambda r: r["slot_idx"])
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "n_sweeps": n_sweeps, "n_positions": n_pos,
        "sweep_duration_s": sweep_dur, "slot_duration_s": slot_dur,
        "nav_trigger_ms": nav, "data_trigger_ms": data,
        "navigator_fixed": bool(fixed), "role_stats": {str(k): v for k, v in stats.items()},
        "order": order_name, "reversed": bool(args.reverse), "order_scores": scores,
    }
    json_path = args.out.rsplit(".", 1)[0] + ".json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"{len(rows)} slots -> {args.out}  (summary: {json_path})")


if __name__ == "__main__":
    main()