"""
Inspect the paired MRI folder ("Mag (0019)", "Mag (0020)", ...) before
building a loader for it.

python inspect_paired_sequence.py --mri_dir /path/to/mag_folder \
    [--latent_dir /path/to/output/latents/<sequence_name>]

Prints: number blocks (runs of consecutive numbers), what one file contains
(2-D slice or 3-D volume), DICOM timing / slice-position tags if present,
and how many camera latent files there are to compare.
"""

import argparse
import os
import re
from collections import Counter


def file_number(name: str):
    m = re.search(r"\((\d+)\)", name) or re.search(r"(\d+)(?!.*\d)", name)
    return int(m.group(1)) if m else None


def runs(numbers):
    blocks, start, prev = [], numbers[0], numbers[0]
    for n in numbers[1:]:
        if n != prev + 1:
            blocks.append((start, prev))
            start = n
        prev = n
    blocks.append((start, prev))
    return blocks


def read_info(path):
    """Shape + key DICOM tags, trying DICOM first, then NIfTI."""
    try:
        import pydicom
        ds = pydicom.dcmread(path, force=True)
        info = {"format": "DICOM", "shape": getattr(ds, "pixel_array", None).shape}
        for tag in ("InstanceNumber", "AcquisitionTime", "ContentTime", "TriggerTime",
                    "SliceLocation", "ImageType", "SeriesDescription", "SeriesNumber",
                    "NumberOfFrames"):
            if tag in ds:
                info[tag] = str(ds.get(tag))
        if "ImagePositionPatient" in ds:
            info["ImagePositionPatient"] = [round(float(v), 2) for v in ds.ImagePositionPatient]
        return info
    except Exception:
        pass
    try:
        import nibabel as nib
        img = nib.load(path)
        return {"format": "NIfTI", "shape": img.shape, "zooms": img.header.get_zooms()}
    except Exception as exc:
        return {"format": f"unreadable ({type(exc).__name__})"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mri_dir", required=True)
    ap.add_argument("--latent_dir", default=None)
    ap.add_argument("--n_show", type=int, default=4, help="Files to show around each boundary")
    args = ap.parse_args()

    names = [f for f in os.listdir(args.mri_dir) if not f.startswith(".")]
    numbered = sorted((file_number(f), f) for f in names if file_number(f) is not None)
    if not numbered:
        raise SystemExit("No numbered files found.")
    numbers = [n for n, _ in numbered]
    by_number = dict(numbered)

    blocks = runs(numbers)
    sizes = Counter(b - a + 1 for a, b in blocks)
    gaps = Counter(blocks[i + 1][0] - blocks[i][1] - 1 for i in range(len(blocks) - 1))
    print(f"MRI files: {len(numbers)}  (numbers {numbers[0]} .. {numbers[-1]})")
    print(f"Blocks of consecutive numbers: {len(blocks)}")
    print(f"  block sizes -> count: {dict(sizes)}")
    print(f"  gap sizes   -> count: {dict(gaps)}")
    print(f"  first blocks: {blocks[:4]}")

    # Files at the start, the end of block 1 and the start of block 2.
    a1, b1 = blocks[0]
    probe = numbers[: args.n_show]
    probe += [n for n in range(b1 - 1, b1 + 1) if n in by_number]
    if len(blocks) > 1:
        a2 = blocks[1][0]
        probe += [n for n in range(a2, a2 + 2) if n in by_number]
    print("\nFile contents / tags:")
    for n in dict.fromkeys(probe):
        print(f"  {by_number[n]}: {read_info(os.path.join(args.mri_dir, by_number[n]))}")

    if args.latent_dir:
        lat = [f for f in os.listdir(args.latent_dir) if f.endswith((".pt", ".npy", ".npz"))]
        lat_numbers = sorted(n for n in (file_number(f) for f in lat) if n is not None)
        print(f"\nCamera latent files: {len(lat)}")
        if lat_numbers:
            print(f"  numbers {lat_numbers[0]} .. {lat_numbers[-1]}, "
                  f"{len(runs(lat_numbers))} block(s) of consecutive numbers")
        print(f"  MRI files / latent files = {len(numbers)} / {len(lat)}")


if __name__ == "__main__":
    main()