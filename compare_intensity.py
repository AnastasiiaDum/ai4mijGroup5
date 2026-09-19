from pathlib import Path
import csv

import nibabel as nib
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Experimental settings—not established best settings.
LOW_PERCENTILE = 1
HIGH_PERCENTILE = 99

PATIENT_IDS = ["Patient_01", "Patient_02", "Patient_03"]


# Ask where the original patient folders are stored.
folder_text = input(
    "Paste the path to the train folder, then press Enter: "
).strip()

data_folder = Path(folder_text).expanduser()

if not data_folder.is_dir():
    raise SystemExit("Folder not found. Check the path and run again.")


# Create a separate folder for comparison results.
output_folder = (
    Path(__file__).resolve().parent / "preprocessing_comparison"
)
output_folder.mkdir(exist_ok=True)

rows = []


for patient_id in PATIENT_IDS:
    print(f"Comparing {patient_id}...")

    scan_path = (
        data_folder / patient_id / f"{patient_id}.nii.gz"
    )

    if not scan_path.is_file():
        raise SystemExit(f"Scan not found: {scan_path}")

    scan_image = nib.load(str(scan_path))
    scan = scan_image.get_fdata(dtype=np.float32)

    if scan.ndim != 3:
        raise SystemExit(f"{patient_id}: expected a 3D scan.")

    if not np.isfinite(scan).all():
        raise SystemExit(f"{patient_id}: invalid numeric values found.")

    spacing = scan_image.header.get_zooms()[:3]

    # Calculate scaling limits using the complete scan.
    scan_min = float(scan.min())
    scan_max = float(scan.max())

    low, high = np.percentile(
        scan, [LOW_PERCENTILE, HIGH_PERCENTILE]
    )
    low = float(low)
    high = float(high)

    if scan_max <= scan_min or high <= low:
        raise SystemExit(
            f"{patient_id}: not enough intensity variation to scale."
        )

    # Measure how many values are outside the clipping limits.
    below_percent = float(
        100 * np.count_nonzero(scan < low) / scan.size
    )
    above_percent = float(
        100 * np.count_nonzero(scan > high) / scan.size
    )

    rows.append({
        "patient": patient_id,
        "baseline_min": scan_min,
        "baseline_max": scan_max,
        "candidate_low": low,
        "candidate_high": high,
        "percent_below_low": below_percent,
        "percent_above_high": above_percent,
    })

    # Choose three positions through the scan.
    # We do not use the reference labels to choose slices.
    slice_indices = sorted({
        int(round(fraction * (scan.shape[2] - 1)))
        for fraction in [0.25, 0.50, 0.75]
    })

    fig, axes = plt.subplots(
        len(slice_indices),
        3,
        figsize=(12, 4 * len(slice_indices)),
        squeeze=False,
    )

    for row_index, slice_index in enumerate(slice_indices):
        original_slice = scan[:, :, slice_index]

        # Method A: minimum–maximum scaling.
        baseline_slice = (
            (original_slice - scan_min) / (scan_max - scan_min)
        )

        # Method B: percentile clipping, then scaling.
        clipped_slice = np.clip(original_slice, low, high)
        candidate_slice = (
            (clipped_slice - low) / (high - low)
        )

        # Identify locations where clipping changes the values.
        clipped_locations = (
            (original_slice < low) | (original_slice > high)
        )

        axes[row_index, 0].imshow(
            baseline_slice.T,
            cmap="gray",
            vmin=0,
            vmax=1,
            origin="lower",
            interpolation="nearest",
        )
        axes[row_index, 0].set_title(
            f"Baseline scaling — slice {slice_index}"
        )

        axes[row_index, 1].imshow(
            candidate_slice.T,
            cmap="gray",
            vmin=0,
            vmax=1,
            origin="lower",
            interpolation="nearest",
        )
        axes[row_index, 1].set_title(
            f"Clipping P{LOW_PERCENTILE}–P{HIGH_PERCENTILE} + scaling"
        )

        axes[row_index, 2].imshow(
            clipped_locations.T,
            cmap="gray",
            vmin=0,
            vmax=1,
            origin="lower",
            interpolation="nearest",
        )
        axes[row_index, 2].set_title(
            "White = values outside clipping limits"
        )

        for axis in axes[row_index]:
            axis.set_aspect(float(spacing[1] / spacing[0]))
            axis.axis("off")

    fig.suptitle(
        f"{patient_id}\n"
        f"Baseline limits: {scan_min:.1f} to {scan_max:.1f} | "
        f"Candidate limits: {low:.1f} to {high:.1f}\n"
        "Both image panels use the same display scale: 0–1",
        fontsize=12,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(
        output_folder / f"{patient_id}_comparison.png",
        dpi=150,
    )
    plt.close(fig)

    print(
        f"  Clipping limits: {low:.1f} to {high:.1f}; "
        f"below: {below_percent:.2f}%; "
        f"above: {above_percent:.2f}%"
    )


# Save the measurements in a table.
with (output_folder / "comparison_summary.csv").open(
    "w", newline=""
) as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

print("\nComparison complete.")
print(f"Results saved in: {output_folder}")
print("Original scans and labels were not changed.")