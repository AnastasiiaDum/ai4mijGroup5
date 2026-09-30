"""Exploratory data analysis for the clean SegTHOR training dataset.

The script never modifies the NIfTI input files. It writes reproducible CSV
summaries, dataset-level plots, and one CT/ground-truth overlay per patient.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import nibabel as nib
import numpy as np


LABEL_NAMES = {
    0: "background",
    1: "esophagus",
    2: "heart",
    3: "trachea",
    4: "aorta",
}

LABEL_COLORS = {
    1: "#00D7D7",
    2: "#FF4D4D",
    3: "#59D65F",
    4: "#E44DE1",
}

EXPECTED_LABELS = set(LABEL_NAMES)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reproducible EDA on the clean SegTHOR dataset."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/segthor_train_full/train"),
        help="Folder containing Patient_XX directories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eda_clean_outputs"),
        help="Destination for CSV files and figures.",
    )
    parser.add_argument(
        "--sample_voxels",
        type=int,
        default=200_000,
        help="Maximum number of CT voxels sampled per patient for the histogram.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path}")

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_overlay(
    ct_data: np.ndarray,
    gt_data: np.ndarray,
    patient: str,
    output_path: Path,
) -> None:
    foreground_per_slice = np.count_nonzero(gt_data > 0, axis=(0, 1))
    slice_index = int(np.argmax(foreground_per_slice))

    ct_slice = ct_data[:, :, slice_index].T
    gt_slice = gt_data[:, :, slice_index].T

    lower, upper = np.percentile(ct_slice, [1, 99])
    if upper <= lower:
        lower, upper = float(ct_slice.min()), float(ct_slice.max())

    masked_labels = np.ma.masked_where(gt_slice == 0, gt_slice)
    overlay_cmap = ListedColormap(
        [
            LABEL_COLORS[1],
            LABEL_COLORS[2],
            LABEL_COLORS[3],
            LABEL_COLORS[4],
        ]
    )

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(ct_slice, cmap="gray", vmin=lower, vmax=upper, origin="lower")
    ax.imshow(
        masked_labels,
        cmap=overlay_cmap,
        vmin=1,
        vmax=4,
        alpha=0.45,
        interpolation="nearest",
        origin="lower",
    )
    ax.set_title(f"{patient} — slice {slice_index}")
    ax.axis("off")
    ax.legend(
        handles=[
            Patch(color=LABEL_COLORS[label], label=LABEL_NAMES[label])
            for label in range(1, 5)
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_slice_geometry(patient_rows: list[dict], output: Path) -> None:
    patients = [row["patient"] for row in patient_rows]
    slices = np.array([row["slices"] for row in patient_rows], dtype=float)
    spacing_z = np.array([row["spacing_z_mm"] for row in patient_rows], dtype=float)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(spacing_z, slices, s=60, alpha=0.8)
    for patient, x_value, y_value in zip(patients, spacing_z, slices):
        ax.annotate(
            patient.replace("Patient_", "P"),
            (x_value, y_value),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )
    ax.set_xlabel("Slice spacing (mm)")
    ax.set_ylabel("Number of axial slices")
    ax.set_title("CT acquisition geometry")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_inplane_spacing(patient_rows: list[dict], output: Path) -> None:
    patients = [row["patient"].replace("Patient_", "P") for row in patient_rows]
    spacing_x = np.array(
        [row["spacing_x_mm"] for row in patient_rows], dtype=float
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(patients, spacing_x, color="#3274A1")
    ax.axhline(
        np.median(spacing_x),
        color="#E1812C",
        linestyle="--",
        label=f"Median = {np.median(spacing_x):.4f} mm",
    )
    ax.set_xlabel("Patient")
    ax.set_ylabel("In-plane spacing X (mm)")
    ax.set_title("Variation in in-plane voxel spacing")
    ax.tick_params(axis="x", rotation=70, labelsize=7)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_intensity_ranges(patient_rows: list[dict], output: Path) -> None:
    patients = [row["patient"].replace("Patient_", "P") for row in patient_rows]
    x_values = np.arange(len(patients))
    p01 = np.array([row["ct_p0_1_hu"] for row in patient_rows], dtype=float)
    p1 = np.array([row["ct_p1_hu"] for row in patient_rows], dtype=float)
    p99 = np.array([row["ct_p99_hu"] for row in patient_rows], dtype=float)
    p999 = np.array([row["ct_p99_9_hu"] for row in patient_rows], dtype=float)

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.vlines(x_values, p01, p999, color="#B8B8B8", linewidth=3, label="0.1–99.9%")
    ax.vlines(x_values, p1, p99, color="#3274A1", linewidth=6, label="1–99%")
    ax.scatter(x_values, p1, color="#173F5F", s=18)
    ax.scatter(x_values, p99, color="#173F5F", s=18)
    ax.set_xticks(x_values)
    ax.set_xticklabels(patients, rotation=70, fontsize=7)
    ax.set_ylabel("CT intensity (HU)")
    ax.set_title("Robust CT intensity ranges across patients")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_intensity_histogram(samples: list[np.ndarray], output: Path) -> None:
    values = np.concatenate(samples)
    display_values = values[(values >= -1200) & (values <= 2000)]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(display_values, bins=250, color="#3274A1", alpha=0.85)
    ax.set_yscale("log")
    ax.set_xlabel("CT intensity (HU)")
    ax.set_ylabel("Sampled voxel count (log scale)")
    ax.set_title("Sampled CT intensity distribution")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_organ_volumes(label_rows: list[dict], output: Path) -> None:
    foreground_rows = [row for row in label_rows if row["label"] != 0]
    data = [
        [
            row["volume_ml"]
            for row in foreground_rows
            if row["label"] == label
        ]
        for label in range(1, 5)
    ]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.boxplot(data)
    ax.set_xticks(range(1, 5))
    ax.set_xticklabels([LABEL_NAMES[label] for label in range(1, 5)])
    ax.set_ylabel("Physical volume (mL)")
    ax.set_title("Organ-volume distribution across patients")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_class_imbalance(label_rows: list[dict], output: Path) -> None:
    percentages = []
    for label in range(5):
        label_values = [
            row["voxel_percentage"]
            for row in label_rows
            if row["label"] == label
        ]
        percentages.append(float(np.median(label_values)))

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(
        [LABEL_NAMES[label] for label in range(5)],
        percentages,
        color=["#777777", "#00D7D7", "#FF4D4D", "#59D65F", "#E44DE1"],
    )
    ax.set_yscale("log")
    ax.set_ylabel("Median percentage of voxels (log scale)")
    ax.set_title("Segmentation class imbalance")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    output = args.output.resolve()
    overlays = output / "overlays"
    output.mkdir(parents=True, exist_ok=True)
    overlays.mkdir(parents=True, exist_ok=True)

    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    patients = sorted(path for path in root.glob("Patient_*") if path.is_dir())
    if not patients:
        raise RuntimeError(f"No Patient_* directories found in {root}")

    patient_rows: list[dict] = []
    label_rows: list[dict] = []
    organ_intensity_rows: list[dict] = []
    histogram_samples: list[np.ndarray] = []
    spacing_counts: Counter = Counter()
    problems: list[str] = []
    rng = np.random.default_rng(args.seed)

    print(f"Found {len(patients)} patients in {root}")

    for index, patient_dir in enumerate(patients, start=1):
        patient = patient_dir.name
        print(f"[{index:02d}/{len(patients)}] Analysing {patient}...")

        ct_path = patient_dir / f"{patient}.nii.gz"
        gt_path = patient_dir / "GT.nii.gz"

        if not ct_path.exists() or not gt_path.exists():
            problems.append(f"{patient}: missing CT or GT file")
            continue

        ct_nifti = nib.load(str(ct_path))
        gt_nifti = nib.load(str(gt_path))

        if ct_nifti.shape != gt_nifti.shape:
            problems.append(
                f"{patient}: CT shape {ct_nifti.shape} != GT shape {gt_nifti.shape}"
            )
            continue

        if not np.allclose(ct_nifti.affine, gt_nifti.affine):
            problems.append(f"{patient}: CT and GT affine matrices differ")
            continue

        ct_data = np.asanyarray(ct_nifti.dataobj)
        gt_data = np.asanyarray(gt_nifti.dataobj).astype(np.uint8, copy=False)
        labels = set(np.unique(gt_data).astype(int).tolist())

        if labels != EXPECTED_LABELS:
            problems.append(
                f"{patient}: labels {sorted(labels)}, expected {sorted(EXPECTED_LABELS)}"
            )
            continue

        shape_x, shape_y, shape_z = ct_nifti.shape
        spacing_x, spacing_y, spacing_z = (
            float(value) for value in ct_nifti.header.get_zooms()[:3]
        )
        voxel_volume_mm3 = spacing_x * spacing_y * spacing_z
        spacing_counts[(spacing_x, spacing_y, spacing_z)] += 1

        percentiles = np.percentile(
            ct_data, [0.1, 1, 5, 50, 95, 99, 99.9]
        )
        patient_rows.append(
            {
                "patient": patient,
                "shape_x": shape_x,
                "shape_y": shape_y,
                "slices": shape_z,
                "spacing_x_mm": spacing_x,
                "spacing_y_mm": spacing_y,
                "spacing_z_mm": spacing_z,
                "physical_x_mm": shape_x * spacing_x,
                "physical_y_mm": shape_y * spacing_y,
                "physical_z_mm": shape_z * spacing_z,
                "ct_min_hu": int(ct_data.min()),
                "ct_p0_1_hu": float(percentiles[0]),
                "ct_p1_hu": float(percentiles[1]),
                "ct_p5_hu": float(percentiles[2]),
                "ct_median_hu": float(percentiles[3]),
                "ct_p95_hu": float(percentiles[4]),
                "ct_p99_hu": float(percentiles[5]),
                "ct_p99_9_hu": float(percentiles[6]),
                "ct_max_hu": int(ct_data.max()),
            }
        )

        total_voxels = int(gt_data.size)
        for label, label_name in LABEL_NAMES.items():
            mask = gt_data == label
            voxel_count = int(np.count_nonzero(mask))
            slices_with_label = int(np.count_nonzero(np.any(mask, axis=(0, 1))))

            label_rows.append(
                {
                    "patient": patient,
                    "label": label,
                    "label_name": label_name,
                    "voxel_count": voxel_count,
                    "voxel_percentage": 100.0 * voxel_count / total_voxels,
                    "volume_ml": voxel_count * voxel_volume_mm3 / 1000.0,
                    "slices_with_label": slices_with_label,
                }
            )

            if label > 0:
                organ_values = ct_data[mask]
                organ_percentiles = np.percentile(organ_values, [1, 50, 99])
                organ_intensity_rows.append(
                    {
                        "patient": patient,
                        "label": label,
                        "label_name": label_name,
                        "mean_hu": float(np.mean(organ_values)),
                        "std_hu": float(np.std(organ_values)),
                        "p1_hu": float(organ_percentiles[0]),
                        "median_hu": float(organ_percentiles[1]),
                        "p99_hu": float(organ_percentiles[2]),
                    }
                )

        flat_ct = ct_data.reshape(-1)
        sample_size = min(args.sample_voxels, flat_ct.size)
        # Sampling with replacement avoids allocating a permutation of the
        # complete volume, which would be unnecessarily memory intensive.
        sample_indices = rng.integers(0, flat_ct.size, size=sample_size)
        histogram_samples.append(flat_ct[sample_indices].astype(np.float32))

        save_overlay(
            ct_data,
            gt_data,
            patient,
            overlays / f"{patient}_overlay.png",
        )

        del ct_data, gt_data

    if problems:
        problem_text = "\n".join(f"- {problem}" for problem in problems)
        raise RuntimeError(f"EDA stopped because integrity checks failed:\n{problem_text}")

    write_csv(output / "patient_summary.csv", patient_rows)
    write_csv(output / "label_summary.csv", label_rows)
    write_csv(output / "organ_intensity_summary.csv", organ_intensity_rows)

    plot_slice_geometry(patient_rows, output / "acquisition_geometry.png")
    plot_inplane_spacing(patient_rows, output / "inplane_spacing.png")
    plot_intensity_ranges(patient_rows, output / "ct_intensity_ranges.png")
    plot_intensity_histogram(
        histogram_samples, output / "ct_intensity_histogram.png"
    )
    plot_organ_volumes(label_rows, output / "organ_volume_distribution.png")
    plot_class_imbalance(label_rows, output / "class_imbalance.png")

    with (output / "eda_summary.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"Patients analysed: {len(patient_rows)}\n")
        handle.write("Expected labels: 0, 1, 2, 3, 4\n")
        handle.write("Integrity checks: PASS\n\n")
        handle.write("Spacing groups:\n")
        for spacing, count in sorted(spacing_counts.items()):
            handle.write(f"  {spacing}: {count} patients\n")

    print(f"\nPASS: EDA completed for {len(patient_rows)} patients.")
    print(f"Results saved to: {output}")


if __name__ == "__main__":
    main(get_args())
