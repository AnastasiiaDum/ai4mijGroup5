import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
import shutil

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import h_maxima
from skimage.segmentation import watershed


def propose_split(gt, spacing, prominence_mm=2.0):
    """
    Uses the merged GT only.

    Hypothesis:
      - narrower watershed basins -> label 1
      - wider watershed basins   -> label 4

    This is the same algorithm used in try_label_split.py.
    """

    mask = gt == 1

    points = np.argwhere(mask)
    if not len(points):
        raise ValueError("No label 1 to split.")

    low = np.maximum(points.min(0) - 2, 0)
    high = np.minimum(points.max(0) + 3, gt.shape)

    box = tuple(
        slice(int(a), int(b))
        for a, b in zip(low, high)
    )

    small_mask = mask[box]

    # Padding supplies explicit background outside the volume boundary.
    padded = np.pad(small_mask, 1)

    distance = ndi.distance_transform_edt(
        padded,
        sampling=spacing
    )[1:-1, 1:-1, 1:-1]

    peaks = h_maxima(distance, prominence_mm)

    markers, count = ndi.label(peaks)

    if count < 2:
        raise ValueError(
            "Fewer than two seed regions: do not force a split."
        )

    basins = watershed(
        -distance,
        markers,
        mask=small_mask
    )

    if not np.all(basins[small_mask] > 0):
        raise ValueError("Unassigned merged voxels.")

    depths = np.array([
        distance[basins == i].max()
        for i in range(1, count + 1)
    ])

    ordered = np.sort(depths)
    gaps = np.diff(ordered)

    if gaps.max() <= 0:
        raise ValueError(
            "No depth separation between basins."
        )

    gap_index = int(np.argmax(gaps))

    cutoff = float(
        (ordered[gap_index] + ordered[gap_index + 1]) / 2
    )

    prediction = gt.copy()

    crop = prediction[box]

    basin_report = []

    for i, depth in enumerate(depths, 1):

        label = 1 if depth < cutoff else 4

        region = basins == i

        crop[region] = label

        basin_report.append({
            "basin": i,
            "voxels": int(region.sum()),
            "maximum_depth_mm": float(depth),
            "assigned_label": label
        })

    # Make sure nothing outside the original merged region changed.
    if not np.array_equal(
        prediction[~mask],
        gt[~mask]
    ):
        raise ValueError(
            "Unexpected change outside merged region."
        )

    # Make sure the original combined region was preserved.
    if not np.array_equal(
        (prediction == 1) | (prediction == 4),
        mask
    ):
        raise ValueError(
            "Combined region changed."
        )

    return prediction, {
        "prominence_mm": prominence_mm,
        "cutoff_mm": cutoff,
        "basins": basin_report,
        "combined_region_preserved": True,
        "other_labels_unchanged": True
    }


def load_labels(path):
    image = nib.load(str(path))

    values = np.asarray(image.dataobj)

    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError(
            "Expected finite 3D labels."
        )

    if not np.isin(
        values,
        [0, 1, 2, 3, 4]
    ).all():
        raise ValueError(
            "Unexpected label values."
        )

    return image, values.astype(np.uint8)


def check_geometry(a, b):
    if (
        a.shape != b.shape
        or not np.allclose(a.affine, b.affine)
        or a.header.get_xyzt_units()[0]
        != b.header.get_xyzt_units()[0]
    ):
        raise ValueError(
            "Images have different geometry."
        )


def dice(a, b, label):
    x = a == label
    y = b == label

    denominator = int(
        x.sum() + y.sum()
    )

    if denominator == 0:
        return None

    return float(
        2 * np.count_nonzero(x & y)
        / denominator
    )


def process_patient(
    patient_dir,
    output_dir,
    prominence_mm=2.0
):
    """
    Process one patient.

    Original files are never modified.
    """

    patient_name = patient_dir.name

    source_gt = patient_dir / "GT.nii.gz"
    source_ct = patient_dir / f"{patient_name}.nii.gz"
    source_gt2 = patient_dir / "GT2.nii.gz"

    if not source_gt.exists():
        raise FileNotFoundError(
            f"Missing GT: {source_gt}"
        )

    if not source_ct.exists():
        raise FileNotFoundError(
            f"Missing CT: {source_ct}"
        )

    print(f"\nProcessing {patient_name}...")

    # Load original GT.
    image, gt = load_labels(source_gt)

    # This experiment expects mm spacing.
    if image.header.get_xyzt_units()[0] != "mm":
        raise ValueError(
            f"{patient_name}: "
            "This experiment expects spacing in millimetres."
        )

    # The algorithm expects label 1 to be the merged region.
    # If label 4 already exists, stop rather than silently modifying it.
    if np.any(gt == 4):
        raise ValueError(
            f"{patient_name}: "
            "Label 4 already exists. "
            "Inspect this patient before using this method."
        )

    spacing = image.header.get_zooms()[:3]

    # IMPORTANT:
    # Prediction is generated without looking at GT2.
    predicted, report = propose_split(
        gt,
        spacing,
        prominence_mm=prominence_mm
    )

    report["patient"] = patient_name
    report["status"] = (
        "EXPERIMENTAL: review before training use"
    )

    report["source_sha256"] = (
        hashlib.sha256(
            source_gt.read_bytes()
        ).hexdigest()
    )

    report["versions"] = {
        package: version(package)
        for package in [
            "numpy",
            "scipy",
            "scikit-image",
            "nibabel"
        ]
    }

    report["original_labels"] = sorted(
        np.unique(gt).astype(int).tolist()
    )

    report["corrected_labels"] = sorted(
        np.unique(predicted).astype(int).tolist()
    )

    # ---------------------------------------------------------
    # Optional Patient 07 reference comparison.
    # ---------------------------------------------------------

    reference = None

    if source_gt2.exists():

        print("  GT2 found - performing reference comparison.")

        reference_image, reference = load_labels(
            source_gt2
        )

        check_geometry(
            image,
            reference_image
        )

        report["reference_sha256"] = (
            hashlib.sha256(
                source_gt2.read_bytes()
            ).hexdigest()
        )

        report["reference_union_matches"] = bool(
            np.array_equal(
                gt == 1,
                (reference == 1)
                | (reference == 4)
            )
        )

        report["dice_original"] = {
            str(k): dice(gt, reference, k)
            for k in [1, 4]
        }

        report["dice_candidate"] = {
            str(k): dice(predicted, reference, k)
            for k in [1, 4]
        }

        report["different_voxels"] = int(
            np.count_nonzero(
                predicted != reference
            )
        )

        report["evaluation_note"] = (
            "Patient 07 development check, "
            "not independent validation or "
            "ENet performance."
        )

        print(
            "  Candidate vs GT2:"
            f" label 1 Dice = "
            f"{report['dice_candidate']['1']:.6f},"
            f" label 4 Dice = "
            f"{report['dice_candidate']['4']:.6f},"
            f" differing voxels = "
            f"{report['different_voxels']}"
        )

    # ---------------------------------------------------------
    # Create output patient directory.
    # ---------------------------------------------------------

    patient_output = output_dir / patient_name

    patient_output.mkdir(
        parents=True,
        exist_ok=False
    )

    # Copy CT unchanged.
    shutil.copy2(
        source_ct,
        patient_output / source_ct.name
    )

    # Copy GT2 if it exists.
    if source_gt2.exists():

        shutil.copy2(
            source_gt2,
            patient_output / "GT2.nii.gz"
        )

    # ---------------------------------------------------------
    # Save corrected GT.
    # ---------------------------------------------------------

    header = image.header.copy()
    header.set_data_dtype(np.uint8)

    result = nib.Nifti1Image(
        predicted,
        image.affine,
        header
    )

    qform, qcode = image.get_qform(
        coded=True
    )

    sform, scode = image.get_sform(
        coded=True
    )

    result.set_qform(
        qform,
        int(qcode)
    )

    result.set_sform(
        sform,
        int(scode)
    )

    corrected_gt = patient_output / "GT.nii.gz"

    nib.save(
        result,
        str(corrected_gt)
    )

    # ---------------------------------------------------------
    # Reload and verify corrected file.
    # ---------------------------------------------------------

    reloaded = nib.load(
        str(corrected_gt)
    )

    check_geometry(
        image,
        reloaded
    )

    reloaded_data = np.asarray(
        reloaded.dataobj
    )

    if not np.array_equal(
        reloaded_data,
        predicted
    ):
        raise ValueError(
            f"{patient_name}: "
            "Saved GT does not match prediction."
        )

    # Verify labels.
    if not np.isin(
        reloaded_data,
        [0, 1, 2, 3, 4]
    ).all():
        raise ValueError(
            f"{patient_name}: "
            "Corrected GT contains invalid labels."
        )

    # ---------------------------------------------------------
    # Save report.
    # ---------------------------------------------------------

    report["output_gt"] = str(
        corrected_gt
    )

    report["original_unchanged"] = True

    report_path = (
        patient_output
        / "label_split_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        f"  Saved corrected GT:"
        f" {corrected_gt}"
    )

    return report


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Apply the validated experimental "
            "label-1 -> label-1/4 watershed split "
            "to all SEGTHOR patients."
        )
    )

    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path(
            "data/data/segthor_part1/train"
        ),
        help=(
            "Directory containing Patient_01, "
            "Patient_02, etc."
        )
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/segthor_corrected/train"
        ),
        help=(
            "Directory where corrected copies "
            "will be written."
        )
    )

    parser.add_argument(
        "--prominence-mm",
        type=float,
        default=2.0,
        help=(
            "H-maxima prominence in millimetres."
        )
    )

    args = parser.parse_args()

    source_dir = args.source_dir
    output_dir = args.output_dir

    if not source_dir.exists():
        raise SystemExit(
            f"Source directory does not exist:\n"
            f"{source_dir.resolve()}"
        )

    # IMPORTANT:
    # Refuse to use an existing output directory.
    # This prevents accidental mixing of old/new results.
    if output_dir.exists():
        raise SystemExit(
            f"Output directory already exists:\n"
            f"{output_dir.resolve()}\n\n"
            "Delete it manually if you want "
            "to regenerate the corrected dataset."
        )

    output_dir.mkdir(
        parents=True
    )

    patients = sorted(
        [
            p
            for p in source_dir.iterdir()
            if p.is_dir()
            and p.name.startswith("Patient_")
        ]
    )

    if not patients:
        raise SystemExit(
            f"No Patient_* directories found in:\n"
            f"{source_dir.resolve()}"
        )

    print(
        f"Found {len(patients)} patients."
    )

    reports = []
    failures = []

    for patient_dir in patients:

        try:

            report = process_patient(
                patient_dir,
                output_dir,
                prominence_mm=args.prominence_mm
            )

            reports.append(report)

        except Exception as exc:

            print(
                f"  ERROR: {patient_dir.name}: {exc}"
            )

            failures.append({
                "patient": patient_dir.name,
                "error": str(exc)
            })

    # ---------------------------------------------------------
    # Overall report.
    # ---------------------------------------------------------

    overall_report = {
        "source_dir": str(
            source_dir.resolve()
        ),
        "output_dir": str(
            output_dir.resolve()
        ),
        "prominence_mm": args.prominence_mm,
        "patients_found": len(patients),
        "patients_successful": len(reports),
        "patients_failed": len(failures),
        "successful_patients": [
            r["patient"]
            for r in reports
        ],
        "failed_patients": failures,
        "status": (
            "EXPERIMENTAL: corrected labels "
            "should be reviewed before training."
        )
    }

    overall_path = (
        output_dir.parent
        / "label_split_all_patients_report.json"
    )

    overall_path.write_text(
        json.dumps(
            overall_report,
            indent=2
        ),
        encoding="utf-8"
    )

    print("\n" + "=" * 60)
    print("FINISHED")
    print("=" * 60)

    print(
        f"Patients found:      {len(patients)}"
    )

    print(
        f"Patients successful: {len(reports)}"
    )

    print(
        f"Patients failed:     {len(failures)}"
    )

    print(
        f"\nCorrected dataset:"
        f"\n{output_dir.resolve()}"
    )

    print(
        f"\nOverall report:"
        f"\n{overall_path.resolve()}"
    )

    if failures:

        print("\nFAILED PATIENTS:")

        for failure in failures:
            print(
                f"  {failure['patient']}: "
                f"{failure['error']}"
            )

        print(
            "\nSome patients failed. "
            "Do not proceed to training until "
            "you inspect those failures."
        )

    else:

        print(
            "\nAll patients processed successfully."
        )

        print(
            "\nOriginal raw GT files were NOT modified."
        )


if __name__ == "__main__":
    main()