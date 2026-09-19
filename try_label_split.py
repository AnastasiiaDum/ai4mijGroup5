import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import h_maxima
from skimage.segmentation import watershed
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def propose_split(gt, spacing, prominence_mm=2.0):
    """Uses merged GT only. Group watershed basins by maximum interior depth.

    Hypothesis: narrower basins are label 1, wider basins label 4.
    This class assignment is an assumption, not an anatomical guarantee.
    """
    mask = gt == 1
    points = np.argwhere(mask)
    if not len(points):
        raise ValueError("No label 1 to split.")
    low = np.maximum(points.min(0) - 2, 0)
    high = np.minimum(points.max(0) + 3, gt.shape)
    box = tuple(slice(int(a), int(b)) for a, b in zip(low, high))
    small_mask = mask[box]
    # Padding supplies an explicit background outside the volume boundary.
    padded = np.pad(small_mask, 1)
    distance = ndi.distance_transform_edt(padded, sampling=spacing)[1:-1, 1:-1, 1:-1]
    peaks = h_maxima(distance, prominence_mm)
    markers, count = ndi.label(peaks)
    if count < 2:
        raise ValueError("Fewer than two seed regions: do not force a split.")
    basins = watershed(-distance, markers, mask=small_mask)
    if not np.all(basins[small_mask] > 0):
        raise ValueError("Unassigned merged voxels.")
    depths = np.array([distance[basins == i].max() for i in range(1, count + 1)])
    ordered = np.sort(depths)
    gaps = np.diff(ordered)
    if gaps.max() <= 0:
        raise ValueError("No depth separation between basins.")
    gap_index = int(np.argmax(gaps))
    cutoff = float((ordered[gap_index] + ordered[gap_index + 1]) / 2)
    prediction = gt.copy()
    crop = prediction[box]
    basin_report = []
    for i, depth in enumerate(depths, 1):
        label = 1 if depth < cutoff else 4
        region = basins == i
        crop[region] = label
        basin_report.append({"basin": i, "voxels": int(region.sum()),
                             "maximum_depth_mm": float(depth), "assigned_label": label})
    if not np.array_equal(prediction[~mask], gt[~mask]):
        raise ValueError("Unexpected change outside merged region.")
    if not np.array_equal((prediction == 1) | (prediction == 4), mask):
        raise ValueError("Combined region changed.")
    return prediction, {"prominence_mm": prominence_mm, "cutoff_mm": cutoff,
                        "basins": basin_report, "combined_region_preserved": True,
                        "other_labels_unchanged": True}


def load_labels(path):
    image = nib.load(str(path))
    values = np.asarray(image.dataobj)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("Expected finite 3D labels.")
    if not np.isin(values, [0, 1, 2, 3, 4]).all():
        raise ValueError("Unexpected label values.")
    return image, values.astype(np.uint8)


def check_geometry(a, b):
    if (a.shape != b.shape or not np.allclose(a.affine, b.affine)
            or a.header.get_xyzt_units()[0] != b.header.get_xyzt_units()[0]):
        raise ValueError("Images have different geometry.")


def dice(a, b, label):
    x, y = a == label, b == label
    denominator = int(x.sum() + y.sum())
    return None if denominator == 0 else float(2 * np.count_nonzero(x & y) / denominator)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("label_split_trial"))
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit("Output directory exists. Choose a new --output-dir; nothing overwritten.")
    source = args.patient_dir / "GT.nii.gz"
    image, gt = load_labels(source)
    if image.header.get_xyzt_units()[0] != "mm":
        raise ValueError("This experiment expects spacing in millimetres.")
    if np.any(gt == 4):
        raise ValueError("Label 4 already exists: inspect this dataset before using this method.")

    # Prediction is complete BEFORE GT2 is opened.
    predicted, report = propose_split(gt, image.header.get_zooms()[:3])
    report["status"] = "EXPERIMENTAL: review before any training use"
    report["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    report["versions"] = {p: version(p) for p in ["numpy", "scipy", "scikit-image", "nibabel"]}
    reference_path = args.patient_dir / "GT2.nii.gz"
    reference = None
    if reference_path.exists():
        reference_image, reference = load_labels(reference_path)
        check_geometry(image, reference_image)
        report["reference_sha256"] = hashlib.sha256(reference_path.read_bytes()).hexdigest()
        report["reference_union_matches"] = bool(np.array_equal(gt == 1, (reference == 1) | (reference == 4)))
        report["dice_original"] = {str(k): dice(gt, reference, k) for k in [1, 4]}
        report["dice_candidate"] = {str(k): dice(predicted, reference, k) for k in [1, 4]}
        report["different_voxels"] = int(np.count_nonzero(predicted != reference))
        report["evaluation_note"] = "Patient 07 development check, not independent validation or ENet performance."

    args.output_dir.mkdir(parents=True)
    header = image.header.copy()
    header.set_data_dtype(np.uint8)
    result = nib.Nifti1Image(predicted, image.affine, header)
    qform, qcode = image.get_qform(coded=True)
    sform, scode = image.get_sform(coded=True)
    result.set_qform(qform, int(qcode))
    result.set_sform(sform, int(scode))
    target = args.output_dir / "GT_candidate_DO_NOT_REPLACE_ORIGINAL.nii.gz"
    nib.save(result, str(target))
    reloaded = nib.load(str(target))
    check_geometry(image, reloaded)
    assert np.array_equal(np.asarray(reloaded.dataobj), predicted)

    if reference is not None:
        # Include known contact examples and the worst disagreement slice.
        errors_per_slice = np.count_nonzero(predicted != reference, axis=(0, 1))
        slices = sorted(set([55, 56, 65, 102, int(np.argmax(errors_per_slice))]))
        slices = [z for z in slices if z < gt.shape[2]]
        ct_path = args.patient_dir / f"{args.patient_dir.name}.nii.gz"
        ct_image = nib.load(str(ct_path))
        check_geometry(image, ct_image)
        ct = np.asarray(ct_image.dataobj)
        vmin, vmax = np.percentile(ct, [1, 99])
        fig, axes = plt.subplots(len(slices), 4, figsize=(12, 3 * len(slices)), squeeze=False)
        cmap = ListedColormap(["cyan", "magenta"])
        for row, z in enumerate(slices):
            coords = np.argwhere(gt[:, :, z] == 1)
            lo = np.maximum(coords.min(0) - 12, 0)
            hi = np.minimum(coords.max(0) + 13, gt.shape[:2])
            sl = (slice(lo[0], hi[0]), slice(lo[1], hi[1]), z)
            for col, values in enumerate([gt[sl], predicted[sl], reference[sl]]):
                ax = axes[row, col]
                ax.imshow(ct[sl].T, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
                show = np.zeros(values.shape, np.uint8)
                show[values == 1] = 1
                show[values == 4] = 2
                ax.imshow(np.ma.masked_where(show.T == 0, show.T), cmap=cmap,
                          vmin=.5, vmax=2.5, alpha=.5, origin="lower", interpolation="nearest")
            axes[row, 3].imshow((predicted[sl] != reference[sl]).T, cmap="gray",
                                vmin=0, vmax=1, origin="lower", interpolation="nearest")
            for col, title in enumerate(["Original GT", "Candidate", "Reference GT2", "White = disagreement"]):
                axes[row, col].set_title(f"Slice {z}: {title}", fontsize=10)
                axes[row, col].set_aspect(float(image.header.get_zooms()[1] / image.header.get_zooms()[0]))
                axes[row, col].axis("off")
        fig.suptitle("Patient 07 DEVELOPMENT CHECK | cyan: label 1, magenta: label 4\nNot an independently validated repair or model prediction")
        fig.tight_layout(rect=(0, 0, 1, .95))
        fig.savefig(args.output_dir / "comparison.png", dpi=130)
        plt.close(fig)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Saved experiment to {args.output_dir.resolve()}. Originals unchanged.")


if __name__ == "__main__":
    main()