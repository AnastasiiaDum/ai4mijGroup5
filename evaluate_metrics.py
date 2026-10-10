#!/usr/bin/env python3
"""Evaluate SegTHOR PNG predictions patient-by-patient.

The script reports:
* per-class 3D Dice (computed after stacking every slice of a patient),
* per-class mean 2D Dice (empty/empty slices are excluded),
* mean foreground Dice (classes 1-4),
* relative volume difference: (predicted voxels - GT voxels) / GT voxels, and
* per-class 3D 95th-percentile Hausdorff distance (HD95) in millimetres.

It accepts masks encoded either as class IDs 0..4 or as the repository's
grayscale values 0, 63, 126, 189 and 252.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from monai.metrics import compute_hausdorff_distance
from PIL import Image


CLASS_NAMES = ("background", "esophagus", "heart", "trachea", "aorta")
N_CLASSES = len(CLASS_NAMES)
FILENAME_PATTERN = re.compile(r"^(Patient_\d+)_([0-9]+)\.png$")


def mask_to_classes(path: Path) -> np.ndarray:
    """Read a PNG mask and convert its values to class IDs 0..4."""
    array = np.asarray(Image.open(path).convert("L"))
    unique = set(np.unique(array).tolist())

    if unique.issubset(set(range(N_CLASSES))):
        classes = array.astype(np.uint8)
    else:
        classes = np.rint(array.astype(np.float32) / 63.0)
        classes = np.clip(classes, 0, N_CLASSES - 1).astype(np.uint8)

    if classes.min() < 0 or classes.max() >= N_CLASSES:
        raise ValueError(f"Invalid class values in {path}")
    return classes


def indexed_pngs(directory: Path) -> dict[str, Path]:
    """Return PNG files indexed by filename, rejecting duplicate names."""
    files: dict[str, Path] = {}
    for path in sorted(directory.rglob("*.png")):
        if path.name in files:
            raise ValueError(f"Duplicate filename found: {path.name}")
        if FILENAME_PATTERN.match(path.name):
            files[path.name] = path

    if not files:
        raise ValueError(
            f"No Patient_XX_slice.png files found inside {directory}"
        )
    return files


def group_filenames_by_patient(filenames: set[str]) -> dict[str, list[str]]:
    groups: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for filename in filenames:
        match = FILENAME_PATTERN.match(filename)
        if match is None:
            continue
        patient_id, slice_number = match.groups()
        groups[patient_id].append((int(slice_number), filename))

    return {
        patient_id: [name for _, name in sorted(items)]
        for patient_id, items in sorted(groups.items())
    }


def dice_score(prediction: np.ndarray, target: np.ndarray) -> float:
    """Dice for two Boolean masks; NaN means absent in both masks."""
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    denominator = int(prediction.sum()) + int(target.sum())
    if denominator == 0:
        return float("nan")
    intersection = int(np.logical_and(prediction, target).sum())
    return 2.0 * intersection / denominator


def relative_volume_difference(
    prediction: np.ndarray, target: np.ndarray
) -> float:
    """Signed volume error; NaN means the target class is absent."""
    predicted_volume = int(prediction.astype(bool).sum())
    target_volume = int(target.astype(bool).sum())
    if target_volume == 0:
        return float("nan")
    return (predicted_volume - target_volume) / target_volume


def hd95_score(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing: tuple[float, float, float],
) -> float:
    """Symmetric 3D HD95 in mm; undefined empty cases are explicit."""
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    prediction_present = bool(prediction.any())
    target_present = bool(target.any())

    if not prediction_present and not target_present:
        return float("nan")
    if prediction_present != target_present:
        return float("inf")

    prediction_tensor = torch.as_tensor(
        prediction[None, None], dtype=torch.float32
    )
    target_tensor = torch.as_tensor(target[None, None], dtype=torch.float32)
    result = compute_hausdorff_distance(
        y_pred=prediction_tensor,
        y=target_tensor,
        include_background=True,
        distance_metric="euclidean",
        percentile=95,
        directed=False,
        spacing=spacing,
    )
    return float(result.item())


def nanmean(values: list[float] | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    valid = array[~np.isnan(array)]
    return float(valid.mean()) if valid.size else float("nan")


def load_spacing(path: Path) -> dict[str, tuple[float, float, float]]:
    """Load and validate original patient spacing stored as (x, y, z)."""
    if not path.is_file():
        raise FileNotFoundError(f"Spacing file not found: {path}")
    with path.open("rb") as spacing_file:
        raw = pickle.load(spacing_file)
    if not isinstance(raw, dict):
        raise ValueError("Spacing file must contain a patient-to-spacing dictionary")

    result: dict[str, tuple[float, float, float]] = {}
    for patient_id, values in raw.items():
        if not isinstance(patient_id, str) or len(values) != 3:
            raise ValueError(f"Invalid spacing entry: {patient_id!r} -> {values!r}")
        spacing = tuple(float(value) for value in values)
        if not all(np.isfinite(spacing)) or not all(value > 0 for value in spacing):
            raise ValueError(f"Invalid spacing for {patient_id}: {spacing}")
        result[patient_id] = spacing
    return result


def resized_spacing_zyx(
    original_spacing_xyz: tuple[float, float, float],
    volume_shape_zyx: tuple[int, int, int],
    source_inplane_shape: tuple[int, int],
) -> tuple[float, float, float]:
    """Convert original (x,y,z) spacing to resized volume (z,y,x) spacing."""
    spacing_x, spacing_y, spacing_z = original_spacing_xyz
    _, output_height, output_width = volume_shape_zyx
    source_height, source_width = source_inplane_shape
    return (
        spacing_z,
        spacing_y * source_height / output_height,
        spacing_x * source_width / output_width,
    )


def evaluate_patient(
    filenames: list[str],
    prediction_files: dict[str, Path],
    target_files: dict[str, Path],
    original_spacing_xyz: tuple[float, float, float],
    source_inplane_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    for filename in filenames:
        prediction = mask_to_classes(prediction_files[filename])
        target = mask_to_classes(target_files[filename])
        if prediction.shape != target.shape:
            raise ValueError(
                f"Shape mismatch for {filename}: "
                f"prediction={prediction.shape}, target={target.shape}"
            )
        predictions.append(prediction)
        targets.append(target)

    prediction_3d = np.stack(predictions, axis=0)
    target_3d = np.stack(targets, axis=0)
    spacing_zyx = resized_spacing_zyx(
        original_spacing_xyz,
        prediction_3d.shape,
        source_inplane_shape,
    )

    dice_3d = np.full(N_CLASSES, np.nan, dtype=float)
    dice_2d = np.full(N_CLASSES, np.nan, dtype=float)
    rvd = np.full(N_CLASSES, np.nan, dtype=float)
    hd95 = np.full(N_CLASSES, np.nan, dtype=float)

    for class_id in range(N_CLASSES):
        pred_class_3d = prediction_3d == class_id
        target_class_3d = target_3d == class_id

        dice_3d[class_id] = dice_score(pred_class_3d, target_class_3d)
        rvd[class_id] = relative_volume_difference(
            pred_class_3d, target_class_3d
        )
        if class_id > 0:
            hd95[class_id] = hd95_score(
                pred_class_3d, target_class_3d, spacing_zyx
            )

        slice_scores = [
            dice_score(pred_slice == class_id, target_slice == class_id)
            for pred_slice, target_slice in zip(predictions, targets)
        ]
        dice_2d[class_id] = nanmean(slice_scores)

    return dice_3d, dice_2d, rvd, hd95


def write_patient_csv(
    path: Path,
    dice_3d: dict[str, np.ndarray],
    dice_2d: dict[str, np.ndarray],
    rvd: dict[str, np.ndarray],
    hd95: dict[str, np.ndarray],
) -> None:
    fieldnames = ["patient_id"]
    for metric_name in ("dice_3d", "dice_2d", "rvd", "hd95_mm"):
        fieldnames.extend(
            f"{metric_name}_{class_name}" for class_name in CLASS_NAMES[1:]
        )
    fieldnames.extend(
        (
            "mean_foreground_dice_3d",
            "mean_foreground_dice_2d",
            "mean_foreground_hd95_mm",
        )
    )

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for patient_id in sorted(dice_3d):
            row: dict[str, str | float] = {"patient_id": patient_id}
            for metric_name, values in (
                ("dice_3d", dice_3d[patient_id]),
                ("dice_2d", dice_2d[patient_id]),
                ("rvd", rvd[patient_id]),
                ("hd95_mm", hd95[patient_id]),
            ):
                for class_id, class_name in enumerate(CLASS_NAMES[1:], start=1):
                    row[f"{metric_name}_{class_name}"] = values[class_id]
            row["mean_foreground_dice_3d"] = nanmean(dice_3d[patient_id][1:])
            row["mean_foreground_dice_2d"] = nanmean(dice_2d[patient_id][1:])
            row["mean_foreground_hd95_mm"] = nanmean(hd95[patient_id][1:])
            writer.writerow(row)


def write_summary_csv(
    path: Path,
    dice_3d: dict[str, np.ndarray],
    dice_2d: dict[str, np.ndarray],
    rvd: dict[str, np.ndarray],
    hd95: dict[str, np.ndarray],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = (
            "class_id",
            "class_name",
            "mean_dice_3d",
            "mean_dice_2d",
            "mean_rvd",
            "mean_absolute_rvd",
            "mean_hd95_mm",
        )
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for class_id, class_name in enumerate(CLASS_NAMES[1:], start=1):
            class_dice_3d = [values[class_id] for values in dice_3d.values()]
            class_dice_2d = [values[class_id] for values in dice_2d.values()]
            class_rvd = [values[class_id] for values in rvd.values()]
            class_hd95 = [values[class_id] for values in hd95.values()]
            writer.writerow(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "mean_dice_3d": nanmean(class_dice_3d),
                    "mean_dice_2d": nanmean(class_dice_2d),
                    "mean_rvd": nanmean(class_rvd),
                    "mean_absolute_rvd": nanmean(np.abs(class_rvd)),
                    "mean_hd95_mm": nanmean(class_hd95),
                }
            )

        all_foreground_3d = [nanmean(values[1:]) for values in dice_3d.values()]
        all_foreground_2d = [nanmean(values[1:]) for values in dice_2d.values()]
        all_foreground_hd95 = [nanmean(values[1:]) for values in hd95.values()]
        writer.writerow(
            {
                "class_id": "1-4",
                "class_name": "mean_foreground",
                "mean_dice_3d": nanmean(all_foreground_3d),
                "mean_dice_2d": nanmean(all_foreground_2d),
                "mean_rvd": "",
                "mean_absolute_rvd": "",
                "mean_hd95_mm": nanmean(all_foreground_hd95),
            }
        )


def run_evaluation(
    prediction_dir: Path,
    target_dir: Path,
    output_dir: Path,
    spacing_file: Path,
    source_inplane_shape: tuple[int, int],
    allow_subset: bool = False,
) -> None:
    prediction_files = indexed_pngs(prediction_dir)
    target_files = indexed_pngs(target_dir)
    patient_spacing = load_spacing(spacing_file)

    prediction_names = set(prediction_files)
    target_names = set(target_files)
    if prediction_names != target_names and not allow_subset:
        missing_predictions = sorted(target_names - prediction_names)
        missing_targets = sorted(prediction_names - target_names)
        raise ValueError(
            "Prediction and GT filenames differ. "
            f"Missing predictions: {missing_predictions[:10]}; "
            f"missing GT files: {missing_targets[:10]}"
        )

    missing_targets = sorted(prediction_names - target_names)
    if missing_targets:
        raise ValueError(f"Predictions without GT files: {missing_targets[:10]}")
    if allow_subset and prediction_names != target_names:
        print(
            "WARNING: subset evaluation enabled; evaluating only "
            f"the {len(prediction_names)} predicted slices."
        )

    patient_files = group_filenames_by_patient(prediction_names)
    missing_spacing = sorted(set(patient_files) - set(patient_spacing))
    if missing_spacing:
        raise ValueError(f"Missing spacing for patients: {missing_spacing}")

    dice_3d: dict[str, np.ndarray] = {}
    dice_2d: dict[str, np.ndarray] = {}
    rvd: dict[str, np.ndarray] = {}
    hd95: dict[str, np.ndarray] = {}

    for patient_id, filenames in patient_files.items():
        (
            patient_dice_3d,
            patient_dice_2d,
            patient_rvd,
            patient_hd95,
        ) = evaluate_patient(
            filenames,
            prediction_files,
            target_files,
            patient_spacing[patient_id],
            source_inplane_shape,
        )
        dice_3d[patient_id] = patient_dice_3d
        dice_2d[patient_id] = patient_dice_2d
        rvd[patient_id] = patient_rvd
        hd95[patient_id] = patient_hd95

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / "dice_3d.npz", **dice_3d)
    np.savez(output_dir / "dice_2d.npz", **dice_2d)
    np.savez(output_dir / "relative_volume_difference.npz", **rvd)
    np.savez(output_dir / "hausdorff_distance_95.npz", **hd95)
    write_patient_csv(
        output_dir / "metrics_per_patient.csv", dice_3d, dice_2d, rvd, hd95
    )
    write_summary_csv(
        output_dir / "metrics_summary.csv", dice_3d, dice_2d, rvd, hd95
    )

    mean_dice = nanmean([nanmean(x[1:]) for x in dice_3d.values()])
    mean_hd95 = nanmean([nanmean(x[1:]) for x in hd95.values()])
    print(f"Evaluated {len(patient_files)} patients and {len(prediction_names)} slices.")
    print(f"Results saved to: {output_dir}")
    print(f"Mean foreground 3D Dice: {mean_dice:.4f}")
    print(f"Mean foreground 3D HD95: {mean_hd95:.4f} mm")


def run_self_test() -> None:
    target = np.array([1, 1, 0, 0], dtype=bool)
    perfect = target.copy()
    partial = np.array([1, 0, 0, 0], dtype=bool)

    assert dice_score(perfect, target) == 1.0
    assert np.isclose(dice_score(partial, target), 2.0 / 3.0)
    assert relative_volume_difference(perfect, target) == 0.0
    assert relative_volume_difference(partial, target) == -0.5
    assert np.isnan(dice_score(np.zeros(4), np.zeros(4)))

    perfect_3d = np.zeros((5, 6, 7), dtype=bool)
    perfect_3d[1:4, 2:5, 2:6] = True
    assert np.isclose(
        hd95_score(perfect_3d, perfect_3d, spacing=(2.5, 2.0, 2.0)),
        0.0,
    )
    assert np.isinf(
        hd95_score(np.zeros_like(perfect_3d), perfect_3d, (2.5, 2.0, 2.0))
    )
    print("PASS: metric formula self-test succeeded, including MONAI HD95.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate patient-level SegTHOR evaluation metrics."
    )
    parser.add_argument("--pred_dir", type=Path)
    parser.add_argument("--gt_dir", type=Path)
    parser.add_argument("--output_dir", type=Path, default=Path("evaluation_results"))
    parser.add_argument(
        "--spacing_file",
        type=Path,
        help="Pickle dictionary mapping Patient_XX to original (x, y, z) spacing.",
    )
    parser.add_argument(
        "--source_inplane_shape",
        type=int,
        nargs=2,
        default=(512, 512),
        metavar=("HEIGHT", "WIDTH"),
        help="Original CT in-plane shape before PNG resizing (default: 512 512).",
    )
    parser.add_argument(
        "--allow_subset",
        action="store_true",
        help="Allow a debug prediction folder to contain fewer slices than GT.",
    )
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()

    if not args.self_test:
        if args.pred_dir is None or args.gt_dir is None:
            parser.error("--pred_dir and --gt_dir are required unless --self_test is used")
        if args.spacing_file is None:
            parser.error("--spacing_file is required unless --self_test is used")
    args.source_inplane_shape = tuple(args.source_inplane_shape)
    return args


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
    else:
        run_evaluation(
            args.pred_dir,
            args.gt_dir,
            args.output_dir,
            args.spacing_file,
            args.source_inplane_shape,
            allow_subset=args.allow_subset,
        )


if __name__ == "__main__":
    main()
