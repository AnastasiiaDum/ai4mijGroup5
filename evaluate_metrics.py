#!/usr/bin/env python3
"""Evaluate SegTHOR PNG predictions patient-by-patient.

The script reports:
* per-class 3D Dice (computed after stacking every slice of a patient),
* per-class mean 2D Dice (empty/empty slices are excluded),
* mean foreground Dice (classes 1-4), and
* relative volume difference: (predicted voxels - GT voxels) / GT voxels.

It accepts masks encoded either as class IDs 0..4 or as the repository's
grayscale values 0, 63, 126, 189 and 252.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
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


def nanmean(values: list[float] | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    valid = array[~np.isnan(array)]
    return float(valid.mean()) if valid.size else float("nan")


def evaluate_patient(
    filenames: list[str],
    prediction_files: dict[str, Path],
    target_files: dict[str, Path],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    dice_3d = np.full(N_CLASSES, np.nan, dtype=float)
    dice_2d = np.full(N_CLASSES, np.nan, dtype=float)
    rvd = np.full(N_CLASSES, np.nan, dtype=float)

    for class_id in range(N_CLASSES):
        pred_class_3d = prediction_3d == class_id
        target_class_3d = target_3d == class_id

        dice_3d[class_id] = dice_score(pred_class_3d, target_class_3d)
        rvd[class_id] = relative_volume_difference(
            pred_class_3d, target_class_3d
        )

        slice_scores = [
            dice_score(pred_slice == class_id, target_slice == class_id)
            for pred_slice, target_slice in zip(predictions, targets)
        ]
        dice_2d[class_id] = nanmean(slice_scores)

    return dice_3d, dice_2d, rvd


def write_patient_csv(
    path: Path,
    dice_3d: dict[str, np.ndarray],
    dice_2d: dict[str, np.ndarray],
    rvd: dict[str, np.ndarray],
) -> None:
    fieldnames = ["patient_id"]
    for metric_name in ("dice_3d", "dice_2d", "rvd"):
        fieldnames.extend(
            f"{metric_name}_{class_name}" for class_name in CLASS_NAMES[1:]
        )
    fieldnames.extend(("mean_foreground_dice_3d", "mean_foreground_dice_2d"))

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for patient_id in sorted(dice_3d):
            row: dict[str, str | float] = {"patient_id": patient_id}
            for metric_name, values in (
                ("dice_3d", dice_3d[patient_id]),
                ("dice_2d", dice_2d[patient_id]),
                ("rvd", rvd[patient_id]),
            ):
                for class_id, class_name in enumerate(CLASS_NAMES[1:], start=1):
                    row[f"{metric_name}_{class_name}"] = values[class_id]
            row["mean_foreground_dice_3d"] = nanmean(dice_3d[patient_id][1:])
            row["mean_foreground_dice_2d"] = nanmean(dice_2d[patient_id][1:])
            writer.writerow(row)


def write_summary_csv(
    path: Path,
    dice_3d: dict[str, np.ndarray],
    dice_2d: dict[str, np.ndarray],
    rvd: dict[str, np.ndarray],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = (
            "class_id",
            "class_name",
            "mean_dice_3d",
            "mean_dice_2d",
            "mean_rvd",
            "mean_absolute_rvd",
        )
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for class_id, class_name in enumerate(CLASS_NAMES[1:], start=1):
            class_dice_3d = [values[class_id] for values in dice_3d.values()]
            class_dice_2d = [values[class_id] for values in dice_2d.values()]
            class_rvd = [values[class_id] for values in rvd.values()]
            writer.writerow(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "mean_dice_3d": nanmean(class_dice_3d),
                    "mean_dice_2d": nanmean(class_dice_2d),
                    "mean_rvd": nanmean(class_rvd),
                    "mean_absolute_rvd": nanmean(np.abs(class_rvd)),
                }
            )

        all_foreground_3d = [nanmean(values[1:]) for values in dice_3d.values()]
        all_foreground_2d = [nanmean(values[1:]) for values in dice_2d.values()]
        writer.writerow(
            {
                "class_id": "1-4",
                "class_name": "mean_foreground",
                "mean_dice_3d": nanmean(all_foreground_3d),
                "mean_dice_2d": nanmean(all_foreground_2d),
                "mean_rvd": "",
                "mean_absolute_rvd": "",
            }
        )


def run_evaluation(
    prediction_dir: Path,
    target_dir: Path,
    output_dir: Path,
    allow_subset: bool = False,
) -> None:
    prediction_files = indexed_pngs(prediction_dir)
    target_files = indexed_pngs(target_dir)

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
    dice_3d: dict[str, np.ndarray] = {}
    dice_2d: dict[str, np.ndarray] = {}
    rvd: dict[str, np.ndarray] = {}

    for patient_id, filenames in patient_files.items():
        patient_dice_3d, patient_dice_2d, patient_rvd = evaluate_patient(
            filenames, prediction_files, target_files
        )
        dice_3d[patient_id] = patient_dice_3d
        dice_2d[patient_id] = patient_dice_2d
        rvd[patient_id] = patient_rvd

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / "dice_3d.npz", **dice_3d)
    np.savez(output_dir / "dice_2d.npz", **dice_2d)
    np.savez(output_dir / "relative_volume_difference.npz", **rvd)
    write_patient_csv(output_dir / "metrics_per_patient.csv", dice_3d, dice_2d, rvd)
    write_summary_csv(output_dir / "metrics_summary.csv", dice_3d, dice_2d, rvd)

    print(f"Evaluated {len(patient_files)} patients and {len(prediction_names)} slices.")
    print(f"Results saved to: {output_dir}")
    print(f"Mean foreground 3D Dice: {nanmean([nanmean(x[1:]) for x in dice_3d.values()]):.4f}")


def run_self_test() -> None:
    target = np.array([1, 1, 0, 0], dtype=bool)
    perfect = target.copy()
    partial = np.array([1, 0, 0, 0], dtype=bool)

    assert dice_score(perfect, target) == 1.0
    assert np.isclose(dice_score(partial, target), 2.0 / 3.0)
    assert relative_volume_difference(perfect, target) == 0.0
    assert relative_volume_difference(partial, target) == -0.5
    assert np.isnan(dice_score(np.zeros(4), np.zeros(4)))
    print("PASS: metric formula self-test succeeded.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate patient-level SegTHOR evaluation metrics."
    )
    parser.add_argument("--pred_dir", type=Path)
    parser.add_argument("--gt_dir", type=Path)
    parser.add_argument("--output_dir", type=Path, default=Path("evaluation_results"))
    parser.add_argument(
        "--allow_subset",
        action="store_true",
        help="Allow a debug prediction folder to contain fewer slices than GT.",
    )
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()

    if not args.self_test and (args.pred_dir is None or args.gt_dir is None):
        parser.error("--pred_dir and --gt_dir are required unless --self_test is used")
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
            allow_subset=args.allow_subset,
        )


if __name__ == "__main__":
    main()
