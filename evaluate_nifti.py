#!/usr/bin/env python3
"""Evaluate SegTHOR NIfTI predictions patient-by-patient, in original space.

NIfTI counterpart of evaluate_metrics.py (which works on PNG slices)

Predictions: --pred_dir/Patient_XX.nii.gz (e.g. from inference.py).
Ground truth: --gt_dir/Patient_XX.nii.gz or --gt_dir/Patient_XX/GT.nii.gz (original data folder).

python evaluate_nifti.py --pred_dir submission/val/pred --gt_dir submission/val/gt --output_dir submission/val
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

from evaluate_metrics import (N_CLASSES, dice_score, hd95_score, nanmean, relative_volume_difference,
                              write_patient_csv, write_summary_csv)


def find_gt(gt_dir: Path, patient_id: str) -> Path:
    for candidate in (gt_dir / f"{patient_id}.nii.gz", gt_dir / patient_id / "GT.nii.gz"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No ground truth for {patient_id} in {gt_dir}")


def load_labels(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    img = nib.load(str(path))
    labels = np.rint(np.asarray(img.dataobj)).astype(np.uint8)
    if labels.ndim != 3 or labels.max() >= N_CLASSES:
        raise ValueError(f"{path}: expected a 3D label volume with values 0..{N_CLASSES - 1}")
    spacing = tuple(float(s) for s in img.header.get_zooms()[:3])
    return labels, spacing


def evaluate_patient(pred: np.ndarray, gt: np.ndarray, spacing) -> tuple[np.ndarray, ...]:
    dice_3d, dice_2d, rvd, hd95 = (np.full(N_CLASSES, np.nan, dtype=float) for _ in range(4))
    for c in range(N_CLASSES):
        p, g = pred == c, gt == c
        dice_3d[c] = dice_score(p, g)
        rvd[c] = relative_volume_difference(p, g)
        if c > 0:
            hd95[c] = hd95_score(p, g, spacing)
        dice_2d[c] = nanmean([dice_score(p[:, :, z], g[:, :, z]) for z in range(gt.shape[2])])
    return dice_3d, dice_2d, rvd, hd95


def run_evaluation(pred_dir: Path, gt_dir: Path, output_dir: Path) -> None:
    pred_files = sorted(pred_dir.glob("Patient_*.nii.gz"))
    if not pred_files:
        raise ValueError(f"No Patient_XX.nii.gz predictions found in {pred_dir}")

    dice_3d, dice_2d, rvd, hd95 = {}, {}, {}, {}
    for pred_path in pred_files:
        patient_id = pred_path.name[:-len(".nii.gz")]
        pred, _ = load_labels(pred_path)
        gt, spacing = load_labels(find_gt(gt_dir, patient_id))
        if pred.shape != gt.shape:
            raise ValueError(f"Shape mismatch for {patient_id}: prediction={pred.shape}, target={gt.shape}")

        (dice_3d[patient_id], dice_2d[patient_id],
         rvd[patient_id], hd95[patient_id]) = evaluate_patient(pred, gt, spacing)
        print(f"   {patient_id}: 3D Dice {nanmean(dice_3d[patient_id][1:]):.4f} | "
              f"HD95 {nanmean(hd95[patient_id][1:]):.2f} mm", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / "dice_3d.npz", **dice_3d)
    np.savez(output_dir / "dice_2d.npz", **dice_2d)
    np.savez(output_dir / "relative_volume_difference.npz", **rvd)
    np.savez(output_dir / "hausdorff_distance_95.npz", **hd95)
    write_patient_csv(output_dir / "metrics_per_patient.csv", dice_3d, dice_2d, rvd, hd95)
    write_summary_csv(output_dir / "metrics_summary.csv", dice_3d, dice_2d, rvd, hd95)

    print(f"Evaluated {len(pred_files)} patients.")
    print(f"Results saved to: {output_dir}")
    print(f"Mean foreground 3D Dice: {nanmean([nanmean(x[1:]) for x in dice_3d.values()]):.4f}")
    print(f"Mean foreground 3D HD95: {nanmean([nanmean(x[1:]) for x in hd95.values()]):.4f} mm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate patient-level SegTHOR metrics on NIfTI volumes.")
    parser.add_argument("--pred_dir", type=Path, required=True, help="Folder with Patient_XX.nii.gz predictions.")
    parser.add_argument("--gt_dir", type=Path, required=True,
                        help="Folder with Patient_XX.nii.gz or Patient_XX/GT.nii.gz ground truth.")
    parser.add_argument("--output_dir", type=Path, default=Path("evaluation_results"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args.pred_dir, args.gt_dir, args.output_dir)
