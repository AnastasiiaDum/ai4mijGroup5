"""
3D post-processing.
Input : folder of predicted label volumes Patient_XX.nii.gz (from inference.py)
Output: cleaned Patient_XX.nii.gz.

Steps:
  lcc -> keep the largest 3D connected component of each class, removed voxels become background. 
  fill_holes -> fill background holes enclosed inside a class.
  smooth -> Gaussian-smooth each class's one-hot mask and re-take the argmax.

destination = results/final_UNET2D
python postprocess.py --run_dir $destination --split val --steps lcc fill_holes smooth   # compare steps vs GT
python postprocess.py --run_dir $destination --split heldout --steps lcc fill_holes      # final honest score
python postprocess.py --run_dir $destination --split test --steps lcc fill_holes         # test set (no GT)

"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import nibabel as nib

import numpy as np
from scipy import ndimage

K = 5
CLASS_NAMES = ("background", "esophagus", "heart", "trachea", "aorta")
STRUCT_26 = ndimage.generate_binary_structure(3, 3)


# input/output
def load_labels(path: Path):
    img = nib.load(str(path))
    vol = np.rint(np.asarray(img.dataobj)).astype(np.uint8)
    return img, vol


def save_labels(ref_img, vol: np.ndarray, path: Path) -> None:
    hdr = ref_img.header.copy()
    hdr.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(vol.astype(np.uint8), ref_img.affine, header=hdr), str(path))


def find_gt(gt_dir: Path, pid: str) -> Path | None:
    for candidate in (gt_dir / pid / "GT.nii.gz", gt_dir / f"{pid}.nii.gz",
                 gt_dir / f"{pid}_GT.nii.gz", gt_dir / pid / f"{pid}_GT.nii.gz"):
        if candidate.exists():
            return candidate
    
    return None


# Helper functions
def bbox_slices(mask: np.ndarray, margin: int = 0):
    """ Bounding box of a boolean mask. """
    
    objs = ndimage.find_objects(mask.astype(np.uint8))
    if not objs or objs[0] is None:
        return None
    
    return tuple(slice(max(s.start - margin, 0), min(s.stop + margin, n)) for s, n in zip(objs[0], mask.shape))


def keep_components(mask: np.ndarray, keep_frac: float) -> np.ndarray:
    lab, n = ndimage.label(mask, structure=STRUCT_26)
    
    if n <= 1:
        return mask
    sizes = np.bincount(lab.ravel())[1:]
    
    if keep_frac <= 0:
        keep = np.zeros(n, dtype=bool)
        keep[int(np.argmax(sizes))] = True # largest only
    else:
        keep = sizes >= keep_frac * sizes.max()
    
    lut = np.zeros(n + 1, dtype=bool)
    lut[1:] = keep
    
    return lut[lab]


# Post-processing steps
def step_lcc(vol, spacing, p) -> None:
    for c in p.lcc_classes:
        mask = vol == c
        box = bbox_slices(mask)
        
        if box is None:
            continue
        
        sub = mask[box]
        removed = sub & ~keep_components(sub, p.keep_frac)
        view = vol[box]
        view[removed] = 0


def step_fill_holes(vol, spacing, p) -> None:
    for c in p.fill_classes:
        mask = vol == c
        box = bbox_slices(mask, margin=1)
        
        if box is None:
            continue
        
        sub = mask[box]
        if p.fill_mode == "3d":
            filled = ndimage.binary_fill_holes(sub)
        else:
            filled = np.stack([ndimage.binary_fill_holes(sub[:, :, z])
                               for z in range(sub.shape[2])], axis=2)
        
        view = vol[box]
        holes = filled & ~sub & (view == 0) # prevent overwriting another organ
        view[holes] = c


def step_smooth(vol, spacing, p) -> None:
    sigma_vox = [p.smooth_sigma_mm / float(s) for s in spacing]
    margin = int(np.ceil(3 * max(sigma_vox))) + 1
    box = bbox_slices(vol > 0, margin=margin)
    
    if box is None:
        return
    
    sub = vol[box]
    best = np.full(sub.shape, -1.0, dtype=np.float32)
    lab = np.zeros(sub.shape, dtype=np.uint8)
    
    for c in range(K):
        g = ndimage.gaussian_filter((sub == c).astype(np.float32), sigma_vox, mode="nearest")
        upd = g > best
        best[upd] = g[upd]
        lab[upd] = c
    
    vol[box] = lab


STEPS = {"lcc": step_lcc, "fill_holes": step_fill_holes, "smooth": step_smooth}


# Metrics
def dice_3d(pred: np.ndarray, gt: np.ndarray) -> float:
    denom = int(pred.sum()) + int(gt.sum())
    
    if denom == 0:
        return float("nan")
    
    return 2.0 * int(np.logical_and(pred, gt).sum()) / denom


def hd95(pred: np.ndarray, gt: np.ndarray, spacing) -> float:
    if not pred.any() or not gt.any():
        return float("nan")
    
    box = bbox_slices(pred | gt, margin=1)
    p, g = pred[box], gt[box]
    ps = p & ~ndimage.binary_erosion(p)
    gs = g & ~ndimage.binary_erosion(g)
    d_to_g = ndimage.distance_transform_edt(~gs, sampling=spacing)
    d_to_p = ndimage.distance_transform_edt(~ps, sampling=spacing)
    
    return float(np.percentile(np.concatenate([d_to_g[ps], d_to_p[gs]]), 95))


def evaluate(vol: np.ndarray, gt: np.ndarray, spacing) -> dict[int, tuple[float, float]]:
    return {c: (dice_3d(vol == c, gt == c), hd95(vol == c, gt == c, spacing)) for c in range(1, K)}


# Driver
def run(args) -> None:
    files = sorted(args.pred_dir.rglob("*.nii.gz"))
    
    if args.patients:
        files = [f for f in files if f.name[:-7] in set(args.patients)]
    
    if not files:
        raise FileNotFoundError(f"No .nii.gz predictions found in {args.pred_dir}")
    
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "postprocess_config.json").write_text(
        json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))

    rows: list[dict] = []
    stage_names = ["raw"]
    for s in args.steps:
        stage_names.append(f"+{s}")

    for f in files:
        pid = f.name[:-len(".nii.gz")]
        img, vol = load_labels(f)
        spacing = tuple(float(s) for s in img.header.get_zooms()[:3])

        gt = None
        if args.gt_dir is not None:
            gt_path = find_gt(args.gt_dir, pid)
            if gt_path is None:
                print(f"   {pid}: no GT found in {args.gt_dir}, skipping evaluation")
            else:
                _, gt = load_labels(gt_path)
                if gt.shape != vol.shape:
                    raise ValueError(f"{pid}: prediction {vol.shape} vs GT {gt.shape}")

        def log(stage: str) -> None:
            if gt is None:
                return
            for c, (d, h) in evaluate(vol, gt, spacing).items():
                rows.append({"stage": stage, "patient": pid, "class": CLASS_NAMES[c], "dice_3d": d, "hd95_mm": h})

        log("raw")
        for name, step in zip(stage_names[1:], args.steps):
            STEPS[step](vol, spacing, args)
            log(name)

        save_labels(img, vol, args.out_dir / f"{pid}.nii.gz")
        print(f"{pid}: processed" + (" + evaluated" if gt is not None else ""))

    if rows:
        report(rows, stage_names, args.out_dir)
    print(f"Done! Cleaned volumes saved to {args.out_dir}")


def report(rows: list[dict], stage_names: list[str], out_dir: Path) -> None:
    with (out_dir / "metrics_per_patient.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["stage", "patient", "class", "dice_3d", "hd95_mm"])
        w.writeheader()
        w.writerows(rows)

    def mean(stage, cls, key):
        v = [r[key] for r in rows if r["stage"] == stage and r["class"] == cls]
        v = np.array(v, dtype=float)
        v = v[~np.isnan(v)]
        return float(v.mean()) if v.size else float("nan")

    classes = CLASS_NAMES[1:]
    summary = []
    
    print("\nMean 3D Dice (higher is better)")
    print(f"{'stage':<18}" + "".join(f"{c:>12}" for c in classes) + f"{'mean_fg':>12}")
    for st in stage_names:
        d = [mean(st, c, "dice_3d") for c in classes]
        print(f"{st:<18}" + "".join(f"{x:12.4f}" for x in d) + f"{np.nanmean(d):12.4f}")
        summary.append({"stage": st, "metric": "dice_3d",
                        **dict(zip(classes, d)), "mean_fg": float(np.nanmean(d))})
    
    print("\nMean HD95 in mm (lower is better)")
    print(f"{'stage':<18}" + "".join(f"{c:>12}" for c in classes) + f"{'mean_fg':>12}")
    for st in stage_names:
        h = [mean(st, c, "hd95_mm") for c in classes]
        print(f"{st:<18}" + "".join(f"{x:12.2f}" for x in h) + f"{np.nanmean(h):12.2f}")
        summary.append({"stage": st, "metric": "hd95_mm",
                        **dict(zip(classes, h)), "mean_fg": float(np.nanmean(h))})

    with (out_dir / "metrics_summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["stage", "metric", *classes, "mean_fg"])
        w.writeheader()
        w.writerows(summary)
    print(f"\nMetrics written to {out_dir}/metrics_per_patient.csv and metrics_summary.csv")


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="3D post-processing & evaluation of SegTHOR predictions")
    
    p.add_argument("--run_dir", type=Path, default=None, help="Training folder used with inference.py.")
    p.add_argument("--split", choices=["val", "heldout", "test"], default=None, help="Which inference output to process.")
    p.add_argument("--data_dir", type=Path, default=Path("data"), help="Folder with the original train/ CTs and GT.")
    p.add_argument("--pred_dir", type=Path, default=None, help="Folder with predicted Patient_XX.nii.gz volumes.")
    p.add_argument("--out_dir", type=Path, default=None, help="Where to write cleaned volumes and metrics.")
    p.add_argument("--gt_dir", type=Path, default=None, help="Optional: folder with ground truth (Patient_XX/GT.nii.gz "
                        "or Patient_XX.nii.gz) to evaluate raw vs post-processed.")
    p.add_argument("--patients", nargs="+", default=None, help="Only process these patient ids.")
    p.add_argument("--steps", nargs="+", default=["lcc", "fill_holes"], choices=list(STEPS),
                   help="Steps in the order they are applied.")
    p.add_argument("--lcc_classes", type=int, nargs="+", default=[1, 2, 3, 4],
                   help="Classes that get largest-component filtering.")
    p.add_argument("--keep_frac", type=float, default=0.0, help="lcc: also keep components >= this fraction of the largest.")
    p.add_argument("--fill_classes", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--fill_mode", choices=["3d", "2d"], default="3d")
    p.add_argument("--smooth_sigma_mm", type=float, default=1.0)
    
    a = p.parse_args()
    
    if a.split is not None and a.run_dir is None:
        p.error("--split needs --run_dir")

    if a.pred_dir is None or a.out_dir is None:
        p.error("give --run_dir with --split, or --pred_dir and --out_dir")
    
    return a


if __name__ == "__main__":
    args = get_args()
    run(args)
