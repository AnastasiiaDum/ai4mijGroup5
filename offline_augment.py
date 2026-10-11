"""Offline augmentation: write augmented (img, gt) PNG pairs to disk ONCE, using the same
Augmenter / AugConfig as the online version.

Output layout (everything goes into train/, because main_hyperparameter pools and re-splits by patient):
    dest/train/img/Patient_01_0042.png          original (hard link, no extra disk)
    dest/train/img/Patient_01aug0_0042.png      augmented copy 0 of patient 01
    dest/train/gt/...                           matching labels

Why the "aug" sits in the PATIENT part of the name (Patient_01aug0_0042):
  * patient_id() still returns Patient_01, so augmented copies always stay on the same side of
    the train/val split as their original patient (no leakage).
  * SliceDataset25D groups slices by everything before the last _<number>, so an augmented copy
    becomes its own volume and its 2.5D stacks only contain slices of that copy.

Only patients in the TRAINING part of the split get augmented copies. Validation (and test)
patients stay original, so Dice is comparable with the online runs.

Standalone use (single train/val/test split only, NOT k-fold CV):
    python offline_augment.py --source_dir data/hu_wide_rs18 --dest_dir data/hu_wide_rs18_aug \
        --aug_json results/aug_UNET25D/best_aug.json --n_copies 1 --per_volume \
        --test_frac 0.1 --val_frac 0.1 --seed 0
Then train with the SAME --test_frac / --val_frac / --seed, --n_folds 0 and --data_dir <dest_dir>.
"""
import argparse
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from tune_aug_effect import AugConfig, Augmenter
from main_hyperparameter import patient_id, split_by_patient

_PAT = re.compile(r"^(.*)_(\d+)$")           # Patient_01_0042 -> ("Patient_01", "0042")
_AUG = re.compile(r"aug\d+_\d+$")            # Patient_01aug0_0042


def is_aug_stem(stem: str) -> bool:
    return _AUG.search(stem) is not None


def source_dirs(src: Path) -> list[Path]:
    dirs = [src / s for s in ('train', 'val') if any((src / s / "gt").glob("*.png"))]
    if not dirs:
        raise FileNotFoundError(f"No train/ or val/ folder with gt/*.png in {src}")
    return dirs


def list_source_stems(src: Path) -> list[str]:
    return sorted(p.stem for d in source_dirs(src) for p in (d / "gt").glob("*.png"))


def single_split_train_patients(stems, test_frac, val_frac, seed) -> set[str]:
    """Patients that end up in TRAIN for the single-split mode of runTraining (same logic)."""
    if test_frac > 0:
        rest, _ = split_by_patient(stems, test_frac, seed)
    else:
        rest = np.arange(len(stems))
    rest_stems = [stems[i] for i in rest]
    tr, _ = split_by_patient(rest_stems, val_frac / (1 - test_frac), seed + 1)
    return {patient_id(rest_stems[i]) for i in tr}


def _params_to_json(p: dict) -> dict:
    sp = p['spatial']
    return {'spatial': None if sp is None else dict(zip(('angle_deg', 'tx_px', 'ty_px', 'scale'),
                                                        map(float, sp))),
            'shift': None if p['shift'] is None else float(p['shift']),
            'noise': bool(p['noise'])}


def _link_or_copy(src: Path, dst: Path):
    try:
        os.link(src, dst)                    # hard link: instant, no extra disk (same drive, NTFS)
    except OSError:
        shutil.copy2(src, dst)


def build_offline_dataset(src, dst, aug: Augmenter, n_copies: int, aug_patients: set[str],
                          per_volume: bool, seed: int) -> dict:
    """per_volume=True: one random draw per (patient, copy), applied to all its slices.
    Needed for 2.5D, otherwise neighbouring slices of a stack would be transformed differently.
    per_volume=False: a fresh draw for every slice (closest to the online 2D behaviour)."""
    src, dst = Path(src), Path(dst)
    if dst.exists():
        shutil.rmtree(dst)
    out_img, out_gt = dst / "train" / "img", dst / "train" / "gt"
    out_img.mkdir(parents=True)
    out_gt.mkdir(parents=True)
    rng = np.random.default_rng(seed)

    volumes: dict[str, list] = {}
    n_orig = 0
    for d in source_dirs(src):
        for img_p in sorted((d / "img").glob("*.png")):
            gt_p = d / "gt" / img_p.name
            if not gt_p.exists():
                raise FileNotFoundError(f"Missing label for {img_p.name}: {gt_p}")
            m = _PAT.match(img_p.stem)
            if m is None:
                raise ValueError(f"Unexpected filename (need <patient>_<slice>): {img_p.name}")
            _link_or_copy(img_p, out_img / img_p.name)
            _link_or_copy(gt_p, out_gt / gt_p.name)
            volumes.setdefault(m.group(1), []).append((m.group(2), img_p, gt_p))
            n_orig += 1

    n_aug = n_skipped = 0
    log: dict[str, dict] = {}   # per_volume: key "Patient_01aug0"; per slice: key "Patient_01aug0_0042"
    for prefix, slices in volumes.items():
        if patient_id(prefix) not in aug_patients:
            continue
        for k in range(n_copies):
            vol_params = aug.draw(rng) if per_volume else None
            if per_volume and aug.is_identity(vol_params):
                n_skipped += len(slices)     # nothing fired: an identical copy adds no information
                continue
            if per_volume:
                log[f"{prefix}aug{k}"] = _params_to_json(vol_params)
            for idx, img_p, gt_p in slices:
                params = vol_params if per_volume else aug.draw(rng)
                if aug.is_identity(params):
                    n_skipped += 1
                    continue
                if not per_volume:
                    log[f"{prefix}aug{k}_{idx}"] = _params_to_json(params)
                img = Image.open(img_p)
                img.load()
                gt = Image.open(gt_p)
                gt.load()
                [aug_img], aug_gt = aug.apply([img], gt, params, rng)
                name = f"{prefix}aug{k}_{idx}.png"
                aug_img.save(out_img / name)
                if params['spatial'] is None:        # label unchanged -> reuse the file
                    _link_or_copy(gt_p, out_gt / name)
                else:
                    aug_gt.save(out_gt / name)
                n_aug += 1

    with open(dst / "aug_log.json", "w") as f:
        json.dump(log, f, indent=1)
    return {"n_original": n_orig, "n_augmented": n_aug, "n_skipped_identity": n_skipped}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source_dir", type=Path, required=True)
    ap.add_argument("--dest_dir", type=Path, required=True)
    ap.add_argument("--aug_json", type=Path, default=None, help="AugConfig json; default = AugConfig().")
    ap.add_argument("--n_copies", type=int, default=1)
    ap.add_argument("--per_volume", action="store_true", help="One draw per patient copy (use for 2.5D).")
    ap.add_argument("--test_frac", type=float, default=0.1)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=float, default=0, help="Split seed, same as main_hyperparameter --seed.")
    ap.add_argument("--aug_seed", type=int, default=0)
    args = ap.parse_args()
    args.seed = int(args.seed)

    if args.dest_dir.exists():
        raise SystemExit(f"{args.dest_dir} already exists - remove it first, nothing overwritten")

    cfg = AugConfig(**json.load(open(args.aug_json))) if args.aug_json else AugConfig()
    stems = list_source_stems(args.source_dir)
    train_p = single_split_train_patients(stems, args.test_frac, args.val_frac, args.seed)
    print(f"Augmenting {len(train_p)} training patients (val/test patients stay original)")

    stats = build_offline_dataset(args.source_dir, args.dest_dir, Augmenter(cfg), args.n_copies,
                                  train_p, args.per_volume, args.aug_seed)
    print(stats)
    print(f"Done: {args.dest_dir.resolve()}")


if __name__ == "__main__":
    main()
