"""Offline augmentation: generate augmented (img, gt) PNG pairs ONCE and save
them to disk, instead of transforming on-the-fly in __getitem__ (dataset1.py).

Uses the exact same augment_pair() as the online version, so the two
approaches are comparable - the only difference is WHEN the random transform
is drawn: once per slice here (fixed forever after), vs fresh every epoch
in the online version.

Only the TRAIN split is augmented. val/ and test/ (if present) are copied
unchanged, so both experiments evaluate on identical, un-augmented data.

Output layout matches what dataset.py's make_dataset() expects:
    dest_dir/train/img/*.png   (originals + augmented copies)
    dest_dir/train/gt/*.png
    dest_dir/val/img/*.png     (unchanged copy of source)
    dest_dir/val/gt/*.png
    dest_dir/test/...          (unchanged copy, if it exists in source)

Run from the project root:
    python offline_augment.py --source_dir data/SEGTHOR --dest_dir data/SEGTHOR_AUG_OFFLINE --n_copies 2

Then train on it exactly like any other dataset:
    python main.py --dataset SEGTHOR_AUG_OFFLINE --mode full --epochs 25 --dest results/segthor_full/ce_offline --gpu
(add "SEGTHOR_AUG_OFFLINE" to datasets_params in main.py first, same K/net/B as "SEGTHOR")
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from augmentation import augment_pair


def copy_unchanged(src_subset_dir: Path, dst_subset_dir: Path):
    if dst_subset_dir.exists():
        shutil.rmtree(dst_subset_dir)
    shutil.copytree(src_subset_dir, dst_subset_dir)


def augment_split(src_subset_dir: Path, dst_subset_dir: Path, n_copies: int, rng: np.random.Generator):
    src_img_dir = src_subset_dir / "img"
    src_gt_dir = src_subset_dir / "gt"
    dst_img_dir = dst_subset_dir / "img"
    dst_gt_dir = dst_subset_dir / "gt"
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_gt_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(src_img_dir.glob("*.png"))
    print(f"  {len(img_paths)} original slices -> {len(img_paths) * (1 + n_copies)} total "
          f"(originals + {n_copies} augmented copie(s) each)")

    for img_path in img_paths:
        gt_path = src_gt_dir / img_path.name
        img = Image.open(img_path)
        gt = Image.open(gt_path)

        # keep the untouched original too, so augmentation ADDS data rather than replacing it
        img.save(dst_img_dir / img_path.name)
        gt.save(dst_gt_dir / img_path.name)

        stem = img_path.stem
        for k in range(n_copies):
            aug_img, aug_gt = augment_pair(img, gt, rng=rng)
            aug_name = f"{stem}_aug{k}.png"
            aug_img.save(dst_img_dir / aug_name)
            aug_gt.save(dst_gt_dir / aug_name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dir", type=Path, required=True, help="e.g. data/SEGTHOR")
    parser.add_argument("--dest_dir", type=Path, required=True, help="e.g. data/SEGTHOR_AUG_OFFLINE")
    parser.add_argument("--n_copies", type=int, default=2,
                         help="augmented copies to generate PER original training slice")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.dest_dir.exists():
        raise SystemExit(f"{args.dest_dir} already exists - remove it first, nothing overwritten")

    rng = np.random.default_rng(args.seed)

    print(f"Augmenting train split ({args.source_dir / 'train'} -> {args.dest_dir / 'train'})")
    augment_split(args.source_dir / "train", args.dest_dir / "train", args.n_copies, rng)

    for subset in ["val", "test"]:
        src = args.source_dir / subset
        if src.exists():
            print(f"Copying {subset} split unchanged ({src} -> {args.dest_dir / subset})")
            copy_unchanged(src, args.dest_dir / subset)

    print(f"\nDone. Dataset ready at {args.dest_dir.resolve()}")


if __name__ == "__main__":
    main()