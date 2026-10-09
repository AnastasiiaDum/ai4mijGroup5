#!/usr/bin/env python3.7

# MIT License

# Copyright (c) 2024 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pickle
import random
import argparse
import warnings
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from typing import Callable

import numpy as np
import nibabel as nib
from skimage.io import imsave
from skimage.transform import resize

from utils import map_, tqdm_

from scipy.ndimage import zoom

def norm_arr(img: np.ndarray) -> np.ndarray:
    casted = img.astype(np.float32)
    shifted = casted - casted.min()
    norm = shifted / shifted.max()
    res = 255 * norm

    assert 0 == res.min(), res.min()
    assert res.max() == 255, res.max()

    return res.astype(np.uint8)

def resample_inplane(
    ct: np.ndarray, gt: np.ndarray, dx: float, dy: float, target: float
) -> tuple[np.ndarray, np.ndarray]:
    """Resample x/y to `target` mm; z is left untouched."""
    factors = (dx / target, dy / target, 1.0)
    ct_r = zoom(ct.astype(np.float32), factors, order=1, mode="nearest")  # linear for CT
    gt_r = zoom(gt, factors, order=0, mode="nearest")                     # nearest for labels
    assert ct_r.shape == gt_r.shape, (ct_r.shape, gt_r.shape)
    return ct_r, gt_r


def center_pad_crop(vol: np.ndarray, size: tuple[int, int], fill: float) -> np.ndarray:
    """Centre-pad or centre-crop x/y to `size`."""
    out = np.full((size[0], size[1], vol.shape[2]), fill, dtype=vol.dtype)
    src, dst = [], []
    for n, m in zip(vol.shape[:2], size):
        if n >= m:
            s = (n - m) // 2
            src.append(slice(s, s + m)); dst.append(slice(0, m))
        else:
            s = (m - n) // 2
            src.append(slice(0, n)); dst.append(slice(s, s + n))
    out[dst[0], dst[1]] = vol[src[0], src[1]]
    return out

def norm_arr_percentile(img: np.ndarray) -> np.ndarray:
    """Clip extreme intensities, then scale the CT to 0–255."""
    image = img.astype(np.float32)

    low, high = np.percentile(image, [1, 99])

    if high <= low:
        raise ValueError("Cannot normalize: clipping limits are equal.")

    clipped = np.clip(image, low, high)
    normalized = (clipped - low) / (high - low)

    return (normalized * 255).clip(0, 255).astype(np.uint8)


def norm_arr_fixed_hu(
    img: np.ndarray,
    hu_min: float = -1000.0,
    hu_max: float = 400.0,
) -> np.ndarray:
    """Clip to a fixed HU window, then scale the CT to 0-255."""
    if hu_max <= hu_min:
        raise ValueError("hu_max must be larger than hu_min.")

    image = img.astype(np.float32)
    clipped = np.clip(image, hu_min, hu_max)
    normalized = (clipped - hu_min) / (hu_max - hu_min)

    return (normalized * 255).clip(0, 255).astype(np.uint8)


def sanity_ct(ct, x, y, z, dx, dy, dz) -> bool:
    assert ct.dtype in [np.int16, np.int32], ct.dtype
    assert -1000 <= ct.min(), ct.min()
    assert ct.max() <= 31743, ct.max()

    assert 0.896 <= dx <= 1.37, dx  # Rounding error
    assert dx == dy
    assert 2 <= dz <= 3.7, dz

    assert (x, y) == (512, 512)
    assert x == y
    assert 135 <= z <= 284, z

    return True


def sanity_gt(gt, ct) -> bool:
    assert gt.shape == ct.shape
    assert gt.dtype in [np.uint8], gt.dtype

    # Do the test on 3d: assume all organs are present..
    # assert set(np.unique(gt)) == set(range(5))

    return True


resize_: Callable = partial(resize, mode="constant", preserve_range=True, anti_aliasing=False)


def slice_patient(
    id_: str,
    dest_path: Path,
    source_path: Path,
    shape: tuple[int, int],
    test_mode: bool = False,
    intensity_mode: str = "baseline",
    hu_min: float = -1000.0,
    hu_max: float = 400.0,
    target_spacing: float | None = None,
) -> tuple[float, float, float]:
    id_path: Path = source_path / ("train" if not test_mode else "test") / id_

    ct_path: Path = (id_path / f"{id_}.nii.gz") if not test_mode else (source_path / "test" / f"{id_}.nii.gz")
    nib_obj = nib.load(str(ct_path))
    ct: np.ndarray = np.asarray(nib_obj.dataobj)
    # dx, dy, dz = nib_obj.header.get_zooms()
    x, y, z = ct.shape
    dx, dy, dz = nib_obj.header.get_zooms()

    assert sanity_ct(ct, *ct.shape, *nib_obj.header.get_zooms())

    gt: np.ndarray
    if not test_mode:
        gt_path: Path = id_path / "GT.nii.gz"
        gt_nib = nib.load(str(gt_path))
        # print(nib_obj.affine, gt_nib.affine)
        gt = np.asarray(gt_nib.dataobj)
        assert sanity_gt(gt, ct)
    else:
        gt = np.zeros_like(ct, dtype=np.uint8)

    if target_spacing is not None:
        ct, gt = resample_inplane(ct, gt, dx, dy, target_spacing)
        ct = center_pad_crop(ct, shape, fill=-1000.0)  
        gt = center_pad_crop(gt, shape, fill=0)        
        dx = dy = target_spacing                      

    if intensity_mode == "percentile":
        norm_ct = norm_arr_percentile(ct)
    elif intensity_mode == "fixed_hu":
        norm_ct = norm_arr_fixed_hu(ct, hu_min=hu_min, hu_max=hu_max)
    else:
        norm_ct = norm_arr(ct)

    to_slice_ct = norm_ct
    to_slice_gt = gt

    for idz in range(z):
        img_slice = resize_(to_slice_ct[:, :, idz], shape).astype(np.uint8)
        gt_slice = resize_(to_slice_gt[:, :, idz], shape, order=0).astype(np.uint8)
        assert img_slice.shape == gt_slice.shape
        gt_slice *= 63
        assert gt_slice.dtype == np.uint8, gt_slice.dtype
        # assert set(np.unique(gt_slice)) <= set(range(5))
        assert set(np.unique(gt_slice)) <= set([0, 63, 126, 189, 252]), np.unique(gt_slice)

        arrays: list[np.ndarray] = [img_slice, gt_slice]

        subfolders: list[str] = ["img", "gt"]
        assert len(arrays) == len(subfolders)
        for save_subfolder, data in zip(subfolders,
                                        arrays):
            filename = f"{id_}_{idz:04d}.png"

            save_path: Path = Path(dest_path, save_subfolder)
            save_path.mkdir(parents=True, exist_ok=True)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                imsave(str(save_path / filename), data)

    return dx, dy, dz


def get_splits(src_path: Path, retains: int, fold: int) -> tuple[list[str], list[str], list[str]]:
    ids: list[str] = sorted(
        p.name
        for p in (src_path / "train").glob("Patient_*")
        if p.is_dir()
    )
    print(f"Founds {len(ids)} in the id list")
    print(ids[:10])
    assert len(ids) > retains

    random.shuffle(ids)  # Shuffle before to avoid any problem if the patients are sorted in any way
    validation_slice = slice(fold * retains, (fold + 1) * retains)
    validation_ids: list[str] = ids[validation_slice]
    assert len(validation_ids) == retains

    training_ids: list[str] = [e for e in ids if e not in validation_ids]
    assert (len(training_ids) + len(validation_ids)) == len(ids)

    test_ids: list[str] = sorted(map_(lambda p: Path(p.stem).stem, (src_path / 'test').glob('*')))
    print(f"Founds {len(test_ids)} test ids")
    print(test_ids[:10])

    return training_ids, validation_ids, test_ids


def main(args: argparse.Namespace):
    src_path: Path = Path(args.source_dir)
    dest_path: Path = Path(args.dest_dir)

    # Assume the clean up is done before calling the script
    assert src_path.exists()
    assert not dest_path.exists()

    training_ids: list[str]
    validation_ids: list[str]
    test_ids: list[str]
    training_ids, validation_ids, test_ids = get_splits(src_path, args.retains, args.fold)

    resolution_dict: dict[str, tuple[float, float, float]] = {}

    split_ids: list[str]
    for mode, split_ids in zip(["train", "val"], [training_ids, validation_ids]):
        dest_mode: Path = dest_path / mode
        print(f"Slicing {len(split_ids)} pairs to {dest_mode}")

        pfun: Callable = partial(
            slice_patient,
            dest_path=dest_mode,
            source_path=src_path,
            shape=tuple(args.shape),
            test_mode=mode == "test",
            intensity_mode=args.intensity_mode,
            hu_min=args.hu_min,
            hu_max=args.hu_max,
            target_spacing=args.target_spacing,
        )
        resolutions: list[tuple[float, float, float]]
        iterator = tqdm_(split_ids)
        match args.process:
            case 1:
                resolutions = list(map(pfun, iterator))
            case -1:
                resolutions = Pool().map(pfun, iterator)
            case _ as p:
                resolutions = Pool(p).map(pfun, iterator)

        for key, val in zip(split_ids, resolutions):
            resolution_dict[key] = val

    with open(dest_path / "spacing.pkl", 'wb') as f:
        pickle.dump(resolution_dict, f, pickle.HIGHEST_PROTOCOL)
        print(f"Saved spacing dictionnary to {f}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Slicing parameters')
    parser.add_argument('--source_dir', type=str, required=True)
    parser.add_argument('--dest_dir', type=str, required=True)

    parser.add_argument('--shape', type=int, nargs="+", default=[256, 256])
    parser.add_argument('--retains', type=int, default=25, help="Number of retained patient for the validation data")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--process', '-p', type=int, default=1,
                        help="The number of cores to use for processing")
    parser.add_argument(
        "--intensity_mode",
        choices=["baseline", "percentile", "fixed_hu"],
        default="baseline",
        help="Choose original scaling, percentile clipping, or a fixed HU window."
    )
    parser.add_argument(
        "--hu_min",
        type=float,
        default=-1000.0,
        help="Lower HU limit used when --intensity_mode fixed_hu."
    )
    parser.add_argument(
        "--hu_max",
        type=float,
        default=400.0,
        help="Upper HU limit used when --intensity_mode fixed_hu."
    )
    parser.add_argument('--target_spacing', type=float, default=None,
                        help="Resample x/y to this spacing in mm before slicing (z unchanged).")

    args = parser.parse_args()
    random.seed(args.seed)

    print(args)

    return args


if __name__ == "__main__":
    main(get_args())
