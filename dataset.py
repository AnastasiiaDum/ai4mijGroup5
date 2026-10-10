#!/usr/bin/env python3

# MIT License

# Copyright (c) 2025 Hoel Kervadec

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

from pathlib import Path
from typing import Callable, Union

from torch import Tensor
from PIL import Image
from torch.utils.data import Dataset
import torch

import re


def make_dataset(root, subset) -> list[tuple[Path, Path | None]]:
    assert subset in ['train', 'val', 'test']

    root = Path(root)
    print(f"> {root=}")

    img_path = root / subset / 'img'
    full_path = root / subset / 'gt'

    images: list[Path] = sorted(img_path.glob("*.png"))
    full_labels: list[Path | None]
    if subset != 'test':
        full_labels = sorted(full_path.glob("*.png"))
    else:
        full_labels = [None] * len(images)

    return list(zip(images, full_labels))


class SliceDataset(Dataset):
    def __init__(self, subset, root_dir, img_transform=None,
                 gt_transform=None, augment=False, equalize=False, debug=False):
        self.root_dir: str = root_dir
        self.img_transform: Callable = img_transform
        self.gt_transform: Callable = gt_transform
        self.augmentation: bool = augment
        self.equalize: bool = equalize

        self.test_mode: bool = subset == 'test'

        self.files = make_dataset(root_dir, subset)
        if debug:
            self.files = self.files[:10]

        print(f">> Created {subset} dataset with {len(self)} images...")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index) -> dict[str, Union[Tensor, int, str]]:
        img_path, gt_path = self.files[index]

        img: Tensor = self.img_transform(Image.open(img_path))

        data_dict = {"images": img,
                     "stems": img_path.stem}

        if not self.test_mode:
            gt: Tensor = self.gt_transform(Image.open(gt_path))

            _, W, H = img.shape
            K, _, _ = gt.shape
            assert gt.shape == (K, W, H)

            data_dict["gts"] = gt

        return data_dict

class SliceDataset25D(Dataset):
    """
    Returns a stack of (2*n_neighbors+1) adjacent slices from the same patient
    as the input, and the ground truth of the center slice only.
    Edge slices are handled by repeating the first/last slice.
    Expects filenames like Patient_01_0042.png.
    """
    _pat = re.compile(r"^(.*)_(\d+)$")

    def __init__(self, subset, root_dir, img_transform=None, gt_transform=None,
                 debug=False, n_neighbors=2):
        self.img_transform = img_transform
        self.gt_transform = gt_transform
        root = Path(root_dir) / subset
        imgs = sorted((root / "img").glob("*.png"))
        gts = sorted((root / "gt").glob("*.png"))
        assert len(imgs) == len(gts)

        # group slices by patient, ordered by slice index
        groups = {}
        for p in imgs:
            m = self._pat.match(p.stem)
            assert m, f"Unexpected filename: {p.name}"
            groups.setdefault(m.group(1), []).append((int(m.group(2)), p))
        pos = {}  # path -> (patient, position in patient's sorted list)
        for pid, lst in groups.items():
            lst.sort(key=lambda t: t[0])
            groups[pid] = [p for _, p in lst]
            for i, p in enumerate(groups[pid]):
                pos[p] = (pid, i)

        self.samples = []  # (list of neighbour paths, center gt path)
        for img_p, gt_p in zip(imgs, gts):
            pid, i = pos[img_p]
            g = groups[pid]
            neigh = [g[min(max(i + o, 0), len(g) - 1)]
                     for o in range(-n_neighbors, n_neighbors + 1)]
            self.samples.append((neigh, gt_p))

        if debug:
            self.samples = self.samples[:10]
        print(f"> Created {subset} 2.5D dataset with {len(self.samples)} samples "
              f"({2 * n_neighbors + 1} slices each)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        neigh, gt_path = self.samples[index]
        stack = torch.cat([self.img_transform(Image.open(p)) for p in neigh], dim=0)
        gt = self.gt_transform(Image.open(gt_path))
        center = neigh[len(neigh) // 2]
        return {"images": stack, "gts": gt, "stems": center.stem}
