"""Configurable augmentation for 2D / 2.5D slices (used by aug_tune.py).

Same operations as your augment_pair, but the strengths live in an AugConfig so
Optuna can tune them. An Augmenter takes a LIST of images (1 for 2D, 2n+1 for 2.5D)
and the centre ground truth. The same spatial transform is applied to every slice
and to the gt, so the stack and the label stay aligned. A probability of 0 disables
an augmentation.

Augmenter is a module-level class, so it can be pickled by DataLoader workers on Windows.
"""
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class AugConfig:
    p_spatial: float = 0.5
    max_rotation_deg: float = 10.0
    max_translation_px: float = 10.0
    scale_delta: float = 0.1       # scale drawn from [1 - d, 1 + d]
    p_shift: float = 0.5
    shift_max: float = 15.0        # intensity shift drawn from [-m, m] (0-255 scale)
    p_noise: float = 0.5
    noise_std: float = 5.0


def _affine_coeffs(size, angle_deg, tx, ty, scale):
    w, h = size
    cx, cy = w / 2.0, h / 2.0
    theta = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    a = cos_t / scale
    b = sin_t / scale
    d = -sin_t / scale
    e = cos_t / scale
    c = cx - a * (cx + tx) - b * (cy + ty)
    f = cy - d * (cx + tx) - e * (cy + ty)
    return (a, b, c, d, e, f)


def _apply_affine(img, coeffs, resample, fillcolor):
    return img.transform(img.size, Image.AFFINE, coeffs, resample=resample, fillcolor=fillcolor)


def _to_img(arr):
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


class Augmenter:
    def __init__(self, cfg: AugConfig):
        self.cfg = cfg

    def __call__(self, imgs, gt=None, rng=None):
        rng = rng or np.random.default_rng()
        c = self.cfg
        assert gt is None or imgs[0].size == gt.size

        if rng.random() < c.p_spatial:
            angle = rng.uniform(-c.max_rotation_deg, c.max_rotation_deg)
            tx = rng.uniform(-c.max_translation_px, c.max_translation_px)
            ty = rng.uniform(-c.max_translation_px, c.max_translation_px)
            scale = rng.uniform(1 - c.scale_delta, 1 + c.scale_delta)
            coeffs = _affine_coeffs(imgs[0].size, angle, tx, ty, scale)
            imgs = [_apply_affine(i, coeffs, Image.BILINEAR, 0) for i in imgs]
            if gt is not None:
                gt = _apply_affine(gt, coeffs, Image.NEAREST, 0)   # NEAREST keeps labels intact

        if rng.random() < c.p_shift:
            shift = rng.uniform(-c.shift_max, c.shift_max)         # one shift for the whole stack
            imgs = [_to_img(np.asarray(i, dtype=np.float32) + shift) for i in imgs]

        if rng.random() < c.p_noise:                               # independent noise per slice
            imgs = [_to_img(np.asarray(i, dtype=np.float32)
                            + rng.normal(0.0, c.noise_std, size=np.asarray(i).shape))
                    for i in imgs]

        return imgs, gt
