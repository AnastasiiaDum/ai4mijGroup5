import numpy as np
from PIL import Image
    
MAX_ROTATION_DEG = 10
MAX_TRANSLATION_PX = 10
SCALE_RANGE = (0.9, 1.1)
INTENSITY_SHIFT_RANGE = (-15, 15)
GAUSSIAN_NOISE_STD = 5.0
P_SPATIAL = 0.5
P_INTENSITY_SHIFT = 0.5
P_NOISE = 0.5
    

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
    return (a,b,c,d,e,f)

def _apply_affine (img, coeffs, resample, fillcolor):
    return img.transform(img.size, Image.AFFINE, coeffs, resample=resample, fillcolor=fillcolor)

def augment_pair(img: Image.Image, gt: Image.Image, rng=None) -> tuple[Image.Image, Image.Image]:
    rng = rng or np.random.default_rng()
    assert gt is None or img.size == gt.size

    if rng.random() < P_SPATIAL:
        angle = rng.uniform(-MAX_ROTATION_DEG, MAX_ROTATION_DEG)
        tx = rng.uniform(-MAX_TRANSLATION_PX, MAX_TRANSLATION_PX)
        ty = rng.uniform(-MAX_TRANSLATION_PX, MAX_TRANSLATION_PX)
        scale = rng.uniform(*SCALE_RANGE)
        coeffs = _affine_coeffs(img.size, angle, tx, ty, scale)
        img = _apply_affine(img, coeffs, Image.BILINEAR, fillcolor = 0)
        if gt is not None: 
            gt = _apply_affine(gt, coeffs, Image.NEAREST, fillcolor = 0)

    if rng.random() < P_INTENSITY_SHIFT:
        shift = rng.uniform(*INTENSITY_SHIFT_RANGE)
        arr = np.asarray(img, dtype=np.float32) + shift
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    if rng.random() < P_NOISE:
        noise = rng.normal(0.0, GAUSSIAN_NOISE_STD, size=np.asarray(img).shape)
        arr = np.asarray(img, dtype=np.float32) + noise
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


    return img, gt

    

        
    