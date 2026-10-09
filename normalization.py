from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def compute_stats(files: list[Path]) -> dict:
    if not files:
        raise RuntimeError("No image files given to compute normalization stats")

    count, total, total_sq = 0, 0.0, 0.0
    for path in files:
        x = np.asarray(Image.open(path).convert("L"), dtype=np.float64) / 255.0
        count += x.size
        total += x.sum()
        total_sq += np.square(x).sum()

    mean = total / count
    std = float(np.sqrt(max(total_sq / count - mean ** 2, 0.0)))
    if std == 0:
        raise ValueError("Standard deviation is zero; check the images.")
    return {"mean": float(mean), "std": std, "n_slices": len(files)}


def normalize(img, stats: dict | None):
    if stats is None:
        return img
    return (img - stats["mean"]) / stats["std"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute normalization stats of a folder of PNGs.")
    parser.add_argument("--img_dir", type=Path, required=True)
    args = parser.parse_args()
    stats = compute_stats(sorted(args.img_dir.glob("*.png")))
    print(f"{stats['n_slices']} slices | mean = {stats['mean']:.6f} | std = {stats['std']:.6f}")