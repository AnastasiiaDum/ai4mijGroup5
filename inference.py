"""
Slice-level predictions are mapped back to the original grid (undoing the optional in-plane 
resampling & centre pad/crop and the 512 -> 256 resize) by interpolating the softmax 
probabilities and taking the argmax, which gives smoother borders than resizing hard labels.

destination = results/final_UNET2D
python inference.py --run_dir $destination --split val        # validation patients
python inference.py --run_dir $destination --split heldout    # held-out patients (test_patients.json)
python inference.py --run_dir $destination --split test       # official test set (data/test)

--run_dir supplies bestweights.pt, norm_stats.json (if present) and the patient list;
--split picks the CT folder and patient list; the output goes to <run_dir>/<split>_pred.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import nibabel as nib

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom

from slice_segthor import center_pad_crop, norm_arr, norm_arr_fixed_hu, norm_arr_percentile, resize_

K = 5 
CLASS_NAMES = ("background", "esophagus", "heart", "trachea", "aorta")


# Model
def build_arch(arch: str, n_neighbors: int) -> tuple[torch.nn.Module, list[int]]:
    if arch == "unet2d":
        from UNet2_25D import UNet2D
        return UNet2D(1, K), [0]
    
    if arch == "resunet2d":
        from ResUNet2_25D import ResUNet2D
        return ResUNet2D(1, K), [0]

    offsets = list(range(-n_neighbors, n_neighbors + 1))
    if arch == "unet25d":
        from UNet2_25D import UNet25D
        return UNet25D(len(offsets), K), offsets
    
    if arch == "resunet25d":
        from ResUNet2_25D import ResUNet25D
        return ResUNet25D(len(offsets), K), offsets
    
    raise ValueError(f"Unknown architecture: {arch}")


def load_net(args, device) -> tuple[torch.nn.Module, list[int]]:
    net, offsets = build_arch(args.arch, args.n_neighbors)

    checkpoint = torch.load(args.weights, map_location=device, weights_only=False)
    if isinstance(checkpoint, torch.nn.Module): # bestmodel.pkl
        net = checkpoint
    else: # bestweights.pt
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        net.load_state_dict(checkpoint)
    
    net.to(device).eval()
    
    return net, offsets


# Case discovery
def find_cases(input_path: Path, wanted: set[str] | None) -> dict[str, Path]:
    """ Find CT files and store them by patient ID. """
    
    if input_path.is_file():
        files = [input_path]
    else:
        files = [p for p in sorted(input_path.rglob("*.nii.gz"))
                 if not p.name.lower().startswith("gt")]
    
    cases: dict[str, Path] = {}
    for p in files:
        pid = p.name[:-len(".nii.gz")]
        
        if wanted is not None and pid not in wanted:
            continue

        cases[pid] = p
    
    if not cases:
        raise FileNotFoundError(f"No CT .nii.gz files found in {input_path}"
                                + (f" for patients {sorted(wanted)}" if wanted else ""))
    
    return cases


# Preprocessing
def preprocess(ct: np.ndarray, zooms, args) -> tuple[np.ndarray, tuple[int, int] | None]:
    """ CT (X, Y, Z) -> uint8 slices (Z, H, W) + in-plane size after resampling. """
    shape = tuple(args.shape)
    resampled_xy = None

    if args.target_spacing is not None:
        dx, dy = float(zooms[0]), float(zooms[1])
        ct = zoom(ct.astype(np.float32),
                  (dx / args.target_spacing, dy / args.target_spacing, 1.0),
                  order=1, mode="nearest")
        resampled_xy = (int(ct.shape[0]), int(ct.shape[1]))
        ct = center_pad_crop(ct, shape, fill=-1000.0)

    if args.intensity_mode == "percentile":
        norm_ct = norm_arr_percentile(ct)
    elif args.intensity_mode == "fixed_hu":
        norm_ct = norm_arr_fixed_hu(ct, hu_min=args.hu_min, hu_max=args.hu_max)
    else:
        norm_ct = norm_arr(ct)

    slices = np.stack([resize_(norm_ct[:, :, z], shape).astype(np.uint8)
                       for z in range(norm_ct.shape[2])], axis=0)
    
    return slices, resampled_xy


# Mapping predictions back to the original grid
def pad_crop_slices(n: int, m: int) -> tuple[slice, slice]:
    """ 
    Inverse of slice_segthor.center_pad_crop along one axis.
    n = size before pad/crop, m = size after. 
    Returns (slice into the network output, slice into the n-sized grid).
    """
    if n >= m: # it was cropped
        s = (n - m) // 2
        return slice(0, m), slice(s, s + m)
    
    s = (m - n) // 2 # it was padded
    
    return slice(s, s + n), slice(0, n)


def probs_to_original(probs: torch.Tensor, orig_xy: tuple[int, int], resampled_xy: tuple[int, int] | None) -> torch.Tensor:
    """ Probs (B, K, H, W) -> uint8 labels (B, X, Y) on the original in-plane grid. """
    
    if resampled_xy is None:
        up = F.interpolate(probs, size=orig_xy, mode="bilinear", align_corners=False)
    else:
        B, Kc, H, W = probs.shape
        full = probs.new_zeros((B, Kc, *resampled_xy))
        
        # area cropped away during training = background
        full[:, 0] = 1.0 
        (sx, dx), (sy, dy) = (pad_crop_slices(n, m) for n, m in zip(resampled_xy, (H, W)))
        full[:, :, dx, dy] = probs[:, :, sx, sy]
        up = F.interpolate(full, size=orig_xy, mode="bilinear", align_corners=True)
    
    return up.argmax(dim=1).to(torch.uint8).cpu()


@torch.no_grad()
def predict_volume(net, slices: np.ndarray, offsets: list[int], norm: dict | None,
                   device, batch_size: int, orig_xy: tuple[int, int],
                   resampled_xy: tuple[int, int] | None) -> np.ndarray:
    
    vol = torch.from_numpy(slices).float().div_(255.0) # (Z, H, W) in [0, 1]
    
    if norm is not None: # same z-score as training
        vol = (vol - norm["mean"]) / norm["std"]
    
    Z = vol.shape[0]
    offs = torch.tensor(offsets)
    out = np.zeros((*orig_xy, Z), dtype=np.uint8)

    for start in range(0, Z, batch_size):
        zs = torch.arange(start, min(start + batch_size, Z))
        idx = (zs[:, None] + offs[None, :]).clamp_(0, Z - 1) # repeat edge slices
        x = vol[idx].to(device) 
        
        probs = F.softmax(net(x), dim=1)
        labels = probs_to_original(probs, orig_xy, resampled_xy) 
        out[:, :, start:start + len(zs)] = labels.permute(1, 2, 0).numpy()
    
    return out


# Main
def run(args) -> None:
    
    print(f"Model: {args.arch}" + (f" (n_neighbors={args.n_neighbors})" if "25d" in args.arch else "")
          + f", weights: {args.weights}")
    print(f"Preprocessing: {args.intensity_mode}"
          + (f" [{args.hu_min:g}, {args.hu_max:g}] HU" if args.intensity_mode == "fixed_hu" else "")
          + f", target_spacing={args.target_spacing} | shape={tuple(args.shape)}"
          + f", normalize={'yes' if args.norm_stats else 'no'}")

    wanted = None
    if args.patients:
        wanted = set(args.patients)
    if args.patients_json:
        wanted = (wanted or set()) | set(json.loads(Path(args.patients_json).read_text()))
    cases = find_cases(args.input, wanted)
    print(f"{len(cases)} patient(s) to predict")

    norm = json.loads(Path(args.norm_stats).read_text()) if args.norm_stats else None
    if norm is not None:
        print(f"z-score normalisation: mean={norm['mean']:.4f} std={norm['std']:.4f}")

    device = torch.device("cuda" if (not args.cpu) and torch.cuda.is_available() else "cpu")
    net, offsets = load_net(args, device)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "inference_config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))

    for pid, path in cases.items():
        nii = nib.load(str(path))
        ct = np.asarray(nii.dataobj)
        if ct.ndim != 3:
            raise ValueError(f"{path}: expected a 3D volume, got shape {ct.shape}")
        zooms = nii.header.get_zooms()

        slices, resampled_xy = preprocess(ct, zooms, args)
        pred = predict_volume(net, slices, offsets, norm, device, args.batch_size,
                              ct.shape[:2], resampled_xy)

        hdr = nii.header.copy()
        hdr.set_data_dtype(np.uint8)
        out_img = nib.Nifti1Image(pred, nii.affine, header=hdr)
        nib.save(out_img, str(args.output / f"{pid}.nii.gz"))

        counts = np.bincount(pred.ravel(), minlength=K)
        missing = [CLASS_NAMES[c] for c in range(1, K) if counts[c] == 0]
        print(f"   {pid}: shape {pred.shape}, voxels per class "
              + ", ".join(f"{CLASS_NAMES[c]}={counts[c]}" for c in range(1, K))
              + (f", WARNING no {', '.join(missing)} predicted" if missing else ""))

    print(f"Done! Predictions saved to {args.output}")


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CT NIfTI -> label NIfTI inference (original space)")

    p.add_argument("--run_dir", type=Path, default=None,
                   help="Training folder (--destination): provides bestweights.pt, "
                        "norm_stats.json and the patient lists.")
    p.add_argument("--split", choices=["val", "heldout", "test"], default=None,
                   help="val = val_patients.json, heldout = test_patients.json (both from --data_dir/train), "
                        "test = official test CTs in --data_dir/test.")
    p.add_argument("--data_dir", type=Path, default=Path("data"),
                   help="Folder with the original train/ and test/ CT folders.")
    p.add_argument("--input", type=Path, default=None,
                   help="A CT .nii.gz file, or a folder searched recursively for CTs (GT files ignored).")
    p.add_argument("--output", type=Path, default=None, help="Folder for Patient_XX.nii.gz predictions.")
    p.add_argument("--weights", type=Path, default=None,
                   help="bestweights.pt (state dict) or bestmodel.pkl (whole model).")
    p.add_argument("--patients", nargs="+", default=None, help="Only predict these patient ids.")
    p.add_argument("--patients_json", type=Path, default=None,
                   help="JSON list of patient ids (val_patients.json / test_patients.json).")
    p.add_argument("--norm_stats", type=Path, default=None,
                   help="norm_stats.json from training with --normalize (auto-detected in --run_dir).")

    # Model must match the trained model
    p.add_argument("--arch", default="unet2d", choices=["unet2d", "unet25d", "resunet2d", "resunet25d"])
    p.add_argument("--n_neighbors", type=int, default=2,
                   help="2.5D only: neighbours per side used at training (1 = 3 slices, 2 = 5 slices).")

    p.add_argument("--shape", type=int, nargs=2, default=[256, 256])
    p.add_argument("--intensity_mode", choices=["baseline", "percentile", "fixed_hu"], default="fixed_hu")
    p.add_argument("--hu_min", type=float, default=-1000.0)
    p.add_argument("--hu_max", type=float, default=400.0)
    p.add_argument("--target_spacing", type=float, default=1.8,
                   help="In-plane resampling in mm used at slicing time (use 0 for none).")

    p.add_argument("--cpu", action="store_true", help="Force CPU (GPU is used automatically if available).")
    p.add_argument("--gpu", action="store_true", help="Kept for compatibility; GPU is now automatic.")
    p.add_argument("--batch_size", type=int, default=16)

    a = p.parse_args()

    # Resolve shortcuts
    if a.split is not None:
        if a.run_dir is None:
            p.error("--split needs --run_dir")
        if a.split == "test":
            a.input = a.input or a.data_dir / "test"
        else:
            a.input = a.input or a.data_dir / "train"
            name = "val_patients.json" if a.split == "val" else "test_patients.json"
            if a.patients is None and a.patients_json is None:
                a.patients_json = a.run_dir / name
        a.output = a.output or a.run_dir / f"{a.split}_pred"
    
    if a.run_dir is not None:
        a.weights = a.weights or a.run_dir / "bestweights.pt"
        if a.norm_stats is None and (a.run_dir / "norm_stats.json").exists():
            a.norm_stats = a.run_dir / "norm_stats.json"
    
    if a.target_spacing is not None and a.target_spacing <= 0:
        a.target_spacing = None
    
    missing = [n for n in ("input", "output", "weights") if getattr(a, n) is None]
    if missing:
        p.error("missing " + ", ".join("--" + m for m in missing) + " (or use --run_dir with --split)")
    
    return a


if __name__ == "__main__":
    run(get_args())
