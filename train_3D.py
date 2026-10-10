"""3D training path for main_hyperparameter.py (UNet3D / ResUNet3D).

Use data, slpit, normalization and optuna callback from main_hyperparameter.py. 
Only the data handling and the training loop differ:

* Training: random sub-volumes of `patch_depth` consecutive slices
  (full 256x256 in-plane) are sampled from each training patient.
* Validation / test: each patient volume is predicted with a sliding window
  along z (50% overlap, softmax averaged), then scored slice by slice with the
  same Dice function as the 2D models, so the numbers are directly comparable.
  Validation predictions are saved as per-slice PNGs, like the 2D models.
"""
from __future__ import annotations
 
import os
import re
import warnings
from collections import defaultdict
from pathlib import Path
from shutil import copytree, rmtree
 
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
 
from losses import DiceCELoss
from normalization import normalize
from utils import dice_coef, probs2class, probs2one_hot, save_images, tqdm_
 
STEM_RE = re.compile(r"^(?P<patient>.+)_(?P<z>\d+)$")
# Windows starts workers by pickling the dataset (all volumes) into each one
WORKERS = 0 if os.name == "nt" else 4
 
 
def group_slices(all_stems: list[str], all_img_paths: list[Path], indices) -> dict[str, list[tuple]]:
    """{patient: [(z, img_path, gt_path, stem), ...] sorted by z} for the given pooled indices."""
    groups: dict[str, list[tuple]] = defaultdict(list)
    for i in indices:
        stem = all_stems[i]
        m = STEM_RE.match(stem)
        assert m, f"Unexpected slice name: {stem}"
        img_path = Path(all_img_paths[i])
        gt_path = img_path.parent.parent / "gt" / img_path.name
        groups[m["patient"]].append((int(m["z"]), img_path, gt_path, stem))
    return {p: sorted(v) for p, v in sorted(groups.items())}
 
 
class VolumeDataset(Dataset):
    """Patient volumes stacked from the per-slice PNGs (kept in memory as uint8).
 
    train=True : each item is a random patch (1, D, H, W) image + (K, D, H, W) one-hot GT;
                 every patient is sampled `samples_per_volume` times per epoch.
    train=False: each item is a whole patient volume (use batch_size=1).
    """
 
    def __init__(self, patients: dict[str, list[tuple]], K: int, patch_depth: int,
                 samples_per_volume: int = 1, train: bool = True):
        self.K, self.depth, self.train = K, patch_depth, train
        self.samples_per_volume = samples_per_volume if train else 1
        self.names: list[str] = []
        self.imgs: list[np.ndarray] = []
        self.gts: list[np.ndarray] = []
        self.stems: list[list[str]] = []
        for patient, slices in patients.items():
            self.names.append(patient)
            self.imgs.append(np.stack([np.asarray(Image.open(s[1]).convert("L")) for s in slices]))
            gt = np.stack([np.asarray(Image.open(s[2]).convert("L")) for s in slices]) // 63
            assert gt.max() < K, (patient, gt.max())
            self.gts.append(gt.astype(np.uint8))
            self.stems.append([s[3] for s in slices])
        mode = "train" if train else "eval"
        print(f">> Created 3D {mode} dataset: {len(self.names)} volumes"
              + (f", {len(self)} patches of {patch_depth} slices per epoch" if train else ""))
 
    def __len__(self) -> int:
        return len(self.names) * self.samples_per_volume
 
    def _to_tensors(self, img: np.ndarray, gt: np.ndarray) -> tuple[Tensor, Tensor]:
        img_t = torch.from_numpy(img.astype(np.float32) / 255.0)[None]        
        gt_t = F.one_hot(torch.from_numpy(gt.astype(np.int64)), self.K)         
        # int32 one-hot, same dtype as the 2D gt_transform (class2one_hot)
        return img_t, gt_t.permute(3, 0, 1, 2).contiguous().to(torch.int32)   
 
    def __getitem__(self, idx: int) -> dict:
        v = idx % len(self.names)
        img, gt = self.imgs[v], self.gts[v]
        if not self.train:
            img_t, gt_t = self._to_tensors(img, gt)
            return {"images": img_t, "gts": gt_t, "stems": self.stems[v], "patient": self.names[v]}
 
        z = img.shape[0]
        if z < self.depth:                                
            pad = self.depth - z
            img = np.pad(img, ((0, pad), (0, 0), (0, 0)))
            gt = np.pad(gt, ((0, pad), (0, 0), (0, 0)))
            z = self.depth
        z0 = int(torch.randint(0, z - self.depth + 1, (1,)))  
        img_t, gt_t = self._to_tensors(img[z0:z0 + self.depth], gt[z0:z0 + self.depth])
        return {"images": img_t, "gts": gt_t}
 
 
def _collate_eval(batch: list[dict]) -> dict:
    assert len(batch) == 1, "use batch_size=1 for whole-volume evaluation"
    item = batch[0]
    return {"images": item["images"][None], "gts": item["gts"][None],
            "stems": item["stems"], "patient": item["patient"]}
 
 
def to_slices(x: Tensor) -> Tensor:
    """(B, K, D, H, W) -> (B*D, K, H, W): lets the 2D loss / Dice code score each slice."""
    b, k, d, h, w = x.shape
    return x.permute(0, 2, 1, 3, 4).reshape(b * d, k, h, w)
 
 
@torch.no_grad()
def predict_volume(net, img: Tensor, patch_depth: int, norm) -> Tensor:
    """Sliding-window prediction along z. img: (1, 1, Z, H, W) in [0, 1]. Returns probs (1, K, Z, H, W)."""
    z = img.shape[2]
    pad = max(0, patch_depth - z)
    if pad:
        img = F.pad(img, (0, 0, 0, 0, 0, pad))
    zp = img.shape[2]
    stride = max(1, patch_depth // 2)
    starts = list(range(0, zp - patch_depth + 1, stride))
    if starts[-1] != zp - patch_depth:
        starts.append(zp - patch_depth)
 
    acc = None
    count = torch.zeros((1, 1, zp, 1, 1), device=img.device)
    for s in starts:
        patch = normalize(img[:, :, s:s + patch_depth], norm)
        probs = F.softmax(net(patch), dim=1)
        if acc is None:
            acc = torch.zeros((1, probs.shape[1], zp, *img.shape[-2:]), device=img.device)
        acc[:, :, s:s + patch_depth] += probs
        count[:, :, s:s + patch_depth] += 1
    return (acc / count)[:, :, :z]
 
 
@torch.no_grad()
def score_volumes(net, dataset: VolumeDataset, device, patch_depth: int, norm,
                  save_dir: Path | None = None) -> Tensor:
    """Per-slice Dice (N_slices, K) over all volumes; optionally saves the predicted slices as PNGs."""
    net.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=min(WORKERS, 2), collate_fn=_collate_eval)
    dices = []
    for data in tqdm_(loader, desc=">> Evaluating volumes"):
        img = data["images"].to(device)
        gt = data["gts"].to(device)
        assert 0 <= img.min() and img.max() <= 1
        probs = predict_volume(net, img, patch_depth, norm)
        probs_2d, gt_2d = to_slices(probs), to_slices(gt)
        dices.append(dice_coef(probs2one_hot(probs_2d), gt_2d).cpu())
        if save_dir is not None:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                K = probs.shape[1]
                mult = 63 if K == 5 else (255 / (K - 1))
                save_images(probs2class(probs_2d) * mult, data["stems"], save_dir)
    return torch.cat(dices)
 
 
def evaluate_3d(net, dataset: VolumeDataset, device, patch_depth: int, norm=None) -> np.ndarray:
    """Mean per-class Dice over all slices, same definition as `evaluate` for the 2D models."""
    return score_volumes(net, dataset, device, patch_depth, norm).mean(dim=0).numpy()
 
 
def train_one_fold_3d(args, net, optimizer, device, train_ds: VolumeDataset, val_ds: VolumeDataset,
                      K: int, B: int, dest: Path, on_epoch_end=None, norm=None) -> float:
    if args.mode != "full":
        raise ValueError("3D models only support --mode full")
    loss_fn = DiceCELoss(K, dice_weight=args.dice_weight, ce_weight=args.ce_weight)
    train_loader = DataLoader(train_ds, batch_size=B, shuffle=True, num_workers=WORKERS, drop_last=True)
    if len(train_loader) == 0:
        raise ValueError(f"batch_size={B} is larger than the {len(train_ds)} training patches per epoch")
 
    log_loss_tra = torch.zeros((args.epochs, len(train_loader)))
    log_dice_val = None
    best_dice = 0.0
 
    for e in range(args.epochs):
        net.train()
        tq_iter = tqdm_(enumerate(train_loader), total=len(train_loader), desc=f">> Training   ({e: 4d})")
        for i, data in tq_iter:
            img = data["images"].to(device)
            gt = data["gts"].to(device)
            assert 0 <= img.min() and img.max() <= 1
            img = normalize(img, norm)
 
            optimizer.zero_grad()
            pred_probs = F.softmax(net(img), dim=1)
            loss = loss_fn(to_slices(pred_probs), to_slices(gt))
            loss.backward()
            optimizer.step()
 
            log_loss_tra[e, i] = loss.item()
            tq_iter.set_postfix({"Loss": f"{log_loss_tra[e, :i + 1].mean():5.2e}"})
 
        iter_dir = dest / f"iter{e:03d}" / "val"
        dice_e = score_volumes(net, val_ds, device, args.patch_depth, norm, save_dir=iter_dir)
        if log_dice_val is None:
            log_dice_val = torch.zeros((args.epochs, *dice_e.shape))
        log_dice_val[e] = dice_e
 
        np.save(dest / "loss_tra.npy", log_loss_tra)
        np.save(dest / "dice_val.npy", log_dice_val)
 
        per_class = dice_e.mean(dim=0)
        current_dice = float(dice_e[:, 1:].mean())
        print(f">> Validation ({e: 4d}): Dice={current_dice:05.3f} "
              + " ".join(f"Dice-{k}={per_class[k]:05.3f}" for k in range(1, K)))

        # Optuna reporting / pruning
        if on_epoch_end is not None:            
            on_epoch_end(e, current_dice)
        if current_dice > best_dice:
            message = f">>> Improved dice at epoch {e}: {best_dice:05.3f}->{current_dice:05.3f} DSC"
            print(message)
            best_dice = current_dice
            (dest / "best_epoch.txt").write_text(message)
            best_folder = dest / "best_epoch"
            if best_folder.exists():
                rmtree(best_folder)
            copytree(dest / f"iter{e:03d}", best_folder)
            torch.save(net, dest / "bestmodel.pkl")
            torch.save(net.state_dict(), dest / "bestweights.pt")
 
    return best_dice