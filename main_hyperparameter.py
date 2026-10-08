#!/usr/bin/env python3

# MIT License

# Copyright (c) 2025 Hoel Kervadec, Caroline Magg

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

import argparse
import warnings
from typing import Any
from pathlib import Path
from pprint import pprint
from operator import itemgetter
from shutil import copytree, rmtree
import re
import json

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset, Subset

from functools import partial 

from dataset import SliceDataset
from ShallowNet import shallowCNN
from ENet import ENet
from utils import (Dcm,
                   class2one_hot,
                   probs2one_hot,
                   probs2class,
                   tqdm_,
                   dice_coef,
                   save_images)

from losses import (CrossEntropy)

from UNet2_25D import UNet25D, UNet2D
from ResUNet2_25D import ResUNet2D, ResUNet25D  
from dataset import SliceDataset, SliceDataset25D
from losses import CrossEntropy, DiceCELoss

# N_NEIGHBORS = 2                      # 5-slice input; try 1 (3 slices) or 3 (7 slices)
DEFAULT_N_NEIGHBORS = 2  


datasets_params: dict[str, dict[str, Any]] = {}

# K for the number of classes
# Avoids the classes with C (often used for the number of Channel)
# datasets_params["TOY2"] = {'K': 2, 'net': shallowCNN, 'B': 2, 'kernels': 8, 'factor': 2}
# datasets_params["SEGTHOR"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}
# datasets_params["SEGTHOR_CLEAN"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}
# datasets_params["SEGTHOR"] = {'K': 5, 'net': UNet25D, 'B': 8, 'use_25d': True}
datasets_params["SEGTHOR_UNET2D"] = {'K': 5, 'net': UNet2D, 'B': 8, 'plain_unet': True}


def img_transform(img):
    img = img.convert('L')
    img = np.array(img)[np.newaxis, ...]
    img = img / 255  # max <= 1
    img = torch.tensor(img, dtype=torch.float32)
    return img
 
 
def gt_transform(K, img):
    img = np.array(img)[...]
    img = img / (255 / (K - 1)) if K != 5 else img / 63  # max <= 1
    img = torch.tensor(img, dtype=torch.int64)[None, ...]  # Add one dimension to simulate batch
    img = class2one_hot(img, K=K)
    return img[0]
 
 
# patient wise splitting
PATIENT_RE = re.compile(r"(Patient_\d+)")
 
 
def patient_id(stem: str) -> str:
    m = PATIENT_RE.search(stem)
    return m.group(1) if m else stem
 
 
def get_stems(ds, root_dir: Path, split: str, debug: bool) -> list[str]:
    stems = sorted(p.stem for p in (root_dir / split / "gt").glob("*.png"))
    if debug:
        stems = stems[:len(ds)]
    assert len(stems) == len(ds), f"{split}: {len(stems)} files vs dataset length {len(ds)}"
    return stems
 
 
def split_by_patient(stems: list[str], frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    groups = np.array([patient_id(s) for s in stems])
    patients = np.unique(groups)
    rng = np.random.RandomState(seed)
    rng.shuffle(patients)
    n_held = max(1, int(round(frac * len(patients))))
    if n_held >= len(patients):
        raise ValueError(f"Cannot hold out {n_held} of {len(patients)} patients")
    held = np.isin(groups, patients[:n_held])
    return np.flatnonzero(~held), np.flatnonzero(held)
 
 
def patient_folds(stems: list[str], n_folds: int, seed: int) -> list[np.ndarray]:
    groups = np.array([patient_id(s) for s in stems])
    patients = np.unique(groups)
    if len(patients) < n_folds:
        raise ValueError(f"{len(patients)} patients, cannot make {n_folds} folds")
    rng = np.random.RandomState(seed)
    rng.shuffle(patients)
    fold_patients = np.array_split(patients, n_folds)
    return [np.flatnonzero(np.isin(groups, fp)) for fp in fold_patients]
 
 
def get_batch_size(args) -> int:
    return args.batch_size or datasets_params[args.dataset]['B']
 
 
def build_model(args, device) -> tuple[nn.Module, torch.optim.Optimizer]:
    K: int = datasets_params[args.dataset]['K']
    use_25d: bool = datasets_params[args.dataset].get('use_25d', False)
 
    if use_25d:
        in_ch: int = 2 * args.n_neighbors + 1
        net = datasets_params[args.dataset]['net'](in_ch, K)
    elif datasets_params[args.dataset].get('plain_unet', False):
        net = datasets_params[args.dataset]['net'](1, K)      # 2D UNet: 1 input channel
    else:
        kernels: int = datasets_params[args.dataset].get('kernels', 8)
        factor: int = datasets_params[args.dataset].get('factor', 2)
        net = datasets_params[args.dataset]['net'](1, K, kernels=kernels, factor=factor)
 
    net.init_weights()
    net.to(device)
 
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999))
    return net, optimizer
 
 
def build_datasets(args) -> tuple[dict[str, Any], dict[str, list[str]]]:
    K: int = datasets_params[args.dataset]['K']
    use_25d: bool = datasets_params[args.dataset].get('use_25d', False)
    root_dir = args.data_dir if args.data_dir is not None else Path("data") / args.dataset
 
    if use_25d:
        make_set = partial(SliceDataset25D, n_neighbors=args.n_neighbors)
    else:
        make_set = SliceDataset
 
    sets: dict[str, Any] = {}
    for split in ('train', 'val'):
        sets[split] = make_set(split,
                               root_dir,
                               img_transform=img_transform,
                               gt_transform=partial(gt_transform, K),
                               debug=args.debug)
 
    stems: dict[str, list[str]] = {s: get_stems(sets[s], root_dir, s, args.debug) for s in sets}
    return sets, stems
 
 
def make_loaders(args, train_ds, val_ds) -> tuple[DataLoader, DataLoader]:
    B = get_batch_size(args)
    # drop_last: a final training batch of size 1 would crash BatchNorm
    train_loader = DataLoader(train_ds, batch_size=B, num_workers=5, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=B, num_workers=5, shuffle=False)
    return train_loader, val_loader
 
 
@torch.no_grad()
def evaluate(net, loader, device) -> np.ndarray:
    net.eval()
    dices = []
    for data in loader:
        img = data['images'].to(device)
        gt = data['gts'].to(device)
        probs = F.softmax(net(img), dim=1)
        dices.append(dice_coef(probs2one_hot(probs), gt).cpu())
    return torch.cat(dices).mean(dim=0).numpy()  # (K,)
 
 
# training fold
def train_one_fold(args, net, optimizer, device, train_loader, val_loader, K, dest: Path,
                   on_epoch_end=None) -> float:
 
    if args.mode == "full":
        loss_fn = DiceCELoss(K, dice_weight=args.dice_weight, ce_weight=args.ce_weight)
    elif args.mode in ["partial"] and args.dataset == 'SEGTHOR':
        loss_fn = CrossEntropy(idk=[0, 1, 3, 4])  # Do not supervise the heart (class 2)
    else:
        raise ValueError(args.mode, args.dataset)
 
    # Notice one has the length of the _loader_, and the other one of the _dataset_
    log_loss_tra: Tensor = torch.zeros((args.epochs, len(train_loader)))
    log_dice_tra: Tensor = torch.zeros((args.epochs, len(train_loader.dataset), K))
    log_loss_val: Tensor = torch.zeros((args.epochs, len(val_loader)))
    log_dice_val: Tensor = torch.zeros((args.epochs, len(val_loader.dataset), K))
 
    best_dice: float = 0
 
    for e in range(args.epochs):
        for m in ['train', 'val']:
            match m:
                case 'train':
                    net.train()
                    opt = optimizer
                    cm = Dcm
                    desc = f">> Training   ({e: 4d})"
                    loader = train_loader
                    log_loss = log_loss_tra
                    log_dice = log_dice_tra
                case 'val':
                    net.eval()
                    opt = None
                    cm = torch.no_grad
                    desc = f">> Validation ({e: 4d})"
                    loader = val_loader
                    log_loss = log_loss_val
                    log_dice = log_dice_val
 
            with cm():  # Either dummy context manager, or the torch.no_grad for validation
                j = 0
                tq_iter = tqdm_(enumerate(loader), total=len(loader), desc=desc)
                for i, data in tq_iter:
                    img = data['images'].to(device)
                    gt = data['gts'].to(device)
 
                    if opt:
                        opt.zero_grad()
 
                    # Sanity tests to see if loaded and encoded the data correctly
                    assert 0 <= img.min() and img.max() <= 1
                    B, _, W, H = img.shape
 
                    pred_logits = net(img)
                    pred_probs = F.softmax(1 * pred_logits, dim=1)
 
                    # Metrics computation, not used for training
                    pred_seg = probs2one_hot(pred_probs)
                    log_dice[e, j:j + B, :] = dice_coef(pred_seg, gt)
 
                    loss = loss_fn(pred_probs, gt)
                    log_loss[e, i] = loss.item()
 
                    if opt:  # Only for training
                        loss.backward()
                        opt.step()
 
                    if m == 'val':
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', category=UserWarning)
                            predicted_class: Tensor = probs2class(pred_probs)
                            mult: int = 63 if K == 5 else (255 / (K - 1))
                            save_images(predicted_class * mult,
                                        data['stems'],
                                        dest / f"iter{e:03d}" / m)
 
                    j += B  # Keep in mind that _in theory_, each batch might have a different size
                    # For the DSC average: do not take the background class (0) into account:
                    postfix_dict: dict[str, str] = {"Dice": f"{log_dice[e, :j, 1:].mean():05.3f}",
                                                    "Loss": f"{log_loss[e, :i + 1].mean():5.2e}"}
                    if K > 2:
                        postfix_dict |= {f"Dice-{k}": f"{log_dice[e, :j, k].mean():05.3f}"
                                         for k in range(1, K)}
                    tq_iter.set_postfix(postfix_dict)
 
        # save it at each epochs, in case the code crashes or decide to stop it early
        np.save(dest / "loss_tra.npy", log_loss_tra)
        np.save(dest / "dice_tra.npy", log_dice_tra)
        np.save(dest / "loss_val.npy", log_loss_val)
        np.save(dest / "dice_val.npy", log_dice_val)
 
        current_dice: float = log_dice_val[e, :, 1:].mean().item()
        if on_epoch_end is not None:  # e.g. Optuna reporting / pruning
            on_epoch_end(e, current_dice)
        if current_dice > best_dice:
            message = f">>> Improved dice at epoch {e}: {best_dice:05.3f}->{current_dice:05.3f} DSC"
            print(message)
            best_dice = current_dice
            with open(dest / "best_epoch.txt", 'w') as f:
                f.write(message)
 
            best_folder = dest / "best_epoch"
            if best_folder.exists():
                rmtree(best_folder)
            copytree(dest / f"iter{e:03d}", Path(best_folder))
 
            torch.save(net, dest / "bestmodel.pkl")
            torch.save(net.state_dict(), dest / "bestweights.pt")
 
    return best_dice
 
 
def runTraining(args, on_epoch_end=None) -> float:
    print(f">>> Setting up to train on {args.dataset} with {args.mode}")
 
    gpu: bool = args.gpu and torch.cuda.is_available()
    device = torch.device("cuda") if gpu else torch.device("cpu")
    print(f">> Picked {device} to run experiments")
 
    K: int = datasets_params[args.dataset]['K']
    B: int = get_batch_size(args)
    sets, stems = build_datasets(args)
    args.dest.mkdir(parents=True, exist_ok=True)
 
    # pool the original train and val folders
    pooled = ConcatDataset([sets['train'], sets['val']])
    all_stems = stems['train'] + stems['val']
 
    # hold out the test patients (none if test_frac == 0)
    if args.test_frac > 0:
        rest_idx, test_idx = split_by_patient(all_stems, args.test_frac, args.seed)
    else:
        rest_idx, test_idx = np.arange(len(all_stems)), np.array([], dtype=int)
    rest_stems = [all_stems[i] for i in rest_idx]
    test_p = {patient_id(all_stems[i]) for i in test_idx}
    print(f">> Test set: {len(test_p)} patients ({len(test_idx)} slices), "
          f"remaining: {len(rest_idx)} slices")
    with open(args.dest / "test_patients.json", "w") as f:
        json.dump(sorted(test_p), f, indent=2)
 
    # Split the remaining patients into train, val pairs
    if args.n_folds > 1:
        folds = patient_folds(rest_stems, args.n_folds, args.seed)
        splits = [(np.concatenate([folds[i] for i in range(args.n_folds) if i != k]), folds[k])
                  for k in range(args.n_folds)]
    else:
        rel_val = args.val_frac / (1 - args.test_frac)  # val fraction relative to what is left
        splits = [split_by_patient(rest_stems, rel_val, args.seed + 1)]
 
    test_loader = None
    if len(test_idx) > 0:
        test_loader = DataLoader(Subset(pooled, test_idx.tolist()),
                                 batch_size=B, num_workers=5, shuffle=False)
 
    run_ids = range(len(splits)) if args.fold is None else [args.fold]
    val_scores: dict[int, float] = {}
    test_scores: dict[int, float] = {}
 
    for k in run_ids:
        if args.n_folds > 1:
            print(f"\n{'=' * 20} Fold {k + 1}/{args.n_folds} {'=' * 20}")
            fold_dir = args.dest / f"fold_{k}"
        else:
            print(f"\n{'=' * 20} Single train/val/test split {'=' * 20}")
            fold_dir = args.dest
 
        tr_rel, va_rel = splits[k]
        train_idx = rest_idx[tr_rel]
        val_idx = rest_idx[va_rel]
 
        # Sanity checks: no patient shared between train / val / test
        tr_p = {patient_id(all_stems[i]) for i in train_idx}
        va_p = {patient_id(all_stems[i]) for i in val_idx}
        assert not (tr_p & va_p), "Patient leakage between train and val!"
        assert not (test_p & (tr_p | va_p)), "Patient leakage into the test set!"
        n_all = len(train_idx) + len(val_idx) + len(test_idx)
        print(f">> train: {len(tr_p)} patients ({len(train_idx)} slices, {len(train_idx) / n_all:.0%}) | "
              f"val: {len(va_p)} patients ({len(val_idx)} slices, {len(val_idx) / n_all:.0%}) | "
              f"test: {len(test_p)} patients ({len(test_idx)} slices, {len(test_idx) / n_all:.0%})")
 
        net, optimizer = build_model(args, device)
        train_loader, val_loader = make_loaders(args,
                                                Subset(pooled, train_idx.tolist()),
                                                Subset(pooled, val_idx.tolist()))
 
        cb = None
        if on_epoch_end is not None:
            cb = lambda e, d, k=k: on_epoch_end(k * args.epochs + e, d)
 
        val_scores[k] = train_one_fold(args, net, optimizer, device,
                                       train_loader, val_loader, K, fold_dir, cb)
 
        with open(fold_dir / "val_patients.json", "w") as f:
            json.dump(sorted(va_p), f, indent=2)
 
        scores = {"val_best_dice": val_scores[k]}
 
        # Test evaluation with the best val weights of this fold
        if not args.skip_test and test_loader is not None:
            net.load_state_dict(torch.load(fold_dir / "bestweights.pt", map_location=device))
            per_class = evaluate(net, test_loader, device)
            test_scores[k] = float(per_class[1:].mean())
            scores |= {"test_dice": test_scores[k],
                       "test_dice_per_class": [float(x) for x in per_class]}
            print(f">>> Fold {k}: best val Dice {val_scores[k]:.4f} | test Dice {test_scores[k]:.4f}")
 
        with open(fold_dir / "scores.json", "w") as f:
            json.dump(scores, f, indent=2)
 
    val_arr = np.array(list(val_scores.values()))
    print(f"\n>>> Val  best Dice per fold: {np.round(val_arr, 4).tolist()}")
    print(f">>> Val  mean +/- std: {val_arr.mean():.4f} +/- {val_arr.std():.4f}")
    summary: dict[str, Any] = {"val_per_fold": val_scores,
                               "val_mean": float(val_arr.mean()),
                               "val_std": float(val_arr.std())}
    if test_scores:
        test_arr = np.array(list(test_scores.values()))
        print(f">>> Test Dice per fold: {np.round(test_arr, 4).tolist()}")
        print(f">>> Test mean +/- std: {test_arr.mean():.4f} +/- {test_arr.std():.4f}")
        summary |= {"test_per_fold": test_scores,
                    "test_mean": float(test_arr.mean()),
                    "test_std": float(test_arr.std())}
 
    if args.fold is None:
        with open(args.dest / "cv_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
 
    return float(val_arr.mean())
 
 
def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
 
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--dataset', default='SEGTHOR_UNET2D', choices=datasets_params.keys())
    parser.add_argument('--mode', default='full', choices=['partial', 'full'])
    parser.add_argument('--dest', type=Path, required=True,
                        help="Destination directory to save the results (predictions and weights).")
 
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--debug', action='store_true',
                        help="Keep only a fraction (10 samples) of the datasets, "
                             "to test the logics around epochs and logging easily.")
 
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=None,
        help="Folder containing the train and val image folders."
    )
 
    parser.add_argument('--test_frac', default=0.1, type=float,
                        help="Fraction of patients held out as the test set.")
    parser.add_argument('--val_frac', default=0.1, type=float,
                        help="Fraction of patients used for validation in the single-split mode "
                             "(--n_folds 0 or 1). Ignored with CV.")
    parser.add_argument('--n_folds', default=0, type=int,
                        help="K-fold CV on the non-test patients. Validation = (1-test_frac)/n_folds "
                             "of all patients. 0 or 1 = single train/val split.")
    parser.add_argument('--fold', default=None, type=int,
                        help="Run only this fold (0-based), e.g. to parallelise folds over several jobs.")
    parser.add_argument('--seed', default=0, type=int,
                        help="Seed for the patient split (keep identical across fold jobs).")
    parser.add_argument('--skip_test', action='store_true',
                        help="Do not evaluate on the test set (used while tuning hyperparameters).")
 
    # Loss
    parser.add_argument('--dice_weight', default=1.0, type=float, help="Weight of the Dice term in DiceCELoss.")
    parser.add_argument('--ce_weight', default=1.0, type=float, help="Weight of the cross-entropy term in DiceCELoss.")
 
    # Hyperparameters (tuned by hp_tune.py)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--batch_size', default=None, type=int,
                        help="Overrides the per-dataset default batch size.")
    parser.add_argument('--n_neighbors', default=DEFAULT_N_NEIGHBORS, type=int,
                        help="Neighbouring slices on each side (2.5D only): input has 2*n+1 channels.")
    return parser
 
 
def main():
    parser = get_parser()
    args = parser.parse_args()
 
    if args.fold is not None and not (args.n_folds > 1 and 0 <= args.fold < args.n_folds):
        parser.error("--fold requires --n_folds > 1 and 0 <= fold < n_folds")
    if not (0 <= args.test_frac < 1) or (args.n_folds <= 1 and not (0 < args.val_frac < 1 - args.test_frac)):
        parser.error("Need 0 <= test_frac < 1 and 0 < val_frac < 1 - test_frac")
 
    pprint(args)
 
    runTraining(args)
 
 
if __name__ == '__main__':
    main()