"""Offline augmentation tuning: every Optuna trial writes an augmented copy of the TRAINING
patients to disk, trains on it, and deletes it again. Same search space, reference trials
and validation split as aug_tune.py (the online version), so the two are comparable.

    python aug_tune_offline.py --dataset SEGTHOR_UNET2D --mode full --gpu --data_dir data/hu_wide_rs18 \
        --normalize --dice_weight 2.7 --ce_weight 1.0 --lr 5e-4 --batch_size 8  \
        --epochs 20 --n_trials 15 --n_copies 1 --dest results/aug_offline_UNET2D --mlflow

Output in --dest: trials.csv, tuning_results.png, best_aug.json.
Validation patients are never augmented; a pre-flight check enforces this before training starts.
"""
import argparse
import copy
import gc
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from shutil import rmtree

import optuna
import torch

from aug_tune import suggest_config, combo, save_results, SPATIAL, SHIFT, NOISE
from tune_aug_effect import Augmenter
from main_hyperparameter import get_parser, runTraining, datasets_params, split_by_patient, patient_id
from offline_augment import (build_offline_dataset, list_source_stems,
                             single_split_train_patients, is_aug_stem)

VAL_FRAC = 0.2   # same as hp_tune.py / aug_tune.py


def parse():
    tp = argparse.ArgumentParser(add_help=False)
    tp.add_argument('--n_trials', type=int, default=30)
    tp.add_argument('--study_name', default=None, help="Defaults to offline_<dataset>.")
    tp.add_argument('--n_copies', type=int, default=1,
                    help="Augmented copies per training slice. Epoch time grows with (1 + n_copies).")
    tp.add_argument('--keep_trials', action='store_true', help="Keep trial result folders.")
    tp.add_argument('--keep_data', action='store_true', help="Keep the generated datasets (large!).")
    tp.add_argument('--mlflow', action='store_true', help="Log every trial to MLflow.")
    tune_args, rest = tp.parse_known_args()
    base_args = get_parser().parse_args(rest)
    return tune_args, base_args


def preflight(data_dir: Path, expected_val: set[str], seed: int) -> None:
    """Replicate the split runTraining will do on the generated folder and check it."""
    stems = sorted(p.stem for p in (data_dir / "train" / "gt").glob("*.png"))
    _, va = split_by_patient(stems, VAL_FRAC, seed + 1)
    val_stems = [stems[i] for i in va]
    got = {patient_id(s) for s in val_stems}
    assert got == expected_val, f"val patients differ from the online split: {sorted(got ^ expected_val)}"
    bad = [s for s in val_stems if is_aug_stem(s)]
    assert not bad, f"{len(bad)} augmented slices ended up in validation, e.g. {bad[0]}"


def make_objective(base_args, tune_args, src_dir, is_25d, train_patients, expected_val):
    if tune_args.mlflow:
        import mlflow

    def objective(trial: optuna.Trial) -> float:
        cfg = suggest_config(trial)
        trial.set_user_attr('aug', asdict(cfg))

        args = copy.deepcopy(base_args)
        args.aug = None                   # offline: no on-the-fly augmentation
        args.aug_json = None
        args.dest = base_args.dest / f"trial_{trial.number:03d}"
        data_dir = base_args.dest / f"trial_{trial.number:03d}_data"
        args.skip_test = True
        args.test_frac = 0.0
        args.val_frac = VAL_FRAC
        args.n_folds = 0
        args.fold = None

        def report(step: int, dice: float):
            if tune_args.mlflow:
                mlflow.log_metric('val_dice', dice, step=step)
            trial.report(dice, step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        run = mlflow.start_run(run_name=f"offline_trial_{trial.number:03d}") if tune_args.mlflow else nullcontext()
        with run:
            try:
                stats = build_offline_dataset(src_dir, data_dir, Augmenter(cfg), tune_args.n_copies,
                                              train_patients, per_volume=is_25d,
                                              seed=base_args.seed * 10_000 + trial.number)
                print(f">> Offline data: {stats}")
                trial.set_user_attr('offline', stats)
                preflight(data_dir, expected_val, base_args.seed)
                args.data_dir = data_dir

                if tune_args.mlflow:
                    mlflow.log_params({'trial': trial.number, 'combo': combo(trial.params),
                                       'lr': args.lr, 'batch_size': args.batch_size,
                                       'n_neighbors': args.n_neighbors, 'n_copies': tune_args.n_copies,
                                       **asdict(cfg)})
                    mlflow.log_metrics({k: v for k, v in stats.items()})

                value = runTraining(args, on_epoch_end=report)
            except torch.cuda.OutOfMemoryError:
                gc.collect()
                torch.cuda.empty_cache()
                raise optuna.TrialPruned()
            finally:
                if not tune_args.keep_data:
                    rmtree(data_dir, ignore_errors=True)
                if not tune_args.keep_trials:
                    rmtree(args.dest, ignore_errors=True)
            if tune_args.mlflow:
                mlflow.log_metric('best_val_dice', value)
            return value

    return objective


if __name__ == '__main__':
    tune_args, base_args = parse()
    base_args.dest.mkdir(parents=True, exist_ok=True)
    is_25d = datasets_params[base_args.dataset].get('use_25d', False)
    src_dir = base_args.data_dir if base_args.data_dir is not None else Path("data") / base_args.dataset

    # patients that may get augmented copies = training patients of the (fixed) single split
    stems = list_source_stems(src_dir)
    train_patients = single_split_train_patients(stems, 0.0, VAL_FRAC, base_args.seed)
    expected_val = {patient_id(s) for s in stems} - train_patients
    print(f">> {len(train_patients)} training patients (augmented), {len(expected_val)} validation patients (untouched)")

    study = optuna.create_study(
        study_name=tune_args.study_name or f"offline_{base_args.dataset}",
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=0, n_startup_trials=8),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=8),
        storage=f"sqlite:///{(base_args.dest / 'optuna.db').as_posix()}",
        load_if_exists=True,
    )

    if len(study.trials) == 0:   # reference trials, identical to aug_tune.py
        off = {'use_spatial': False, 'use_shift': False, 'use_noise': False}
        study.enqueue_trial(off)
        study.enqueue_trial({**off, 'use_spatial': True, **SPATIAL})
        study.enqueue_trial({**off, 'use_shift': True, **SHIFT})
        study.enqueue_trial({**off, 'use_noise': True, **NOISE})
        study.enqueue_trial({'use_spatial': True, 'use_shift': True, 'use_noise': True,
                             **SPATIAL, **SHIFT, **NOISE})

    study.optimize(make_objective(base_args, tune_args, src_dir, is_25d, train_patients, expected_val),
                   n_trials=tune_args.n_trials)

    print("\n>>> Best trial")
    for k, v in study.best_trial.user_attrs['aug'].items():
        print(f"    {k} = {v}")
    print(f"    best validation Dice = {study.best_value:.4f}")
    save_results(study, base_args.dest)
