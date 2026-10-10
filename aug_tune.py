"""Tune augmentation with Optuna: which augmentations to use AND how strong they are.
lr / batch size / n_neighbors stay fixed (pass the best values from hp_tune.py).

    python aug_tune.py --dataset SEGTHOR_UNET25D --mode full --gpu --data_dir data/hu_wide_rs18 \
        --normalize --dice_weight 2.7 --ce_weight 1.0 --lr 5e-4 --batch_size 8 --n_neighbors 2 \
        --epochs 20 --n_trials 30 --dest results/aug_UNET25D [--mlflow]

The first 5 trials are fixed references: no augmentation, each augmentation alone with
your original values, and all three together. After that TPE explores combinations and magnitudes.
Output in --dest: trials.csv, tuning_results.png, best_aug.json (use it with --aug_json).
"""
import argparse
import copy
import gc
import json
from contextlib import nullcontext
from dataclasses import asdict
from shutil import rmtree

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import optuna
import torch
from optuna.trial import TrialState

from augmentation import AugConfig, Augmenter
from main_hyperparameter import get_parser, runTraining

SPATIAL = {'p_spatial': 0.5, 'max_rotation_deg': 10.0, 'max_translation_px': 10.0, 'scale_delta': 0.1}
SHIFT = {'p_shift': 0.5, 'shift_max': 15.0}
NOISE = {'p_noise': 0.5, 'noise_std': 5.0}


def parse():
    tp = argparse.ArgumentParser(add_help=False)
    tp.add_argument('--n_trials', type=int, default=30)
    tp.add_argument('--study_name', default=None, help="Defaults to aug_<dataset>.")
    tp.add_argument('--keep_trials', action='store_true')
    tp.add_argument('--mlflow', action='store_true', help="Log every trial to MLflow.")
    tune_args, rest = tp.parse_known_args()
    base_args = get_parser().parse_args(rest)
    return tune_args, base_args


def suggest_config(trial: optuna.Trial) -> AugConfig:
    cfg = AugConfig(p_spatial=0.0, p_shift=0.0, p_noise=0.0)       # everything off unless chosen
    if trial.suggest_categorical('use_spatial', [True, False]):
        cfg.p_spatial = trial.suggest_float('p_spatial', 0.2, 1.0)
        cfg.max_rotation_deg = trial.suggest_float('max_rotation_deg', 2.0, 25.0)
        cfg.max_translation_px = trial.suggest_float('max_translation_px', 0.0, 30.0)
        cfg.scale_delta = trial.suggest_float('scale_delta', 0.0, 0.25)
    if trial.suggest_categorical('use_shift', [True, False]):
        cfg.p_shift = trial.suggest_float('p_shift', 0.2, 1.0)
        cfg.shift_max = trial.suggest_float('shift_max', 5.0, 40.0)
    if trial.suggest_categorical('use_noise', [True, False]):
        cfg.p_noise = trial.suggest_float('p_noise', 0.2, 1.0)
        cfg.noise_std = trial.suggest_float('noise_std', 1.0, 15.0)
    return cfg


def combo(params: dict) -> str:
    names = [n for n, k in (('spatial', 'use_spatial'), ('shift', 'use_shift'), ('noise', 'use_noise'))
             if params.get(k)]
    return '+'.join(names) or 'none'


def make_objective(base_args, tune_args):
    if tune_args.mlflow:
        import mlflow

    def objective(trial: optuna.Trial) -> float:
        cfg = suggest_config(trial)
        trial.set_user_attr('aug', asdict(cfg))

        args = copy.deepcopy(base_args)
        args.aug = Augmenter(cfg)
        args.dest = base_args.dest / f"trial_{trial.number:03d}"
        args.skip_test = True
        args.test_frac = 0.0
        args.val_frac = 0.2      # same split as hp_tune.py, same patients in every trial
        args.n_folds = 0
        args.fold = None

        def report(step: int, dice: float):
            if tune_args.mlflow:
                mlflow.log_metric('val_dice', dice, step=step)
            trial.report(dice, step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        run = mlflow.start_run(run_name=f"aug_trial_{trial.number:03d}") if tune_args.mlflow else nullcontext()
        with run:
            if tune_args.mlflow:
                mlflow.log_params({'trial': trial.number, 'combo': combo(trial.params),
                                   'lr': args.lr, 'batch_size': args.batch_size,
                                   'n_neighbors': args.n_neighbors, **asdict(cfg)})
            try:
                value = runTraining(args, on_epoch_end=report)
            except torch.cuda.OutOfMemoryError:
                gc.collect()
                torch.cuda.empty_cache()
                raise optuna.TrialPruned()
            finally:
                if not tune_args.keep_trials:
                    rmtree(args.dest, ignore_errors=True)
            if tune_args.mlflow:
                mlflow.log_metric('best_val_dice', value)
            return value

    return objective


def save_results(study: optuna.Study, out) -> None:
    study.trials_dataframe().to_csv(out / 'trials.csv', index=False)
    best = study.best_trial
    with open(out / 'best_aug.json', 'w') as f:
        json.dump(best.user_attrs['aug'], f, indent=2)

    done = [t for t in study.trials if t.state == TrialState.COMPLETE]
    combos = sorted({combo(t.params) for t in done})

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].scatter([combos.index(combo(t.params)) for t in done], [t.value for t in done])
    ax[0].scatter(combos.index(combo(best.params)), best.value, marker='*', s=250, c='red', zorder=3)
    ax[0].set_xticks(range(len(combos)))
    ax[0].set_xticklabels(combos, rotation=30)
    ax[0].set_ylabel('best validation Dice')
    ax[0].set_title('Dice per augmentation combination')

    vals = [t.value for t in done]
    ax[1].plot([t.number for t in done], vals, 'o', label='trial')
    ax[1].plot([t.number for t in done], [max(vals[:i + 1]) for i in range(len(vals))], '-', label='best so far')
    ax[1].set_xlabel('trial number')
    ax[1].set_title('Optimization history')
    ax[1].legend()

    fig.tight_layout()
    fig.savefig(out / 'tuning_results.png', dpi=150)
    print(f"Saved trials.csv, best_aug.json and tuning_results.png in {out}")


if __name__ == '__main__':
    tune_args, base_args = parse()
    base_args.dest.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name=tune_args.study_name or f"aug_{base_args.dataset}",
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=0, n_startup_trials=8),
        # augmented runs converge slower, so give them a longer warm-up before pruning
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=8),
        storage=f"sqlite:///{(base_args.dest / 'optuna.db').as_posix()}",
        load_if_exists=True,
    )

    if len(study.trials) == 0:   # reference trials
        off = {'use_spatial': False, 'use_shift': False, 'use_noise': False}
        study.enqueue_trial(off)                                                    # no augmentation
        study.enqueue_trial({**off, 'use_spatial': True, **SPATIAL})                # spatial only
        study.enqueue_trial({**off, 'use_shift': True, **SHIFT})                    # intensity only
        study.enqueue_trial({**off, 'use_noise': True, **NOISE})                    # noise only
        study.enqueue_trial({'use_spatial': True, 'use_shift': True, 'use_noise': True,
                             **SPATIAL, **SHIFT, **NOISE})                          # your original setup

    study.optimize(make_objective(base_args, tune_args), n_trials=tune_args.n_trials)

    print("\n>>> Best trial")
    for k, v in study.best_trial.user_attrs['aug'].items():
        print(f"    {k} = {v}")
    print(f"    best validation Dice = {study.best_value:.4f}")
    save_results(study, base_args.dest)
