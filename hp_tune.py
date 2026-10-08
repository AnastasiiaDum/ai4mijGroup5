"""Tune lr, batch size (and n_neighbors for 2.5D models) with Optuna. One study per model.

Have to remove results if it didnt succeed (change to model trained):
Remove-Item -Recurse -Force results\hp_UNET2D

Run UNet
    2D:
    python hp_tune.py --dataset SEGTHOR_UNET2D --mode full --dest results/hp_UNET2D --gpu --data_dir data/SEGTHOR --epochs 20 --n_trials 30 --dice_weight 1.0 --ce_weight 1.0
    2.5D:
    python hp_tune.py --dataset SEGTHOR_UNET25D --mode full --dest results/hp_UNET25D --gpu --data_dir data/SEGTHOR --epochs 20 --n_trials 40 --dice_weight 1.0 --ce_weight 1.0

Run ResUNet
    2D:
    python hp_tune.py --dataset SEGTHOR_RESUNET2D --mode full --dest results/hp_RESUNET2D --gpu --data_dir data/SEGTHOR --epochs 20 --n_trials 30 --dice_weight 1.0 --ce_weight 1.0
    2.5:
    python hp_tune.py --dataset SEGTHOR_RESUNET25D --mode full --dest results/hp_RESUNET25D --gpu --data_dir data/SEGTHOR --epochs 20 --n_trials 40 --dice_weight 1.0 --ce_weight 1.0
    
Running 4 models at once:
foreach ($M in "UNET2D","UNET25D","RESUNET2D","RESUNET25D") {
  python hp_tune.py --dataset "SEGTHOR_$M" --mode full --dest "results/hp_$M" --gpu --data_dir data/SEGTHOR --epochs 20 --n_trials 30 --dice_weight 1.0 --ce_weight 1.0
}
"""
import argparse
import copy
import gc
from shutil import rmtree

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import optuna
import torch
from optuna.trial import TrialState

from main_hyperparameter import get_parser, runTraining, datasets_params


def parse():
    tp = argparse.ArgumentParser(add_help=False)
    tp.add_argument('--n_trials', type=int, default=30)
    tp.add_argument('--study_name', default=None, help="Defaults to hp_<dataset>.")
    tp.add_argument('--lr_min', type=float, default=1e-4)
    tp.add_argument('--lr_max', type=float, default=2e-3)
    tp.add_argument('--batch_sizes', type=int, nargs='+', default=[4, 8, 16])
    tp.add_argument('--neighbors', type=int, nargs='+', default=[1, 2, 3],
                    help="n_neighbors choices (2.5D only): 1/2/3 -> 3/5/7 input slices.")
    tp.add_argument('--keep_trials', action='store_true')
    tune_args, rest = tp.parse_known_args()
    base_args = get_parser().parse_args(rest)
    return tune_args, base_args


def make_objective(base_args, tune_args):
    is_25d = datasets_params[base_args.dataset].get('use_25d', False)

    def objective(trial: optuna.Trial) -> float:
        args = copy.deepcopy(base_args)
        args.lr = trial.suggest_float('lr', tune_args.lr_min, tune_args.lr_max, log=True)
        args.batch_size = trial.suggest_categorical('batch_size', tune_args.batch_sizes)
        if is_25d:
            args.n_neighbors = trial.suggest_categorical('n_neighbors', tune_args.neighbors)

        args.dest = base_args.dest / f"trial_{trial.number:03d}"
        args.skip_test = True
        args.test_frac = 0.0
        args.val_frac = 0.2      # 80% train / 20% val, same patients in every trial
        args.n_folds = 0
        args.fold = None

        def report(step: int, dice: float):
            trial.report(dice, step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        try:
            return runTraining(args, on_epoch_end=report)
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            raise optuna.TrialPruned()
        finally:
            if not tune_args.keep_trials:
                rmtree(args.dest, ignore_errors=True)

    return objective


def save_results(study: optuna.Study, out) -> None:
    study.trials_dataframe().to_csv(out / 'trials.csv', index=False)
    done = [t for t in study.trials if t.state == TrialState.COMPLETE]
    names = list(study.best_params)

    fig, ax = plt.subplots(1, len(names) + 1, figsize=(4.5 * (len(names) + 1), 4))
    for a, name in zip(ax, names):
        a.scatter([t.params[name] for t in done], [t.value for t in done])
        a.scatter(study.best_params[name], study.best_value, marker='*', s=250, c='red', zorder=3)
        if name == 'lr':
            a.set_xscale('log')
        a.set_xlabel(name)
        a.set_ylabel('best validation Dice')
        a.set_title(f'Dice vs {name}')

    vals = [t.value for t in done]
    best_so_far = [max(vals[:i + 1]) for i in range(len(vals))]
    ax[-1].plot([t.number for t in done], vals, 'o', label='trial')
    ax[-1].plot([t.number for t in done], best_so_far, '-', label='best so far')
    ax[-1].set_xlabel('trial number')
    ax[-1].set_title('Optimization history')
    ax[-1].legend()

    fig.tight_layout()
    fig.savefig(out / 'tuning_results.png', dpi=150)
    print(f"Saved {out / 'trials.csv'} and {out / 'tuning_results.png'}")


if __name__ == '__main__':
    tune_args, base_args = parse()
    base_args.dest.mkdir(parents=True, exist_ok=True)
    is_25d = datasets_params[base_args.dataset].get('use_25d', False)

    study = optuna.create_study(
        study_name=tune_args.study_name or f"hp_{base_args.dataset}",
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=0, n_startup_trials=8),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=8),
        storage=f"sqlite:///{(base_args.dest / 'optuna.db').as_posix()}",
        load_if_exists=True,
    )

    if len(study.trials) == 0:   # baseline (current setting)
        baseline = {'lr': 5e-4, 'batch_size': 8}
        if is_25d:
            baseline['n_neighbors'] = 2
        if baseline['batch_size'] in tune_args.batch_sizes and \
                (not is_25d or 2 in tune_args.neighbors):
            study.enqueue_trial(baseline)

    study.optimize(make_objective(base_args, tune_args), n_trials=tune_args.n_trials)

    print("\n>>> Best trial")
    for k, v in study.best_params.items():
        print(f"    {k} = {v}")
    print(f"    best validation Dice = {study.best_value:.4f}")
    save_results(study, base_args.dest)