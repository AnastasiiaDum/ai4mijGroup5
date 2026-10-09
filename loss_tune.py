"""Tune the Dice vs CrossEntropy weight ratio with Optuna.

Run code:
python loss_tune.py --dataset SEGTHOR_UNET2D --mode full --dest results/tune_unet2d --gpu --data_dir data/hu_wide_rs18 --epochs 20 --n_trials 30

View optuna results:
pip install optuna-dashboard
optuna-dashboard sqlite:///results/tune_unet2d/optuna.db

"""


import argparse
import copy
from shutil import rmtree

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import optuna
from optuna.trial import TrialState

from main import get_parser, runTraining


def parse():
    tp = argparse.ArgumentParser(add_help=False)
    tp.add_argument('--n_trials', type=int, default=15)
    tp.add_argument('--study_name', default='dice_ce_ratio')
    tp.add_argument('--min_ratio', type=float, default=0.1,
                    help="Lower bound for dice_weight / ce_weight.")
    tp.add_argument('--max_ratio', type=float, default=10.0,
                    help="Upper bound for dice_weight / ce_weight.")
    tp.add_argument('--keep_trials', action='store_true',
                    help="Keep per-trial output folders (predictions, weights). Off by default: they get big.")
    tune_args, rest = tp.parse_known_args()
    base_args = get_parser().parse_args(rest)
    return tune_args, base_args


def make_objective(base_args, tune_args):
    def objective(trial: optuna.Trial) -> float:
        ratio = trial.suggest_float('dice_over_ce', tune_args.min_ratio, tune_args.max_ratio, log=True)

        args = copy.deepcopy(base_args)
        args.ce_weight = 1.0
        args.dice_weight = ratio
        args.dest = base_args.dest / f"trial_{trial.number:03d}"
        args.skip_test = True
        args.test_frac = 0.0
        args.val_frac = 0.2    # 80% train / 20% val
        args.n_folds = 0       # no CV
        args.fold = None  

        def report(step: int, dice: float):
            trial.report(dice, step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        try:
            # mean best validation dice over the folds that were run
            return runTraining(args, on_epoch_end=report)
        finally:
            if not tune_args.keep_trials:
                rmtree(args.dest, ignore_errors=True)

    return objective

def save_results(study: optuna.Study, out) -> None:
    study.trials_dataframe().to_csv(out / 'trials.csv', index=False)

    done = [t for t in study.trials if t.state == TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == TrialState.PRUNED]

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

    # ratio vs Dice
    ax[0].scatter([t.params['dice_over_ce'] for t in done], [t.value for t in done],
                  label='completed')
    if pruned:
        ax[0].scatter([t.params['dice_over_ce'] for t in pruned],
                      [t.intermediate_values[max(t.intermediate_values)] for t in pruned],
                      facecolors='none', edgecolors='gray', label='pruned (last epoch Dice)')
    ax[0].scatter(study.best_params['dice_over_ce'], study.best_value,
                  marker='*', s=250, c='red', label='best', zorder=3)
    ax[0].set_xscale('log')
    ax[0].set_xlabel('dice_weight / ce_weight')
    ax[0].set_ylabel('best validation Dice')
    ax[0].set_title('Dice vs loss ratio')
    ax[0].legend()

    # optimization history
    vals = [t.value for t in done]
    nums = [t.number for t in done]
    best_so_far = [max(vals[:i + 1]) for i in range(len(vals))]
    ax[1].plot(nums, vals, 'o', label='trial')
    ax[1].plot(nums, best_so_far, '-', label='best so far')
    ax[1].set_xlabel('trial number')
    ax[1].set_ylabel('best validation Dice')
    ax[1].set_title('Optimization history')
    ax[1].legend()

    fig.tight_layout()
    fig.savefig(out / 'tuning_results.png', dpi=150)
    print(f"\nSaved {out / 'trials.csv'} and {out / 'tuning_results.png'}")

if __name__ == '__main__':
    tune_args, base_args = parse()
    base_args.dest.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name=tune_args.study_name,
        direction='maximize',
        # n_startup_trials = x number of random values before choosing values
        # based on top performing values from earlier runs
        sampler=optuna.samplers.TPESampler(seed=0, n_startup_trials=6),
        # n_startup_trials = running first x epochs fully
        # n_warmup_steps = let the combination train for x epochs before eliminating it
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5),
        storage=f"sqlite:///{(base_args.dest / 'optuna.db').as_posix()}",
        load_if_exists=True,  
    )

    if len(study.trials) == 0:
        study.enqueue_trial({'dice_over_ce': 1.0}) 

    study.optimize(make_objective(base_args, tune_args), n_trials=tune_args.n_trials)

    print("\n>>> Best trial")
    print(f"    dice_weight / ce_weight = {study.best_params['dice_over_ce']:.3f}")
    print(f"    mean best Dice          = {study.best_value:.4f}")
    print(f"\nFinal run:  python main.py ... --ce_weight 1.0 --dice_weight {study.best_params['dice_over_ce']:.3f}")

    save_results(study, base_args.dest)