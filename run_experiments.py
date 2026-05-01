"""
Reproduce All Experiments from the NFL Paper

Runs every configuration reported in the paper tables:
- Tables 1-2: CIL on CIFAR-100, Tiny-ImageNet (10/20 tasks)
- Table (ImageNet combined): CIL/TIL on ImageNet-1000 (10/20/50 tasks)
- Table (ViT): CIL on CIFAR-100, ImageNet-R, ImageNet-A with ViT-B/16
- All baselines: EWC, SI, LwF, DER++, SGD (lower bound)

External baselines (DCNet, NISPA, SpaceNet, PEC, DyTox, MEMO, iCaRL,
CL-LoRA, EWC-LoRA) must be run from their own repositories.

Anonymous submission - NeurIPS 2026

Usage:
    python run_experiments.py --all --n_runs 10
    python run_experiments.py --ours-only --n_runs 10
    python run_experiments.py --dataset cifar100 --num_tasks 10 --method nfl+ --n_runs 10
"""

import os
import sys
import argparse
import json
import subprocess
from datetime import datetime
import numpy as np


# ---------------------------------------------------------------------------
# Best hyperparameters from paper (Table 5, first-task HPO)
# ---------------------------------------------------------------------------
BEST_HP = {
    'nfl': {
        'cifar100':     {'lr': 0.1,  'temperature': 2.0, 'alpha': 0.3},
        'tinyimagenet': {'lr': 0.03, 'temperature': 2.0, 'alpha': 0.3},
        'imagenet1000': {'lr': 0.03, 'temperature': 2.0, 'alpha': 0.5},
    },
    'nfl+': {
        'cifar100':     {'lr': 0.1,  'temperature': 2.0, 'eta': 0.5, 'omega': 0.5, 'lr_adam': 0.001},
        'tinyimagenet': {'lr': 0.03, 'temperature': 2.0, 'eta': 0.5, 'omega': 0.5, 'lr_adam': 0.001},
        'imagenet1000': {'lr': 0.03, 'temperature': 2.0, 'eta': 0.5, 'omega': 0.5, 'lr_adam': 0.001},
    },
    'nfl+lora': {
        'cifar100':    {'lr': 0.001, 'temperature': 2.0, 'alpha': 0.5, 'fisher_lambda': 1.0, 'lora_rank': 8},
        'imagenet-r':  {'lr': 0.001, 'temperature': 2.0, 'alpha': 0.5, 'fisher_lambda': 1.0, 'lora_rank': 8},
        'imagenet-a':  {'lr': 0.001, 'temperature': 2.0, 'alpha': 0.5, 'fisher_lambda': 1.0, 'lora_rank': 8},
    },
}

# ---------------------------------------------------------------------------
# Experiment configurations — every row in every table
# ---------------------------------------------------------------------------
RESNET_CONFIGS = [
    # CIFAR-100 (10/20 tasks) — Tables 1-2
    {'dataset': 'cifar100',     'num_tasks': 10, 'scenario': 'cil', 'epochs': 100},
    {'dataset': 'cifar100',     'num_tasks': 20, 'scenario': 'cil', 'epochs': 100},
    {'dataset': 'cifar100',     'num_tasks': 10, 'scenario': 'til', 'epochs': 100},
    {'dataset': 'cifar100',     'num_tasks': 20, 'scenario': 'til', 'epochs': 100},
    # Tiny-ImageNet (10/20 tasks)
    {'dataset': 'tinyimagenet', 'num_tasks': 10, 'scenario': 'cil', 'epochs': 100},
    {'dataset': 'tinyimagenet', 'num_tasks': 20, 'scenario': 'cil', 'epochs': 100},
    {'dataset': 'tinyimagenet', 'num_tasks': 10, 'scenario': 'til', 'epochs': 100},
    {'dataset': 'tinyimagenet', 'num_tasks': 20, 'scenario': 'til', 'epochs': 100},
    # ImageNet-1000 (10/20/50 tasks)
    {'dataset': 'imagenet1000', 'num_tasks': 10, 'scenario': 'cil', 'epochs': 200},
    {'dataset': 'imagenet1000', 'num_tasks': 20, 'scenario': 'cil', 'epochs': 200},
    {'dataset': 'imagenet1000', 'num_tasks': 50, 'scenario': 'cil', 'epochs': 200},
    {'dataset': 'imagenet1000', 'num_tasks': 10, 'scenario': 'til', 'epochs': 200},
    {'dataset': 'imagenet1000', 'num_tasks': 20, 'scenario': 'til', 'epochs': 200},
    {'dataset': 'imagenet1000', 'num_tasks': 50, 'scenario': 'til', 'epochs': 200},
]

VIT_CONFIGS = [
    # ViT-B/16 CIL experiments (Table: LoRA-based)
    {'dataset': 'cifar100',    'num_tasks': 10, 'scenario': 'cil', 'epochs': 50},
    {'dataset': 'imagenet-r',  'num_tasks': 10, 'scenario': 'cil', 'epochs': 50},
    {'dataset': 'imagenet-r',  'num_tasks': 20, 'scenario': 'cil', 'epochs': 50},
    {'dataset': 'imagenet-a',  'num_tasks': 10, 'scenario': 'cil', 'epochs': 50},
    {'dataset': 'imagenet-a',  'num_tasks': 20, 'scenario': 'cil', 'epochs': 50},
]

OUR_METHODS = ['nfl', 'nfl+']
BASELINE_METHODS = ['ewc', 'si', 'lwf', 'der++', 'sgd']
VIT_METHOD = 'nfl+lora'


def get_args():
    parser = argparse.ArgumentParser(description='Run NFL experiments')
    parser.add_argument('--all', action='store_true',
                        help='Run ALL experiments (ours + baselines)')
    parser.add_argument('--ours-only', action='store_true',
                        help='Run only NFL/NFL+/NFL+LoRA')
    parser.add_argument('--baselines-only', action='store_true',
                        help='Run only baseline methods')
    parser.add_argument('--vit-only', action='store_true',
                        help='Run only ViT experiments')
    parser.add_argument('--method', type=str, default=None,
                        help='Single method to run')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Single dataset')
    parser.add_argument('--num_tasks', type=int, default=None)
    parser.add_argument('--scenario', type=str, default=None,
                        choices=['cil', 'til'])
    parser.add_argument('--n_runs', type=int, default=10,
                        help='Number of runs per config (10 in paper)')
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--save_dir', type=str, default='./experiments')
    parser.add_argument('--gpu', type=int, default=0)
    return parser.parse_args()


def run_single(method, config, seed, data_root, save_dir, gpu):
    """Launch a single train.py run as a subprocess."""
    hp = BEST_HP.get(method, {}).get(config['dataset'], {})

    cmd = [
        sys.executable, 'train.py',
        '--method', method,
        '--dataset', config['dataset'],
        '--num_tasks', str(config['num_tasks']),
        '--scenario', config['scenario'],
        '--epochs', str(config['epochs']),
        '--seed', str(seed),
        '--data_root', data_root,
        '--save_dir', save_dir,
        '--gpu', str(gpu),
        '--batch_size', '64',
        '--val_split', '0.1',
        '--early_stopping', '10',
    ]

    for k, v in hp.items():
        cmd.extend([f'--{k}', str(v)])

    exp_name = f"{method}_{config['dataset']}_{config['num_tasks']}t_{config['scenario']}_s{seed}"
    print(f"\n>>> {exp_name}")
    print(f"    cmd: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    FAILED: {result.stderr[-500:]}")
    return result


def aggregate(results_dir):
    """Collect results.json files and aggregate across seeds."""
    all_results = []
    for root, _, files in os.walk(results_dir):
        for f in files:
            if f == 'results.json':
                with open(os.path.join(root, f)) as fp:
                    all_results.append(json.load(fp))
    if not all_results:
        return None

    accs = [r['final_metrics']['accuracy'] for r in all_results]
    bwts = [r['final_metrics']['backward_transfer'] for r in all_results]
    pss  = [r['final_metrics']['plasticity_stability'] for r in all_results]

    return {
        'n_runs': len(all_results),
        'accuracy':  {'mean': float(np.mean(accs)), 'std': float(np.std(accs))},
        'bwt':       {'mean': float(np.mean(bwts)), 'std': float(np.std(bwts))},
        'ps':        {'mean': float(np.mean(pss)),  'std': float(np.std(pss))},
    }


def print_table(results_dict):
    """Pretty-print aggregated results."""
    header = f"{'Method':<12} {'Dataset':<14} {'Tasks':<6} {'Scen':<5} {'ACC (%)':<18} {'BWT':<14} {'PS':<14}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for key in sorted(results_dict.keys()):
        r = results_dict[key]
        parts = key.split('|')
        method, ds, tasks, scen = parts
        acc = f"{r['accuracy']['mean']*100:.2f} ± {r['accuracy']['std']*100:.2f}"
        bwt = f"{r['bwt']['mean']:.2f} ± {r['bwt']['std']:.2f}"
        ps  = f"{r['ps']['mean']:.2f} ± {r['ps']['std']:.2f}"
        print(f"{method:<12} {ds:<14} {tasks:<6} {scen:<5} {acc:<18} {bwt:<14} {ps:<14}")
    print("=" * len(header))


def main():
    args = get_args()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = os.path.join(args.save_dir, f'run_{timestamp}')
    os.makedirs(base_dir, exist_ok=True)

    # Determine what to run
    jobs = []  # list of (method, config)

    if args.method and args.dataset and args.num_tasks and args.scenario:
        # Single experiment
        config = {
            'dataset': args.dataset,
            'num_tasks': args.num_tasks,
            'scenario': args.scenario,
            'epochs': 200 if 'imagenet1000' in args.dataset else 100,
        }
        jobs.append((args.method, config))

    elif args.ours_only:
        for m in OUR_METHODS:
            for c in RESNET_CONFIGS:
                jobs.append((m, c))
        for c in VIT_CONFIGS:
            jobs.append((VIT_METHOD, c))

    elif args.baselines_only:
        for m in BASELINE_METHODS:
            for c in RESNET_CONFIGS:
                jobs.append((m, c))

    elif args.vit_only:
        for c in VIT_CONFIGS:
            jobs.append((VIT_METHOD, c))

    elif args.all:
        for m in OUR_METHODS:
            for c in RESNET_CONFIGS:
                jobs.append((m, c))
        for m in BASELINE_METHODS:
            for c in RESNET_CONFIGS:
                jobs.append((m, c))
        for c in VIT_CONFIGS:
            jobs.append((VIT_METHOD, c))
    else:
        print("Specify --all, --ours-only, --baselines-only, --vit-only,")
        print("or provide --method, --dataset, --num_tasks, --scenario.")
        return

    # Run
    all_aggregated = {}
    for method, config in jobs:
        key = f"{method}|{config['dataset']}|{config['num_tasks']}|{config['scenario']}"
        exp_dir = os.path.join(base_dir, key.replace('|', '_'))
        os.makedirs(exp_dir, exist_ok=True)

        for run_idx in range(args.n_runs):
            seed = 42 + run_idx
            run_dir = os.path.join(exp_dir, f'run_{run_idx}')
            run_single(method, config, seed, args.data_root, run_dir, args.gpu)

        agg = aggregate(exp_dir)
        if agg:
            all_aggregated[key] = agg
            with open(os.path.join(exp_dir, 'aggregated.json'), 'w') as f:
                json.dump(agg, f, indent=2)

    # Summary
    print_table(all_aggregated)

    with open(os.path.join(base_dir, 'summary.json'), 'w') as f:
        json.dump(all_aggregated, f, indent=2)
    print(f"\nAll results saved to: {base_dir}")


if __name__ == '__main__':
    main()
