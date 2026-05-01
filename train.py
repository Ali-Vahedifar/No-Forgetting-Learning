"""
Main Training Script for NFL, NFL+, and NFL+LoRA Continual Learning Methods

Implements the complete training pipeline for CIL and TIL scenarios.
Matches the experiment setup described in Section 4 of the paper.

Anonymous submission - NeurIPS 2026

Usage:
    python train.py --method nfl+ --dataset cifar100 --num_tasks 10 --scenario cil
    python train.py --method nfl+lora --dataset imagenet-r --num_tasks 10 --scenario cil
    python train.py --method ewc --dataset cifar100 --num_tasks 10 --scenario cil
"""

import os
import argparse
import json
import random
import numpy as np
import torch
from copy import deepcopy
from datetime import datetime
from typing import Dict, Any

from models.nfl import (
    NFLModel, NFLPlusModel, NFLPlusLoRAModel,
    NFLTrainer, NFLPlusTrainer, NFLPlusLoRATrainer
)
from models.backbone import get_backbone
from models.baselines import get_baseline, BASELINE_REGISTRY, EXTERNAL_BASELINES
from data.datasets import get_dataset
from utils.metrics import (
    MetricsTracker, evaluate_model, evaluate_model_til,
    compute_all_metrics
)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='NFL Continual Learning Training'
    )

    # Method and dataset
    all_methods = ['nfl', 'nfl+', 'nfl+lora'] + list(BASELINE_REGISTRY.keys())
    parser.add_argument('--method', type=str, default='nfl+',
                        choices=all_methods,
                        help='CL method to use')
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar100', 'tinyimagenet', 'imagenet1000',
                                 'imagenet-r', 'imagenet-a'],
                        help='Dataset to use')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Root directory for datasets')

    # Task configuration
    parser.add_argument('--num_tasks', type=int, default=10,
                        help='Number of tasks')
    parser.add_argument('--scenario', type=str, default='cil',
                        choices=['cil', 'til'],
                        help='CIL or TIL scenario')

    # Training hyperparameters — paper uses Adam for all methods
    parser.add_argument('--epochs', type=int, default=100,
                        help='Max epochs per step')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (default: auto per dataset)')
    parser.add_argument('--lr_adam', type=float, default=0.001,
                        help='Learning rate for Adam (Auto-Encoder)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (unused when optimizer=adam)')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay')

    # NFL/NFL+ specific hyperparameters
    parser.add_argument('--temperature', type=float, default=2.0,
                        help='KD temperature parameter (p)')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Dual-KD balance weight (NFL step 4, NFL+LoRA step 4)')
    parser.add_argument('--eta', type=float, default=0.5,
                        help='Bias-corrected KD weight (NFL+ step 5)')
    parser.add_argument('--omega', type=float, default=0.5,
                        help='Reconstruction loss weight (NFL+ Auto-Encoder)')

    # NFL+LoRA specific
    parser.add_argument('--lora_rank', type=int, default=8,
                        help='LoRA rank r')
    parser.add_argument('--fisher_lambda', type=float, default=1.0,
                        help='Fisher regularization strength lambda')
    parser.add_argument('--fisher_decay', type=float, default=0.95,
                        help='Fisher accumulation decay gamma')

    # Baseline specific
    parser.add_argument('--ewc_lambda', type=float, default=400.0,
                        help='EWC regularization strength')
    parser.add_argument('--si_lambda', type=float, default=1.0,
                        help='SI regularization strength')
    parser.add_argument('--buffer_size', type=int, default=2000,
                        help='Replay buffer size for memory-based methods')

    # Other settings
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation split ratio (10%% of training data)')
    parser.add_argument('--early_stopping', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--save_dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID to use')

    return parser.parse_args()


def get_default_lr(method: str, dataset: str) -> float:
    """
    Return the best LR from Table 5 of the paper.
    
    Paper hyperparameters (first-task HPO):
        NFL  CIFAR-100: 0.1,  TinyImageNet/ImageNet: 0.03
        NFL+ CIFAR-100: 0.1,  TinyImageNet/ImageNet: 0.03
        NFL+LoRA: 0.001 (Adam with LoRA)
    """
    if method == 'nfl+lora':
        return 0.001
    lr_map = {
        ('nfl', 'cifar100'): 0.1,
        ('nfl+', 'cifar100'): 0.1,
    }
    return lr_map.get((method, dataset), 0.03)


def get_default_epochs(dataset: str) -> int:
    """Paper: 100 for CIFAR/Tiny, 200 for ImageNet-1000."""
    if 'imagenet1000' in dataset:
        return 200
    return 100


def create_nfl_model(args, num_initial_classes, device):
    """Create NFL or NFL+ model with ResNet-18 backbone."""
    backbone = get_backbone(arch='resnet18', dataset=args.dataset)
    feature_dim = backbone.feature_dim

    if args.method == 'nfl':
        model = NFLModel(backbone, feature_dim, num_initial_classes)
    else:  # nfl+
        model = NFLPlusModel(backbone, feature_dim, num_initial_classes)

    return model.to(device)


def create_nfl_lora_model(args, num_initial_classes, device):
    """Create NFL+LoRA model with ViT-B/16 backbone."""
    from models.vit_lora import get_vit_lora_backbone

    backbone = get_vit_lora_backbone(
        model_name='vit_base_patch16_224',
        pretrained=True,
        lora_rank=args.lora_rank,
        fisher_decay=args.fisher_decay
    )
    feature_dim = backbone.feature_dim
    model = NFLPlusLoRAModel(backbone, feature_dim, num_initial_classes)
    return model.to(device)


def create_baseline_model(args, num_total_classes, device):
    """Create a simple model for baselines (backbone + single linear head)."""
    import torch.nn as nn

    backbone = get_backbone(arch='resnet18', dataset=args.dataset)
    feature_dim = backbone.feature_dim

    class SimpleModel(nn.Module):
        def __init__(self, backbone, feature_dim, num_classes):
            super().__init__()
            self.backbone = backbone
            self.head = nn.Linear(feature_dim, num_classes)

        def forward(self, x, task_id=None):
            return self.head(self.backbone(x))

        def add_classes(self, new_classes):
            old = self.head
            new_head = nn.Linear(old.in_features, old.out_features + new_classes)
            with torch.no_grad():
                new_head.weight[:old.out_features] = old.weight
                new_head.bias[:old.out_features] = old.bias
            self.head = new_head

    model = SimpleModel(backbone, feature_dim, num_total_classes)
    return model.to(device)


def create_trainer(args, model, device):
    """Create appropriate trainer for the method."""
    lr = args.lr if args.lr is not None else get_default_lr(args.method, args.dataset)

    if args.method == 'nfl':
        return NFLTrainer(
            model=model, device=device,
            temperature=args.temperature, alpha=args.alpha,
            lr=lr, weight_decay=args.weight_decay, momentum=args.momentum
        )
    elif args.method == 'nfl+':
        return NFLPlusTrainer(
            model=model, device=device,
            temperature=args.temperature,
            eta=args.eta, omega=args.omega,
            lr=lr, lr_adam=args.lr_adam,
            weight_decay=args.weight_decay, momentum=args.momentum
        )
    elif args.method == 'nfl+lora':
        return NFLPlusLoRATrainer(
            model=model, device=device,
            temperature=args.temperature, alpha=args.alpha,
            fisher_lambda=args.fisher_lambda,
            lr=lr, weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"Unknown NFL method: {args.method}")


def run_nfl_experiment(args, dataset, device):
    """Run NFL / NFL+ / NFL+LoRA experiment."""
    metrics_tracker = MetricsTracker(args.num_tasks)
    num_initial_classes = dataset.classes_per_task
    epochs = args.epochs if args.epochs != 100 else get_default_epochs(args.dataset)

    # Create model
    if args.method == 'nfl+lora':
        model = create_nfl_lora_model(args, num_initial_classes, device)
    else:
        model = create_nfl_model(args, num_initial_classes, device)

    trainer = create_trainer(args, model, device)
    print(f"Model: {args.method.upper()}")
    print(f"Feature dim: {model.feature_dim}")
    print(f"Classes per task: {dataset.classes_per_task}")

    for task_id in range(args.num_tasks):
        print(f"\n{'='*60}")
        print(f"Task {task_id + 1}/{args.num_tasks}")
        print(f"{'='*60}")

        train_loader = dataset.get_train_loader(task_id, args.batch_size)
        val_loader = dataset.get_val_loader(task_id, args.batch_size)

        if task_id == 0:
            # --- First task ---
            if args.method == 'nfl+lora':
                trainer.train_first_task(
                    train_loader, val_loader, epochs, args.early_stopping
                )
            elif args.method == 'nfl+':
                trainer.train_step1(
                    train_loader, val_loader, epochs, args.early_stopping
                )
                trainer.train_step2_autoencoder(
                    train_loader, val_loader, epochs // 2, args.early_stopping
                )
            else:  # nfl
                trainer.train_step1(
                    train_loader, val_loader, epochs, args.early_stopping
                )
        else:
            # --- Subsequent tasks ---
            model.add_task(dataset.classes_per_task)

            if args.method == 'nfl+lora':
                trainer.train_new_task(
                    train_loader, val_loader, epochs, args.early_stopping
                )
            elif args.method == 'nfl+':
                soft_targets = trainer.compute_soft_targets(train_loader)
                # Step 3 (FFT)
                trainer.train_step2(
                    train_loader, val_loader, epochs, args.early_stopping
                )
                # Step 4 (TTF)
                trainer.train_step3(
                    train_loader, soft_targets, val_loader, epochs,
                    args.early_stopping
                )
                updated_soft_targets = trainer.compute_updated_logits(train_loader)
                # Bias correction
                trainer.train_bias_correction(
                    val_loader, epochs=50, early_stopping_patience=args.early_stopping
                )
                # Step 5 (TTT + bias)
                trainer.train_step5(
                    train_loader, soft_targets, updated_soft_targets,
                    val_loader, epochs, args.early_stopping
                )
                # AE for next task
                trainer.train_step2_autoencoder(
                    train_loader, val_loader, epochs // 2, args.early_stopping
                )
            else:  # nfl
                trainer.train_new_task(
                    train_loader, val_loader, epochs, args.early_stopping
                )

        # --- Evaluate ---
        print(f"\nEvaluating after Task {task_id + 1}...")
        test_loaders = [dataset.get_test_loader(i, args.batch_size)
                        for i in range(task_id + 1)]

        if args.scenario == 'cil':
            task_offsets = [dataset.get_class_offset(i) for i in range(task_id + 1)]
            accuracies = evaluate_model(model, test_loaders, device, task_offsets)
        else:
            accuracies = evaluate_model_til(model, test_loaders, device)

        metrics_tracker.update(task_id, accuracies)
        current = metrics_tracker.get_metrics_at_task(task_id)
        print(f"Task {task_id + 1} Metrics: {current}")
        for i, acc in enumerate(accuracies):
            print(f"  Task {i + 1} Accuracy: {acc * 100:.2f}%")

    return model, metrics_tracker


def run_baseline_experiment(args, dataset, device):
    """Run a baseline method experiment."""
    metrics_tracker = MetricsTracker(args.num_tasks)
    total_classes = dataset.num_classes
    epochs = args.epochs if args.epochs != 100 else get_default_epochs(args.dataset)

    model = create_baseline_model(args, dataset.classes_per_task, device)

    # Method-specific kwargs
    kwargs = {'lr': args.lr or 0.001}
    if args.method == 'ewc':
        kwargs['ewc_lambda'] = args.ewc_lambda
    elif args.method == 'si':
        kwargs['si_lambda'] = args.si_lambda
    elif args.method == 'lwf':
        kwargs['temperature'] = args.temperature
    elif args.method == 'der++':
        kwargs['buffer_size'] = args.buffer_size

    baseline = get_baseline(args.method, model, device, **kwargs)

    cumulative_classes = 0
    for task_id in range(args.num_tasks):
        print(f"\n{'='*60}")
        print(f"Task {task_id + 1}/{args.num_tasks} [{args.method.upper()}]")
        print(f"{'='*60}")

        train_loader = dataset.get_train_loader(task_id, args.batch_size)
        val_loader = dataset.get_val_loader(task_id, args.batch_size)

        if task_id > 0:
            model.add_classes(dataset.classes_per_task)
            model = model.to(device)

        num_old_classes = cumulative_classes
        cumulative_classes += dataset.classes_per_task

        if args.method == 'lwf':
            baseline.train_task(
                train_loader, val_loader, epochs=epochs,
                early_stopping_patience=args.early_stopping,
                num_old_classes=num_old_classes
            )
            baseline.end_task()
        elif args.method == 'ewc':
            baseline.train_task(
                train_loader, val_loader, epochs=epochs,
                early_stopping_patience=args.early_stopping
            )
            baseline.end_task(train_loader)
        elif args.method == 'si':
            baseline.train_task(
                train_loader, val_loader, epochs=epochs,
                early_stopping_patience=args.early_stopping
            )
            baseline.end_task()
        else:
            baseline.train_task(
                train_loader, val_loader, epochs=epochs,
                early_stopping_patience=args.early_stopping
            )

        # Evaluate
        test_loaders = [dataset.get_test_loader(i, args.batch_size)
                        for i in range(task_id + 1)]
        if args.scenario == 'cil':
            task_offsets = [dataset.get_class_offset(i) for i in range(task_id + 1)]
            accuracies = evaluate_model(model, test_loaders, device, task_offsets)
        else:
            accuracies = evaluate_model_til(model, test_loaders, device)

        metrics_tracker.update(task_id, accuracies)
        current = metrics_tracker.get_metrics_at_task(task_id)
        print(f"Task {task_id + 1} Metrics: {current}")

    return model, metrics_tracker


def main():
    args = get_args()
    set_seed(args.seed)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Save directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(
        args.save_dir,
        f"{args.method}_{args.dataset}_{args.num_tasks}tasks_{args.scenario}_{timestamp}"
    )
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Load dataset
    print(f"\nLoading {args.dataset} with {args.num_tasks} tasks...")
    dataset = get_dataset(
        dataset_name=args.dataset,
        data_root=args.data_root,
        num_tasks=args.num_tasks,
        val_split=args.val_split,
        seed=args.seed
    )

    # Run experiment
    if args.method in ('nfl', 'nfl+', 'nfl+lora'):
        model, metrics_tracker = run_nfl_experiment(args, dataset, device)
    elif args.method in BASELINE_REGISTRY:
        model, metrics_tracker = run_baseline_experiment(args, dataset, device)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    # Print and save
    metrics_tracker.print_summary()
    final = metrics_tracker.get_final_metrics()
    results = {
        'method': args.method,
        'dataset': args.dataset,
        'num_tasks': args.num_tasks,
        'scenario': args.scenario,
        'seed': args.seed,
        'final_metrics': {
            'accuracy': final.accuracy,
            'backward_transfer': final.backward_transfer,
            'plasticity_stability': final.plasticity_stability,
            'plasticity': final.plasticity,
            'stability': final.stability
        },
        'accuracy_matrix': metrics_tracker.get_accuracy_matrix().tolist(),
        'hyperparameters': vars(args)
    }

    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    torch.save(model.state_dict(), os.path.join(save_dir, 'model.pt'))
    print(f"\nResults saved to: {save_dir}")

    return results


if __name__ == '__main__':
    main()
