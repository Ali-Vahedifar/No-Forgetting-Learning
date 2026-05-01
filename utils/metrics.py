"""
Evaluation Metrics for Continual Learning

Implements ACC, BWT, and the proposed PS (Plasticity-Stability) metric.

Anonymous submission - ICML 2026
"""

import numpy as np
import torch
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class CLMetrics:
    """Container for continual learning metrics."""
    accuracy: float
    backward_transfer: float
    plasticity_stability: float
    plasticity: float
    stability: float
    
    def __str__(self) -> str:
        return (
            f"ACC: {self.accuracy:.4f} | "
            f"BWT: {self.backward_transfer:.4f} | "
            f"PS: {self.plasticity_stability:.4f} | "
            f"P: {self.plasticity:.4f} | "
            f"S: {self.stability:.4f}"
        )


class AccuracyMatrix:
    """
    Manages the accuracy matrix A for continual learning evaluation.
    
    A[i,j] = accuracy on task j after training on task i
    """
    
    def __init__(self, num_tasks: int):
        """
        Args:
            num_tasks: Total number of tasks
        """
        self.num_tasks = num_tasks
        self.matrix = np.zeros((num_tasks, num_tasks))
        
    def update(self, current_task: int, task_accuracies: List[float]):
        """
        Update accuracy matrix after training on a task.
        
        Args:
            current_task: Index of the task just trained (0-indexed)
            task_accuracies: List of accuracies on tasks 0 to current_task
        """
        if len(task_accuracies) != current_task + 1:
            raise ValueError(
                f"Expected {current_task + 1} accuracies, got {len(task_accuracies)}"
            )
        for j, acc in enumerate(task_accuracies):
            self.matrix[current_task, j] = acc
            
    def get_accuracy(self, train_task: int, eval_task: int) -> float:
        """Get A[train_task, eval_task]."""
        return self.matrix[train_task, eval_task]
    
    def to_numpy(self) -> np.ndarray:
        """Return the accuracy matrix as numpy array."""
        return self.matrix.copy()


def compute_average_accuracy(acc_matrix: np.ndarray) -> float:
    """
    Compute Average Accuracy (ACC) after training on all T tasks.
    
    ACC_T = (1/T) * sum_{k=1}^{T} A_{T,k}
    
    Args:
        acc_matrix: Accuracy matrix A of shape (T, T)
        
    Returns:
        Average accuracy
    """
    T = acc_matrix.shape[0]
    return np.mean(acc_matrix[T-1, :T])


def compute_backward_transfer(acc_matrix: np.ndarray) -> float:
    """
    Compute Backward Transfer (BWT).
    
    BWT_T = (1/(T-1)) * sum_{k=1}^{T-1} (A_{T,k} - A_{k,k})
    
    Measures the influence of learning task T on previous tasks.
    Negative BWT indicates forgetting.
    
    Args:
        acc_matrix: Accuracy matrix A of shape (T, T)
        
    Returns:
        Backward transfer (typically negative for forgetting)
    """
    T = acc_matrix.shape[0]
    if T <= 1:
        return 0.0
    
    bwt_sum = 0.0
    for k in range(T - 1):
        bwt_sum += acc_matrix[T-1, k] - acc_matrix[k, k]
    
    return bwt_sum / (T - 1)


def compute_plasticity(acc_matrix: np.ndarray) -> float:
    """
    Compute Plasticity (P).
    
    P = (1/(T-1)) * sum_{k=2}^{T} (A_{k,k} - A_{k-1,k}) / (1 - A_{k-1,k})
    
    Measures learning efficiency - how much of the "unlearned" knowledge gap
    was closed during training.
    
    Args:
        acc_matrix: Accuracy matrix A of shape (T, T)
        
    Returns:
        Plasticity score
    """
    T = acc_matrix.shape[0]
    if T <= 1:
        return 1.0
    
    plasticity_sum = 0.0
    for k in range(1, T):
        # A_{k-1,k} is the accuracy on task k before training on it
        # A_{k,k} is the accuracy on task k after training on it
        prev_acc = acc_matrix[k-1, k]
        curr_acc = acc_matrix[k, k]
        
        # Avoid division by zero
        if prev_acc < 1.0:
            plasticity_sum += (curr_acc - prev_acc) / (1.0 - prev_acc)
        else:
            # Perfect previous accuracy means no room to improve
            plasticity_sum += 1.0 if curr_acc >= prev_acc else 0.0
    
    return plasticity_sum / (T - 1)


def compute_stability(acc_matrix: np.ndarray) -> float:
    """
    Compute Stability (S).
    
    S = 1 - (1/(T-1)) * sum_{k=1}^{T-1} (A_{k,k} - A_{T,k})
    
    Measures retention rate of previously learned tasks.
    Assumes A_{k,k} >= A_{T,k} (non-negative forgetting).
    
    Args:
        acc_matrix: Accuracy matrix A of shape (T, T)
        
    Returns:
        Stability score
    """
    T = acc_matrix.shape[0]
    if T <= 1:
        return 1.0
    
    forgetting_sum = 0.0
    for k in range(T - 1):
        forgetting = acc_matrix[k, k] - acc_matrix[T-1, k]
        # If backward transfer increases accuracy, forgetting is 0
        forgetting_sum += max(0.0, forgetting)
    
    return 1.0 - forgetting_sum / (T - 1)


def compute_plasticity_stability(acc_matrix: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute the Plasticity-Stability (PS) ratio.
    
    PS_T = (2 * P * S) / (P + S)
    
    This is the harmonic mean of plasticity and stability.
    
    Args:
        acc_matrix: Accuracy matrix A of shape (T, T)
        
    Returns:
        Tuple of (PS ratio, Plasticity, Stability)
    """
    P = compute_plasticity(acc_matrix)
    S = compute_stability(acc_matrix)
    
    # Harmonic mean
    if P + S > 0:
        PS = (2 * P * S) / (P + S)
    else:
        PS = 0.0
    
    return PS, P, S


def compute_all_metrics(acc_matrix: np.ndarray) -> CLMetrics:
    """
    Compute all continual learning metrics.
    
    Args:
        acc_matrix: Accuracy matrix A of shape (T, T)
        
    Returns:
        CLMetrics object containing all metrics
    """
    acc = compute_average_accuracy(acc_matrix)
    bwt = compute_backward_transfer(acc_matrix)
    ps, p, s = compute_plasticity_stability(acc_matrix)
    
    return CLMetrics(
        accuracy=acc,
        backward_transfer=bwt,
        plasticity_stability=ps,
        plasticity=p,
        stability=s
    )


def evaluate_model(
    model: torch.nn.Module,
    test_loaders: List[torch.utils.data.DataLoader],
    device: torch.device,
    task_offset: Optional[List[int]] = None
) -> List[float]:
    """
    Evaluate model on all tasks seen so far.
    
    Args:
        model: The model to evaluate
        test_loaders: List of test data loaders for each task
        device: Evaluation device
        task_offset: Optional class offset for each task (for CIL)
        
    Returns:
        List of accuracies for each task
    """
    model.eval()
    accuracies = []
    
    with torch.no_grad():
        for task_id, loader in enumerate(test_loaders):
            correct = 0
            total = 0
            
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                # Add class offset for CIL scenario
                if task_offset is not None and task_id < len(task_offset):
                    batch_y = batch_y + task_offset[task_id]
                
                outputs = model(batch_x)
                _, predicted = outputs.max(1)
                
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()
            
            accuracies.append(correct / total if total > 0 else 0.0)
    
    return accuracies


def evaluate_model_til(
    model: torch.nn.Module,
    test_loaders: List[torch.utils.data.DataLoader],
    device: torch.device
) -> List[float]:
    """
    Evaluate model in Task Incremental Learning (TIL) setting.
    
    In TIL, task identity is known during inference.
    
    Args:
        model: The model to evaluate
        test_loaders: List of test data loaders for each task
        device: Evaluation device
        
    Returns:
        List of accuracies for each task
    """
    model.eval()
    accuracies = []
    
    with torch.no_grad():
        for task_id, loader in enumerate(test_loaders):
            correct = 0
            total = 0
            
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                # Use task-specific head
                outputs = model(batch_x, task_id=task_id)
                _, predicted = outputs.max(1)
                
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()
            
            accuracies.append(correct / total if total > 0 else 0.0)
    
    return accuracies


class MetricsTracker:
    """
    Track and compute metrics throughout continual learning.
    """
    
    def __init__(self, num_tasks: int):
        """
        Args:
            num_tasks: Total number of tasks
        """
        self.num_tasks = num_tasks
        self.acc_matrix = AccuracyMatrix(num_tasks)
        self.metrics_history = []
        
    def update(self, current_task: int, task_accuracies: List[float]):
        """
        Update tracker after training on a task.
        
        Args:
            current_task: Index of task just trained (0-indexed)
            task_accuracies: Accuracies on tasks 0 to current_task
        """
        self.acc_matrix.update(current_task, task_accuracies)
        
        # Compute metrics on partial matrix
        partial_matrix = self.acc_matrix.matrix[:current_task+1, :current_task+1]
        metrics = compute_all_metrics(partial_matrix)
        self.metrics_history.append(metrics)
        
    def get_final_metrics(self) -> CLMetrics:
        """Get final metrics after all tasks."""
        if len(self.metrics_history) == 0:
            raise RuntimeError("No metrics recorded yet")
        return self.metrics_history[-1]
    
    def get_metrics_at_task(self, task: int) -> CLMetrics:
        """Get metrics after training on a specific task."""
        if task >= len(self.metrics_history):
            raise IndexError(f"Task {task} not yet recorded")
        return self.metrics_history[task]
    
    def get_accuracy_matrix(self) -> np.ndarray:
        """Get the full accuracy matrix."""
        return self.acc_matrix.to_numpy()
    
    def print_summary(self):
        """Print a summary of metrics."""
        print("\n" + "="*60)
        print("Continual Learning Metrics Summary")
        print("="*60)
        
        final = self.get_final_metrics()
        print(f"\nFinal Metrics: {final}")
        
        print("\nMetrics progression across tasks:")
        for i, m in enumerate(self.metrics_history):
            print(f"  Task {i+1}: ACC={m.accuracy:.4f}, BWT={m.backward_transfer:.4f}, PS={m.plasticity_stability:.4f}")
        
        print("\nAccuracy Matrix:")
        matrix = self.get_accuracy_matrix()
        print("     " + "  ".join([f"T{j+1:2d}" for j in range(matrix.shape[1])]))
        for i in range(matrix.shape[0]):
            row_str = f"T{i+1:2d}: " + "  ".join([f"{matrix[i,j]:.2f}" for j in range(i+1)])
            print(row_str)
        print("="*60)


def compute_forgetting_per_task(acc_matrix: np.ndarray) -> List[float]:
    """
    Compute forgetting for each task.
    
    Forgetting_k = A_{k,k} - A_{T,k}
    
    Args:
        acc_matrix: Accuracy matrix of shape (T, T)
        
    Returns:
        List of forgetting values for tasks 0 to T-2
    """
    T = acc_matrix.shape[0]
    forgetting = []
    
    for k in range(T - 1):
        f = acc_matrix[k, k] - acc_matrix[T-1, k]
        forgetting.append(max(0.0, f))
    
    return forgetting


def compute_forward_transfer(
    acc_matrix: np.ndarray, 
    random_baseline: Optional[List[float]] = None
) -> float:
    """
    Compute Forward Transfer (FWT).
    
    FWT_T = (1/(T-1)) * sum_{k=2}^{T} (A_{k-1,k} - b_k)
    
    Measures how well the model generalizes to future tasks before seeing them.
    
    Args:
        acc_matrix: Accuracy matrix of shape (T, T)
        random_baseline: Random baseline accuracy for each task (default: 0)
        
    Returns:
        Forward transfer score
    """
    T = acc_matrix.shape[0]
    if T <= 1:
        return 0.0
    
    if random_baseline is None:
        random_baseline = [0.0] * T
    
    fwt_sum = 0.0
    for k in range(1, T):
        fwt_sum += acc_matrix[k-1, k] - random_baseline[k]
    
    return fwt_sum / (T - 1)
