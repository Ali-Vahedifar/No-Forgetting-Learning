

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import datasets, transforms
from typing import List, Tuple, Optional, Dict
from PIL import Image


class ContinualLearningDataset:
    """
    Base class for continual learning dataset management.
    
    Handles splitting data into tasks and creating data loaders.
    """
    
    def __init__(
        self,
        dataset_name: str,
        data_root: str,
        num_tasks: int,
        train_transform: transforms.Compose,
        test_transform: transforms.Compose,
        val_split: float = 0.1,
        seed: int = 42
    ):
        """
        Args:
            dataset_name: Name of the dataset
            data_root: Root directory for data
            num_tasks: Number of tasks to split into
            train_transform: Transforms for training data
            test_transform: Transforms for test data
            val_split: Fraction of training data for validation
            seed: Random seed for reproducibility
        """
        self.dataset_name = dataset_name
        self.data_root = data_root
        self.num_tasks = num_tasks
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.val_split = val_split
        self.seed = seed
        
        # Will be populated by subclasses
        self.num_classes = None
        self.classes_per_task = None
        self.class_order = None
        
        # Data storage
        self.train_datasets = []
        self.val_datasets = []
        self.test_datasets = []
        
        # Class offset for CIL (cumulative classes before each task)
        self.class_offsets = []
        
    def get_task_classes(self, task_id: int) -> List[int]:
        """Get class indices for a specific task."""
        start_idx = task_id * self.classes_per_task
        end_idx = start_idx + self.classes_per_task
        return self.class_order[start_idx:end_idx]
    
    def get_class_offset(self, task_id: int) -> int:
        """Get the class offset for a task (for CIL)."""
        return self.class_offsets[task_id]
    
    def get_train_loader(self, task_id: int, batch_size: int = 64) -> DataLoader:
        """Get training data loader for a task."""
        return DataLoader(
            self.train_datasets[task_id],
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
    
    def get_val_loader(self, task_id: int, batch_size: int = 64) -> DataLoader:
        """Get validation data loader for a task."""
        return DataLoader(
            self.val_datasets[task_id],
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    
    def get_test_loader(self, task_id: int, batch_size: int = 64) -> DataLoader:
        """Get test data loader for a task."""
        return DataLoader(
            self.test_datasets[task_id],
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    
    def get_all_test_loaders(self, batch_size: int = 64) -> List[DataLoader]:
        """Get test loaders for all tasks seen so far."""
        return [self.get_test_loader(i, batch_size) for i in range(len(self.test_datasets))]


class TaskSubset(Dataset):
    """
    Subset of a dataset for a specific task with label remapping.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        indices: List[int],
        class_mapping: Dict[int, int],
        transform: Optional[transforms.Compose] = None
    ):
        """
        Args:
            dataset: Full dataset
            indices: Indices of samples for this task
            class_mapping: Mapping from original labels to task-local labels
            transform: Optional transform override
        """
        self.dataset = dataset
        self.indices = indices
        self.class_mapping = class_mapping
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        real_idx = self.indices[idx]
        img, label = self.dataset[real_idx]
        
        if self.transform is not None:
            img = self.transform(img)
        
        # Remap label to task-local index
        local_label = self.class_mapping[label]
        
        return img, local_label


class CIFAR100CL(ContinualLearningDataset):
    """
    CIFAR-100 dataset for continual learning.
    
    100 classes, typically split into 10 or 20 tasks.
    Image size: 32x32
    """
    
    def __init__(
        self,
        data_root: str = './data',
        num_tasks: int = 10,
        val_split: float = 0.1,
        seed: int = 42
    ):
        # Standard CIFAR transforms
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761]
            )
        ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761]
            )
        ])
        
        super().__init__(
            'cifar100', data_root, num_tasks,
            train_transform, test_transform, val_split, seed
        )
        
        self.num_classes = 100
        self.classes_per_task = self.num_classes // num_tasks
        
        # Shuffle class order for randomized task creation
        np.random.seed(seed)
        self.class_order = np.random.permutation(self.num_classes).tolist()
        
        # Load datasets
        self._setup_datasets()
        
    def _setup_datasets(self):
        """Create task-specific datasets."""
        # Load full CIFAR-100
        full_train = datasets.CIFAR100(
            self.data_root, train=True, download=True,
            transform=self.train_transform
        )
        full_test = datasets.CIFAR100(
            self.data_root, train=False, download=True,
            transform=self.test_transform
        )
        
        # Get labels
        train_labels = np.array(full_train.targets)
        test_labels = np.array(full_test.targets)
        
        cumulative_classes = 0
        
        for task_id in range(self.num_tasks):
            task_classes = self.get_task_classes(task_id)
            self.class_offsets.append(cumulative_classes)
            
            # Create class mapping (original -> task-local)
            class_mapping = {c: i for i, c in enumerate(task_classes)}
            
            # Find indices for this task
            train_mask = np.isin(train_labels, task_classes)
            test_mask = np.isin(test_labels, task_classes)
            
            train_indices = np.where(train_mask)[0].tolist()
            test_indices = np.where(test_mask)[0].tolist()
            
            # Split training into train/val
            np.random.seed(self.seed + task_id)
            np.random.shuffle(train_indices)
            
            val_size = int(len(train_indices) * self.val_split)
            val_indices = train_indices[:val_size]
            train_indices = train_indices[val_size:]
            
            # Create subsets
            train_dataset = TaskSubset(
                full_train, train_indices, class_mapping, self.train_transform
            )
            val_dataset = TaskSubset(
                full_train, val_indices, class_mapping, self.test_transform
            )
            test_dataset = TaskSubset(
                full_test, test_indices, class_mapping, self.test_transform
            )
            
            self.train_datasets.append(train_dataset)
            self.val_datasets.append(val_dataset)
            self.test_datasets.append(test_dataset)
            
            cumulative_classes += len(task_classes)


class TinyImageNetCL(ContinualLearningDataset):
    """
    Tiny-ImageNet dataset for continual learning.
    
    200 classes, typically split into 10 or 20 tasks.
    Image size: 64x64
    """
    
    def __init__(
        self,
        data_root: str = './data/tiny-imagenet-200',
        num_tasks: int = 10,
        val_split: float = 0.1,
        seed: int = 42
    ):
        train_transform = transforms.Compose([
            transforms.RandomCrop(64, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        super().__init__(
            'tinyimagenet', data_root, num_tasks,
            train_transform, test_transform, val_split, seed
        )
        
        self.num_classes = 200
        self.classes_per_task = self.num_classes // num_tasks
        
        np.random.seed(seed)
        self.class_order = np.random.permutation(self.num_classes).tolist()
        
        self._setup_datasets()
        
    def _setup_datasets(self):
        """Create task-specific datasets."""
        # Load Tiny-ImageNet using ImageFolder
        train_dir = os.path.join(self.data_root, 'train')
        val_dir = os.path.join(self.data_root, 'val')
        
        if not os.path.exists(train_dir):
            raise RuntimeError(
                f"Tiny-ImageNet not found at {self.data_root}. "
                "Please download and extract it first."
            )
        
        full_train = datasets.ImageFolder(train_dir, transform=self.train_transform)
        full_val = datasets.ImageFolder(val_dir, transform=self.test_transform)
        
        train_labels = np.array([s[1] for s in full_train.samples])
        test_labels = np.array([s[1] for s in full_val.samples])
        
        cumulative_classes = 0
        
        for task_id in range(self.num_tasks):
            task_classes = self.get_task_classes(task_id)
            self.class_offsets.append(cumulative_classes)
            
            class_mapping = {c: i for i, c in enumerate(task_classes)}
            
            train_mask = np.isin(train_labels, task_classes)
            test_mask = np.isin(test_labels, task_classes)
            
            train_indices = np.where(train_mask)[0].tolist()
            test_indices = np.where(test_mask)[0].tolist()
            
            # Split training into train/val
            np.random.seed(self.seed + task_id)
            np.random.shuffle(train_indices)
            
            val_size = int(len(train_indices) * self.val_split)
            val_indices = train_indices[:val_size]
            train_indices = train_indices[val_size:]
            
            train_dataset = TaskSubset(
                full_train, train_indices, class_mapping, self.train_transform
            )
            val_dataset = TaskSubset(
                full_train, val_indices, class_mapping, self.test_transform
            )
            test_dataset = TaskSubset(
                full_val, test_indices, class_mapping, self.test_transform
            )
            
            self.train_datasets.append(train_dataset)
            self.val_datasets.append(val_dataset)
            self.test_datasets.append(test_dataset)
            
            cumulative_classes += len(task_classes)


class ImageNet1000CL(ContinualLearningDataset):
    """
    ImageNet-1000 dataset for continual learning.
    
    1000 classes, typically split into 10 or 20 tasks.
    Image size: 224x224
    """
    
    def __init__(
        self,
        data_root: str = './data/imagenet',
        num_tasks: int = 10,
        val_split: float = 0.1,
        seed: int = 42
    ):
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        super().__init__(
            'imagenet1000', data_root, num_tasks,
            train_transform, test_transform, val_split, seed
        )
        
        self.num_classes = 1000
        self.classes_per_task = self.num_classes // num_tasks
        
        np.random.seed(seed)
        self.class_order = np.random.permutation(self.num_classes).tolist()
        
        self._setup_datasets()
        
    def _setup_datasets(self):
        """Create task-specific datasets."""
        train_dir = os.path.join(self.data_root, 'train')
        val_dir = os.path.join(self.data_root, 'val')
        
        if not os.path.exists(train_dir):
            raise RuntimeError(
                f"ImageNet not found at {self.data_root}. "
                "Please download and extract it first."
            )
        
        full_train = datasets.ImageFolder(train_dir, transform=self.train_transform)
        full_val = datasets.ImageFolder(val_dir, transform=self.test_transform)
        
        train_labels = np.array([s[1] for s in full_train.samples])
        test_labels = np.array([s[1] for s in full_val.samples])
        
        cumulative_classes = 0
        
        for task_id in range(self.num_tasks):
            task_classes = self.get_task_classes(task_id)
            self.class_offsets.append(cumulative_classes)
            
            class_mapping = {c: i for i, c in enumerate(task_classes)}
            
            train_mask = np.isin(train_labels, task_classes)
            test_mask = np.isin(test_labels, task_classes)
            
            train_indices = np.where(train_mask)[0].tolist()
            test_indices = np.where(test_mask)[0].tolist()
            
            np.random.seed(self.seed + task_id)
            np.random.shuffle(train_indices)
            
            val_size = int(len(train_indices) * self.val_split)
            val_indices = train_indices[:val_size]
            train_indices = train_indices[val_size:]
            
            train_dataset = TaskSubset(
                full_train, train_indices, class_mapping, self.train_transform
            )
            val_dataset = TaskSubset(
                full_train, val_indices, class_mapping, self.test_transform
            )
            test_dataset = TaskSubset(
                full_val, test_indices, class_mapping, self.test_transform
            )
            
            self.train_datasets.append(train_dataset)
            self.val_datasets.append(val_dataset)
            self.test_datasets.append(test_dataset)
            
            cumulative_classes += len(task_classes)


class ImageNetRCL(ContinualLearningDataset):
    """
    ImageNet-R (Hendrycks et al., 2021) for ViT-B/16 experiments.
    
    200 classes of renditions (art, cartoons, deviantart, etc.) of ImageNet classes.
    Used for evaluating robustness to distribution shift.
    """
    
    def __init__(
        self,
        data_root: str = './data/imagenet-r',
        num_tasks: int = 10,
        val_split: float = 0.1,
        seed: int = 42
    ):
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        super().__init__(
            'imagenet_r', data_root, num_tasks,
            train_transform, test_transform, val_split, seed
        )
        self.num_classes = 200
        self.classes_per_task = self.num_classes // num_tasks
        np.random.seed(seed)
        self.class_order = np.random.permutation(self.num_classes).tolist()
        self._setup_datasets()
    
    def _setup_datasets(self):
        if not os.path.exists(self.data_root):
            raise RuntimeError(
                f"ImageNet-R not found at {self.data_root}. "
                "Download from: https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar"
            )
        full_dataset = datasets.ImageFolder(self.data_root, transform=self.train_transform)
        all_labels = np.array([s[1] for s in full_dataset.samples])
        
        cumulative_classes = 0
        for task_id in range(self.num_tasks):
            task_classes = self.get_task_classes(task_id)
            self.class_offsets.append(cumulative_classes)
            class_mapping = {c: i for i, c in enumerate(task_classes)}
            mask = np.isin(all_labels, task_classes)
            indices = np.where(mask)[0].tolist()
            
            np.random.seed(self.seed + task_id)
            np.random.shuffle(indices)
            
            # 70/20/10 split matching paper
            n = len(indices)
            n_test = int(n * 0.2)
            n_val = int(n * 0.1)
            test_indices = indices[:n_test]
            val_indices = indices[n_test:n_test + n_val]
            train_indices = indices[n_test + n_val:]
            
            self.train_datasets.append(
                TaskSubset(full_dataset, train_indices, class_mapping, self.train_transform)
            )
            self.val_datasets.append(
                TaskSubset(full_dataset, val_indices, class_mapping, self.test_transform)
            )
            self.test_datasets.append(
                TaskSubset(full_dataset, test_indices, class_mapping, self.test_transform)
            )
            cumulative_classes += len(task_classes)


class ImageNetACL(ContinualLearningDataset):
    """
    ImageNet-A (Hendrycks et al., 2021) — natural adversarial examples.
    
    200 classes of naturally occurring adversarial examples for ImageNet.
    """
    
    def __init__(
        self,
        data_root: str = './data/imagenet-a',
        num_tasks: int = 10,
        val_split: float = 0.1,
        seed: int = 42
    ):
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        super().__init__(
            'imagenet_a', data_root, num_tasks,
            train_transform, test_transform, val_split, seed
        )
        self.num_classes = 200
        self.classes_per_task = self.num_classes // num_tasks
        np.random.seed(seed)
        self.class_order = np.random.permutation(self.num_classes).tolist()
        self._setup_datasets()
    
    def _setup_datasets(self):
        if not os.path.exists(self.data_root):
            raise RuntimeError(
                f"ImageNet-A not found at {self.data_root}. "
                "Download from: https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar"
            )
        full_dataset = datasets.ImageFolder(self.data_root, transform=self.train_transform)
        all_labels = np.array([s[1] for s in full_dataset.samples])
        
        cumulative_classes = 0
        for task_id in range(self.num_tasks):
            task_classes = self.get_task_classes(task_id)
            self.class_offsets.append(cumulative_classes)
            class_mapping = {c: i for i, c in enumerate(task_classes)}
            mask = np.isin(all_labels, task_classes)
            indices = np.where(mask)[0].tolist()
            
            np.random.seed(self.seed + task_id)
            np.random.shuffle(indices)
            
            n = len(indices)
            n_test = int(n * 0.2)
            n_val = int(n * 0.1)
            test_indices = indices[:n_test]
            val_indices = indices[n_test:n_test + n_val]
            train_indices = indices[n_test + n_val:]
            
            self.train_datasets.append(
                TaskSubset(full_dataset, train_indices, class_mapping, self.train_transform)
            )
            self.val_datasets.append(
                TaskSubset(full_dataset, val_indices, class_mapping, self.test_transform)
            )
            self.test_datasets.append(
                TaskSubset(full_dataset, test_indices, class_mapping, self.test_transform)
            )
            cumulative_classes += len(task_classes)


def get_dataset(
    dataset_name: str,
    data_root: str = './data',
    num_tasks: int = 10,
    val_split: float = 0.1,
    seed: int = 42
) -> ContinualLearningDataset:
    """
    Get continual learning dataset by name.
    
    Args:
        dataset_name: Name of dataset
        data_root: Root directory for data
        num_tasks: Number of tasks to split into
        val_split: Fraction of training data for validation
        seed: Random seed
        
    Returns:
        ContinualLearningDataset instance
    """
    dataset_map = {
        'cifar100': CIFAR100CL,
        'cifar-100': CIFAR100CL,
        'tinyimagenet': TinyImageNetCL,
        'tiny-imagenet': TinyImageNetCL,
        'tiny_imagenet': TinyImageNetCL,
        'imagenet1000': ImageNet1000CL,
        'imagenet-1000': ImageNet1000CL,
        'imagenet': ImageNet1000CL,
        'imagenet-r': ImageNetRCL,
        'imagenet_r': ImageNetRCL,
        'imagenetr': ImageNetRCL,
        'imagenet-a': ImageNetACL,
        'imagenet_a': ImageNetACL,
        'imageneta': ImageNetACL,
    }
    
    dataset_name_lower = dataset_name.lower()
    if dataset_name_lower not in dataset_map:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(set(dataset_map.values()))}"
        )
    
    return dataset_map[dataset_name_lower](
        data_root=data_root,
        num_tasks=num_tasks,
        val_split=val_split,
        seed=seed
    )
