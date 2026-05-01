
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from copy import deepcopy


class KnowledgeDistillationLoss(nn.Module):
    """
    Knowledge Distillation loss function as defined in Equations 7-9 of the paper.
    
    L_KD(H_t, O_t) = -sum_i h_i * log(o_i)
    
    where h_i and o_i are temperature-scaled softmax outputs.
    """
    
    def __init__(self, temperature: float = 2.0):
        """
        Args:
            temperature: Temperature parameter p for smoothing probability distribution.
                        Values typically > 1 (default: 2.0)
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(self, soft_targets: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """
        Compute KD loss between soft targets and model outputs.
        
        Args:
            soft_targets: Logits from the teacher model (H_t)
            outputs: Logits from the student model (O_t)
            
        Returns:
            KD loss value
        """
        # Apply temperature scaling (Equations 8-9)
        h = F.softmax(soft_targets / self.temperature, dim=1)
        o = F.log_softmax(outputs / self.temperature, dim=1)
        
        # KD loss (Equation 7)
        loss = -torch.sum(h * o, dim=1).mean()
        
        # Scale by temperature^2 as per Hinton et al.
        return loss * (self.temperature ** 2)


class UnderCompleteAutoEncoder(nn.Module):
    """
    Under-complete Auto-Encoder for feature preservation in NFL+.
    
    The Auto-Encoder captures the most relevant features required for previous tasks.
    When a new task is introduced, it ensures critical features are preserved by
    enforcing a reconstruction loss.
    
    Architecture: Feature_dim -> bottleneck_dim -> Feature_dim
    
    CORRECTED: Bias correction now properly projects to num_old_classes dimension
    for element-wise multiplication with H_t.
    """
    
    def __init__(self, feature_dim: int, bottleneck_dim: int, num_old_classes: int):
        """
        Args:
            feature_dim: Dimension of input features (from backbone)
            bottleneck_dim: Dimension of bottleneck layer (smaller than feature_dim)
            num_old_classes: Number of classes in old tasks (for bias correction)
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.bottleneck_dim = bottleneck_dim
        self.num_old_classes = num_old_classes
        
        # Auto-Encoder layers
        self.encoder = nn.Linear(feature_dim, bottleneck_dim)
        self.decoder = nn.Linear(bottleneck_dim, feature_dim)
        self.activation = nn.ReLU()
        
        # Bias correction parameters (Equation 15)
        # These project from bottleneck_dim to num_old_classes for element-wise
        # multiplication with H_t (which has shape [batch, num_old_classes])
        self.bias_projection = nn.Linear(bottleneck_dim, num_old_classes)
        self.w_bias = nn.Parameter(torch.ones(num_old_classes))
        self.b_bias = nn.Parameter(torch.zeros(num_old_classes))
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode features to bottleneck representation: σ(W_Enc * P)"""
        return self.activation(self.encoder(x))
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from bottleneck to feature space: W_Dec * z"""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct features through bottleneck.
        R(P) = W_Dec * σ(W_Enc * P)
        """
        z = self.encode(x)
        return self.decode(z)
    
    def compute_bias_correction(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute bias correction transformation Γ (Equation 15).
        
        Γ(P_{t+1}) = w_bias * proj(σ(W_Enc * P_{t+1})) + b_bias
        
        CORRECTED: Projects to num_old_classes dimension for element-wise
        multiplication with H_t.
        
        Args:
            features: Features from backbone for new task data [batch, feature_dim]
            
        Returns:
            Bias correction factors [batch, num_old_classes]
        """
        encoded = self.encode(features)  # [batch, bottleneck_dim]
        projected = self.bias_projection(encoded)  # [batch, num_old_classes]
        return self.w_bias * projected + self.b_bias  # [batch, num_old_classes]
    
    def update_num_old_classes(self, new_num_old_classes: int):
        """Update bias correction layers for new number of old classes."""
        device = self.w_bias.device
        
        # Create new layers
        new_projection = nn.Linear(self.bottleneck_dim, new_num_old_classes).to(device)
        new_w_bias = nn.Parameter(torch.ones(new_num_old_classes, device=device))
        new_b_bias = nn.Parameter(torch.zeros(new_num_old_classes, device=device))
        
        # Copy old weights if possible
        old_classes = min(self.num_old_classes, new_num_old_classes)
        with torch.no_grad():
            new_projection.weight[:old_classes] = self.bias_projection.weight[:old_classes]
            new_projection.bias[:old_classes] = self.bias_projection.bias[:old_classes]
            new_w_bias[:old_classes] = self.w_bias[:old_classes]
            new_b_bias[:old_classes] = self.b_bias[:old_classes]
        
        self.bias_projection = new_projection
        self.w_bias = new_w_bias
        self.b_bias = new_b_bias
        self.num_old_classes = new_num_old_classes


class MultiHeadClassifier(nn.Module):
    """
    Multi-headed classifier supporting both CIL and TIL scenarios.
    
    For CIL: Single unified output layer over all classes
    For TIL: Separate task-specific heads (can use task identity during inference)
    """
    
    def __init__(self, feature_dim: int, initial_classes: int):
        """
        Args:
            feature_dim: Input feature dimension from backbone
            initial_classes: Number of classes in the first task
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.heads = nn.ModuleList([nn.Linear(feature_dim, initial_classes)])
        self.num_classes_per_task = [initial_classes]
        
    def add_task_head(self, num_classes: int):
        """Add a new task head for incremental learning."""
        new_head = nn.Linear(self.feature_dim, num_classes)
        self.heads.append(new_head)
        self.num_classes_per_task.append(num_classes)
        
    def forward(self, features: torch.Tensor, task_id: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass through classifier.
        
        Args:
            features: Features from backbone [batch_size, feature_dim]
            task_id: If provided (TIL), use only that task's head.
                    If None (CIL), concatenate outputs from all heads.
                    
        Returns:
            Logits [batch_size, num_classes]
        """
        if task_id is not None:
            # TIL: Use specific task head
            return self.heads[task_id](features)
        else:
            # CIL: Concatenate all heads
            outputs = [head(features) for head in self.heads]
            return torch.cat(outputs, dim=1)
    
    def get_old_task_output(self, features: torch.Tensor, num_old_tasks: int) -> torch.Tensor:
        """Get outputs for old task classes only."""
        outputs = [self.heads[i](features) for i in range(num_old_tasks)]
        return torch.cat(outputs, dim=1)
    
    def get_new_task_output(self, features: torch.Tensor) -> torch.Tensor:
        """Get output for the newest task only."""
        return self.heads[-1](features)
    
    @property
    def total_classes(self) -> int:
        """Total number of classes across all tasks."""
        return sum(self.num_classes_per_task)
    
    @property
    def num_tasks(self) -> int:
        """Number of tasks learned so far."""
        return len(self.heads)
    
    @property
    def num_old_classes(self) -> int:
        """Number of classes in all tasks except the newest."""
        if len(self.num_classes_per_task) <= 1:
            return self.num_classes_per_task[0] if self.num_classes_per_task else 0
        return sum(self.num_classes_per_task[:-1])


class NFLModel(nn.Module):
    """
    No Forgetting Learning (NFL) model.
    
    Implements the 4-step training process:
    1. Initial Task Training
    2. New Task Introduction (FFT - freeze backbone & old head, train new head)
    3. Teacher & Old Head Alignment (TTF - freeze new head, train backbone & old head)
    4. Final Consolidation (TTT - train all with dual logit targets)
    """
    
    def __init__(self, backbone: nn.Module, feature_dim: int, initial_classes: int):
        """
        Args:
            backbone: Feature extraction network (e.g., ResNet-18)
            feature_dim: Output dimension of backbone
            initial_classes: Number of classes in the first task
        """
        super().__init__()
        self.backbone = backbone
        self.classifier = MultiHeadClassifier(feature_dim, initial_classes)
        self.feature_dim = feature_dim
        
        # Store soft targets for knowledge distillation
        self.stored_logits = None  # H_t from Equation 2
        self.updated_logits = None  # H'_t from Equation 6
        
    def forward(self, x: torch.Tensor, task_id: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images [batch_size, C, H, W]
            task_id: Task identifier for TIL (None for CIL)
            
        Returns:
            Logits [batch_size, num_classes]
        """
        features = self.backbone(x)
        return self.classifier(features, task_id)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from backbone."""
        return self.backbone(x)
    
    def add_task(self, num_classes: int):
        """Add a new task head."""
        self.classifier.add_task_head(num_classes)


class NFLPlusModel(NFLModel):
    """
    NFL+ model with Auto-Encoder for feature preservation.
    
    Extends NFL with:
    - Under-complete Auto-Encoder for preserving informative features
    - Bias correction mechanism for addressing class imbalance
    
    Implements the 5-step training process:
    1. Initial Task Training
    2. Auto-Encoder Training (on OLD task data X_t)
    3. New Task Introduction
    4. Teacher & Old Head Alignment  
    5. Final Consolidation with Bias Correction
    """
    
    def __init__(
        self, 
        backbone: nn.Module, 
        feature_dim: int, 
        initial_classes: int,
        bottleneck_dim: Optional[int] = None
    ):
        """
        Args:
            backbone: Feature extraction network
            feature_dim: Output dimension of backbone
            initial_classes: Number of classes in first task
            bottleneck_dim: Auto-encoder bottleneck dimension (default: feature_dim // 4)
        """
        super().__init__(backbone, feature_dim, initial_classes)
        
        if bottleneck_dim is None:
            bottleneck_dim = feature_dim // 4
        
        self.bottleneck_dim = bottleneck_dim
        self.autoencoder = UnderCompleteAutoEncoder(
            feature_dim, bottleneck_dim, initial_classes
        )
        
        # Store frozen backbone for feature space constraint (set before Step 5)
        self.frozen_backbone = None
        
    def set_frozen_backbone(self):
        """Store a copy of the backbone for feature space constraint in Step 5."""
        self.frozen_backbone = deepcopy(self.backbone)
        for param in self.frozen_backbone.parameters():
            param.requires_grad = False
            
    def get_frozen_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get features from frozen backbone (θ*_s in Equation 17)."""
        if self.frozen_backbone is None:
            raise RuntimeError("Frozen backbone not set. Call set_frozen_backbone() first.")
        return self.frozen_backbone(x)
    
    def compute_bias_corrected_logits(
        self, 
        features: torch.Tensor, 
        stored_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute bias-corrected logits (Equation 16).
        
        H̃_t = Γ(P_{t+1}) ⊙ H_t
        
        CORRECTED: Now both tensors have matching dimensions for element-wise multiplication.
        
        Args:
            features: Features from backbone [batch, feature_dim]
            stored_logits: Original stored logits H_t [batch, num_old_classes]
            
        Returns:
            Bias-corrected logits [batch, num_old_classes]
        """
        gamma = self.autoencoder.compute_bias_correction(features)  # [batch, num_old_classes]
        return gamma * stored_logits  # Element-wise multiplication
    
    def add_task(self, num_classes: int):
        """Add a new task head and update Auto-Encoder for new class count."""
        super().add_task(num_classes)
        # Update Auto-Encoder bias correction for the new total of old classes
        self.autoencoder.update_num_old_classes(self.classifier.num_old_classes)


class NFLTrainer:
    """
    Trainer for NFL method implementing the 4-step training process.
    
    CORRECTED: Properly stores and uses original θ_t for H'_t computation.
    """
    
    def __init__(
        self,
        model: NFLModel,
        device: torch.device,
        temperature: float = 2.0,
        alpha: float = 0.5,
        lr: float = 0.1,
        weight_decay: float = 0.0,
        momentum: float = 0.9
    ):
        """
        Args:
            model: NFL model instance
            device: Training device
            temperature: KD temperature parameter (p in paper)
            alpha: Weight for original stability vs recent stability (Equation 6)
            lr: Learning rate for SGD
            weight_decay: Weight decay for SGD
            momentum: SGD momentum
        """
        self.model = model
        self.device = device
        self.temperature = temperature
        self.alpha = alpha
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        
        self.kd_loss = KnowledgeDistillationLoss(temperature)
        self.ce_loss = nn.CrossEntropyLoss()
        
        # Track current task
        self.current_task = 0
        
        # CORRECTED: Store original old task head state for H'_t computation
        self.original_old_head_states = []
        
    def _get_optimizer(self, parameters) -> torch.optim.Optimizer:
        """Create SGD optimizer."""
        return torch.optim.SGD(
            parameters,
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )
    
    def _freeze_parameters(self, module: nn.Module):
        """Freeze all parameters in a module."""
        for param in module.parameters():
            param.requires_grad = False
            
    def _unfreeze_parameters(self, module: nn.Module):
        """Unfreeze all parameters in a module."""
        for param in module.parameters():
            param.requires_grad = True
    
    def train_step1(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int,
        early_stopping_patience: int = 10
    ) -> Dict[str, List[float]]:
        """
        Step 1: Initial Task Training
        
        Train NN^1 with L^1 = E[L_CE(Y_t, O^1_t)] (Equation 1)
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum training epochs
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Training history
        """
        self._unfreeze_parameters(self.model)
        optimizer = self._get_optimizer(self.model.parameters())
        
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.ce_loss(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            
            # Validation
            val_loss, val_acc = self._validate(val_loader)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = deepcopy(self.model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    self.model.load_state_dict(best_state)
                    break
        
        # CORRECTED: Store original old task head states after Step 1
        self._store_original_old_heads()
                    
        return history
    
    def _store_original_old_heads(self):
        """Store the state of all current heads as 'original' for H'_t computation."""
        self.original_old_head_states = []
        for head in self.model.classifier.heads:
            self.original_old_head_states.append(deepcopy(head.state_dict()))
    
    def compute_soft_targets(
        self, 
        data_loader: torch.utils.data.DataLoader
    ) -> torch.Tensor:
        """
        Compute soft targets H_t for new task data (Equation 2).
        
        NN^1(X_{t+1}, θ*_s, θ*_t) → H_t
        
        Args:
            data_loader: Data loader for new task data
            
        Returns:
            Soft targets tensor
        """
        self.model.eval()
        all_logits = []
        
        with torch.no_grad():
            for batch_x, _ in data_loader:
                batch_x = batch_x.to(self.device)
                # Get outputs only for old task classes
                features = self.model.get_features(batch_x)
                old_logits = self.model.classifier.get_old_task_output(
                    features, 
                    self.model.classifier.num_tasks - 1
                )
                all_logits.append(old_logits.cpu())
                
        return torch.cat(all_logits, dim=0)
    
    def train_step2(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int,
        early_stopping_patience: int = 10
    ) -> Dict[str, List[float]]:
        """
        Step 2: New Task Introduction (FFT configuration)
        
        Freeze θ*_s and θ*_t, train only θ_{t+1}
        L^2 = E[L_CE(Y_{t+1}, O^2_{t+1})] (Equation 3)
        
        Args:
            train_loader: Training data loader for new task
            val_loader: Validation data loader
            epochs: Maximum training epochs
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Training history
        """
        # Freeze backbone and old task heads (FFT)
        self._freeze_parameters(self.model.backbone)
        for i in range(self.model.classifier.num_tasks - 1):
            self._freeze_parameters(self.model.classifier.heads[i])
        
        # Train only new task head
        self._unfreeze_parameters(self.model.classifier.heads[-1])
        
        optimizer = self._get_optimizer(self.model.classifier.heads[-1].parameters())
        
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                # Get only new task output
                features = self.model.get_features(batch_x)
                outputs = self.model.classifier.get_new_task_output(features)
                
                loss = self.ce_loss(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            
            val_loss, val_acc = self._validate_new_task(val_loader)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = deepcopy(self.model.classifier.heads[-1].state_dict())
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Step 2 early stopping at epoch {epoch}")
                    self.model.classifier.heads[-1].load_state_dict(best_state)
                    break
                    
        return history
    
    def train_step3(
        self,
        train_loader: torch.utils.data.DataLoader,
        soft_targets: torch.Tensor,
        val_loader: torch.utils.data.DataLoader,
        epochs: int,
        early_stopping_patience: int = 10
    ) -> Dict[str, List[float]]:
        """
        Step 3: Teacher & Old Head Alignment (TTF configuration)
        
        Freeze θ*_{t+1}, train θ_s and θ_t
        L^3 = E[L_KD(H_t, O^3_t) + L_CE(Y_{t+1}, O^3_{t+1})] (Equation 4)
        
        Args:
            train_loader: Training data loader
            soft_targets: Pre-computed soft targets H_t
            val_loader: Validation data loader
            epochs: Maximum training epochs
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Training history
        """
        # Freeze new task head (TTF)
        self._freeze_parameters(self.model.classifier.heads[-1])
        
        # Unfreeze backbone and old task heads
        self._unfreeze_parameters(self.model.backbone)
        for i in range(self.model.classifier.num_tasks - 1):
            self._unfreeze_parameters(self.model.classifier.heads[i])
        
        # Collect trainable parameters
        trainable_params = list(self.model.backbone.parameters())
        for i in range(self.model.classifier.num_tasks - 1):
            trainable_params.extend(self.model.classifier.heads[i].parameters())
        
        optimizer = self._get_optimizer(trainable_params)
        
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            
            for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Get corresponding soft targets
                start_idx = batch_idx * train_loader.batch_size
                end_idx = start_idx + batch_x.size(0)
                batch_soft_targets = soft_targets[start_idx:end_idx].to(self.device)
                
                optimizer.zero_grad()
                
                features = self.model.get_features(batch_x)
                
                # Old task output for KD loss
                old_outputs = self.model.classifier.get_old_task_output(
                    features, 
                    self.model.classifier.num_tasks - 1
                )
                
                # New task output for CE loss
                new_outputs = self.model.classifier.get_new_task_output(features)
                
                # Combined loss (Equation 4)
                kd_loss = self.kd_loss(batch_soft_targets, old_outputs)
                ce_loss = self.ce_loss(new_outputs, batch_y)
                loss = kd_loss + ce_loss
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            
            val_loss, val_acc = self._validate(val_loader)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = deepcopy(self.model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Step 3 early stopping at epoch {epoch}")
                    self.model.load_state_dict(best_state)
                    break
                    
        return history
    
    def compute_updated_logits(
        self,
        data_loader: torch.utils.data.DataLoader
    ) -> torch.Tensor:
        """
        Compute updated logits H'_t (Equation 6).
        
        NN^4(X_{t+1}, θ^u_s, θ_t) → H'_t
        
        CORRECTED: Uses updated backbone θ^u_s but ORIGINAL old task head θ_t from Step 1.
        
        Args:
            data_loader: Data loader for new task data
            
        Returns:
            Updated soft targets
        """
        self.model.eval()
        
        # Save current head states
        current_head_states = []
        for i in range(self.model.classifier.num_tasks - 1):
            current_head_states.append(
                deepcopy(self.model.classifier.heads[i].state_dict())
            )
        
        # CORRECTED: Temporarily restore original old task heads from Step 1
        for i in range(len(self.original_old_head_states)):
            if i < self.model.classifier.num_tasks - 1:
                self.model.classifier.heads[i].load_state_dict(
                    self.original_old_head_states[i]
                )
        
        all_logits = []
        
        with torch.no_grad():
            for batch_x, _ in data_loader:
                batch_x = batch_x.to(self.device)
                features = self.model.get_features(batch_x)  # Uses updated backbone θ^u_s
                old_logits = self.model.classifier.get_old_task_output(
                    features,
                    self.model.classifier.num_tasks - 1
                )  # Uses original θ_t
                all_logits.append(old_logits.cpu())
        
        # Restore current head states
        for i, state in enumerate(current_head_states):
            self.model.classifier.heads[i].load_state_dict(state)
                
        return torch.cat(all_logits, dim=0)
    
    def train_step4(
        self,
        train_loader: torch.utils.data.DataLoader,
        soft_targets_original: torch.Tensor,
        soft_targets_updated: torch.Tensor,
        val_loader: torch.utils.data.DataLoader,
        epochs: int,
        early_stopping_patience: int = 10
    ) -> Dict[str, List[float]]:
        """
        Step 4: Final Consolidation (TTT configuration)
        
        Train all parameters with dual logit targets.
        L^4 = E[α*L_KD(H_t, O^4_t) + (1-α)*L_KD(H'_t, O'^4_t) + L_CE(Y_{t+1}, O^4_{t+1})]
        (Equation 6)
        
        Args:
            train_loader: Training data loader
            soft_targets_original: Original soft targets H_t
            soft_targets_updated: Updated soft targets H'_t
            val_loader: Validation data loader
            epochs: Maximum training epochs
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Training history
        """
        # Unfreeze all parameters (TTT)
        self._unfreeze_parameters(self.model)
        
        optimizer = self._get_optimizer(self.model.parameters())
        
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            
            for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Get soft targets
                start_idx = batch_idx * train_loader.batch_size
                end_idx = start_idx + batch_x.size(0)
                batch_soft_original = soft_targets_original[start_idx:end_idx].to(self.device)
                batch_soft_updated = soft_targets_updated[start_idx:end_idx].to(self.device)
                
                optimizer.zero_grad()
                
                features = self.model.get_features(batch_x)
                
                # Old task outputs
                old_outputs = self.model.classifier.get_old_task_output(
                    features,
                    self.model.classifier.num_tasks - 1
                )
                
                # New task output
                new_outputs = self.model.classifier.get_new_task_output(features)
                
                # Combined loss with dual logit targets (Equation 6)
                kd_loss_original = self.kd_loss(batch_soft_original, old_outputs)
                kd_loss_updated = self.kd_loss(batch_soft_updated, old_outputs)
                ce_loss = self.ce_loss(new_outputs, batch_y)
                
                loss = (self.alpha * kd_loss_original + 
                       (1 - self.alpha) * kd_loss_updated + 
                       ce_loss)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            
            val_loss, val_acc = self._validate(val_loader)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = deepcopy(self.model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Step 4 early stopping at epoch {epoch}")
                    self.model.load_state_dict(best_state)
                    break
        
        # Update stored original heads for next task
        self._store_original_old_heads()
                    
        return history
    
    def _validate(self, val_loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """Validate on all tasks seen so far."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                loss = self.ce_loss(outputs, batch_y)
                total_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()
                
        return total_loss / len(val_loader), correct / total
    
    def _validate_new_task(self, val_loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """Validate only on the new task."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                features = self.model.get_features(batch_x)
                outputs = self.model.classifier.get_new_task_output(features)
                loss = self.ce_loss(outputs, batch_y)
                total_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()
                
        return total_loss / len(val_loader), correct / total
    
    def train_new_task(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int = 100,
        early_stopping_patience: int = 10
    ) -> Dict[str, Dict[str, List[float]]]:
        """
        Complete training pipeline for a new task (Steps 2-4).
        
        Call this after adding a new task head to the model.
        
        Args:
            train_loader: Training data loader for new task
            val_loader: Validation data loader
            epochs: Maximum epochs per step
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Training history for all steps
        """
        history = {}
        
        # Compute soft targets H_t using current model
        print("Computing soft targets H_t...")
        soft_targets = self.compute_soft_targets(train_loader)
        
        # Step 2: Train new task head (FFT)
        print("Step 2: Training new task head (FFT)...")
        history['step2'] = self.train_step2(
            train_loader, val_loader, epochs, early_stopping_patience
        )
        
        # Step 3: Train backbone and old heads (TTF)
        print("Step 3: Training backbone and old heads (TTF)...")
        history['step3'] = self.train_step3(
            train_loader, soft_targets, val_loader, epochs, early_stopping_patience
        )
        
        # Compute updated soft targets H'_t
        print("Computing updated soft targets H'_t...")
        updated_soft_targets = self.compute_updated_logits(train_loader)
        
        # Step 4: Final consolidation (TTT)
        print("Step 4: Final consolidation (TTT)...")
        history['step4'] = self.train_step4(
            train_loader, soft_targets, updated_soft_targets,
            val_loader, epochs, early_stopping_patience
        )
        
        self.current_task += 1
        
        return history


class NFLPlusTrainer(NFLTrainer):
    """
    Trainer for NFL+ method implementing the 5-step training process.
    
    CORRECTED: 
    - Auto-Encoder is trained on OLD task data (X_t, Y_t) in Step 2
    - Bias correction parameters trained separately before Step 5
    - Proper feature space constraint implementation
    """
    
    def __init__(
        self,
        model: NFLPlusModel,
        device: torch.device,
        temperature: float = 2.0,
        eta: float = 0.5,
        omega: float = 0.5,
        lr: float = 0.1,
        lr_adam: float = 0.001,
        weight_decay: float = 0.0,
        momentum: float = 0.9
    ):
        """
        Args:
            model: NFL+ model instance
            device: Training device
            temperature: KD temperature parameter (p in paper)
            eta: Weight for bias-corrected stability vs recent stability (Equation 17)
            omega: Weight for reconstruction loss in Auto-Encoder (Ω in paper)
            lr: Learning rate for SGD
            lr_adam: Learning rate for Adam (Auto-Encoder training)
            weight_decay: Weight decay for SGD
            momentum: SGD momentum
        """
        # Note: NFL+ uses eta (not alpha) for its final step (Step 5).
        # Pass alpha=0.5 as default since NFL+ does not call train_step4.
        super().__init__(model, device, temperature, alpha=0.5, lr=lr,
                         weight_decay=weight_decay, momentum=momentum)
        self.eta = eta
        self.omega = omega
        self.lr_adam = lr_adam
        
    def train_step2_autoencoder(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int,
        early_stopping_patience: int = 10
    ) -> Dict[str, List[float]]:
        """
        Step 2: Auto-Encoder Training (on OLD task data X_t)
        
        CORRECTED: This is trained on the OLD task data, not new task data.
        
        argmin_R E_{(X_t, Y_t)}[Ω||R(P) - P||² + L_CE(θ_t(R(P)), Y_t)]
        where P = θ_s(X_t)
        
        Args:
            train_loader: Training data loader for CURRENT/OLD task (X_t, Y_t)
            val_loader: Validation data loader for current task
            epochs: Maximum training epochs
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Training history
        """
        # Freeze backbone and classifier, train only Auto-Encoder
        self._freeze_parameters(self.model.backbone)
        self._freeze_parameters(self.model.classifier)
        
        # Auto-Encoder uses Adam optimizer
        optimizer = torch.optim.Adam(
            self.model.autoencoder.parameters(),
            lr=self.lr_adam,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        for epoch in range(epochs):
            self.model.autoencoder.train()
            train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                # Get features P = θ_s(X_t)
                with torch.no_grad():
                    features = self.model.get_features(batch_x)
                
                # Reconstruct: R(P) = W_Dec * σ(W_Enc * P)
                reconstructed = self.model.autoencoder(features)
                
                # Reconstruction loss: ||R(P) - P||²
                recon_loss = F.mse_loss(reconstructed, features)
                
                # Classification loss: L_CE(θ_t(R(P)), Y_t)
                # Pass reconstructed features through old task classifier
                old_task_output = self.model.classifier.get_old_task_output(
                    reconstructed,
                    self.model.classifier.num_tasks  # All current tasks
                )
                ce_loss = self.ce_loss(old_task_output, batch_y)
                
                # Combined loss (Equation 12)
                loss = self.omega * recon_loss + ce_loss
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            
            # Validation
            val_loss = self._validate_autoencoder(val_loader)
            history['val_loss'].append(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = deepcopy(self.model.autoencoder.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Auto-Encoder early stopping at epoch {epoch}")
                    self.model.autoencoder.load_state_dict(best_state)
                    break
        
        # Unfreeze for subsequent steps
        self._unfreeze_parameters(self.model.backbone)
        self._unfreeze_parameters(self.model.classifier)
                    
        return history
    
    def train_bias_correction(
        self,
        val_loader: torch.utils.data.DataLoader,
        epochs: int = 50,
        early_stopping_patience: int = 10
    ) -> Dict[str, List[float]]:
        """
        Train bias correction parameters (before Step 5).
        
        Minimization: E[||Γ(X_{t+1}) - 1||²]
        
        This promotes unbiased learning by keeping Γ close to identity.
        
        Args:
            val_loader: Validation data loader (cross-validation set)
            epochs: Maximum training epochs
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Training history
        """
        # Freeze everything except bias correction parameters
        self._freeze_parameters(self.model)
        
        # Only train bias correction parameters
        self.model.autoencoder.w_bias.requires_grad = True
        self.model.autoencoder.b_bias.requires_grad = True
        self.model.autoencoder.bias_projection.weight.requires_grad = True
        self.model.autoencoder.bias_projection.bias.requires_grad = True
        
        bias_params = [
            self.model.autoencoder.w_bias,
            self.model.autoencoder.b_bias,
            self.model.autoencoder.bias_projection.weight,
            self.model.autoencoder.bias_projection.bias
        ]
        
        optimizer = torch.optim.Adam(bias_params, lr=self.lr_adam)
        
        history = {'train_loss': []}
        best_loss = float('inf')
        patience_counter = 0
        best_state = {
            'w_bias': self.model.autoencoder.w_bias.data.clone(),
            'b_bias': self.model.autoencoder.b_bias.data.clone(),
            'proj_weight': self.model.autoencoder.bias_projection.weight.data.clone(),
            'proj_bias': self.model.autoencoder.bias_projection.bias.data.clone()
        }
        
        for epoch in range(epochs):
            total_loss = 0.0
            
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(self.device)
                
                optimizer.zero_grad()
                
                # Get features
                with torch.no_grad():
                    features = self.model.get_features(batch_x)
                
                # Compute Γ(X_{t+1})
                gamma = self.model.autoencoder.compute_bias_correction(features)
                
                # Loss: ||Γ(X_{t+1}) - 1||²
                ones = torch.ones_like(gamma)
                loss = F.mse_loss(gamma, ones)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            total_loss /= len(val_loader)
            history['train_loss'].append(total_loss)
            
            if total_loss < best_loss:
                best_loss = total_loss
                patience_counter = 0
                best_state = {
                    'w_bias': self.model.autoencoder.w_bias.data.clone(),
                    'b_bias': self.model.autoencoder.b_bias.data.clone(),
                    'proj_weight': self.model.autoencoder.bias_projection.weight.data.clone(),
                    'proj_bias': self.model.autoencoder.bias_projection.bias.data.clone()
                }
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Bias correction early stopping at epoch {epoch}")
                    self.model.autoencoder.w_bias.data = best_state['w_bias']
                    self.model.autoencoder.b_bias.data = best_state['b_bias']
                    self.model.autoencoder.bias_projection.weight.data = best_state['proj_weight']
                    self.model.autoencoder.bias_projection.bias.data = best_state['proj_bias']
                    break
        
        # Unfreeze model for next steps
        self._unfreeze_parameters(self.model)
                    
        return history
    
    def train_step5(
        self,
        train_loader: torch.utils.data.DataLoader,
        soft_targets_original: torch.Tensor,
        soft_targets_updated: torch.Tensor,
        val_loader: torch.utils.data.DataLoader,
        epochs: int,
        early_stopping_patience: int = 10
    ) -> Dict[str, List[float]]:
        """
        Step 5: Final Consolidation with Bias Correction (TTT configuration)
        
        L^5 = E[η*L_KD(H̃_t, O^5_t) + (1-η)*L_KD(H'_t, O'^5_t) + 
              ||σ(W_Enc*θ_s(X_{t+1})) - σ(W_Enc*θ*_s(X_{t+1}))||² +
              L_CE(Y_{t+1}, O^5_{t+1})]
        (Equation 17)
        
        Args:
            train_loader: Training data loader
            soft_targets_original: Original soft targets H_t
            soft_targets_updated: Updated soft targets H'_t
            val_loader: Validation data loader
            epochs: Maximum training epochs
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Training history
        """
        # Store frozen backbone state for feature space constraint
        self.model.set_frozen_backbone()
        
        # Unfreeze all parameters except Auto-Encoder encoder/decoder (keep those fixed)
        self._unfreeze_parameters(self.model.backbone)
        self._unfreeze_parameters(self.model.classifier)
        self._freeze_parameters(self.model.autoencoder.encoder)
        self._freeze_parameters(self.model.autoencoder.decoder)
        
        # Trainable params: backbone + classifier + bias correction
        trainable_params = list(self.model.backbone.parameters())
        trainable_params.extend(self.model.classifier.parameters())
        trainable_params.extend([
            self.model.autoencoder.w_bias,
            self.model.autoencoder.b_bias,
            self.model.autoencoder.bias_projection.weight,
            self.model.autoencoder.bias_projection.bias
        ])
        
        optimizer = self._get_optimizer(trainable_params)
        
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            
            for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Get soft targets
                start_idx = batch_idx * train_loader.batch_size
                end_idx = start_idx + batch_x.size(0)
                batch_soft_original = soft_targets_original[start_idx:end_idx].to(self.device)
                batch_soft_updated = soft_targets_updated[start_idx:end_idx].to(self.device)
                
                optimizer.zero_grad()
                
                # Current features
                features = self.model.get_features(batch_x)
                
                # Frozen features for constraint
                with torch.no_grad():
                    frozen_features = self.model.get_frozen_features(batch_x)
                
                # Compute bias-corrected targets: H̃_t = Γ(P_{t+1}) ⊙ H_t
                bias_corrected_targets = self.model.compute_bias_corrected_logits(
                    features, batch_soft_original
                )
                
                # Old task outputs
                old_outputs = self.model.classifier.get_old_task_output(
                    features,
                    self.model.classifier.num_tasks - 1
                )
                
                # New task output
                new_outputs = self.model.classifier.get_new_task_output(features)
                
                # KD losses
                kd_loss_bias_corrected = self.kd_loss(bias_corrected_targets, old_outputs)
                kd_loss_updated = self.kd_loss(batch_soft_updated, old_outputs)
                
                # Feature space constraint: ||σ(W_Enc*θ_s) - σ(W_Enc*θ*_s)||²
                encoded_current = self.model.autoencoder.encode(features)
                with torch.no_grad():
                    encoded_frozen = self.model.autoencoder.encode(frozen_features)
                feature_constraint = F.mse_loss(encoded_current, encoded_frozen)
                
                # CE loss
                ce_loss = self.ce_loss(new_outputs, batch_y)
                
                # Combined loss (Equation 17)
                loss = (self.eta * kd_loss_bias_corrected + 
                       (1 - self.eta) * kd_loss_updated +
                       feature_constraint +
                       ce_loss)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            
            val_loss, val_acc = self._validate(val_loader)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = deepcopy(self.model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Step 5 early stopping at epoch {epoch}")
                    self.model.load_state_dict(best_state)
                    break
        
        # Update stored original heads for next task
        self._store_original_old_heads()
                    
        return history
    
    def _validate_autoencoder(
        self, 
        val_loader: torch.utils.data.DataLoader
    ) -> float:
        """Validate Auto-Encoder reconstruction."""
        self.model.autoencoder.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                features = self.model.get_features(batch_x)
                reconstructed = self.model.autoencoder(features)
                
                recon_loss = F.mse_loss(reconstructed, features)
                old_task_output = self.model.classifier.get_old_task_output(
                    reconstructed,
                    self.model.classifier.num_tasks
                )
                ce_loss = self.ce_loss(old_task_output, batch_y)
                
                total_loss += (self.omega * recon_loss + ce_loss).item()
                
        return total_loss / len(val_loader)
    
    def train_first_task(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int = 100,
        early_stopping_patience: int = 10
    ) -> Dict[str, Dict[str, List[float]]]:
        """
        Complete training pipeline for the FIRST task (Steps 1-2 for NFL+).
        
        For NFL+, this includes:
        - Step 1: Initial task training
        - Step 2: Auto-Encoder training on this task's data (X_t)
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum epochs per step
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Training history
        """
        history = {}
        
        # Step 1: Initial task training
        print("Step 1: Initial task training...")
        history['step1'] = self.train_step1(
            train_loader, val_loader, epochs, early_stopping_patience
        )
        
        # Step 2: Train Auto-Encoder on CURRENT task data (this is X_t for next task)
        print("Step 2: Training Auto-Encoder on current task data...")
        history['step2_ae'] = self.train_step2_autoencoder(
            train_loader, val_loader, epochs, early_stopping_patience
        )
        
        self.current_task += 1
        
        return history
    
    def train_new_task(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int = 100,
        early_stopping_patience: int = 10
    ) -> Dict[str, Dict[str, List[float]]]:
        """
        Complete training pipeline for a new task (Steps 3-5 for NFL+).
        
        For NFL+:
        - Step 3: Train new task head (FFT) - same as NFL Step 2
        - Step 4: Train backbone and old heads (TTF) - same as NFL Step 3
        - Train bias correction parameters
        - Step 5: Final consolidation with bias correction
        - Then train Auto-Encoder on this task's data for NEXT task
        
        Args:
            train_loader: Training data loader for new task
            val_loader: Validation data loader
            epochs: Maximum epochs per step
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Training history for all steps
        """
        history = {}
        
        # Compute soft targets H_t using current model
        print("Computing soft targets H_t...")
        soft_targets = self.compute_soft_targets(train_loader)
        
        # Step 3: Train new task head (FFT) - corresponds to NFL Step 2
        print("Step 3: Training new task head (FFT)...")
        history['step3'] = self.train_step2(
            train_loader, val_loader, epochs, early_stopping_patience
        )
        
        # Step 4: Train backbone and old heads (TTF) - corresponds to NFL Step 3
        print("Step 4: Training backbone and old heads (TTF)...")
        history['step4'] = self.train_step3(
            train_loader, soft_targets, val_loader, epochs, early_stopping_patience
        )
        
        # Compute updated soft targets H'_t
        print("Computing updated soft targets H'_t...")
        updated_soft_targets = self.compute_updated_logits(train_loader)
        
        # Train bias correction parameters
        print("Training bias correction parameters...")
        history['bias_correction'] = self.train_bias_correction(
            val_loader, epochs=50, early_stopping_patience=early_stopping_patience
        )
        
        # Step 5: Final consolidation with bias correction
        print("Step 5: Final consolidation with bias correction...")
        history['step5'] = self.train_step5(
            train_loader, soft_targets, updated_soft_targets,
            val_loader, epochs, early_stopping_patience
        )
        
        # Train Auto-Encoder on this task's data for NEXT task
        print("Training Auto-Encoder for next task...")
        history['ae_for_next'] = self.train_step2_autoencoder(
            train_loader, val_loader, epochs, early_stopping_patience
        )
        
        self.current_task += 1
        
        return history


# ---------------------------------------------------------------------------
# NFL+LoRA: Adaptation for Vision Transformers (Section 3.4)
# ---------------------------------------------------------------------------

class NFLPlusLoRAModel(nn.Module):
    """
    NFL+LoRA model for pre-trained Vision Transformers.
    
    Pre-trained ViT weights W_0 remain frozen. All backbone updates are
    confined to a low-rank subspace via LoRA. After each task, LoRA weights
    are merged into the base model and a fresh LoRA module is initialized,
    keeping memory constant regardless of task count.
    
    Follows the 4-step NFL structure (Steps 1-4) with:
    - LoRA parameters replacing full backbone updates
    - Fisher-weighted regularization replacing the Auto-Encoder constraint
    """
    
    def __init__(
        self,
        backbone,  # ViTLoRABackbone
        feature_dim: int,
        initial_classes: int,
    ):
        super().__init__()
        self.backbone = backbone
        self.classifier = MultiHeadClassifier(feature_dim, initial_classes)
        self.feature_dim = feature_dim
    
    def forward(
        self, x: torch.Tensor, task_id: Optional[int] = None
    ) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features, task_id)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def add_task(self, num_classes: int):
        self.classifier.add_task_head(num_classes)


class NFLPlusLoRATrainer:
    """
    Trainer for NFL+LoRA implementing the 4-step process with Fisher
    regularization (Equation 18 in the paper).
    
    L^6 = alpha L_KD(H_t, O^6_t) + (1-alpha) L_KD(H'_t, O'^6_t)
          + L_CE(Y_{t+1}, O^6_{t+1})
          + (lambda/2) vec(AB)^T F_{t-1}^cum vec(AB)
    """
    
    def __init__(
        self,
        model: NFLPlusLoRAModel,
        device: torch.device,
        temperature: float = 2.0,
        alpha: float = 0.5,
        fisher_lambda: float = 1.0,
        lr: float = 0.001,
        weight_decay: float = 0.0,
    ):
        self.model = model
        self.device = device
        self.temperature = temperature
        self.alpha = alpha
        self.fisher_lambda = fisher_lambda
        self.lr = lr
        self.weight_decay = weight_decay
        
        self.kd_loss = KnowledgeDistillationLoss(temperature)
        self.ce_loss = nn.CrossEntropyLoss()
        self.current_task = 0
        
        # Store original old task head states for H'_t computation
        self.original_old_head_states = []
    
    def _get_optimizer(self, parameters):
        return torch.optim.Adam(
            parameters, lr=self.lr, weight_decay=self.weight_decay
        )
    
    def _freeze_parameters(self, module: nn.Module):
        for param in module.parameters():
            param.requires_grad = False
    
    def _unfreeze_parameters(self, module: nn.Module):
        for param in module.parameters():
            param.requires_grad = True
    
    def _store_original_old_heads(self):
        self.original_old_head_states = []
        for head in self.model.classifier.heads:
            self.original_old_head_states.append(deepcopy(head.state_dict()))
    
    def _validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                outputs = self.model(batch_x)
                loss = self.ce_loss(outputs, batch_y)
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()
        return total_loss / max(len(val_loader), 1), correct / max(total, 1)
    
    def _validate_new_task(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                features = self.model.get_features(batch_x)
                outputs = self.model.classifier.get_new_task_output(features)
                loss = self.ce_loss(outputs, batch_y)
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()
        return total_loss / max(len(val_loader), 1), correct / max(total, 1)
    
    def train_step1(self, train_loader, val_loader, epochs,
                    early_stopping_patience=10):
        """Step 1: Train LoRA + head with CE loss."""
        self.model.backbone.unfreeze_lora()
        self._unfreeze_parameters(self.model.classifier)
        
        params = self.model.backbone.get_lora_parameters()
        params += list(self.model.classifier.parameters())
        optimizer = self._get_optimizer(params)
        
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.ce_loss(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= max(len(train_loader), 1)
            history['train_loss'].append(train_loss)
            val_loss, val_acc = self._validate(val_loader)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = deepcopy(self.model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    self.model.load_state_dict(best_state)
                    break
        self._store_original_old_heads()
        return history
    
    def compute_soft_targets(self, data_loader):
        """Compute H_t from old-task heads."""
        self.model.eval()
        all_logits = []
        with torch.no_grad():
            for batch_x, _ in data_loader:
                batch_x = batch_x.to(self.device)
                features = self.model.get_features(batch_x)
                old_logits = self.model.classifier.get_old_task_output(
                    features, self.model.classifier.num_tasks - 1
                )
                all_logits.append(old_logits.cpu())
        return torch.cat(all_logits, dim=0)
    
    def train_step2(self, train_loader, val_loader, epochs,
                    early_stopping_patience=10):
        """Step 2 (FFT): Freeze LoRA + old heads, train only new head."""
        self.model.backbone.freeze_lora()
        for i in range(self.model.classifier.num_tasks - 1):
            self._freeze_parameters(self.model.classifier.heads[i])
        self._unfreeze_parameters(self.model.classifier.heads[-1])
        optimizer = self._get_optimizer(self.model.classifier.heads[-1].parameters())
        
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                features = self.model.get_features(batch_x)
                outputs = self.model.classifier.get_new_task_output(features)
                loss = self.ce_loss(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= max(len(train_loader), 1)
            history['train_loss'].append(train_loss)
            val_loss, val_acc = self._validate_new_task(val_loader)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = deepcopy(self.model.classifier.heads[-1].state_dict())
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    self.model.classifier.heads[-1].load_state_dict(best_state)
                    break
        return history
    
    def train_step3(self, train_loader, soft_targets, val_loader, epochs,
                    early_stopping_patience=10):
        """Step 3 (TTF): Freeze new head, train LoRA + old heads."""
        self._freeze_parameters(self.model.classifier.heads[-1])
        self.model.backbone.unfreeze_lora()
        for i in range(self.model.classifier.num_tasks - 1):
            self._unfreeze_parameters(self.model.classifier.heads[i])
        
        params = self.model.backbone.get_lora_parameters()
        for i in range(self.model.classifier.num_tasks - 1):
            params += list(self.model.classifier.heads[i].parameters())
        optimizer = self._get_optimizer(params)
        
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                start = batch_idx * train_loader.batch_size
                end = start + batch_x.size(0)
                batch_soft = soft_targets[start:end].to(self.device)
                optimizer.zero_grad()
                features = self.model.get_features(batch_x)
                old_out = self.model.classifier.get_old_task_output(
                    features, self.model.classifier.num_tasks - 1
                )
                new_out = self.model.classifier.get_new_task_output(features)
                loss = self.kd_loss(batch_soft, old_out) + self.ce_loss(new_out, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= max(len(train_loader), 1)
            history['train_loss'].append(train_loss)
            val_loss, val_acc = self._validate(val_loader)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = deepcopy(self.model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    self.model.load_state_dict(best_state)
                    break
        return history
    
    def compute_updated_logits(self, data_loader):
        """Compute H'_t using updated backbone + original old head theta_t."""
        self.model.eval()
        current_states = []
        for i in range(self.model.classifier.num_tasks - 1):
            current_states.append(
                deepcopy(self.model.classifier.heads[i].state_dict())
            )
        for i in range(len(self.original_old_head_states)):
            if i < self.model.classifier.num_tasks - 1:
                self.model.classifier.heads[i].load_state_dict(
                    self.original_old_head_states[i]
                )
        all_logits = []
        with torch.no_grad():
            for batch_x, _ in data_loader:
                batch_x = batch_x.to(self.device)
                features = self.model.get_features(batch_x)
                old_logits = self.model.classifier.get_old_task_output(
                    features, self.model.classifier.num_tasks - 1
                )
                all_logits.append(old_logits.cpu())
        for i, state in enumerate(current_states):
            self.model.classifier.heads[i].load_state_dict(state)
        return torch.cat(all_logits, dim=0)
    
    def train_step4(self, train_loader, soft_targets_original,
                    soft_targets_updated, val_loader, epochs,
                    early_stopping_patience=10):
        """Step 4 (TTT): Joint fine-tuning with dual KD + Fisher penalty (Eq. 18)."""
        self.model.backbone.unfreeze_lora()
        self._unfreeze_parameters(self.model.classifier)
        
        params = self.model.backbone.get_lora_parameters()
        params += list(self.model.classifier.parameters())
        optimizer = self._get_optimizer(params)
        
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                start = batch_idx * train_loader.batch_size
                end = start + batch_x.size(0)
                batch_orig = soft_targets_original[start:end].to(self.device)
                batch_upd = soft_targets_updated[start:end].to(self.device)
                optimizer.zero_grad()
                features = self.model.get_features(batch_x)
                old_out = self.model.classifier.get_old_task_output(
                    features, self.model.classifier.num_tasks - 1
                )
                new_out = self.model.classifier.get_new_task_output(features)
                kd_orig = self.kd_loss(batch_orig, old_out)
                kd_upd = self.kd_loss(batch_upd, old_out)
                ce = self.ce_loss(new_out, batch_y)
                fisher_pen = self.model.backbone.compute_fisher_penalty()
                loss = (self.alpha * kd_orig
                        + (1 - self.alpha) * kd_upd
                        + ce
                        + (self.fisher_lambda / 2) * fisher_pen)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= max(len(train_loader), 1)
            history['train_loss'].append(train_loss)
            val_loss, val_acc = self._validate(val_loader)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = deepcopy(self.model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    self.model.load_state_dict(best_state)
                    break
        self._store_original_old_heads()
        return history
    
    def end_task(self, train_loader):
        """Post-task: compute Fisher, accumulate, merge LoRA, reset."""
        fisher = self.model.backbone.compute_fisher(
            train_loader, self.model.classifier, self.device
        )
        self.model.backbone.accumulate_fisher(fisher)
        self.model.backbone.merge_and_reset_all()
    
    def train_first_task(self, train_loader, val_loader, epochs,
                         early_stopping_patience=10):
        """Complete first task: Step 1 then end_task."""
        history = {'step1': self.train_step1(
            train_loader, val_loader, epochs, early_stopping_patience
        )}
        self.end_task(train_loader)
        self.current_task += 1
        return history
    
    def train_new_task(self, train_loader, val_loader, epochs=100,
                       early_stopping_patience=10):
        """Complete pipeline: Steps 2-4 then end_task."""
        history = {}
        soft_targets = self.compute_soft_targets(train_loader)
        history['step2'] = self.train_step2(
            train_loader, val_loader, epochs, early_stopping_patience
        )
        history['step3'] = self.train_step3(
            train_loader, soft_targets, val_loader, epochs, early_stopping_patience
        )
        soft_targets_updated = self.compute_updated_logits(train_loader)
        history['step4'] = self.train_step4(
            train_loader, soft_targets, soft_targets_updated,
            val_loader, epochs, early_stopping_patience
        )
        self.end_task(train_loader)
        self.current_task += 1
        return history
