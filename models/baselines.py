"""
Baseline Continual Learning Methods

Wraps Avalanche library implementations for standard baselines.
For methods not in Avalanche, provides standalone implementations.

All baselines compared in the paper:
    Memory-free: EWC, SI, LwF, PEC, SpaceNet, NISPA, DCNet
    Memory-based: iCaRL, DER++, DyTox, MEMO
    LoRA-based (ViT): CL-LoRA, EWC-LoRA

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from copy import deepcopy

try:
    from avalanche.training.supervised import (
        EWC as AvalancheEWC,
        SynapticIntelligence as AvalancheSI,
        LwF as AvalancheLwF,
        ICaRL as AvalancheICaRL,
    )
    from avalanche.training.plugins import (
        EWCPlugin, SynapticIntelligencePlugin, LwFPlugin,
        ReplayPlugin,
    )
    AVALANCHE_AVAILABLE = True
except ImportError:
    AVALANCHE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Standalone baseline implementations (no Avalanche dependency)
# ---------------------------------------------------------------------------

class EWCBaseline:
    """
    Elastic Weight Consolidation (Kirkpatrick et al., 2017).
    
    Regularizes important parameters using the diagonal Fisher Information
    Matrix estimated after each task.
    """
    
    def __init__(self, model, device, lr=0.001, ewc_lambda=400.0,
                 fisher_sample_size=500):
        self.model = model
        self.device = device
        self.lr = lr
        self.ewc_lambda = ewc_lambda
        self.fisher_sample_size = fisher_sample_size
        self.ce_loss = nn.CrossEntropyLoss()
        
        self.fisher_dict = {}
        self.param_dict = {}
    
    def compute_fisher(self, data_loader):
        """Estimate diagonal Fisher on current task data."""
        self.model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()
                  if p.requires_grad}
        count = 0
        for batch_x, batch_y in data_loader:
            if count >= self.fisher_sample_size:
                break
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            self.model.zero_grad()
            out = self.model(batch_x)
            loss = self.ce_loss(out, batch_y)
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data ** 2
            count += batch_x.size(0)
        for n in fisher:
            fisher[n] /= max(count, 1)
        return fisher
    
    def store_params(self):
        self.param_dict = {n: p.data.clone()
                           for n, p in self.model.named_parameters()
                           if p.requires_grad}
    
    def ewc_penalty(self):
        penalty = 0.0
        for n, p in self.model.named_parameters():
            if n in self.fisher_dict and n in self.param_dict:
                penalty += (self.fisher_dict[n] * (p - self.param_dict[n]) ** 2).sum()
        return penalty
    
    def train_task(self, train_loader, val_loader, epochs=100,
                   early_stopping_patience=10):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        best_val_loss = float('inf')
        patience = 0
        best_state = None
        
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                out = self.model(batch_x)
                loss = self.ce_loss(out, batch_y)
                loss += self.ewc_lambda * self.ewc_penalty()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            val_loss = self._validate(val_loader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0
                best_state = deepcopy(self.model.state_dict())
            else:
                patience += 1
                if patience >= early_stopping_patience:
                    self.model.load_state_dict(best_state)
                    break
    
    def end_task(self, data_loader):
        fisher = self.compute_fisher(data_loader)
        for n in fisher:
            if n in self.fisher_dict:
                self.fisher_dict[n] += fisher[n]
            else:
                self.fisher_dict[n] = fisher[n]
        self.store_params()
    
    def _validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                out = self.model(batch_x)
                total_loss += self.ce_loss(out, batch_y).item()
        return total_loss / max(len(val_loader), 1)


class SIBaseline:
    """
    Synaptic Intelligence (Zenke et al., 2017).
    
    Tracks parameter importance through online path integral along
    the training trajectory.
    """
    
    def __init__(self, model, device, lr=0.001, si_lambda=1.0, epsilon=1e-3):
        self.model = model
        self.device = device
        self.lr = lr
        self.si_lambda = si_lambda
        self.epsilon = epsilon
        self.ce_loss = nn.CrossEntropyLoss()
        
        self.omega = {}
        self.prev_params = {}
        self.big_omega = {}
        self.small_omega = {}
        
        self._init_si()
    
    def _init_si(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.prev_params[n] = p.data.clone()
                self.big_omega[n] = torch.zeros_like(p)
                self.small_omega[n] = torch.zeros_like(p)
    
    def si_penalty(self):
        penalty = 0.0
        for n, p in self.model.named_parameters():
            if n in self.big_omega:
                penalty += (self.big_omega[n] * (p - self.prev_params[n]) ** 2).sum()
        return penalty
    
    def train_task(self, train_loader, val_loader, epochs=100,
                   early_stopping_patience=10):
        # Reset small omega for this task
        for n in self.small_omega:
            self.small_omega[n].zero_()
        
        init_params = {n: p.data.clone()
                       for n, p in self.model.named_parameters()
                       if p.requires_grad}
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        best_val_loss = float('inf')
        patience = 0
        best_state = None
        
        for epoch in range(epochs):
            self.model.train()
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                out = self.model(batch_x)
                loss = self.ce_loss(out, batch_y)
                loss += self.si_lambda * self.si_penalty()
                loss.backward()
                
                # Track path integral
                for n, p in self.model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        self.small_omega[n] += -p.grad.data * (p.data - init_params[n])
                
                optimizer.step()
            
            val_loss = self._validate(val_loader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0
                best_state = deepcopy(self.model.state_dict())
            else:
                patience += 1
                if patience >= early_stopping_patience:
                    self.model.load_state_dict(best_state)
                    break
    
    def end_task(self):
        for n, p in self.model.named_parameters():
            if n in self.small_omega:
                delta = p.data - self.prev_params[n]
                self.big_omega[n] += self.small_omega[n] / (delta ** 2 + self.epsilon)
                self.prev_params[n] = p.data.clone()
    
    def _validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                total_loss += self.ce_loss(self.model(batch_x), batch_y).item()
        return total_loss / max(len(val_loader), 1)


class LwFBaseline:
    """
    Learning without Forgetting (Li & Hoiem, 2017).
    
    Uses knowledge distillation from old model outputs on new data
    to preserve old task performance.
    """
    
    def __init__(self, model, device, lr=0.001, temperature=2.0, lwf_alpha=1.0):
        self.model = model
        self.device = device
        self.lr = lr
        self.temperature = temperature
        self.lwf_alpha = lwf_alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.old_model = None
    
    def _kd_loss(self, old_logits, new_logits):
        old_probs = F.softmax(old_logits / self.temperature, dim=1)
        new_log_probs = F.log_softmax(new_logits / self.temperature, dim=1)
        return -(old_probs * new_log_probs).sum(dim=1).mean() * self.temperature ** 2
    
    def train_task(self, train_loader, val_loader, epochs=100,
                   early_stopping_patience=10, num_old_classes=0):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        best_val_loss = float('inf')
        patience = 0
        best_state = None
        
        for epoch in range(epochs):
            self.model.train()
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                out = self.model(batch_x)
                loss = self.ce_loss(out[:, num_old_classes:], batch_y)
                
                if self.old_model is not None and num_old_classes > 0:
                    with torch.no_grad():
                        old_out = self.old_model(batch_x)[:, :num_old_classes]
                    loss += self.lwf_alpha * self._kd_loss(
                        old_out, out[:, :num_old_classes]
                    )
                loss.backward()
                optimizer.step()
            
            val_loss = self._validate(val_loader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0
                best_state = deepcopy(self.model.state_dict())
            else:
                patience += 1
                if patience >= early_stopping_patience:
                    self.model.load_state_dict(best_state)
                    break
    
    def end_task(self):
        self.old_model = deepcopy(self.model)
        self.old_model.eval()
        for p in self.old_model.parameters():
            p.requires_grad = False
    
    def _validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                total_loss += self.ce_loss(self.model(batch_x), batch_y).item()
        return total_loss / max(len(val_loader), 1)


class DERPPBaseline:
    """
    Dark Experience Replay++ (Buzzega et al., 2020).
    
    Stores both exemplars and their logits (dark experience) in a
    reservoir buffer, replaying both data and soft targets.
    """
    
    def __init__(self, model, device, lr=0.001, buffer_size=2000,
                 alpha=0.1, beta=0.5):
        self.model = model
        self.device = device
        self.lr = lr
        self.buffer_size = buffer_size
        self.alpha = alpha
        self.beta = beta
        self.ce_loss = nn.CrossEntropyLoss()
        
        self.buffer_x = []
        self.buffer_y = []
        self.buffer_logits = []
        self.buffer_count = 0
    
    def _reservoir_update(self, x, y, logits):
        """Reservoir sampling for buffer management."""
        for i in range(x.size(0)):
            if self.buffer_count < self.buffer_size:
                self.buffer_x.append(x[i].cpu())
                self.buffer_y.append(y[i].cpu())
                self.buffer_logits.append(logits[i].cpu())
            else:
                j = torch.randint(0, self.buffer_count + 1, (1,)).item()
                if j < self.buffer_size:
                    self.buffer_x[j] = x[i].cpu()
                    self.buffer_y[j] = y[i].cpu()
                    self.buffer_logits[j] = logits[i].cpu()
            self.buffer_count += 1
    
    def _sample_buffer(self, batch_size):
        if len(self.buffer_x) == 0:
            return None, None, None
        idx = torch.randint(0, len(self.buffer_x), (min(batch_size, len(self.buffer_x)),))
        bx = torch.stack([self.buffer_x[i] for i in idx]).to(self.device)
        by = torch.stack([self.buffer_y[i] for i in idx]).to(self.device)
        bl = torch.stack([self.buffer_logits[i] for i in idx]).to(self.device)
        return bx, by, bl
    
    def train_task(self, train_loader, val_loader, epochs=100,
                   early_stopping_patience=10):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        best_val_loss = float('inf')
        patience = 0
        best_state = None
        
        for epoch in range(epochs):
            self.model.train()
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                out = self.model(batch_x)
                loss = self.ce_loss(out, batch_y)
                
                # Replay from buffer
                buf_x, buf_y, buf_logits = self._sample_buffer(batch_x.size(0))
                if buf_x is not None:
                    buf_out = self.model(buf_x)
                    loss += self.alpha * self.ce_loss(buf_out, buf_y)
                    loss += self.beta * F.mse_loss(buf_out, buf_logits)
                
                loss.backward()
                optimizer.step()
                
                # Update buffer with current batch logits
                with torch.no_grad():
                    current_logits = self.model(batch_x)
                self._reservoir_update(batch_x, batch_y, current_logits)
            
            val_loss = self._validate(val_loader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0
                best_state = deepcopy(self.model.state_dict())
            else:
                patience += 1
                if patience >= early_stopping_patience:
                    self.model.load_state_dict(best_state)
                    break
    
    def _validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                total_loss += self.ce_loss(self.model(batch_x), batch_y).item()
        return total_loss / max(len(val_loader), 1)


class SGDLowerBound:
    """SGD Lower Bound — plain fine-tuning with no CL strategy."""
    
    def __init__(self, model, device, lr=0.001):
        self.model = model
        self.device = device
        self.lr = lr
        self.ce_loss = nn.CrossEntropyLoss()
    
    def train_task(self, train_loader, val_loader, epochs=100,
                   early_stopping_patience=10):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        best_val_loss = float('inf')
        patience = 0
        best_state = None
        for epoch in range(epochs):
            self.model.train()
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                loss = self.ce_loss(self.model(batch_x), batch_y)
                loss.backward()
                optimizer.step()
            val_loss = self._validate(val_loader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0
                best_state = deepcopy(self.model.state_dict())
            else:
                patience += 1
                if patience >= early_stopping_patience:
                    self.model.load_state_dict(best_state)
                    break
    
    def _validate(self, val_loader):
        self.model.eval()
        total = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                total += self.ce_loss(self.model(bx), by).item()
        return total / max(len(val_loader), 1)


# ---------------------------------------------------------------------------
# Registry for all methods
# ---------------------------------------------------------------------------

BASELINE_REGISTRY = {
    'ewc': EWCBaseline,
    'si': SIBaseline,
    'lwf': LwFBaseline,
    'der++': DERPPBaseline,
    'sgd': SGDLowerBound,
}

# Methods requiring separate repositories (referenced in paper):
EXTERNAL_BASELINES = {
    'dcnet': 'https://github.com/xxx/DCNet',
    'nispa': 'https://github.com/xxx/NISPA',
    'spacenet': 'https://github.com/xxx/SpaceNet',
    'pec': 'https://github.com/xxx/PEC',
    'dytox': 'https://github.com/arthurdouillard/dytox',
    'memo': 'https://github.com/xxx/MEMO',
    'icarl': 'Avalanche: avalanche.training.supervised.ICaRL',
    'cl-lora': 'https://github.com/xxx/CL-LoRA',
    'ewc-lora': 'https://github.com/xxx/EWC-LoRA',
}


def get_baseline(name: str, model, device, **kwargs):
    """
    Instantiate a baseline method by name.
    
    Args:
        name: Method name (see BASELINE_REGISTRY)
        model: The neural network model
        device: torch device
        **kwargs: Method-specific hyperparameters
    
    Returns:
        Baseline trainer instance
    
    Raises:
        ValueError if method not found
    """
    name_lower = name.lower()
    if name_lower in BASELINE_REGISTRY:
        return BASELINE_REGISTRY[name_lower](model, device, **kwargs)
    elif name_lower in EXTERNAL_BASELINES:
        raise ValueError(
            f"'{name}' requires an external implementation. "
            f"See: {EXTERNAL_BASELINES[name_lower]}"
        )
    else:
        available = list(BASELINE_REGISTRY.keys()) + list(EXTERNAL_BASELINES.keys())
        raise ValueError(
            f"Unknown method '{name}'. Available: {available}"
        )
