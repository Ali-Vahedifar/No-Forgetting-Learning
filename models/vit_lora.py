"""
Vision Transformer with Low-Rank Adaptation (LoRA) for NFL+LoRA.

Implements ViT-B/16 with LoRA modules as described in Section 3.4 of the paper.
Pre-trained weights from ImageNet-21K remain frozen; only LoRA parameters
(A, B matrices) and task heads are trainable.

Anonymous submission - NeurIPS 2026
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict
from copy import deepcopy

try:
    import timm
except ImportError:
    timm = None


class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation layer: ΔW = A @ B
    
    The effective weight is W = W_0 + A @ B, where W_0 is frozen.
    Only A and B are trainable.
    """
    
    def __init__(self, in_features: int, out_features: int, rank: int = 8):
        super().__init__()
        self.rank = rank
        self.in_features = in_features
        self.out_features = out_features
        
        # LoRA matrices: A ∈ R^{d_O × r}, B ∈ R^{r × d_I}
        self.A = nn.Parameter(torch.zeros(out_features, rank))
        self.B = nn.Parameter(torch.zeros(rank, in_features))
        
        # Initialize A with Kaiming, B with zeros (standard LoRA init)
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)
        
        self.scaling = 1.0 / rank
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute ΔW @ x = (A @ B) @ x, scaled."""
        return F.linear(x, self.A @ self.B * self.scaling)
    
    def get_delta_weight(self) -> torch.Tensor:
        """Return ΔW = A @ B (for merging into base weights)."""
        return (self.A @ self.B * self.scaling).detach()
    
    def reset_parameters(self):
        """Re-initialize for next task (after merging)."""
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation.
    Base weights W_0 are frozen; output = W_0 @ x + (A @ B) @ x.
    """
    
    def __init__(self, base_linear: nn.Linear, rank: int = 8):
        super().__init__()
        self.base_linear = base_linear
        # Freeze base weights
        for param in self.base_linear.parameters():
            param.requires_grad = False
        
        self.lora = LoRALayer(
            base_linear.in_features, base_linear.out_features, rank
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(x)
        lora_out = self.lora(x)
        return base_out + lora_out
    
    def merge_lora(self):
        """Merge learned LoRA into base weights: W_t = W_{t-1} + A*B*."""
        delta = self.lora.get_delta_weight()
        self.base_linear.weight.data += delta
    
    def reset_lora(self):
        """Initialize fresh LoRA for next task."""
        self.lora.reset_parameters()
    
    def merge_and_reset(self):
        """Merge then reset — constant memory regardless of task count."""
        self.merge_lora()
        self.reset_lora()


class ViTLoRABackbone(nn.Module):
    """
    ViT-B/16 backbone with LoRA adaptation for NFL+LoRA.
    
    Pre-trained ViT weights W_0 remain frozen throughout training.
    LoRA modules are applied to the query and value projections
    in each transformer block.
    
    After each task:
    1. Compute diagonal Fisher F_t and accumulate F_t^cum = γ F_{t-1}^cum + F_t
    2. Merge: W_t = W_{t-1} + A*B*
    3. Reset: fresh A, B ← 0
    """
    
    def __init__(
        self,
        model_name: str = 'vit_base_patch16_224',
        pretrained: bool = True,
        lora_rank: int = 8,
        fisher_decay: float = 0.95
    ):
        super().__init__()
        
        if timm is None:
            raise ImportError(
                "timm is required for ViT backbone. Install with: "
                "pip install timm --break-system-packages"
            )
        
        # Load pre-trained ViT
        self.vit = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0
        )
        self.feature_dim = self.vit.embed_dim  # 768 for ViT-B
        self.lora_rank = lora_rank
        self.fisher_decay = fisher_decay
        
        # Freeze all pre-trained weights
        for param in self.vit.parameters():
            param.requires_grad = False
        
        # Apply LoRA to query and value projections in each attention block
        self.lora_layers: Dict[str, LoRALinear] = {}
        self._apply_lora_to_attention()
        
        # Fisher information accumulator
        self.fisher_cum = None
        self._init_fisher()
    
    def _apply_lora_to_attention(self):
        """Replace Q, V projections with LoRA-adapted versions."""
        for name, block in self.vit.blocks.named_children():
            attn = block.attn
            
            # Replace qkv projection — ViT uses fused qkv, so we wrap it
            if hasattr(attn, 'qkv'):
                lora_qkv = LoRALinear(attn.qkv, rank=self.lora_rank)
                attn.qkv = lora_qkv
                self.lora_layers[f'block_{name}_qkv'] = lora_qkv
            
            # Replace output projection
            if hasattr(attn, 'proj'):
                lora_proj = LoRALinear(attn.proj, rank=self.lora_rank)
                attn.proj = lora_proj
                self.lora_layers[f'block_{name}_proj'] = lora_proj
    
    def _init_fisher(self):
        """Initialize Fisher accumulator to zeros."""
        self.fisher_cum = {}
        for name, lora_linear in self.lora_layers.items():
            lora = lora_linear.lora
            self.fisher_cum[f'{name}_A'] = torch.zeros_like(lora.A)
            self.fisher_cum[f'{name}_B'] = torch.zeros_like(lora.B)
    
    def get_lora_parameters(self) -> List[nn.Parameter]:
        """Get all trainable LoRA parameters."""
        params = []
        for lora_linear in self.lora_layers.values():
            params.append(lora_linear.lora.A)
            params.append(lora_linear.lora.B)
        return params
    
    def compute_fisher(
        self,
        data_loader: torch.utils.data.DataLoader,
        classifier: nn.Module,
        device: torch.device,
        num_samples: int = 500
    ) -> Dict[str, torch.Tensor]:
        """
        Estimate diagonal Fisher Information Matrix over ΔW-space.
        
        Since W_0 is frozen, ∂L/∂W = ∂L/∂ΔW, so Fisher is computed
        in ΔW-space without additional overhead.
        """
        fisher = {}
        for name, lora_linear in self.lora_layers.items():
            lora = lora_linear.lora
            fisher[f'{name}_A'] = torch.zeros_like(lora.A)
            fisher[f'{name}_B'] = torch.zeros_like(lora.B)
        
        self.eval()
        count = 0
        
        for batch_x, batch_y in data_loader:
            if count >= num_samples:
                break
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            features = self.forward(batch_x)
            outputs = classifier(features)
            loss = F.cross_entropy(outputs, batch_y)
            loss.backward()
            
            for name, lora_linear in self.lora_layers.items():
                lora = lora_linear.lora
                if lora.A.grad is not None:
                    fisher[f'{name}_A'] += lora.A.grad.data ** 2
                if lora.B.grad is not None:
                    fisher[f'{name}_B'] += lora.B.grad.data ** 2
            
            self.zero_grad()
            count += batch_x.size(0)
        
        # Normalize
        for key in fisher:
            fisher[key] /= max(count, 1)
        
        return fisher
    
    def accumulate_fisher(self, fisher: Dict[str, torch.Tensor]):
        """F_t^cum = γ F_{t-1}^cum + F_t"""
        for key in fisher:
            if key in self.fisher_cum:
                self.fisher_cum[key] = (
                    self.fisher_decay * self.fisher_cum[key].to(fisher[key].device)
                    + fisher[key]
                )
            else:
                self.fisher_cum[key] = fisher[key].clone()
    
    def compute_fisher_penalty(self) -> torch.Tensor:
        """
        Compute (λ/2) vec(AB)^T F_{t-1}^cum vec(AB).
        
        Approximated as sum over layers of element-wise F * (param^2).
        """
        penalty = torch.tensor(0.0, device=next(self.parameters()).device)
        
        for name, lora_linear in self.lora_layers.items():
            lora = lora_linear.lora
            f_A = self.fisher_cum.get(f'{name}_A')
            f_B = self.fisher_cum.get(f'{name}_B')
            
            if f_A is not None:
                f_A = f_A.to(lora.A.device)
                penalty = penalty + (f_A * lora.A ** 2).sum()
            if f_B is not None:
                f_B = f_B.to(lora.B.device)
                penalty = penalty + (f_B * lora.B ** 2).sum()
        
        return penalty
    
    def merge_and_reset_all(self):
        """Merge all LoRA weights and reinitialize for next task."""
        for lora_linear in self.lora_layers.values():
            lora_linear.merge_and_reset()
    
    def freeze_lora(self):
        """Freeze LoRA parameters (for FFT step)."""
        for lora_linear in self.lora_layers.values():
            lora_linear.lora.A.requires_grad = False
            lora_linear.lora.B.requires_grad = False
    
    def unfreeze_lora(self):
        """Unfreeze LoRA parameters."""
        for lora_linear in self.lora_layers.values():
            lora_linear.lora.A.requires_grad = True
            lora_linear.lora.B.requires_grad = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — returns CLS token features."""
        return self.vit(x)


def get_vit_lora_backbone(
    model_name: str = 'vit_base_patch16_224',
    pretrained: bool = True,
    lora_rank: int = 8,
    fisher_decay: float = 0.95
) -> ViTLoRABackbone:
    """
    Create ViT-B/16 backbone with LoRA for NFL+LoRA.
    
    Args:
        model_name: timm model name (default: 'vit_base_patch16_224')
        pretrained: Use ImageNet-21K pretrained weights
        lora_rank: LoRA rank r (default: 8)
        fisher_decay: Fisher decay factor γ (default: 0.95)
    
    Returns:
        ViTLoRABackbone instance
    """
    return ViTLoRABackbone(
        model_name=model_name,
        pretrained=pretrained,
        lora_rank=lora_rank,
        fisher_decay=fisher_decay
    )
