"""
 CUDA Graph pruning scheduler and fixed-shape runtime state.

This module provides immutable configuration and external state management
to enable CUDA Graph capture for both dense and sparse forward passes.

Key Design Principles:
1. StaticPruneConfig: Immutable (frozen dataclass) - set once at model init
2. RuntimeState: Fixed-shape index buffers shared by dense/sparse CUDA Graphs
3. PruningScheduler: External owner of those buffers
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict
import torch


@dataclass(frozen=True)
class StaticPruneConfig:
    """
    Immutable pruning configuration - set once at model initialization, never modified.

    This enables CUDA Graph capture and prevents torch.compile recompilation.
    All values are compile-time constants after initialization.
    """
    # Model architecture (read from LlamaConfig)
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    num_layers: int = 32
    max_seq_len: int = 611

    # Pruning configuration
    mlp_prune_ratio: float = 0.25
    attn_prune_ratio: float = 0.25
    skip_layers: Tuple[int, ...] = (0, 1, 2)

    # Metric calculation settings
    history_metric_ratio: float = 0.9
    tc_multiple: int = 64  # Tensor core alignment
    data_type_max: float = 65504.0  # BF16 max value
    transpose_down_linear: bool = True

    @property
    def num_prunable_layers(self) -> int:
        """Number of layers that will be pruned (excluding skip_layers)"""
        return self.num_layers - len(self.skip_layers)

    @property
    def mlp_keep_dims(self) -> int:
        """
        Fixed number of MLP dimensions to keep (aligned to tc_multiple).
        This is constant and enables static tensor shapes for CUDA Graph.
        """
        num_prune = int(self.mlp_prune_ratio * self.intermediate_size)
        # Round up to tc_multiple
        num_prune = ((num_prune + self.tc_multiple - 1) // self.tc_multiple) * self.tc_multiple
        # Ensure we keep at least tc_multiple dims
        num_prune = min(num_prune, self.intermediate_size - self.tc_multiple)
        return self.intermediate_size - num_prune

    @property
    def attn_keep_heads(self) -> int:
        """Fixed number of attention heads to keep"""
        num_prune = int(self.attn_prune_ratio * self.num_attention_heads)
        return self.num_attention_heads - num_prune

    @property
    def attn_keep_dims(self) -> int:
        """Fixed number of attention dimensions to keep (heads * head_dim)"""
        return self.attn_keep_heads * self.head_dim

    def get_prunable_layer_indices(self) -> Tuple[int, ...]:
        """Returns tuple of layer indices that will be pruned"""
        return tuple(i for i in range(self.num_layers) if i not in self.skip_layers)


@dataclass
class RuntimeState:
    """
    Mutable index state managed by PruningScheduler outside CUDA Graph.

    All tensor buffers have FIXED shapes to enable CUDA Graph capture.
    The indices tensors are updated in-place using copy_() on metric_stream.
    """
    # Pre-allocated fixed-size indices buffers
    # Shape: [num_prunable_layers, keep_dims]
    mlp_indices_buffer: Optional[torch.Tensor] = None
    attn_qk_indices_buffer: Optional[torch.Tensor] = None
    attn_vo_indices_buffer: Optional[torch.Tensor] = None

    # Number of remaining heads per layer (for sparse forward reshape)
    # Shape: [num_prunable_layers]
    remain_num_heads: Optional[torch.Tensor] = None

    # Layer index to buffer index mapping
    # e.g., {3: 0, 4: 1, ...} maps layer_idx to buffer position
    layer_to_buffer_idx: Dict[int, int] = field(default_factory=dict)


class PruningScheduler:
    """Own fixed-shape pruning indices used by the dense/sparse CUDA Graphs."""

    def __init__(
        self,
        static_cfg: StaticPruneConfig,
        device: torch.device,
        dtype: torch.dtype
    ):
        self.static_cfg = static_cfg
        self.device = device
        self.dtype = dtype
        self.state = RuntimeState()
        self._init_buffers()

    def _init_buffers(self):
        """
        Initialize all buffers with fixed shapes.

        Called once during scheduler creation. After this, only in-place
        updates via copy_() are allowed to maintain CUDA Graph compatibility.
        """
        cfg = self.static_cfg
        num_layers = cfg.num_prunable_layers

        # MLP indices buffer: [num_prunable_layers, mlp_keep_dims]
        self.state.mlp_indices_buffer = torch.zeros(
            num_layers, cfg.mlp_keep_dims,
            dtype=torch.int64, device=self.device
        )

        # Attention indices buffers: [num_prunable_layers, attn_keep_dims]
        self.state.attn_qk_indices_buffer = torch.zeros(
            num_layers, cfg.attn_keep_dims,
            dtype=torch.int64, device=self.device
        )
        self.state.attn_vo_indices_buffer = torch.zeros(
            num_layers, cfg.attn_keep_dims,
            dtype=torch.int64, device=self.device
        )

        # Number of heads per layer: [num_prunable_layers]
        self.state.remain_num_heads = torch.full(
            (num_layers,), cfg.attn_keep_heads,
            dtype=torch.int64, device=self.device
        )

        # Build layer index mapping and initialize default indices
        buffer_idx = 0
        for layer_idx in range(cfg.num_layers):
            if layer_idx not in cfg.skip_layers:
                self.state.layer_to_buffer_idx[layer_idx] = buffer_idx

                # Initialize with default sequential indices: [0, 1, 2, ..., keep_dims-1]
                self.state.mlp_indices_buffer[buffer_idx] = torch.arange(
                    cfg.mlp_keep_dims, device=self.device, dtype=torch.int64
                )
                self.state.attn_qk_indices_buffer[buffer_idx] = torch.arange(
                    cfg.attn_keep_dims, device=self.device, dtype=torch.int64
                )
                self.state.attn_vo_indices_buffer[buffer_idx] = torch.arange(
                    cfg.attn_keep_dims, device=self.device, dtype=torch.int64
                )

                buffer_idx += 1

    def get_buffer_idx(self, layer_idx: int) -> Optional[int]:
        """
        Get buffer index for a layer.

        Returns None for skip layers (no pruning buffer).
        """
        return self.state.layer_to_buffer_idx.get(layer_idx)

    def get_mlp_indices(self, buffer_idx: int) -> torch.Tensor:
        """Get MLP indices for a specific buffer position"""
        return self.state.mlp_indices_buffer[buffer_idx]

    def get_attn_indices(self, buffer_idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Get attention indices for a specific buffer position.

        Returns:
            (qk_indices, vo_indices, num_heads)
        """
        return (
            self.state.attn_qk_indices_buffer[buffer_idx],
            self.state.attn_vo_indices_buffer[buffer_idx],
            int(self.state.remain_num_heads[buffer_idx].item())
        )

    def update_mlp_indices(self, buffer_idx: int, indices: torch.Tensor):
        """
        Update MLP indices for a layer (in-place copy).

        Called on metric_stream after metric computation.
        indices must have shape [mlp_keep_dims].
        """
        self.state.mlp_indices_buffer[buffer_idx].copy_(indices)

    def update_attn_indices(
        self,
        buffer_idx: int,
        qk_indices: torch.Tensor,
        vo_indices: torch.Tensor,
        num_heads: int
    ):
        """
        Update attention indices for a layer (in-place copy).

        Called on metric_stream after metric computation.
        qk_indices and vo_indices must have shape [attn_keep_dims].
        """
        self.state.attn_qk_indices_buffer[buffer_idx].copy_(qk_indices)
        self.state.attn_vo_indices_buffer[buffer_idx].copy_(vo_indices)
        self.state.remain_num_heads[buffer_idx] = num_heads

    def update_all_indices(
        self,
        buffer_idx: int,
        mlp_indices: torch.Tensor,
        attn_indices: torch.Tensor,
        num_heads: int
    ):
        """
        Update both MLP and attention indices for a layer.

        Convenience method that combines update_mlp_indices and update_attn_indices.
        """
        self.update_mlp_indices(buffer_idx, mlp_indices)
        self.update_attn_indices(buffer_idx, attn_indices, attn_indices, num_heads)


def create_scheduler_from_model_config(
    model_config,
    mlp_prune_ratio: float = 0.25,
    attn_prune_ratio: float = 0.25,
    skip_layers: Tuple[int, ...] = (0, 1, 2),
    max_seq_len: int = 611,
    history_metric_ratio: float = 0.9,
    device: torch.device = None,
    dtype: torch.dtype = None
) -> PruningScheduler:
    """
    Factory function to create PruningScheduler from a HuggingFace model config.

    Args:
        model_config: LlamaConfig or similar HF config object
        mlp_prune_ratio: Ratio of MLP dimensions to prune
        attn_prune_ratio: Ratio of attention heads to prune
        skip_layers: Layers to skip (no pruning)
        max_seq_len: Maximum sequence length
        history_metric_ratio: Weight assigned to historical metric in the global/local combination
        device: Target device
        dtype: Target dtype

    Returns:
        PruningScheduler instance
    """
    static_cfg = StaticPruneConfig(
        hidden_size=model_config.hidden_size,
        intermediate_size=model_config.intermediate_size,
        num_attention_heads=model_config.num_attention_heads,
        num_key_value_heads=getattr(model_config, 'num_key_value_heads', model_config.num_attention_heads),
        head_dim=getattr(model_config, 'head_dim', model_config.hidden_size // model_config.num_attention_heads),
        num_layers=model_config.num_hidden_layers,
        max_seq_len=max_seq_len,
        mlp_prune_ratio=mlp_prune_ratio,
        attn_prune_ratio=attn_prune_ratio,
        skip_layers=skip_layers,
        history_metric_ratio=history_metric_ratio,
    )

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dtype is None:
        dtype = torch.bfloat16

    return PruningScheduler(static_cfg, device, dtype)
