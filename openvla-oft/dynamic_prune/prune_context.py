from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class PruneConfig:
    model_name: str = "llama-2-7b"
    batch_size: int = 1
    max_seq_len: int = 611
    mode: str = "dynamic"
    cust_tgt_modules: Any = "default"
    prune_ratio: Any = field(default_factory=lambda: [0.25, 0.25])
    attn_prune_ratio: float = 0.25
    mlp_prune_ratio: float = 0.25
    prune_flag: bool = False
    async_prune: bool = True
    history_metric_ratio: float = 0.9
    history_update: bool = True
    use_triton: bool = True
    transpose_down_linear: bool = True
    use_fused_kernel: bool = True
    nbatch_enabled: bool = True


@dataclass
class PruneDerivedConfig:
    data_type: Any = None
    data_type_max: Any = None
    tc_multiple: int = 64
    skip_layers: List[int] = field(default_factory=lambda: [0, 1, 2])
    qk_prune_way: str = "whole"
    vo_prune_way: str = "whole"
    # Fixed exponential moving-average momentum used by dynamic metrics.
    ema_momentum: float = 0.95
    dataset_name: Optional[str] = None
    tokenizer: Any = None


@dataclass
class PruneRuntimeState:
    calibration_stage: bool = False
    cur_batch_index: int = -1
    current_action_step: int = 0
    is_prune: bool = False
    prune_count: int = 0
    visual_embedding_list: List[Any] = field(default_factory=list)
    pre_is_prune: int = 0
    similarity_history: List[float] = field(default_factory=list)
    similarity_threshold: float = 0.25
    recent_prune_flags: List[bool] = field(default_factory=list)
    already_prune_attn: bool = False
    already_prune_mlp: bool = False
    prune_ratio_new: Any = None
    pad_tokens: Any = None
    nonpad_tokens_denominator: Any = None
    first_one_indices: Any = None


@dataclass
class PruneStats:
    real_prune_ratio: Any = None
    real_attn_keep_ratio: Any = None
    velocity_fields_std: List[Any] = field(default_factory=list)
    lm_out_eps: Any = None


@dataclass
class PruneContext:
    config: PruneConfig = field(default_factory=PruneConfig)
    derived: PruneDerivedConfig = field(default_factory=PruneDerivedConfig)
    state: PruneRuntimeState = field(default_factory=PruneRuntimeState)
    stats: PruneStats = field(default_factory=PruneStats)


prune_context = PruneContext()
