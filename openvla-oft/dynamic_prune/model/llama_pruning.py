import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..prune_context import prune_context as cfg
from ..module import check_skip_layers
from .triton_linear import (
    up_channel_level_indices_linear, 
    down_channel_level_indices_linear, 
    down_channel_transpose_level_indices_linear, 
)

class LlamaPruningModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.forward = self.model.forward
        self.add_pruner()

    def add_pruner(self):
        self._find_and_replace()
        mark_no_trainable(self.model)
    
    def _create_new_module(self, target, key): 
        kwargs = {
            "key": key,
            "dev": target.weight.device,
            "is_GQA": self.model.language_model.config.num_key_value_heads != self.model.language_model.config.num_attention_heads 
        }
        
        if not isinstance(target, torch.nn.Linear):
            raise ValueError(
                f"Target module {target} is not supported. "
                f"Currently, only `torch.nn.Linear` is supported."
            )
        new_module = Linear(target.in_features, target.out_features, bias=False, **kwargs)

        return new_module


    def _find_and_replace(self):
        is_target_modules_in_base_model = False
        key_list = [key for key, _ in self.model.named_modules()]
        target_modules = cfg.config.cust_tgt_modules
        for key in key_list:
            if not _check_target_module_exists(target_modules, key):
                continue
            
            if check_skip_layers(key):
                continue

            is_target_modules_in_base_model = True
            parent, target, target_name = _get_submodules(self.model, key)
            
            new_module = self._create_new_module(target, key)
            
            self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            print(
                f"Target modules {target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        use_transposed_down_projection = (
            cfg.config.use_triton
            and cfg.config.transpose_down_linear
            and child_name in {"o_proj", "down_proj"}
        )
        if use_transposed_down_projection:
            new_module.weight = nn.Parameter(old_module.weight.permute(1,0).contiguous())
        else: 
            new_module.weight = old_module.weight
        new_module.weight.requires_grad = False
        new_module.device = old_module.weight.device
        new_module.is_pruned = True

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)


def mark_no_trainable(model: nn.Module) -> None:
    for _, parameter in model.named_parameters():
        parameter.requires_grad = False

def _get_submodules(model, key):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name

def _check_target_module_exists(target_modules, key):
    if isinstance(target_modules, str):
        target_module_found = re.fullmatch(target_modules, key)
    else:
        target_module_found = any(key.endswith(target_key) for target_key in target_modules)
    return target_module_found

class EriLayer:
    def __init__(self, in_features: int, out_features: int, **kwargs):
        self.key = kwargs["key"]

    def extract_in_dim_weight(self, weight, indices):
        return  weight[:, indices]
           
    def extract_out_dim_weight(self, weight, indices):
        return  weight[indices, :]


class ChannelMetricHistory:
    """Per-input-channel running metric state shared by calibration and inference."""

    def __init__(self, num_channels: int, device):
        self.value = torch.zeros(num_channels, device=device, dtype=torch.float32)
        self.nsamples = torch.zeros(num_channels, device=device, dtype=torch.int32)

    def to(self, device):
        self.value = self.value.to(device)
        self.nsamples = self.nsamples.to(device)
        return self

    def to_dtype(self, dtype):
        self.value = self.value.to(dtype)

    def update_ema(self, inp, update_indices, aggregate, momentum, data_type_max):
        device = inp.device
        self.to(device)
        update_indices = update_indices.to(device)
        self.value[update_indices] *= momentum
        current_metric = aggregate(inp, data_type_max)
        self.value[update_indices] += (1 - momentum) * current_metric

    def update_running_mean(self, inp, update_indices, sample_count, aggregate, data_type_max):
        device = inp.device
        self.to(device)
        update_indices = update_indices.to(device)
        denominator = self.nsamples[update_indices] + sample_count
        self.value[update_indices] *= self.nsamples[update_indices] / denominator
        current_metric = aggregate(inp, data_type_max)
        self.value[update_indices] += current_metric * sample_count / denominator
        self.nsamples[update_indices] += sample_count


class Linear(nn.Linear, EriLayer):
    def __init__(
        self,
        in_features,
        out_features,
        bias,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, bias=bias)
        EriLayer.__init__(self, in_features=in_features, out_features=out_features, **kwargs)
        self.is_GQA = kwargs["is_GQA"]
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False
        self.layer_type = "linear"
        self.in_features = in_features
        if ("o_proj" in self.key or "down_proj" in self.key):
            self.metric_history = ChannelMetricHistory(in_features, self.weight.data.device)

    @property
    def nsamples(self):
        return self.metric_history.nsamples

    @nsamples.setter
    def nsamples(self, value):
        self.metric_history.nsamples = value

    @property
    def scaler_inp(self):
        return self.metric_history.value

    @scaler_inp.setter
    def scaler_inp(self, value):
        self.metric_history.value = value

    @staticmethod
    def _aggregate_squared_activation(inp, sample_count, clamp):
        squared_norm = torch.linalg.vector_norm(inp, ord=2, dim=0) ** 2
        if clamp:
            squared_norm = torch.clamp(squared_norm, max=cfg.derived.data_type_max)
        per_token_metric = squared_norm / sample_count
        if clamp:
            per_token_metric = torch.clamp(per_token_metric, max=cfg.derived.data_type_max)
        return torch.sum(per_token_metric, dim=0)

    def update_global_metric_score_distribution_ema(self, inp, update_indices):
        if not cfg.config.history_update:
            return
        if len(inp.shape) == 2:
            raise ValueError(f"Input shape {inp.shape} is not supported. Please provide a 3D tensor.")
        momentum = cfg.derived.ema_momentum
        if cfg.state.calibration_stage:
            sample_count = inp.shape[0]
            clamp = False
        elif cfg.state.pad_tokens is not None:
            cfg.state.nonpad_tokens_denominator = cfg.state.nonpad_tokens_denominator.to(inp.device)
            sample_count = cfg.state.nonpad_tokens_denominator[:inp.shape[1]]
            clamp = True
        else:
            sample_count = inp.shape[0]
            clamp = True
        self.metric_history.update_ema(
            inp, update_indices,
            lambda value, _: self._aggregate_squared_activation(value, sample_count, clamp),
            momentum, cfg.derived.data_type_max,
        )

    def update_global_metric_score_distribution(self, inp, update_indices):
        if not cfg.config.history_update:
            return
        
        if len(inp.shape) == 2:
            raise ValueError(f"Input shape {inp.shape} is not supported. Please provide a 3D tensor.")
        
        if cfg.state.calibration_stage:
            sample_count = inp.shape[0]
            aggregate = lambda value, _: torch.sum(torch.linalg.vector_norm(value, ord=2, dim=0) ** 2, dim=0)
        else:
            if cfg.state.pad_tokens is not None:
                cfg.state.nonpad_tokens_denominator = cfg.state.nonpad_tokens_denominator.to(inp.device)
                sample_count = cfg.state.nonpad_tokens_denominator[:inp.shape[1]].sum()
                denominator_for_tokens = cfg.state.nonpad_tokens_denominator[:inp.shape[1]]
                aggregate = lambda value, _: self._aggregate_squared_activation(value, denominator_for_tokens, clamp=True)
            else:
                sample_count = inp.shape[0]
                aggregate = lambda value, _: self._aggregate_squared_activation(value, sample_count, clamp=True)
            self.metric_history.to_dtype(cfg.derived.data_type)
        self.metric_history.update_running_mean(inp, update_indices, sample_count, aggregate, cfg.derived.data_type_max)

    def get_global_metric_score_distribution(self):
        return self.scaler_inp

    def free(self):
        if hasattr(self, "scaler_inp"):
            self.scaler_inp = None
        if hasattr(self, "scaler_row"):
            self.scaler_row = None
        torch.cuda.empty_cache()  

    def return_global_metric_info(self):
        if ("o_proj" in self.key or "down_proj" in self.key):
            return {"nsamples": self.nsamples, "scaler_inp": self.scaler_inp}
        else:
            return None

    def set_global_metric_to_data_type(self):
        if ("o_proj" in self.key or "down_proj" in self.key):
            self.metric_history.to_dtype(cfg.derived.data_type)
    
    def forward(self, x: torch.Tensor, **kwargs):
        if cfg.config.use_triton: 
            return self.triton_forward(x, **kwargs)
        else:
            return self.torch_forward(x, **kwargs)
    
    def triton_forward(self, x: torch.Tensor, **kwargs):
        with torch.no_grad():
            previous_dtype = x.dtype
            if cfg.state.calibration_stage:
                if "o_proj" in self.key or "down_proj" in self.key:
                    self.update_global_metric_score_distribution(x, torch.arange(self.in_features, dtype=torch.long, device=x.device))
                    if cfg.config.transpose_down_linear: 
                        result = torch.matmul(x, self.weight)
                    else: 
                        result = F.linear(x, self.weight)
                else: 
                    result = F.linear(x, self.weight)
                result = result.to(previous_dtype)
                return result
            else:
                if cfg.state.cur_batch_index == 0:
                    self.set_global_metric_to_data_type()
                    
                if kwargs.get("cal_mlp_metric_scores", False):
                    result = F.linear(x, self.weight, bias=None)
                    result = result.to(previous_dtype)
                    return result
                elif kwargs.get("cal_attn_metric_scores", False):
                    result = F.linear(x, self.weight, bias=None)
                    result = result.to(previous_dtype)
                    return result

                # dynamic prune
                if cfg.config.mode == "dynamic":
                    # update metric score
                    if ("o_proj" in self.key or "down_proj" in self.key) and cfg.state.is_prune:
                        if "in_dim_indices" in kwargs:
                            self.update_global_metric_score_distribution_ema(x, kwargs["in_dim_indices"])
                        else:
                            self.update_global_metric_score_distribution_ema(x, torch.arange(self.in_features, dtype=torch.long, device=x.device))
                    if "out_dim_indices" in kwargs:
                        result = up_channel_level_indices_linear(x, self.weight, kwargs["out_dim_indices"])
                    elif "in_dim_indices" in kwargs:
                        if cfg.config.transpose_down_linear: 
                            result = down_channel_transpose_level_indices_linear(x, self.weight, kwargs["in_dim_indices"])
                        else: 
                            result = down_channel_level_indices_linear(x, self.weight, kwargs["in_dim_indices"])
                    elif cfg.config.transpose_down_linear and ("o_proj" in self.key or "down_proj" in self.key):
                        result = torch.matmul(x, self.weight)
                    else:
                        result = F.linear(x, self.weight, bias=None)
                    result = result.to(previous_dtype)
                return result
    
    def torch_forward(self, x: torch.Tensor, **kwargs):
        with torch.no_grad():
            previous_dtype = x.dtype
            if cfg.state.calibration_stage:
                if "o_proj" in self.key or "down_proj" in self.key:
                    self.update_global_metric_score_distribution(x, torch.arange(self.in_features, dtype=torch.long, device=x.device))
                result = F.linear(x, self.weight, bias=None)
                result = result.to(previous_dtype)
                return result
            else:
                if cfg.state.cur_batch_index == 0:
                    self.set_global_metric_to_data_type()
                    
                if kwargs.get("cal_mlp_metric_scores", False):
                    result = F.linear(x, self.weight, bias=None)
                    result = result.to(previous_dtype)
                    return result
                elif kwargs.get("cal_attn_metric_scores", False):
                    result = F.linear(x, self.weight, bias=None)
                    result = result.to(previous_dtype)
                    return result
                
                # dynamic prune
                if cfg.config.mode == "dynamic":
                    weight = self.weight
                    # update metric score
                    if ("o_proj" in self.key or "down_proj" in self.key) and cfg.state.is_prune:
                        if "in_dim_indices" in kwargs:
                            self.update_global_metric_score_distribution_ema(x, kwargs["in_dim_indices"])
                        else:
                            self.update_global_metric_score_distribution_ema(x, torch.arange(self.in_features, dtype=torch.long, device=x.device))
                    # up linear
                    if "out_dim_indices" in kwargs:
                        if ("k_proj" in self.key or "v_proj" in self.key) and self.is_GQA:
                            weight = weight
                        else:
                            weight = self.extract_out_dim_weight(weight, kwargs["out_dim_indices"])
                        result = F.linear(x, weight, bias=None)
                    # down linear
                    elif "in_dim_indices" in kwargs:
                        weight = self.extract_in_dim_weight(weight, kwargs["in_dim_indices"])
                        result = F.linear(x, weight, bias=None)
                    else:
                        result = F.linear(x, weight, bias=None)
                    result = result.to(previous_dtype)
                return result
