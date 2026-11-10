import torch
from ..prune_context import prune_context as cfg
from ..module import nearest_multiple, check_skip_layers

BLOCK_SIZE = 64

class HiddenRepresentationPruning():

    def __init__(self, cfg, key, model_config=None):
        self.key = key
        if not isinstance(cfg.config.prune_ratio, float) and len(cfg.config.prune_ratio) == 2:
            if "attention" in self.key:
                self.prune_ratio = cfg.config.prune_ratio[0]
            else:
                self.prune_ratio = cfg.config.prune_ratio[1]
        else:
            self.prune_ratio = cfg.config.prune_ratio
        self.model_config = model_config

    def adjust_prune_ratio(self, prune_ratio, model_config):
        """
        Adjust the per-layer pruning ratio upward to account for skipped layers and match the global ratio.
        """
        if model_config is not None:
            if prune_ratio == 0:
                return 0
            
            num_hidden_layers = model_config.num_hidden_layers
            prune_ratio = round(num_hidden_layers / (num_hidden_layers - (len(cfg.derived.skip_layers))) * prune_ratio, 2) 
            return prune_ratio
        else:
            return cfg.config.prune_ratio
    
    def handle_attn_pad_tokens(self, activation):
        # Handle padding tokens for the complete batch and sequence.
        if cfg.state.pad_tokens is not None:
            cfg.state.pad_tokens = cfg.state.pad_tokens.to(activation.device)
            padding_mask = cfg.state.pad_tokens[:, :activation.shape[1]]

            activation[padding_mask] = 0
            sample_count = (activation.size(0) - torch.sum(padding_mask, dim=0) + 1e-3).to(activation.dtype).unsqueeze(1)
            return activation, sample_count
        else:
            return activation, activation.size(0)
    
    def handle_mlp_pad_tokens(self, activation: torch.Tensor):
        if cfg.state.pad_tokens is not None:
            cfg.state.pad_tokens = cfg.state.pad_tokens.to(activation.device)
            padding_mask = cfg.state.pad_tokens[:, :activation.shape[1]]

            sample_count = (activation.size(0) - torch.sum(padding_mask, dim=0) + 1e-3).to(activation.dtype).unsqueeze(1)
            return activation, sample_count
        else:
            return activation, activation.size(0)

    def cal_attn_prune_metric(self, activation, weight, global_metric_score_distribution=None):
        activation, sample_count = self.handle_attn_pad_tokens(activation)
        
        local_activation_metric = torch.sum(activation**2, dim=0) / sample_count
        local_activation_metric = torch.clamp(local_activation_metric, max=cfg.derived.data_type_max)
        local_activation_metric = torch.sum(local_activation_metric, dim=0)
        if global_metric_score_distribution is not None:
            combined_activation = cfg.config.history_metric_ratio * global_metric_score_distribution + (1 - cfg.config.history_metric_ratio) * local_activation_metric
            combined_activation = combined_activation.clamp(max=cfg.derived.data_type_max)
            metric_scores = torch.linalg.vector_norm((combined_activation.reshape((1,-1)) * torch.pow(weight, 2)).reshape(self.model_config.hidden_size, self.model_config.num_attention_heads, -1), ord=2, dim=(0, 2)).clamp(max=cfg.derived.data_type_max)

            return metric_scores
        else:
            local_activation_metric = local_activation_metric.clamp(max=cfg.derived.data_type_max)
            metric_scores = torch.linalg.vector_norm((local_activation_metric.reshape((1,-1)) * torch.pow(weight, 2)).reshape(self.model_config.hidden_size, self.model_config.num_attention_heads, -1), ord=2, dim=(0, 2)).clamp(max=cfg.derived.data_type_max)
            return metric_scores

    def cal_mlp_prune_metric(
        self, 
        activation, 
        weight, 
        global_metric_score_distribution = None
    ):
        activation, sample_count = self.handle_mlp_pad_tokens(activation)

        local_activation_metric = torch.sum(activation**2, dim=0) / sample_count
        local_activation_metric = torch.clamp(local_activation_metric, max=cfg.derived.data_type_max)
        local_activation_metric = torch.sum(local_activation_metric, dim=0)

        if global_metric_score_distribution is not None:
            combined_activation = cfg.config.history_metric_ratio * global_metric_score_distribution + (1 - cfg.config.history_metric_ratio) * local_activation_metric
            combined_activation = combined_activation.clamp(max=cfg.derived.data_type_max)
            metric_scores = torch.linalg.vector_norm(
                (combined_activation.reshape((1,-1)) * torch.pow(weight, 2)),
                ord=2,
                dim=0
            ).clamp(max=cfg.derived.data_type_max)

            return metric_scores
        else:
            metric_scores = torch.linalg.vector_norm(
                (local_activation_metric.reshape((1,-1)) * torch.pow(weight, 2)),
                ord=2,
                dim=0
            ).clamp(max=cfg.derived.data_type_max)
            return metric_scores

    def cal_attn_calib_prune_metric(self, calib, weight):
        calib = calib.to(weight.device)
        calib = calib.to(torch.float32)
        weight = weight.to(torch.float32)
        calib = calib.clamp(max=cfg.derived.data_type_max)
        metric_scores = torch.linalg.vector_norm((calib.reshape((1,-1)) * torch.pow(weight, 2)).reshape(self.model_config.hidden_size, self.model_config.num_attention_heads, -1), ord=2, dim=(0, 2)).clamp(max=cfg.derived.data_type_max)
    
        return metric_scores

    def cal_mlp_calib_prune_metric(self, calib, weight):
        calib = calib.to(weight.device)
        calib = calib.to(torch.float32)

        weight = weight.to(torch.float32)
        calib = calib.clamp(max=cfg.derived.data_type_max)
        metric_scores = torch.linalg.vector_norm((calib.reshape((1,-1)) * torch.pow(weight, 2)), ord=2, dim=0).clamp(max=cfg.derived.data_type_max)
        
        return metric_scores
    
    def sort_attn_metric(self, metric_scores, num_heads, head_dim, prune_way, prune_module, multiple, pruning_ratio=None):
        if prune_way == None:
            return None, None, num_heads

        prune_ratio = pruning_ratio if pruning_ratio is not None else self.prune_ratio
        if "whole" in prune_way:    
            metric_scores = metric_scores.reshape(num_heads, -1)
            summed_metrics = torch.clamp(metric_scores.sum(dim=-1), max=cfg.derived.data_type_max)
            sorted_value, sorted_indices = torch.sort(summed_metrics, dim=0)
            num_prune_heads = int(prune_ratio * num_heads)
            heads_to_preserve = sorted_indices[num_prune_heads:]
            heads_to_preserve = torch.sort(heads_to_preserve)[0]
            full_indices_to_preserve = (torch.arange(head_dim, device=metric_scores.device) + heads_to_preserve.unsqueeze(1) * head_dim).view(-1)
            num_heads = num_heads - num_prune_heads
            return full_indices_to_preserve, num_heads, heads_to_preserve
        else:
            raise NotImplementedError

    def sort_mlp_metric(self, metric_scores, multiple, pruning_ratio=None):        
        prune_ratio = pruning_ratio if pruning_ratio is not None else self.prune_ratio
        sorted_value, sorted_indices = torch.sort(metric_scores, dim=0)
        num_prune = int(prune_ratio * metric_scores.shape[0])
        num_prune = num_prune if num_prune % BLOCK_SIZE == 0 else (num_prune // BLOCK_SIZE + 1) * BLOCK_SIZE
        num_prune = nearest_multiple(num_prune, metric_scores.shape[0], multiple)
        # return torch.sort(sorted_indices[num_prune:])[0], torch.sort(sorted_indices[:num_prune])[0]
        return torch.sort(sorted_indices[num_prune:])[0], sorted_indices[:num_prune]
        # return sorted_indices[num_prune:], sorted_indices[:num_prune]
    
