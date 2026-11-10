import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_SIMILARITY_THRESHOLD = 0.25
SIMILARITY_WINDOW_SIZE = 5
SIMILARITY_THRESHOLD_PERCENTILE = 5


def reset_eval_prune_state(cfg):
    """Reset run-level pruning counters derived from static config."""
    cfg.state.prune_count = 0
    cfg.state.prune_ratio_new = cfg.config.prune_ratio
    return cfg


def reset_episode_prune_state(cfg, similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD):
    """Reset frame-level pruning state at the beginning of each rollout."""
    cfg.state.current_action_step = 0
    cfg.state.visual_embedding_list = []
    cfg.state.pre_is_prune = 0
    cfg.state.similarity_history = []
    cfg.state.similarity_threshold = similarity_threshold
    cfg.state.prune_ratio_new = cfg.config.prune_ratio
    cfg.state.recent_prune_flags = []
    cfg.state.already_prune_attn = False
    cfg.state.already_prune_mlp = False
    return cfg


def calc_similarity(f1: torch.Tensor, f2: torch.Tensor):
    return torch.mean(F.cosine_similarity(f1, f2, dim=1))


def adjust_similarity_threshold(cfg, current_similarity_score):
    cfg.state.similarity_history.append(current_similarity_score.item())

    if len(cfg.state.similarity_history) > SIMILARITY_WINDOW_SIZE:
        cfg.state.similarity_history.pop(0)

    if len(cfg.state.similarity_history) < SIMILARITY_WINDOW_SIZE:
        return cfg.state.similarity_threshold

    cfg.state.similarity_threshold = np.percentile(
        np.array(cfg.state.similarity_history),
        SIMILARITY_THRESHOLD_PERCENTILE,
    )
    return cfg.state.similarity_threshold


def update_similarity_prune_state(cfg, projected_patch_embeddings):
    """Update cfg.state.is_prune from current visual embeddings and recent history."""
    cfg.state.visual_embedding_list.append(projected_patch_embeddings[0])

    if not hasattr(cfg.state, "recent_prune_flags"):
        cfg.state.recent_prune_flags = []

    if len(cfg.state.visual_embedding_list) == 1:
        cfg.state.is_prune = True
        cfg.state.already_prune_attn = False
        cfg.state.already_prune_mlp = False
    else:
        similarity_with_last_prune = calc_similarity(
            cfg.state.visual_embedding_list[-1],
            cfg.state.visual_embedding_list[-2],
        )
        recently_pruned = len(cfg.state.recent_prune_flags) >= 2 and any(cfg.state.recent_prune_flags[-2:])
        cfg.state.is_prune = similarity_with_last_prune < cfg.state.similarity_threshold and not recently_pruned

        if cfg.state.is_prune:
            cfg.state.already_prune_attn = False
            cfg.state.already_prune_mlp = False

        cfg.state.recent_prune_flags.append(cfg.state.is_prune)
        if len(cfg.state.recent_prune_flags) > 2:
            cfg.state.recent_prune_flags.pop(0)

    if cfg.state.is_prune:
        cfg.state.pre_is_prune = cfg.state.current_action_step
        cfg.state.prune_count += 1
        print("prune")

    if len(cfg.state.visual_embedding_list) > 1:
        similarity_with_previous_frame = calc_similarity(
            cfg.state.visual_embedding_list[-1],
            cfg.state.visual_embedding_list[-2],
        )
        adjust_similarity_threshold(cfg, similarity_with_previous_frame)

    return cfg.state.is_prune
