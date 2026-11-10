from ..prune_context import prune_context as cfg


def make_prune_model(model):
    from .llama_pruning import LlamaPruningModel

    if 'llama' in cfg.config.model_name:
        return LlamaPruningModel(model)
    raise ValueError('Only LLaMA-backed OpenVLA pruning is supported')
