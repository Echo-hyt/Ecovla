import re

from ..prune_context import prune_context as cfg


def check_skip_layers(name):
    layer_order_matches = re.findall(r"\d+", name)
    if not layer_order_matches:
        raise ValueError(f"Layer order not found in the layer key {name}")

    return int(layer_order_matches[0]) in cfg.derived.skip_layers


def nearest_multiple(num_prune, total, multiple):
    remain = (total - num_prune) % multiple
    if remain == 0:
        return num_prune
    return num_prune - (multiple - remain)
