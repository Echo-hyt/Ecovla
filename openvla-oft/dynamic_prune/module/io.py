import os
import re
from pathlib import Path

import torch

from ..prune_context import prune_context


def makedir_exist_ok(path):
    os.makedirs(path, exist_ok=True)


def _calib_info_dir():
    return Path(__file__).resolve().parents[1] / "output" / "calibration"


def _safe_path_component(value, default):
    value = default if value is None else str(value).strip()
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value)
    return value.strip(".-") or default


def _context_value(section_name, key, default=None):
    section = getattr(prune_context, section_name)
    value = getattr(section, key, default)
    return default if value is None else value


def _calib_info_name():
    dataset = _safe_path_component(
        _context_value("derived", "dataset_name"), "unknown_dataset"
    )
    model = _safe_path_component(
        _context_value("config", "model_name"), "unknown_model"
    )
    max_seq_len = int(_context_value("config", "max_seq_len", 0))
    return _calib_info_dir() / dataset / model / f"seq_{max_seq_len}.pt"


def get_calib_saving_info_path():
    return str(_calib_info_name())


def check_calib_saving_info():
    return os.path.exists(get_calib_saving_info_path())


def load(path):
    return torch.load(path, map_location=lambda storage, loc: storage)


def save(input_data, path):
    makedir_exist_ok(os.path.dirname(path))
    torch.save(input_data, path)


def load_calib_saving_info(model):
    calibration_data = load(get_calib_saving_info_path())
    named_modules = dict(model.named_modules())

    for key, value in calibration_data.items():
        module_name, attr_name = key.rsplit("+", 1)
        print(f"Applying calibration data to {module_name}.{attr_name}...")
        module = named_modules[module_name]
        setattr(module, attr_name, value.to(module.weight.device))

    print("Calibration data applied to the model successfully.")


def save_calib_info(model):
    calib_info = {}
    for name, module in model.named_modules():
        if not hasattr(module, "return_global_metric_info"):
            continue

        module_info = module.return_global_metric_info()
        if module_info is None:
            continue

        for key, value in module_info.items():
            calib_info[f"{name}+{key}"] = value

    save(calib_info, get_calib_saving_info_path())
    print("Calibration data saved successfully.")
