import subprocess

import torch

from ..prune_context import prune_context as cfg


def _resolve_hardware_config():
    cudatoolkit_version = float(torch.version.cuda)
    cudnn_version = float(torch.backends.cudnn.version())

    gpu_name = ""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=gpu_name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        )
        gpu_name = result.stdout.strip()
        print(f"GPU Name: {gpu_name}")
    except Exception as exc:
        print(f"An error occurred: {exc}")
    cfg.derived.data_type = torch.float16
    cfg.derived.data_type_max = torch.finfo(cfg.derived.data_type).max

    if cfg.derived.data_type in (torch.float16, torch.float32):
        if cudatoolkit_version >= 11 and cudnn_version >= 7630:
            cfg.derived.tc_multiple = 8 if gpu_name == "NVIDIA GeForce RTX 4090" else 64
        else:
            cfg.derived.tc_multiple = 64


def _normalize_prune_config():
    model_name = cfg.config.model_name
    cfg.config.batch_size = int(cfg.config.batch_size)
    cfg.config.max_seq_len = int(cfg.config.max_seq_len)
    cfg.derived.ema_momentum = 0.95

    if isinstance(cfg.config.cust_tgt_modules, str):
        cfg.config.cust_tgt_modules = cfg.config.cust_tgt_modules.split("+")
    if "llama" in model_name and cfg.config.cust_tgt_modules != ["default"]:
        cfg.config.cust_tgt_modules = [module.replace("-", "_") for module in cfg.config.cust_tgt_modules]
    elif "opt" in model_name and cfg.config.cust_tgt_modules != ["default"]:
        cfg.config.cust_tgt_modules = [module.replace("-", "_") for module in cfg.config.cust_tgt_modules]
    elif "llama" in model_name:
        cfg.config.cust_tgt_modules = TRANSFORMERS_MODELS_TO_ERI_TARGET_MODULES_MAPPING["llama"]
    elif "opt" in model_name:
        cfg.config.cust_tgt_modules = TRANSFORMERS_MODELS_TO_ERI_TARGET_MODULES_MAPPING["opt"]
    else:
        raise ValueError("Not valid model name")
def _initialize_runtime_state():
    cfg.state.calibration_stage = False
    cfg.derived.skip_layers = [0, 1, 2]
    cfg.state.cur_batch_index = -1
    cfg.derived.qk_prune_way = "whole"
    cfg.derived.vo_prune_way = "whole"

    cfg.state.pad_tokens = None


def process_control():
    _resolve_hardware_config()
    _normalize_prune_config()
    _initialize_runtime_state()
    print("cfg: ", cfg, flush=True)


TRANSFORMERS_MODELS_TO_ERI_TARGET_MODULES_MAPPING = {
    "opt": ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
    "llama": ["q_proj", "v_proj", "o_proj", "k_proj", "gate_proj", "up_proj", "down_proj"],
}
