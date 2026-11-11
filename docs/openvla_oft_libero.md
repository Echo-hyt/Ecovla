# Running OpenVLA-OFT on LIBERO

This guide explains how to run OpenVLA-OFT evaluations on LIBERO using this repository.

## 1. Set Up the Environment

First, follow the setup instructions in the [official OpenVLA-OFT repository](https://github.com/moojink/openvla-oft):

- Follow [SETUP.md](https://github.com/moojink/openvla-oft/blob/main/SETUP.md) to create and activate a Conda environment and install dependencies such as PyTorch and Flash Attention.
- Follow the [Setup section of LIBERO.md](https://github.com/moojink/openvla-oft/blob/main/LIBERO.md#setup) to install the LIBERO simulation environment and evaluation dependencies.

After completing the setup, install this repository's code and the bundled Transformers fork in the same environment. Run the following commands from the repository root:

```bash
cd openvla-oft
python -m pip install -e .
python -m pip install -e ./transformers-openvla-oft
python -m pip install -r experiments/robot/libero/libero_requirements.txt
```

## 2. Prepare the Model Checkpoint

Prepare an OpenVLA-OFT checkpoint that supports the LIBERO task suite.

When running on another machine, replace `--pretrained_checkpoint` with the actual checkpoint path.

## 3. Run the Evaluation

Activate the configured environment and run the following command:

```bash
cd openvla-oft

CUDA_VISIBLE_DEVICES=gpu-id python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint /path/to/openvla-oft/ckpt \
  --prune True \
  --num_trials_per_task 50 \
  --attn_prune_ratio 0.40 \
  --mlp_prune_ratio 0.40 \
  --task_suite_name libero_10
```
