"""
run_libero_eval.py

Evaluates a trained policy in a LIBERO simulation benchmark task suite.
"""

import json
import logging
import os
import sys
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union
import random
import draccus
import numpy as np
import tqdm
from libero.libero import benchmark
import torch
from torch.utils.data import DataLoader
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
import wandb
from dynamic_prune.model import make_prune_model
from dynamic_prune.prune_context import prune_context as cfg
from dynamic_prune.module import (
    process_control,
    reset_episode_prune_state,
    reset_eval_prune_state,
    check_calib_saving_info,
    load_calib_saving_info,
    save_calib_info,
)
import time
from dynamic_prune.pruning_scheduler import create_scheduler_from_model_config

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK

# Define task suite constants
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,  # longest training demo has 193 steps
    TaskSuite.LIBERO_OBJECT: 280,  # longest training demo has 254 steps
    TaskSuite.LIBERO_GOAL: 300,  # longest training demo has 270 steps
    TaskSuite.LIBERO_10: 520,  # longest training demo has 505 steps
    TaskSuite.LIBERO_90: 400,  # longest training demo has 373 steps
}


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

os.environ["PYTHONUNBUFFERED"] = "1"

@dataclass
class GenerateConfig:
    # fmt: off
    
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = "/data/shared_workspace/huangyuting/ckpts/openvla/openvla-7b-oft-finetuned-libero-spatial-object-goal-10"     # Pretrained checkpoint path
    # Static pruning parameters.
    # ===========================================================
    batch_size: int = 1 
    image_aug: bool = True
    calibration_dataset_name: str = None
    calibration_nsample: int = 1024
    data_root_dir: Path = Path("/data/shared_workspace/tangzhipeng/data/modified_libero_rlds/")
    shuffle_buffer_size: int = 100_000
    # ==============================================================

    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
    num_diffusion_steps_inference: int = 50          # (When `diffusion==True`) Number of diffusion steps used for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy

    lora_rank: int = 32                              # Rank of LoRA weight matrix (MAKE SURE THIS MATCHES TRAINING!)

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_10"               # Task suite
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50    # Number of rollouts per task
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project

    seed: int = 7                                    # Random Seed (for reproducibility)

    # prune control
    prune: bool = False
    model_name: str = "llama-2-7b"
    max_seq_len: int = 611
    attn_prune_ratio: float = 0.25
    mlp_prune_ratio: float = 0.25
    mode: str = "dynamic"
    cust_tgt_modules: str = "default"
    async_prune: bool = True
    use_cuda_graphs: bool = True
    history_metric_ratio: float = 0.9
    history_update: bool = True 
    use_triton: bool = True
    transpose_down_linear: bool = True
    use_fused_kernel: bool = True


def validate_config(generate_cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert generate_cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    if "image_aug" in str(generate_cfg.pretrained_checkpoint):
        assert generate_cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (generate_cfg.load_in_8bit and generate_cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Validate task suite
    assert generate_cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {generate_cfg.task_suite_name}"


def initialize_model(generate_cfg: GenerateConfig):
    """Initialize model and associated components."""
    # Load model
    model = get_model(generate_cfg)

    # Load proprio projector if needed
    proprio_projector = None
    if generate_cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            generate_cfg,
            model.llm_dim,
            proprio_dim=8,  # 8-dimensional proprio for LIBERO
        )

    # Load action head if needed
    action_head = None
    if generate_cfg.use_l1_regression or generate_cfg.use_diffusion:
        action_head = get_action_head(generate_cfg, model.llm_dim)

    # Load noisy action projector if using diffusion
    noisy_action_projector = None
    if generate_cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(generate_cfg, model.llm_dim)

    # Get OpenVLA processor if needed
    processor = None
    if generate_cfg.model_family == "openvla":
        
        processor = get_processor(generate_cfg)
        # processor.tokenizer.model_max_length = 128
        processor.tokenizer.model_max_length = 40
        check_unnorm_key(generate_cfg, model)

    return model, action_head, proprio_projector, noisy_action_projector, processor


def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    """Check that the model contains the action un-normalization key."""
    # Initialize unnorm_key
    unnorm_key = cfg.task_suite_name

    # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
    # with the suffix "_no_noops" in the dataset name)
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"
    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"

    # Set the unnorm_key in cfg
    cfg.unnorm_key = unnorm_key


def setup_logging(generate_cfg: GenerateConfig):
    """Set up logging to file and optionally to wandb."""
    # Create run ID
    run_id = f"EVAL-{generate_cfg.task_suite_name}-{generate_cfg.model_family}-{DATE_TIME}"
    if generate_cfg.run_id_note is not None:
        run_id += f"--{generate_cfg.run_id_note}"

    # Set up local logging
    os.makedirs(generate_cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(generate_cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging if enabled
    if generate_cfg.use_wandb:
        wandb.init(
            entity=generate_cfg.wandb_entity,
            project=generate_cfg.wandb_project,
            name=run_id,
        )

    return log_file, local_log_filepath, run_id


def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def load_initial_states(generate_cfg: GenerateConfig, task_suite, task_id: int, log_file=None):
    """Load initial states for the given task."""
    # Get default initial states
    initial_states = task_suite.get_task_init_states(task_id)

    # If using custom initial states, load them from file
    if generate_cfg.initial_states_path != "DEFAULT":
        with open(generate_cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {generate_cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    else:
        log_message("Using default initial states", log_file)
        return initial_states, None


def prepare_observation(obs, resize_size):
    """Prepare observation for policy input."""
    # Get preprocessed images
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)

    # Resize images to size expected by model
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

    # Prepare observations dict
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }

    return observation, img  # Return both processed observation and original image for replay


def process_action(action, model_family):
    """Process action before sending to environment."""
    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
    action = normalize_gripper_action(action, binarize=True)

    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    if model_family == "openvla":
        action = invert_gripper_action(action)

    return action


def run_episode(
    generate_cfg: GenerateConfig,
    env,
    task_description: str,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    initial_state=None,
    log_file=None,
):
    """Run a single episode in the environment."""
    # Reset environment
    start_time = time.time()
    env.reset()

    # Set initial state if provided
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    # Initialize action queue
    if generate_cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        print(f"WARNING: generate_cfg.num_open_loop_steps ({generate_cfg.num_open_loop_steps}) does not match the NUM_ACTIONS_CHUNK "
              f"({NUM_ACTIONS_CHUNK}) constant defined in prismatic.vla.constants! For best performance (in terms of "
               "both speed and success rate), we recommend executing the full action chunk.")
    action_queue = deque(maxlen=generate_cfg.num_open_loop_steps)

    t = 0
    replay_images = []
    max_steps = TASK_MAX_STEPS[generate_cfg.task_suite_name]

    # Run episode
    success = False
    reset_episode_prune_state(cfg)

    while t < max_steps + generate_cfg.num_steps_wait:
        # Do nothing for the first few timesteps to let objects stabilize
        if t < generate_cfg.num_steps_wait:
            obs, reward, done, info = env.step(get_libero_dummy_action(generate_cfg.model_family))
            t += 1
            continue

        # Prepare observation
        observation, img = prepare_observation(obs, resize_size)

        replay_images.append(img)

        if len(action_queue) == 0:
            output = get_action(
                generate_cfg,
                model,
                observation,
                task_description,
                processor=processor,
                action_head=action_head,
                proprio_projector=proprio_projector,
                noisy_action_projector=noisy_action_projector,
                use_film=generate_cfg.use_film,
            )
            action_queue.extend(output["actions"][:generate_cfg.num_open_loop_steps])
            cfg.state.cur_batch_index = cfg.state.cur_batch_index+1
            cfg.state.current_action_step += 1

        # Get action from queue
        action = action_queue.popleft()

        # Process action
        action = process_action(action, generate_cfg.model_family)
        obs, reward, done, info = env.step(action.tolist())
        if done:
            success = True
            break
        t += 1
    end_time = time.time()
    episode_time = end_time - start_time
    print(f"[Episode Time] Duration: {episode_time:.2f} seconds") 

    return success, replay_images


def run_task(
    generate_cfg: GenerateConfig,
    task_suite,
    task_id: int,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    total_episodes=0,
    total_successes=0,
    log_file=None,
):
    """Run evaluation for a single task."""
    # Get task
    task = task_suite.get_task(task_id)

    # Get initial states
    initial_states, all_initial_states = load_initial_states(generate_cfg, task_suite, task_id, log_file)

    # Initialize environment and get task description
    env, task_description = get_libero_env(task, generate_cfg.model_family, resolution=generate_cfg.env_img_res)

    # Start episodes
    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(generate_cfg.num_trials_per_task)):
        log_message(f"\nTask: {task_description}", log_file)

        # Handle initial state
        if generate_cfg.initial_states_path == "DEFAULT":
            # Use default initial state
            initial_state = initial_states[episode_idx]
        else:
            # Get keys for fetching initial episode state from JSON
            initial_states_task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{episode_idx}"

            # Skip episode if expert demonstration failed to complete the task
            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                log_message(f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!", log_file)
                continue

            # Get initial state
            initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])

        log_message(f"Starting episode {task_episodes + 1}...", log_file)

        # Run episode
        success, replay_images = run_episode(
            generate_cfg,
            env,
            task_description,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            initial_state,
            log_file,
        )

        # Update counters
        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1

        # Save replay video
        save_rollout_video(
            replay_images, total_episodes, success=success, task_description=task_description, log_file=log_file
        )

        # Log results
        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes completed so far: {total_episodes}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)

    # Log task results
    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    log_message(f"Current task success rate: {task_success_rate}", log_file)
    log_message(f"Current total success rate: {total_success_rate}", log_file)

    # Log to wandb if enabled
    if generate_cfg.use_wandb:
        wandb.log(
            {
                f"success_rate/{task_description}": task_success_rate,
                f"num_episodes/{task_description}": task_episodes,
            }
        )

    return total_episodes, total_successes


def run_calibration(
    generate_cfg: GenerateConfig, 
    model, 
    data_loader, 
    proprio_projector, 
    device
): 
    # rng = random.Random(44)
    # calibration_data_idx_list = sorted(rng.sample(range(0, len(data_loader)), generate_cfg.calibration_nsample))
    calibration_data_idx_list = sorted(random.sample(range(0, len(data_loader)), generate_cfg.calibration_nsample))
    with torch.no_grad():
        model.eval()
        count = 0
        for i, input in enumerate(data_loader): 
            if i != calibration_data_idx_list[count]: 
                continue
            model(
                    input_ids=input["input_ids"].to(device),
                    attention_mask=input["attention_mask"].to(device),
                    pixel_values=input["pixel_values"].to(torch.bfloat16).to(device),
                    labels=input["labels"].to(device),
                    output_hidden_states=True,
                    proprio=input["proprio"].to(torch.bfloat16).to(device) if generate_cfg.use_proprio else None,
                    proprio_projector=proprio_projector if generate_cfg.use_proprio else None,
                    use_film=generate_cfg.use_film,
                )
            count += 1
            if count >= generate_cfg.calibration_nsample:
                break
    return


def apply_generate_config(generate_cfg: GenerateConfig) -> None:
    """Copy command-line pruning parameters into the runtime config."""
    cfg.config.model_name = generate_cfg.model_name
    cfg.config.batch_size = generate_cfg.batch_size
    cfg.config.max_seq_len = generate_cfg.max_seq_len
    cfg.config.attn_prune_ratio = generate_cfg.attn_prune_ratio
    cfg.config.mlp_prune_ratio = generate_cfg.mlp_prune_ratio
    cfg.config.prune_ratio = [generate_cfg.attn_prune_ratio, generate_cfg.mlp_prune_ratio]
    cfg.config.mode = generate_cfg.mode
    cfg.config.cust_tgt_modules = generate_cfg.cust_tgt_modules
    cfg.config.prune_flag = generate_cfg.prune
    cfg.config.async_prune = generate_cfg.async_prune
    cfg.config.use_triton = generate_cfg.use_triton
    cfg.config.transpose_down_linear = generate_cfg.transpose_down_linear
    cfg.config.use_fused_kernel = generate_cfg.use_fused_kernel
    cfg.config.history_metric_ratio = generate_cfg.history_metric_ratio
    cfg.config.history_update = generate_cfg.history_update


def initialize_cuda_graphs(generate_cfg: GenerateConfig, model) -> None:
    if not (generate_cfg.use_cuda_graphs and generate_cfg.prune):
        return

    logger.info("Initializing CUDA Graph optimized pruning scheduler...")
    llm_config = model.language_model.model.config
    pruning_scheduler = create_scheduler_from_model_config(
        model_config=llm_config,
        mlp_prune_ratio=cfg.config.mlp_prune_ratio,
        attn_prune_ratio=cfg.config.attn_prune_ratio,
        skip_layers=(0, 1, 2),
        max_seq_len=cfg.config.max_seq_len,
        history_metric_ratio=cfg.config.history_metric_ratio,
        device=torch.device("cuda:0"),
        dtype=torch.bfloat16,
    )

    model.language_model.model.set_scheduler(pruning_scheduler)
    model.language_model.model.init_dual_cuda_graphs()

    logger.info("CUDA Graph initialization complete. Dense and Sparse graphs captured.")
    logger.info(f"  MLP keep dims: {pruning_scheduler.static_cfg.mlp_keep_dims}")
    logger.info(f"  Attention keep heads: {pruning_scheduler.static_cfg.attn_keep_heads}")


def prepare_calibration(
    generate_cfg: GenerateConfig,
    model,
    processor,
    proprio_projector,
    calibration_dataset_name: str,
) -> None:
    tokenizer = ActionTokenizer(processor.tokenizer)
    cfg.derived.tokenizer = tokenizer
    processor.tokenizer.model_max_length = 40

    # Calibration skips metric updates while cur_batch_index is zero.
    cfg.state.cur_batch_index = -1
    if generate_cfg.prune:
        cfg.state.calibration_stage = True
        if check_calib_saving_info():
            load_calib_saving_info(model)
        else:
            collator = PaddedCollatorForActionPrediction(
                processor.tokenizer.model_max_length,
                processor.tokenizer.pad_token_id,
                padding_side="right",
            )
            batch_transform = RLDSBatchTransform(
                    tokenizer,
                    processor.tokenizer,
                    image_transform=processor.image_processor.apply_transform,
                    prompt_builder_fn=PurePromptBuilder,
                    use_wrist_image=generate_cfg.num_images_in_input > 1,
                    use_proprio=generate_cfg.use_proprio,
                )
            train_dataset = RLDSDataset(
                generate_cfg.data_root_dir,
                calibration_dataset_name,
                batch_transform,
                resize_resolution=tuple(model.config.image_sizes),
                shuffle_buffer_size=generate_cfg.shuffle_buffer_size,
                image_aug=generate_cfg.image_aug,
            )
            calibration_data_loader = DataLoader(
                train_dataset,
                batch_size=generate_cfg.batch_size,
                sampler=None,
                collate_fn=collator,
                num_workers=0,
            )
            run_calibration(
                    generate_cfg,
                    model,
                calibration_data_loader,
                proprio_projector,
                device=torch.device("cuda:0"),
            )
            save_calib_info(model)
            exit()
        cfg.state.calibration_stage = False
    else:
        cfg.state.calibration_stage = False

    cfg.state.cur_batch_index = 0


@draccus.wrap()
def eval_libero(generate_cfg: GenerateConfig) -> float:
    """Main function to evaluate a trained policy on LIBERO benchmark tasks."""
    apply_generate_config(generate_cfg)

    process_control()
    reset_eval_prune_state(cfg)
    
    calibration_dataset_name = generate_cfg.calibration_dataset_name if generate_cfg.calibration_dataset_name is not None else f"{generate_cfg.task_suite_name}_no_noops"
    # calibration_dataset_name = "libero_90"
    cfg.derived.dataset_name = calibration_dataset_name.replace("_no_noops", "")
    
    
    # Validate configuration
    validate_config(generate_cfg)

    # Set random seed
    set_seed_everywhere(generate_cfg.seed)

    # Initialize model and components
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(generate_cfg)
    # model, action_head, proprio_projector, noisy_action_projector, processor

    if generate_cfg.prune:
        model = make_prune_model(model)

    initialize_cuda_graphs(generate_cfg, model)
    prepare_calibration(generate_cfg, model, processor, proprio_projector, calibration_dataset_name)
    
    # Get expected image dimensions
    resize_size = get_image_resize_size(generate_cfg)

    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(generate_cfg)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[generate_cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    log_message(f"Task suite: {generate_cfg.task_suite_name}", log_file)

    # Start evaluation
    total_episodes, total_successes = 0, 0

    if generate_cfg.prune and cfg.config.use_triton:
        warmup_triton_kernel()

    for task_id in tqdm.tqdm(range(num_tasks)):
        total_episodes, total_successes = run_task(
            generate_cfg,
            task_suite,
            task_id,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            total_episodes,
            total_successes,
            log_file,
        )

    # Calculate final success rate
    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    # Log final results
    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)
    log_message("prune count: {}".format(cfg.state.prune_count), log_file)

    # Log to wandb if enabled
    if generate_cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": final_success_rate,
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)

    # Close log file
    if log_file:
        log_file.close()
        

    return final_success_rate


def warmup_triton_kernel(): 
    from dynamic_prune.model.triton_linear import (
        up_channel_level_indices_linear, 
        down_channel_level_indices_linear, 
        down_channel_transpose_level_indices_linear, 
        DEVICE, 
        BLOCK_SIZE_N, 
        BLOCK_SIZE_K, 
        DIM
    )
    a = torch.rand((611, 4096), device=DEVICE, dtype=torch.bfloat16) - 0.5
    b = torch.rand((DIM, 4096), device=DEVICE, dtype=torch.bfloat16) - 0.5
    index_list = [_ for _ in range(DIM // BLOCK_SIZE_N * BLOCK_SIZE_N)]
    index_list = sorted(random.sample(index_list, int(len(index_list) // BLOCK_SIZE_N * 0.8) * BLOCK_SIZE_N))
    index_list = torch.tensor(index_list, dtype=torch.int32, device=DEVICE)
    
    # unsqueeze for add batch_size
    up_channel_level_indices_linear(a.unsqueeze(0), b, index_list)

    a = torch.rand((611, int(DIM // BLOCK_SIZE_K * 0.8) * BLOCK_SIZE_K), device=DEVICE, dtype=torch.bfloat16) - 0.5
    b = torch.rand((4096, DIM), device=DEVICE, dtype=torch.bfloat16) - 0.5
    index_list = [_ for _ in range(DIM)]
    index_list = sorted(random.sample(index_list, int(len(index_list) // BLOCK_SIZE_K * 0.8) * BLOCK_SIZE_K))
    index_list = torch.tensor(index_list, dtype=torch.int32, device=DEVICE)
    
    down_channel_level_indices_linear(a.unsqueeze(0), b, index_list)
    
    a = torch.rand((611, int(DIM // BLOCK_SIZE_K * 0.8) * BLOCK_SIZE_K), device=DEVICE, dtype=torch.bfloat16) - 0.5
    b = torch.rand((DIM, 4096), device=DEVICE, dtype=torch.bfloat16) - 0.5
    index_list = [_ for _ in range(DIM)]
    index_list = sorted(random.sample(index_list, int(len(index_list) // BLOCK_SIZE_K * 0.8) * BLOCK_SIZE_K))
    index_list = torch.tensor(index_list, dtype=torch.int32, device=DEVICE)
    
    down_channel_transpose_level_indices_linear(a.unsqueeze(0), b, index_list)

if __name__ == "__main__":
    eval_libero()
