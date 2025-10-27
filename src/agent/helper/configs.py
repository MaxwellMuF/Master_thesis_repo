from dataclasses import dataclass
import argparse
from typing import Tuple, Any


# ---------------------------------------------------------------------------
# Helper: robust string→bool converter
# ---------------------------------------------------------------------------
def _to_bool(value: Any) -> bool:
    """
    string to bool if nessesary. Help out for arg parser.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(int(value))
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "f", "no", "n", "off"}:
            return False
    raise argparse.ArgumentTypeError(f"Cannot convert to bool: {value!r}")


# ---------------------------------------------------------------------------
# Parser: argparse with types 
# ---------------------------------------------------------------------------
def _parse_arguments() -> argparse.Namespace:
    """
    Argparse parser with explicit types.
    """
    parser = argparse.ArgumentParser(description="Configure model parameters")

    # Data input
    parser.add_argument("--path_data", type=str, default="/workspace/data/mimic-iv/icd-10/hosp/",
                        help="Path of dataset.")
    parser.add_argument("--name_data", type=str, default="subval_1.parquet",
                        help="Name of dataset.")
    parser.add_argument("--path_prompt_set", type=str, default="/workspace/data/prompts_agent/",
                        help="Path of prompts.")
    parser.add_argument("--name_prompts_data", type=str, default="one_long.json",
                        help="Name of the prompt set.")

    # Model
    parser.add_argument("--name_model", type=str, default="Qwen/Qwen3-8B",
                        help="Model name from Hugging Face.")

    # Inference
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Sampling temperature.")
    parser.add_argument("--max_tokens", type=int, default=10000,
                        help="Maximum number of generated tokens (use 10000, not 10_000).")
    parser.add_argument("--role_to_add_note", type=str, default="user",
                        help="Role to attach the admission note to (e.g., 'system' or 'user').")
    parser.add_argument("--enable_thinking", type=_to_bool, default=True,
                        help="Allow/force model to generate </think>.")
    parser.add_argument("--continue_final_message", type=_to_bool, default=False,
                        help="XOR with add_generation_prompt.")
    parser.add_argument("--add_generation_prompt", type=_to_bool, default=True,
                        help="XOR with continue_final_message.")

    # Hardware
    parser.add_argument("--type_gpu", type=str, default="a100",
                        help="Kind of GPU, e.g., a100 or h200.")

    # Data output
    parser.add_argument("--path_output", type=str, default="/workspace/data/output/",
                        help="Output path for results.")
    
    # Task pipeline random_seed
    parser.add_argument("--name_prompt_task_1", type=str, default="one_task.json",
                        help="File name of task prompt(s)")
    parser.add_argument("--name_prompt_task_2", type=str, default="one_task_2.json",
                        help="File name of task_2 prompt(s)")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Set seed for choosing example to improve prompt")
    parser.add_argument("--max_iterations", type=int, default=20,
                        help="Number of iterations of data and tasks pipelines ~ number of generated prompts")
    parser.add_argument("--improve_prompt", type=_to_bool, default=True,
                        help="Set to False, skip the task pipelines, i.e. not changing the prompt. You can test n prompt by setting the max_iteration to the number of prompts to be tested")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataclasses: pipeline configurations 
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PipelineDataConfig:
    # Data input
    path_data: str
    name_data: str
    path_prompt_set: str
    name_prompts_data: str

    # Model
    name_model: str

    # Inference
    temperature: float
    max_tokens: int
    role_to_add_note: str
    enable_thinking: bool
    continue_final_message: bool
    add_generation_prompt: bool

    # Hardware
    type_gpu: str

    # Data output
    path_output: str


@dataclass(frozen=True)
class PipelineTaskConfig:
    name_prompt_task_1: str
    name_prompt_task_2: str
    random_seed: int
    max_iterations: int
    improve_prompt: bool


# ---------------------------------------------------------------------------
# Builder: build configs (Namespace → dataclasses; same fixed order)
# ---------------------------------------------------------------------------
def _build_configs(args: argparse.Namespace) -> Tuple[PipelineDataConfig, PipelineTaskConfig]:
    """
    Build typed config objects from parsed CLI args in the fixed order.
    """
    p1 = PipelineDataConfig(
        # Data input
        path_data               = args.path_data,
        name_data               = args.name_data,
        path_prompt_set         = args.path_prompt_set,
        name_prompts_data       = args.name_prompts_data,

        # Model
        name_model              = args.name_model,

        # Inference
        temperature             = args.temperature,
        max_tokens              = args.max_tokens,
        role_to_add_note        = args.role_to_add_note,
        enable_thinking         = args.enable_thinking,
        continue_final_message  = args.continue_final_message,
        add_generation_prompt   = args.add_generation_prompt,

        # Hardware
        type_gpu                = args.type_gpu,

        # Data output
        path_output             = args.path_output,
        


    )

    # Pipeline 2 placeholders for now
    p2 = PipelineTaskConfig(
        # Task pipeline
        name_prompt_task_1      = args.name_prompt_task_1,
        name_prompt_task_2      = args.name_prompt_task_2,
        random_seed             = args.random_seed,
        max_iterations          = args.max_iterations,
        improve_prompt          = args.improve_prompt)

    return p1, p2


def parse_and_build_configs() -> Tuple[PipelineDataConfig, PipelineTaskConfig]:
    """
    Run arg parser and build configs.
    """
    args = _parse_arguments()
    return _build_configs(args)
