"""Adapter utilities bridging local protocol modules with verl trainer hooks."""

from .data_preprocessor import preprocess_trajectories
from .env_bridge import BridgeResult, ToolExecutor, run_env_bridge_step
from .mask_injector import build_response_mask, inject_response_mask
from .reprompt_adapter import build_self_distillation_batch
from .reward_function import reward_fn

__all__ = [
    "BridgeResult",
    "ToolExecutor",
    "build_response_mask",
    "build_self_distillation_batch",
    "inject_response_mask",
    "preprocess_trajectories",
    "reward_fn",
    "run_env_bridge_step",
]
