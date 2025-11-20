# SPDX-License-Identifier: MIT
"""
Configuration helpers for the ECOT RLDS integration layer.

Only lightweight defaults and validation utilities live here; the real
implementation will be filled in later steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union


@dataclass(frozen=True)
class ECOTDefaults:
    """Default knobs for the ECOT RLDS adapter."""

    # Data paths
    data_root_dir: str = None  # Required, no default
    data_mix: str = None  # Required, no default
    
    # Image and action parameters
    image_size: list = None  # Default: [224, 224]
    future_action_window_size: int = 15
    past_action_window_size: int = 0
    action_dim: int = 7
    
    # Dataset parameters
    shuffle_buffer_size: int = 256_000
    image_aug: bool = False
    scheduled_stage: int = 1
    reasoning_json: str = "/share/project/lvjing/datas/embodied_features_bridge.json"
    load_proprio: bool = True
    lower_case_instruction: bool = True
    train: bool = True
    
    # Thinking token configuration (optional, defaults handled in dataset.py)
    thinking_token_count: int = 2
    thinking_token: str = "<|thinking|>"
    start_of_thinking_token: str = "<|start_of_thinking|>"
    end_of_thinking_token: str = "<|end_of_thinking|>"
    tag2think_count: Optional[dict] = None  # Default handled in dataset.py


def _extract(cfg: Any, key: str, default: Any = None) -> Any:
    """
    Safely extract an attribute / key from OmegaConf / dict-like / object configs.
    
    Handles:
    - None: returns default
    - dict: uses .get(key, default)
    - OmegaConf objects: uses OmegaConf.get(cfg, key, default) or safe attribute access
    - Objects with attributes: uses getattr with default
    """
    if cfg is None:
        return default
    
    # Handle dict
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    
    # Handle OmegaConf objects (DictConfig, ListConfig, etc.)
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf
        if isinstance(cfg, (DictConfig, ListConfig)):
            # Use OmegaConf.get() which safely handles missing keys
            return OmegaConf.get(cfg, key, default)
    except ImportError:
        pass
    except Exception:
        # If OmegaConf access fails, fall through to other methods
        pass
    
    # Handle objects with attributes (including OmegaConf when key exists as attribute)
    if hasattr(cfg, key):
        try:
            return getattr(cfg, key, default)
        except Exception:
            # Attribute access failed, try other methods
            pass
    
    # Try dict-style access as last resort
    try:
        return cfg[key]
    except (KeyError, AttributeError, TypeError):
        # Key doesn't exist or object doesn't support indexing
        return default


def _ensure_tuple_of_ints(values: Union[Sequence[int], Tuple[int, int]]) -> Tuple[int, int]:
    if values is None:
        raise ValueError("image_size must be provided and contain two integers.")
    if len(values) != 2:
        raise ValueError(f"image_size must have length 2, got {values}")
    return int(values[0]), int(values[1])


def validate_and_normalize_cfg(ecot_cfg: Dict[str, Any], global_cfg: Any) -> Dict[str, Any]:
    """
    Validate and normalize configuration for the ECOT RLDS adapter.
    
    Parameters can be specified in two ways (priority order):
    1. In ecot_cfg (datasets.vla_data.ecot.*) - takes priority
    2. In global_cfg (framework.action_model.*, datasets.vla_data.image_size) - fallback
    
    This allows flexibility while maintaining backward compatibility with existing QwenGR00T config structure.
    """

    defaults = ECOTDefaults()

    # --- Extract ECOT-specific parameters (required in ecot_cfg) ---
    data_root_dir = _extract(ecot_cfg, "data_root_dir", defaults.data_root_dir)
    data_mix = _extract(ecot_cfg, "data_mix", defaults.data_mix)
    
    if not data_root_dir:
        raise ValueError("datasets.vla_data.ecot.data_root_dir must be specified.")
    if not data_mix:
        raise ValueError("datasets.vla_data.ecot.data_mix must be specified.")

    # --- Extract image size (priority: ecot_cfg > global_cfg > default) ---
    image_size_raw = _extract(ecot_cfg, "image_size", None)
    if image_size_raw is None and global_cfg is not None:
        # Fallback to global config (only if global_cfg is provided)
        datasets_cfg = _extract(global_cfg, "datasets", {})
        if datasets_cfg:  # Only proceed if datasets exists
            vla_data_cfg = _extract(datasets_cfg, "vla_data", {})
            if vla_data_cfg:  # Only proceed if vla_data exists
                image_size_raw = _extract(vla_data_cfg, "image_size", None)
    
    # Use default if still None
    if image_size_raw is None:
        image_size = (224, 224) if defaults.image_size is None else _ensure_tuple_of_ints(defaults.image_size)
    else:
        image_size = _ensure_tuple_of_ints(image_size_raw)

    # --- Extract action parameters (priority: ecot_cfg > global_cfg > default) ---
    future_action_window_size = _extract(ecot_cfg, "future_action_window_size", None)
    past_action_window_size = _extract(ecot_cfg, "past_action_window_size", None)
    action_dim = _extract(ecot_cfg, "action_dim", None)
    
    # Fallback to global config if not in ecot_cfg (only if global_cfg is provided and not None)
    if (future_action_window_size is None or past_action_window_size is None or action_dim is None) and global_cfg is not None:
        framework_cfg = _extract(global_cfg, "framework", {})
        if framework_cfg:  # Only proceed if framework exists
            action_model_cfg = _extract(framework_cfg, "action_model", {})
            
            if future_action_window_size is None:
                future_action_window_size = _extract(action_model_cfg, "future_action_window_size", None)
            if past_action_window_size is None:
                past_action_window_size = _extract(action_model_cfg, "past_action_window_size", None)
            if action_dim is None:
                action_dim = _extract(action_model_cfg, "action_dim", None)
    
    # Use defaults if still None (after checking both ecot_cfg and global_cfg)
    if future_action_window_size is None:
        future_action_window_size = defaults.future_action_window_size
    if past_action_window_size is None:
        past_action_window_size = defaults.past_action_window_size
    if action_dim is None:
        action_dim = defaults.action_dim
    
    future_action_window_size = int(future_action_window_size)
    past_action_window_size = int(past_action_window_size)
    action_dim = int(action_dim)
    
    if future_action_window_size <= 0:
        raise ValueError("future_action_window_size must be positive.")
    
    chunk_len = past_action_window_size + 1 + future_action_window_size

    # Dataset parameters
    shuffle_buffer_size = int(_extract(ecot_cfg, "shuffle_buffer_size", defaults.shuffle_buffer_size))
    if shuffle_buffer_size <= 0:
        raise ValueError(f"shuffle_buffer_size must be positive, got {shuffle_buffer_size}")
    image_aug = bool(_extract(ecot_cfg, "image_aug", defaults.image_aug))
    scheduled_stage = int(_extract(ecot_cfg, "scheduled_stage", defaults.scheduled_stage))
    reasoning_json = _extract(ecot_cfg, "reasoning_json", defaults.reasoning_json)
    load_proprio = bool(_extract(ecot_cfg, "load_proprio", defaults.load_proprio))
    lower_case_instruction = bool(_extract(ecot_cfg, "lower_case_instruction", defaults.lower_case_instruction))
    train = bool(_extract(ecot_cfg, "train", defaults.train))

    # Thinking token configuration (optional - defaults used if None)
    thinking_token_count = _extract(ecot_cfg, "thinking_token_count", defaults.thinking_token_count)
    thinking_token = _extract(ecot_cfg, "thinking_token", defaults.thinking_token)
    start_of_thinking_token = _extract(ecot_cfg, "start_of_thinking_token", defaults.start_of_thinking_token)
    end_of_thinking_token = _extract(ecot_cfg, "end_of_thinking_token", defaults.end_of_thinking_token)
    tag2think_count = _extract(ecot_cfg, "tag2think_count", defaults.tag2think_count)

    # Return simple dict - all values are directly from ecot_cfg or defaults
    return {
        "data_root_dir": Path(data_root_dir),
        "data_mix": data_mix,
        "image_size": image_size,
        "future_action_window_size": future_action_window_size,
        "past_action_window_size": past_action_window_size,
        "chunk_len": chunk_len,
        "action_dim": action_dim,
        "shuffle_buffer_size": shuffle_buffer_size,
        "image_aug": image_aug,
        "scheduled_stage": scheduled_stage,
        "reasoning_json": reasoning_json,
        "load_proprio": load_proprio,
        "lower_case_instruction": lower_case_instruction,
        "train": train,
        "thinking_token_count": thinking_token_count,
        "thinking_token": thinking_token,
        "start_of_thinking_token": start_of_thinking_token,
        "end_of_thinking_token": end_of_thinking_token,
        "tag2think_count": tag2think_count,
    }



