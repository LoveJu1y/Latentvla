# SPDX-License-Identifier: MIT
"""
Dataset adapter for ECOT RLDS integration.

This module wraps the Prismatic RLDS pipeline and converts RLDS batches into
StarVLA-compatible sample dictionaries. It intentionally lives outside of the
`prismatic/` tree so we can integrate without modifying upstream sources.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

import numpy as np
from torch.utils.data import IterableDataset

from prismatic.vla.datasets.rlds import make_interleaved_dataset
from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights
from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType

from .transforms import ensure_image_size, to_pil_list_from_numpy

logger = logging.getLogger(__name__)


def _to_str(value: Any) -> str:
    """
    Safely decode bytes / numpy scalar strings to Python str.

    Handles various input types:
    - bytes / np.bytes_: decode to UTF-8 string
    - np.str_: convert to Python str
    - np.ndarray (scalar): extract item and recursively convert
    - str: return as-is
    - None / empty: return empty string
    - Other types: convert to string

    Parameters
    ----------
    value:
        Input that may be bytes, numpy scalar, or already a string.

    Returns
    -------
    str
        Decoded/converted string value.
    """
    # Handle None and empty values
    if value is None:
        return ""
    
    # Handle numpy string types
    if isinstance(value, (np.bytes_, np.str_)):
        return str(value)
    
    # Handle Python bytes
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError as e:
            logger.warning(f"[ECOT RLDS] Failed to decode bytes as UTF-8: {e}, using fallback")
            return value.decode("utf-8", errors="replace")
    
    # Handle numpy arrays (scalar or 0-d)
    if isinstance(value, np.ndarray):
        if value.shape == ():
            # Scalar array - extract item and recursively convert
            return _to_str(value.item())
        elif value.size == 0:
            # Empty array
            return ""
        else:
            raise ValueError(
                f"Cannot decode non-scalar numpy array to string: shape={value.shape}, "
                f"dtype={value.dtype}"
            )
    
    # Handle already-string types
    if isinstance(value, str):
        return value
    
    # Fallback: convert to string
    return str(value)


class ECOTBatchTransform:
    """
    Convert RLDS batches to StarVLA sample dictionaries.

    This transform handles:
    - Image conversion (numpy -> PIL)
    - Action window alignment (to chunk_len)
    - Language instruction decoding
    - CoT reasoning text extraction
    - Optional proprioceptive state extraction
    """

    def __init__(
        self,
        image_size: Tuple[int, int],
        chunk_len: int,
        action_dim: int,
        lower_case_instruction: bool = True,
        load_proprio: bool = True,
    ) -> None:
        self.image_size = image_size
        self.chunk_len = chunk_len
        self.action_dim = action_dim
        self.lower_case_instruction = lower_case_instruction
        self.load_proprio = load_proprio
        self._action_mismatch_warned = False

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform a single RLDS batch.

        Parameters
        ----------
        rlds_batch:
            Raw batch dictionary emitted by ``make_interleaved_dataset``.

        Returns
        -------
        dict
            StarVLA sample dictionary containing ``image``, ``lang``, ``action``,
            and optional ``state`` / ``reasoning`` fields.
        """
        try:
            # Check required fields with detailed error messages (Task 2.8)
            if "observation" not in rlds_batch:
                available_keys = list(rlds_batch.keys())
                raise KeyError(
                    f"Missing 'observation' in RLDS batch. "
                    f"Available keys: {available_keys}"
                )
            observation = rlds_batch["observation"]
            
            if not isinstance(observation, dict):
                raise TypeError(
                    f"Expected 'observation' to be a dict, got {type(observation)}"
                )
            
            if "image_primary" not in observation:
                obs_keys = list(observation.keys())
                raise KeyError(
                    f"Missing 'image_primary' in observation. "
                    f"Available observation keys: {obs_keys}"
                )
            
            if "action" not in rlds_batch:
                available_keys = list(rlds_batch.keys())
                raise KeyError(
                    f"Missing 'action' in RLDS batch. "
                    f"Available keys: {available_keys}"
                )
            
            if "task" not in rlds_batch:
                available_keys = list(rlds_batch.keys())
                raise KeyError(
                    f"Missing 'task' in RLDS batch. "
                    f"Available keys: {available_keys}"
                )
            
            task = rlds_batch["task"]
            if not isinstance(task, dict):
                raise TypeError(
                    f"Expected 'task' to be a dict, got {type(task)}"
                )
            
            if "language_instruction" not in task:
                task_keys = list(task.keys())
                raise KeyError(
                    f"Missing 'language_instruction' in task. "
                    f"Available task keys: {task_keys}"
                )

            # 1. Extract and convert image
            # Note: RLDS frame_transform may handle resize via resize_size parameter
            # We only do minimal conversion (numpy -> PIL) here
            # Qwen processor will handle final resize/normalize if needed
            primary = observation["image_primary"]
            images = to_pil_list_from_numpy(primary)
            
            # Optional resize for consistency with LeRobot dataset behavior
            # LeRobot does resize to (224, 224) in __getitem__, but this is mainly
            # for performance (reducing memory/transfer). Qwen processor can handle
            # different image sizes, but we keep this for consistency.
            # If RLDS resize_size is set correctly, images should already be resized.
            if self.image_size:
                images = ensure_image_size(images, self.image_size)

            # 2. Extract and decode language instruction (Task 2.5)
            lang_value = rlds_batch["task"]["language_instruction"]
            lang = _to_str(lang_value).strip()
            if not lang:
                logger.warning(
                    "[ECOT RLDS] Empty language instruction detected, using fallback"
                )
                lang = "perform the task"  # Fallback for empty instructions
            if self.lower_case_instruction:
                lang = lang.lower()

            # 3. Extract and process reasoning (CoT) (Task 2.5)
            # Handle missing or empty reasoning gracefully
            reasoning_raw = rlds_batch.get("reasoning")
            if reasoning_raw is None:
                reasoning_raw = b""
            reasoning_text = _to_str(reasoning_raw).strip()
            
            # No dropout applied - use reasoning text as-is
            # reasoning_subset is kept for compatibility but will be empty
            reasoning_subset = ""

            # Combine instruction and reasoning (which already contains thinking tokens from Prismatic)
            # This is the key step: reasoning_text already has thinking tokens inserted by make_interleaved_dataset
            if reasoning_text:
                # Add @ delimiter between instruction and reasoning for clear boundary detection
                # This works for both stage 0 (no thinking tokens) and stage 2+ (with thinking tokens)
                # Format: "instruction @ reasoning" or "instruction @ <|start|>...<|end|> reasoning"
                lang = lang + " @ " + reasoning_text

            # 4. Extract and align action sequence
            # RLDS returns [window_size, future_action_window_size+1, action_dim]
            # For window_size=1, this is [1, future+1, action_dim]
            # We need to flatten to [chunk_len, action_dim] where chunk_len = past + 1 + future
            action = np.asarray(rlds_batch["action"], dtype=np.float32)

            # Handle different input shapes
            if action.ndim == 3:
                # [window_size, future+1, action_dim] - take first window (current frame)
                if action.shape[0] > 1:
                    logger.debug(
                        f"[ECOT RLDS] Action has window_size={action.shape[0]}, "
                        f"taking first window (current frame)"
                    )
                action = action[0]  # [future+1, action_dim]
            elif action.ndim == 2:
                # Already flattened to [future+1, action_dim] or [chunk_len, action_dim]
                pass
            else:
                raise ValueError(
                    f"Unexpected action shape: {action.shape}. "
                    f"Expected [window, future+1, dim] or [future+1, dim]"
                )

            # Verify action dimension matches expected
            if action.shape[-1] != self.action_dim:
                raise ValueError(
                    f"Action dimension mismatch: expected {self.action_dim}, "
                    f"got {action.shape[-1]}"
                )

            # Align action sequence length to chunk_len
            # chunk_len = past_action_window_size + 1 + future_action_window_size
            # RLDS provides: current + future (future_action_window_size + 1 steps)
            # If past_action_window_size > 0, we would need to pad with past actions,
            # but since past_action_window_size is typically 0, we just need to ensure
            # the length matches chunk_len (which should equal future + 1)
            if action.shape[0] != self.chunk_len:
                if action.shape[0] > self.chunk_len:
                    # Truncate to last chunk_len steps (keep most recent actions)
                    action = action[-self.chunk_len :]
                    if not self._action_mismatch_warned:
                        logger.warning(
                            f"[ECOT RLDS] Action sequence length ({action.shape[0]}) > chunk_len ({self.chunk_len}), "
                            f"truncating to last {self.chunk_len} steps. "
                            f"Consider setting future_action_window_size={self.chunk_len - 1} in RLDS config."
                        )
                        self._action_mismatch_warned = True
                else:
                    # Pad with zeros (not recommended - indicates config mismatch)
                    # This happens if RLDS future_action_window_size < model future_action_window_size
                    padding = np.zeros(
                        (self.chunk_len - action.shape[0], self.action_dim), dtype=action.dtype
                    )
                    action = np.concatenate([action, padding], axis=0)
                    if not self._action_mismatch_warned:
                        logger.warning(
                            f"[ECOT RLDS] Action sequence length ({action.shape[0]}) < chunk_len ({self.chunk_len}), "
                            f"padding with zeros. This may degrade performance! "
                            f"Please ensure RLDS future_action_window_size={self.chunk_len - 1} "
                            f"matches model future_action_window_size."
                        )
                        self._action_mismatch_warned = True

            # Final shape verification
            if action.shape != (self.chunk_len, self.action_dim):
                raise ValueError(
                    f"Action shape mismatch after alignment: "
                    f"expected {(self.chunk_len, self.action_dim)}, got {action.shape}"
                )

            # 5. Extract state (proprio) if requested (Task 2.6)
            state = None
            if self.load_proprio:
                proprio = observation.get("proprio")
                if proprio is not None:
                    try:
                        state = np.asarray(proprio, dtype=np.float32)
                        
                        # Handle different proprio shapes
                        if state.ndim == 0:
                            # Scalar - expand to 1D
                            state = np.array([state], dtype=np.float32)
                        elif state.ndim == 1:
                            # Already 1D: [state_dim]
                            pass
                        elif state.ndim == 2:
                            # 2D: [window_size, state_dim] or [time_steps, state_dim]
                            if state.shape[0] > 1:
                                # Take first frame (current frame) for window_size > 1
                                state = state[0]  # [state_dim]
                            else:
                                # Squeeze window dimension if window_size == 1
                                state = state[0]  # [state_dim]
                        else:
                            logger.warning(
                                f"[ECOT RLDS] Unexpected proprio shape: {state.shape}, "
                                f"flattening to 1D"
                            )
                            state = state.flatten()
                        
                        # Final verification: state should be 1D
                        if state.ndim != 1:
                            raise ValueError(
                                f"Proprio state must be 1D after processing, got shape: {state.shape}"
                            )
                        
                    except Exception as e:
                        logger.warning(
                            f"[ECOT RLDS] Failed to extract proprio state: {e}, "
                            f"skipping state field"
                        )
                        state = None
                else:
                    # Proprio not available in observation
                    if self.load_proprio:
                        logger.debug(
                            "[ECOT RLDS] load_proprio=True but 'proprio' not found in observation"
                        )

            # 6. Extract dataset name
            dataset_name = _to_str(rlds_batch.get("dataset_name", "unknown_dataset"))

            # Build sample dictionary
            sample: Dict[str, Any] = {
                "image": images,
                "lang": lang,
                "action": action,
                "reasoning": reasoning_text,
                "reasoning_subset": reasoning_subset,
                "dataset_name": dataset_name,
            }
            if state is not None:
                sample["state"] = state

            return sample

        except (KeyError, ValueError, TypeError) as e:
            # Enhanced error logging (Task 2.8)
            logger.error(f"[ECOT RLDS] Failed to transform batch: {type(e).__name__}: {e}")
            logger.error(f"  Available batch keys: {list(rlds_batch.keys())}")
            if "observation" in rlds_batch:
                obs = rlds_batch["observation"]
                if isinstance(obs, dict):
                    logger.error(f"  Observation keys: {list(obs.keys())}")
                else:
                    logger.error(f"  Observation type: {type(obs)}")
            if "task" in rlds_batch:
                task = rlds_batch["task"]
                if isinstance(task, dict):
                    logger.error(f"  Task keys: {list(task.keys())}")
                else:
                    logger.error(f"  Task type: {type(task)}")
            if "action" in rlds_batch:
                action = rlds_batch["action"]
                logger.error(f"  Action type: {type(action)}, shape: {getattr(action, 'shape', 'N/A')}")
            raise
        except Exception as e:
            # Catch-all for unexpected errors
            logger.error(f"[ECOT RLDS] Unexpected error during batch transformation: {type(e).__name__}: {e}")
            logger.error(f"  Available batch keys: {list(rlds_batch.keys())}")
            logger.exception("  Full traceback:")
            raise


class ECOTRLDSDataset(IterableDataset):
    """
    Iterable dataset that yields StarVLA-formatted samples built from RLDS data.

    This class wraps the Prismatic RLDS pipeline and applies the ECOTBatchTransform
    to convert RLDS batches into StarVLA-compatible sample dictionaries.
    """

    def __init__(
        self,
        cfg: Any,
    ) -> None:
        """
        Initialize the ECOT RLDS dataset.

        Parameters
        ----------
        cfg:
            Global configuration object (OmegaConf or dict-like). Can contain:
            - ``datasets.vla_data.ecot.*``: ECOT-specific configuration (takes priority)
            - ``framework.action_model.*``: Action model parameters (fallback if not in ecot.*)
            - ``datasets.vla_data.image_size``: Image size (fallback if not in ecot.*)
        """
        super().__init__()

        # Import here to avoid circular imports
        from .config import validate_and_normalize_cfg

        # 1. Extract ecot config and validate
        # Parameters can be in ecot.* (priority) or in framework.* / datasets.vla_data.* (fallback)
        from .config import _extract
        
        # Safely extract ecot_cfg using _extract to handle OmegaConf and dict
        datasets_cfg = _extract(cfg, "datasets", {})
        vla_data_cfg = _extract(datasets_cfg, "vla_data", {})
        ecot_cfg = _extract(vla_data_cfg, "ecot", {})
        
        # Validate and get all parameters - ecot_cfg takes priority, global_cfg is fallback
        cfg_dict = validate_and_normalize_cfg(ecot_cfg, cfg)
        
        # Store config dict for easy access
        self.cfg_dict = cfg_dict

        # Extract key parameters directly
        self.image_size = cfg_dict["image_size"]
        self.future_action_window_size = cfg_dict["future_action_window_size"]
        self.chunk_len = cfg_dict["chunk_len"]
        self.action_dim = cfg_dict["action_dim"]

        # 2. Determine mixture specification
        data_mix = self.cfg_dict["data_mix"]
        if data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[data_mix]
        else:
            # Assume single dataset name
            mixture_spec = [(data_mix, 1.0)]
            logger.info(f"[ECOT RLDS] Using single dataset '{data_mix}' (not in OXE_NAMED_MIXTURES)")

        # 3. Build RLDS dataset kwargs
        num_parallel_calls = 16  # Default for frame transforms
        dataset_kwargs_list, weights = get_oxe_dataset_kwargs_and_weights(
            self.cfg_dict["data_root_dir"],
            mixture_spec,
            load_camera_views=("primary",),  # Can be extended to multi-view later
            load_depth=False,
            load_proprio=self.cfg_dict["load_proprio"],
            load_language=True,
            action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
        )
        
        # Add reasoning_json path to each dataset config
        reasoning_json_path = Path(self.cfg_dict["reasoning_json"]) if isinstance(self.cfg_dict["reasoning_json"], str) else self.cfg_dict["reasoning_json"]
        
        # Add reasoning_dataset_path to each dataset kwargs
        for dataset_kwargs in dataset_kwargs_list:
            dataset_kwargs["reasoning_dataset_path"] = str(reasoning_json_path)

        # 4. Configure frame transforms
        frame_transform_kwargs: Dict[str, Any] = dict(
            resize_size=self.image_size,
            num_parallel_calls=num_parallel_calls,
        )
        if self.cfg_dict["image_aug"]:
            frame_transform_kwargs.update(
                image_augment_kwargs=dict(
                    random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                    random_brightness=[0.2],
                    random_contrast=[0.8, 1.2],
                    random_saturation=[0.8, 1.2],
                    random_hue=[0.05],
                    augment_order=[
                        "random_resized_crop",
                        "random_brightness",
                        "random_contrast",
                        "random_saturation",
                        "random_hue",
                    ],
                )
            )

        # 5. Build config object for Prismatic (expects cfg.vla.* attributes)
        # Simple object that directly reads from cfg_dict - no complex wrapping
        default_tag2think_count = {
            "TASK": 1,
            "PLAN": 1,
            "VISIBLE OBJECTS": 1,
            "SUBTASK REASONING": 1,
            "SUBTASK": 1,
            "MOVE REASONING": 1,
            "MOVE": 1,
            "GRIPPER POSITION": 1,
        }
        
        # Get tag2think_count - convert to dict if needed
        tag2think_raw = cfg_dict["tag2think_count"]
        if tag2think_raw is None:
            tag2think_count = default_tag2think_count
        elif isinstance(tag2think_raw, dict):
            tag2think_count = tag2think_raw
        else:
            # Try to convert (e.g., from OmegaConf)
            try:
                tag2think_count = dict(tag2think_raw)
            except (TypeError, ValueError):
                tag2think_count = default_tag2think_count
        
        # Simple config object for Prismatic - directly from cfg_dict
        class VLAConfig:
            def __init__(self):
                self.thinking_token_count = cfg_dict["thinking_token_count"]
                self.thinking_token = cfg_dict["thinking_token"]
                self.start_of_thinking_token = cfg_dict["start_of_thinking_token"]
                self.end_of_thinking_token = cfg_dict["end_of_thinking_token"]
                self.tag2think_count = tag2think_count
        
        class PrismaticConfig:
            def __init__(self):
                self.vla = VLAConfig()
        
        prismatic_cfg = PrismaticConfig()
        
        # 6. Build RLDS config
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=1,  # Only take current frame observation
                future_action_window_size=self.future_action_window_size,  # Must match model config
                skip_unlabeled=True,
                goal_relabeling_strategy="uniform",
            ),
            frame_transform_kwargs=frame_transform_kwargs,
            dataset_kwargs_list=dataset_kwargs_list,
            shuffle_buffer_size=self.cfg_dict["shuffle_buffer_size"],
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=self.cfg_dict["train"],
            scheduled_stage=self.cfg_dict["scheduled_stage"],
            cfg=prismatic_cfg,  # Pass config with vla attribute for reasoning construction
        )

        # 7. Build RLDS dataset
        logger.info(f"[ECOT RLDS] Building dataset with mixture '{data_mix}'")
        logger.info(f"[ECOT RLDS] Action window: future={self.future_action_window_size}, chunk_len={self.chunk_len}")
        logger.info(f"[ECOT RLDS] Image size: {self.image_size}, CoT stage: {self.cfg_dict['scheduled_stage']}")
        logger.info(f"[ECOT RLDS] Reasoning JSON: {self.cfg_dict.get('reasoning_json', 'default')}")
        logger.info(f"[ECOT RLDS] Thinking tokens: {prismatic_cfg.vla.thinking_token}, "
                   f"start={prismatic_cfg.vla.start_of_thinking_token}, "
                   f"end={prismatic_cfg.vla.end_of_thinking_token}")
        logger.info(f"[ECOT RLDS] Tag2think_count: {prismatic_cfg.vla.tag2think_count}")

        try:
            self.rlds_dataset, self._dataset_length, self._dataset_statistics = make_interleaved_dataset(**rlds_config)
            
            # Clean statistics for JSON serialization (Task 2.7)
            self._dataset_statistics = self._clean_statistics(self._dataset_statistics)
        except Exception as e:
            logger.error(f"[ECOT RLDS] Failed to build RLDS dataset: {type(e).__name__}: {e}")
            logger.error(f"  Data root dir: {self.cfg_dict['data_root_dir']}")
            logger.error(f"  Data mix: {data_mix}")
            logger.error(f"  Mixture spec: {mixture_spec}")
            logger.error(f"  Number of datasets: {len(dataset_kwargs_list)}")
            if dataset_kwargs_list:
                logger.error(f"  First dataset name: {dataset_kwargs_list[0].get('name', 'unknown')}")
            logger.exception("  Full traceback:")
            raise

        # 8. Create batch transform
        self.batch_transform = ECOTBatchTransform(
            image_size=self.image_size,
            chunk_len=self.chunk_len,
            action_dim=self.action_dim,
            lower_case_instruction=self.cfg_dict["lower_case_instruction"],
            load_proprio=self.cfg_dict["load_proprio"],
        )

        # 9. Logging flag for first sample
        self._log_once = True

        logger.info(f"[ECOT RLDS] Dataset initialized: length={self._dataset_length}")

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over the dataset, yielding StarVLA-formatted samples."""
        sample_count = 0
        try:
            for rlds_batch in self.rlds_dataset.as_numpy_iterator():
                try:
                    sample = self.batch_transform(rlds_batch)
                    sample_count += 1

                    # Log first sample for debugging
                    if self._log_once:
                        logger.info(f"[ECOT RLDS] First sample:")
                        logger.info(f"  image: {len(sample['image'])} views, size={sample['image'][0].size}")
                        logger.info(f"  lang: {sample['lang'][:60]}...")
                        logger.info(f"  action: {sample['action'].shape}")
                        logger.info(f"  reasoning: {sample['reasoning'][:80] if sample['reasoning'] else '(empty)'}...")
                        if "state" in sample:
                            logger.info(f"  state: {sample['state'].shape}")
                        self._log_once = False

                    yield sample
                except Exception as e:
                    # Enhanced error handling for individual batch transformation (Task 2.8)
                    logger.error(
                        f"[ECOT RLDS] Failed to transform batch at sample {sample_count}: "
                        f"{type(e).__name__}: {e}"
                    )
                    # Re-raise to stop iteration on critical errors
                    # (Alternatively, could skip and continue with warning)
                    raise
        except Exception as e:
            # Error during dataset iteration
            logger.error(
                f"[ECOT RLDS] Error during dataset iteration after {sample_count} samples: "
                f"{type(e).__name__}: {e}"
            )
            logger.exception("  Full traceback:")
            raise

    def __len__(self) -> int:
        """Return the dataset length."""
        return int(self._dataset_length)

    @staticmethod
    def _clean_statistics(stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively clean statistics dictionary, converting non-serializable types.
        
        Converts:
        - np.ndarray -> list
        - np.integer, np.floating -> Python int/float
        - Path -> str
        - Other types preserved as-is
        
        Parameters
        ----------
        stats:
            Statistics dictionary that may contain numpy types.
        
        Returns
        -------
        dict
            Cleaned dictionary with only JSON-serializable types.
        """
        cleaned = {}
        for key, value in stats.items():
            if isinstance(value, dict):
                cleaned[key] = ECOTRLDSDataset._clean_statistics(value)
            elif isinstance(value, np.ndarray):
                cleaned[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                cleaned[key] = value.item()
            elif isinstance(value, Path):
                cleaned[key] = str(value)
            elif isinstance(value, (list, tuple)):
                # Recursively clean list/tuple elements
                cleaned[key] = [
                    ECOTRLDSDataset._clean_statistics(item) if isinstance(item, dict)
                    else item.item() if isinstance(item, (np.integer, np.floating))
                    else item.tolist() if isinstance(item, np.ndarray)
                    else str(item) if isinstance(item, Path)
                    else item
                    for item in value
                ]
            else:
                cleaned[key] = value
        return cleaned

    @property
    def dataset_statistics(self) -> Dict[str, Any]:
        """
        Return cached dataset statistics (q01/q99, etc.).
        
        Statistics are cleaned and JSON-serializable.
        """
        return self._dataset_statistics

