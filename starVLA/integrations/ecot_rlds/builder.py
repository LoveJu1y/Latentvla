# SPDX-License-Identifier: MIT
"""
Builder utilities for ECOT RLDS datasets and dataloaders.

Provides dataset construction and DataLoader creation for ECOT RLDS integration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch.distributed as dist
from accelerate.logging import get_logger
from torch.utils.data import DataLoader

from .collate import collate_fn_ecot
from .dataset import ECOTRLDSDataset

logger = get_logger(__name__)


def get_vla_dataset_ecot(cfg: Any) -> ECOTRLDSDataset:
    """
    Construct the ECOT RLDS dataset adapter.

    Parameters
    ----------
    cfg:
        Global configuration object containing:
        - ``datasets.vla_data.ecot.*``: ECOT-specific configuration (data_root_dir, data_mix, etc.)
          Optional: image_size, action_dim, future_action_window_size, etc. (if not provided, will read from global config)
        - ``framework.action_model.*``: Action model parameters (used if not in ecot.*)
        - ``datasets.vla_data.image_size``: Image size (used if not in ecot.*)
        - ``datasets.vla_data.per_device_batch_size``: Batch size (for DataLoader)
        - ``datasets.vla_data.num_workers``: Number of DataLoader workers (for DataLoader)

    Returns
    -------
    ECOTRLDSDataset
        Initialized ECOT RLDS dataset instance.
    """
    return ECOTRLDSDataset(cfg)


def make_dataloader_ecot(cfg: Any) -> DataLoader:
    """
    Create a DataLoader for the ECOT RLDS dataset.

    This function:
    1. Constructs the ECOT RLDS dataset
    2. Creates a PyTorch DataLoader with appropriate batch size and workers
    3. Saves dataset statistics to disk (on rank 0 only, for distributed training)

    Parameters
    ----------
    cfg:
        Global configuration object. Must contain:
        - ``datasets.vla_data.per_device_batch_size``: Batch size per device
        - ``datasets.vla_data.num_workers``: Number of DataLoader workers (default: 4)
        - ``output_dir``: Output directory for saving statistics (optional)
        - ECOT-specific config (see ``get_vla_dataset_ecot``)

    Returns
    -------
    torch.utils.data.DataLoader
        DataLoader wrapping the ECOT RLDS dataset.
    """
    # 1. Construct dataset
    dataset = get_vla_dataset_ecot(cfg)

    # 2. Extract DataLoader parameters (safe access using _extract)
    from .config import _extract
    
    datasets_cfg = _extract(cfg, "datasets", {})
    vla_data_cfg = _extract(datasets_cfg, "vla_data", {})
    batch_size = _extract(vla_data_cfg, "per_device_batch_size", 1)
    num_workers = _extract(vla_data_cfg, "num_workers", 4)  # Default to 4, consistent with LeRobot

    # 3. Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn_ecot,
    )

    # 4. Save dataset statistics (on rank 0 only, for distributed training)
    output_dir_raw = _extract(cfg, "output_dir", None)
    if output_dir_raw:
        # Check if distributed training is initialized
        if not dist.is_initialized() or dist.get_rank() == 0:
            output_dir = Path(output_dir_raw)
            save_dataset_statistics(output_dir, dataset.dataset_statistics)
    else:
        logger.warning(
            "[ECOT RLDS] cfg.output_dir not found. Dataset statistics will not be saved. "
            "This is OK for testing, but statistics should be saved during training."
        )

    return dataloader


def save_dataset_statistics(run_dir: Path, stats: dict) -> None:
    """
    Save dataset statistics to JSON file for inference (action denormalization).

    The statistics are already cleaned and JSON-serializable (via ECOTRLDSDataset._clean_statistics).
    This function saves the statistics in the same format as LeRobot datasets for consistency.

    Parameters
    ----------
    run_dir:
        Path to training run directory (will save dataset_statistics.json here).
    stats:
        Statistics dictionary from dataset.dataset_statistics.
        Expected format: {dataset_name: {action: {...}, proprio: {...}, num_transitions: ..., num_trajectories: ...}}
        All values should already be JSON-serializable (numpy arrays converted to lists, etc.).
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "dataset_statistics.json"

    try:
        # Statistics are already cleaned by ECOTRLDSDataset._clean_statistics
        # so they should be directly JSON-serializable
        with open(out_path, "w") as f:
            json.dump(stats, f, indent=2)
        
        # Log summary of saved statistics
        logger.info(f"[ECOT RLDS] Saved dataset statistics to {out_path}")
        if stats:
            logger.info(f"[ECOT RLDS] Statistics summary:")
            for dataset_name, dataset_stats in stats.items():
                num_transitions = dataset_stats.get("num_transitions", "unknown")
                num_trajectories = dataset_stats.get("num_trajectories", "unknown")
                logger.info(f"  {dataset_name}: {num_transitions} transitions, {num_trajectories} trajectories")
                if "action" in dataset_stats:
                    action_stats = dataset_stats["action"]
                    if "q01" in action_stats and "q99" in action_stats:
                        logger.info(f"    Action normalization: q01={len(action_stats['q01'])} dims, q99={len(action_stats['q99'])} dims")
    except Exception as e:
        logger.error(f"[ECOT RLDS] Failed to save dataset statistics to {out_path}: {e}")
        logger.exception("  Full traceback:")
        # Don't raise - statistics saving is not critical for training
        # But log the error so it can be investigated

