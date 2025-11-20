#!/usr/bin/env python3
"""
Test script to verify ECOT RLDS dataset can be loaded and analyze reasoning data.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from starVLA.integrations.ecot_rlds.dataset import ECOTRLDSDataset
from starVLA.integrations.ecot_rlds.builder import get_vla_dataset_ecot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_DATA_ROOT_DIR = "/share/project/emllm_mnt.1d/mnt/sfs/baaiei/jyShi/rt_newData"
DEFAULT_DATA_MIX = "bridge"


def create_minimal_config(
    data_root_dir: str,
    data_mix: str,
    image_size: list = [224, 224],
    action_dim: int = 7,
    future_action_window_size: int = 15,
    past_action_window_size: int = 0,
    all_in_ecot: bool = True,
) -> dict:
    """
    Create a minimal configuration for testing.
    
    Parameters
    ----------
    all_in_ecot: bool
        If True, all parameters are in ecot.* (simple and straightforward).
        If False, action parameters are in framework.* and image_size in datasets.vla_data.*
        (maintains QwenGR00T config structure, ecot.* takes priority if specified).
    """
    if all_in_ecot:
        # Option 1: All ECOT configuration in ecot.* (simple and straightforward)
        return {
            "datasets": {
                "vla_data": {
                    "dataset_py": "ecot_rlds",
                    "per_device_batch_size": 1,
                    "image_size": image_size,
                    "num_workers": 0,
                    "ecot": {
                        # All ECOT configuration is here - simple and straightforward
                        "data_root_dir": data_root_dir,
                        "data_mix": data_mix,
                        "image_size": image_size,
                        "action_dim": action_dim,
                        "future_action_window_size": future_action_window_size,
                        "past_action_window_size": past_action_window_size,
                        "scheduled_stage": 1,
                        "shuffle_buffer_size": 1000,  # Small for testing
                        "image_aug": False,
                        "reasoning_json": "/share/project/lvjing/datas/embodied_features_bridge.json",
                        "load_proprio": True,
                        "lower_case_instruction": True,
                        "train": True,
                        # Thinking token config (optional - uses defaults if not specified)
                        # "thinking_token": "<|thinking|>",
                        # "start_of_thinking_token": "<|start_of_thinking|>",
                        # "end_of_thinking_token": "<|end_of_thinking|>",
                        # "thinking_token_count": 2,
                    },
                },
            },
        }
    else:
        # Option 2: Maintain QwenGR00T config structure (framework.* and datasets.vla_data.*)
        # ecot.* only contains ECOT-specific parameters
        # Action parameters and image_size are read from global config (framework.* and datasets.vla_data.*)
        return {
            "datasets": {
                "vla_data": {
                    "dataset_py": "ecot_rlds",
                    "image_size": image_size,  # In datasets.vla_data.* (QwenGR00T structure)
                    "per_device_batch_size": 1,
                    "num_workers": 0,
                    "ecot": {
                        # Only ECOT-specific parameters here
                        "data_root_dir": data_root_dir,
                        "data_mix": data_mix,
                        "scheduled_stage": 1,
                        "shuffle_buffer_size": 1000,  # Small for testing
                        "image_aug": False,
                        "reasoning_json": "/share/project/lvjing/datas/embodied_features_bridge.json",
                        "load_proprio": True,
                        "lower_case_instruction": True,
                        "train": True,
                        # Action parameters and image_size are read from framework.* and datasets.vla_data.*
                        # (ecot.* takes priority if specified)
                    },
                },
            },
            "framework": {
                "action_model": {
                    # Action parameters in framework.* (QwenGR00T structure)
                    "action_dim": action_dim,
                    "future_action_window_size": future_action_window_size,
                    "past_action_window_size": past_action_window_size,
                },
            },
        }


def analyze_dataset(dataset: ECOTRLDSDataset, max_samples: int = None):
    """
    Analyze dataset and collect statistics, focusing on reasoning field.
    """
    logger.info("=" * 80)
    logger.info("Analyzing ECOT RLDS Dataset")
    logger.info("=" * 80)
    
    # Statistics
    stats = {
        "total_samples": 0,
        "reasoning": {
            "lengths": [],
            "non_empty_count": 0,
            "empty_count": 0,
            "with_cot_tokens": 0,
            "multi_line": 0,
            "examples": [],
        },
        "action": {
            "shapes": [],
            "ranges": [],
        },
        "has_state": 0,
    }
    
    logger.info(f"\nIterating through dataset (max_samples={max_samples or 'all'})...")
    
    # Iterate through all samples
    for i, sample in enumerate(dataset):
        if max_samples and i >= max_samples:
            break
        
        stats["total_samples"] += 1
        
        # Analyze reasoning (重点)
        reasoning = sample.get("reasoning", "")
        reasoning_len = len(reasoning)
        stats["reasoning"]["lengths"].append(reasoning_len)
        
        # Log first sample in detail with reasoning
        if i == 0:
            logger.info(f"\n--- Sample 0 (First Sample) ---")
            logger.info(f"  image: {len(sample.get('image', []))} views")
            logger.info(f"  lang: {sample.get('lang', '')[:60]}...")
            logger.info(f"  action: shape={sample.get('action', np.array([])).shape}")
            logger.info(f"  dataset_name: {sample.get('dataset_name', 'unknown')}")
            if "state" in sample:
                logger.info(f"  state: shape={sample['state'].shape}")
            logger.info(f"  reasoning: length={reasoning_len}")
            if reasoning_len > 0:
                logger.info(f"    Preview: {reasoning[:300]}...")
                if "\n" in reasoning:
                    lines = reasoning.split("\n")
                    logger.info(f"    Lines: {len(lines)}")
                    for line_idx, line in enumerate(lines[:5]):
                        if line.strip():
                            logger.info(f"      [{line_idx+1}] {line[:100]}")
        
        # Collect reasoning statistics
        if reasoning_len > 0:
            stats["reasoning"]["non_empty_count"] += 1
            reasoning_stripped = reasoning.strip()
            if len(reasoning_stripped) > 0:
                if "TH" in reasoning or "ST" in reasoning or "MV" in reasoning:
                    stats["reasoning"]["with_cot_tokens"] += 1
                if "\n" in reasoning:
                    stats["reasoning"]["multi_line"] += 1
                # Collect first 10 examples for detailed analysis
                if len(stats["reasoning"]["examples"]) < 10:
                    stats["reasoning"]["examples"].append({
                        "index": i,
                        "length": reasoning_len,
                        "preview": reasoning[:400],
                        "has_cot_tokens": "TH" in reasoning or "ST" in reasoning or "MV" in reasoning,
                        "is_multi_line": "\n" in reasoning,
                    })
        else:
            stats["reasoning"]["empty_count"] += 1
        
        # Analyze action
        action = sample.get("action")
        if action is not None:
            stats["action"]["shapes"].append(action.shape)
            stats["action"]["ranges"].append((action.min(), action.max()))
        
        # Check state
        if "state" in sample:
            stats["has_state"] += 1
        
        # Progress log
        if (i + 1) % 1000 == 0:
            logger.info(f"  Processed {i+1} samples...")
    
    # Print statistics
    logger.info(f"\n{'='*80}")
    logger.info(f"Dataset Statistics (total samples: {stats['total_samples']})")
    logger.info(f"{'='*80}")
    
    # Reasoning statistics
    reasoning_stats = stats["reasoning"]
    if reasoning_stats["lengths"]:
        avg_length = sum(reasoning_stats["lengths"]) / len(reasoning_stats["lengths"])
        max_length = max(reasoning_stats["lengths"])
        min_length = min(reasoning_stats["lengths"])
        
        logger.info(f"\nReasoning Statistics:")
        logger.info(f"  Total samples: {stats['total_samples']}")
        logger.info(f"  Non-empty: {reasoning_stats['non_empty_count']} ({100*reasoning_stats['non_empty_count']/stats['total_samples']:.1f}%)")
        logger.info(f"  Empty: {reasoning_stats['empty_count']} ({100*reasoning_stats['empty_count']/stats['total_samples']:.1f}%)")
        logger.info(f"  Average length: {avg_length:.1f} chars")
        logger.info(f"  Min length: {min_length} chars")
        logger.info(f"  Max length: {max_length} chars")
        logger.info(f"  With CoT tokens (TH/ST/MV): {reasoning_stats['with_cot_tokens']} ({100*reasoning_stats['with_cot_tokens']/stats['total_samples']:.1f}%)")
        logger.info(f"  Multi-line: {reasoning_stats['multi_line']} ({100*reasoning_stats['multi_line']/stats['total_samples']:.1f}%)")
        
        # Show examples with detailed info
        if reasoning_stats["examples"]:
            logger.info(f"\n  Example Reasoning Texts (first {len(reasoning_stats['examples'])}):")
            for idx, example in enumerate(reasoning_stats["examples"], 1):
                logger.info(f"    [{example['index']}] Length: {example['length']} chars | "
                           f"CoT tokens: {example['has_cot_tokens']} | "
                           f"Multi-line: {example['is_multi_line']}")
                logger.info(f"        {example['preview']}...")
    
    # Action statistics (simplified)
    if stats["action"]["shapes"]:
        unique_shapes = set(stats["action"]["shapes"])
        logger.info(f"\nAction Statistics:")
        logger.info(f"  Unique shapes: {unique_shapes}")
        if stats["action"]["ranges"]:
            all_mins = [r[0] for r in stats["action"]["ranges"]]
            all_maxs = [r[1] for r in stats["action"]["ranges"]]
            logger.info(f"  Value range: [{min(all_mins):.3f}, {max(all_maxs):.3f}]")
    
    # State statistics (simplified)
    if stats["has_state"] > 0:
        logger.info(f"\nState Statistics:")
        logger.info(f"  Samples with state: {stats['has_state']} ({100*stats['has_state']/stats['total_samples']:.1f}%)")
    
    logger.info(f"\n{'='*80}\n")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Test ECOT RLDS dataset loading")
    parser.add_argument(
        "--data_root_dir",
        type=str,
        default=DEFAULT_DATA_ROOT_DIR,
        help=f"Root directory containing RLDS datasets (default: {DEFAULT_DATA_ROOT_DIR})",
    )
    parser.add_argument(
        "--data_mix",
        type=str,
        default=DEFAULT_DATA_MIX,
        help=f"Dataset mixture name or single dataset name (default: {DEFAULT_DATA_MIX})",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to analyze (default: all)",
    )
    parser.add_argument(
        "--config_yaml",
        type=str,
        default=None,
        help="Path to YAML config file (optional, overrides other args)",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        default=[224, 224],
        help="Image size [H, W] (default: 224 224)",
    )
    parser.add_argument(
        "--action_dim",
        type=int,
        default=7,
        help="Action dimension (default: 7)",
    )
    parser.add_argument(
        "--future_action_window_size",
        type=int,
        default=15,
        help="Future action window size (default: 15)",
    )
    
    args = parser.parse_args()
    
    # Load config from YAML or create minimal config
    if args.config_yaml:
        logger.info(f"Loading config from {args.config_yaml}")
        cfg = OmegaConf.load(args.config_yaml)
    else:
        logger.info("Creating minimal test config")
        cfg_dict = create_minimal_config(
            data_root_dir=args.data_root_dir,
            data_mix=args.data_mix,
            image_size=args.image_size,
            action_dim=args.action_dim,
            future_action_window_size=args.future_action_window_size,
        )
        cfg = OmegaConf.create(cfg_dict)
    
    # Safely extract config values for logging
    from starVLA.integrations.ecot_rlds.config import _extract
    datasets_cfg = _extract(cfg, "datasets", {})
    vla_data_cfg = _extract(datasets_cfg, "vla_data", {})
    ecot_cfg = _extract(vla_data_cfg, "ecot", {})
    data_mix = _extract(ecot_cfg, "data_mix", "unknown")
    image_size = _extract(vla_data_cfg, "image_size", _extract(ecot_cfg, "image_size", [224, 224]))
    
    # Safely extract action_dim (could be in ecot.* or framework.*)
    action_dim_ecot = _extract(ecot_cfg, "action_dim", None)
    if action_dim_ecot is None:
        framework_cfg = _extract(cfg, "framework", {})
        action_model_cfg = _extract(framework_cfg, "action_model", {})
        action_dim = _extract(action_model_cfg, "action_dim", 7)
    else:
        action_dim = action_dim_ecot
    
    logger.info(f"Config: data_mix={data_mix}, "
               f"image_size={image_size}, "
               f"action_dim={action_dim}, "
               f"max_samples={args.max_samples or 'all'}")
    
    try:
        # Create dataset
        logger.info("Creating dataset...")
        dataset = get_vla_dataset_ecot(cfg)
        logger.info(f"✓ Dataset created: length={len(dataset)}")
        
        # Analyze dataset (default: analyze all samples)
        analyze_dataset(dataset, max_samples=args.max_samples)
        
        logger.info("✅ Analysis complete!")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

