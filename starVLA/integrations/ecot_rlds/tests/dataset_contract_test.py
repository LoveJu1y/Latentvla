#!/usr/bin/env python3
"""
Unit tests for ECOT RLDS dataset contract validation.

This module provides strict contract tests to verify that dataset samples
conform to the expected format for StarVLA training.

Task 2.9: Dataset contract tests
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from omegaconf import OmegaConf
from starVLA.integrations.ecot_rlds.dataset import ECOTRLDSDataset
from starVLA.integrations.ecot_rlds.builder import get_vla_dataset_ecot

# Default test configuration
DEFAULT_DATA_ROOT_DIR = "/share/project/emllm_mnt.1d/mnt/sfs/baaiei/jyShi/rt_newData"
DEFAULT_DATA_MIX = "bridge"


def create_test_config(
    data_root_dir: str = DEFAULT_DATA_ROOT_DIR,
    data_mix: str = DEFAULT_DATA_MIX,
    image_size: list = [224, 224],
    action_dim: int = 7,
    future_action_window_size: int = 15,
    past_action_window_size: int = 0,
) -> dict:
    """Create a minimal test configuration."""
    return {
        "datasets": {
            "vla_data": {
                "dataset_py": "ecot_rlds",
                "image_size": image_size,
                "per_device_batch_size": 1,
                "num_workers": 0,
                "ecot": {
                    "data_root_dir": data_root_dir,
                    "data_mix": data_mix,
                    "scheduled_stage": 0,
                    "shuffle_buffer_size": 1000,  # Small for testing
                    "image_aug": False,
                    "load_proprio": True,
                    "lower_case_instruction": True,
                    "train": True,
                },
            },
        },
        "framework": {
            "action_model": {
                "action_dim": action_dim,
                "future_action_window_size": future_action_window_size,
                "past_action_window_size": past_action_window_size,
            },
        },
    }


@pytest.mark.skip(reason="Requires RLDS data and TensorFlow - run manually with test_dataset_load.py")
def test_ecot_rlds_dataset_contract():
    """
    Test that dataset samples conform to the expected contract.
    
    Expected fields:
    - image: List[PIL.Image], at least one image
    - lang: str, non-empty
    - action: np.ndarray, shape [chunk_len, action_dim]
    - reasoning: str (may be empty)
    - reasoning_subset: str (kept for compatibility)
    - dataset_name: str
    - state: np.ndarray, shape [state_dim] (optional, if load_proprio=True)
    """
    # Load test configuration
    cfg_dict = create_test_config()
    cfg = OmegaConf.create(cfg_dict)
    
    # Create dataset
    dataset = ECOTRLDSDataset(cfg)
    
    # Verify dataset has length
    assert len(dataset) > 0, "Dataset should have non-zero length"
    
    # Get chunk parameters
    chunk_len = dataset.chunk_len
    action_dim = dataset.action_dim
    
    # Take a sample
    sample = next(iter(dataset))
    
    # 1. Verify image field
    assert "image" in sample, "Sample must contain 'image' field"
    assert isinstance(sample["image"], list), f"image should be list, got {type(sample['image'])}"
    assert len(sample["image"]) > 0, "image list should not be empty"
    
    from PIL import Image
    img = sample["image"][0]
    assert isinstance(img, Image.Image), f"image[0] should be PIL.Image, got {type(img)}"
    # PIL.Image.size returns (width, height)
    assert img.size == (224, 224), f"Expected image size (224, 224), got {img.size}"
    
    # 2. Verify lang field
    assert "lang" in sample, "Sample must contain 'lang' field"
    assert isinstance(sample["lang"], str), f"lang should be str, got {type(sample['lang'])}"
    assert len(sample["lang"]) > 0, "lang should not be empty"
    
    # 3. Verify action field
    assert "action" in sample, "Sample must contain 'action' field"
    import numpy as np
    action = sample["action"]
    assert isinstance(action, np.ndarray), f"action should be np.ndarray, got {type(action)}"
    assert action.shape == (chunk_len, action_dim), \
        f"action shape should be ({chunk_len}, {action_dim}), got {action.shape}"
    assert action.dtype == np.float32, f"action dtype should be float32, got {action.dtype}"
    
    # 4. Verify reasoning field
    assert "reasoning" in sample, "Sample must contain 'reasoning' field"
    assert isinstance(sample["reasoning"], str), f"reasoning should be str, got {type(sample['reasoning'])}"
    # reasoning may be empty, which is acceptable
    
    # 5. Verify reasoning_subset field
    assert "reasoning_subset" in sample, "Sample must contain 'reasoning_subset' field"
    assert isinstance(sample["reasoning_subset"], str), \
        f"reasoning_subset should be str, got {type(sample['reasoning_subset'])}"
    
    # 6. Verify dataset_name field
    assert "dataset_name" in sample, "Sample must contain 'dataset_name' field"
    assert isinstance(sample["dataset_name"], str), \
        f"dataset_name should be str, got {type(sample['dataset_name'])}"
    
    # 7. Verify state field (optional, if load_proprio=True)
    if "state" in sample:
        state = sample["state"]
        assert isinstance(state, np.ndarray), f"state should be np.ndarray, got {type(state)}"
        assert state.ndim == 1, f"state should be 1D, got shape {state.shape}"
        assert state.dtype == np.float32, f"state dtype should be float32, got {state.dtype}"
        assert state.shape[0] > 0, "state should have positive dimension"


@pytest.mark.skip(reason="Requires RLDS data and TensorFlow - run manually with test_dataset_load.py")
def test_dataset_statistics():
    """Test that dataset statistics are available and properly formatted."""
    cfg_dict = create_test_config()
    cfg = OmegaConf.create(cfg_dict)
    
    dataset = ECOTRLDSDataset(cfg)
    stats = dataset.dataset_statistics
    
    # Statistics should be a dict
    assert isinstance(stats, dict), "dataset_statistics should be a dict"
    
    # Should contain at least some keys
    assert len(stats) > 0, "dataset_statistics should not be empty"
    
    # Statistics should be JSON-serializable (no numpy types)
    import json
    try:
        json.dumps(stats)
    except (TypeError, ValueError) as e:
        pytest.fail(f"dataset_statistics should be JSON-serializable: {e}")


@pytest.mark.skip(reason="Requires RLDS data and TensorFlow - run manually with test_dataset_load.py")
def test_multiple_samples():
    """Test that multiple samples can be retrieved and are consistent."""
    cfg_dict = create_test_config()
    cfg = OmegaConf.create(cfg_dict)
    
    dataset = ECOTRLDSDataset(cfg)
    
    # Get multiple samples
    samples = []
    for i, sample in enumerate(dataset):
        if i >= 3:  # Test 3 samples
            break
        samples.append(sample)
    
    assert len(samples) == 3, "Should be able to retrieve 3 samples"
    
    # Verify all samples have the same structure
    for i, sample in enumerate(samples):
        assert "image" in sample, f"Sample {i} missing 'image'"
        assert "lang" in sample, f"Sample {i} missing 'lang'"
        assert "action" in sample, f"Sample {i} missing 'action'"
        assert sample["action"].shape == (dataset.chunk_len, dataset.action_dim), \
            f"Sample {i} has incorrect action shape"


if __name__ == "__main__":
    # Allow running directly without pytest
    print("Note: These tests require RLDS data and TensorFlow.")
    print("For manual testing, use: python -m starVLA.integrations.ecot_rlds.tests.test_dataset_load")
    print("\nTo run with pytest:")
    print("  pytest starVLA/integrations/ecot_rlds/tests/dataset_contract_test.py -v")
