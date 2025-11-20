#!/usr/bin/env python3
"""
Test script to verify ECOT RLDS builder functions work correctly.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from starVLA.integrations.ecot_rlds.builder import get_vla_dataset_ecot, make_dataloader_ecot, save_dataset_statistics

DEFAULT_DATA_ROOT_DIR = "/share/project/emllm_mnt.1d/mnt/sfs/baaiei/jyShi/rt_newData"
DEFAULT_DATA_MIX = "bridge"


def create_test_config():
    """Create a minimal test configuration."""
    return {
        "datasets": {
            "vla_data": {
                "dataset_py": "ecot_rlds",
                "image_size": [224, 224],
                "per_device_batch_size": 2,
                "num_workers": 0,  # Use 0 for testing to avoid multiprocessing issues
                "ecot": {
                    "data_root_dir": DEFAULT_DATA_ROOT_DIR,
                    "data_mix": DEFAULT_DATA_MIX,
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
                "action_dim": 7,
                "future_action_window_size": 15,
                "past_action_window_size": 0,
            },
        },
    }


def test_get_vla_dataset_ecot():
    """Test that get_vla_dataset_ecot returns a dataset."""
    print("=" * 80)
    print("Test: get_vla_dataset_ecot")
    print("=" * 80)
    
    cfg_dict = create_test_config()
    cfg = OmegaConf.create(cfg_dict)
    
    dataset = get_vla_dataset_ecot(cfg)
    
    assert dataset is not None, "Dataset should not be None"
    assert hasattr(dataset, "__len__"), "Dataset should have __len__"
    assert hasattr(dataset, "__iter__"), "Dataset should have __iter__"
    assert hasattr(dataset, "dataset_statistics"), "Dataset should have dataset_statistics property"
    
    print(f"✓ Dataset created: length={len(dataset)}")
    print(f"✓ Dataset statistics available: {list(dataset.dataset_statistics.keys())}")
    
    # Test iteration
    sample_count = 0
    for sample in dataset:
        sample_count += 1
        if sample_count >= 2:
            break
    
    print(f"✓ Dataset iteration works: got {sample_count} samples")
    print("✅ get_vla_dataset_ecot test passed!\n")


def test_make_dataloader_ecot():
    """Test that make_dataloader_ecot returns a DataLoader."""
    print("=" * 80)
    print("Test: make_dataloader_ecot")
    print("=" * 80)
    
    cfg_dict = create_test_config()
    cfg = OmegaConf.create(cfg_dict)
    
    # Test without output_dir (testing scenario)
    dataloader = make_dataloader_ecot(cfg)
    
    assert dataloader is not None, "DataLoader should not be None"
    assert hasattr(dataloader, "__iter__"), "DataLoader should have __iter__"
    
    print(f"✓ DataLoader created: batch_size={dataloader.batch_size}, num_workers={dataloader.num_workers}")
    
    # Test iteration
    batch_count = 0
    for batch in dataloader:
        batch_count += 1
        assert isinstance(batch, list), "Batch should be a list"
        assert len(batch) > 0, "Batch should not be empty"
        assert isinstance(batch[0], dict), "Batch elements should be dicts"
        assert "image" in batch[0], "Sample should have 'image' key"
        assert "lang" in batch[0], "Sample should have 'lang' key"
        assert "action" in batch[0], "Sample should have 'action' key"
        if batch_count >= 2:
            break
    
    print(f"✓ DataLoader iteration works: got {batch_count} batches")
    print("✅ make_dataloader_ecot test passed!\n")


def test_save_dataset_statistics():
    """Test that save_dataset_statistics saves statistics correctly."""
    print("=" * 80)
    print("Test: save_dataset_statistics")
    print("=" * 80)
    
    # Create mock statistics (cleaned format)
    mock_stats = {
        "bridge_data_v2": {
            "action": {
                "q01": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                "q99": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
                "mean": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                "std": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            },
            "proprio": {
                "q01": [0.0, 0.0, 0.0],
                "q99": [1.0, 1.0, 1.0],
            },
            "num_transitions": 1000,
            "num_trajectories": 100,
        }
    }
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "test_run"
        
        # Save statistics
        save_dataset_statistics(run_dir, mock_stats)
        
        # Verify file exists
        stats_file = run_dir / "dataset_statistics.json"
        assert stats_file.exists(), "Statistics file should exist"
        
        # Verify file content
        import json
        with open(stats_file, "r") as f:
            loaded_stats = json.load(f)
        
        assert loaded_stats == mock_stats, "Loaded statistics should match saved statistics"
        print(f"✓ Statistics saved to {stats_file}")
        print(f"✓ Statistics loaded and verified")
        print("✅ save_dataset_statistics test passed!\n")


def test_make_dataloader_with_output_dir():
    """Test that make_dataloader_ecot saves statistics when output_dir is provided."""
    print("=" * 80)
    print("Test: make_dataloader_ecot with output_dir")
    print("=" * 80)
    
    cfg_dict = create_test_config()
    cfg = OmegaConf.create(cfg_dict)
    
    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg.output_dir = tmpdir
        
        dataloader = make_dataloader_ecot(cfg)
        
        # Verify statistics file was created
        stats_file = Path(tmpdir) / "dataset_statistics.json"
        assert stats_file.exists(), "Statistics file should be created when output_dir is provided"
        
        # Verify file is valid JSON
        import json
        with open(stats_file, "r") as f:
            stats = json.load(f)
        
        assert isinstance(stats, dict), "Statistics should be a dict"
        assert len(stats) > 0, "Statistics should not be empty"
        
        print(f"✓ Statistics file created at {stats_file}")
        print(f"✓ Statistics contain {len(stats)} dataset(s)")
        print("✅ make_dataloader_ecot with output_dir test passed!\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("ECOT RLDS Builder Tests")
    print("=" * 80 + "\n")
    
    try:
        test_get_vla_dataset_ecot()
        test_make_dataloader_ecot()
        test_save_dataset_statistics()
        test_make_dataloader_with_output_dir()
        
        print("=" * 80)
        print("✅ All builder tests passed!")
        print("=" * 80)
        return 0
    except Exception as e:
        print(f"\n❌ Test failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

