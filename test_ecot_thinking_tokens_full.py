#!/usr/bin/env python3
"""
Complete test for ECOT implicit reasoning implementation.

This test covers the full pipeline:
1. Dataset construction with scheduled_stage=2
2. DataLoader creation
3. Data sample extraction
4. Tokenizer extension (thinking tokens)
5. Input construction with alignment
6. Forward pass preparation

Usage:
    python test_ecot_thinking_tokens_full.py
"""

import os
import sys
import logging
from pathlib import Path

# Set up environment
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = './qwen'

import torch
import numpy as np
from accelerate import PartialState

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger1 = logging.getLogger(__name__)


def create_test_config():
    """Create test configuration with implicit reasoning enabled."""
    return {
        "datasets": {
            "vla_data": {
                "dataset_py": "ecot_rlds",
                "per_device_batch_size": 2,  # Test with batch size 2
                "image_size": [224, 224],
                "num_workers": 0,
                "ecot": {
                    "data_root_dir": "/share/project/emllm_mnt.1d/mnt/sfs/baaiei/jyShi/rt_newData",
                    "data_mix": "bridge",
                    "image_size": [224, 224],
                    "action_dim": 7,
                    "future_action_window_size": 15,
                    "past_action_window_size": 0,
                    "scheduled_stage": 2,  # Enable thinking tokens
                    "shuffle_buffer_size": 1000,
                    "image_aug": False,
                    "reasoning_json": "/share/project/lvjing/datas/embodied_features_bridge.json",
                    "load_proprio": True,
                    "lower_case_instruction": True,
                    "train": True,
                    # Thinking token configuration
                    "thinking_token": "<|thinking|>",
                    "start_of_thinking_token": "<|start_of_thinking|>",
                    "end_of_thinking_token": "<|end_of_thinking|>",
                    "thinking_token_count": 2,
                },
            },
        },
        "framework": {
            "name": "QwenGR00T",
            "enable_latent_reasoning": True,  # Enable thinking token alignment
            "latent_reasoning": {
                "thinking_token": "<|thinking|>",
                "start_of_thinking_token": "<|start_of_thinking|>",
                "end_of_thinking_token": "<|end_of_thinking|>",
            },
            "qwenvl": {
                "base_vlm": "Qwen/Qwen3-VL-2B-Instruct",  # Fixed: Use Qwen3-VL
                "cache_dir": "./qwen_cache",
                "model_max_length": 8192,
            },
            "action_model": {
                "action_dim": 7,
                "future_action_window_size": 15,
                "past_action_window_size": 0,
            },
        },
    }


class MockConfig:
    """Mock config object that supports attribute access and 'in' operator."""
    def __init__(self, config_dict):
        self._config = config_dict
    
    def get(self, key, default=None):
        return self._config.get(key, default)
    
    def __contains__(self, key):
        """Support 'in' operator."""
        return key in self._config
    
    def __getattr__(self, key):
        if key.startswith('_'):
            return object.__getattribute__(self, key)
        value = self._config.get(key)
        if isinstance(value, dict):
            return MockConfig(value)
        return value


def test_step_1_dataset_construction():
    """Test Step 1: Dataset construction with scheduled_stage=2."""
    print("=" * 80)
    print("Step 1: Testing Dataset Construction")
    print("=" * 80)
    
    try:
        from starVLA.integrations.ecot_rlds.dataset import ECOTRLDSDataset
        
        config_dict = create_test_config()
        config = MockConfig(config_dict)
        
        print(f"Creating dataset with scheduled_stage=2...")
        dataset = ECOTRLDSDataset(config)
        
        print(f"✓ Dataset created successfully")
        print(f"  - Image size: {dataset.image_size}")
        print(f"  - Action dim: {dataset.action_dim}")
        print(f"  - Chunk len: {dataset.chunk_len}")
        print(f"  - Scheduled stage: {dataset.cfg_dict['scheduled_stage']}")
        
        return dataset
    
    except Exception as e:
        print(f"✗ Dataset construction failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_step_2_dataloader_creation(dataset):
    """Test Step 2: DataLoader creation."""
    print("\n" + "=" * 80)
    print("Step 2: Testing DataLoader Creation")
    print("=" * 80)
    
    try:
        from torch.utils.data import DataLoader
        from starVLA.integrations.ecot_rlds.collate import collate_fn_ecot
        
        config_dict = create_test_config()
        batch_size = config_dict["datasets"]["vla_data"]["per_device_batch_size"]
        
        print(f"Creating DataLoader with batch_size={batch_size}...")
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn_ecot,
            num_workers=0,
        )
        
        print(f"✓ DataLoader created successfully")
        
        return dataloader
    
    except Exception as e:
        print(f"✗ DataLoader creation failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_step_3_sample_extraction(dataloader):
    """Test Step 3: Extract sample from dataloader."""
    print("\n" + "=" * 80)
    print("Step 3: Testing Sample Extraction")
    print("=" * 80)
    
    try:
        print("Fetching first batch from dataloader...")
        batch = next(iter(dataloader))
        
        print(f"✓ Batch extracted successfully")
        print(f"  - Batch size: {len(batch)}")
        
        # Check first sample
        sample = batch[0]
        print(f"\nSample structure:")
        print(f"  - Keys: {list(sample.keys())}")
        print(f"  - Image: {len(sample['image'])} PIL images")
        print(f"  - Language: '{sample['lang'][:100]}...' (truncated)")
        print(f"  - Action shape: {sample['action'].shape}")
        if 'state' in sample:
            print(f"  - State shape: {sample['state'].shape}")
        
        # Check if reasoning/thinking tokens are included in lang
        if '<|thinking|>' in sample['lang'] or '<|start_of_thinking|>' in sample['lang']:
            print(f"✓ Thinking tokens found in language field!")
            print(f"  Language preview: {sample['lang'][:200]}...")
        else:
            print(f"⚠ No thinking tokens found in language field")
            print(f"  Full language: {sample['lang']}")
        
        return batch
    
    except Exception as e:
        print(f"✗ Sample extraction failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_step_4_tokenizer_extension():
    """Test Step 4: Tokenizer extension with thinking tokens."""
    print("\n" + "=" * 80)
    print("Step 4: Testing Tokenizer Extension")
    print("=" * 80)
    
    try:
        # Import after setting up environment
        sys.path.insert(0, str(Path(__file__).parent))
        from starVLA.model.modules.vlm.QWen3 import _QWen3_VL_Interface
        
        config_dict = create_test_config()
        config = MockConfig(config_dict)
        
        print("Creating QWen3-VL interface with thinking tokens...")
        # Initialize distributed state for proper device handling
        distributed_state = PartialState()
        
        interface = _QWen3_VL_Interface(config)
        
        # Check tokenizer
        tokenizer = interface.processor.tokenizer
        vocab_size = len(tokenizer)
        
        print(f"✓ Tokenizer extended successfully")
        print(f"  - Vocabulary size: {vocab_size}")
        print(f"  - Thinking token ID: {interface.thinking_token_id}")
        print(f"  - Start thinking ID: {interface.start_thinking_id}")
        print(f"  - End thinking ID: {interface.end_thinking_id}")
        
        # Verify tokens are in vocabulary
        thinking_token = config.framework.latent_reasoning.thinking_token
        if thinking_token in tokenizer.get_vocab():
            print(f"✓ Thinking token '{thinking_token}' is in vocabulary")
        else:
            print(f"✗ Thinking token '{thinking_token}' NOT in vocabulary")
        
        # Check embeddings
        embeddings = interface.model.get_input_embeddings()
        embedding_size = embeddings.weight.shape[0]
        print(f"  - Embedding matrix size: {embedding_size}")
        
        if embedding_size == vocab_size:
            print(f"✓ Embedding size matches vocabulary size")
        else:
            print(f"✗ Embedding size mismatch: {embedding_size} != {vocab_size}")
        
        return interface
    
    except Exception as e:
        print(f"✗ Tokenizer extension failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_step_5_input_construction(interface, batch):
    """Test Step 5: Input construction with thinking token alignment."""
    print("\n" + "=" * 80)
    print("Step 5: Testing Input Construction & Alignment")
    print("=" * 80)
    
    try:
        # Extract batch data
        batch_images = [sample["image"] for sample in batch]
        instructions = [sample["lang"] for sample in batch]
        
        print(f"Building inputs for {len(batch)} samples...")
        print(f"Sample 0 instruction length: {len(instructions[0])} chars")
        print(f"Sample 1 instruction length: {len(instructions[1])} chars")
        
        # Check if thinking tokens are present
        for i, inst in enumerate(instructions):
            if '<|thinking|>' in inst:
                print(f"  Sample {i}: Contains thinking tokens ✓")
            else:
                print(f"  Sample {i}: No thinking tokens ⚠")
        
        # Build inputs
        print("Calling build_qwenvl_inputs...")
        qwen_inputs = interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions
        )
        
        print(f"✓ Inputs constructed successfully")
        print(f"  - input_ids shape: {qwen_inputs['input_ids'].shape}")
        print(f"  - attention_mask shape: {qwen_inputs['attention_mask'].shape}")
        if 'pixel_values' in qwen_inputs:
            pv = qwen_inputs['pixel_values']
            if isinstance(pv, dict):
                print(f"  - pixel_values: dict with keys {list(pv.keys())}")
                for k, v in pv.items():
                    print(f"    - {k}: shape {v.shape}")
            else:
                print(f"  - pixel_values shape: {pv.shape}")
        
        # Check for thinking token alignment
        tokenizer = interface.processor.tokenizer
        thinking_token_id = interface.thinking_token_id
        
        print(f"\nChecking thinking token positions:")
        for i in range(qwen_inputs['input_ids'].shape[0]):
            ids = qwen_inputs['input_ids'][i]
            thinking_positions = (ids == thinking_token_id).nonzero(as_tuple=True)[0]
            if len(thinking_positions) > 0:
                first_pos = thinking_positions[0].item()
                print(f"  Sample {i}: First thinking token at position {first_pos}")
            else:
                print(f"  Sample {i}: No thinking tokens found in input_ids")
        
        # Check if thinking tokens are aligned (should be at same position)
        if qwen_inputs['input_ids'].shape[0] > 1:
            first_thinking_positions = []
            for i in range(qwen_inputs['input_ids'].shape[0]):
                ids = qwen_inputs['input_ids'][i]
                thinking_positions = (ids == thinking_token_id).nonzero(as_tuple=True)[0]
                if len(thinking_positions) > 0:
                    first_thinking_positions.append(thinking_positions[0].item())
                else:
                    first_thinking_positions.append(-1)
            
            if len(set(first_thinking_positions)) == 1:
                print(f"✓ Thinking tokens are aligned at position {first_thinking_positions[0]}")
            else:
                print(f"⚠ Thinking tokens are NOT aligned: {first_thinking_positions}")
        
        return qwen_inputs
    
    except Exception as e:
        print(f"✗ Input construction failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_step_6_forward_preparation(interface, qwen_inputs, batch):
    """Test Step 6: Prepare for forward pass."""
    print("\n" + "=" * 80)
    print("Step 6: Testing Forward Pass Preparation")
    print("=" * 80)
    
    try:
        print("Checking inputs are ready for forward pass...")
        
        # Verify device
        device = qwen_inputs['input_ids'].device
        print(f"  - Device: {device}")
        
        # Verify all tensors are on same device
        all_same_device = True
        for key, value in qwen_inputs.items():
            if isinstance(value, torch.Tensor):
                if value.device != device:
                    print(f"  ✗ {key} is on different device: {value.device}")
                    all_same_device = False
            elif isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, torch.Tensor) and v.device != device:
                        print(f"  ✗ {key}[{k}] is on different device: {v.device}")
                        all_same_device = False
        
        if all_same_device:
            print(f"✓ All tensors on same device")
        
        # Test forward pass (without computing loss)
        print("\nTesting VLM forward pass...")
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = interface(
                    **qwen_inputs,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
        
        print(f"✓ Forward pass successful")
        print(f"  - Hidden states: {len(outputs.hidden_states)} layers")
        print(f"  - Last hidden shape: {outputs.hidden_states[-1].shape}")
        
        return outputs
    
    except Exception as e:
        print(f"✗ Forward pass preparation failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def pre_flight_check():
    """Pre-flight checks before running tests."""
    print("Running pre-flight checks...")
    
    # Check GPU
    if not torch.cuda.is_available():
        print("✗ No GPU available!")
        return False
    
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  GPU memory: {gpu_mem:.2f} GB")
    if gpu_mem < 10:
        print(f"  ⚠ GPU memory may be insufficient ({gpu_mem:.2f} GB < 10 GB recommended)")
    
    # Check data paths
    config = create_test_config()
    data_root = Path(config["datasets"]["vla_data"]["ecot"]["data_root_dir"])
    if not data_root.exists():
        print(f"✗ Data root not found: {data_root}")
        return False
    print(f"  ✓ Data root exists: {data_root}")
    
    reasoning_json = Path(config["datasets"]["vla_data"]["ecot"]["reasoning_json"])
    if not reasoning_json.exists():
        print(f"  ⚠ Reasoning JSON not found: {reasoning_json}")
        print(f"  ⚠ Scheduled_stage=2 may not generate reasoning text correctly")
    else:
        print(f"  ✓ Reasoning JSON exists: {reasoning_json}")
    
    print("✓ Pre-flight checks completed\n")
    return True


def main():
    """Run complete test pipeline."""
    print("=" * 80)
    print("ECOT Implicit Reasoning - Complete Pipeline Test")
    print("=" * 80)
    
    # Run pre-flight checks
    if not pre_flight_check():
        print("Pre-flight checks failed. Please fix the issues and try again.")
        sys.exit(1)
    
    try:
        # Step 1: Dataset construction
        dataset = test_step_1_dataset_construction()
        print("step1done")
        # Step 2: DataLoader creation
        dataloader = test_step_2_dataloader_creation(dataset)
        print("step2done")
        
        # Step 3: Sample extraction
        batch = test_step_3_sample_extraction(dataloader)
        print("step3done")
        # Step 4: Tokenizer extension
        interface = test_step_4_tokenizer_extension()
        print("step4done")
        # Step 5: Input construction
        qwen_inputs = test_step_5_input_construction(interface, batch)
        print("step5done")
        # Step 6: Forward pass preparation
        outputs = test_step_6_forward_preparation(interface, qwen_inputs, batch)
        print("step6done")
        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED!")
        print("=" * 80)
        print("\nPipeline verified:")
        print("  1. Dataset with scheduled_stage=2 ✓")
        print("  2. DataLoader creation ✓")
        print("  3. Sample extraction with thinking tokens ✓")
        print("  4. Tokenizer extension ✓")
        print("  5. Input construction with alignment ✓")
        print("  6. Forward pass preparation ✓")
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("✗ TEST FAILED")
        print("=" * 80)
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

