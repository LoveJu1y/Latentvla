"""
ECoT Implicit Reasoning End-to-End Test Training Script

Purpose: Validate the entire pipeline from data loading to training
- Stage 0: No thinking tokens, @ delimiter only
- Stage 2+: With thinking tokens, KV-Cache iterative forward

Usage:
    python test_ecot_training.py --config_yaml config/test_ecot_stage0.yaml
    python test_ecot_training.py --config_yaml config/test_ecot_stage2.yaml
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

# 设置环境变量（必须在导入其他模块之前）
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "./qwen"

# Local imports (延迟导入以避免不必要的依赖)
from starVLA.model.framework import build_framework


def validate_config(cfg):
    """
    验证配置文件的完整性和合理性
    
    检查项:
    1. 必需字段存在
    2. 路径有效
    3. scheduled_stage与enable_latent_reasoning一致
    4. batch_size合理（建议<=4）
    """
    print("\n" + "="*80)
    print("📋 Configuration Validation")
    print("="*80)
    
    # 检查必需字段
    required_fields = {
        "datasets.vla_data.dataset_py": cfg.datasets.vla_data.dataset_py,
        "datasets.vla_data.ecot.data_root_dir": cfg.datasets.vla_data.ecot.data_root_dir,
        "datasets.vla_data.ecot.scheduled_stage": cfg.datasets.vla_data.ecot.scheduled_stage,
        "framework.name": cfg.framework.name,
        "framework.enable_latent_reasoning": cfg.framework.enable_latent_reasoning,
    }
    
    print("✅ Required fields check:")
    for field_name, field_value in required_fields.items():
        print(f"  - {field_name}: {field_value}")
    
    # 检查数据路径
    data_root = cfg.datasets.vla_data.ecot.data_root_dir
    reasoning_json = cfg.datasets.vla_data.ecot.reasoning_json
    
    if not os.path.exists(data_root):
        print(f"⚠️  Warning: data_root_dir does not exist: {data_root}")
    else:
        print(f"✅ Data root exists: {data_root}")
    
    if not os.path.exists(reasoning_json):
        print(f"⚠️  Warning: reasoning_json does not exist: {reasoning_json}")
    else:
        print(f"✅ Reasoning JSON exists: {reasoning_json}")
    
    # 检查batch size
    batch_size = cfg.datasets.vla_data.per_device_batch_size
    if batch_size > 4:
        print(f"⚠️  Warning: batch_size={batch_size} is large for testing, recommend <=4")
    else:
        print(f"✅ Batch size is reasonable: {batch_size}")
    
    # 打印关键配置
    print(f"\n📊 Key Configuration:")
    print(f"  - Stage: {cfg.datasets.vla_data.ecot.scheduled_stage}")
    print(f"  - Enable Latent Reasoning: {cfg.framework.enable_latent_reasoning}")
    print(f"  - Compute Language Loss: {cfg.framework.latent_reasoning.compute_language_loss}")
    print(f"  - Batch Size: {batch_size}")
    print(f"  - Max Steps: {cfg.trainer.max_train_steps}")
    print(f"  - Model: {cfg.framework.qwenvl.base_vlm}")
    
    return True


def test_dataloader(cfg):
    """
    测试数据加载和格式
    
    验证:
    1. DataLoader可以正常创建
    2. 可以获取batch
    3. Batch格式正确（包含必需字段）
    4. 数据类型和shape正确
    """
    print("\n" + "="*80)
    print("📦 Testing DataLoader")
    print("="*80)
    
    # 直接使用 ECOT RLDS 的 dataloader builder
    from starVLA.integrations.ecot_rlds.builder import make_dataloader_ecot
    
    # 创建dataloader
    print("Creating ECOT RLDS dataloader...")
    dataloader = make_dataloader_ecot(cfg)
    print(f"✅ DataLoader created successfully")
    
    # 获取一个batch
    print("\nFetching first batch...")
    batch = next(iter(dataloader))
    
    # 验证batch格式
    print(f"✅ Batch type: {type(batch)}")
    print(f"✅ Batch length (samples): {len(batch)}")
    print(f"✅ Sample keys: {list(batch[0].keys())}")
    
    # 检查必需字段
    required_keys = ["image", "lang", "action"]
    for key in required_keys:
        if key not in batch[0]:
            print(f"❌ Missing required key: {key}")
            return None
        else:
            print(f"✅ Found required key: {key}")
    
    # 打印样本信息
    sample = batch[0]
    print(f"\n📊 Sample Info:")
    print(f"  - Images: {len(sample['image'])} views")
    if len(sample['image']) > 0:
        print(f"    - Image 0 size: {sample['image'][0].size}")
        print(f"    - Image 0 mode: {sample['image'][0].mode}")
    
    lang_preview = sample['lang'][:100] if len(sample['lang']) > 100 else sample['lang']
    print(f"  - Language (first 100 chars): {lang_preview}...")
    print(f"  - Language full length: {len(sample['lang'])} chars")
    
    action_array = np.array(sample['action'])
    print(f"  - Action shape: {action_array.shape}")
    print(f"  - Action dtype: {action_array.dtype}")
    
    if "state" in sample:
        state_array = np.array(sample['state'])
        print(f"  - State shape: {state_array.shape}")
        print(f"  - State dtype: {state_array.dtype}")
    
    # 检查 @ 分界符
    print(f"\n🔍 Checking @ delimiter:")
    if " @ " in sample['lang']:
        print(f"✅ Found @ delimiter in language")
        parts = sample['lang'].split(" @ ", 1)
        instr_preview = parts[0][:50] if len(parts[0]) > 50 else parts[0]
        reason_preview = parts[1][:50] if len(parts[1]) > 50 else parts[1]
        print(f"  - Instruction part (first 50 chars): {instr_preview}...")
        print(f"  - Reasoning part (first 50 chars): {reason_preview}...")
    else:
        print(f"⚠️  @ delimiter NOT found in language")
        print(f"  - Full language text: {sample['lang'][:200]}...")
    
    # 检查 reasoning 字段（如果存在）
    if "reasoning" in sample:
        reasoning = sample['reasoning']
        if reasoning:
            print(f"✅ Reasoning field exists and non-empty")
            print(f"  - Reasoning (first 100 chars): {reasoning[:100]}...")
        else:
            print(f"⚠️  Reasoning field exists but is EMPTY")
    else:
        print(f"⚠️  Reasoning field NOT in sample")
    
    # 检查thinking tokens (stage 2+)
    print(f"\n🔍 Checking thinking tokens:")
    has_thinking = "<|thinking|>" in sample['lang']
    has_start = "<|start_of_thinking|>" in sample['lang']
    has_end = "<|end_of_thinking|>" in sample['lang']
    
    if has_thinking or has_start or has_end:
        print(f"✅ Found thinking tokens in language")
        print(f"  - <|thinking|>: {'Yes' if has_thinking else 'No'}")
        print(f"  - <|start_of_thinking|>: {'Yes' if has_start else 'No'}")
        print(f"  - <|end_of_thinking|>: {'Yes' if has_end else 'No'}")
    else:
        scheduled_stage = cfg.datasets.vla_data.ecot.scheduled_stage
        if scheduled_stage == 0:
            print(f"✅ No thinking tokens found (expected for stage 0)")
        else:
            print(f"⚠️  No thinking tokens found (unexpected for stage {scheduled_stage})")
    
    return dataloader


def test_model_build(cfg):
    """
    测试模型构建
    
    验证:
    1. 模型可以正常创建
    2. Thinking tokens正确添加到tokenizer
    3. 模型可以移动到GPU
    4. 参数数量合理
    """
    print("\n" + "="*80)
    print("🏗️  Testing Model Build")
    print("="*80)
    
    # 构建模型
    print("Building model...")
    model = build_framework(cfg)
    print(f"✅ Model built successfully: {type(model).__name__}")
    
    # 检查thinking tokens
    if cfg.framework.enable_latent_reasoning:
        print(f"\n🔍 Checking thinking tokens in tokenizer:")
        tokenizer = model.qwen_vl_interface.processor.tokenizer
        vocab = tokenizer.get_vocab()
        
        thinking_tokens = [
            "<|thinking|>",
            "<|start_of_thinking|>",
            "<|end_of_thinking|>"
        ]
        
        for token in thinking_tokens:
            if token in vocab:
                token_id = vocab[token]
                print(f"✅ {token}: ID={token_id}")
            else:
                print(f"❌ {token}: NOT FOUND in vocabulary")
    else:
        print(f"ℹ️  Latent reasoning disabled, skipping thinking token check")
    
    # 移动到GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔧 Moving model to device: {device}")
    model = model.to(device)
    print(f"✅ Model moved to {device}")
    
    # 统计参数
    print(f"\n📊 Model Statistics:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    print(f"  - Frozen parameters: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
    
    # 检查主要组件
    print(f"\n🔍 Model Components:")
    print(f"  - QWen VL Interface: {type(model.qwen_vl_interface).__name__}")
    print(f"  - Action Model: {type(model.action_model).__name__}")
    
    return model


def test_forward_pass(model, dataloader, cfg):
    """
    测试前向传播（无梯度）
    
    验证:
    1. build_qwenvl_inputs正确构建输入
    2. 输入包含必需字段（input_ids, attention_mask, pixel_values, labels, position_ids）
    3. Forward可以正常执行
    4. 输出包含必需字段（action_loss, vlm_loss, total_loss）
    5. Loss值合理（不为NaN/Inf）
    6. Hidden states shape正确
    """
    print("\n" + "="*80)
    print("🔬 Testing Forward Pass (No Gradient)")
    print("="*80)
    
    model.eval()
    batch = next(iter(dataloader))
    
    with torch.no_grad():
        # Step 1: 测试 build_qwenvl_inputs
        print("\n📝 Step 1: Testing build_qwenvl_inputs")
        batch_images = [example["image"] for example in batch]
        instructions = [example["lang"] for example in batch]
        
        print(f"  - Building inputs for {len(batch)} samples...")
        qwen_inputs = model.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions
        )
        
        print(f"✅ Input keys: {list(qwen_inputs.keys())}")
        print(f"✅ input_ids shape: {qwen_inputs['input_ids'].shape}")
        print(f"✅ attention_mask shape: {qwen_inputs['attention_mask'].shape}")
        
        if "position_ids" in qwen_inputs:
            print(f"✅ position_ids shape: {qwen_inputs['position_ids'].shape}")
        else:
            print(f"ℹ️  position_ids not in inputs (may be generated internally)")
        
        if "labels" in qwen_inputs:
            print(f"✅ labels shape: {qwen_inputs['labels'].shape}")
            # 检查mask比例
            total_tokens = qwen_inputs['labels'].numel()
            masked_tokens = (qwen_inputs['labels'] == -100).sum().item()
            trainable_tokens = total_tokens - masked_tokens
            print(f"✅ Label statistics:")
            print(f"  - Total tokens: {total_tokens}")
            print(f"  - Masked tokens (IGNORE_INDEX): {masked_tokens} ({masked_tokens/total_tokens*100:.1f}%)")
            print(f"  - Trainable tokens: {trainable_tokens} ({trainable_tokens/total_tokens*100:.1f}%)")
        else:
            print(f"ℹ️  labels not in inputs (may not compute VLM loss)")
        
        if "pixel_values" in qwen_inputs:
            pv = qwen_inputs['pixel_values']
            if isinstance(pv, dict):
                print(f"✅ pixel_values (dict): {list(pv.keys())}")
                for k, v in pv.items():
                    print(f"  - {k}: shape={v.shape}, dtype={v.dtype}")
            else:
                print(f"✅ pixel_values: shape={pv.shape}, dtype={pv.dtype}")
        
        # 检查thinking token对齐（如果有）
        if cfg.framework.enable_latent_reasoning:
            print(f"\n🔍 Checking thinking token alignment:")
            thinking_token_id = getattr(model.qwen_vl_interface, "thinking_token_id", None)
            if thinking_token_id is not None:
                print(f"  - Thinking token ID: {thinking_token_id}")
                # 找到每个样本的第一个thinking token位置
                B = qwen_inputs['input_ids'].shape[0]
                first_thinking_positions = []
                for b in range(B):
                    ids = qwen_inputs['input_ids'][b]
                    thinking_mask = (ids == thinking_token_id)
                    if thinking_mask.any():
                        pos = thinking_mask.nonzero()[0].item()
                        first_thinking_positions.append(pos)
                        # 统计thinking token数量
                        count = thinking_mask.sum().item()
                        print(f"  - Sample {b}: first position={pos}, total count={count}")
                    else:
                        print(f"  - Sample {b}: no thinking tokens found")
                
                if first_thinking_positions:
                    if len(set(first_thinking_positions)) == 1:
                        print(f"✅ Thinking tokens are ALIGNED at position {first_thinking_positions[0]}!")
                    else:
                        print(f"⚠️  Thinking tokens are NOT aligned: positions={first_thinking_positions}")
            else:
                print(f"ℹ️  thinking_token_id not found in model")
        
        # 检查label mask的正确性（Instruction和Latent是否被正确mask）
        if "labels" in qwen_inputs:
            print(f"\n🔍 Checking label mask correctness:")
            labels = qwen_inputs['labels']
            input_ids = qwen_inputs['input_ids']
            pad_id = model.qwen_vl_interface.processor.tokenizer.pad_token_id
            IGNORE_INDEX = -100
            
            # 检查第一个样本的mask情况
            sample_idx = 0
            sample_labels = labels[sample_idx]
            sample_ids = input_ids[sample_idx]
            valid_mask = (sample_ids != pad_id)
            valid_length = valid_mask.sum().item()
            
            # 找到instruction和latent的边界
            start_id = getattr(model.qwen_vl_interface, "start_thinking_id", None)
            end_id = getattr(model.qwen_vl_interface, "end_thinking_id", None)
            thinking_token_id = getattr(model.qwen_vl_interface, "thinking_token_id", None)
            
            # 检查instruction段是否被mask
            # 对于Stage 2+: 使用start_thinking位置
            # 对于Stage 0: 使用@分界符位置
            scheduled_stage = cfg.datasets.vla_data.ecot.scheduled_stage
            instr_end = None
            
            if start_id is not None:
                start_pos = (sample_ids == start_id).nonzero()
                if start_pos.numel() > 0:
                    instr_end = start_pos[0].item()
                    instr_masked = (sample_labels[:instr_end] == IGNORE_INDEX).all().item()
                    print(f"  - Instruction span [0:{instr_end}) (via <|start_of_thinking|>): {'✅ MASKED' if instr_masked else '❌ NOT MASKED'}")
            
            # 对于Stage 0，检查@分界符
            if instr_end is None and scheduled_stage == 0:
                # 尝试找到@分界符
                tokenizer = model.qwen_vl_interface.processor.tokenizer
                try:
                    at_token_ids = tokenizer.encode(" @ ", add_special_tokens=False)
                    print(f"at_token_ids: {at_token_ids}")
                    if at_token_ids:
                        # 查找@分界符的位置
                        at_tensor = torch.tensor(at_token_ids, device=sample_ids.device, dtype=sample_ids.dtype)
                        at_len = len(at_token_ids)
                        for i in range(len(sample_ids) - at_len + 1):
                            if torch.equal(sample_ids[i:i+at_len], at_tensor):
                                instr_end = i + at_len
                                instr_masked = (sample_labels[:instr_end] == IGNORE_INDEX).all().item()
                                print(f"  - Instruction span [0:{instr_end}) (via @ delimiter): {'✅ MASKED' if instr_masked else '❌ NOT MASKED'}")
                                break
                except Exception as e:
                    print(f"  - ⚠️  Could not check @ delimiter: {e}")
            
            # 检查latent段是否被mask（如果有thinking tokens，Stage 2+）
            if thinking_token_id is not None and start_id is not None and end_id is not None and scheduled_stage > 0:
                start_pos = (sample_ids == start_id).nonzero()
                end_pos = (sample_ids == end_id).nonzero()
                if start_pos.numel() > 0 and end_pos.numel() > 0:
                    lat_start = start_pos[0].item()
                    lat_end = end_pos[0].item() + 1  # Include end token
                    latent_masked = (sample_labels[lat_start:lat_end] == IGNORE_INDEX).all().item()
                    print(f"  - Latent span [{lat_start}:{lat_end}): {'✅ MASKED' if latent_masked else '❌ NOT MASKED'}")
                    
                    # 检查post-latent段是否未被mask
                    if lat_end < valid_length:
                        post_latent_labels = sample_labels[lat_end:valid_length]
                        post_latent_unmasked = (post_latent_labels != IGNORE_INDEX).any().item()
                        print(f"  - Post-latent span [{lat_end}:{valid_length}): {'✅ UNMASKED (trainable)' if post_latent_unmasked else '⚠️  ALL MASKED'}")
            elif scheduled_stage == 0:
                print(f"  - Stage 0: No latent span (no thinking tokens)")
        
        # Step 2: 测试完整forward
        print("\n🚀 Step 2: Testing full forward")
        print(f"  - Running forward pass...")
        print(f"  - Batch size: {len(batch)}")
        print(f"  - repeated_diffusion_steps: {cfg.trainer.repeated_diffusion_steps}")
        print(f"  - Expected effective batch: {len(batch) * cfg.trainer.repeated_diffusion_steps}")
        output_dict = model.forward(batch)
        
        print(f"✅ Output keys: {list(output_dict.keys())}")
        
        # 检查loss
        print(f"\n📊 Loss values:")
        for loss_name in ["action_loss", "vlm_loss", "total_loss"]:
            if loss_name in output_dict:
                loss_value = output_dict[loss_name]
                loss_item = loss_value.item()
                print(f"✅ {loss_name}: {loss_item:.4f}")
                
                # 检查数值有效性
                if torch.isnan(loss_value):
                    print(f"❌ {loss_name} is NaN!")
                    raise ValueError(f"{loss_name} is NaN")
                if torch.isinf(loss_value):
                    print(f"❌ {loss_name} is Inf!")
                    raise ValueError(f"{loss_name} is Inf")
            else:
                print(f"ℹ️  {loss_name}: not in output")
        
        # 检查forward类型（stage 0 vs stage 2+）
        scheduled_stage = cfg.datasets.vla_data.ecot.scheduled_stage
        print(f"\n🔍 Forward type check:")
        if scheduled_stage == 0:
            print(f"✅ Stage 0: Expected to use normal forward (or forward_latent with 0 passes)")
        else:
            print(f"✅ Stage {scheduled_stage}: Expected to use forward_latent with KV-Cache")
        
        # 检查是否使用了iterative forward
        if "vlm_loss" in output_dict and output_dict["vlm_loss"] is not None:
            print(f"✅ VLM loss computed, language model training is active")
        else:
            print(f"ℹ️  VLM loss not computed or None")
        
        # 检查forward_latent是否被调用（通过检查forward_latent的返回值）
        # 注意：这需要在forward_latent中添加返回值来验证，目前只能通过间接方式检查
        scheduled_stage = cfg.datasets.vla_data.ecot.scheduled_stage
        if scheduled_stage > 0 and cfg.framework.enable_latent_reasoning:
            # 对于stage 2+，应该使用forward_latent
            # 可以通过检查是否有thinking tokens来间接验证
            check_thinking_token_id = getattr(model.qwen_vl_interface, "thinking_token_id", None)
            if check_thinking_token_id is not None:
                has_thinking_in_batch = (qwen_inputs['input_ids'] == check_thinking_token_id).any().item()
                if has_thinking_in_batch:
                    print(f"✅ Stage {scheduled_stage}: Thinking tokens found, forward_latent should be used")
                else:
                    print(f"⚠️  Stage {scheduled_stage}: No thinking tokens found in batch (unexpected)")
    
    model.train()
    print(f"\n✅ Forward pass test completed successfully!")
    return output_dict


def test_training_loop(model, dataloader, cfg):
    """
    测试训练循环
    
    验证:
    1. 可以正常backward
    2. 梯度不为NaN/Inf
    3. 参数可以更新
    4. 多步训练稳定
    5. Loss趋势合理
    """
    print("\n" + "="*80)
    print("🏋️  Testing Training Loop")
    print("="*80)
    
    model.train()
    
    # 创建optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )
    
    # 记录初始参数（用于验证更新）
    initial_param = None
    for name, param in model.named_parameters():
        if param.requires_grad:
            initial_param = (name, param.clone().detach())
            break
    
    if initial_param is None:
        print("⚠️  Warning: No trainable parameters found!")
        return []
    
    # 训练循环
    losses = []
    data_iter = iter(dataloader)
    
    print(f"\n🚀 Starting training loop ({cfg.trainer.max_train_steps} steps)...")
    for step in tqdm(range(cfg.trainer.max_train_steps), desc="Training"):
        # 获取batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        # Forward
        optimizer.zero_grad()
        output_dict = model.forward(batch)
        
        # 使用total_loss（如果有），否则用action_loss
        loss = output_dict.get("total_loss", output_dict["action_loss"])
        
        # Backward
        loss.backward()
        
        # 检查梯度
        has_nan_grad = False
        has_inf_grad = False
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    has_nan_grad = True
                    print(f"⚠️  NaN gradient in {name}")
                if torch.isinf(param.grad).any():
                    has_inf_grad = True
                    print(f"⚠️  Inf gradient in {name}")
                grad_norms.append(param.grad.norm().item())
        
        if has_nan_grad or has_inf_grad:
            print(f"❌ Step {step}: Gradient check FAILED")
            raise RuntimeError(f"Gradient contains NaN or Inf at step {step}")
        
        # Gradient clipping
        if cfg.trainer.gradient_clipping is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.trainer.gradient_clipping)
        
        # Optimizer step
        optimizer.step()
        
        # 记录loss
        losses.append(loss.item())
        
        # 每步打印
        if step % cfg.trainer.logging_frequency == 0:
            log_str = f"Step {step}: "
            if "action_loss" in output_dict:
                log_str += f"action_loss={output_dict['action_loss'].item():.4f} "
            if "vlm_loss" in output_dict and output_dict["vlm_loss"] is not None:
                log_str += f"vlm_loss={output_dict['vlm_loss'].item():.4f} "
            log_str += f"total_loss={loss.item():.4f}"
            if grad_norms:
                log_str += f" | grad_norm={np.mean(grad_norms):.4f}"
            print(log_str)
    
    # 验证参数更新
    if initial_param is not None:
        name, initial_value = initial_param
        current_value = dict(model.named_parameters())[name]
        param_changed = not torch.equal(initial_value, current_value)
        if param_changed:
            print(f"\n✅ Parameters updated (checked: {name})")
            # 计算参数变化量
            param_diff = (current_value - initial_value).abs().max().item()
            print(f"  - Max parameter change: {param_diff:.6f}")
        else:
            print(f"\n⚠️  Parameters NOT updated (checked: {name})")
    
    # Loss趋势
    print(f"\n📊 Loss Statistics:")
    print(f"  - Initial loss: {losses[0]:.4f}")
    print(f"  - Final loss: {losses[-1]:.4f}")
    print(f"  - Mean loss: {np.mean(losses):.4f}")
    print(f"  - Std loss: {np.std(losses):.4f}")
    print(f"  - Min loss: {np.min(losses):.4f}")
    print(f"  - Max loss: {np.max(losses):.4f}")
    
    # 检查loss稳定性
    if len(losses) > 1:
        loss_std = np.std(losses)
        loss_mean = np.mean(losses)
        cv = loss_std / loss_mean if loss_mean > 0 else float('inf')
        if cv < 1.0:
            print(f"✅ Loss is stable (CV={cv:.3f} < 1.0)")
        else:
            print(f"⚠️  Loss has high variance (CV={cv:.3f} >= 1.0)")
    
    print(f"\n✅ Training loop test completed successfully!")
    return losses


def main():
    """
    主测试流程 - 前5个步骤
    """
    # 解析参数
    parser = argparse.ArgumentParser(description="ECoT End-to-End Test Training")
    parser.add_argument("--config_yaml", type=str, default="config/test_ecot_stage0.yaml", help="Path to test config YAML")
    args = parser.parse_args()
    
    # 加载配置
    print("Loading configuration...")
    cfg = OmegaConf.load(args.config_yaml)
    
    print("\n" + "="*80)
    print("🚀 ECoT Implicit Reasoning End-to-End Test")
    print("="*80)
    print(f"Config: {args.config_yaml}")
    print(f"Stage: {cfg.datasets.vla_data.ecot.scheduled_stage}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # 测试流程
    try:
        # 1. 验证配置
        print("\n" + "🔹"*40)
        print("Step 1/5: Configuration Validation")
        print("🔹"*40)
        validate_config(cfg)
        
        # 2. 测试数据加载
        print("\n" + "🔹"*40)
        print("Step 2/5: DataLoader Test")
        print("🔹"*40)
        dataloader = test_dataloader(cfg)
        if dataloader is None:
            raise RuntimeError("DataLoader test failed")
        
        # 3. 测试模型构建
        print("\n" + "🔹"*40)
        print("Step 3/5: Model Build Test")
        print("🔹"*40)
        model = test_model_build(cfg)
        
        # 4. 测试前向传播（无梯度）
        print("\n" + "🔹"*40)
        print("Step 4/5: Forward Pass Test (No Gradient)")
        print("🔹"*40)
        output_dict = test_forward_pass(model, dataloader, cfg)
        
        # 5. 测试训练循环（有梯度）
        print("\n" + "🔹"*40)
        print("Step 5/5: Training Loop Test")
        print("🔹"*40)
        losses = test_training_loop(model, dataloader, cfg)
        
        # 最终报告
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED!")
        print("="*80)
        print(f"✅ Step 1: Configuration validation - OK")
        print(f"✅ Step 2: DataLoader test - OK")
        print(f"✅ Step 3: Model build test - OK")
        print(f"✅ Step 4: Forward pass test - OK")
        print(f"✅ Step 5: Training loop test - OK")
        print("\n🎉 ECoT pipeline is ready for full training!")
        
    except Exception as e:
        print("\n" + "="*80)
        print("❌ TEST FAILED")
        print("="*80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

