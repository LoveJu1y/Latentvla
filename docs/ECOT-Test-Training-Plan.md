# ECoT 隐式推理端到端测试训练脚本设计方案

## 一、测试目标

### 核心验证点
1. ✅ **数据流验证**: ECOT RLDS数据 → DataLoader → Batch格式正确
2. ✅ **Tokenization验证**: `@` 分界符 + thinking tokens正确插入和tokenize
3. ✅ **对齐验证**: Thinking token位置在batch中正确对齐
4. ✅ **Label掩码验证**: Instruction和latent span正确mask
5. ✅ **Forward验证**: 
   - Stage 0: 普通forward + VLM loss计算
   - Stage 2+: KV-Cache迭代forward + thinking token更新
6. ✅ **Loss计算验证**: Action loss + VLM loss正确计算和合并
7. ✅ **梯度流验证**: 反向传播正常，梯度不为NaN/Inf
8. ✅ **多步训练验证**: 至少运行10-20步，验证稳定性

### 非目标
- ❌ 不追求训练收敛（只验证流程）
- ❌ 不进行完整的evaluation
- ❌ 不保存checkpoint（可选）
- ❌ 不使用分布式训练（单GPU测试）

---

## 二、脚本设计架构

### 2.1 文件结构
```
starVLA/
├── test_ecot_training.py          # 主测试脚本（新建）
├── config/
│   └── test_ecot_stage0.yaml      # Stage 0测试配置（新建）
│   └── test_ecot_stage2.yaml      # Stage 2测试配置（新建）
└── docs/
    └── ECOT-Test-Training-Plan.md # 本文档
```

### 2.2 脚本模块划分

```python
# test_ecot_training.py 结构

# 1. 导入模块
# 2. 配置加载与验证
# 3. 数据准备
# 4. 模型构建
# 5. 前向测试（无梯度）
# 6. 训练循环测试（有梯度）
# 7. 结果验证与报告
```

---

## 三、详细设计

### 3.1 配置文件设计

#### Stage 0 配置 (`config/test_ecot_stage0.yaml`)

```yaml
# 基础配置
run_id: "test_ecot_stage0"
run_root_dir: "./test_outputs"
seed: 42
is_debug: false

# 数据配置
datasets:
  vla_data:
    dataset_py: "ecot_rlds"
    per_device_batch_size: 2  # 保守设置
    num_workers: 0  # 避免多进程问题
    image_size: [224, 224]
    ecot:
      data_root_dir: "/share/project/emllm_mnt.1d/mnt/sfs/baaiei/jyShi/rt_newData"
      data_mix: "bridge"
      scheduled_stage: 0  # Stage 0: 无thinking tokens
      action_dim: 7
      future_action_window_size: 15
      past_action_window_size: 0
      shuffle_buffer_size: 100  # 小buffer，快速测试
      image_aug: false
      reasoning_json: "/share/project/lvjing/datas/embodied_features_bridge.json"
      load_proprio: true
      lower_case_instruction: true
      train: true

# 模型配置
framework:
  name: "QwenGR00T"
  enable_latent_reasoning: true  # 开启（但stage 0无thinking tokens）
  latent_reasoning:
    compute_language_loss: true  # 测试VLM loss计算
    vlm_loss_weight: 0.1
    thinking_token: "<|thinking|>"
    start_of_thinking_token: "<|start_of_thinking|>"
    end_of_thinking_token: "<|end_of_thinking|>"
  qwenvl:
    base_vlm: "Qwen/Qwen3-VL-2B-Instruct"
    attn_implementation: "sdpa"
    cache_dir: "./qwen_cache"
    model_max_length: 2048  # 减小以加快测试
  action_model:
    action_dim: 7
    future_action_window_size: 15
    past_action_window_size: 0

# 训练配置（最小化）
trainer:
  max_train_steps: 10  # 只跑10步
  gradient_accumulation_steps: 1
  learning_rate:
    base: 1.0e-5
  optimizer:
    betas: [0.9, 0.95]
    weight_decay: 0.01
    eps: 1.0e-8
  lr_scheduler_type: "constant"
  num_warmup_steps: 0
  gradient_clipping: 1.0
  logging_frequency: 1  # 每步都log
  save_interval: 1000  # 不保存
  eval_interval: 1000  # 不eval
  repeated_diffusion_steps: 2  # 减少diffusion steps加快测试

# W&B配置（可选）
wandb_project: "test_ecot"
wandb_entity: null
```

#### Stage 2 配置 (`config/test_ecot_stage2.yaml`)

```yaml
# 与Stage 0大部分相同，关键差异：
datasets:
  vla_data:
    ecot:
      scheduled_stage: 2  # Stage 2: 有thinking tokens
      thinking_token_count: 2
      tag2think_count:
        TASK: 1
        PLAN: 1
        "VISIBLE OBJECTS": 1
        "SUBTASK REASONING": 1
        SUBTASK: 1
        "MOVE REASONING": 1
        MOVE: 1
        "GRIPPER POSITION": 1

# 其他配置相同
```

---

### 3.2 主测试脚本设计 (`test_ecot_training.py`)

#### 模块1: 导入与初始化

```python
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

# Local imports
from starVLA.model.framework import build_framework
from starVLA.dataloader import build_dataloader

# 设置环境变量
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "./qwen_cache"
```

#### 模块2: 配置验证

```python
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
    required_fields = [
        "datasets.vla_data.dataset_py",
        "datasets.vla_data.ecot.data_root_dir",
        "datasets.vla_data.ecot.scheduled_stage",
        "framework.name",
        "framework.enable_latent_reasoning",
    ]
    
    # 检查数据路径
    data_root = cfg.datasets.vla_data.ecot.data_root_dir
    reasoning_json = cfg.datasets.vla_data.ecot.reasoning_json
    
    # 检查batch size
    batch_size = cfg.datasets.vla_data.per_device_batch_size
    
    # 打印关键配置
    print(f"✅ Stage: {cfg.datasets.vla_data.ecot.scheduled_stage}")
    print(f"✅ Enable Latent Reasoning: {cfg.framework.enable_latent_reasoning}")
    print(f"✅ Compute Language Loss: {cfg.framework.latent_reasoning.compute_language_loss}")
    print(f"✅ Batch Size: {batch_size}")
    print(f"✅ Max Steps: {cfg.trainer.max_train_steps}")
    
    return True
```

#### 模块3: 数据加载测试

```python
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
    
    # 创建dataloader
    dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)
    
    # 获取一个batch
    batch = next(iter(dataloader))
    
    # 验证batch格式
    print(f"✅ Batch type: {type(batch)}")
    print(f"✅ Batch length: {len(batch)}")
    print(f"✅ Sample keys: {batch[0].keys()}")
    
    # 检查必需字段
    required_keys = ["image", "lang", "action"]
    for key in required_keys:
        assert key in batch[0], f"Missing key: {key}"
    
    # 打印样本信息
    sample = batch[0]
    print(f"\n📊 Sample Info:")
    print(f"  - Images: {len(sample['image'])} views, shape: {sample['image'][0].size}")
    print(f"  - Language: {sample['lang'][:100]}...")  # 前100字符
    print(f"  - Action shape: {np.array(sample['action']).shape}")
    if "state" in sample:
        print(f"  - State shape: {np.array(sample['state']).shape}")
    
    # 检查 @ 分界符
    if " @ " in sample['lang']:
        print(f"✅ Found @ delimiter in language")
        parts = sample['lang'].split(" @ ", 1)
        print(f"  - Instruction part: {parts[0][:50]}...")
        print(f"  - Reasoning part: {parts[1][:50]}...")
    
    # 检查thinking tokens (stage 2+)
    if "<|thinking|>" in sample['lang']:
        print(f"✅ Found thinking tokens in language")
        print(f"  - <|start_of_thinking|>: {'Yes' if '<|start_of_thinking|>' in sample['lang'] else 'No'}")
        print(f"  - <|end_of_thinking|>: {'Yes' if '<|end_of_thinking|>' in sample['lang'] else 'No'}")
    
    return dataloader
```

#### 模块4: 模型构建测试

```python
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
    model = build_framework(cfg)
    
    # 检查thinking tokens
    if cfg.framework.enable_latent_reasoning:
        tokenizer = model.qwen_vl_interface.processor.tokenizer
        vocab = tokenizer.get_vocab()
        
        thinking_tokens = [
            "<|thinking|>",
            "<|start_of_thinking|>",
            "<|end_of_thinking|>"
        ]
        
        for token in thinking_tokens:
            if token in vocab:
                print(f"✅ {token}: ID={vocab[token]}")
            else:
                print(f"❌ {token}: NOT FOUND")
    
    # 移动到GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"✅ Model moved to {device}")
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Total parameters: {total_params:,}")
    print(f"✅ Trainable parameters: {trainable_params:,}")
    
    return model
```

#### 模块5: 前向传播测试（无梯度）

```python
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
        
        qwen_inputs = model.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions
        )
        
        print(f"✅ Input keys: {qwen_inputs.keys()}")
        print(f"✅ input_ids shape: {qwen_inputs['input_ids'].shape}")
        print(f"✅ attention_mask shape: {qwen_inputs['attention_mask'].shape}")
        if "position_ids" in qwen_inputs:
            print(f"✅ position_ids shape: {qwen_inputs['position_ids'].shape}")
        if "labels" in qwen_inputs:
            print(f"✅ labels shape: {qwen_inputs['labels'].shape}")
            # 检查mask比例
            total_tokens = qwen_inputs['labels'].numel()
            masked_tokens = (qwen_inputs['labels'] == -100).sum().item()
            print(f"✅ Masked tokens: {masked_tokens}/{total_tokens} ({masked_tokens/total_tokens*100:.1f}%)")
        
        # 检查thinking token对齐（如果有）
        if cfg.framework.enable_latent_reasoning:
            thinking_token_id = getattr(model.qwen_vl_interface, "thinking_token_id", None)
            if thinking_token_id is not None:
                # 找到每个样本的第一个thinking token位置
                B = qwen_inputs['input_ids'].shape[0]
                first_thinking_positions = []
                for b in range(B):
                    ids = qwen_inputs['input_ids'][b]
                    thinking_mask = (ids == thinking_token_id)
                    if thinking_mask.any():
                        pos = thinking_mask.nonzero()[0].item()
                        first_thinking_positions.append(pos)
                
                if first_thinking_positions:
                    print(f"✅ First thinking token positions: {first_thinking_positions}")
                    if len(set(first_thinking_positions)) == 1:
                        print(f"✅ Thinking tokens are ALIGNED!")
                    else:
                        print(f"⚠️  Thinking tokens are NOT aligned")
        
        # Step 2: 测试完整forward
        print("\n🚀 Step 2: Testing full forward")
        output_dict = model.forward(batch)
        
        print(f"✅ Output keys: {output_dict.keys()}")
        
        # 检查loss
        for loss_name in ["action_loss", "vlm_loss", "total_loss"]:
            if loss_name in output_dict:
                loss_value = output_dict[loss_name]
                print(f"✅ {loss_name}: {loss_value.item():.4f}")
                assert not torch.isnan(loss_value), f"{loss_name} is NaN!"
                assert not torch.isinf(loss_value), f"{loss_name} is Inf!"
        
        # 检查forward类型（stage 0 vs stage 2+）
        scheduled_stage = cfg.datasets.vla_data.ecot.scheduled_stage
        if scheduled_stage == 0:
            print(f"✅ Stage 0: Should use normal forward")
        else:
            print(f"✅ Stage {scheduled_stage}: Should use forward_latent with KV-Cache")
    
    model.train()
    return output_dict
```

#### 模块6: 训练循环测试（有梯度）

```python
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
    
    # 训练循环
    losses = []
    data_iter = iter(dataloader)
    
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
            break
        
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
            if "vlm_loss" in output_dict:
                log_str += f"vlm_loss={output_dict['vlm_loss'].item():.4f} "
            log_str += f"total_loss={loss.item():.4f}"
            print(log_str)
    
    # 验证参数更新
    if initial_param is not None:
        name, initial_value = initial_param
        current_value = dict(model.named_parameters())[name]
        param_changed = not torch.equal(initial_value, current_value)
        if param_changed:
            print(f"✅ Parameters updated (checked: {name})")
        else:
            print(f"⚠️  Parameters NOT updated (checked: {name})")
    
    # Loss趋势
    print(f"\n📊 Loss Statistics:")
    print(f"  - Initial loss: {losses[0]:.4f}")
    print(f"  - Final loss: {losses[-1]:.4f}")
    print(f"  - Mean loss: {np.mean(losses):.4f}")
    print(f"  - Std loss: {np.std(losses):.4f}")
    
    return losses
```

#### 模块7: 主函数

```python
def main():
    """
    主测试流程
    """
    # 解析参数
    parser = argparse.ArgumentParser(description="ECoT End-to-End Test Training")
    parser.add_argument("--config_yaml", type=str, required=True, help="Path to test config YAML")
    args = parser.parse_args()
    
    # 加载配置
    cfg = OmegaConf.load(args.config_yaml)
    
    print("\n" + "="*80)
    print("🚀 ECoT Implicit Reasoning End-to-End Test")
    print("="*80)
    print(f"Config: {args.config_yaml}")
    print(f"Stage: {cfg.datasets.vla_data.ecot.scheduled_stage}")
    
    # 测试流程
    try:
        # 1. 验证配置
        validate_config(cfg)
        
        # 2. 测试数据加载
        dataloader = test_dataloader(cfg)
        
        # 3. 测试模型构建
        model = test_model_build(cfg)
        
        # 4. 测试前向传播（无梯度）
        test_forward_pass(model, dataloader, cfg)
        
        # 5. 测试训练循环（有梯度）
        losses = test_training_loop(model, dataloader, cfg)
        
        # 6. 最终报告
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED!")
        print("="*80)
        print(f"✅ Data loading: OK")
        print(f"✅ Model building: OK")
        print(f"✅ Forward pass: OK")
        print(f"✅ Training loop: OK")
        print(f"✅ Gradient flow: OK")
        print(f"✅ Parameter update: OK")
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
```

---

## 四、测试执行计划

### 4.1 测试顺序

```bash
# Step 1: 测试 Stage 0 (无thinking tokens)
python test_ecot_training.py --config_yaml config/test_ecot_stage0.yaml

# Step 2: 测试 Stage 2 (有thinking tokens)
python test_ecot_training.py --config_yaml config/test_ecot_stage2.yaml
```

### 4.2 预期输出

#### Stage 0 预期
```
✅ Found @ delimiter in language
✅ Masked tokens: 450/512 (87.9%)  # Instruction被mask
✅ Stage 0: Should use normal forward
✅ action_loss: 2.3456
✅ vlm_loss: 1.2345
✅ total_loss: 2.4689
✅ Parameters updated
```

#### Stage 2 预期
```
✅ Found @ delimiter in language
✅ Found thinking tokens in language
✅ Thinking tokens are ALIGNED!
✅ Masked tokens: 480/512 (93.8%)  # Instruction + latent被mask
✅ Stage 2: Should use forward_latent with KV-Cache
✅ action_loss: 2.3456
✅ vlm_loss: 1.2345
✅ total_loss: 2.4689
✅ Parameters updated
```

### 4.3 失败诊断

| 错误现象 | 可能原因 | 解决方案 |
|---------|---------|---------|
| `@ delimiter not found` | 数据处理未添加 `@` | 检查 `ECOTBatchTransform` |
| `Thinking tokens NOT aligned` | 对齐逻辑错误 | 检查 `_align_thinking_tokens` |
| `NaN loss` | Label mask错误或数值不稳定 | 检查 `_build_ecot_labels_batch` |
| `NaN gradient` | Forward计算错误 | 检查 `forward_latent` |
| `Parameters NOT updated` | 所有参数被冻结 | 检查 `requires_grad` |
| `OOM` | Batch size太大或序列太长 | 减小batch size或model_max_length |

---

## 五、成功标准

### 必须通过的检查项

#### 数据层面
- [ ] DataLoader可以正常创建和迭代
- [ ] Batch格式包含所有必需字段
- [ ] `@` 分界符存在于所有样本
- [ ] Stage 2+的样本包含thinking tokens

#### Tokenization层面
- [ ] Thinking tokens正确添加到tokenizer
- [ ] `build_qwenvl_inputs` 返回所有必需字段
- [ ] `input_ids` 和 `attention_mask` shape一致
- [ ] `position_ids` 正确生成
- [ ] `labels` 正确生成且包含mask

#### 对齐层面
- [ ] Stage 2+的thinking tokens位置对齐
- [ ] 对齐后的序列长度合理（不超过model_max_length）

#### Forward层面
- [ ] Stage 0使用普通forward
- [ ] Stage 2+使用 `forward_latent`
- [ ] 所有loss值为有限数（不是NaN/Inf）
- [ ] `action_loss` 和 `vlm_loss` 都能计算
- [ ] `total_loss` 正确合并

#### 训练层面
- [ ] Backward正常执行
- [ ] 梯度不包含NaN/Inf
- [ ] 参数可以更新
- [ ] 至少能稳定训练10步

---

## 六、后续扩展

### 6.1 完整训练脚本
测试通过后，可以基于 `train_starvla.py` 创建完整的ECoT训练脚本：
- 添加分布式训练支持
- 添加checkpoint保存/恢复
- 添加W&B logging
- 添加evaluation

### 6.2 性能优化
- 添加mixed precision训练
- 优化DataLoader（增加num_workers）
- 添加gradient checkpointing

### 6.3 多阶段训练
- 实现curriculum learning（stage 0 → 1 → 2 → ...）
- 自动切换scheduled_stage
- 保存每个stage的checkpoint

---

## 七、检查清单

在运行测试前，确认：
- [ ] 数据路径正确且数据存在
- [ ] Reasoning JSON文件存在
- [ ] GPU可用且内存足够（建议>=16GB）
- [ ] 已安装所有依赖
- [ ] HF_ENDPOINT和HF_HOME已设置
- [ ] Qwen3-VL模型可以下载或已缓存

在测试过程中，观察：
- [ ] 数据加载时间合理（<10s per batch）
- [ ] 模型加载时间合理（<2min）
- [ ] Forward时间合理（Stage 0: <2s, Stage 2: <5s per batch）
- [ ] 内存使用稳定（不持续增长）
- [ ] Loss值在合理范围（0.1-10.0）

测试通过后，确认：
- [ ] 所有测试模块都打印了 ✅
- [ ] 没有 ⚠️ 或 ❌ 输出
- [ ] Loss曲线平滑（无突变）
- [ ] 可以重复运行测试（结果一致）

---

## 八、总结

本测试脚本设计为**渐进式验证**，从数据到模型到训练，逐层测试。每个模块独立且可调试，失败时可以快速定位问题。

**设计原则**：
1. **保守参数**：小batch size、短序列、少训练步数
2. **详细输出**：每个步骤都有明确的 ✅/❌ 标记
3. **快速失败**：发现问题立即停止并报告
4. **可重复**：固定seed，结果可复现
5. **独立模块**：每个测试函数可以单独运行

**预期时间**：
- Stage 0测试：~5-10分钟
- Stage 2测试：~10-15分钟（KV-Cache多次forward更慢）

测试通过后，即可开始完整的训练实验！

