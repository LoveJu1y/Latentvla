# ECoT 隐式推理训练脚本集成方案

## 一、概述

本文档详细规划如何将 ECoT 隐式推理功能集成到 `train_starvla.py` 训练脚本中，支持：
- Stage 0: 无 thinking tokens，使用普通 forward
- Stage 2+: 有 thinking tokens，使用 KV-Cache 迭代 forward
- VLM Loss 计算和记录
- 多阶段训练（Curriculum Learning，可选）

---

## 二、当前状态分析

### 2.1 已实现的功能

#### QwenGR00T 框架层（`starVLA/model/framework/QwenGR00T.py`）
- ✅ 已支持 `forward_latent` 调用（第 103-116 行）
- ✅ 已支持 `vlm_loss` 计算（第 162-167 行）
- ✅ 已支持 `total_loss` 合并（`action_loss + vlm_loss_weight * vlm_loss`）
- ✅ 已支持根据 `enable_latent_reasoning` 自动选择 forward 路径

#### QWen3 接口层（`starVLA/model/modules/vlm/QWen3.py`）
- ✅ 已实现 `forward_latent` 方法（KV-Cache 迭代 forward）
- ✅ 已实现 `_build_ecot_labels_batch` 方法（Label masking）
- ✅ 已实现 `_align_thinking_tokens` 方法（Thinking token 对齐）
- ✅ 已实现 `build_qwenvl_inputs` 方法（支持隐式推理路径）

### 2.2 需要修改的部分

#### train_starvla.py
- ❌ `_train_step` 方法：未处理 `vlm_loss` 和 `total_loss`
- ❌ `_log_metrics` 方法：未记录 `vlm_loss`
- ❌ `_log_training_config` 方法：未记录 ECoT 配置
- ❌ 缺少配置验证：未验证 ECoT 相关配置
- ❌ 缺少多阶段训练支持：未实现 Curriculum Learning

---

## 三、详细修改方案

### 3.1 修改点 1: `_train_step` 方法

**文件**: `starVLA/training/train_starvla.py`  
**位置**: 第 389-414 行  
**优先级**: ⭐⭐⭐ 必须修改

#### 当前代码问题
```python
def _train_step(self, batch_vla, batch_vlm=None):
    # ...
    output_dict = self.model.forward(batch_vla)
    action_loss = output_dict["action_loss"]
    total_loss = action_loss  # ❌ 只使用 action_loss，忽略了 vlm_loss
    # ...
    return {
        "action_dit_loss": action_loss.item(),  # ❌ 未返回 vlm_loss
    }
```

#### 修改方案

```python
def _train_step(self, batch_vla, batch_vlm=None):
    """execute single training step"""
    with self.accelerator.accumulate(self.model):
        self.optimizer.zero_grad()

        # VLA task forward propagation
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output_dict = self.model.forward(batch_vla)

            # ✅ 修改点1: 优先使用模型返回的 total_loss（如果存在）
            # 模型已经根据配置计算了 total_loss = action_loss + vlm_loss_weight * vlm_loss
            if "total_loss" in output_dict:
                total_loss = output_dict["total_loss"]
            else:
                # 回退：只使用 action_loss（向后兼容）
                total_loss = output_dict["action_loss"]

        # VLA backward propagation
        self.accelerator.backward(total_loss)

        # gradient clipping
        if self.config.trainer.gradient_clipping is not None:
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

        # optimizer step
        self.optimizer.step()
        self.lr_scheduler.step()

    # ✅ 修改点2: 返回完整的 metrics，包括 vlm_loss（如果存在）
    metrics = {
        "action_loss": output_dict["action_loss"].item(),
    }
    
    # 添加 vlm_loss（如果存在且已计算）
    if "vlm_loss" in output_dict and output_dict["vlm_loss"] is not None:
        metrics["vlm_loss"] = output_dict["vlm_loss"].item()
    
    # 添加 total_loss（用于监控，确保与 backward 使用的 loss 一致）
    metrics["total_loss"] = total_loss.item()
    
    return metrics
```

#### 修改理由
1. **使用模型计算的 total_loss**：模型已根据配置计算 `total_loss = action_loss + vlm_loss_weight * vlm_loss`，应直接使用
2. **记录 vlm_loss**：便于监控和调试 VLM loss 的变化
3. **向后兼容**：如果模型未返回 `total_loss`，回退到只使用 `action_loss`

---

### 3.2 修改点 2: `_log_training_config` 方法

**文件**: `starVLA/training/train_starvla.py`  
**位置**: 第 380-387 行  
**优先级**: ⭐⭐ 重要

#### 修改方案

```python
def _log_training_config(self):
    """record training config"""
    if self.accelerator.is_main_process:
        logger.info("***** Training Configuration *****")
        logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
        logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
        logger.info(f"  Gradient accumulation steps = {self.config.trainer.gradient_accumulation_steps}")
        logger.info(f"  Total batch size = {self.total_batch_size}")
        
        # ✅ 新增: 记录 ECoT 隐式推理配置
        enable_latent_reasoning = self.config.framework.get("enable_latent_reasoning", False)
        if enable_latent_reasoning:
            latent_cfg = self.config.framework.get("latent_reasoning", {})
            compute_language_loss = latent_cfg.get("compute_language_loss", False)
            vlm_loss_weight = latent_cfg.get("vlm_loss_weight", 0.1)
            
            # 获取 scheduled_stage（可能在 ecot.* 中）
            try:
                scheduled_stage = self.config.datasets.vla_data.ecot.get("scheduled_stage", 0)
            except (AttributeError, KeyError):
                scheduled_stage = 0
            
            logger.info("***** ECoT Implicit Reasoning Configuration *****")
            logger.info(f"  Enable Latent Reasoning: {enable_latent_reasoning}")
            logger.info(f"  Scheduled Stage: {scheduled_stage}")
            logger.info(f"  Compute Language Loss: {compute_language_loss}")
            if compute_language_loss:
                logger.info(f"  VLM Loss Weight: {vlm_loss_weight}")
            else:
                logger.info(f"  VLM Loss: Disabled (compute_language_loss=False)")
```

#### 修改理由
- 便于确认训练配置是否正确
- 帮助调试时快速了解当前使用的模式
- 记录关键参数（stage、loss weight）便于复现

---

### 3.3 修改点 3: 配置验证函数（新增）

**文件**: `starVLA/training/train_starvla.py`  
**位置**: 在 `main` 函数中，`build_model` 之前  
**优先级**: ⭐⭐ 重要

#### 修改方案

```python
def validate_ecot_config(cfg):
    """
    验证 ECoT 隐式推理相关配置的完整性和一致性
    
    检查项:
    1. enable_latent_reasoning 与 scheduled_stage 的一致性
    2. compute_language_loss 与 enable_latent_reasoning 的关系
    3. vlm_loss_weight 是否存在且合理
    4. thinking tokens 配置是否存在
    """
    enable_latent_reasoning = cfg.framework.get("enable_latent_reasoning", False)
    
    if not enable_latent_reasoning:
        logger.info("ECoT implicit reasoning is disabled (enable_latent_reasoning=False)")
        return True
    
    logger.info("Validating ECoT implicit reasoning configuration...")
    
    # 检查 latent_reasoning 配置
    latent_cfg = cfg.framework.get("latent_reasoning", {})
    if not latent_cfg:
        logger.warning("⚠️  enable_latent_reasoning=True but latent_reasoning config is missing")
        logger.warning("   Using default values for latent_reasoning")
    else:
        # 检查 compute_language_loss
        compute_language_loss = latent_cfg.get("compute_language_loss", False)
        if compute_language_loss:
            vlm_loss_weight = latent_cfg.get("vlm_loss_weight", 0.1)
            if not (0.0 <= vlm_loss_weight <= 1.0):
                logger.warning(f"⚠️  vlm_loss_weight={vlm_loss_weight} is outside recommended range [0.0, 1.0]")
            logger.info(f"✅ VLM loss will be computed with weight: {vlm_loss_weight}")
        else:
            logger.info("ℹ️  VLM loss computation is disabled (compute_language_loss=False)")
            logger.info("   Only action_loss will be used for training")
    
    # 检查 scheduled_stage
    try:
        scheduled_stage = cfg.datasets.vla_data.ecot.scheduled_stage
        logger.info(f"✅ ECoT scheduled_stage: {scheduled_stage}")
        
        if scheduled_stage == 0:
            logger.info("   Stage 0: No thinking tokens, using normal forward")
        elif scheduled_stage >= 2:
            logger.info(f"   Stage {scheduled_stage}: With thinking tokens, using forward_latent")
        else:
            logger.warning(f"⚠️  Unusual scheduled_stage: {scheduled_stage} (expected 0 or >=2)")
    except (AttributeError, KeyError):
        logger.warning("⚠️  enable_latent_reasoning=True but scheduled_stage not found in config")
        logger.warning("   Assuming scheduled_stage=0")
    
    # 检查 thinking tokens 配置
    if latent_cfg:
        thinking_token = latent_cfg.get("thinking_token", "<|thinking|>")
        start_token = latent_cfg.get("start_of_thinking_token", "<|start_of_thinking|>")
        end_token = latent_cfg.get("end_of_thinking_token", "<|end_of_thinking|>")
        logger.info(f"✅ Thinking tokens: {thinking_token}, {start_token}, {end_token}")
    
    logger.info("✅ ECoT configuration validation completed")
    return True
```

#### 在 main 函数中调用

```python
def main(cfg) -> None:
    logger.info("VLA Training :: Warming Up")

    # ✅ 新增: 验证 ECoT 配置
    validate_ecot_config(cfg)
    
    # create output directory and save config
    output_dir = setup_directories(cfg=cfg)
    # ... 其余代码保持不变 ...
```

#### 修改理由
- 提前发现配置错误，避免训练中途失败
- 提供清晰的配置信息，便于调试
- 验证配置合理性（如 vlm_loss_weight 范围）

---

### 3.4 修改点 4: 移除 QwenGR00T 中的调试日志（可选）

**文件**: `starVLA/model/framework/QwenGR00T.py`  
**位置**: 第 99-101 行、105 行、116 行、119 行  
**优先级**: ⭐ 可选

#### 修改方案

```python
# 第 99-101 行：改为 debug 级别或移除
# 原代码：
logger.info(f"[QwenGR00T.forward] enable_latent_reasoning={enable_latent_reasoning}, ...")

# 修改为：
logger.debug(f"[QwenGR00T.forward] enable_latent_reasoning={enable_latent_reasoning}, ...")

# 第 105、116、119 行：同样改为 debug 级别或移除
```

#### 修改理由
- 减少日志噪音
- 保留关键错误和警告信息
- 可通过日志级别控制是否显示

---

### 3.5 修改点 5: 多阶段训练支持（可选，Curriculum Learning）

**文件**: `starVLA/training/train_starvla.py`  
**位置**: 在 `VLATrainer` 类中新增方法  
**优先级**: ⭐ 可选（高级功能）

#### 功能说明
支持在训练过程中动态切换 `scheduled_stage`，实现渐进式训练：
- Step 0-1000: Stage 0（无 thinking tokens）
- Step 1000-2000: Stage 1（少量 thinking tokens）
- Step 2000+: Stage 2（完整 thinking tokens）

#### 修改方案

##### 5.1 在 `__init__` 中添加

```python
class VLATrainer(TrainerUtils):
    def __init__(self, cfg, model, vla_train_dataloader, optimizer, lr_scheduler, accelerator):
        # ... 现有代码 ...
        
        # ✅ 新增: 多阶段训练支持
        self.curriculum_learning = cfg.trainer.get("curriculum_learning", {})
        self.enable_curriculum = self.curriculum_learning.get("enable", False)
        
        if self.enable_curriculum:
            self.stage_schedule = self.curriculum_learning.get("stage_schedule", {})
            # stage_schedule 格式: {step: stage}
            # 例如: {0: 0, 1000: 1, 2000: 2}
            logger.info(f"✅ Curriculum Learning enabled: {self.stage_schedule}")
        else:
            self.stage_schedule = None
```

##### 5.2 添加获取当前 stage 的方法

```python
def _get_current_stage(self):
    """
    根据训练步数返回当前的 scheduled_stage
    
    Returns:
        int: 当前应该使用的 scheduled_stage
    """
    if not self.enable_curriculum:
        # 不使用 curriculum learning，返回配置中的固定 stage
        try:
            return self.config.datasets.vla_data.ecot.get("scheduled_stage", 0)
        except (AttributeError, KeyError):
            return 0
    
    # 根据 step 查找对应的 stage
    current_step = self.completed_steps
    stages = sorted(self.stage_schedule.keys())
    
    # 找到当前 step 应该使用的 stage
    for step_threshold in reversed(stages):
        if current_step >= step_threshold:
            target_stage = self.stage_schedule[step_threshold]
            return target_stage
    
    # 默认返回第一个 stage（step 0 对应的 stage）
    return self.stage_schedule.get(0, 0)
```

##### 5.3 在训练循环中检查 stage 切换

```python
def train(self):
    """execute training loop"""
    # ... 现有代码 ...
    
    # main training loop
    while self.completed_steps < self.config.trainer.max_train_steps:
        # ✅ 新增: 检查是否需要切换 stage（如果启用 curriculum learning）
        if self.enable_curriculum:
            current_stage = self._get_current_stage()
            # 检查是否需要更新 dataloader 的 stage
            # 注意：这需要重新创建 dataloader，实现较复杂
            # 可以考虑在每个 epoch 开始时检查
        
        # ... 其余训练循环代码保持不变 ...
```

#### 注意事项
1. **Dataloader 重建**：切换 stage 需要重新创建 dataloader，可能影响训练效率
2. **Checkpoint 兼容性**：不同 stage 的 checkpoint 可能不兼容
3. **实现复杂度**：需要修改 dataloader 以支持动态 stage 切换

#### 配置文件示例

```yaml
trainer:
  # ... 现有配置 ...
  
  # 新增: 多阶段训练（Curriculum Learning）
  curriculum_learning:
    enable: false  # 是否启用多阶段训练
    stage_schedule:
      0: 0      # step 0 开始使用 stage 0
      1000: 1   # step 1000 切换到 stage 1
      2000: 2   # step 2000 切换到 stage 2
```

---

## 四、配置文件扩展

### 4.1 训练配置文件示例

#### 完整配置示例（Stage 2）

```yaml
# 基础配置
run_id: "ecot_stage2_training"
run_root_dir: "./outputs"
seed: 42
is_debug: false

# 数据配置
datasets:
  vla_data:
    dataset_py: "ecot_rlds"
    per_device_batch_size: 4
    num_workers: 4
    image_size: [224, 224]
    ecot:
      data_root_dir: "/path/to/data"
      data_mix: "bridge"
      scheduled_stage: 2  # Stage 2: 有 thinking tokens
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
      action_dim: 7
      future_action_window_size: 15
      past_action_window_size: 0
      shuffle_buffer_size: 1000
      image_aug: false
      reasoning_json: "/path/to/reasoning.json"
      load_proprio: true
      lower_case_instruction: true
      train: true

# 模型配置
framework:
  name: "QwenGR00T"
  enable_latent_reasoning: true  # ✅ 开启隐式推理
  latent_reasoning:
    compute_language_loss: true  # ✅ 计算 VLM loss
    vlm_loss_weight: 0.1  # ✅ VLM loss 权重
    thinking_token: "<|thinking|>"
    start_of_thinking_token: "<|start_of_thinking|>"
    end_of_thinking_token: "<|end_of_thinking|>"
  qwenvl:
    base_vlm: "Qwen/Qwen3-VL-2B-Instruct"
    attn_implementation: "flash_attention_2"
    cache_dir: "./qwen_cache"
    model_max_length: 2048
  action_model:
    action_dim: 7
    future_action_window_size: 15
    past_action_window_size: 0

# 训练配置
trainer:
  max_train_steps: 10000
  gradient_accumulation_steps: 1
  learning_rate:
    base: 1.0e-5
  optimizer:
    betas: [0.9, 0.95]
    weight_decay: 0.01
    eps: 1.0e-8
  lr_scheduler_type: "cosine"
  num_warmup_steps: 100
  gradient_clipping: 1.0
  logging_frequency: 10
  save_interval: 1000
  eval_interval: 500
  repeated_diffusion_steps: 2
  
  # ✅ 可选: 多阶段训练
  curriculum_learning:
    enable: false
    stage_schedule:
      0: 0
      2000: 1
      5000: 2

# W&B配置
wandb_project: "ecot_training"
wandb_entity: null
```

---

## 五、修改实施步骤

### 步骤 1: 核心功能修改（必须）

1. ✅ 修改 `_train_step` 方法
   - 使用 `total_loss`（如果存在）
   - 返回 `vlm_loss` 和 `total_loss` 到 metrics

2. ✅ 修改 `_log_training_config` 方法
   - 添加 ECoT 配置日志

3. ✅ 添加配置验证函数
   - 在 `main` 函数中调用 `validate_ecot_config`

### 步骤 2: 代码清理（可选）

4. ⚪ 移除 QwenGR00T 中的调试日志
   - 将 `logger.info` 改为 `logger.debug`

### 步骤 3: 高级功能（可选）

5. ⚪ 实现多阶段训练支持
   - 添加 `_get_current_stage` 方法
   - 在训练循环中检查 stage 切换
   - 实现 dataloader 动态重建（如果需要）

---

## 六、测试验证

### 6.1 测试用例

#### 测试 1: Stage 0 训练（无 thinking tokens）
```bash
# 使用 test_ecot_stage0.yaml 配置
python train_starvla.py --config_yaml config/test_ecot_stage0.yaml
```

**验证点**：
- ✅ 配置验证通过
- ✅ 训练日志显示 "Stage 0: No thinking tokens"
- ✅ `vlm_loss` 被正确计算（如果 `compute_language_loss=True`）
- ✅ `total_loss = action_loss + vlm_loss_weight * vlm_loss`
- ✅ W&B 日志包含 `vlm_loss` 和 `total_loss`

#### 测试 2: Stage 2 训练（有 thinking tokens）
```bash
# 使用 test_ecot_stage2.yaml 配置
python train_starvla.py --config_yaml config/test_ecot_stage2.yaml
```

**验证点**：
- ✅ 配置验证通过
- ✅ 训练日志显示 "Stage 2: With thinking tokens"
- ✅ 使用 `forward_latent` 路径
- ✅ `vlm_loss` 被正确计算
- ✅ Thinking tokens 正确对齐
- ✅ Label masking 正确工作

#### 测试 3: 禁用 VLM Loss
```yaml
framework:
  latent_reasoning:
    compute_language_loss: false  # 禁用 VLM loss
```

**验证点**：
- ✅ `vlm_loss` 不在 metrics 中
- ✅ `total_loss = action_loss`
- ✅ 训练正常进行

#### 测试 4: 向后兼容性
```yaml
framework:
  enable_latent_reasoning: false  # 禁用隐式推理
```

**验证点**：
- ✅ 使用普通 forward 路径
- ✅ 不计算 `vlm_loss`
- ✅ `total_loss = action_loss`
- ✅ 训练正常进行

### 6.2 成功标准

#### 必须通过的检查项

1. **配置验证**
   - [ ] 配置验证函数正常执行
   - [ ] 配置错误时给出清晰警告
   - [ ] 配置正确时显示确认信息

2. **Loss 计算**
   - [ ] `vlm_loss` 正确计算（如果启用）
   - [ ] `total_loss` 正确合并
   - [ ] Loss 值不为 NaN/Inf

3. **日志记录**
   - [ ] 训练配置日志包含 ECoT 信息
   - [ ] W&B 日志包含 `vlm_loss` 和 `total_loss`
   - [ ] 控制台日志显示正确的 stage 信息

4. **训练稳定性**
   - [ ] 可以稳定训练至少 100 步
   - [ ] 梯度正常（不为 NaN/Inf）
   - [ ] 参数正常更新

---

## 七、预期效果

### 7.1 训练日志示例

#### Stage 0 训练日志
```
***** Training Configuration *****
  Total optimization steps = 10000
  Per device batch size = 4
  Gradient accumulation steps = 1
  Total batch size = 4

***** ECoT Implicit Reasoning Configuration *****
  Enable Latent Reasoning: True
  Scheduled Stage: 0
  Compute Language Loss: True
  VLM Loss Weight: 0.1

Step 0, Loss: {'action_loss': 2.3456, 'vlm_loss': 1.2345, 'total_loss': 2.4689, 'learning_rate': 1e-05}
Step 10, Loss: {'action_loss': 2.1234, 'vlm_loss': 1.1234, 'total_loss': 2.2357, 'learning_rate': 1e-05}
...
```

#### Stage 2 训练日志
```
***** Training Configuration *****
  Total optimization steps = 10000
  Per device batch size = 4
  Gradient accumulation steps = 1
  Total batch size = 4

***** ECoT Implicit Reasoning Configuration *****
  Enable Latent Reasoning: True
  Scheduled Stage: 2
  Compute Language Loss: True
  VLM Loss Weight: 0.1

Step 0, Loss: {'action_loss': 2.3456, 'vlm_loss': 1.2345, 'total_loss': 2.4689, 'learning_rate': 1e-05}
Step 10, Loss: {'action_loss': 2.1234, 'vlm_loss': 1.1234, 'total_loss': 2.2357, 'learning_rate': 1e-05}
...
```

### 7.2 W&B 监控指标

训练过程中，W&B 将记录以下指标：
- `action_loss`: Action prediction loss
- `vlm_loss`: VLM language modeling loss（如果启用）
- `total_loss`: Combined loss（`action_loss + vlm_loss_weight * vlm_loss`）
- `learning_rate`: 当前学习率
- `epoch`: 当前 epoch
- `data_time`: 数据加载时间
- `model_time`: 模型前向+反向时间

---

## 八、潜在问题和解决方案

### 问题 1: VLM Loss 为 None

**现象**: `output_dict["vlm_loss"]` 为 None

**可能原因**:
1. `compute_language_loss=False`
2. `labels` 未正确传递到模型
3. 模型 forward 路径未计算 loss

**解决方案**:
- 检查配置：确认 `compute_language_loss=True`
- 检查 `build_qwenvl_inputs`：确认返回了 `labels`
- 检查模型 forward：确认 `forward_latent` 或普通 forward 返回了 `loss`

### 问题 2: Total Loss 计算错误

**现象**: `total_loss` 与预期不符

**可能原因**:
1. `vlm_loss_weight` 配置错误
2. Loss 合并逻辑错误

**解决方案**:
- 检查配置：确认 `vlm_loss_weight` 值合理（建议 0.01-0.5）
- 检查模型代码：确认 `total_loss = action_loss + vlm_loss_weight * vlm_loss`

### 问题 3: 训练不稳定

**现象**: Loss 出现 NaN 或 Inf

**可能原因**:
1. `vlm_loss_weight` 过大
2. Label masking 错误
3. 数值溢出

**解决方案**:
- 减小 `vlm_loss_weight`（如从 0.1 降到 0.01）
- 检查 label masking：确认 instruction 和 thinking tokens 被正确 mask
- 添加 gradient clipping
- 检查学习率是否过大

### 问题 4: 性能下降

**现象**: Stage 2+ 训练速度明显变慢

**可能原因**:
1. KV-Cache 迭代 forward 开销大
2. Thinking tokens 数量过多

**解决方案**:
- 减少 thinking tokens 数量
- 优化 KV-Cache 使用
- 减小 batch size
- 使用 gradient checkpointing

---

## 九、实施时间估算

### 核心功能修改（必须）
- 修改 `_train_step`: ~30 分钟
- 修改 `_log_training_config`: ~15 分钟
- 添加配置验证: ~45 分钟
- **总计**: ~1.5 小时

### 代码清理（可选）
- 移除调试日志: ~15 分钟

### 高级功能（可选）
- 多阶段训练支持: ~2-3 小时（包括测试）

---

## 十、后续优化方向

### 10.1 性能优化
1. **Mixed Precision 训练**: 已支持（通过 `torch.autocast`）
2. **Gradient Checkpointing**: 可添加到 `forward_latent` 中
3. **DataLoader 优化**: 增加 `num_workers`，使用 prefetch

### 10.2 功能扩展
1. **多阶段训练**: 实现完整的 Curriculum Learning
2. **动态 Loss Weight**: 根据训练进度调整 `vlm_loss_weight`
3. **Stage 自动切换**: 根据 loss 趋势自动切换 stage

### 10.3 监控和调试
1. **Loss 分解监控**: 分别监控 instruction、thinking、post-thinking 的 loss
2. **Thinking Token 分析**: 分析 thinking tokens 的激活情况
3. **可视化工具**: 可视化 label masking 和 thinking token 位置

---

## 十一、总结

本方案详细规划了将 ECoT 隐式推理功能集成到训练脚本的修改点：

### 必须修改（核心功能）
1. ✅ `_train_step`: 处理 `vlm_loss` 和 `total_loss`
2. ✅ `_log_training_config`: 记录 ECoT 配置
3. ✅ 配置验证: 添加 `validate_ecot_config` 函数

### 可选修改（增强功能）
4. ⚪ 代码清理: 移除调试日志
5. ⚪ 多阶段训练: 实现 Curriculum Learning

### 关键设计原则
- **向后兼容**: 不启用隐式推理时，代码仍正常工作
- **配置驱动**: 通过配置文件控制行为，避免硬编码
- **清晰日志**: 提供详细的配置和训练信息
- **错误处理**: 优雅处理配置错误和异常情况

实施这些修改后，训练脚本将完全支持 ECoT 隐式推理训练，包括 Stage 0 和 Stage 2+ 两种模式。

