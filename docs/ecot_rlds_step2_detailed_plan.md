## 第二步骤详细实施计划：RLDS 数据适配层（dataset.py）

### 目标
- 实现 RLDS → StarVLA 的数据适配器，支持多时间步动作窗口（action chunking）。
- 完整保留并传递 CoT（Chain-of-Thought）推理文本，支持隐式推理训练。
- 确保与 `QwenGR00T` 模型配置完全对齐（动作窗口、状态、图像等）。

### 背景与关键约束

#### 1. QwenGR00T 的动作窗口配置
从 `QwenGR00T.py` 可知：
```python
self.future_action_window_size = config.framework.action_model.future_action_window_size  # 默认 15
self.past_action_window_size = config.framework.action_model.past_action_window_size      # 默认 0
self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size        # 默认 16
```

- **训练时**：模型期望 `action` 形状为 `[B, chunk_len, action_dim]`，即 `[B, 16, 7]`。
- **推理时**：预测未来 `chunk_len` 步的连续动作。
- **关键**：RLDS 数据集必须提供 `past + current + future` 的完整动作序列。

#### 2. RLDS 的时间窗口机制
Prismatic 的 RLDS 管线通过 `traj_transform_kwargs` 控制：
```python
traj_transform_kwargs=dict(
    window_size=1,                          # 观测窗口（通常为 1，只取当前帧）
    future_action_window_size=7,            # 未来动作步数（需与模型对齐）
    skip_unlabeled=True,
    goal_relabeling_strategy="uniform",
)
```

- `window_size=1`：只取当前时刻的观测（图像、状态）。
- `future_action_window_size`：控制返回的动作序列长度（当前 + 未来）。
- **输出**：`action` 形状为 `[window_size, future_action_window_size + 1, action_dim]`，即 `[1, 8, 7]`（如果设为 7）。

#### 3. CoT 数据的结构
从 `prismatic/vla/datasets/rlds/dataset.py` 可知：
- `reasoning` 字段通过 TF lookup table 从 JSON 索引中获取。
- 格式：`"@TAG1@content1@TAG2@content2..."`（按阶段与 dropout 处理后）。
- **关键**：每个时间步都有对应的 reasoning，但通常只使用当前时刻的推理文本。

#### 4. 隐式推理的训练需求
- **第一阶段**（当前）：仅用连续动作监督，保留 `reasoning` 字段供后续使用。
- **第二阶段**（可选）：将 `reasoning` 作为 assistant 响应，计算语言建模损失。
- **关键设计**：
  - `reasoning` 应对应"当前时刻"的推理（即 `window_size=1` 时的第 0 帧）。
  - 动作序列覆盖"当前+未来"，与推理时刻对齐。

---

### 数据流与字段映射

#### RLDS 批次输出（来自 `as_numpy_iterator`）
```python
rlds_batch = {
    "observation": {
        "image_primary": np.ndarray,  # [window_size, H, W, 3]，通常 [1, 224, 224, 3]
        "proprio": np.ndarray,        # [window_size, state_dim]，可选
        "timestep": np.ndarray,       # [window_size]
    },
    "task": {
        "language_instruction": bytes,  # b"pick up the cup"
    },
    "action": np.ndarray,              # [window_size, future_action_window_size+1, action_dim]
                                       # 例如 [1, 16, 7]（如果 future=15）
    "dataset_name": bytes,             # b"bridge_dataset"
    "reasoning": bytes,                # b"@TASK@...@ACTION@..."（CoT 字符串）
}
```

#### StarVLA 期望的样例格式
```python
sample = {
    "image": [PIL.Image, ...],         # 多视角列表，每个 PIL.Image 尺寸为 (H, W)
    "lang": str,                       # "pick up the cup"（小写）
    "action": np.ndarray,              # [chunk_len, action_dim]，例如 [16, 7]
    "state": np.ndarray,               # [chunk_len, state_dim] 或 [state_dim]（可选）
    "reasoning": str,                  # CoT 文本（已 dropout）
    "reasoning_subset": str,           # 保留的 CoT 组件标记（如 "[TH, ST, MV]"）
}
```

---

### 实施计划（9 个子任务）

#### 任务 2.1：配置读取与校验（`config.py` 增强）
**目标**：从全局 `cfg` 提取并校验所有必需参数。

**实现要点**：
- 在 `config.py` 新增 `validate_and_normalize_cfg(ecot_cfg, global_cfg)`：
  ```python
  def validate_and_normalize_cfg(ecot_cfg, global_cfg):
      """
      从 cfg 提取并校验 ECOT 数据集所需的所有参数。
      
      Args:
          ecot_cfg: cfg.datasets.vla_data.ecot（子配置）
          global_cfg: 完整 cfg（用于提取 framework.action_model 等）
      
      Returns:
          normalized_cfg: dict，包含所有标准化后的参数
      """
      # 1. 提取动作窗口参数
      future_action_window_size = global_cfg.framework.action_model.future_action_window_size  # 15
      past_action_window_size = global_cfg.framework.action_model.past_action_window_size      # 0
      chunk_len = past_action_window_size + 1 + future_action_window_size                      # 16
      action_dim = global_cfg.framework.action_model.action_dim                                # 7
      
      # 2. 提取图像参数
      image_size = global_cfg.datasets.vla_data.image_size  # [224, 224]
      
      # 3. 提取 ECOT 专属参数（带默认值）
      data_root_dir = ecot_cfg.get("data_root_dir")
      data_mix = ecot_cfg.get("data_mix")
      scheduled_stage = ecot_cfg.get("scheduled_stage", ECOT_DEFAULTS["scheduled_stage"])
      reasoning_dropout_prob = ecot_cfg.get("reasoning_dropout_prob", ECOT_DEFAULTS["reasoning_dropout_prob"])
      shuffle_buffer_size = ecot_cfg.get("shuffle_buffer_size", ECOT_DEFAULTS["shuffle_buffer_size"])
      image_aug = ecot_cfg.get("image_aug", ECOT_DEFAULTS["image_aug"])
      reasoning_json = ecot_cfg.get("reasoning_json", ECOT_DEFAULTS["reasoning_json"])
      load_proprio = ecot_cfg.get("load_proprio", True)
      lower_case_instruction = ecot_cfg.get("lower_case_instruction", True)
      
      # 4. 校验必需参数
      if not data_root_dir:
          raise ValueError("ecot.data_root_dir is required")
      if not data_mix:
          raise ValueError("ecot.data_mix is required")
      
      # 5. 校验范围
      if not (0 <= reasoning_dropout_prob <= 1):
          raise ValueError(f"reasoning_dropout_prob must be in [0, 1], got {reasoning_dropout_prob}")
      if scheduled_stage < 0:
          raise ValueError(f"scheduled_stage must be >= 0, got {scheduled_stage}")
      
      return {
          # 动作窗口
          "future_action_window_size": future_action_window_size,
          "past_action_window_size": past_action_window_size,
          "chunk_len": chunk_len,
          "action_dim": action_dim,
          # 图像
          "image_size": tuple(image_size),
          # RLDS
          "data_root_dir": Path(data_root_dir),
          "data_mix": data_mix,
          "shuffle_buffer_size": shuffle_buffer_size,
          "image_aug": image_aug,
          "train": True,  # 默认训练模式
          # CoT
          "scheduled_stage": scheduled_stage,
          "reasoning_dropout_prob": reasoning_dropout_prob,
          "reasoning_json": reasoning_json,
          # 其他
          "load_proprio": load_proprio,
          "lower_case_instruction": lower_case_instruction,
      }
  ```

**验收**：
- 缺少必需参数时抛出清晰异常。
- 使用完整 YAML 时能正确读取全部参数。

---

#### 任务 2.2：RLDS 数据集构建（`dataset.py` 核心）
**目标**：调用 Prismatic 的 `make_interleaved_dataset`，传入正确的窗口参数。

**实现要点**：
```python
class ECOTRLDSDataset(torch.utils.data.IterableDataset):
    def __init__(self, cfg):
        # 1. 校验并提取配置
        self.cfg_dict = validate_and_normalize_cfg(cfg.datasets.vla_data.ecot, cfg)
        
        # 2. 构建 RLDS 配置
        # 关键：future_action_window_size 必须与模型一致
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.cfg_dict["data_root_dir"],
            OXE_NAMED_MIXTURES[self.cfg_dict["data_mix"]],
            load_camera_views=("primary",),  # 可扩展为多视角
            load_depth=False,
            load_proprio=self.cfg_dict["load_proprio"],
            load_language=True,
            action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
        )
        
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=1,  # 只取当前帧观测
                future_action_window_size=self.cfg_dict["future_action_window_size"],  # 15
                skip_unlabeled=True,
                goal_relabeling_strategy="uniform",
            ),
            frame_transform_kwargs=dict(
                resize_size=self.cfg_dict["image_size"],  # (224, 224)
                num_parallel_calls=16,
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=self.cfg_dict["shuffle_buffer_size"],
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(per_dataset_kwargs),
            traj_read_threads=len(per_dataset_kwargs),
            train=self.cfg_dict["train"],
            scheduled_stage=self.cfg_dict["scheduled_stage"],
            cfg=cfg,  # 传入完整 cfg 供 reasoning 构建使用
        )
        
        # 3. 构建 RLDS 数据集
        self.rlds_dataset, self.dataset_length, self.dataset_statistics = \
            make_interleaved_dataset(**rlds_config)
        
        # 4. 创建批次变换器
        self.batch_transform = ECOTBatchTransform(
            image_size=self.cfg_dict["image_size"],
            chunk_len=self.cfg_dict["chunk_len"],
            action_dim=self.cfg_dict["action_dim"],
            reasoning_dropout_prob=self.cfg_dict["reasoning_dropout_prob"],
            lower_case_instruction=self.cfg_dict["lower_case_instruction"],
            load_proprio=self.cfg_dict["load_proprio"],
        )
        
        # 5. 日志
        self._log_once = True
    
    def __iter__(self):
        for rlds_batch in self.rlds_dataset.as_numpy_iterator():
            sample = self.batch_transform(rlds_batch)
            
            # 首次日志（调试用）
            if self._log_once:
                logger.info(f"[ECOT RLDS] First sample:")
                logger.info(f"  image: {len(sample['image'])} views, size={sample['image'][0].size}")
                logger.info(f"  lang: {sample['lang'][:60]}...")
                logger.info(f"  action: {sample['action'].shape}")
                logger.info(f"  reasoning: {sample['reasoning'][:80]}...")
                if "state" in sample:
                    logger.info(f"  state: {sample['state'].shape}")
                self._log_once = False
            
            yield sample
    
    def __len__(self):
        return self.dataset_length
```

**验收**：
- 能成功创建数据集对象。
- `dataset_length` 和 `dataset_statistics` 非空。

---

#### 任务 2.3：动作窗口对齐与裁剪（`ECOTBatchTransform`）
**目标**：将 RLDS 返回的动作序列对齐到模型期望的 `chunk_len`。

**关键问题**：
- RLDS 返回 `[window_size, future_action_window_size+1, action_dim]`，例如 `[1, 16, 7]`。
- 需要展平为 `[chunk_len, action_dim]`，即 `[16, 7]`。
- 如果 RLDS 配置的 `future_action_window_size` 与模型不一致，需要裁剪或填充。

**实现要点**：
```python
class ECOTBatchTransform:
    def __init__(self, image_size, chunk_len, action_dim, reasoning_dropout_prob, 
                 lower_case_instruction, load_proprio):
        self.image_size = image_size
        self.chunk_len = chunk_len
        self.action_dim = action_dim
        self.reasoning_dropout_prob = reasoning_dropout_prob
        self.lower_case_instruction = lower_case_instruction
        self.load_proprio = load_proprio
        self._action_mismatch_warned = False
    
    def __call__(self, rlds_batch):
        # 1. 提取动作（关键）
        action = rlds_batch["action"]  # [window_size, future+1, action_dim]
        
        # 展平时间维度：只取 window_size=0（当前帧）
        action = action[0]  # [future+1, action_dim]，例如 [16, 7]
        
        # 对齐到 chunk_len
        if action.shape[0] > self.chunk_len:
            # 截取末尾 chunk_len 步
            action = action[-self.chunk_len:]
            if not self._action_mismatch_warned:
                logger.warning(
                    f"[ECOT RLDS] Action sequence length ({action.shape[0]}) > chunk_len ({self.chunk_len}), "
                    f"truncating to last {self.chunk_len} steps."
                )
                self._action_mismatch_warned = True
        elif action.shape[0] < self.chunk_len:
            # 填充零（不推荐，应在配置时避免）
            padding = np.zeros((self.chunk_len - action.shape[0], self.action_dim), dtype=action.dtype)
            action = np.concatenate([action, padding], axis=0)
            if not self._action_mismatch_warned:
                logger.warning(
                    f"[ECOT RLDS] Action sequence length ({action.shape[0]}) < chunk_len ({self.chunk_len}), "
                    f"padding with zeros. This may degrade performance!"
                )
                self._action_mismatch_warned = True
        
        # 最终形状：[chunk_len, action_dim]
        assert action.shape == (self.chunk_len, self.action_dim), \
            f"Action shape mismatch: expected {(self.chunk_len, self.action_dim)}, got {action.shape}"
        
        # ... 其他字段处理（见后续任务）
        
        return {
            "action": action,
            # ...
        }
```

**验收**：
- 动作形状始终为 `[chunk_len, action_dim]`。
- 长度不匹配时只打印一次告警。

---

#### 任务 2.4：图像转换与多视角处理（`transforms.py` + `ECOTBatchTransform`）
**目标**：将 RLDS 的图像数组转为 PIL，支持多视角扩展。

**实现要点**：
```python
# 在 ECOTBatchTransform.__call__ 中：
def __call__(self, rlds_batch):
    # 2. 提取图像
    image_primary = rlds_batch["observation"]["image_primary"]  # [window_size, H, W, 3]
    
    # 只取当前帧（window_size=0）
    image_primary = image_primary[0]  # [H, W, 3]
    
    # 转为 PIL 并 resize
    pil_images = [to_pil_and_resize(image_primary, self.image_size)]
    
    # 多视角扩展（未来）：
    # if "image_wrist" in rlds_batch["observation"]:
    #     image_wrist = rlds_batch["observation"]["image_wrist"][0]
    #     pil_images.append(to_pil_and_resize(image_wrist, self.image_size))
    
    return {
        "image": pil_images,
        # ...
    }

# 在 transforms.py 中：
def to_pil_and_resize(img_array: np.ndarray, target_size: tuple) -> Image.Image:
    """
    将 numpy 数组转为 PIL.Image 并 resize。
    
    Args:
        img_array: [H, W, 3] uint8 数组
        target_size: (H, W) 目标尺寸
    
    Returns:
        PIL.Image，尺寸为 (W, H)（PIL 约定）
    """
    if img_array.dtype != np.uint8:
        img_array = (img_array * 255).astype(np.uint8)
    
    pil_img = Image.fromarray(img_array)
    
    # PIL.Image.resize 参数为 (width, height)
    pil_img = pil_img.resize((target_size[1], target_size[0]), Image.BILINEAR)
    
    return pil_img
```

**验收**：
- 样例中 `image` 为 `[PIL.Image]` 列表。
- 每个图像尺寸为 `(W, H)`，符合 `image_size`。

---

#### 任务 2.5：语言与推理字段解码（`ECOTBatchTransform`）
**目标**：安全解码 RLDS 的 bytes 字段，支持 CoT dropout。

**实现要点**：
```python
def _to_str(self, value) -> str:
    """安全解码 RLDS 字段为字符串"""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    elif isinstance(value, np.ndarray):
        if value.size == 1:
            return str(value.item())
        else:
            raise ValueError(f"Cannot convert multi-element array to string: {value}")
    elif isinstance(value, str):
        return value
    elif value is None or (isinstance(value, np.ndarray) and value.size == 0):
        return ""
    else:
        return str(value)

def __call__(self, rlds_batch):
    # 3. 提取语言指令
    lang = self._to_str(rlds_batch["task"]["language_instruction"])
    if self.lower_case_instruction:
        lang = lang.lower()
    
    # 4. 提取推理文本（关键）
    reasoning_raw = self._to_str(rlds_batch["reasoning"])
    
    # CoT dropout（复用 prismatic 的逻辑）
    from prismatic.vla.datasets.datasets import reasoning_dropout
    reasoning, reasoning_subset = reasoning_dropout(
        reasoning_raw, 
        dropout_prob=self.reasoning_dropout_prob
    )
    
    return {
        "lang": lang,
        "reasoning": reasoning,
        "reasoning_subset": reasoning_subset,
        # ...
    }
```

**验收**：
- `lang` 为非空字符串。
- `reasoning` 为 CoT 文本（可能为空，取决于 dropout）。
- `reasoning_subset` 记录保留的组件（如 `"[TH, ST, MV]"`）。

---

#### 任务 2.6：状态（proprio）处理（`ECOTBatchTransform`）
**目标**：可选地提取并对齐状态序列。

**实现要点**：
```python
def __call__(self, rlds_batch):
    # 5. 提取状态（可选）
    state = None
    if self.load_proprio and "proprio" in rlds_batch["observation"]:
        proprio = rlds_batch["observation"]["proprio"]  # [window_size, state_dim]
        
        # 只取当前帧
        state = proprio[0]  # [state_dim]
        
        # 可选：扩展为与动作相同的时间维度
        # state = np.tile(state[None, :], (self.chunk_len, 1))  # [chunk_len, state_dim]
    
    result = {
        # ...
    }
    
    if state is not None:
        result["state"] = state
    
    return result
```

**验收**：
- `load_proprio=True` 时，样例包含 `state` 字段。
- `load_proprio=False` 时，样例不包含 `state` 字段。

---

#### 任务 2.7：数据统计清理（`dataset.py`）
**目标**：将 `dataset_statistics` 转为可 JSON 序列化的格式。

**实现要点**：
```python
def _clean_statistics(self, stats: dict) -> dict:
    """递归清理统计数据，移除不可序列化的类型"""
    cleaned = {}
    for key, value in stats.items():
        if isinstance(value, dict):
            cleaned[key] = self._clean_statistics(value)
        elif isinstance(value, np.ndarray):
            cleaned[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            cleaned[key] = value.item()
        elif isinstance(value, Path):
            cleaned[key] = str(value)
        else:
            cleaned[key] = value
    return cleaned

# 在 __init__ 中：
self.dataset_statistics = self._clean_statistics(self.dataset_statistics)
```

**验收**：
- `dataset.dataset_statistics` 可用 `json.dumps()` 序列化。

---

#### 任务 2.8：错误处理与日志（`ECOTBatchTransform`）
**目标**：对关键字段缺失或格式错误给出清晰提示。

**实现要点**：
```python
def __call__(self, rlds_batch):
    try:
        # 检查必需字段
        if "observation" not in rlds_batch:
            raise KeyError("Missing 'observation' in RLDS batch")
        if "image_primary" not in rlds_batch["observation"]:
            raise KeyError("Missing 'image_primary' in observation")
        if "action" not in rlds_batch:
            raise KeyError("Missing 'action' in RLDS batch")
        if "task" not in rlds_batch or "language_instruction" not in rlds_batch["task"]:
            raise KeyError("Missing 'language_instruction' in task")
        
        # ... 正常处理 ...
        
    except Exception as e:
        logger.error(f"[ECOT RLDS] Failed to transform batch: {e}")
        logger.error(f"  Available keys: {list(rlds_batch.keys())}")
        raise
```

**验收**：
- 数据格式错误时异常信息清晰。

---

#### 任务 2.9：最小测试（`tests/dataset_contract_test.py`）
**目标**：编写单元测试验证数据契约。

**实现要点**：
```python
import pytest
from starVLA.integrations.ecot_rlds.dataset import ECOTRLDSDataset
from omegaconf import OmegaConf

@pytest.mark.skip(reason="Requires RLDS data and TensorFlow")
def test_ecot_rlds_dataset_contract():
    # 加载测试配置
    cfg = OmegaConf.load("path/to/test_config.yaml")
    
    # 创建数据集
    dataset = ECOTRLDSDataset(cfg)
    
    # 取一个样例
    sample = next(iter(dataset))
    
    # 验证字段
    assert "image" in sample
    assert isinstance(sample["image"], list)
    assert len(sample["image"]) > 0
    assert sample["image"][0].size == (224, 224)  # (W, H)
    
    assert "lang" in sample
    assert isinstance(sample["lang"], str)
    assert len(sample["lang"]) > 0
    
    assert "action" in sample
    assert sample["action"].shape == (16, 7)  # (chunk_len, action_dim)
    
    assert "reasoning" in sample
    assert isinstance(sample["reasoning"], str)
    
    if "state" in sample:
        assert sample["state"].shape[0] == 7  # state_dim
```

**验收**：
- 测试通过（或可 skip，但代码无误）。

---

### 关键设计决策

#### 1. 动作窗口对齐策略
- **推荐**：在 YAML 中将 `traj_transform_kwargs.future_action_window_size` 设为 15（与模型一致）。
- **兜底**：适配层支持截断/填充，但会打印告警。

#### 2. CoT 数据的时间对齐
- **当前帧推理**：`reasoning` 对应 `window_size=0` 的时刻，与"当前动作"对齐。
- **未来动作**：动作序列覆盖"当前+未来 15 步"，推理文本指导整个序列的生成。

#### 3. 多视角图像扩展
- **当前**：只加载 `primary` 视角。
- **未来**：在 `get_oxe_dataset_kwargs_and_weights` 中设置 `load_camera_views=("primary", "wrist")`，并在 `ECOTBatchTransform` 中处理多视角。

#### 4. 状态（proprio）的使用
- **默认**：`load_proprio=True`，提供 `[state_dim]` 形状的状态。
- **可选**：扩展为 `[chunk_len, state_dim]`，与动作时间维度对齐（需在模型侧支持）。

---

### 配置示例（YAML）

```yaml
datasets:
  vla_data:
    dataset_py: ecot_rlds
    image_size: [224, 224]
    per_device_batch_size: 16
    ecot:
      data_root_dir: playground/Datasets/OXE_RLDS
      data_mix: bridge
      scheduled_stage: 0              # 0=全量 CoT
      reasoning_dropout_prob: 0.2     # 20% dropout
      shuffle_buffer_size: 256000
      image_aug: false
      reasoning_json: /share/project/lvjing/datas/embodied_features_bridge.json
      load_proprio: true              # 加载状态
      lower_case_instruction: true    # 指令小写

framework:
  action_model:
    action_dim: 7
    future_action_window_size: 15     # 关键：与 RLDS 对齐
    past_action_window_size: 0
    # chunk_len = 0 + 1 + 15 = 16
```

---

### 验收清单

- [ ] 配置校验：缺少必需参数时报错清晰。
- [ ] RLDS 构建：能成功创建数据集，`dataset_length` 非零。
- [ ] 动作对齐：`action.shape == (16, 7)`，长度不匹配时打印告警。
- [ ] 图像转换：`image` 为 `[PIL.Image]`，尺寸正确。
- [ ] 语言解码：`lang` 非空，`reasoning` 包含 CoT 文本。
- [ ] 状态处理：`load_proprio=True` 时存在 `state` 字段。
- [ ] 统计清理：`dataset_statistics` 可 JSON 序列化。
- [ ] 错误处理：异常信息清晰，便于调试。
- [ ] 单元测试：契约测试通过（或可 skip）。

---

### 后续步骤

完成第二步骤后，进入第三步骤：
- 在 `builder.py` 中封装 `make_dataloader_ecot`。
- 在 `build_dataloader` 中接入 `ecot_rlds` 分支。
- 编写 smoke test 验证端到端流程。

---

### 附录：关键代码片段汇总

#### A. 配置校验（`config.py`）
```python
def validate_and_normalize_cfg(ecot_cfg, global_cfg):
    # 提取动作窗口
    future_action_window_size = global_cfg.framework.action_model.future_action_window_size
    past_action_window_size = global_cfg.framework.action_model.past_action_window_size
    chunk_len = past_action_window_size + 1 + future_action_window_size
    action_dim = global_cfg.framework.action_model.action_dim
    
    # 提取图像参数
    image_size = global_cfg.datasets.vla_data.image_size
    
    # 提取 ECOT 参数
    data_root_dir = ecot_cfg.get("data_root_dir")
    data_mix = ecot_cfg.get("data_mix")
    # ... 其他参数 ...
    
    # 校验
    if not data_root_dir or not data_mix:
        raise ValueError("data_root_dir and data_mix are required")
    
    return {
        "future_action_window_size": future_action_window_size,
        "chunk_len": chunk_len,
        "action_dim": action_dim,
        "image_size": tuple(image_size),
        "data_root_dir": Path(data_root_dir),
        "data_mix": data_mix,
        # ...
    }
```

#### B. 动作对齐（`ECOTBatchTransform`）
```python
def __call__(self, rlds_batch):
    action = rlds_batch["action"][0]  # [future+1, action_dim]
    
    if action.shape[0] > self.chunk_len:
        action = action[-self.chunk_len:]
        if not self._action_mismatch_warned:
            logger.warning(f"Truncating action sequence to {self.chunk_len} steps")
            self._action_mismatch_warned = True
    elif action.shape[0] < self.chunk_len:
        padding = np.zeros((self.chunk_len - action.shape[0], self.action_dim))
        action = np.concatenate([action, padding], axis=0)
        if not self._action_mismatch_warned:
            logger.warning(f"Padding action sequence with zeros")
            self._action_mismatch_warned = True
    
    return {"action": action, ...}
```

#### C. 图像转换（`transforms.py`）
```python
def to_pil_and_resize(img_array: np.ndarray, target_size: tuple) -> Image.Image:
    if img_array.dtype != np.uint8:
        img_array = (img_array * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_array)
    pil_img = pil_img.resize((target_size[1], target_size[0]), Image.BILINEAR)
    return pil_img
```

#### D. CoT dropout（`ECOTBatchTransform`）
```python
from prismatic.vla.datasets.datasets import reasoning_dropout

reasoning_raw = self._to_str(rlds_batch["reasoning"])
reasoning, reasoning_subset = reasoning_dropout(
    reasoning_raw, 
    dropout_prob=self.reasoning_dropout_prob
)
```

---

### 总结

本计划将第二步骤拆解为 9 个可独立验收的子任务，覆盖配置、RLDS 构建、动作对齐、图像/语言/推理处理、状态、统计、错误处理与测试。完成后，数据适配层将稳定支持多时间步动作窗口与 CoT 隐式推理，为后续训练打下坚实基础。

