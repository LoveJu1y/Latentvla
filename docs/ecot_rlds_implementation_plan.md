## ECoT RLDS 集成代码实施方案（分步骤）

本方案描述如何在不修改 `prismatic/` 源码的前提下，将 ECoT（Prismatic/RLDS）数据以“非侵入式”的方式接入 `starVLA` 的训练流程。方案按步骤可直接落地，覆盖依赖、目录与骨架、数据适配、入口接入、配置、测试与验证等。

---

### 0. 依赖与准备
- 新增依赖清单（可选）：`requirements-ecot-rlds.txt`
  - 建议包含：`tensorflow`, `tensorflow_datasets`, `dlimp`, `huggingface_hub`（版本以 `prismatic` 兼容为准）。
- 数据与资源检查：
  - RLDS/OXE 数据（TFDS 结构）是否就绪。
  - reasoning JSON 路径：默认 `/share/project/lvjing/datas/embodied_features_bridge.json`；若不存在则由 `prismatic` 逻辑自动拉取并缓存。
- 说明：`prismatic` 侧已强制 `tf.config.set_visible_devices([], "GPU")`，避免与 PyTorch GPU 冲突。

验收
- 依赖可单独安装，不影响默认环境。
- 本地存在或可自动获取 reasoning JSON。

---

### 1. 新建目录与骨架文件
在 `starVLA` 内新增独立子模块（非侵入式）：
```
starVLA/integrations/ecot_rlds/
  __init__.py                 # 导出构建 API
  config.py                   # 默认常量、字段映射、参数校验
  transforms.py               # 轻量图像/多视角处理（PIL转换、resize）
  collate.py                  # collate_fn（返回样例列表，与现有 forward 兼容）
  dataset.py                  # 适配器数据集（核心）
  builder.py                  # get_vla_dataset_ecot/make_dataloader_ecot 入口
  README.md                   # 使用说明
  tests/
    smoke_test.py             # 冒烟测试：加载→取batch→前向
    dataset_contract_test.py  # 数据契约校验：字段/类型/形状
```

关键接口（文件/函数签名）
```python
# starVLA/integrations/ecot_rlds/__init__.py
from .builder import get_vla_dataset_ecot, make_dataloader_ecot

# starVLA/integrations/ecot_rlds/config.py
ECOT_DEFAULTS = {
    "shuffle_buffer_size": 256000,
    "image_aug": False,
    "scheduled_stage": 0,
    "reasoning_dropout_prob": 0.2,
    "reasoning_json": "/share/project/lvjing/datas/embodied_features_bridge.json",
}
def validate_and_normalize_cfg(ecot_cfg, global_cfg) -> dict: ...
def resolve_chunk_params(global_cfg) -> tuple[int, int, int]:  # past, current, future

# starVLA/integrations/ecot_rlds/transforms.py
def to_pil_list_from_numpy(primary: np.ndarray) -> list: ...
def ensure_image_size(pils: list, image_size: tuple[int, int]) -> list: ...

# starVLA/integrations/ecot_rlds/collate.py
def collate_fn_ecot(samples: list[dict]) -> list[dict]:
    return samples  # 返回样例列表，兼容 QwenGR00T.forward
```

验收
- 目录结构完整，可被 `import`。

---

### 2. 实现数据适配层（dataset.py，核心）
目标：只读复用 `prismatic` 的 RLDS 构建流程，自定义 batch transform，直接产出 StarVLA 期望的“样例字典”。

只读依赖
- `from prismatic.vla.datasets.rlds import make_interleaved_dataset`
- `from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights`
- `from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType`

实现要点
- 自定义 `ECOTBatchTransform`（参考 `prismatic.vla.datasets.datasets.RLDSBatchTransform`，但输出换为 StarVLA 结构）：
  - 输入：`rlds_batch`（来自 TFDS as_numpy_iterator）。
  - 输出：`dict`，字段：
    - `image`: `[PIL.Image, ...]`（主视角 `image_primary` 转 PIL，多视角保留列表；按 `image_size` 调整尺寸）。
    - `lang`: `task.language_instruction`（可统一 lower）。
    - `action`: 连续动作 `(chunk_len, action_dim)`；与 `future_action_window_size` 对齐（必要时截断/填充并记录一次性告警）。
    - `reasoning`: CoT 字符串（按阶段与 dropout 后）。
    - `state`（可选）：若存在 `proprio` 则拼接提供。
- 构建 `ECOTRLDSDataset(IterableDataset)`：
  - 组装与 `prismatic` 一致的 `rlds_config`（`traj_transform_kwargs`、`frame_transform_kwargs`、混合权重等），但替换 `batch_transform` 为 `ECOTBatchTransform`。
  - 保留 `dataset_length` 与 `dataset_statistics`（q01/q99 等）。
- 窗口长度对齐：
  - 从 `cfg.framework.action_model.future_action_window_size` 获取期望 `future`；保证与 `traj_transform_kwargs.future_action_window_size` 一致。
  - 若不一致，适配层对 `action` 做裁剪/填充，并在首次批次打印一次性告警。

关键接口（函数签名）
```python
class ECOTBatchTransform:
    def __init__(self, image_size: tuple[int, int], reasoning_dropout_prob: float, prompt_builder=None): ...
    def __call__(self, rlds_batch: dict) -> dict: ...

class ECOTRLDSDataset(torch.utils.data.IterableDataset):
    def __init__(self, data_root_dir: Path, data_mix: str, image_size: tuple[int, int],
                 shuffle_buffer_size: int, train: bool, image_aug: bool,
                 scheduled_stage: int, reasoning_dropout_prob: float,
                 future_action_window_size: int, cfg_obj: object | None = None) -> None: ...
    def __iter__(self): ...
    def __len__(self) -> int: ...
    @property
    def dataset_statistics(self) -> dict: ...
```

验收
- 可实例化数据集并迭代返回样例（dict）。

---

### 3. 入口与 DataLoader 封装（builder.py）
职责
- 读取全局 `cfg`，抽取：
  - `datasets.vla_data.image_size`、`per_device_batch_size`；
  - `datasets.vla_data.ecot.*` 子配置；
  - `framework.action_model.future_action_window_size`。
- 构造 `ECOTRLDSDataset`。
- 封装 `DataLoader`：
  - `batch_size=cfg.datasets.vla_data.per_device_batch_size`；
  - `collate_fn=collate_fn_ecot`；
  - `num_workers` 与现有 VLA 一致（如 4~8）。
- 保存统计：
  - `save_dataset_statistics(run_dir, dataset.dataset_statistics)`（可复用现有工具或在 builder 内实现）。

关键接口（函数签名）
```python
def get_vla_dataset_ecot(cfg) -> torch.utils.data.Dataset: ...
def make_dataloader_ecot(cfg) -> torch.utils.data.DataLoader: ...
def save_dataset_statistics(run_dir: Path, stats: dict) -> None: ...
```

验收
- 通过 `make_dataloader_ecot(cfg)` 返回 `DataLoader` 并可迭代。

---

### 4. 接入 build_dataloader（单点改动）
目标文件：`starVLA/dataloader/__init__.py`
- 在 `build_dataloader()` 增加：
  ```python
  elif dataset_py == "ecot_rlds":
      from starVLA.integrations.ecot_rlds.builder import make_dataloader_ecot
      return make_dataloader_ecot(cfg)
  ```

验收
- YAML 切换 `datasets.vla_data.dataset_py: ecot_rlds` 后走新通道。

---

### 5. YAML 配置与 Schema
- 示例：
```yaml
datasets:
  vla_data:
    dataset_py: ecot_rlds
    image_size: [224, 224]
    per_device_batch_size: 16
    ecot:
      data_root_dir: playground/Datasets/OXE_RLDS
      data_mix: bridge
      scheduled_stage: 0
      reasoning_dropout_prob: 0.2
      shuffle_buffer_size: 256000
      image_aug: false
      reasoning_json: /share/project/lvjing/datas/embodied_features_bridge.json

framework:
  action_model:
    action_dim: 7
    future_action_window_size: 15
    past_action_window_size: 0
```

验收
- 使用该 YAML 可启动训练并从 ECOT RLDS 通道读取数据。

---

### 6. 冒烟与契约测试
- `tests/smoke_test.py`：
  - 加载 `make_dataloader_ecot(cfg)`，取 1 个 batch，检查：
    - `len(batch)`；
    - `type(batch[0]["image"][0])` 为 `PIL.Image.Image`；
    - `batch[0]["action"].shape == (chunk_len, action_dim)`；
    - `batch[0]["lang"]` 为非空字符串；
    - 是否存在 `reasoning` 键。
  - 调用一次 `model.forward(batch)`，观察 `action_loss` 合理。
- `tests/dataset_contract_test.py`：
  - 严格校验每个样例字段、类型、形状。
  - 如发生窗口补齐/裁剪，首批打印一次性告警，测试日志中可见。

验收
- 冒烟与契约测试通过。

---

### 7. 小规模训练验证
- 使用小 batch（如 4）与 200 步：
  - 观察 `action_loss` 收敛；
  - 监控 `data_time/model_time`；
  - 检查 `dataset_statistics.json` 生成。
- 如性能瓶颈明显，尝试：
  - 下调 `shuffle_buffer_size`；
  - 关闭/降低 `image_aug`；
  - 调整 `num_workers`。

验收
- 训练稳定、无 OOM/死锁；loss 曲线合理。

---

### 8. （可选）接入 CoT 语言建模损失
- 不改 `prismatic/`，适配层仅保留 `reasoning`：
  - 在 Qwen Prompt 构建中注入 `reasoning` 作为 assistant 响应（新增开关）。
  - 构造 `labels` 并仅对 assistant 段计损。
- 训练端合并：
  - `total_loss = action_loss + λ * lm_loss`，`λ` 由 `cfg.trainer.loss_scale.vlm_cot` 控制（默认 0）。
- 增加指标：`lm_loss`、`total_loss`、`vlm_cot_weight`。

验收
- 启用后训练稳定；`lm_loss` 与 `action_loss` 均合理。

---

### 回滚与安全
- 回滚：将 YAML `dataset_py` 改回 `lerobot_datasets` 即可。
- 不修改 `prismatic/` 源码，改动可控；额外依赖单独清单，避免污染默认环境。

---

### 完成标准（总）
- 新增 `starVLA/integrations/ecot_rlds/` 子模块，接口稳定、可维护。
- YAML 可在 LeRobot/RLDS 通道间无缝切换。
- 首期动作监督通道稳定，统计与日志完整；为 CoT 语言损失打下基础。


