## StarVLA 集成 ECoT（Prismatic/RLDS）数据计划

### 目标
- 将 `prismatic`/ECoT 的 RLDS 数据（含隐式推理与 CoT 标注）无缝纳入现有 `starVLA` 训练流程。
- 第一阶段以“连续动作监督”为主；保留/传递“推理文本”字段，便于后续开启语言建模损失与阶段式思维 token 机制。

### 背景与现状
- 训练主循环见 `starVLA/training/train_starvla.py`，通过 `build_dataloader()` 拉取 VLA 数据，`QwenGR00T.forward()` 期望批次样例为若干 dict 的列表，至少包含：
  - `image`: `[PIL.Image, ...]` 多视角图像
  - `lang`: `str` 指令
  - `action`: `(T, action_dim)` 连续动作序列
- 现有数据接入是 LeRobot 体系（`starVLA/dataloader/lerobot_datasets.py`）。
- 新增的 `prismatic/vla/datasets` 已实现 RLDS → 模型输入的核心流水（OXE 混合、reasoning 构造/阶段/Dropout、统计等）。

### 总体方案
新增一个“数据适配层”，将 ECoT RLDS 的批次转换为 `starVLA` 期望的样例结构；在 `build_dataloader()` 中增设入口；在 YAML 中增加 RLDS 与 ECoT 专属超参；随后逐步开启语言建模损失（可选）。

---

### 分阶段实施计划

#### 阶段 A：环境与依赖
- 安装/校验 Prismatic RLDS 要求依赖：
  - `tensorflow`, `tensorflow_datasets`, `dlimp`, `huggingface_hub`（具体版本以 `prismatic` 兼容为准）。
  - TensorFlow GPU 屏蔽：Prismatic 代码内已 `tf.config.set_visible_devices([], "GPU")`，避免与 PyTorch 冲突。
- 准备 reasoning 索引 JSON：
  - 默认路径：`/share/project/lvjing/datas/embodied_features_bridge.json`；若不存在，按 Prismatic 逻辑自动从 HF 拉取并缓存。

#### 阶段 B：数据适配层（核心）
- 新增文件 `starVLA/dataloader/prismatic_rlds.py`，功能：
  1) 构建 RLDS Dataset：
     - 复用 `prismatic.vla.datasets.datasets.RLDSDataset`，从配置读取：
       - `data_root_dir`、`data_mix`（OXE 预设或单数据集名）。
       - `resize_resolution`（对齐 `cfg.datasets.vla_data.image_size`）。
       - `image_aug`、`shuffle_buffer_size`、`train`。
       - `scheduled_stage`（阶段式 CoT）与 `reasoning_dropout_prob`（隐式推理保留率）。
       - `future_action_window_size` 对齐 `cfg.framework.action_model.future_action_window_size`（默认 15，整体 chunk=16）。
  2) 批次变换输出映射为 StarVLA 样例列表：
     - `image`: 将 RLDS 帧转换为 `PIL.Image` 列表（多视角时保留列表，尺寸按 `image_size`）。
     - `lang`: 使用 `language_instruction`，如存在 `CoT_prompt`，则以其格式化（与现有 Qwen 接入一致）。
     - `reasoning`/`cot`: 保留完整 CoT 字符串（已按阶段/Dropout处理）；额外保存 `subset`（可选）。
     - `action`: 连续动作序列，形如 `(chunk_len, action_dim)`；确保与模型设置一致（必要时截断/填充，打印告警）。
     - `state`（可选）：如 RLDS 含 `proprio`，则拼接后提供，供后续扩展。
  3) `collate_fn`：保持返回“样例列表”的策略（适配当前 `QwenGR00T.forward`）。若后续启用文本损失，再扩展聚合 `input_ids/labels`。
  4) `save_dataset_statistics`：从 `dataset.dataset_statistics`（含 q01/q99）持久化到 `run_dir`，便于日志与后处理。

#### 阶段 C：集成入口
- 修改 `starVLA/dataloader/__init__.py::build_dataloader`：
  - 新增 `elif dataset_py == "prismatic_rlds":` 分支，调用 `get_vla_dataset_rlds()` 并返回 `DataLoader`。
  - `DataLoader` 参数与 LeRobot 分支保持一致：`batch_size`、`num_workers`、统一 `collate_fn`。

#### 阶段 D：模型与训练兼容性
- 第一阶段维持“动作监督”路径：`QwenGR00T.forward()` 从样例中读取 `image/lang/action`，按 `chunk_len` 对齐监督；无需改模型。
- 第二阶段（可选）加入语言建模损失（CoT 监督）：
  - 在 Qwen 接口构建 Chat Prompt 时，将 `reasoning` 作为 assistant 回复加入对话；传入 `labels` 计算 LM Loss。
  - 训练端合并损失：`total_loss = action_loss + λ * lm_loss`，通过 `cfg.trainer.loss_scale.vlm_cot` 控制（默认 0）。
  - 与 `CoT_prompt` 分工：`CoT_prompt` 用于输入提示；`reasoning` 作为输出标签。

#### 阶段 E：验证与监控
- Sanity Check：
  - 取 1-2 个 batch，核对：
    - `image` 类型/尺寸（PIL，按 `image_size` 重设）；
    - `lang` 文本；
    - `action` 形状 `[chunk_len, action_dim]`；
    - `reasoning` 字符串（阶段化/Dropout 符合预期）。
  - 打印/保存 `dataset_statistics.json`（q01/q99）。
- 前向与小规模训练：
  - 单步 `model.forward(batch)` 无异常，loss 数值合理；
  - 跑 100~500 steps：loss 收敛趋势、吞吐、W&B 指标稳定；
  - 如启用 LM Loss：校验仅对 assistant 段计损且 mask 正确。

---

### 架构与接口变更点
1) 新增：`starVLA/dataloader/prismatic_rlds.py`
   - `get_vla_dataset_rlds(data_cfg)`：构建 RLDSDataset 与 DatasetStatistics；返回 `torch.utils.data.Dataset`。
   - `collate_fn`：返回样例列表，字段：`image`/`lang`/`action`/`reasoning`（可选 `state`）。
2) 修改：`starVLA/dataloader/__init__.py`
   - `build_dataloader(cfg, dataset_py=...)` 新增 `prismatic_rlds` 分支。
3) 配置：训练 YAML 新增/扩展项（见下）。

---

### 配置变更（示例片段）
```yaml
datasets:
  vla_data:
    dataset_py: prismatic_rlds
    # RLDS 根目录（TFDS 格式/OXE 数据）
    data_root_dir: playground/Datasets/OXE_RLDS
    # OXE 预设混合名或单一数据集名（示例：bridge / bridge_rt_1 / nyu_franka_play_dataset_converted_externally_to_rlds）
    data_mix: bridge
    # 统一图像尺寸（与模型训练保持一致）
    image_size: [224, 224]
    per_device_batch_size: 16

    # ECoT / RLDS 专属
    rlds:
      shuffle_buffer_size: 256000
      image_aug: false
      scheduled_stage: 0            # 0=全量 CoT；>0 逐步裁剪
      reasoning_dropout_prob: 0.2   # 推理文本随机保留率
      reasoning_json: /share/project/lvjing/datas/embodied_features_bridge.json

framework:
  action_model:
    action_dim: 7
    future_action_window_size: 15  # RLDS 窗口对齐 → chunk_len = 16
    past_action_window_size: 0

trainer:
  loss_scale:
    vla: 1.0
    vlm_cot: 0.0   # 第二阶段再开启，例如 0.05~0.2
```

---

### 风险与缓解
- 动作窗口不匹配：以配置驱动，确保 RLDS 与模型 `future_action_window_size` 一致；适配层做截断/填充并记录告警。
- 资源/性能瓶颈：适当下调 `shuffle_buffer_size`、关闭/降低 `image_aug`、减少并行线程。
- LM Loss 掩码风险：启用后仅计算 assistant 段；与 Qwen Chat 模板严格对齐，避免 label shift。
- 数据缺项：如缺 `proprio`、缺 `reasoning`，保持可选键，必要时回退为动作监督。

---

### 里程碑（建议）
1) D1：依赖安装、数据与 reasoning 资源检查（含本地/自动拉取）。
2) D2：完成 `prismatic_rlds.py` 与 `build_dataloader` 接入。
3) D3：配置与最小可运行验证（取 batch，前向跑通，统计保存）。
4) D4：小规模训练（100~500 steps），监控与日志校验。
5) D5（可选）：接入 CoT 语言建模损失，权重调参与对比实验。
6) D6：文档完善与清理。

---

### 验收清单
- 能通过 `dataset_py: prismatic_rlds` 拉取 RLDS+ECoT 数据，样例结构与现有模型完全兼容。
- 训练能跑通，动作损失正常下降；统计文件与 W&B 指标齐全。
- 可按配置切换 CoT 阶段（`scheduled_stage`）与 Dropout（`reasoning_dropout_prob`）。
- （可选）启用 `vlm_cot` 后总损失稳定，且推理文本监督有效。

---

### 备注
- 保持与现有 `QwenGR00T` 结构兼容，优先保证“动作监督”路径稳定；CoT 文本监督在第二阶段逐步开放。

---

## 非侵入式集成定义（重要）
- 不修改 `prismatic/` 源码，仅以只读方式 import 其 RLDS 管线与 ECoT 构造。
- 在 `starVLA` 内新增独立子模块，所有对接逻辑（数据、配置、日志、异常处理）均放置于该子模块内，便于开关与维护。
- 与现有 LeRobot 数据通道并存，互不影响；通过 YAML `dataset_py` 选择启用。

### 目录结构与模块职责（拟新增）
```
starVLA/integrations/ecot_rlds/
  __init__.py                 # 导出构建 API 与常量
  config.py                   # 默认常量、字段映射、超参数边界与校验
  dataset.py                  # 适配器数据集（RLDSDataset 包装、字段映射、窗口对齐、统计访问）
  collate.py                  # collate_fn（样例列表返回；保留LM损失扩展插口）
  builder.py                  # get_vla_dataset_ecot/make_dataloader_ecot 构建入口
  transforms.py               # 轻量图像/多视角处理（保持非侵入、仅做PIL/resize等）
  README.md                   # 使用说明、配方、常见问题
  tests/
    smoke_test.py             # 冒烟：加载→取batch→前向
    dataset_contract_test.py  # 数据契约校验（字段/形状/类型）
```

### 公共 API 规范
- 构建数据集（不触碰外部状态）：
  - `get_vla_dataset_ecot(cfg) -> torch.utils.data.Dataset`
    - 仅解析 `cfg.datasets.vla_data` 与其子项 `ecot`；不依赖全局变量。
- 构建 DataLoader：
  - `make_dataloader_ecot(cfg) -> torch.utils.data.DataLoader`
    - 读取 `per_device_batch_size`、`num_workers`，使用 `collate_fn` 返回“样例列表”以兼容现有 `QwenGR00T.forward`。
- 统计输出：
  - `save_dataset_statistics(run_dir: Path, stats: dict) -> None`
    - 写出 `dataset_statistics.json`，包含 `action.q01/q99` 等键，保持与现有日志工具兼容。

### 配置 Schema（严格）
- 总入口：`datasets.vla_data.dataset_py: ecot_rlds`
- 复用字段：
  - `datasets.vla_data.image_size: [H, W]`
  - `datasets.vla_data.per_device_batch_size: int`
  - `framework.action_model.future_action_window_size: int`（与 RLDS 窗口一致，默认 15）
- 新增子配置：`datasets.vla_data.ecot.*`
  - `data_root_dir: str`（TFDS 数据根目录）
  - `data_mix: str`（OXE 预设混合名或单一数据名）
  - `scheduled_stage: int`（0=全量CoT；>0 逐步裁剪）
  - `reasoning_dropout_prob: float [0,1]`
  - `shuffle_buffer_size: int`（默认 256000）
  - `image_aug: bool`（默认 false）
  - `reasoning_json: str`（本地路径；不存在时按 prismatic 逻辑自动拉取）
  - `num_parallel_calls: int | 'autotune'`（可选，默认遵循 prismatic）
  - `num_parallel_reads: int | 'autotune'`（可选）
  - `train: bool`（默认 true）

### 数据样例契约（接口稳定性保障）
- 每个样例为 `dict`，最少包含：
  - `image`: `List[PIL.Image.Image]`（多视角保留列表；尺寸按 `image_size`）  
  - `lang`: `str`（来自 `language_instruction`，可被 `CoT_prompt` 模板化后输入 Qwen）  
  - `action`: `np.ndarray | torch.Tensor`，形状 `[chunk_len, action_dim]`  
    - 其中 `chunk_len = past_action_window_size + 1 + future_action_window_size`（默认 16）  
  - 可选：
    - `state`: `np.ndarray | torch.Tensor`（如 RLDS 提供 `proprio`）  
    - `reasoning`: `str`（阶段/Dropout 后的 CoT 字符串）  
- collate 后的 batch：`List[Dict]`（不做 tensor 堆叠，交由 `QwenGR00T.forward` 内部处理）。

### 依赖与安装建议
- 额外依赖（与 prismatic 一致）：`tensorflow`、`tensorflow_datasets`、`dlimp`、`huggingface_hub`。
- 建议提供独立清单：`requirements-ecot-rlds.txt`，避免影响默认环境。
- TF 与 Torch 并存注意：
  - Prismatic 已显式关闭 TF 的 GPU：`tf.config.set_visible_devices([], "GPU")`，无需额外修改。

### 性能优化与资源规划
- RLDS 管线关键参数：
  - `shuffle_buffer_size`：内存敏感；可从 256k 降到 64k/32k，根据机器内存与吞吐调优。
  - `num_parallel_calls/reads`：多核 CPU 上可提高，但需观察 I/O 饱和。
  - `image_aug: false`：首期关闭，训练稳定后再评估开启带来的收益与开销。
- DataLoader 侧：
  - `num_workers` 建议 ≥ CPU 物理核数/2；逐步调优找到最佳点。
  - 若 I/O 成瓶颈，考虑将图像 resize 前置到 RLDS/TFDS 侧（保持非侵入的前提下仅在适配层做 PIL resize）。

### 错误处理与降级策略
- 资源文件缺失（`reasoning_json`）：
  - 自动按 prismatic 逻辑从 HF 拉取；失败则打印可操作提示并允许仅动作监督继续（`reasoning` 字段置空）。
- 动作窗口不匹配：
  - 适配层做截断/填充，首个 batch 打印一次性告警，记录到日志。
- 字段缺失：
  - `proprio` 缺失时跳过 `state`；`reasoning` 缺失时仅做动作监督；均不应中断训练。

### 日志与监控
- 记录以下关键项到训练日志/W&B：
  - 数据通道：`dataset_py=ecot_rlds`、`data_mix`、`scheduled_stage`、`reasoning_dropout_prob`
  - 窗口参数：`future_action_window_size`、`chunk_len`、对齐策略（截断/填充）
  - 吞吐：`data_time`、`model_time`（已有）
  - （启用 LM Loss 时）`lm_loss`、`loss_scale.vlm_cot`
- 输出 `dataset_statistics.json` 于 `run_dir`。

### 测试与验证矩阵（建议）
- 功能冒烟：
  - 单数据集/单机 CPU：加载→取 1 个 batch→前向（无异常）
  - 多视角样例：`image` 列表长度符合预期
- 兼容性：
  - `future_action_window_size` ∈ {7, 15} 等常见设置
  - `image_size` ∈ {(224,224), (336,336)}
  - `scheduled_stage` ∈ {0, 3, len-1}
- 性能：
  - `shuffle_buffer_size` ∈ {32k, 64k, 256k}
  - `num_workers` ∈ {4, 8, 16}
- 稳定性：
  - 100~500 steps 小规模训练；loss 下降、无 OOM/死锁

### 兼容性与迁移
- 兼容现有 LeRobot 通道：通过 YAML 切换，无需改代码。
- 模型无感知：`QwenGR00T` 接口与损失路径不变；未来启用 LM Loss 仅需在训练端合并权重。

### 安全与合规（数据侧）
- 确认 RLDS 数据与 reasoning JSON 的许可与分发合规；如需远程拉取，记录来源与版本。
- 训练日志不泄漏敏感数据（如路径、私有仓库 URL）。

### 发布与回滚计划
- 发布：
  - 合并 `starVLA/integrations/ecot_rlds/` 子模块与 `build_dataloader` 新分支。
  - 提供示例 YAML 与使用 README。
- 回滚：
  - 将 `dataset_py` 改回 `lerobot_datasets` 即可；新增模块对核心路径无影响。

### 开放问题与后续路线
- 是否在数据侧前置统一 resize/颜色增广以降低运行时开销？（需评估收益/复杂度）
- CoT LM Loss 的启用时机与权重建议？（建议在动作收敛后逐步打开 0.05~0.2）
- `reasoning` 输入是否用于条件提示而非仅标签？（需在 Qwen Prompt 设计中谨慎处理，避免信息泄漏）

### 术语表（简要）
- RLDS：Robotics Learning Data Standard，TFDS 管线输出的机器人数据标准格式。
- OXE：Open X-Embodiment，跨形体的大规模机器人数据集合。
- ECoT：Embodied Chain-of-Thought，带隐式推理/思维标注的数据或方法。
- CoT：Chain-of-Thought，链式推理文本。


