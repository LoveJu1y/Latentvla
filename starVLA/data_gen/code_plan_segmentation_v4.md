## Bridge Data V2 几何分段 + 多模态标注代码计划（v4.1）

本计划基于 `segmentation_plan.md` 中的 v4.1 方案，只描述代码结构与模块划分，不涉及具体实现细节。

目标：在 **不修改训练 pipeline** 的前提下，在 `starVLA/data_gen` 目录实现一套两阶段的离线标注脚本：

- 阶段一（CPU）：几何分段 `bridge_geometric_segmentation.py`
- 阶段二（GPU）：VLM 推理 + GroundingDINO `bridge_vlm_grounding.py`

最终生成：

- `/share/project/baishuanghao/data/bridge_orig_lerobot/annotations/segmentation_full_v1.jsonl`

---

## 1. 总体结构与中间文件

### 1.1 脚本与文件布局

- `starVLA/data_gen/bridge_geometric_segmentation.py`
  - 负责：几何分段（基于 state/XYZ + gripper 的物理子任务划分）。
  - 输出：`intermediate/geo_segments.jsonl`

- `starVLA/data_gen/bridge_vlm_grounding.py`
  - 负责：VLM + GroundingDINO，多模态 CoT 与 BBox 标注。
  - 输入：`intermediate/geo_segments.jsonl`
  - 输出：`annotations/segmentation_full_v1.jsonl`

- 其他辅助：
  - 复用 `segmentation_plan.md` 和 `README_segmentation.md` 作为文档说明。
  - 若需要，可在 `intermediate/` 下增加 debug 文件（例如几何统计）。

### 1.2 数据输入与路径

- 数据根目录（来自方案）：  
  `/share/project/baishuanghao/data/bridge_orig_lerobot`

- 关键 meta 文件：
  - `meta/info.json`：包含 `chunks_size`, `splits` 等。
  - `meta/tasks.jsonl`：`task_index -> task`，用于获取指令文本。
  - `meta/episodes.jsonl`：`episode_index -> task_index` 等映射关系。

- 轨迹数据（LeRobot 格式）：
  - `data/chunk-{chunk_index:03d}/episode_{episode_index:06d}.parquet`
  - 主要使用字段：
    - `observation.state`：`[T, 8]`，包含 `x, y, z, roll, pitch, yaw, pad, gripper`
    - `action`：`[T, 7]`，包含 `x, y, z, roll, pitch, yaw, gripper`

- 视频数据：
  - 路径模板从 `meta/info.json["video_path"]` 获取，用于从 mp4 提取关键帧。

---

## 2. 阶段一：bridge_geometric_segmentation.py（几何分段）

### 2.1 入口与配置

函数：`main()`

- 解析命令行参数：
  - `--bridge_root`：数据根目录，默认 `/share/project/baishuanghao/data/bridge_orig_lerobot`
  - `--split`：数据划分，默认 `train`
  - `--output`：几何分段输出文件路径，默认 `intermediate/geo_segments.jsonl`
  - 几何相关超参：
    - `--bspline_degree`（默认 3）
    - `--smooth_factor`（样条平滑系数）
    - `--th_open`, `--th_close`（gripper 二值化阈值）
    - `--score_window`（score 峰值 refinement 的窗口大小）
    - `--max_episodes`（调试用）

- 从 `meta/info.json` 读取：
  - `chunks_size`
  - `splits[split]` → episode index 范围（例如 `"0:53192"`）

- 遍历 episode index：
  - 对每个 episode 调用 `process_episode(...)` 得到结果字典；
  - 将结果序列化为 JSON 行写入 `geo_segments.jsonl`。

### 2.2 数据加载模块

函数：`load_episode(root: Path, episode_index: int, chunk_size: int) -> dict[str, np.ndarray]`

- 计算 chunk 索引：
  - `chunk_index = episode_index // chunk_size`
- 拼接 parquet 路径：
  - `data/chunk-{chunk_index:03d}/episode_{episode_index:06d}.parquet`
- 使用 `pyarrow.parquet.read_table` 读取：
  - 将 `observation.state` 和 `action` 列还原为 `[T, D]` 的 numpy 数组。
- 返回：
  - `{"state": state_array, "action": action_array}`

### 2.3 特征与几何量计算

函数：`derive_trajectory_features(state: np.ndarray, action: np.ndarray) -> dict`

- 提取：
  - `pos_t = state[:, 0:3]`（末端位置）
  - `z_t = state[:, 2]`（高度）
  - `gripper_raw`：
    - 优先使用 `action[:, -1]`，若方差极小，再 fallback 为 `state[:, -1]`
  - 简单速度：
    - `vel_simple = norm(pos_t[t] - pos_t[t-1])`

函数：`fit_bspline(pos: np.ndarray) -> (tck, u) | None`

- 若 `T` 太小（例如 `< 5`）或 `pos` 变化极小：
  - 返回 `None`，后续使用简化规则分段。
- 否则：
  - 使用 `scipy.interpolate.splprep` 拟合参数样条（3 次 B-spline）。

函数：`compute_curvature_and_velocity(tck, u) -> (kappa: np.ndarray, v: np.ndarray)`

- 使用样条的一阶导数 `r'(u)`、二阶导数 `r''(u)`：
  - 速度：`v(u) = ||r'(u)||`
  - 曲率：`κ(u) = ||r'(u) × r''(u)|| / ||r'(u)||^3`
- 输出长度为 `T` 的 `kappa` 与 `v` 序列。

若 B-spline 拟合失败：

- fallback 使用位置差分近似：
  - `v_t = vel_simple`
  - `kappa_t` 可以用方向变化/二阶差分等简化形式近似，或置为常数。

### 2.4 gripper 事件检测

函数：`binarize_gripper(g_raw: np.ndarray, th_open: float, th_close: float) -> np.ndarray`

- 对 `g_raw` 做平滑（例如移动平均）；
- 归一化到 `[0, 1]`；
- 使用双阈值：
  - `<= th_open` → 0（open）
  - `>= th_close` → 1（close）
  - 中间区域继承前一帧状态。

函数：`detect_gripper_edges(g_bin: np.ndarray) -> (t_close_list, t_open_list)`

- open→close（0→1）记作抓取事件 `t_close`；
- close→open（1→0）记作放置事件 `t_open`。

### 2.5 score 计算与事件 refinement

函数：`compute_score(kappa: np.ndarray, v: np.ndarray) -> np.ndarray`

- 对 κ 和 v 做归一化：
  - `κ_norm = (κ - min) / (max - min + eps)`
  - `v_norm = (v - min) / (max - min + eps)`
- 定义：
  - `score = κ_norm * (1 - v_norm)`

函数：`refine_event_by_score(score: np.ndarray, around_idx: int, window: int) -> int`

- 在 `[around_idx - window, around_idx + window]` 内寻找 `score` 最大的位置；
- 返回该 index 作为 refined 事件时间。

在 `process_episode()` 中：

- 对每个 `t_close` 及其后最近的 `t_open`：
  - 调用 `refine_event_by_score` 微调：
    - `t_close_refined`, `t_open_refined`

### 2.6 周期与 segment 构造

函数：`build_segments_for_cycle(T: int, t_close: int, t_open: int, vel: np.ndarray, z: np.ndarray, score: np.ndarray, ...) -> list[segment_dict]`

- 输入：
  - 轨迹长度 `T`
  - 精炼后的 `t_close`, `t_open`
  - 辅助量 `vel`, `z`, `score`

- 输出一个周期内的 4 个物理 primitive 段：
  - `move_to_object`
  - `grasp_object`
  - `move_to_goal`
  - `place_object`

- 边界构造：
  - 以 `t_close`、`t_open` 为 anchor，结合窗口大小，构造粗略区间；
  - 根据速度/高度/score 对抓取、放置段的中心进行微调和收缩；
  - 确保段之间连续、不重叠，顺序固定；
  - 若某段太短（例如 `< 2` 帧），合并到前一段或后一段。

- `geometry_debug`：
  - 为每个 segment 计算并保存：
    - `avg_curvature`
    - `avg_velocity`

函数：`build_dense_labels(T: int, cycles: list) -> dict`

- 初始化长度为 `T` 的 `subtask_id`、`cycle_id`：
  - 对每个 cycle、每个 segment：
    - 使用全局映射表将 `primitive_type` 映射到整数 id；
    - 对 `[start_t, end_t]` 范围内的帧填入对应 id 和 `cycle_id`。
- 返回：
  - `{"subtask_id": [...], "cycle_id": [...]}`  
  - 此阶段不填 bbox（由第二阶段生成）。

### 2.7 单 episode 处理与输出

函数：`process_episode(episode_index: int, root: Path, chunk_size: int, ...) -> dict`

- 步骤：
  1. 调用 `load_episode` 读取 state/action。
  2. 调用 `derive_trajectory_features` 得到 `pos`, `z`, `gripper_raw` 等。
  3. 若 B-spline 条件满足：
     - `fit_bspline` → `compute_curvature_and_velocity`；
     - 否则走简化速度/曲率方案。
  4. `binarize_gripper` → `detect_gripper_edges`。
  5. 对每个 `t_close` 及对应 `t_open`：
     - 用 `compute_score` + `refine_event_by_score` 微调事件位置；
     - 调用 `build_segments_for_cycle` 生成该周期 segments。
  6. 调用 `build_dense_labels` 生成 `subtask_id` 等稠密标签。
  7. 设置 `segmentation_quality`：
     - 若成功找到至少一个周期并生成 segments，则为 `"high"`；
     - 否则为 `"low"`。
  8. 组装输出 dict：
     - `episode_index`
     - `segmentation_method`（例如 `"geometric_kinematic_v1"`）
     - `cycles`（仅包含几何字段：`primitive_type`, `start_t`, `end_t`, `geometry_debug`）
     - `dense_labels.subtask_id`, `dense_labels.cycle_id`

在 `main()` 中：

- 逐 episode 调用 `process_episode`，将结果写入 `intermediate/geo_segments.jsonl`。

---

## 3. 阶段二：bridge_vlm_grounding.py（VLM + GroundingDINO）

### 3.1 入口与配置

函数：`main()`

- 命令行参数：
  - `--bridge_root`：数据根目录
  - `--geo_segments`：几何分段输入文件，默认 `intermediate/geo_segments.jsonl`
  - `--output`：最终标注文件，默认 `annotations/segmentation_full_v1.jsonl`
  - VLM 相关配置：
    - 模型类型（本地 Qwen2-VL/Qwen3-VL 或 API）
    - checkpoint/host 地址
  - GroundingDINO 相关配置：
    - 模型权重路径
    - detection 阈值等
  - `--max_episodes`：调试时限制处理 episode 数量

- 初始化：
  - 加载 VLM 模型；
  - 加载 GroundingDINO 模型。

- 循环读取 `geo_segments.jsonl` 中的 episode 记录：
  - 调用 `annotate_episode_with_vlm_grounding(...)`；
  - 将结果写入 `segmentation_full_v1.jsonl`。

### 3.2 任务指令与图像读取

函数：`load_task_instruction(root: Path, episode_index: int) -> str`

- 从 `meta/episodes.jsonl` 中获取该 `episode_index` 对应的 `task_index`；
- 再从 `meta/tasks.jsonl` 中获取对应 `task` 文本；
- 若任务为空字符串或缺失，可填 `"unknown task"` 或跳过该 episode。

函数：`load_frame_image(root: Path, episode_index: int, frame_idx: int, camera_key: str = "observation.images.image_0") -> PIL.Image`

- 从 `meta/info.json` 的 `video_path` 模板生成 mp4 路径；
- 使用视频解码库按 `frame_idx` 读取单帧图像；
- 返回 `PIL.Image` 对象。

### 3.3 关键帧选择

函数：`select_keyframe_for_segment(segment: dict, strategy: str = "default") -> int`

- 默认策略可以先简单实现为使用 segment 末帧 (`end_t`)；
- 后续可按 primitive 类型细化：
  - `move_to_object`：末帧；
  - `grasp_object`：段中心帧；
  - `move_to_goal`：中间或接近目标处的一帧；
  - `place_object`：末帧。

### 3.4 VLM 语义推理

函数：`run_vlm_reasoning(image: Image, instruction: str, primitive_type: str) -> dict`

- 构造 prompt（遵循 `segmentation_plan.md` 中的说明）：
  - 包含：
    - 当前 Phase / primitive_type；
    - 任务指令 `instruction`；
    - 要求：
      - 简短思维链描述当前阶段的动作；
      - 明确抽取被操作物体的具体名称（如 `"green sponge"`），不输出坐标。

- 调用 VLM 模型，解析输出：
  - `segment_cot: str`
  - `target_object_ref: str`（简短物体名）

- 若 VLM 输出中未能可靠抽出物体名称：
  - 设 `target_object_ref = "unknown"` 或留空，并保留 COT。

### 3.5 GroundingDINO 视觉定位

函数：`run_grounding_dino(image: Image, object_ref: str) -> dict | None`

- 将 `object_ref` 作为 GroundingDINO 的 text prompt；
- 得到若干候选框及其置信度；
- 选择最高置信度框，构造：
  - `bbox_2d: [ymin, xmin, ymax, xmax]`（像素坐标）
  - `confidence: float`

- 若置信度低于阈值或无候选：
  - 返回 `None`，表示本段 Grounding 失败。

### 3.6 Segment 多模态标注融合

函数：`annotate_segment_with_vlm_and_grounding(segment: dict, episode_index: int, root: Path, instruction: str, vlm_model, grounding_model) -> dict`

- 步骤：
  1. 使用 `select_keyframe_for_segment` 获取 `frame_idx`；
  2. 利用 `load_frame_image` 读取该帧图像；
  3. 调用 `run_vlm_reasoning`：
     - 获得 `segment_cot`, `target_object_ref`；
  4. 调用 `run_grounding_dino`（若 `target_object_ref` 非空）：
     - 获得 `bbox_2d`, `confidence`；
  5. 在原 segment dict 上新增字段：
     - `"target_object_ref"`
     - `"grounding": { "bbox_2d": ..., "confidence": ..., "frame_idx": ... }`
     - `"segment_cot"`

- 对 Grounding 失败的情况：
  - `grounding` 可以为 `null` 或只包含 `frame_idx` 与 `confidence=0.0`。

### 3.7 稠密 BBox 生成

函数：`build_dense_active_bbox(num_steps: int, cycles: list, default_value=None) -> list`

- 初始化 `active_bbox` 长度为 `num_steps`，全部填 `default_value`（例如 `null`）。

- 遍历每一个 segment：
  - 若 `grounding.bbox_2d` 存在且可靠：
    - 对该 segment 对应的 `[start_t, end_t]` 范围：
      - 可以直接复制同一 bbox；
      - 或者根据需要做轻量插值。
  - 若不存在 bbox，则保持 `default_value`。

- 返回 `active_bbox` 列表，用于填入 `dense_labels.active_bbox`。

### 3.8 单 episode 的整体处理

函数：`annotate_episode_with_vlm_grounding(geo_episode: dict, root: Path, vlm_model, grounding_model) -> dict`

- 输入：
  - 来自 `geo_segments.jsonl` 的一条记录：
    - `episode_index`
    - `segmentation_method`
    - `cycles`（含几何分段信息）
    - `dense_labels.subtask_id`, `dense_labels.cycle_id`

- 步骤：
  1. 读取任务指令：`instruction = load_task_instruction(root, episode_index)`。
  2. 遍历 `cycles` 与其中的 `segments`：
     - 对每个 segment 调用 `annotate_segment_with_vlm_and_grounding`。
  3. 根据更新后的 `cycles` 调用 `build_dense_active_bbox`：
     - 得到 `dense_labels.active_bbox`。
  4. 构造最终 episode 级输出字典：
     - 保留原有的几何字段：
       - `episode_index`
       - `segmentation_method`
       - `cycles`（现在包含 `primitive_type` + `segment_cot` + `target_object_ref` + `grounding`）
       - `dense_labels.subtask_id`, `dense_labels.cycle_id`
     - 新增全局字段：
       - `reasoning_model`：如 `"qwen2-vl-7b"`
       - `grounding_model`：如 `"grounding-dino-swin-t"`
       - `dense_labels.active_bbox`

在 `main()` 中：

- 逐行读取 `geo_segments.jsonl`，调用上述函数；
- 将结果写入 `annotations/segmentation_full_v1.jsonl`。

---

## 4. 扩展与注意事项

- **鲁棒性**
  - B-spline 失败或轨迹过短时，要自动退化为简单的基于 gripper + 速度的分段。
  - VLM 无法可靠抽取物体名时，将 `target_object_ref` 记为 `"unknown"` 并跳过 Grounding。
  - GroundingDINO 检测不到时，对应段的 bbox 允许为 `null`。

- **子任务映射表**
  - 建议在一个统一模块/常量中定义：
    - `primitive_type -> subtask_id` 的映射表；
  - 确保几何分段和后续训练使用同一套编码。

- **性能与调试**
  - 阶段一（几何）可一次性跑完所有 episodes。
  - 阶段二可先用 `--max_episodes` 只处理少数 episodes 做可视化检查，再全量跑。

以上即为基于 v4.1 设计的完整代码结构与模块划分计划，后续具体实现时可严格按本文件中的函数划分与输入输出约定进行。 

