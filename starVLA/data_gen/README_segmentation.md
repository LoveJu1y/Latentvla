## BRIDGE-LeRobot 轨迹分段脚本说明

本目录下的代码仅用于**离线标注**，不会修改训练代码。

当前包含：

- `segmentation_plan.md`：子任务类型与分段方案设计文档。
- `bridge_segmentation.py`：基于 state/action 的规则分段脚本。

---

### 1. 依赖与数据路径

- 依赖：
  - `pyarrow`（用于读取 parquet）
- 默认数据路径：
  - `/share/project/baishuanghao/data/bridge_orig_lerobot`
  - 该目录下应包含：
    - `data/chunk-*/episode_*.parquet`
    - `meta/info.json`
    - `meta/modality.json`
    - `meta/episodes.jsonl` / `meta/tasks.jsonl`（可选，用于后续 COT）

---

### 2. 子任务类型

脚本中使用的子任务类型详见 `segmentation_plan.md`，核心为：

- `move_to_object`
- `grasp_object`
- `move_to_goal`
- `place_object`
- `retract`

内部使用 `SEGMENT_TYPES` 列表将它们映射为整数 `subtask_id`（按列表顺序）。

---

### 3. 脚本功能：`bridge_segmentation.py`

主要流程：

1. 从 `info.json` 读取：
   - `chunks_size`：每个 chunk 中包含多少 episodes。
   - `splits`：如 `"train": "0:53192"`。
2. 对指定 split 中的每个 `episode_index`：
   - 从 `data/chunk-xxx/episode_xxxxxx.parquet` 读取该 episode。
   - 解析 `observation.state` 和 `action`：
     - 位置 `x,y,z`、高度 `z`、velocity（通过位置差分）；
     - gripper 信号（优先用 action 的最后一维）。
   - 平滑并二值化 gripper，将其离散为 open/close；
   - 检测 gripper 的 open→close（抓取）和 close→open（放置）事件；
   - 对每一对 `(t_close, t_open)` 调用基于规则的分段逻辑，得到：
     - `move_to_object` / `grasp_object` / `move_to_goal` / `place_object` / `retract` 子段；
   - 构建 `per_step_labels`：
     - `subtask_id[t]`：当前时间步所属子任务；
     - `cycle_id[t]`：当前时间步所属的 pick-and-place 周期。
3. 将每个 episode 的分段结果写入一个 JSONL 文件。

---

### 4. 运行方式

在仓库根目录或本目录下运行：

```bash
python -m starVLA.data_gen.bridge_segmentation \
  --bridge_root /share/project/baishuanghao/data/bridge_orig_lerobot \
  --split train \
  --output /share/project/baishuanghao/data/bridge_orig_lerobot/annotations/segmentation.jsonl
```

可选参数：

- `--th_open`：
  - gripper 离散化为「open」的阈值，默认 `0.3`。
- `--th_close`：
  - gripper 离散化为「close」的阈值，默认 `0.7`。
- `--window`：
  - 在 gripper 事件附近构造 `grasp` / `place` 段的半窗口大小，默认 `2`。
- `--max_episodes`：
  - 仅处理前 N 条 episode，用于调试；默认 `-1` 表示处理 split 中全部 episodes。

---

### 5. 输出格式

输出文件为 JSONL，每行对应一个 episode，例如：

```json
{
  "dataset_root": "/share/project/baishuanghao/data/bridge_orig_lerobot",
  "episode_index": 123,
  "num_steps": 75,
  "segmentation_quality": "high",
  "cycles": [
    {
      "cycle_id": 0,
      "segments": [
        {
          "segment_id": 0,
          "segment_type": "move_to_object",
          "start_t": 0,
          "end_t": 18
        },
        {
          "segment_id": 1,
          "segment_type": "grasp_object",
          "start_t": 19,
          "end_t": 25
        }
      ]
    }
  ],
  "per_step_labels": {
    "subtask_id": [0, 0, 0, ...],
    "cycle_id":   [0, 0, 0, ...]
  }
}
```

- `segmentation_quality`：
  - `"high"`：检测到至少一个完整周期，并成功分段；
  - `"low"`：未检测到成对的 gripper 事件或分段不可靠。
- `per_step_labels.subtask_id`：
  - 值为 `SEGMENT_TYPES` 中的索引，从 `0` 开始。

---

### 6. 与后续 COT / bbox 标注的接口

- 本脚本只负责**基于 state/action 的子任务分段**；
- 后续可以在本目录新增：
  - `bridge_cot_generation.py`：读取 `segmentation.jsonl` + `tasks.jsonl`，为每个 segment 生成文本 COT；
  - bbox 标注脚本：在关键帧上调用 SAM / GroundingDINO，仅框目标物体，并可将结果对齐到本脚本的时间步标注。

