Bridge Data V2 轨迹分段与多模态 CoT 标注方案 (v4.0)

版本: v4.1 (Geometric-Segmentation + VLM-Reasoning + GroundingDINO)
日期: 2023-11
目标: 为 LeRobot 格式数据生成包含时序分段、语义 CoT 及 关键物体 BBox 的高密度标注。

1. 核心设计哲学 (Design Philosophy)

本方案旨在通过“硬数学”、“软 AI”与“强视觉”的结合，解决机器人数据标注中的三个核心问题：

WHEN (切分点): 使用 微分几何 (Differential Geometry)。

痛点: VLM 对时间边界不敏感，且视频 token 消耗巨大。

解法: 利用轨迹的曲率 (Curvature) 和 速度 (Velocity) 极值，结合 夹爪 (Gripper) 状态，从物理层面精准定位动作切换点。

WHAT (语义描述): 使用 VLM (Vision-Language Model)。

痛点: 几何数据无法理解“抓的是苹果还是梨”。

解法: 将分段后的关键帧送入 VLM，结合任务指令生成自然语言 CoT 和 目标物体名称。

WHERE (视觉定位): 使用 Grounding DINO (Open-Set Detection)。

痛点: VLM 直接生成的坐标往往存在幻觉，精度不足。

解法: 采用 Pipeline 策略。VLM 负责识别“要找什么”（例如 "green sponge"），Grounding DINO 负责在图像中精准定位该物体。

数据根目录: /share/project/baishuanghao/data/bridge_orig_lerobot

2. 详细处理流水线 (Pipeline)

阶段一：几何运动学分段 (Step 1: Geometric Segmentation)

算力需求: CPU Only (极快)

利用 B-Spline 拟合轨迹，计算曲率与速度，寻找运动奇异点。

输入: observation.state (XYZ, Gripper)

算法逻辑:

拟合: 对 XYZ 轨迹进行 3 次 B-Spline 拟合，去除手抖噪声。

微分: 解析计算曲率 $\kappa(t)$ 和速度 $v(t)$。

检测: 寻找 $Score(t) = \text{Norm}(\kappa) \times (1 - \text{Norm}(v))$ 的峰值点。

融合: 将几何峰值与夹爪开合点 (Open $\leftrightarrow$ Close) 合并。

输出: 带有基础物理标签的 Segments。

move_to_object (Approach)

grasp_object (Action)

move_to_goal (Transport)

place_object (Action)

阶段二：语义推理与视觉定位 (Step 2: Reasoning & Grounding)

算力需求: GPU (VLM + Detection Model)

对每个 Segment 的关键帧进行多模态级联分析。

步骤 2a: 语义推理 (Semantic Reasoning)

模型: Qwen2-VL-7B-Instruct 或 GPT-4o

输入: 关键帧 + 指令 + 物理标签

任务:

生成思维链 (CoT)。

提取操作对象的具体名称 (Target Object Name)。

Prompt:

"Phase: {primitive_type}.

Briefly describe the action reasoning.

Crucial: Extract the specific name of the object being manipulated (e.g., 'green sponge'). Do not output coordinates."

步骤 2b: 视觉定位 (Visual Grounding)

模型: Grounding DINO (SOTA Open-Vocabulary Detector)

输入: 关键帧 + Object Name (来自步骤 2a)

任务: 检测该名称对应的物体 BBox。

逻辑:

输入 Prompt: "green sponge"

输出: Box [0.2, 0.4, 0.3, 0.5] + Confidence 0.85

过滤: 保留置信度最高的 Box。

3. 数据 Schema 定义

标注结果保存为 JSONL 文件，每行对应一个 Episode。
输出路径: /share/project/baishuanghao/data/bridge_orig_lerobot/annotations/segmentation_full_v1.jsonl

{
  "episode_index": 123,
  "segmentation_method": "geometric_kinematic_v1",
  "reasoning_model": "qwen3-vl-7b",
  "grounding_model": "grounding-dino-swin-t",
  "cycles": [
    {
      "cycle_id": 0,
      "segments": [
        {
          "segment_id": 0,
          "start_t": 0,
          "end_t": 18,
          // [Phase 1] 几何分析生成的物理标签
          "primitive_type": "move_to_object",
          
          // [Phase 2a] VLM 提取的物体名称
          "target_object_ref": "green sponge",
          
          // [Phase 2b] Grounding DINO 生成的 BBox
          "grounding": {
            "bbox_2d": [150, 200, 300, 350], // [ymin, xmin, ymax, xmax]
            "confidence": 0.88,
            "frame_idx": 18
          },
          
          // [Phase 2a] VLM 生成的思维链
          "segment_cot": "Approaching the green sponge. The robot moves the open gripper downwards to align with the sponge.",
          
          // [Debug] 几何辅助信息
          "geometry_debug": { "avg_curvature": 0.05, "avg_velocity": 0.12 }
        },
        // ...
      ]
    }
  ],
  // 稠密标签用于 Dataloader 快速读取
  "dense_labels": {
    "subtask_id": [0, 0, ..., 1, 1], 
    "active_bbox": [[150,200,300,350], ...] 
  }
}



4. 脚本开发规划

建议在 starVLA/data_gen 下拆分为两个独立的脚本。

脚本 A: bridge_geometric_segmentation.py (同前)

职责: 计算物理分段。输出 intermediate/geo_segments.jsonl。

脚本 B: bridge_vlm_grounding.py (更新)

职责: 级联推理。

读取: 加载 geo_segments.jsonl。

推理 VLM:

调用 Qwen3-VL

获取 segment_cot 和 target_object_ref。

推理 Grounding DINO:

加载 Grounding DINO 模型。

将 target_object_ref 作为 Text Prompt 输入。，

获取最高置信度的 BBox。

写入: 生成最终 segmentation_full_v1.jsonl。