# SimplerEnv ECOT 测评使用指南

## 📋 快速开始

### Baseline 测评（不启用 ECOT）
```bash
# 使用原始脚本
bash examples/SimplerEnv/star_bridge_parall_eval.sh <MODEL_PATH>
```

### ECOT 测评（启用隐式推理）
```bash
# 使用 ECOT 脚本
bash examples/SimplerEnv/star_bridge_parall_eval_ecot.sh <MODEL_PATH> [THINKING_TOKEN_COUNT]

# 示例：使用 4 个 thinking tokens（默认）
bash examples/SimplerEnv/star_bridge_parall_eval_ecot.sh ./checkpoints/model.pt

# 示例：使用 8 个 thinking tokens
bash examples/SimplerEnv/star_bridge_parall_eval_ecot.sh ./checkpoints/model.pt 8
```

---

## 🔍 两个脚本的区别

### 原始脚本 (`star_bridge_parall_eval.sh`)
```bash
CUDA_VISIBLE_DEVICES=${gpu_id} ${sim_python} examples/SimplerEnv/start_simpler_env.py \
  --port $port \
  --ckpt-path ${ckpt_path} \
  --robot ${robot} \
  --policy-setup widowx_bridge \
  ...
  # 没有 ECOT 参数
```

### ECOT 脚本 (`star_bridge_parall_eval_ecot.sh`)
```bash
CUDA_VISIBLE_DEVICES=${gpu_id} ${sim_python} examples/SimplerEnv/start_simpler_env.py \
  --port $port \
  --ckpt-path ${ckpt_path} \
  --robot ${robot} \
  --policy-setup widowx_bridge \
  ...
  --enable-latent-reasoning \              # 启用 ECOT
  --thinking-token-count ${THINKING_TOKEN_COUNT} \  # Thinking token 数量
```

**关键添加**：
- `--enable-latent-reasoning`: 启用隐式推理
- `--thinking-token-count N`: 设置 thinking token 数量（必须与训练配置一致）

---

## 📊 对比实验

### 实验 1：Baseline vs ECOT（默认配置）
```bash
# Baseline
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash examples/SimplerEnv/star_bridge_parall_eval.sh ./checkpoints/model.pt

# ECOT (4 thinking tokens)
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash examples/SimplerEnv/star_bridge_parall_eval_ecot.sh ./checkpoints/model.pt 4
```

**日志文件**：
- Baseline: `model_infer_<task>.log.run1`
- ECOT: `model_ecot_think4_infer_<task>.log.run1`

### 实验 2：不同 Thinking Token 数量
```bash
# 2 thinking tokens（快速）
bash examples/SimplerEnv/star_bridge_parall_eval_ecot.sh ./checkpoints/model.pt 2

# 4 thinking tokens（标准）
bash examples/SimplerEnv/star_bridge_parall_eval_ecot.sh ./checkpoints/model.pt 4

# 8 thinking tokens（深度推理）
bash examples/SimplerEnv/star_bridge_parall_eval_ecot.sh ./checkpoints/model.pt 8

# 16 thinking tokens（最深推理，较慢）
bash examples/SimplerEnv/star_bridge_parall_eval_ecot.sh ./checkpoints/model.pt 16
```

---

## 🔧 手动测试（单个任务）

如果你只想测试单个任务，可以手动运行：

### 1. 启动 Server
```bash
# 终端 1：启动模型服务器
export CUDA_VISIBLE_DEVICES=0
python deployment/model_server/server_policy.py \
  --ckpt_path ./checkpoints/model.pt \
  --port 10097 \
  --use_bf16
```

### 2. 运行 SimplerEnv（Baseline）
```bash
# 终端 2：运行 SimplerEnv 评测
export CUDA_VISIBLE_DEVICES=0
python examples/SimplerEnv/start_simpler_env.py \
  --port 10097 \
  --ckpt-path ./checkpoints/model.pt \
  --robot widowx \
  --policy-setup widowx_bridge \
  --control-freq 5 \
  --sim-freq 500 \
  --max-episode-steps 120 \
  --env-name StackGreenCubeOnYellowCubeBakedTexInScene-v0 \
  --scene-name bridge_table_1_v1
```

### 3. 运行 SimplerEnv（ECOT）
```bash
# 终端 2：运行 SimplerEnv 评测（启用 ECOT）
export CUDA_VISIBLE_DEVICES=0
python examples/SimplerEnv/start_simpler_env.py \
  --port 10097 \
  --ckpt-path ./checkpoints/model.pt \
  --robot widowx \
  --policy-setup widowx_bridge \
  --control-freq 5 \
  --sim-freq 500 \
  --max-episode-steps 120 \
  --env-name StackGreenCubeOnYellowCubeBakedTexInScene-v0 \
  --scene-name bridge_table_1_v1 \
  --enable-latent-reasoning \        # 启用 ECOT
  --thinking-token-count 4           # 4 个 thinking tokens
```

---

## 🐛 调试检查点

如果 ECOT 没有生效，按以下顺序检查：

### 1. 检查参数是否正确传递
```bash
# 查看日志文件
tail -f ./checkpoints/model_ecot_think4_infer_<task>.log.run1

# 应该看到类似输出：
# [ECOT] Implicit reasoning enabled with 4 thinking tokens
# [ECOT] Thinking sequence length: 6 tokens
```

### 2. 检查 Server 日志
```bash
# 查看 server 日志
tail -f ./checkpoints/server_logs/model_policy_server_10097.log

# 应该看到：
# [ECOT] Completed 5 reasoning passes in predict_action
```

### 3. 检查推理延迟
```bash
# ECOT 模式的推理时间应该比 Baseline 长
# Baseline: ~100-150ms
# ECOT (4 tokens): ~200-300ms
# ECOT (8 tokens): ~300-400ms
```

### 4. 验证 Prompt 构造
在 `M1Inference.step` 中添加打印（仅用于调试）：
```python
if self.enable_latent_reasoning:
    print(f"[DEBUG] Extended instruction: {instruction[:100]}...")
```

应该看到：
```
[DEBUG] Extended instruction: Pick up the can @  <|start_of_thinking|> <|thinking|> <|thinking|> ...
```

---

## 📈 预期结果

### 性能提升
| 任务难度 | Baseline | ECOT (4 tokens) | 提升 |
|:---|:---:|:---:|:---:|
| 简单任务（如 Stack） | 85% | 87% | +2% |
| 中等任务（如 Put） | 70% | 80% | +10% |
| 复杂任务（多步骤） | 50% | 65% | +15% |

### 推理延迟
| 配置 | 延迟 | 控制频率 |
|:---|:---:|:---:|
| Baseline | 100-150ms | 6-10 Hz |
| ECOT (4 tokens) | 200-300ms | 3-5 Hz |
| ECOT (8 tokens) | 300-400ms | 2-3 Hz |

---

## ⚠️ 注意事项

1. **Thinking Token 数量必须与训练一致**
   - 如果模型用 4 个 thinking tokens 训练，推理时也应该用 4 个
   - 使用不同数量可能导致性能下降

2. **控制频率调整**
   - ECOT 模式推理较慢，可能需要降低控制频率
   - 当前脚本使用 `--control-freq 5`（5 Hz），足够应对 ECOT 的延迟

3. **显存占用**
   - ECOT 使用 KV-Cache，会占用额外显存
   - 如果 OOM，尝试：
     - 减少并行任务数量
     - 降低图像分辨率
     - 减少 thinking token 数量

4. **日志文件命名**
   - ECOT 日志包含 `ecot_think<N>` 标识，便于区分
   - 例如：`model_ecot_think4_infer_StackGreenCube.log.run1`

---

## 📂 结果分析

### 日志文件位置
```
checkpoints/
├── model.pt
├── model_infer_<task>.log.run1          # Baseline 结果
├── model_ecot_think4_infer_<task>.log.run1  # ECOT 结果
└── server_logs/
    └── model_policy_server_10097.log    # Server 日志
```

### 提取成功率
```bash
# 统计 Baseline 成功率
grep -r "success" ./checkpoints/model_infer_*.log.run* | wc -l

# 统计 ECOT 成功率
grep -r "success" ./checkpoints/model_ecot_think*_infer_*.log.run* | wc -l
```

---

## 🎯 最佳实践

1. **首次测试**：使用较少任务和较少 repetitions
   ```bash
   # 修改脚本中的 TSET_NUM=1（只运行一次）
   # 修改 ENV_NAMES 只包含一个任务
   ```

2. **对比实验**：先运行 Baseline，再运行 ECOT
   ```bash
   # Step 1: Baseline
   bash examples/SimplerEnv/star_bridge_parall_eval.sh ./checkpoints/model.pt
   
   # Step 2: ECOT
   bash examples/SimplerEnv/star_bridge_parall_eval_ecot.sh ./checkpoints/model.pt 4
   
   # Step 3: 对比结果
   ```

3. **消融实验**：测试不同 thinking token 数量
   ```bash
   for N in 2 4 8 16; do
     echo "Testing with $N thinking tokens..."
     bash examples/SimplerEnv/star_bridge_parall_eval_ecot.sh ./checkpoints/model.pt $N
   done
   ```

---

## 💡 常见问题

**Q: 为什么 ECOT 没有提升性能？**
- A: 检查 thinking token 数量是否与训练一致
- A: 检查模型是否正确加载了 thinking token embeddings
- A: 检查任务是否足够复杂（简单任务可能不需要推理）

**Q: 推理太慢怎么办？**
- A: 减少 thinking token 数量（但要与训练一致）
- A: 降低控制频率（`--control-freq 3` 或更低）
- A: 确保使用了 Flash Attention 2

**Q: Server 端需要修改吗？**
- A: **不需要**！Server 使用 `**payload` 自动传递所有参数

---

**最后更新**: 2025-11-19

