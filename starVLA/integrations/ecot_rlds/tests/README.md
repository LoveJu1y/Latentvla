# ECOT RLDS 数据集测试

## 测试脚本

### `test_dataset_load.py`

用于验证 ECOT RLDS 数据集能否正常加载和迭代的冒烟测试。

#### 使用方法

**方式 1：直接运行（使用默认配置）**

测试脚本已内置默认配置，可以直接运行：

```bash
cd /share/project/lvjing/starVLA
python -m starVLA.integrations.ecot_rlds.tests.test_dataset_load
```

默认配置：
- `data_root_dir`: `/share/project/emllm_mnt.1d/mnt/sfs/baaiei/jyShi/rt_newData`
- `data_mix`: `bridge`
- `num_samples`: `3`

**方式 2：使用命令行参数（覆盖默认值）**

```bash
cd /share/project/lvjing/starVLA
python -m starVLA.integrations.ecot_rlds.tests.test_dataset_load \
    --data_root_dir /path/to/OXE_RLDS \
    --data_mix bridge \
    --num_samples 3
```

**方式 3：使用 YAML 配置文件**

```bash
python -m starVLA.integrations.ecot_rlds.tests.test_dataset_load \
    --config_yaml /path/to/config.yaml \
    --num_samples 3
```

#### 参数说明

- `--data_root_dir`: RLDS 数据根目录（必需，除非使用 `--config_yaml`）
- `--data_mix`: 数据集混合名称或单个数据集名称（必需，除非使用 `--config_yaml`）
- `--num_samples`: 要测试的样本数量（默认：3）
- `--config_yaml`: YAML 配置文件路径（可选，会覆盖其他参数）
- `--image_size`: 图像尺寸 [H W]（默认：224 224）
- `--action_dim`: 动作维度（默认：7）
- `--future_action_window_size`: 未来动作窗口大小（默认：15）

#### 测试内容

测试脚本会验证：

1. **数据集创建**：能否成功创建 `ECOTRLDSDataset` 实例
2. **数据集长度**：`len(dataset)` 是否有效
3. **数据集统计**：`dataset_statistics` 是否可用
4. **样本契约**：每个样本是否包含必需的字段：
   - `image`: `List[PIL.Image]`，至少一个图像
   - `lang`: `str`，非空字符串
   - `action`: `np.ndarray`，形状 `[chunk_len, action_dim]`
   - `reasoning`: `str`（可能为空，取决于 dropout）
   - `reasoning_subset`: `str`（如 `"[TH, ST, MV]"`）
   - `dataset_name`: `str`
   - `state`: `np.ndarray`，形状 `[state_dim]`（可选，如果 `load_proprio=True`）

#### 示例输出

```
2024-01-01 12:00:00 [INFO] Creating dataset...
2024-01-01 12:00:01 [INFO] [ECOT RLDS] Building dataset with mixture 'bridge'
2024-01-01 12:00:02 [INFO] [ECOT RLDS] Dataset initialized: length=12345
2024-01-01 12:00:02 [INFO] ✓ Dataset created: length=12345
2024-01-01 12:00:02 [INFO] ✓ Dataset statistics available: ['action', 'state']
2024-01-01 12:00:02 [INFO] ================================================================================
2024-01-01 12:00:02 [INFO] Testing dataset contract...
2024-01-01 12:00:02 [INFO] ================================================================================
2024-01-01 12:00:02 [INFO] 
--- Sample 1 ---
2024-01-01 12:00:03 [INFO] ✓ image: 1 views, size=(224, 224)
2024-01-01 12:00:03 [INFO] ✓ lang: pick up the cup...
2024-01-01 12:00:03 [INFO] ✓ action: shape=(16, 7), dtype=float32
2024-01-01 12:00:03 [INFO]   action range: [-1.234, 2.345]
2024-01-01 12:00:03 [INFO] ✓ reasoning: @TASK@ pick up the cup @ACTION@ move forward...
2024-01-01 12:00:03 [INFO] ✓ reasoning_subset: [TH, ST, MV]
2024-01-01 12:00:03 [INFO] ✓ dataset_name: bridge_dataset
2024-01-01 12:00:03 [INFO] ✓ state: shape=(7,), dtype=float32
...
2024-01-01 12:00:05 [INFO] ================================================================================
2024-01-01 12:00:05 [INFO] ✓ Successfully validated 3 samples
2024-01-01 12:00:05 [INFO] ================================================================================
2024-01-01 12:00:05 [INFO] ✅ All tests passed!
```

#### 故障排除

**问题 1：找不到 RLDS 数据**

```
FileNotFoundError: RLDS dataset not found at /path/to/OXE_RLDS
```

**解决方案**：
- 确认 `--data_root_dir` 路径正确
- 确认数据已下载并解压到指定目录

**问题 2：TensorFlow 相关错误**

```
ImportError: cannot import name 'make_interleaved_dataset'
```

**解决方案**：
- 确认已安装 `tensorflow`、`tensorflow_datasets`、`dlimp`
- 确认 `prismatic` 模块在 Python 路径中

**问题 3：动作窗口不匹配警告**

```
[WARNING] Action sequence length (8) < chunk_len (16), padding with zeros
```

**解决方案**：
- 检查 `future_action_window_size` 配置是否与模型配置一致
- 确保 RLDS 配置的 `future_action_window_size` 等于 `chunk_len - 1`

**问题 4：内存不足**

如果数据集很大，可以：
- 减少 `--num_samples` 参数
- 减小 `shuffle_buffer_size`（在配置中）
- 使用较小的数据集进行测试

#### 下一步

测试通过后，可以：
1. 在 `build_dataloader` 中接入 `ecot_rlds` 分支
2. 运行小规模训练验证端到端流程
3. 检查 W&B 指标和 loss 曲线

