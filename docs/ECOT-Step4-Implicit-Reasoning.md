# 第4步：基于 last-hidden 的多次前向隐式推理（KV-Cache）+ 掩码损失（使用 @ 作为分界）

本文档规定 Step 4 的实现细节：图像通路保持不变（不手动处理 pixel_values），语言掩码与 ECoT 语义一致（Instruction 与 Latent 全遮蔽，只训练 Post-Latent），并实现基于 KV-Cache 的“last-hidden 反馈式”多次前向推理（不考虑降级方案）。

## 目标与边界
- 不改动图像通路：DataLoader → Qwen3-VL processor，`pixel_values` 完全由 processor 生成和管理。
- 语言掩码（ECoT 语义）：
  - Instruction 段：第一个分界符（本方案使用 `@`）之前的所有 token → IGNORE
  - Latent 段：`<|start_of_thinking|>` 到 `<|end_of_thinking|>` → IGNORE
  - Post-Latent 段：`<|end_of_thinking|>` 之后 → 参与 LM Loss
- 多次前向：对每个 latent token，取“该 token 前一位置”的最后隐状态，覆盖该 latent token 的嵌入，并推进到下一个 latent；使用 KV-Cache 加速短前向；完成全部 latent 后做一次全量前向。
- 本阶段仅实现 KV-Cache 版，不考虑降级（embedding hook additive delta）方案。

## 涉及文件
- `starVLA/model/modules/vlm/QWen3.py`
- `starVLA/model/framework/QwenGR00T.py`
- 不改动：`starVLA/integrations/ecot_rlds/dataset.py`（已将 instruction+reasoning 拼为 `lang`）

## 设计细节

### 1) pixel_values 处理
- 保持现状：两条路径（普通/对齐）都使用 `processor.apply_chat_template(...)` 返回的 `BatchFeature`（input_ids、attention_mask、pixel_values）。
- QWen3 接口内不手动改写/堆叠/设备迁移 `pixel_values`，只在返回前统一 `to(self.model.device)`。

### 2) span 定位与 label 掩码（QWen3）
- 新增 `find_ecot_spans(input_ids, start_id, think_id, end_id, use_at=True, at_ids=None)`：
  - 因本方案指定使用 `@` 作为分界，`use_at=True`，当 `<|start_of_thinking|>` 未定位到 instruction 终止位置时，使用 `@` 的首次出现作为 Instruction 结束边界；
  - 返回 per-sample 的 `(instruction_end_idx, latent_start_idx, latent_end_idx)`；
  - 注意 `@` 可能被切分为多个 token，应匹配 token 序列首位置。
- 新增 `build_ecot_labels(input_ids, pad_id, spans)`：
  - `labels = input_ids.clone()`；
  - 将 pad → IGNORE_INDEX；
  - `[0:instruction_end_idx)` 与 `[latent_start_idx:latent_end_idx)` → IGNORE_INDEX；
  - `[latent_end_idx:)` → 保留用于训练。
- 集成点：
  - 在 `build_qwenvl_inputs`（普通/对齐路径）末尾：若 `framework.latent_reasoning.compute_language_loss=True` 且未传入 `solutions`，则附加 `labels`。

### 3) 多次前向（KV-Cache + last-hidden 覆盖，QWen3）
实现与 prismatic 思路一致、参考你提供的伪代码（仅保留 KV-Cache 方案）：

1) 准备
- 从 tokenized batch 获取 `input_ids`、`attention_mask`、（如需）`position_ids = arange(T)`、`pixel_values`；
- 通过 `model.get_input_embeddings()` 获取文本嵌入层，`inputs_embeds = embedding(input_ids)`；
- `latent_indices = (input_ids == thinking_token_id).nonzero()` → (N_latent, 2)；
  - 按 batch 划成 `latent_lists = [[...] for i in B]`；
  - `max_n_latents = max(len(l) for l in latent_lists)`；

2) 初始范围与缓存
- 若 `max_n_latents == 0`：跳过循环，直接一次全量前向（带 labels 可得 `vlm_loss`）；
- 若存在 latent：
  - `next_compute_range = (0, earliest_latent_pos)`，其中 `earliest_latent_pos = latent_indices[:,1].min()`；
  - `kv_cache = None`；`hidden_states_offset = 0`；

3) 多次短前向 + 覆盖
```
for pass_idx in range(max_n_latents):
    if kv_cache is None:
        # 第一次短前向（无 cache）
        outputs = base_causallm(
            inputs_embeds = inputs_embeds[:, s:e, :],
            attention_mask = attention_mask[:, s:e],
            position_ids   = position_ids[:, s:e],
            output_hidden_states=True,
        )
        hidden_states_offset = 0
    else:
        # 使用 KV-Cache 的短前向
        past_key_values = [(k[:, :, :s, :], v[:, :, :s, :]) for (k,v) in kv_cache]
        outputs = base_causallm(
            inputs_embeds = inputs_embeds[:, s:e, :],
            attention_mask = attention_mask[:, :e],
            position_ids   = position_ids[:, s:e],
            past_key_values = past_key_values,
            output_hidden_states=True,
        )
        hidden_states_offset = s

    # 取最后一层 hidden states
    hs = outputs.hidden_states[-1]   # [B, L_sub, H]
    kv_cache = outputs.past_key_values

    # 反馈覆盖：对每个样本，取本 pass 的 latent 位置 token_idx
    # 将 hs[batch_idx, token_idx-1-hidden_states_offset, :] 覆盖到 inputs_embeds[batch_idx, token_idx, :]
    # 使用“分解-重组”避免 in-place

    # 推进范围：下一个范围 (e, T) 或 (e, e+1)，每个 pass 推进一个 latent
    next_compute_range = (e, T if pass_idx+1>=max_n_latents else e+1)
```

4) 最终一次全量前向（可带缓存）
- 构造前缀 `past_key_values`（若 `kv_cache` 不为 None）；
- 以最终的 `inputs_embeds`、完整 `attention_mask`、`position_ids` 与（可选）`labels` 调用一次全量前向，`output_hidden_states=True`，得到完整 logits/hidden_states 与（可选）masked LM Loss。

> 说明：上文中的 `base_causallm(...)` 为 Qwen3-VL 文本分支对应的 HF 接口（在我们封装的 VLM 中调用原生 forward 时需传入 vision 条件 `pixel_values`；在 QWen3 接口内将根据 HF 的多模态前向 API 进行适配，保证“短前向 + vision 条件 + cache”的契合）。

### 4) 在 QwenGR00T 中合并 VLM Loss
- 读取：
  - `framework.latent_reasoning.compute_language_loss`（bool）
  - `framework.latent_reasoning.vlm_loss_weight`（float，默认 0.1）
- 若 QWen3 返回的输入中附加了 `labels`，HF VLM 将返回 `.loss` → 记为 `vlm_loss`；
- 返回：
  - `{"action_loss": action_loss, "vlm_loss": (可选), "total_loss": action_loss + w * vlm_loss}`
  - 训练可选择以 `total_loss` 作为优化目标。

## 配置项（关键）
```yaml
framework:
  latent_reasoning:
    compute_language_loss: true        # 开启 masked LM Loss
    vlm_loss_weight: 0.1              # 与动作损失的权重
    use_at_delimiter: true            # 使用 '@' 作为 Instruction 分界回退
    at_delimiter_token: '@'
    strategy: inputs_embeds_kv        # 仅实现 KV-Cache + inputs_embeds 覆盖方案
    num_reasoning_passes: 1           # pass>1 时按 pass 级别重复覆盖（可后续拓展）
  qwenvl:
    model_max_length: 8192
```

## 测试
### 单元测试（QWen3）
- 构造含 `@<|start_of_thinking|>...<|end_of_thinking|>@` 的短样本，验证：
  - span 定位：`instruction_end_idx`、`latent_start_idx`、`latent_end_idx`；
  - 掩码：pad/Instruction/Latent=IGNORE，Post-Latent=非 IGNORE。
- 小 batch（B=2）验证：
  - 第一 latent 对齐仍然成立（来自 Step 3 的对齐）；
  - 多次短前向的切片范围、`hidden_states_offset` 与缓存使用正确。

### 集成测试（QwenGR00T）
- `compute_language_loss=True` 时：
  - VLM 返回 `vlm_loss` 且非 NaN；
  - 可以得到 `total_loss = action_loss + w * vlm_loss`；
- 验证 pixel 路径与 Step 2/3 完全一致，`pixel_values` 形状不变。

## 备注
- 本阶段严格使用 `@` 作为 Instruction 分界回退，优先 `<|start_of_thinking|>`/`<|end_of_thinking|>` 确定 Latent 段。
- 仅实现 KV-Cache 方案；若未来需要降级策略（embedding hook additive delta），可单独扩展，但不在本阶段范围内。


