"""
constants.py

Action Chunking核心常量定义

该文件定义了Action Chunking所需的核心常量，包括：
- Token相关常量（IGNORE_INDEX, ACTION_TOKEN_BEGIN_IDX等）
- Action Chunking配置（NUM_ACTIONS_CHUNK, ACTION_DIM）

这些常量将被数据处理、模型训练和推理过程使用。
"""

# === Llama 2 Token Constants ===
IGNORE_INDEX = -100  # PyTorch标准ignore index，用于loss计算时忽略某些位置
STOP_INDEX = 2  # '</s>' token id

# ACTION_TOKEN_BEGIN_IDX应该与ActionTokenizer中的action_token_begin_idx对齐
# 对于256-bin的ActionTokenizer，这个值通常是vocab_size - 257
# 假设vocab_size=32000（Llama-2），则为32000-257=31743
ACTION_TOKEN_BEGIN_IDX = 31743
ACTION_TOKEN_END_IDX = 31743 + 256 - 1

# === Bridge数据集 Action Chunking配置 ===
NUM_ACTIONS_CHUNK = 8  # 一次预测8步action（参考OFT论文，8步在Bridge上效果最好）
ACTION_DIM = 7  # 每步action的维度: [x, y, z, roll, pitch, yaw, gripper]

# === 说明 ===
# NUM_ACTIONS_CHUNK * ACTION_DIM = 56
# 这意味着我们需要56个action tokens作为占位符
# 每个action token在LLM中会产生一个hidden state
# Action Head会将这56个hidden states转换为8步action

# === 与原ECoT的区别 ===
# 原ECoT: 每次预测1步action (7维) → 使用ActionTokenizer离散化为7个tokens
# 新方案: 每次预测8步action (8×7=56维) → 使用56个占位符tokens + Action Head回归

