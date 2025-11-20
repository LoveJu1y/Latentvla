"""
train_utils.py

训练辅助函数，用于Action Chunking训练。

该模块提供了训练Action Head所需的核心工具函数：
1. extract_action_hidden_states: 从LLM输出中提取action tokens对应的hidden states
2. compute_l1_loss: 计算L1 regression loss
3. run_forward_pass: 执行模型forward和loss计算的核心函数（结合OFT + ECoT）

这些函数将在训练循环中被调用，用于：
- 从VLM的forward输出中提取必要的信息
- 计算Action Head的训练loss
- 支持ECoT的隐式推理和OFT的action head

参考：OpenVLA-OFT训练流程 + ECoT隐式推理
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any
from transformers.modeling_outputs import CausalLMOutputWithPast
from prismatic.vla.constants import ACTION_TOKEN_BEGIN_IDX, NUM_ACTIONS_CHUNK, ACTION_DIM, ACTION_TOKEN_END_IDX


def extract_action_hidden_states(
    hidden_states: torch.Tensor,
    num_action_tokens: int = NUM_ACTIONS_CHUNK * ACTION_DIM,
) -> torch.Tensor:
    """
    从LLM的hidden states中提取action tokens对应位置的状态。
    
    Following OFT's approach: Direct position-based slicing.
    Action tokens are always at the end of the sequence (before EOS).
    
    Args:
        hidden_states: (batch, seq_len, hidden_dim) - LLM最后一层的hidden states
        num_action_tokens: action tokens数量（默认56 = 8 actions × 7 dims）
    
    Returns:
        action_hidden_states: (batch, num_action_tokens, hidden_dim)
    
    Note:
        This assumes action tokens are always at the end of the sequence,
        which is guaranteed by our prompt construction:
        "USER: ... ASSISTANT: {reasoning} {action_tokens}"
        
        OFT reference: modeling_prismatic.py line 914-918
    """
    # Direct position-based slicing (OFT approach)
    # Extract the last num_action_tokens positions
    return hidden_states[:, -num_action_tokens-1:-1, :]


def compute_l1_loss(
    predicted_actions: torch.Tensor,
    ground_truth_actions: torch.Tensor,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    计算L1 loss（Mean Absolute Error）。
    
    L1 loss相比L2 loss的优势：
    - 对outliers不敏感（更robust）
    - 梯度更稳定（不会因为大误差而产生很大的梯度）
    - 在action预测任务中效果通常更好
    
    Args:
        predicted_actions: (batch, chunk_len, action_dim) - 预测的actions
                          例如: (4, 8, 7)
        ground_truth_actions: (batch, chunk_len, action_dim) - 真实的actions
                             例如: (4, 8, 7)
        mask: (batch, chunk_len) - 可选的mask，标记有效的actions
             例如: (4, 8)，值为True/False
             用于处理padding或无效的action steps
    
    Returns:
        loss: scalar tensor - L1 loss
    
    示例：
        >>> pred = torch.randn(4, 8, 7)
        >>> gt = torch.randn(4, 8, 7)
        >>> loss = compute_l1_loss(pred, gt)
        >>> loss.backward()  # 可以反向传播
        
        # 使用mask的情况（例如某些steps无效）
        >>> mask = torch.ones(4, 8, dtype=torch.bool)
        >>> mask[0, -2:] = False  # 第一个样本的最后2步无效
        >>> loss = compute_l1_loss(pred, gt, mask)
    """
    if mask is not None:
        # 扩展mask以匹配action的维度
        # (batch, chunk_len) → (batch, chunk_len, 1)
        mask_expanded = mask.unsqueeze(-1)
        
        # 只计算mask为True的位置的loss
        # 将mask为False的位置的值置为0
        masked_pred = predicted_actions * mask_expanded
        masked_gt = ground_truth_actions * mask_expanded
        
        # 计算总loss
        loss = F.l1_loss(masked_pred, masked_gt, reduction='sum')
        
        # 归一化：除以有效样本数
        num_valid = mask.sum()
        if num_valid > 0:
            loss = loss / num_valid
        else:
            # 如果没有有效样本，返回0（避免除0）
            loss = loss * 0  # 保持requires_grad
    else:
        # 计算所有位置的loss（默认情况）
        loss = F.l1_loss(predicted_actions, ground_truth_actions, reduction='mean')
    
    return loss


# === 未来可能需要的辅助函数 ===

def compute_mse_loss(
    predicted_actions: torch.Tensor,
    ground_truth_actions: torch.Tensor,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    计算MSE loss（Mean Squared Error）。
    
    保留此函数以便未来实验对比L1 vs L2 loss。
    目前我们主要使用L1 loss，但MSE在某些情况下可能更好。
    
    Args:
        predicted_actions: (batch, chunk_len, action_dim)
        ground_truth_actions: (batch, chunk_len, action_dim)
        mask: (batch, chunk_len) - 可选
    
    Returns:
        loss: scalar tensor
    """
    if mask is not None:
        mask_expanded = mask.unsqueeze(-1)
        masked_pred = predicted_actions * mask_expanded
        masked_gt = ground_truth_actions * mask_expanded
        
        loss = F.mse_loss(masked_pred, masked_gt, reduction='sum')
        num_valid = mask.sum()
        if num_valid > 0:
            loss = loss / num_valid
        else:
            loss = loss * 0
    else:
        loss = F.mse_loss(predicted_actions, ground_truth_actions, reduction='mean')
    
    return loss


# === 工具函数：用于调试和可视化 ===

def print_action_stats(actions: torch.Tensor, name: str = "Actions"):
    """
    打印action的统计信息（用于调试）。
    
    Args:
        actions: (batch, chunk_len, action_dim) 或 (chunk_len, action_dim)
        name: 标识符
    """
    print(f"\n=== {name} Statistics ===")
    print(f"Shape: {actions.shape}")
    print(f"Mean: {actions.mean().item():.4f}")
    print(f"Std: {actions.std().item():.4f}")
    print(f"Min: {actions.min().item():.4f}")
    print(f"Max: {actions.max().item():.4f}")
    print(f"Requires grad: {actions.requires_grad}")

