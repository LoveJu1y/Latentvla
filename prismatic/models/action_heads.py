"""
action_heads.py

Action Head实现，用于从LLM hidden states回归连续actions。

该模块提供了基于MLP的Action Head，作为传统VLM token prediction的替代方案。
核心思想是：从LLM的hidden states中直接回归出连续的action值，而不是预测离散的action tokens。

主要组件：
- MLPResNetBlock: MLP残差块，使用Pre-LayerNorm和残差连接
- MLPResNet: 堆叠多个MLPResNetBlock的网络
- L1RegressionActionHead: 使用L1 loss回归连续actions的Action Head

参考：OpenVLA-OFT (https://arxiv.org/abs/2502.19645)
"""

import torch
import torch.nn as nn
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK


class MLPResNetBlock(nn.Module):
    """
    MLP残差块。
    
    使用Pre-LayerNorm风格（类似Transformer），即先LayerNorm再FFN，然后加残差连接。
    这种结构在训练深层网络时更稳定。
    
    参考：https://arxiv.org/pdf/2002.04745.pdf (On Layer Normalization in the Transformer Architecture)
    """
    
    def __init__(self, dim: int):
        """
        Args:
            dim: 隐藏层维度
        """
        super().__init__()
        self.dim = dim
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：x + FFN(x)
        
        Args:
            x: (batch_size, hidden_dim)
        
        Returns:
            output: (batch_size, hidden_dim)
        """
        identity = x
        x = self.ffn(x)
        x = x + identity
        return x


class MLPResNet(nn.Module):
    """
    带残差连接的MLP网络。
    
    结构：
        LayerNorm → Linear → ReLU → [MLPResNetBlock × N] → LayerNorm → Linear
    
    这种结构在保持模型表达能力的同时，通过残差连接避免梯度消失问题。
    """
    
    def __init__(self, num_blocks: int, input_dim: int, hidden_dim: int, output_dim: int):
        """
        Args:
            num_blocks: MLPResNetBlock的数量
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
        """
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        
        # 堆叠多个残差块
        self.mlp_resnet_blocks = nn.ModuleList([
            MLPResNetBlock(dim=hidden_dim) for _ in range(num_blocks)
        ])
        
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: (batch_size, input_dim)
        
        Returns:
            output: (batch_size, output_dim)
        """
        x = self.layer_norm1(x)      # 输入归一化
        x = self.fc1(x)               # 投影到hidden_dim
        x = self.relu(x)              # 激活
        
        # 通过残差块
        for block in self.mlp_resnet_blocks:
            x = block(x)
        
        x = self.layer_norm2(x)       # 输出归一化
        x = self.fc2(x)               # 投影到output_dim
        return x


class L1RegressionActionHead(nn.Module):
    """
    基于L1回归的Action Head。
    
    核心思想：
    1. LLM生成的hidden states中，有一部分对应于action tokens的位置
    2. 这些hidden states包含了预测actions所需的信息
    3. 通过MLP直接回归出连续的action值，而不是预测离散的tokens
    
    工作流程：
        输入: (batch, 56, hidden_dim)  # 56 = NUM_ACTIONS_CHUNK * ACTION_DIM = 8 * 7
            ↓ reshape
        (batch, 8, 7*hidden_dim)  # 每8步action对应7个tokens的hidden states拼接
            ↓ MLPResNet
        输出: (batch, 8, 7)  # 8步action，每步7维
    
    优势：
    - 直接输出连续值，无需离散化和反离散化
    - 并行预测多步action，提高推理速度
    - L1 loss训练稳定，易于优化
    
    示例：
        >>> action_head = L1RegressionActionHead(input_dim=4096, hidden_dim=4096)
        >>> hidden_states = torch.randn(2, 56, 4096)
        >>> actions = action_head.predict_action(hidden_states)
        >>> actions.shape  # torch.Size([2, 8, 7])
    """
    
    def __init__(
        self, 
        input_dim: int = 4096, 
        hidden_dim: int = 4096, 
        action_dim: int = 7,
    ):
        """
        Args:
            input_dim: LLM hidden states的维度（通常是4096 for Llama-2 7B）
            hidden_dim: MLP的隐藏层维度
            action_dim: 单步action的维度（Bridge数据集为7）
        """
        super().__init__()
        self.action_dim = action_dim
        
        # MLP: 输入是action_dim个token的hidden states拼接
        # 例如：7个tokens，每个4096维 → 输入是28672维
        self.model = MLPResNet(
            num_blocks=2,
            input_dim=input_dim * ACTION_DIM,  # 7个token的hidden states拼接
            hidden_dim=hidden_dim,
            output_dim=action_dim,
        )
    
    def predict_action(self, actions_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        从action tokens的hidden states预测actions。
        
        Args:
            actions_hidden_states: (batch_size, chunk_len * action_dim, hidden_dim)
                                  = (batch_size, 56, hidden_dim)
                                  来自LLM最后一层，对应action tokens位置的hidden states
        
        Returns:
            action: (batch_size, chunk_len, action_dim) = (batch_size, 8, 7)
                   预测的action序列
        
        示例：
            >>> hidden_states = torch.randn(4, 56, 4096)  # batch=4
            >>> actions = action_head.predict_action(hidden_states)
            >>> actions.shape  # torch.Size([4, 8, 7])
        """
        batch_size = actions_hidden_states.shape[0]
        
        # Reshape: (batch, 56, hidden_dim) → (batch, 8, 7*hidden_dim)
        # 将每8步action对应的7个tokens的hidden states拼接起来
        rearranged_actions_hidden_states = actions_hidden_states.reshape(
            batch_size, NUM_ACTIONS_CHUNK, -1
        )
        # 形状: (batch, 8, 28672) 其中 28672 = 7 * 4096
        
        # 通过MLP预测action: (batch, 8, 28672) → (batch, 8, 7)
        action = self.model(rearranged_actions_hidden_states)
        
        return action

