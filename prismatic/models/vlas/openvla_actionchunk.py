"""
openvla_actionchunk.py

OpenVLA with Action Chunking support (Wrapper/Composition模式)

设计理念：
1. 组合模式：包含一个OpenVLA实例（而不是继承）
2. Wrapper模式：在OpenVLA基础上添加Action Head能力
3. 职责分离：VLA负责基础推理，Wrapper负责Action预测
4. 完整保留ECoT的隐式推理能力

优势：
- ✅ 灵活：可以传入任何OpenVLA实例
- ✅ 解耦：不依赖OpenVLA的内部实现
- ✅ 清晰：明确的HAS-A关系（Wrapper包含VLA）
- ✅ 易测试：可以mock VLA进行单元测试
- ✅ 符合设计原则：组合优于继承（GoF）
"""

import torch
import torch.nn as nn
from typing import Optional, Any
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.models.vlas.openvla import OpenVLA
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, ACTION_DIM
from prismatic.models.vlms.prismatic import PrismaticVLM

class OpenVLA_ActionChunk(nn.Module):
    """
    OpenVLA Action Chunking Wrapper (组合模式)
    
    使用组合模式包装OpenVLA，添加：
    1. Action Head（L1 regression）
    2. 双重loss计算（Token Loss + L1 Loss）
    3. 完整保留ECoT的隐式推理能力
    
    核心设计：
    - self.vla: 包含OpenVLA实例（组合）
    - self.action_head: Action Head作为类成员
    - forward: 调用self.vla.forward()，转发所有参数
    - 在forward内部计算L1 loss并合并
    - 可通过use_action_head参数切换模式
    
    与继承方案的对比：
    - 继承：class OpenVLA_ActionChunk(OpenVLA)  # IS-A
    - 组合：class OpenVLA_ActionChunk(nn.Module) + self.vla = vla  # HAS-A ✅
    """
    
    def __init__(
        self,
        vlm: PrismaticVLM,  #   核心：接收OpenVLA实例（组合）
        use_action_head: bool = True,  # 是否启用Action Head
        action_head_hidden_dim: int = None,  # Action Head隐藏层维度（默认=LLM hidden dim）
        l1_loss_weight: float = 1.0,  # L1 loss权重
        token_loss_weight: float = 1.0,  # Token loss权重
    ):
        """
        初始化 OpenVLA_ActionChunk Wrapper
        
        Args:
            vlm: PrismaticVLM实例（已加载的模型） 
            use_action_head: 是否启用Action Head
            action_head_hidden_dim: Action Head隐藏层维度
            l1_loss_weight: L1 loss的权重
            token_loss_weight: Token loss的权重
        """
        super().__init__()
        
        # === Step 1: 保存VLA实例（组合模式的核心）===
        self.vlm = vlm  #   OpenVLA作为成员变量
        
        # === Step 2: 配置参数 ===
        self.use_action_head = use_action_head
        self.l1_loss_weight = l1_loss_weight
        self.token_loss_weight = token_loss_weight
        
        # === Step 3: 创建Action Head ===
        if use_action_head:
            # 从VLA获取LLM的hidden dimension
            llm_hidden_dim = vlm.llm_backbone.embed_dim
            
            # 如果没有指定action_head_hidden_dim，使用LLM的hidden_dim
            if action_head_hidden_dim is None:
                action_head_hidden_dim = llm_hidden_dim
            
            # 创建Action Head
            self.action_head = L1RegressionActionHead(
                input_dim=llm_hidden_dim,
                hidden_dim=action_head_hidden_dim,
                action_dim=ACTION_DIM,
            )
            
            print(f"✅ Initialized Action Head with input_dim={llm_hidden_dim}, "
                  f"hidden_dim={action_head_hidden_dim}, action_dim={ACTION_DIM}")
        else:
            self.action_head = None
            print("ℹ️  Action Head disabled (use_action_head=False)")
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        actions: Optional[torch.FloatTensor] = None,  # 新增：ground truth actions
        position_ids: Optional[torch.LongTensor] = None,
        # === ECoT隐式推理参数（完整保留） ===
        scheduled_stage: int = 0,
        start_thinking_id: int = 0,
        end_thinking_id: int = 0,
        latent_token_id: int = 0,
        cfg: Optional[Any] = None,
        # === 其他参数 ===
        output_hidden_states: bool = None,  # 如果use_action_head=True，强制为True
        return_action_loss_only: bool = False,  # 是否只返回action loss（调试用）
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward with Action Head support
        
        核心设计：
        1. 调用 self.vlm.forward() 转发到内部VLA（完全保留ECoT逻辑） 
        2. 如果 use_action_head=True，额外计算 L1 loss
        3. 返回 combined loss
        
        Args:
            input_ids: 输入token IDs
            attention_mask: Attention mask
            pixel_values: 图像输入
            labels: Token prediction的标签（用于token loss）
            actions: Ground truth actions，shape (batch, 8, 7)（用于L1 loss）
            scheduled_stage: ECoT的训练阶段（0-8）
            start_thinking_id: Thinking token开始ID（ECoT）
            end_thinking_id: Thinking token结束ID（ECoT）
            latent_token_id: Latent token ID（ECoT）
            cfg: 配置对象（ECoT）
            output_hidden_states: 是否输出hidden states
            return_action_loss_only: 是否只返回action loss（调试用）
        
        Returns:
            CausalLMOutputWithPast: 包含loss、logits、hidden_states等
        """
        
        # === Step 1: 确定是否需要输出hidden states ===
        # 如果使用Action Head，必须输出hidden states
        need_hidden_states = output_hidden_states or (self.use_action_head and self.action_head is not None)
        
        # === Step 2: 调用VLA的forward（组合模式：转发调用）===
        output: CausalLMOutputWithPast = self.vlm.forward(  #   使用self.vla而不是super()
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            position_ids=position_ids,
            output_hidden_states=need_hidden_states,
            # ECoT隐式推理参数（完整传递）
            scheduled_stage=scheduled_stage,
            start_thinking_id=start_thinking_id,
            end_thinking_id=end_thinking_id,
            latent_token_id=latent_token_id,
            cfg=cfg,
            **kwargs,
        )
        
        # === Step 3: 如果不使用Action Head，直接返回VLA的输出 ===
        if not self.use_action_head or self.action_head is None:
            return output
        
        # === Step 4: 使用Action Head模式 - 计算L1 loss ===
        if actions is not None and need_hidden_states:
            # 4.1 提取hidden states
            last_hidden_states = output.hidden_states[-1]  # (batch, seq_len, hidden_dim)
            
            # 4.2 提取action tokens的hidden states
            # Action tokens总是在序列末尾（EOS之前）
            # 序列结构: [BOS] + [vision_patches] + [text_tokens] + [action_tokens] + [EOS]
            # ⭐ 简化：直接从末尾提取，跳过EOS（参考OFT: last_hidden_states[:, num_patches:-1]）
            num_action_tokens = NUM_ACTIONS_CHUNK * ACTION_DIM  # 56
            action_hidden_states = last_hidden_states[:, -(num_action_tokens+1):-1, :]
            # Shape: (batch, 56, hidden_dim)
            
            # 4.4 通过Action Head预测actions
            predicted_actions = self.action_head.predict_action(action_hidden_states)
            # Shape: (batch, 8, 7)
            
            # 4.5 计算L1 Loss
            l1_loss = torch.nn.functional.l1_loss(predicted_actions, actions.to('cuda'))
            
            # 4.6 合并Loss
            token_loss = output.loss if output.loss is not None else 0.0
            
            if return_action_loss_only:
                # 调试模式：只返回action loss
                combined_loss = l1_loss
            else:
                # 正常模式：合并token loss和L1 loss
                combined_loss = self.token_loss_weight * token_loss + self.l1_loss_weight * l1_loss
            
            # 4.7 更新output
            output.loss = combined_loss
            
            # 4.8 添加额外信息到output（用于logging）
            # 注意：CausalLMOutputWithPast没有这些字段，但我们可以添加自定义属性
            output.token_loss = token_loss
            output.l1_loss = l1_loss
            output.predicted_actions = predicted_actions
        
        return output
    
    def predict_action(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.FloatTensor,
        # ECoT参数（推理时可能需要）
        scheduled_stage: int = 8,  # 推理时通常使用最高stage
        latent_token_id: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        """
        推理模式：预测actions
        
        这是一个便捷方法，用于推理时只获取action预测，不计算loss。
        
        Args:
            input_ids: 输入token IDs
            attention_mask: Attention mask
            pixel_values: 图像输入
            scheduled_stage: ECoT的推理阶段（默认8，完全隐式推理）
            latent_token_id: Latent token ID
        
        Returns:
            predicted_actions: (batch, 8, 7)
        """
        if not self.use_action_head or self.action_head is None:
            raise RuntimeError("Action Head is not enabled! Set use_action_head=True during initialization.")
        
        # 调用forward，不传入labels和actions（推理模式）
        with torch.no_grad():
            output = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=None,
                actions=None,
                scheduled_stage=scheduled_stage,
                latent_token_id=latent_token_id,
                output_hidden_states=True,
                **kwargs,
            )
            
            # 提取action hidden states
            last_hidden_states = output.hidden_states[-1]
            
            # 直接从末尾提取action tokens，跳过EOS
            num_action_tokens = NUM_ACTIONS_CHUNK * ACTION_DIM
            action_hidden_states = last_hidden_states[:, -(num_action_tokens+1):-1, :]
            
            # 预测actions
            predicted_actions = self.action_head.predict_action(action_hidden_states)
        
        return predicted_actions
    
    # === 便捷方法：转发VLA的属性访问 ===
    
    @property
    def vision_backbone(self):
        """转发：访问VLA的vision_backbone"""
        return self.vlm.vision_backbone
    
    @property
    def llm_backbone(self):
        """转发：访问VLA的llm_backbone"""
        return self.vlm.llm_backbone
    
    def get_action_head_parameters(self):
        """获取Action Head的参数（用于optimizer配置）"""
        if self.action_head is not None:
            return self.action_head.parameters()
        else:
            return []
    
    def num_action_head_parameters(self) -> int:
        """获取Action Head的参数数量"""
        if self.action_head is not None:
            return sum(p.numel() for p in self.action_head.parameters())
        else:
            return 0

