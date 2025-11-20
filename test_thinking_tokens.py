#!/usr/bin/env python3
"""
测试 Qwen3-VL Interface 的 thinking tokens 添加功能
"""
import os
import sys
from pathlib import Path
from accelerate import PartialState
_ = PartialState()
# 设置 Hugging Face 国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 设置缓存目录到本地 ./ 目录
cache_dir = Path("./qwen").absolute()
cache_dir.mkdir(exist_ok=True)
os.environ["HF_HOME"] = str(cache_dir)
os.environ["TRANSFORMERS_CACHE"] = str(cache_dir / "transformers")
os.environ["HF_HUB_CACHE"] = str(cache_dir / "hub")

print(f"使用 Hugging Face 镜像: {os.environ['HF_ENDPOINT']}")
print(f"缓存目录: {cache_dir}")

# 导入必要的库
from omegaconf import OmegaConf, DictConfig
import torch

# 创建测试配置
test_config = {
    "framework": {
        "qwenvl": {
            "base_vlm": "Qwen/Qwen3-VL-2B-Instruct",
            "cache_dir": str(cache_dir)
        },
        "enable_latent_reasoning": True,
        "latent_reasoning": {
            "thinking_token": "<|thinking|>",
            "start_of_thinking_token": "<|start_of_thinking|>",
            "end_of_thinking_token": "<|end_of_thinking|>",
            "thinking_token_count": 2
        }
    },
    "datasets": {
        "vla_data": {}
    }
}

cfg = OmegaConf.create(test_config)

print("\n" + "="*60)
print("测试 1: 启用隐式推理模式")
print("="*60)

try:
    from starVLA.model.modules.vlm.QWen3 import _QWen3_VL_Interface
    
    print("\n正在初始化 Qwen3-VL Interface...")
    qwen_vl = _QWen3_VL_Interface(config=cfg)
    
    print("\n✅ 模型初始化成功!")
    
    # 检查 thinking token IDs
    print(f"\nThinking Token IDs:")
    print(f"  - thinking_token_id: {qwen_vl.thinking_token_id}")
    print(f"  - start_thinking_id: {qwen_vl.start_thinking_id}")
    print(f"  - end_thinking_id: {qwen_vl.end_thinking_id}")
    
    # 验证 tokenizer
    tokenizer = qwen_vl.processor.tokenizer
    print(f"\nTokenizer 词汇表大小: {len(tokenizer)}")
    
    # 测试 token 转换
    if qwen_vl.thinking_token_id is not None:
        thinking_token_str = tokenizer.convert_ids_to_tokens(qwen_vl.thinking_token_id)
        start_token_str = tokenizer.convert_ids_to_tokens(qwen_vl.start_thinking_id)
        end_token_str = tokenizer.convert_ids_to_tokens(qwen_vl.end_thinking_id)
        
        print(f"\nToken 字符串验证:")
        print(f"  - thinking_token: {thinking_token_str}")
        print(f"  - start_token: {start_token_str}")
        print(f"  - end_token: {end_token_str}")
        
        # 验证 embeddings
        embeddings = qwen_vl.model.get_input_embeddings()
        print(f"\nEmbedding 层大小: {embeddings.weight.shape[0]}")
        
        # 检查新 token 的 embeddings 是否已初始化
        thinking_emb = embeddings.weight.data[qwen_vl.thinking_token_id]
        start_emb = embeddings.weight.data[qwen_vl.start_thinking_id]
        end_emb = embeddings.weight.data[qwen_vl.end_thinking_id]
        
        print(f"\nEmbedding 检查:")
        print(f"  - thinking_token embedding shape: {thinking_emb.shape}")
        print(f"  - start_token embedding shape: {start_emb.shape}")
        print(f"  - end_token embedding shape: {end_emb.shape}")
        print(f"  - thinking_token embedding norm: {thinking_emb.norm().item():.4f}")
        print(f"  - start_token embedding norm: {start_emb.norm().item():.4f}")
        print(f"  - end_token embedding norm: {end_emb.norm().item():.4f}")
        
        # 验证 embeddings 不为零
        if thinking_emb.norm() > 0 and start_emb.norm() > 0 and end_emb.norm() > 0:
            print("\n✅ Thinking token embeddings 已正确初始化!")
        else:
            print("\n❌ 警告: 某些 thinking token embeddings 可能未正确初始化")
    
    print("\n" + "="*60)
    print("测试 2: 禁用隐式推理模式（向后兼容性）")
    print("="*60)
    
    # 测试禁用模式
    test_config_no_latent = test_config.copy()
    test_config_no_latent["framework"]["enable_latent_reasoning"] = False
    cfg_no_latent = OmegaConf.create(test_config_no_latent)
    
    print("\n正在初始化 Qwen3-VL Interface（禁用隐式推理）...")
    qwen_vl_no_latent = _QWen3_VL_Interface(config=cfg_no_latent)
    
    print(f"\nThinking Token IDs (应该为 None):")
    print(f"  - thinking_token_id: {qwen_vl_no_latent.thinking_token_id}")
    print(f"  - start_thinking_id: {qwen_vl_no_latent.start_thinking_id}")
    print(f"  - end_thinking_id: {qwen_vl_no_latent.end_thinking_id}")
    
    if (qwen_vl_no_latent.thinking_token_id is None and 
        qwen_vl_no_latent.start_thinking_id is None and 
        qwen_vl_no_latent.end_thinking_id is None):
        print("\n✅ 向后兼容性测试通过!")
    else:
        print("\n❌ 警告: 向后兼容性可能有问题")
    
    print("\n" + "="*60)
    print("所有测试完成!")
    print("="*60)
    
except Exception as e:
    print(f"\n❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

