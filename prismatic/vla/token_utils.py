"""
token_utils.py

Utilities for managing special tokens for Latent Reasoning (ECoT).

Note: Following OFT's approach, action tokens use the EXISTING vocabulary
(last 256 tokens) via ActionTokenizer. We do NOT add new action tokens.
Only latent reasoning tokens need to be added for ECoT's implicit reasoning.
"""

from typing import List
from transformers import PreTrainedTokenizerBase


def get_latent_reasoning_tokens() -> List[str]:
    """
    获取Latent Reasoning所需的thinking token
    
    Returns:
        List containing the thinking token: ["<|latent|>"]
    """
    return ["<|latent|>"]


def add_latent_tokens(
    tokenizer: PreTrainedTokenizerBase,
    verbose: bool = True,
) -> PreTrainedTokenizerBase:
    """
    添加Latent Reasoning所需的特殊token（ECoT隐式推理）
    
    Following OFT's approach: Action tokens use EXISTING vocabulary (last 256 tokens).
    ActionTokenizer automatically maps actions to these tokens (31743-31999).
    
    Only the latent token needs to be added for ECoT's implicit reasoning mechanism.
    
    Args:
        tokenizer: HuggingFace tokenizer
        verbose: 是否打印信息
    
    Returns:
        Modified tokenizer with latent token added
    
    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = add_latent_tokens(tokenizer)
        Added 1 special token to tokenizer
        Latent token: <|latent|> (ID: 32000)
        New vocabulary size: 32001
    """
    special_tokens = get_latent_reasoning_tokens()
    
    # 添加到tokenizer
    num_added = tokenizer.add_tokens(special_tokens)
    
    if verbose:
        latent_token_id = tokenizer.convert_tokens_to_ids("<|latent|>")
        print(f"Added {num_added} special token(s) to tokenizer")
        print(f"  - Latent token: <|latent|> (ID: {latent_token_id})")
        print(f"New vocabulary size: {len(tokenizer)}")
        print(f"\nNote: Action tokens use existing vocabulary (IDs 31743-31999),")
        print(f"      managed by ActionTokenizer. No need to add new action tokens.")
    
    return tokenizer


def verify_latent_token(tokenizer: PreTrainedTokenizerBase, verbose: bool = True) -> bool:
    """
    验证latent token是否正确添加到tokenizer
    
    Args:
        tokenizer: HuggingFace tokenizer
        verbose: 是否打印详细信息
    
    Returns:
        True if latent token is present, False otherwise
    """
    latent_token = "<|latent|>"
    token_id = tokenizer.convert_tokens_to_ids(latent_token)
    
    if token_id == tokenizer.unk_token_id:
        if verbose:
            print(f"❌ Latent token '{latent_token}' not found in vocabulary")
        return False
    
    if verbose:
        print(f"✅ Latent token '{latent_token}' found (ID: {token_id})")
        print(f"Current vocabulary size: {len(tokenizer)}")
    
    return True


def get_latent_token_id(tokenizer: PreTrainedTokenizerBase) -> int:
    """
    获取latent reasoning token的token ID
    
    Args:
        tokenizer: HuggingFace tokenizer
    
    Returns:
        Token ID for <|latent|>
    """
    return tokenizer.convert_tokens_to_ids("<|latent|>")

