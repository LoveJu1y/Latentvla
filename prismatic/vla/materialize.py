"""
materialize.py

Factory class for initializing Open-X RLDS-backed datasets, given specified data mixture parameters; provides and
exports individual functions for clear control flow.
"""

from pathlib import Path
from typing import Optional, Tuple, Type

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import PaddedCollatorForActionPrediction, PaddedCollatorForActionPredictionWithLatentAlignment
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import EpisodicRLDSDataset, RLDSBatchTransform, RLDSDataset


def get_vla_dataset_and_collator(
    data_root_dir: Path,
    data_mix: str,
    image_transform: ImageTransform,
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: Type[PromptBuilder],
    default_image_resolution: Tuple[int, int, int],
    padding_side: str = "right",
    predict_stop_token: bool = True,
    shuffle_buffer_size: int = 100_000,
    train: bool = True,
    episodic: bool = False,
    image_aug: bool = False,
    scheduled_stage: int = 0,
    cfg: Optional[object] = None,
) -> Tuple[Dataset, ActionTokenizer, PaddedCollatorForActionPrediction]:
    """Initialize RLDS Dataset (wraps TFDS), ActionTokenizer, and initialize transform/collation functions."""
    action_tokenizer = ActionTokenizer(tokenizer)
    batch_transform = RLDSBatchTransform(
        action_tokenizer, tokenizer, image_transform, prompt_builder_fn, predict_stop_token=predict_stop_token
    )
    
    # Choose collator based on whether latent reasoning is enabled
    if cfg is not None and hasattr(cfg, 'latent_reason') and cfg.latent_reason:
        # Use enhanced collator with thinking token alignment for latent reasoning
        latent_token_id = tokenizer.convert_tokens_to_ids(cfg.vla.thinking_token) if hasattr(cfg.vla, 'thinking_token') else None
        collator = PaddedCollatorForActionPredictionWithLatentAlignment(
            tokenizer.model_max_length, 
            tokenizer.pad_token_id, 
            padding_side=padding_side,
            latent_token_id=latent_token_id
        )
    else:
        # Use standard collator for regular training
        collator = PaddedCollatorForActionPrediction(
            tokenizer.model_max_length, tokenizer.pad_token_id, padding_side=padding_side
        )

    # Build RLDS Iterable Dataset
    cls = RLDSDataset if not episodic else EpisodicRLDSDataset
    dataset = cls(
        data_root_dir,
        data_mix,
        batch_transform,
        resize_resolution=default_image_resolution[1:],
        shuffle_buffer_size=shuffle_buffer_size,
        train=train,
        image_aug=image_aug,
        scheduled_stage=scheduled_stage,
        cfg=cfg,
    )

    return dataset, action_tokenizer, collator
