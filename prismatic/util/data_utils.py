"""
data_utils.py

General utilities and classes for facilitating data loading and collation.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Sequence, Tuple, Optional

import torch
from torch.nn.utils.rnn import pad_sequence

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


def tree_map(fn: Callable, tree: dict) -> dict:
    """Maps a function over a nested dictionary."""
    return {k: tree_map(fn, v) if isinstance(v, dict) else fn(v) for k, v in tree.items()}


def tree_map_with_key(fn: Callable, tree: dict, keys: Sequence = ()) -> dict:
    """Maps a function over a nested dictionary."""
    return {
        k: tree_map_with_key(fn, v, (*keys, k)) if isinstance(v, dict) else fn((*keys, k), v) for k, v in tree.items()
    }


@dataclass
class PaddedCollatorForLanguageModeling:
    model_max_length: int
    pad_token_id: int
    default_image_resolution: Tuple[int, int, int]
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        self.dummy_pixel_values = torch.zeros(self.default_image_resolution, dtype=self.pixel_values_dtype)

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]

        # For now, we only support Tokenizers with `padding_side = "right"` during Training (but plan to extend!)
        #   => Handle padding via RNN Utils => `pad_sequence`
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # === Handle "unimodal" (language-only) vs. "multimodal" ===

        # Some examples are "language-only" --> build a Tensor of `multimodal_indices` that we can slice into easily
        multimodal_indices = torch.tensor(
            [idx for idx in range(len(pixel_values)) if pixel_values[idx] is not None], dtype=torch.long
        )

        # Stack all `pixel_values` --> depending on type (torch.Tensor, or Dict[str, torch.Tensor]) & presence of None
        if len(multimodal_indices) == 0:
            pixel_values = torch.stack([self.dummy_pixel_values for _ in range(len(input_ids))])
        elif isinstance(pv_example := pixel_values[multimodal_indices[0]], torch.Tensor):
            pixel_values = torch.stack(
                [
                    pixel_values[idx] if idx in multimodal_indices else self.dummy_pixel_values
                    for idx in range(len(input_ids))
                ]
            )
        elif isinstance(pv_example, dict):
            pixel_values = {
                k: torch.stack(
                    [
                        pixel_values[idx][k] if idx in multimodal_indices else self.dummy_pixel_values
                        for idx in range(len(input_ids))
                    ]
                )
                for k in pv_example
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            multimodal_indices=multimodal_indices,
        )


@dataclass
class PaddedCollatorForActionPrediction:
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]
        if "dataset_name" in instances[0]:
            dataset_names = [instance["dataset_name"] for instance in instances]
        else:
            dataset_names = None

        # For now, we only support Tokenizers with `padding_side = "right"` during training
        #   => Handle padding via RNN Utils => `pad_sequence`
        assert self.padding_side == "right", f"Invalid Tokenizer `{self.padding_side = }`"
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # [Contract] For VLA Training =>> No "Unimodal" Data!
        assert all([pv is not None for pv in pixel_values]), "Invalid VLA Example with `pixel_values = None`!"

        # Stack all `pixel_values` --> depending on type is torch.Tensor or Dict[str, torch.Tensor]
        if isinstance(pixel_values[0], torch.Tensor):
            pixel_values = torch.stack(pixel_values)
        elif isinstance(pixel_values[0], dict):
            pixel_values = {
                k: torch.stack([pixel_values[idx][k] for idx in range(len(input_ids))]) for k in pixel_values[0]
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        output = dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        
        # === Handle Actions (for Action Head if enabled) ===
        # Support for action chunking (ECoT + OFT): stack actions if present
        # Actions are numpy arrays from RLDS, need to convert to tensors
        if "actions" in instances[0]:
            actions_list = [
                torch.tensor(instance["actions"], dtype=torch.float32) 
                if not isinstance(instance["actions"], torch.Tensor) 
                else instance["actions"]
                for instance in instances
            ]
            output["actions"] = torch.stack(actions_list)  # Shape: (batch, 8, 7)
        
        return output


@dataclass
class PaddedCollatorForActionPredictionWithLatentAlignment:
    """
    Enhanced collator that aligns thinking token positions across batch samples for efficient latent reasoning training.
    
    This collator implements the key insight from the MyCollator example: by padding sequences
    to align thinking token starting positions, we can maintain batch processing efficiency
    while handling different thinking token positions across samples.
    """
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32
    latent_token_id: Optional[int] = None
    label_pad_token_id: int = IGNORE_INDEX

    def _create_bidirectional_action_mask(
        self,
        base_attention_mask: torch.Tensor,
        action_token_mask: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor:
        """
        Create a 4D attention mask that allows bidirectional attention among action tokens.
        
        Args:
            base_attention_mask: (batch, seq_len) - 1 for valid tokens, 0 for padding
            action_token_mask: (batch, seq_len) - True for action tokens
            batch_size: batch size
            seq_len: sequence length
        
        Returns:
            attention_mask: (batch, 1, seq_len, seq_len) - 0 for attend, -inf for mask
        """
        # Create causal mask (lower triangular)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.float32, device=base_attention_mask.device),
            diagonal=1
        )  # Upper triangle = 1 (will be masked)
        
        # Expand to batch dimension: (seq_len, seq_len) -> (batch, seq_len, seq_len)
        causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Find action token positions for each sample in the batch
        for batch_idx in range(batch_size):
            action_positions = action_token_mask[batch_idx].nonzero(as_tuple=False).squeeze(-1)
            
            if len(action_positions) > 0:
                # Allow bidirectional attention among action tokens
                # Set causal_mask[action_pos, action_pos] = 0 for all pairs
                action_grid = torch.meshgrid(action_positions, action_positions, indexing='ij')
                causal_mask[batch_idx, action_grid[0], action_grid[1]] = 0
        
        # Convert to attention mask format: 0 = attend, -inf = mask
        # causal_mask: 1 = mask, 0 = attend
        attention_mask = causal_mask.masked_fill(causal_mask == 1, float('-inf'))
        attention_mask = attention_mask.masked_fill(causal_mask == 0, 0.0)
        
        # Handle padding: set all attention from/to padding positions to -inf
        # base_attention_mask: (batch, seq_len) - 1 = valid, 0 = padding
        padding_mask = ~base_attention_mask.bool()  # (batch, seq_len)
        
        # Expand padding mask for attention matrix
        # From positions (rows)
        padding_mask_expanded_from = padding_mask.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq_len, seq_len)
        # To positions (cols)
        padding_mask_expanded_to = padding_mask.unsqueeze(2).expand(-1, -1, seq_len)  # (batch, seq_len, seq_len)
        
        # Mask out padding positions
        attention_mask = attention_mask.masked_fill(padding_mask_expanded_from | padding_mask_expanded_to, float('-inf'))
        
        # Add head dimension: (batch, seq_len, seq_len) -> (batch, 1, seq_len, seq_len)
        attention_mask = attention_mask.unsqueeze(1)
        
        return attention_mask
    
    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    # === 高效生成伪造的input_ids和labels（第30位是第一个thinking token）===
        # batch_size = len(instances)
        # first_latent_position = 100  # 所有样本的第一个latent token都在这个位置
        # num_latent_tokens = 2       # 每个样本有2个latent token
        # latent_token_id = self.latent_token_id if self.latent_token_id is not None else 32001
        
        # # 计算每个样本的长度
        # extra_lengths = 100 # [20, 35, 50, 65, ...]
        # total_lengths = first_latent_position + num_latent_tokens + extra_lengths
        # max_length = total_lengths
        
        # # 生成基础tensor：所有位置都是随机token
        # base_tokens = torch.randint(100, 1000, (batch_size, max_length), dtype=torch.long)
        
        # # 设置BOS token（位置0）
        # base_tokens[:, 0] = 1
        
        # # 设置thinking tokens（位置30-31）
        # base_tokens[:, first_latent_position:first_latent_position + num_latent_tokens] = latent_token_id
        
        # # 后面的token保持随机即可，不需要特殊处理
        
        # # 转换为列表格式，每个样本是一个1D tensor
        # input_ids_list = [base_tokens[i] for i in range(batch_size)]
        # labels_list = [base_tokens[i] for i in range(batch_size)]
        
        # # === 保持原有的pixel_values处理逻辑 ===
        # pixel_values = [instance["pixel_values"] for instance in instances]
        # if "dataset_name" in instances[0]:
        #     dataset_names = [instance["dataset_name"] for instance in instances]
        # else:
        #     dataset_names = None

        # # For now, we only support Tokenizers with `padding_side = "right"` during training
        # #   => Handle padding via RNN Utils => `pad_sequence`
        # assert self.padding_side == "right", f"Invalid Tokenizer `{self.padding_side = }`"
        # input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=self.pad_token_id)
        # labels = pad_sequence(labels_list, batch_first=True, padding_value=IGNORE_INDEX)

        # # Truncate (if necessary)
        # input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # # Get `attention_mask` by checking for `pad_token_id`
        # attention_mask = input_ids.ne(self.pad_token_id)

        # # [Contract] For VLA Training =>> No "Unimodal" Data!
        # assert all([pv is not None for pv in pixel_values]), "Invalid VLA Example with `pixel_values = None`!"

        # # Stack all `pixel_values` --> depending on type is torch.Tensor or Dict[str, torch.Tensor]
        # if isinstance(pixel_values[0], torch.Tensor):
        #     pixel_values = torch.stack(pixel_values)
        # elif isinstance(pixel_values[0], dict):
        #     pixel_values = {
        #         k: torch.stack([pixel_values[idx][k] for idx in range(len(input_ids))]) for k in pixel_values[0]
        #     }
        # else:
        #     raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # output = dict(
        #     pixel_values=pixel_values,
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     labels=labels,
        # )
        # if dataset_names is not None:
        #     output["dataset_names"] = dataset_names
        # return output
    #     """
    #     Collate instances with latent token alignment using pad_sequence for memory efficiency.
    #     """
        # === Step 0: Basic data extraction ===
        input_ids_list = [instance["input_ids"] for instance in instances]
        labels_list = [instance["labels"] for instance in instances]
        pixel_values = [instance["pixel_values"] for instance in instances]
        
        dataset_names = [instance.get("dataset_name") for instance in instances] if "dataset_name" in instances[0] else None
        
        # Check padding side consistency
        assert self.padding_side == "right", f"Invalid Tokenizer `{self.padding_side = }`"
        # print(f"self.latent_token_id = {self.latent_token_id}")
        # === Step 1: Calculate alignment padding ===
        if self.latent_token_id is not None:
            # Find the earliest thinking token position in each sample
            earliest_latent_positions = []
            for ids in input_ids_list:
                # More efficient: use torch operations instead of converting to list
                latent_mask = (ids == self.latent_token_id)
                if latent_mask.any():
                    earliest_latent_positions.append(latent_mask.nonzero()[0].item())
                else:
                    earliest_latent_positions.append(-1)

            valid_positions = [pos for pos in earliest_latent_positions if pos >= 0]
            if valid_positions:
                # Align to the latest position (correct alignment strategy)
                latest_latent_pos = max(valid_positions)
                
                # Check if alignment would exceed model_max_length
                max_original_length = max(len(ids) for ids in input_ids_list)
                if latest_latent_pos + max_original_length > self.model_max_length:
                    # If alignment would be too long, skip alignment and use regular padding
                    print(f"max_original_length = {max_original_length}")
                    print(f"latest_latent_pos = {latest_latent_pos}")
                    print(f"self.model_max_length = {self.model_max_length}")
                    print(f"Warning: Latent token alignment would exceed model_max_length ({self.model_max_length}). Using regular padding instead.")
                else:
                    # Apply pre-padding to align latent tokens
                    aligned_input_ids = []
                    aligned_labels = []
                    
                    for i, (input_ids, labels) in enumerate(zip(input_ids_list, labels_list)):
                        if earliest_latent_positions[i] != -1:
                            # Sample has latent token - align it
                            pad_count = latest_latent_pos - earliest_latent_positions[i]
                        else:
                            # Sample has no latent token - add padding to align with others
                            pad_count = latest_latent_pos
                            print(f"input_ids: {input_ids}")
                            print(f"labels: {labels}")
                            print(f"collator Warning: no latent token in sample {i}")
                        
                        if pad_count > 0:
                            # Add padding at the beginning
                            pad_tensor = torch.full((pad_count,), self.pad_token_id, dtype=input_ids.dtype)
                            label_pad_tensor = torch.full((pad_count,), self.label_pad_token_id, dtype=labels.dtype)
                            
                            aligned_input_ids.append(torch.cat([pad_tensor, input_ids]))
                            aligned_labels.append(torch.cat([label_pad_tensor, labels]))
                        else:
                            aligned_input_ids.append(input_ids)
                            aligned_labels.append(labels)
                    
                    input_ids_list = aligned_input_ids
                    labels_list = aligned_labels

        # === Step 2: Use pad_sequence for efficient padding ===
        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels_list, batch_first=True, padding_value=self.label_pad_token_id)

        # === Step 3: Truncate to model_max_length ===
        input_ids = input_ids[:, :self.model_max_length]
        labels = labels[:, :self.model_max_length]

        # === Step 3.5: Mask action tokens in labels (OFT-style) ===
        # Action tokens should not contribute to token loss, only to L1 loss via Action Head
        from prismatic.vla.constants import ACTION_TOKEN_BEGIN_IDX, ACTION_TOKEN_END_IDX
        action_token_mask = (input_ids >= ACTION_TOKEN_BEGIN_IDX) & (input_ids <= ACTION_TOKEN_END_IDX)
        labels[action_token_mask] = self.label_pad_token_id  # Set to IGNORE_INDEX (-100)
        
        # === Step 4: Get attention mask with bidirectional action attention ===
        # Create base attention mask (1 for valid tokens, 0 for padding)
        base_attention_mask = input_ids.ne(self.pad_token_id)  # (batch, seq_len)
        
        # Create 4D attention mask to allow bidirectional attention among action tokens
        # Shape: (batch, 1, seq_len, seq_len)
        batch_size, seq_len = input_ids.shape
        attention_mask = self._create_bidirectional_action_mask(
            base_attention_mask, action_token_mask, batch_size, seq_len
        )

        # === Step 5: Handle pixel values ===
        assert all([pv is not None for pv in pixel_values]), "Invalid VLA Example with `pixel_values = None`!"
        if isinstance(pixel_values[0], torch.Tensor):
            pixel_values_stacked = torch.stack(pixel_values)
        elif isinstance(pixel_values[0], dict):
            pixel_values_stacked = {
                k: torch.stack([pv[k] for pv in pixel_values]) for k in pixel_values[0]
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values[0])}")
        
        # === Step 6: Return collated batch ===
        output = dict(
            pixel_values=pixel_values_stacked,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        
        # Support for action chunking (ECoT + OFT): stack actions if present
        # Actions are numpy arrays from RLDS, need to convert to tensors
        if "actions" in instances[0]:
            actions_list = [
                torch.tensor(instance["actions"], dtype=torch.float32) 
                if not isinstance(instance["actions"], torch.Tensor) 
                else instance["actions"]
                for instance in instances
            ]
            output["actions"] = torch.stack(actions_list)
        
        return output