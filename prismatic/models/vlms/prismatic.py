"""
prismatic.py

PyTorch Module defining a PrismaticVLM, our general interface for defining the various different VLMs in our work.

Notes:
    - For now, we don't subclass `transformers.PretrainedModel` (or CausalLM). Instead, we assume a very limited subset
      of the {Model}ForCausalLM API that enables dispatch to the underlying LLM's `generate` utilities (feeding inputs
      through our custom projection shim).
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Type, Union,Any

import torch
from PIL import Image
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.models.backbones.llm import LLMBackbone
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import VisionBackbone
from prismatic.models.vlms.base_vlm import VLM
from prismatic.overwatch import initialize_overwatch
from prismatic.util.nn_utils import FusedMLPProjector, LinearProjector, MLPProjector

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


class PrismaticVLM(VLM):
    def __init__(
        self,
        model_id: str,
        vision_backbone: VisionBackbone,
        llm_backbone: LLMBackbone,
        enable_mixed_precision_training: bool = True,
        arch_specifier: str = "gelu-mlp",
        **kwargs,
    ) -> None:
        super().__init__(
            "prismatic",
            model_id,
            vision_backbone,
            llm_backbone,
            enable_mixed_precision_training=enable_mixed_precision_training,
        )

        # Set Weight Initialization Seed for Projector Consistency
        torch.manual_seed(vision_backbone.embed_dim)

        # Initialize Projection (Adapter) based on `arch_specifier`
        self.arch_specifier = arch_specifier
        if arch_specifier == "linear":
            self.projector = LinearProjector(vision_backbone.embed_dim, llm_backbone.embed_dim)
        elif arch_specifier.endswith("fused-gelu-mlp"):
            self.projector = FusedMLPProjector(vision_backbone.embed_dim, llm_backbone.embed_dim)
        elif arch_specifier.endswith("gelu-mlp"):
            self.projector = MLPProjector(vision_backbone.embed_dim, llm_backbone.embed_dim)
        else:
            raise ValueError(f"PrismaticVLM with `{arch_specifier = }` is not supported!")

        # Trackers
        self.vision_backbone_requires_grad = False

        # Set Module Keys =>> used in Checkpoint Saving / Model Loading
        self.all_module_keys = ["vision_backbone", "llm_backbone", "projector"]
        self.trainable_module_keys = []

        # === Generation Utilities ===
        #   => For computing likelihoods --> get tokens corresponding to "True", "False" and "Yes", "No"
        self.string2idx = {}
        for trigger_string in ["True", "False", "Yes", "No"] + [chr(ord("A") + i) for i in range(26)]:
            token_idx_list = self.llm_backbone.tokenizer.encode(trigger_string, add_special_tokens=False)
            assert len(token_idx_list) == 1, f'String "{trigger_string}" is tokenized as more than one token!'
            self.string2idx[trigger_string] = token_idx_list[0]
    def _save_kv_cache_hook(self,module,_,output):
        if hasattr(output,'past_key_values'):
            self.kv_cache_store[id(module)] = output.past_key_values


    @classmethod
    def from_pretrained(
        cls,
        pretrained_checkpoint: Path,
        model_id: str,
        vision_backbone: VisionBackbone,
        llm_backbone: LLMBackbone,
        enable_mixed_precision_training: bool = True,
        arch_specifier: str = "gelu-mlp",
        freeze_weights: bool = True,
        **kwargs,
    ) -> PrismaticVLM:
        """Initialize a PrismaticVLM from a pretrained checkpoint, freezing all weights, tailored for inference."""
        vlm = cls(
            model_id,
            vision_backbone,
            llm_backbone,
            enable_mixed_precision_training=enable_mixed_precision_training,
            arch_specifier=arch_specifier,
            **kwargs,
        )

        # Load from Checkpoint (Custom --> should load both *projector* and *llm* weights)
        model_state_dict = torch.load(pretrained_checkpoint, map_location="cpu")["model"]
        assert (
            "projector" in model_state_dict and "llm_backbone" in model_state_dict
        ), "PrismaticVLM `from_pretrained` expects checkpoint with keys for `projector` AND `llm_backbone`!"
        
        vlm.projector.load_state_dict(model_state_dict["projector"])
        
        # Load LLM backbone with strict=False to allow embedding size mismatch
        # This is necessary when loading checkpoints after adding new tokens (e.g., thinking tokens)
        missing_keys, unexpected_keys = vlm.llm_backbone.load_state_dict(model_state_dict["llm_backbone"], strict=False)
        
        # Log any missing or unexpected keys (excluding embedding-related keys which are expected)
        embedding_keys = ["llm.model.embed_tokens.weight", "llm.lm_head.weight"]
        filtered_missing = [k for k in missing_keys if not any(emb_key in k for emb_key in embedding_keys)]
        filtered_unexpected = [k for k in unexpected_keys if not any(emb_key in k for emb_key in embedding_keys)]
        
        if filtered_missing:
            overwatch.warning(f"Missing keys when loading LLM backbone (non-embedding): {filtered_missing}")
        if filtered_unexpected:
            overwatch.warning(f"Unexpected keys when loading LLM backbone (non-embedding): {filtered_unexpected}")
        
        if "vision_backbone" in model_state_dict.keys():
            vlm.vision_backbone.load_state_dict(model_state_dict["vision_backbone"])

        # Freeze Weights
        if freeze_weights:
            vlm.requires_grad_(False)
            vlm.eval()

        return vlm

    def get_prompt_builder(self, system_prompt: Optional[str] = None) -> PromptBuilder:
        prompt_initializer: Type[PromptBuilder] = self.llm_backbone.prompt_builder_fn
        return prompt_initializer(self.model_family, system_prompt=system_prompt)

    def freeze_backbones(self, stage: str) -> None:
        """
        This function sets `requires_grad_` on each of the component modules explicitly, depending on stage.

        We support two separate stages --> "align" and "finetune".
            => "align" --> vision_backbone*, llm_backbone* are frozen; only the `projector` is trained.
            => "finetune" --> vision_backbone* is frozen; both `projector` and `llm_backbone` are trained.

        :param stage: Pretraining stage in < "align" | "finetune" | "full-finetune" | "vla-train" | "vla-full-train" >
        """
        if stage == "align":
            self.vision_backbone.requires_grad_(False)
            self.llm_backbone.requires_grad_(False)
            self.projector.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["projector"]

            # Update Trackers
            self.vision_backbone_requires_grad = False

            # Explicitly Log Frozen / Trainable Components
            overwatch.info(f"[Frozen]    🥶 =>> Vision Backbone `{self.vision_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[Frozen]    🥶 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Projector `{self.arch_specifier}`", ctx_level=1)

        elif stage in {"finetune", "vla-train"}:
            self.vision_backbone.requires_grad_(False)
            self.llm_backbone.requires_grad_(True)
            self.projector.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["projector", "llm_backbone"]

            # Update Trackers
            self.vision_backbone_requires_grad = False

            # Explicitly Log Frozen / Unfrozen Components
            overwatch.info(f"[Frozen]    🥶 =>> Vision Backbone `{self.vision_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Projector `{self.arch_specifier}`", ctx_level=1)

        elif stage in {"full-finetune", "vla-full-train"}:
            self.vision_backbone.dtype = torch.float32
            self.vision_backbone.requires_grad_(True)
            self.llm_backbone.requires_grad_(True)
            self.projector.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["vision_backbone", "projector", "llm_backbone"]

            # Update Trackers
            self.vision_backbone_requires_grad = True

            # Explicitly Log Frozen / Unfrozen Components
            overwatch.info(f"[TRAINABLE] 🔥 =>> Vision Backbone `{self.vision_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Projector `{self.arch_specifier}`", ctx_level=1)

        elif stage in {"last-layer-finetune", "vla-last-layer-train"}:
            self.vision_backbone.requires_grad_(False)
            self.projector.requires_grad_(False)
            self.llm_backbone.requires_grad_(False)

            # Unfreeze final LLM layer
            for module in self.llm_backbone.last_layer_finetune_modules:
                module.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["llm_backbone"]

            # Update Trackers
            self.vision_backbone_requires_grad = False

            # Explicitly Log Frozen / Unfrozen Components
            # fmt: off
            overwatch.info(f"[Frozen]                    🥶   =>> Vision Backbone `{self.vision_backbone.identifier}`", ctx_level=1)  # noqa: E501
            overwatch.info(f"[Frozen, except last layer] 🥶🔥 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)  # noqa: E501
            overwatch.info(f"[Frozen]                    🥶   =>> Projector `{self.arch_specifier}`", ctx_level=1)
            # fmt: on

        elif stage in {"vla-sandwich-train"}:
            self.vision_backbone.dtype = torch.float32
            self.vision_backbone.requires_grad_(True)
            self.projector.requires_grad_(True)
            self.llm_backbone.requires_grad_(False)

            # Unfreeze final LLM layer
            for module in self.llm_backbone.last_layer_finetune_modules:
                module.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["vision_backbone", "projector", "llm_backbone"]

            # Update Trackers
            self.vision_backbone_requires_grad = True

            # Explicitly Log Frozen / Unfrozen Components
            # fmt: off
            overwatch.info(f"[TRAINABLE]                 🔥   =>> Vision Backbone `{self.vision_backbone.identifier}`", ctx_level=1)  # noqa: E501
            overwatch.info(f"[Frozen, except last layer] 🥶🔥 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)  # noqa: E501
            overwatch.info(f"[TRAINABLE]                 🔥   =>> Projector `{self.arch_specifier}`", ctx_level=1)
            # fmt: on

        else:
            raise ValueError(f"Stage `{stage}` is not supported for LLaVa! Try < align | finetune >")

        overwatch.debug("##################################################")
        overwatch.debug("#####      Trainable Network Parameters:     #####")
        overwatch.debug("##################################################")
        for name, param in self.named_parameters():
            if param.requires_grad:
                overwatch.debug(name)

    def load_from_checkpoint(self, stage: str, run_dir: Path, pretrained_checkpoint: Optional[Path] = None) -> None:
        """Load weights from checkpoint (if required by the given stage)."""
        assert stage in {"align", "finetune", "full-finetune"}, f"Stage {stage} is not supported!"

        # If we're running a `no-align` architecture, we're good!
        if self.arch_specifier.startswith("no-align"):
            overwatch.info(
                f"PrismaticVLM with `{self.arch_specifier = }` does not require pretrained weights!", ctx_level=1
            )
            return

        # Otherwise, handle stage-specific logic!
        if stage == "align":
            overwatch.info("Stage `align` does not require pretrained weights =>> Starting Training", ctx_level=1)
            return

        # Otherwise, load from `pretrained_checkpoint` or match on `run_dir` (s/+stage-finetune/+stage-align/g)
        overwatch.info("Stage `finetune` requires `align` pretrained weights", ctx_level=1)

        # Config specifies path to a checkpoint to load
        if pretrained_checkpoint is not None:
            overwatch.info(f"Loading from Provided Checkpoint `{pretrained_checkpoint}`", ctx_level=1)
            model_state_dict = torch.load(pretrained_checkpoint)["model"]
            self.projector.load_state_dict(model_state_dict["projector"])

            return

        # [Contract] If no `pretrained_checkpoint`, assume `align` lives in the run directory; string substitution!
        model, scale, _, seed = run_dir.name.split("+")
        align_dirs = [
            d
            for d in run_dir.parent.iterdir()
            if (d.name.startswith(f"{model}+{scale}") and d.name.endswith(f"+stage-align+{seed}"))
        ]
        assert len(align_dirs) == 1, "Multiple or No Valid Pretrained Directories Exist -- Double Check `runs`!"
        if (pretrained_checkpoint := (align_dirs[0] / "checkpoints" / "latest-checkpoint.pt")).exists():
            overwatch.info(f"Loading from Discovered Checkpoint `{pretrained_checkpoint}`", ctx_level=1)
            model_state_dict = torch.load(pretrained_checkpoint)["model"]
            self.projector.load_state_dict(model_state_dict["projector"])
        else:
            raise ValueError(f"Could not find valid `align` checkpoint at {pretrained_checkpoint}!")

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return an FSDP _or_policy over the policies returned by each individual backbone (and our VLM policy)."""
        vision_fsdp_wrapping_policy = self.vision_backbone.get_fsdp_wrapping_policy()
        llm_fsdp_wrapping_policy = self.llm_backbone.get_fsdp_wrapping_policy()
        # llm_fsdp_wrapping_policy = partial(
        #     transformer_auto_wrap_policy,
        #     transformer_layer_cls={LlamaDecoderLayer},
        # )
        # Get Prismatic Wrapping Policy =>> just a module wrapping policy around `self.projector`
        prismatic_fsdp_wrapping_policy = partial(
            _module_wrap_policy,
            module_classes={LinearProjector, MLPProjector, FusedMLPProjector},
        )

        # Return union (_or_) over constituent policies
        #   => Note: there is *not* a fall-through policy; any module that isn't covered by the above constituents will
        #            automatically be folded into the root VLM FSDP instance.
        return partial(
            _or_policy,
            policies=[
                vision_fsdp_wrapping_policy,
                llm_fsdp_wrapping_policy,
                prismatic_fsdp_wrapping_policy,
            ],
        )

    # Note =>> We're not explicitly subclassing `PreTrainedModel` because we don't need the bloat; however, `forward()`
    #          *must* match the signature of a `{Model}ForCausalLM` so that we can inherit from `GenerationMixin`

    # ruff: noqa: C901
    
    # def forward(
    #     self,
    #     input_ids: Optional[torch.LongTensor] = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     pixel_values: Optional[torch.FloatTensor] = None,
    #     labels: Optional[torch.LongTensor] = None,
    #     inputs_embeds: Optional[torch.FloatTensor] = None,
    #     past_key_values: Optional[List[torch.FloatTensor]] = None,
    #     use_cache: Optional[bool] = None,
    #     output_attentions: Optional[bool] = None,
    #     output_hidden_states: Optional[bool] = None,
    #     return_dict: Optional[bool] = None,
    #     multimodal_indices: Optional[torch.LongTensor] = None,
    #     cfg: Optional[Any] = None,
    #     ) -> CausalLMOutputWithPast:
    #     """Run a forward pass through the VLM, returning a CausalLMOutputWithPast instance (contains loss)."""

    #     # Handle Inference (leverage cache, short-circuit on just LLM forward)
    #     if input_ids.shape[1] == 1 and past_key_values is not None:
    #         # We're leveraging the cache, so just redirect to `self.llm_backbone` with `input_ids` and `past_key_values`
    #         output = self.llm_backbone(
    #             input_ids=input_ids,
    #             attention_mask=None,
    #             position_ids=None,
    #             past_key_values=past_key_values,
    #             inputs_embeds=None,
    #             labels=None,
    #             use_cache=use_cache,
    #             output_attentions=output_attentions,
    #             output_hidden_states=output_hidden_states,
    #             return_dict=return_dict,
    #         )
    #         return output

    #     elif input_ids.shape[1] == 1 or pixel_values is None:
    #         raise RuntimeError("Invalid `forward()` call!")

    #     # Handle Multimodal Indices is None --> pretend like the batch is fully multimodal (always image + text)!
    #     if multimodal_indices is None:
    #         multimodal_indices = torch.arange(len(input_ids), dtype=torch.long, device=input_ids.device)

    #     # Handle Multimodal Indices is Empty (len == 0) --> simple unimodal forward
    #     elif len(multimodal_indices) == 0:
    #         return self.llm_backbone(
    #             input_ids=input_ids,
    #             attention_mask=attention_mask,
    #             position_ids=None,
    #             past_key_values=past_key_values,
    #             inputs_embeds=None,
    #             labels=labels,
    #             use_cache=use_cache,
    #             output_attentions=output_attentions,
    #             output_hidden_states=output_hidden_states,
    #             return_dict=return_dict,
    #         )

    #     # Run Visual Feature Extraction
    #     with torch.set_grad_enabled(self.vision_backbone_requires_grad):
    #         if isinstance(pixel_values, dict):
    #             patch_features = self.vision_backbone({k: pixel_values[k][multimodal_indices] for k in pixel_values})
    #         else:
    #             patch_features = self.vision_backbone(pixel_values[multimodal_indices])

    #     # Projection Logic :: [bsz, num_patches, llm_embed_dim] =>> num_patches = (2 *) (256 + 1) for ViT-L + CLS
    #     projected_patch_embeddings = self.projector(patch_features)
    #     projected_patch_attention_mask = None
    #     if attention_mask is not None:
    #         projected_patch_attention_mask = torch.full(
    #             (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
    #             True,
    #             dtype=attention_mask.dtype,
    #             device=attention_mask.device,
    #         )

    #     # Get Input Embeddings from LLM Backbone :: [bsz, input_seq_len, llm_embed_dim]
    #     input_embeddings = self.llm_backbone.embed_input_ids(input_ids.to(self.device))

    #     # Build Multimodal Embeddings (and build resulting attention mask)
    #     multimodal_embeddings = torch.cat(
    #         [
    #             input_embeddings[multimodal_indices, :1, :],
    #             projected_patch_embeddings,
    #             input_embeddings[multimodal_indices, 1:, :],
    #         ],
    #         dim=1,
    #     )
    #     multimodal_attention_mask = None
    #     if attention_mask is not None:
    #         multimodal_attention_mask = torch.cat(
    #             [
    #                 attention_mask[multimodal_indices, :1],
    #                 projected_patch_attention_mask,
    #                 attention_mask[multimodal_indices, 1:],
    #             ],
    #             dim=1,
    #         )

    #     # [Contract] We assume the first token of `labels` (associated with <BOS>) is already marked as "IGNORE"
    #     #   => We'll ignore the per-token outputs for each of the patch embeddings as well!
    #     multimodal_labels = None
    #     if labels is not None:
    #         projected_patch_labels = torch.full(
    #             (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
    #             IGNORE_INDEX,
    #             dtype=labels.dtype,
    #             device=labels.device,
    #         )
    #         multimodal_labels = torch.cat(
    #             [labels[multimodal_indices, :1], projected_patch_labels, labels[multimodal_indices, 1:]], dim=1
    #         )

    #     # === Add Unimodal Handling ===

    #     # Create Fused Embeddings, Attention Mask, and Labels by Merging with "unimodal" Inputs (if applicable)
    #     unimodal_indices = torch.tensor(
    #         [idx for idx in range(len(input_ids)) if idx not in multimodal_indices],
    #         dtype=torch.long,
    #         device=multimodal_indices.device,
    #     )

    #     # No "unimodal" data --> Fused == Multimodal
    #     if len(unimodal_indices) == 0:
    #         fused_embeddings = multimodal_embeddings
    #         fused_attention_mask = multimodal_attention_mask
    #         fused_labels = multimodal_labels

    #     else:
    #         # Otherwise --> Merge w/ unimodal data

    #         # This doesn't matter --> but in the "normal" case this is the embedding of the <PAD> token
    #         #   => NOTE :: Verified that `zeros/randn/empty/<PAD> embedding` all return the same result!
    #         unimodal_embeddings_pad = torch.zeros(
    #             (len(unimodal_indices), projected_patch_embeddings.shape[1], input_embeddings.shape[2]),
    #             dtype=input_embeddings.dtype,
    #             device=input_embeddings.device,
    #         )
    #         unimodal_attention_pad = torch.full(
    #             (len(unimodal_indices), projected_patch_embeddings.shape[1]),
    #             False,
    #             dtype=attention_mask.dtype,
    #             device=attention_mask.device,
    #         )
    #         unimodal_labels_pad = torch.full(
    #             (len(unimodal_indices), projected_patch_embeddings.shape[1]),
    #             IGNORE_INDEX,
    #             dtype=labels.dtype,
    #             device=labels.device,
    #         )

    #         unimodal_embeddings = torch.cat([input_embeddings[unimodal_indices], unimodal_embeddings_pad], dim=1)
    #         unimodal_attention_mask = torch.cat([attention_mask[unimodal_indices], unimodal_attention_pad], dim=1)
    #         unimodal_labels = torch.cat([labels[unimodal_indices], unimodal_labels_pad], dim=1)

    #         # Create "Fused" Tensors by Stacking Multimodal & Unimodal
    #         fused_embeddings = torch.vstack([multimodal_embeddings, unimodal_embeddings])
    #         fused_attention_mask = torch.vstack([multimodal_attention_mask, unimodal_attention_mask])
    #         fused_labels = torch.vstack([multimodal_labels, unimodal_labels])

    #     # Run LLM Forward --> returns CausalLMOutputWithPast!
    #     return self.llm_backbone(
    #         input_ids=None,
    #         attention_mask=fused_attention_mask,
    #         position_ids=None,
    #         past_key_values=past_key_values,
    #         inputs_embeds=fused_embeddings,
    #         labels=fused_labels,
    #         use_cache=use_cache,
    #         output_attentions=output_attentions,
    #         output_hidden_states=output_hidden_states,
    #         return_dict=return_dict,
    #     )


# ##################################################################################################################################3
#     # === GenerationMixin Methods ===
#     #   => Note: The following methods override the functionality of `transformers.GenerationMixin`; these expect the
#     #            contract in each of the function signatures, and also expect our `forward` function to roughly take
#     #            the same arguments as the underlying LLM (see `LlamaModelForCausalLM` as an example)

    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        **kwargs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Borrowed from `LlamaForCausalLM` --> in general, just handles caching logic during generation."""
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # Make sure `pixel_values` are preserved in `model_inputs`
        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
            }
        )

        return model_inputs

    @torch.inference_mode()
    def generate_batch(
        self,
        pixel_values: Union[torch.Tensor, Dict[str, torch.Tensor]],
        texts: List[str],
        return_string_probabilities: Optional[List[str]] = None,
        **kwargs: str,
    ) -> Union[List[str], List[List[float]]]:
        # For now, only support generation with a batch size of 1 for simplicity
        tokenizer = self.llm_backbone.tokenizer

        # Prepare Inputs
        batch_input_ids = [
            tokenizer(text, truncation=True, return_tensors="pt").input_ids.to(self.device) for text in texts
        ]
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.device) for k, v in pixel_values.items()}
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # Create Output Lists
        gen_texts, gen_probabilities = [], []

        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = self.llm_backbone.half_precision_dtype
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.enable_mixed_precision_training):
            for idx, input_ids in enumerate(batch_input_ids):
                if isinstance(pixel_values, torch.Tensor):
                    pixel_values = pixel_values[idx]
                elif isinstance(pixel_values, dict):
                    pixel_values = {k: pixel_values[k][idx] for k in pixel_values}
                else:
                    raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

                # Handle `return_string_probabilities`
                if return_string_probabilities is None:
                    full_out_ids = super().generate(input_ids=input_ids, pixel_values=pixel_values, **kwargs)
                    gen_ids = full_out_ids[0, input_ids.shape[1] :]

                    # Decode `gen_ids` and strip any <EOS> tokens
                    gen_texts.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())

                else:
                    full_out_dict = super().generate(
                        input_ids=input_ids,
                        pixel_values=pixel_values,
                        output_scores=True,
                        return_dict_in_generate=True,
                        **kwargs,
                    )

                    # Generation pattern should usually be [TOKEN] <EOS> for True/False and Yes/No Generations
                    gen_ids = full_out_dict.sequences[0, input_ids.shape[1] :]

                    # [Debug] Verify that the first token generated is in `self.string2idx.values()`
                    # assert gen_ids[0] in self.string2idx.values(), "Generated ID not in mapping!"

                    # Decode `gen_ids` and strip any <EOS> tokens
                    gen_texts.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())

                    # Get all token probabilities --> softmax over logits
                    token_probs = torch.softmax(full_out_dict.scores[0][0], dim=0)

                    # Get *normalized* probabilities for all values in `return_token_probabilities`
                    slice_idxs = torch.tensor([self.string2idx[s] for s in return_string_probabilities])
                    string_probs_unnormalized = token_probs[slice_idxs]
                    string_probs = string_probs_unnormalized / string_probs_unnormalized.sum()
                    gen_probabilities.append(string_probs.cpu().numpy().tolist())

        return gen_texts if return_string_probabilities is None else gen_probabilities

    @torch.inference_mode()
    def generate(self, image: Image, prompt_text: str, **kwargs: str) -> str:
        # For now, only support generation with a batch size of 1 for simplicity
        image_transform, tokenizer = self.vision_backbone.image_transform, self.llm_backbone.tokenizer

        # Prepare Inputs
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.device)
        pixel_values = image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.device) for k, v in pixel_values.items()}
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = self.llm_backbone.half_precision_dtype
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.enable_mixed_precision_training):
            # fmt: off
            generated_ids = super().generate(
                input_ids=input_ids,            # Shape: [1, seq]
                pixel_values=pixel_values,      # Shape: [1, 3, res, res] or Dict[str, Shape[1, 3, res, res]]
                **kwargs
            )
            # fmt: on

        generated_text = tokenizer.decode(generated_ids[0, input_ids.shape[1] :], skip_special_tokens=True).strip()

        return generated_text
    # def forward(
    #     self,
    #     input_ids: torch.LongTensor,
    #     attention_mask: torch.Tensor,
    #     pixel_values: torch.FloatTensor,
    #     labels: Optional[torch.LongTensor] = None,
    #     position_ids: Optional[torch.LongTensor] = None,
    #     scheduled_stage: int = 0,  # 隐式推理阶段
    #     start_thinking_id: int = 0,
    #     end_thinking_id: int = 0,
    #     latent_token_id: int = 0,
    #     cfg: Optional[Any] = None,
    #     **kwargs,
    # ) -> CausalLMOutputWithPast:
    #     """
    #     Optimized forward pass with latent 'thinking token' feedback for multi-stage implicit reasoning.
        
    #     Key optimization: Since the collator aligns thinking token positions across batch samples,
    #     we can use a much simpler and more efficient approach.
        
    #     Args:
    #         input_ids: [B, seq_len] token IDs (with aligned thinking token positions)
    #         attention_mask: [B, seq_len] attention mask
    #         pixel_values: [B, C, H, W] image tensor
    #         labels: optional [B, seq_len] labels for loss
    #         position_ids: optional position IDs
    #         scheduled_stage: current reasoning stage
    #         latent_token_id: ID of the thinking token
    #     """
    #     from prismatic.vla.datasets.datasets import IGNORE_INDEX
        
    #     # === Step 1: Visual feature extraction ===
    #     # Handle multimodal indices like the original forward method
    #     multimodal_indices = torch.arange(len(input_ids), dtype=torch.long, device=input_ids.device)
        
    #     with torch.set_grad_enabled(self.vision_backbone_requires_grad):
    #         if isinstance(pixel_values, dict):
    #             patch_features = self.vision_backbone({k: pixel_values[k][multimodal_indices] for k in pixel_values})
    #         else:
    #             patch_features = self.vision_backbone(pixel_values[multimodal_indices])
        
    #     projected_patch_embeddings = self.projector(patch_features)

    #     # Attention mask for image patches
    #     projected_patch_attention_mask = torch.full(
    #         (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
    #         True,
    #         dtype=attention_mask.dtype,
    #         device=attention_mask.device,
    #     )

    #     # === Step 2: Build multimodal embeddings ===
    #     input_embeddings = self.llm_backbone.embed_input_ids(input_ids.to(self.device))
        
    #     # Build multimodal embeddings (text + image patches) - handle multimodal_indices properly
    #     multimodal_embeddings = torch.cat(
    #         [
    #             input_embeddings[multimodal_indices, :1, :],               # <BOS>
    #             projected_patch_embeddings,
    #             input_embeddings[multimodal_indices, 1:, :],               # rest of text
    #         ],
    #         dim=1,
    #     )
    #     multimodal_attention_mask = torch.cat(
    #         [
    #             attention_mask[multimodal_indices, :1],
    #             projected_patch_attention_mask,
    #             attention_mask[multimodal_indices, 1:],
    #         ],
    #         dim=1,
    #     )

    #     # Build multimodal labels
    #     multimodal_labels = None
    #     if labels is not None:
    #         projected_patch_labels = torch.full(
    #             (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
    #             IGNORE_INDEX,
    #             dtype=labels.dtype,
    #             device=labels.device,
    #         )
    #         multimodal_labels = torch.cat(
    #             [labels[multimodal_indices, :1], projected_patch_labels, labels[multimodal_indices, 1:]], dim=1
    #         )

    #     # === Step 3: Handle thinking tokens (simplified due to collator alignment) ===
    #     # Find thinking token positions in the original input_ids
    #     thinking_token_mask = (input_ids == latent_token_id)
        
    #     # Early return if no thinking tokens
    #     if not thinking_token_mask.any():
    #         print(f"Warning: no thinking token in sample forward pass")
    #         print(f"input ids: {input_ids}")
    #         return self.llm_backbone(
    #             input_ids=None,
    #             inputs_embeds=multimodal_embeddings,
    #             attention_mask=multimodal_attention_mask,
    #             labels=multimodal_labels,
    #             position_ids=position_ids,
    #             **kwargs,
    #         )

    #     # Calculate number of thinking tokens per sample (should be same across batch due to collator)
    #     thinking_token_count = scheduled_stage * cfg.vla.thinking_token_count
        
    #     # Find the first thinking token position (should be same across batch due to collator)
    #     first_thinking_pos = thinking_token_mask.nonzero(as_tuple=True)[1][0].item()
        
    #     # Adjust position for multimodal embeddings (account for image patches)
    #     multimodal_first_thinking_pos = first_thinking_pos + projected_patch_embeddings.shape[1] + 1  # +1 for BOS
        
    #     # === Step 3.5: Mark thinking tokens as IGNORE_INDEX in labels ===
    #     # 将整个 thinking token 序列（start + thinking + end）标记为 IGNORE_INDEX
    #     if multimodal_labels is not None:
    #         # 计算 thinking token 序列的结束位置
    #         # start_token + thinking_tokens + end_token
    #         thinking_sequence_length = 1 + thinking_token_count + 1  # start + thinking + end
    #         thinking_end_pos = multimodal_first_thinking_pos + thinking_sequence_length
            
    #         # 确保不超出序列长度
    #         thinking_end_pos = min(thinking_end_pos, multimodal_labels.shape[1])
            
    #         # 将 thinking token 序列标记为 IGNORE_INDEX
    #         multimodal_labels[:, multimodal_first_thinking_pos:thinking_end_pos] = IGNORE_INDEX
        
    #     # === Step 4: Iterative thinking token processing ===
    #     inputs_embeds = multimodal_embeddings.clone()
    #     kv_cache = None
        
    #     # Safety check: ensure we don't exceed sequence length
    #     max_seq_len = inputs_embeds.shape[1]
        
    #     # Create full position IDs once
    #     batch_size = inputs_embeds.shape[0]
    #     full_position_ids = torch.arange(max_seq_len, device=inputs_embeds.device).unsqueeze(0).expand(batch_size, -1)
        
    #     # Initialize compute range
    #     next_compute_range = (0, multimodal_first_thinking_pos)
        
    #     # Process each thinking token iteratively
        for thinking_idx in range(thinking_token_count):
            current_thinking_pos = multimodal_first_thinking_pos + thinking_idx
            
            # Safety check: ensure current_thinking_pos is within bounds
            if current_thinking_pos >= max_seq_len:
                print(f"Warning: current_thinking_pos {current_thinking_pos} >= max_seq_len {max_seq_len}, skipping")
                break
            import torch.distributed as dist
            rank = dist.get_rank()

            # 准备要传入的参数
            # current_embeds = inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :]
            # current_mask = multimodal_attention_mask[:, :next_compute_range[1]]
            # current_pos_ids = full_position_ids[:, next_compute_range[0]:next_compute_range[1]]
            
            # # 打印关键信息
            # print(f"[Rank {rank}] Step 4, thinking_idx {thinking_idx}:")
            # print(f"  - next_compute_range: {next_compute_range}")
            # print(f"  - embeds shape: {current_embeds.shape}")
            # print(f"  - mask shape: {current_mask.shape}")
            # print(f"  - pos_ids shape: {current_pos_ids.shape}")
            
            # Forward pass using the correct compute range
            if kv_cache is None:
                # First pass: process from beginning to current thinking token (exclusive)
                # 临时禁用 gradient checkpointing 以确保 KV cache 工作
                # self.llm_backbone.llm.gradient_checkpointing_disable()
                
                outputs = self.llm_backbone(
                    input_ids=None,
                    inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
                    attention_mask=multimodal_attention_mask[:, next_compute_range[0]:next_compute_range[1]],
                    position_ids=full_position_ids[:, next_compute_range[0]:next_compute_range[1]],
                    past_key_values=None,
                    **kwargs,
                )
                
                # 恢复原始 gradient checkpointing 设置
                # self.llm_backbone.llm.gradient_checkpointing_enable()
                hidden_states_offset = 0
            else:
                # Subsequent passes: use KV cache
                past_key_values = []
                for k, v in kv_cache:
                    past_k = k[:, :, :next_compute_range[0], :]
                    past_v = v[:, :, :next_compute_range[0], :]
                    past_key_values.append((past_k, past_v))
                
                # 临时禁用 gradient checkpointing 以确保 KV cache 工作
                # self.llm_backbone.llm.gradient_checkpointing_disable()
                
                outputs = self.llm_backbone(
                    input_ids=None,
                    inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
                    attention_mask=multimodal_attention_mask[:, :next_compute_range[1]],
                    position_ids=full_position_ids[:, next_compute_range[0]:next_compute_range[1]],
                    past_key_values=past_key_values,
                    use_cache=True,  # 显式启用 KV cache
                    **kwargs,
                )
                
                # 恢复原始 gradient checkpointing 设置
                # self.llm_backbone.llm.gradient_checkpointing_enable()
                hidden_states_offset = next_compute_range[0]
            
    #         # Update KV cache (check if it exists)
    #         # 使用 detach() 来避免 FSDP 的 pre_backward_hooks 干扰
    #         if outputs.past_key_values is not None:
    #             kv_cache = outputs.past_key_values
            
            
    #         # Update compute range for next iteration
    #         next_compute_range = (
    #             next_compute_range[1],
    #             current_thinking_pos + 1 if thinking_idx + 1 < thinking_token_count else max_seq_len
    #         )
            
    #         # Get hidden states and update thinking token embeddings
    #         hidden_states = outputs.hidden_states[-1]

    #         # Safety check: ensure hidden_states has content
    #         if hidden_states.shape[1] > 0:
    #             # 计算对应于当前 thinking 位置前一个 token 的本地下标
    #             local_idx = (current_thinking_pos - 1) - hidden_states_offset

    #             # 边界检查
    #             if local_idx < 0 or local_idx >= hidden_states.shape[1]:
    #                 print(
    #                     f"Warning: local_idx {local_idx} out of bounds for hidden_states len {hidden_states.shape[1]} at thinking_idx {thinking_idx}"
    #                 )
    #                 break

    #             # 参考实现：分解为列表后替换再 stack，避免 in-place，保持计算图
    #             tensor_list = [
    #                 [inputs_embeds[batch_idx, pos, :] for pos in range(inputs_embeds.shape[1])]
    #                 for batch_idx in range(inputs_embeds.shape[0])
    #             ]

    #             for batch_idx in range(inputs_embeds.shape[0]):
    #                 tensor_list[batch_idx][current_thinking_pos] = hidden_states[batch_idx, local_idx, :]

    #             inputs_embeds = torch.stack(
    #                 [torch.stack(tensor_list[batch_idx]) for batch_idx in range(inputs_embeds.shape[0])], dim=0
    #             )
    #         else:
    #             print(f"Warning: hidden_states is empty at thinking_idx {thinking_idx}")
    #             break

    #     # === Step 5: Final forward pass for remaining tokens ===
    #     # Use the updated compute range for final pass
    #     # 临时禁用 gradient checkpointing 以确保 KV cache 工作
    #     # self.llm_backbone.llm.gradient_checkpointing_disable()
    #     # import torch.distributed as dist
    #     rank = dist.get_rank()

    #     # 准备要传入的参数
    #     # current_embeds = inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :]
    #     # current_mask = multimodal_attention_mask[:, :next_compute_range[1]]
    #     # current_pos_ids = full_position_ids[:, next_compute_range[0]:next_compute_range[1]]
        
    #     # # 打印关键信息
    #     # print(f"[Rank {rank}] Step 5, thinking_idx {thinking_idx}:")
    #     # print(f"  - next_compute_range: {next_compute_range}")
    #     # print(f"  - embeds shape: {current_embeds.shape}")
    #     # print(f"  - mask shape: {current_mask.shape}")
    #     # print(f"  - pos_ids shape: {current_pos_ids.shape}")
    #     final_output = self.llm_backbone(
    #         input_ids=None,
    #         inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
    #         attention_mask=multimodal_attention_mask[:, :next_compute_range[1]],
    #         position_ids=full_position_ids[:, next_compute_range[0]:next_compute_range[1]],
    #         past_key_values=(
    #             [
    #                 (
    #                     k[:, :, :next_compute_range[0], :],
    #                     v[:, :, :next_compute_range[0], :],
    #                 )
    #                 for k, v in kv_cache
    #             ]
    #             if kv_cache is not None
    #             else None
    #         ),
    #         labels=multimodal_labels[:, next_compute_range[0]:next_compute_range[1]] if multimodal_labels is not None else None,
    #         use_cache=True,  # 显式启用 KV cache
    #         **kwargs,
    #     )
        
    #     # 恢复原始 gradient checkpointing 设置
    #     # self.llm_backbone.llm.gradient_checkpointing_enable()
        
    #     # Create full logits tensor to match training strategy expectations
    #     batch_size, remaining_seq_len, vocab_size = final_output.logits.shape
    #     full_seq_len = multimodal_embeddings.shape[1]
        
    #     # Initialize full logits with very negative values
    #     full_logits = torch.full(
    #         (batch_size, full_seq_len, vocab_size),
    #         -float('inf'),
    #         dtype=final_output.logits.dtype,
    #         device=final_output.logits.device
    #     )
        
    #     # Fill in the actual logits for the processed segment
    #     full_logits[:, next_compute_range[0]:next_compute_range[1], :] = final_output.logits
        
    #     # Create the final output with the full logits
    #     final_output.logits = full_logits
        
    #     return final_output


    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        scheduled_stage: int = 0,
        start_thinking_id: int = 0,
        end_thinking_id: int = 0,
        latent_token_id: int = 0,
        cfg: Optional[Any] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        [GEMINI] 这是一个 `forward` 的测试版本，完整实现了“无 K/V Cache，全量重计算”的方案。

        核心改动:
        1.  完全移除了 `past_key_values` (K/V Cache) 的使用。
        2.  在迭代的 `for` 循环中，每次都从序列的开头进行前向传播，直到当前需要计算的位置。
        3.  最终的 `forward pass` 也是一次完整的、从头到尾的计算。
        4.  大大简化了 `logits` 的拼接逻辑，因为最终的 `forward pass` 直接产出了完整的 logits。

        这个版本旨在最大程度地降低显存激活（activations）的峰值占用，并可以与 FSDP 的激活检查点（Activation Checkpointing）配合使用。
        代价是计算量显著增加，训练速度会变慢。
        """

        # === Step 1 & 2: 视觉特征提取与多模态嵌入构建 (无变化) ===
        # 这部分逻辑与 K/V Cache 无关，保持原样。
        multimodal_indices = torch.arange(len(input_ids), dtype=torch.long, device=input_ids.device)

        with torch.set_grad_enabled(self.vision_backbone_requires_grad):
            if isinstance(pixel_values, dict):
                patch_features = self.vision_backbone({k: pixel_values[k][multimodal_indices] for k in pixel_values})
            else:
                patch_features = self.vision_backbone(pixel_values[multimodal_indices])

        projected_patch_embeddings = self.projector(patch_features)
        projected_patch_attention_mask = torch.full(
            (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
            True,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )

        input_embeddings = self.llm_backbone.embed_input_ids(input_ids.to(self.device))
        
        # ⭐ [OFT-style] Zero out action token embeddings to prevent information leakage
        # Action tokens are used as placeholders for parallel decoding
        from prismatic.vla.constants import ACTION_TOKEN_BEGIN_IDX,ACTION_TOKEN_END_IDX
        action_mask = (input_ids >= ACTION_TOKEN_BEGIN_IDX) & (input_ids <= ACTION_TOKEN_END_IDX) # (batch, seq_len)
        action_mask_expanded = action_mask.unsqueeze(-1)  # (batch, seq_len, 1)
        input_embeddings = input_embeddings * (~action_mask_expanded).float()  # Zero out action embeddings

        multimodal_embeddings = torch.cat(
            [
                input_embeddings[multimodal_indices, :1, :],
                projected_patch_embeddings,
                input_embeddings[multimodal_indices, 1:, :],
            ],
            dim=1,
        )
        multimodal_attention_mask = torch.cat(
            [
                attention_mask[multimodal_indices, :1],
                projected_patch_attention_mask,
                attention_mask[multimodal_indices, 1:],
            ],
            dim=1,
        )

        multimodal_labels = None
        if labels is not None:
            projected_patch_labels = torch.full(
                (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device,
            )
            multimodal_labels = torch.cat(
                [labels[multimodal_indices, :1], projected_patch_labels, labels[multimodal_indices, 1:]], dim=1
            )

        # === Step 3 & 3.5: 处理 Thinking Tokens (无变化) ===
        # 这部分逻辑与 K/V Cache 无关，保持原样。
        thinking_token_mask = (input_ids == latent_token_id)

        if not thinking_token_mask.any():
            # ... (提前返回的逻辑保持不变) ...
            return self.llm_backbone(
                inputs_embeds=multimodal_embeddings,
                attention_mask=multimodal_attention_mask,
                labels=multimodal_labels,
                position_ids=position_ids,
                use_cache=False,  # [!!] MODIFIED: 确保这里也禁用 cache
                **kwargs,
            )

        thinking_token_count = scheduled_stage * cfg.vla.thinking_token_count
        first_thinking_pos = thinking_token_mask.nonzero(as_tuple=True)[1][0].item()
        multimodal_first_thinking_pos = first_thinking_pos + projected_patch_embeddings.shape[1] + 1
        
        if multimodal_labels is not None:
            thinking_sequence_length = 1 + thinking_token_count + 1
            thinking_end_pos = multimodal_first_thinking_pos + thinking_sequence_length - 2
            thinking_end_pos = min(thinking_end_pos, multimodal_labels.shape[1])
            multimodal_labels[:, multimodal_first_thinking_pos:thinking_end_pos] = IGNORE_INDEX
            
        # === Step 4: 迭代式思考处理 (无 K/V CACHE 版本) ===
        
        # [!!] MODIFIED: 唯一的“状态”是不断被更新的 inputs_embeds
        inputs_embeds = multimodal_embeddings.clone()
        
        max_seq_len = inputs_embeds.shape[1]
        batch_size = inputs_embeds.shape[0]
        full_position_ids = torch.arange(max_seq_len, device=inputs_embeds.device).unsqueeze(0).expand(batch_size, -1)

        for thinking_idx in range(thinking_token_count):
            current_thinking_pos = multimodal_first_thinking_pos + thinking_idx
            
            if current_thinking_pos >= max_seq_len:
                break

            # [!!] MODIFIED: 每次迭代的计算范围都是从头开始，到当前思考位置之前
            # 我们需要处理到 `current_thinking_pos` 之前的位置，以获取该位置的 `hidden_state`
            compute_until_pos = current_thinking_pos
            
            # 如果 `compute_until_pos` 为0，说明是第一个 thinking token，没有前文，跳过
            if compute_until_pos == 0:
                continue

            # 【重要】在这里，可以配合 FSDP 开启激活检查点
            # self.llm_backbone.gradient_checkpointing_enable()

            outputs = self.llm_backbone(
                input_ids=None,
                inputs_embeds=inputs_embeds[:, :compute_until_pos, :],
                attention_mask=multimodal_attention_mask[:, :compute_until_pos],
                position_ids=full_position_ids[:, :compute_until_pos],
                past_key_values=None,      # [!!] REMOVED: 不再使用 cache
                use_cache=False,           # [!!] MODIFIED: 明确禁用 cache
                output_hidden_states=True, # 仍然需要 hidden_states 来进行反馈
                **kwargs,
            )

            # [!!] REMOVED: 所有与 kv_cache 和 hidden_states_offset 相关的逻辑都已删除

            # 从输出中获取最后一个时间步的 hidden_state
            hidden_states = outputs.hidden_states[-1]
            
            if hidden_states.shape[1] > 0:
                last_hidden_state = hidden_states[:, -1, :] # 形状为 [B, D]

                # [!!] MODIFIED: 使用“分解-重组”模式来更新 inputs_embeds，以保证梯度流
                # 注意: 这个操作性能较低，但能确保梯度正确性。
                tensor_list = [
                    [inputs_embeds[batch_idx, pos, :] for pos in range(inputs_embeds.shape[1])]
                    for batch_idx in range(inputs_embeds.shape[0])
                ]

                for batch_idx in range(inputs_embeds.shape[0]):
                    tensor_list[batch_idx][current_thinking_pos] = last_hidden_state[batch_idx, :]

                inputs_embeds = torch.stack(
                    [torch.stack(tensor_list[batch_idx]) for batch_idx in range(inputs_embeds.shape[0])], dim=0
                )
            else:
                print(f"Warning: hidden_states is empty at thinking_idx {thinking_idx}")
                break

        # === Step 5: 最终的前向传播 (无 K/V CACHE 版本) ===
        # [!!] MODIFIED: 经过循环，inputs_embeds 已经包含了所有“思考”的结果。
        # 我们现在只需要对这个完整的、最终的 inputs_embeds 进行一次标准的前向传播。
        
        final_output = self.llm_backbone(
            input_ids=None,
            inputs_embeds=inputs_embeds,                  # 传入完整的、更新后的 embeds
            attention_mask=multimodal_attention_mask,     # 传入完整的 mask
            position_ids=full_position_ids,               # 传入完整的 position_ids
            past_key_values=None,                         # 不使用 cache
            labels=multimodal_labels,                     # 传入完整的 labels
            use_cache=False,                              # 禁用 cache
            **kwargs,
        )
        
        # [!!] SIMPLIFIED: 不再需要手动拼接 logits！
        # 因为上面的一次完整 `forward` pass 已经为我们计算出了所有位置的 logits 和最终的 loss。
        # `final_output` 对象已经包含了所有我们需要的东西。
        # print('no kv')
        return final_output