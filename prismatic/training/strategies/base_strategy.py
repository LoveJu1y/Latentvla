"""
base_strategy.py

Abstract class definition of a (distributed) training strategy, with full annotations of class methods, utility
functions, and initialization logic.

Training Strategies (DDP, FSDP-Grad, FSDP-Full) tend to have a lot of repeated components; this class does a lot of
heavy lifting.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from time import sleep
from typing import Callable, Optional, Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset
from tqdm import tqdm
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.models.vlms import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.training.metrics import Metrics, VLAMetrics
from prismatic.util import check_bloat16_supported
from prismatic.util.batching_utils import SplitModalitySampler
from prismatic.util.cot_utils import get_cot_tags_list
from prismatic.util.data_utils import PaddedCollatorForActionPrediction, PaddedCollatorForLanguageModeling
from prismatic.vla.action_tokenizer import ActionTokenizer

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

# LLaMA EOS Token
EOS_TOKEN = 2

from prismatic.models.vlas.openvla_actionchunk import OpenVLA_ActionChunk
# === Abstract Base Class for an arbitrary Training Strategy ===
class TrainingStrategy(ABC):
    def __init__(
        self,
        vla:OpenVLA_ActionChunk,
        # vlm: PrismaticVLM,
        device_id: int,
        stage: str,
        epochs: int,
        max_steps: Optional[int],
        global_batch_size: int,
        per_device_batch_size: int,
        learning_rate: float,
        weight_decay: float,
        max_grad_norm: float,
        lr_scheduler_type: str,
        warmup_ratio: float,
        enable_gradient_checkpointing: bool = True,
        enable_mixed_precision_training: bool = True,
        reduce_in_full_precision: bool = False,
        mixed_precision_dtype: torch.dtype = torch.bfloat16,
        worker_init_fn: Optional[Callable[[int], None]] = None,
        **_: str,
    ) -> None:
        self.vla, self.device_id, self.stage = vla, device_id, stage
        self.vlm = vla.vlm
        # Get relevant VLM instance parameters before they get (potentially) wrapped
        self.all_module_keys, self.trainable_module_keys = self.vla.vlm.all_module_keys, self.vla.vlm.trainable_module_keys
        self.llm_transformer_layer_cls = self.vla.vlm.llm_backbone.transformer_layer_cls

        # Optimization Parameters
        self.epochs, self.max_steps = epochs, max_steps
        self.global_batch_size, self.per_device_batch_size = global_batch_size, per_device_batch_size

        self.learning_rate, self.weight_decay, self.max_grad_norm = learning_rate, weight_decay, max_grad_norm
        self.lr_scheduler_type, self.warmup_ratio = lr_scheduler_type, warmup_ratio

        # Generic Strategy Parameters
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.enable_mixed_precision_training = enable_mixed_precision_training
        self.reduce_in_full_precision = reduce_in_full_precision
        self.mixed_precision_dtype = mixed_precision_dtype

        # DataLoader Parameters
        self.worker_init_fn = worker_init_fn

        # Optimizers & Scheduler (initialized in `run_setup`)
        self.optimizer, self.lr_scheduler = None, None

        # Lightweight Validation
        assert (
            self.global_batch_size % self.per_device_batch_size == 0
        ), "Per-device batch size must evenly divide global batch size!"
        self.grad_accumulation_steps = self.global_batch_size // self.per_device_batch_size // overwatch.world_size()
        if self.enable_mixed_precision_training:
            assert self.mixed_precision_dtype == torch.bfloat16, "Only BF16 mixed precision training is supported!"
            assert check_bloat16_supported(), "BFloat16 is not supported on this hardware; unset `mixed_precision`"

    @abstractmethod
    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: Optional[float] = None,
        only_trainable: bool = True,
    ) -> None: ...

    @abstractmethod
    def run_setup(self, run_dir: Path, n_train_examples: int) -> None: ...

    @abstractmethod
    def clip_grad_norm(self) -> None: ...

    def run_training(
        self,
        dataset: Dataset,
        collator: PaddedCollatorForLanguageModeling,
        metrics: Metrics,
        stage: str = "finetune",
        batch_construction_strategy: str = "split-modality",
        seed: int = 7,
    ) -> None:
        """Run the training loop for the given `dataset` and `collator`; log losses, results to `metrics`"""
        if "finetune" in stage and batch_construction_strategy == "split-modality":
            # Instantiate the split-modality sampler; if you want to extend with other batch construction schemes,
            #   (e.g., grouping by length) =>> can easily add them here!
            modality_lengths = dataset.get_modality_lengths()
            sampler = SplitModalitySampler(
                dataset,
                modality_lengths,
                global_batch_size=self.global_batch_size,
                num_replicas=overwatch.world_size(),
                rank=overwatch.rank(),
                seed=seed,
                drop_last=False,
            )

        else:
            sampler = DistributedSampler(
                dataset,
                num_replicas=overwatch.world_size(),
                rank=overwatch.rank(),
                shuffle=True,
                seed=seed,
                drop_last=False,
            )

        # Create a DataLoader with the initialized sampler, per-device-bsz, and collator
        dataloader = DataLoader(
            dataset,
            batch_size=self.per_device_batch_size,
            sampler=sampler,
            collate_fn=collator,
            num_workers=2,
            worker_init_fn=self.worker_init_fn,
        )

        # Max Steps vs. Epochs Computation
        steps_per_epoch = len(dataloader) // self.grad_accumulation_steps
        if self.max_steps is not None and steps_per_epoch < self.max_steps:
            # Just set `epochs` to some large number --> we'll short-circuit based on steps anyway
            self.epochs = 100

        # === Train ===
        status = metrics.get_status()
        with tqdm(
            total=(
                (self.epochs * (len(dataloader) // self.grad_accumulation_steps))
                if self.max_steps is None
                else self.max_steps
            ),
            desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            for epoch in range(self.epochs):
                self.vla.train()
                sampler.set_epoch(epoch)

                # Zero-Gradients (just in case)
                self.optimizer.zero_grad()

                # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
                #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!
                for train_idx, batch in enumerate(dataloader):
                    # [Contract] self.vlm.forward() must automatically compute `loss` and return!
                    with torch.autocast(
                        "cuda",
                        dtype=self.mixed_precision_dtype,
                        enabled=self.enable_mixed_precision_training,
                    ):
                        output: CausalLMOutputWithPast = self.vlm(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            pixel_values=batch["pixel_values"],
                            labels=batch["labels"],
                            multimodal_indices=batch["multimodal_indices"],
                        )
                        loss = output.loss

                    # Commit Loss (Prior to Gradient Accumulation Normalization)
                    metrics.commit(loss=loss)

                    # Normalize Loss to account for Gradient Accumulation --> Backward!
                    # [IMPORTANT] Technically speaking, doing gradient accumulation in this way is "incorrect"; this is
                    #             because in general, each batch has a *different number of masked out tokens* (because
                    #             we're instruct-tuning). Taking the mean over two unbalanced means != the right thing!
                    #
                    #             HOWEVER -- at least at the 7B scale, the "naive" approach is just as performant as
                    #             the "correct" implementation, without adding extra complexity.
                    #
                    # That being said =>> at the 13B scale, *no matter what we tried, ANY gradient accumulation is just
                    #   really bad for downstream performance. Initial investigation shows that BF16 accumulation
                    #   just really tanks in precision... and don't have a good/clean way to fix this. Would love for
                    #   someone to PR and fix this (and I'd greatly appreciate it!!!)
                    normalized_loss = loss / self.grad_accumulation_steps
                    normalized_loss.backward()

                    # Step =>> Only if Done w/ Gradient Accumulation
                    if (train_idx + 1) % self.grad_accumulation_steps == 0:
                        metrics.commit(update_step_time=True)

                        # Clip Gradients --> this is custom, per-strategy because of DDP vs. FSDP locality-assumptions
                        self.clip_grad_norm()

                        # Optimizer & LR Scheduler Step
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()

                        # Push Metrics
                        metrics.commit(global_step=metrics.global_step + 1, lr=self.lr_scheduler.get_last_lr()[0])
                        status = metrics.push()

                        # Check for Termination & Save Final Checkpoint (in case `max_steps` is not None)
                        if self.max_steps is not None and metrics.global_step >= self.max_steps:
                            self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                            dist.barrier()

                            return

                        # Update Progress Bar
                        progress.update()
                        progress.set_description(status)

            # Save checkpoint at end each epoch (if `self.max_steps` is None)
            if self.max_steps is None:
                self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                dist.barrier()

    # === VLA Training ===

    def run_vla_training(
        self,
        vla_dataset: IterableDataset,
        collator: PaddedCollatorForActionPrediction,
        action_tokenizer: ActionTokenizer,
        metrics: VLAMetrics,
        save_interval: int = 2500,
        save_full_model: bool = True,
    ) -> None:
        """Run the VLA training loop for the given `dataset` and `collator`; log losses, action metrics to `metrics`."""
        assert isinstance(vla_dataset, IterableDataset), "VLA training expects an IterableDataset!"
        assert self.grad_accumulation_steps == 1, "VLA training does not support gradient accumulation!"

        # Create a DataLoader =>> Set `num_workers` to 0; RLDS loader handles parallelism!
        dataloader = DataLoader(
            vla_dataset,
            batch_size=self.per_device_batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,
            worker_init_fn=self.worker_init_fn,
        )

        # === Train ===
        status = metrics.get_status()
        with tqdm(
            total=(self.epochs * len(dataloader)) if self.max_steps is None else self.max_steps,
            desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            self.vlm.train()

            # Zero Gradients (just in case)
            self.optimizer.zero_grad()

            # [Contract] DataLoader wraps RLDS Loader (`.as_numpy_iterator() =>> implicit `.repeat()`)
            #   => This means looping over the DataLoader is basically "infinite" (so no outer loop over epochs).
            #      Slightly breaks default PyTorch semantics, which is why we adaptively compute `epoch` below.
            for batch in dataloader:
                # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
                #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!
                with torch.autocast(
                    "cuda", dtype=self.mixed_precision_dtype, enabled=self.enable_mixed_precision_training
                ):
                    # [Contract] self.vlm.forward() must automatically compute `loss` and return!
                    output: CausalLMOutputWithPast = self.vlm(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        pixel_values=batch["pixel_values"],
                        labels=batch["labels"],
                    )
                    loss = output.loss

                # Commit Loss =>> Backward!
                metrics.commit(loss=loss)
                loss.backward()

                # === Compute Action Token Accuracy & L1 Loss ===

                # To compute action token accuracy, we need to identify the locations of the action tokens
                # in both `output.logits` and `batch["labels"]`. We know that when "right" padding, we
                # insert `self.vlm.vision_backbone.num_patches` at index 1.
                #
                # Computing `action_prediction_accuracy` is then pretty straightforward:
                #   1) Extract "aligned" predictions & labels
                #   2) Compute boolean "mask" where "labels > 2" (where 2 is ID for `EOS_TOKEN`)
                #           => If masking out EOS, then it's just "labels != -100 (IGNORE_INDEX)
                #   3) Compute masked accuracy as `(preds == logits) & mask` --> sum/divide by # unmasked!

                token_preds = output.logits[:, self.vlm.vision_backbone.num_patches : -1].argmax(dim=2)
                token_gt = batch["labels"][:, 1:].to(token_preds.device)
                action_mask = token_gt > action_tokenizer.action_token_begin_idx

                def get_masks(tokens, tags):
                    tag_tokens = dict()

                    for tag in tags:
                        encoded_tags = self.vlm.llm_backbone.tokenizer.encode_plus(tag, return_tensors="pt")
                        tag_ids = encoded_tags["input_ids"][0]
                        tag_tokens[tag] = tag_ids[1:].to(tokens.device)

                    tag_masks = dict()
                    prev_tag = None
                    prev_pos = 0

                    def make_mask(a, b):
                        mask = torch.zeros_like(tokens)
                        mask[a:b] = 1
                        return mask

                    for i in range(len(tokens) - 1):
                        for tag, tag_ids in tag_tokens.items():
                            if i + len(tag_ids) > len(tokens):
                                continue

                            if torch.all(tokens[i : i + len(tag_ids)] == tag_ids):
                                tag_masks[prev_tag] = make_mask(prev_pos, i)
                                prev_tag = tag
                                prev_pos = i + len(tag_ids)

                    tag_masks[prev_tag] = make_mask(prev_pos, len(tokens))

                    for tag in tags:
                        if tag not in tag_masks:
                            tag_masks[tag] = make_mask(0, 0)

                    return tag_masks

                prompt_tags = get_cot_tags_list()

                def get_final_masks(tokens, tags):
                    final_masks = {tag: [] for tag in tags}

                    for group in tokens:
                        group_masks = get_masks(group, tags)

                        for tag in tags:
                            final_masks[tag].append(group_masks[tag])

                    for tag in tags:
                        final_masks[tag] = torch.stack(final_masks[tag], dim=0)

                    return final_masks

                # Dense reasoning metrics
                if metrics.global_step % 100 == 0:
                    final_pred_masks = get_final_masks(token_preds, prompt_tags)
                    final_gt_masks = get_final_masks(token_gt, prompt_tags)

                    # Compute accuracy for each tag
                    for tag in prompt_tags:
                        correct_tags = [0, 0]

                        for reasoning_pred, mask_pred, reasoning_gt, mask_gt in zip(
                            token_preds, final_pred_masks[tag], token_gt, final_gt_masks[tag]
                        ):
                            tag_pred = torch.masked_select(reasoning_pred, mask_pred.bool())
                            tag_gt = torch.masked_select(reasoning_gt, mask_gt.bool())

                            max_size = max(len(tag_pred), len(tag_gt))
                            tag_pred = torch.nn.functional.pad(tag_pred, (0, max_size - len(tag_pred)))
                            tag_gt = torch.nn.functional.pad(tag_gt, (0, max_size - len(tag_gt)))

                            correct_tags[0] += (tag_pred == tag_gt).sum().float()
                            correct_tags[1] += len(tag_gt)

                        if correct_tags[1] > 0:
                            tag_accuracy = correct_tags[0] / correct_tags[1]
                            metrics.commit(**{f"{tag[:-1].lower()}_tag_accuracy": tag_accuracy})

                # Compute Accuracy
                correct_action_preds = (token_preds == token_gt) & action_mask
                action_accuracy = correct_action_preds.sum().float() / action_mask.sum().float()

                # Compute L1 Loss on Predicted (Continuous) Actions
                continuous_actions_pred = torch.tensor(
                    action_tokenizer.decode_token_ids_to_actions(token_preds[action_mask].cpu().numpy())
                )
                continuous_actions_gt = torch.tensor(
                    action_tokenizer.decode_token_ids_to_actions(token_gt[action_mask].cpu().numpy())
                )
                action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)
                
                # === Action Head Metrics (if enabled) ===
                if hasattr(output, 'l1_loss') and hasattr(output, 'token_loss'):
                    metrics.commit(token_loss_separate=output.token_loss if isinstance(output.token_loss, (int, float)) else output.token_loss.item())
                    metrics.commit(action_head_l1=output.l1_loss.item())

                # Compute Accuracy on non-action (ie CoT) tokens
                cot_mask = (token_gt > EOS_TOKEN) & ~action_mask
                correct_cot_preds = (token_preds == token_gt) & cot_mask
                cot_accuracy = correct_cot_preds.sum().float() / cot_mask.sum().float()

                # Commit Metrics
                metrics.commit(
                    action_accuracy=action_accuracy,
                    cot_accuracy=cot_accuracy,
                    l1_loss=action_l1_loss,
                    update_step_time=True,
                )

                # Compute metrics per dataset --> only on rank_zero since we don't log them on other workers anyway
                if overwatch.is_rank_zero():
                    datasets = set(batch["dataset_names"])
                    if len(datasets) > 1:
                        for ds in datasets:
                            ds_mask = torch.tensor([elem == ds for elem in batch["dataset_names"]])
                            action_accuracy_ds = (
                                correct_action_preds[ds_mask].sum().float() / action_mask[ds_mask].sum().float()
                            )
                            continuous_actions_pred_ds = torch.tensor(
                                action_tokenizer.decode_token_ids_to_actions(
                                    token_preds[ds_mask][action_mask[ds_mask]].cpu().numpy()
                                )
                            )
                            continuous_actions_gt_ds = torch.tensor(
                                action_tokenizer.decode_token_ids_to_actions(
                                    token_gt[ds_mask][action_mask[ds_mask]].cpu().numpy()
                                )
                            )
                            action_l1_loss_ds = torch.nn.functional.l1_loss(
                                continuous_actions_pred_ds, continuous_actions_gt_ds
                            )
                            metrics.commit_for_dataset(
                                dataset_name=ds.decode(), action_accuracy=action_accuracy_ds, l1_loss=action_l1_loss_ds
                            )

                # === Gradient Step ===

                # Clip Gradients --> this is custom, per-strategy because of DDP vs. FSDP locality assumptions
                self.clip_grad_norm()

                # Optimizer & LR Scheduler Step
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                # Compute epoch value using number of completed gradient steps
                epoch = (metrics.global_step + 1) // (len(vla_dataset) // self.global_batch_size)

                # Push Metrics
                metrics.commit(global_step=metrics.global_step + 1, epoch=epoch, lr=self.lr_scheduler.get_last_lr()[0])
                status = metrics.push()

                # Check for Save Interval or Max Steps & Save Checkpoint
                if (terminate := (self.max_steps is not None and metrics.global_step >= self.max_steps)) or (
                    (metrics.global_step % save_interval) == 0
                ):
                    self.save_checkpoint(
                        metrics.run_dir, metrics.global_step, epoch, loss.item(), only_trainable=not save_full_model
                    )
                    dist.barrier()

                    if terminate:
                        return

                # Update Progress Bar
                progress.update()
                progress.set_description(status)
                
    def run_vla_training_latent(
        self,
        vla_dataset: IterableDataset,
        collator: PaddedCollatorForActionPrediction,
        action_tokenizer: ActionTokenizer,
        metrics: VLAMetrics,
        save_interval: int = 25000,
        save_full_model: bool = True,
        scheduled_stage: int = 0,
        start_thinking_id: int = 0,
        end_thinking_id: int = 0,
        latent_token_id: int = 0,
        cfg: Optional[Any] = None,
    ) -> None:
        """Run the VLA training loop with stage-aware latent reasoning for the given `dataset` and `collator`."""
        assert isinstance(vla_dataset, IterableDataset), "VLA training expects an IterableDataset!"
        assert self.grad_accumulation_steps == 1, "VLA training does not support gradient accumulation!"
        
        overwatch.info(f"Starting VLA Latent Training for Stage {scheduled_stage}")
        
        # Create a DataLoader =>> Set `num_workers` to 0; RLDS loader handles parallelism!
        dataloader = DataLoader(
            vla_dataset,
            batch_size=self.per_device_batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,
            worker_init_fn=self.worker_init_fn,
        )
        
        # === Train ===
        status = metrics.get_status()
        
        # Calculate steps per epoch based on dataset size and global batch size
        dataset_length = len(vla_dataset)
        steps_per_epoch = dataset_length // self.global_batch_size
        overwatch.info(f"Dataset length: {dataset_length}, Global batch size: {self.global_batch_size}, Steps per epoch: {steps_per_epoch}")
        
        ratio_per_stage = [0.4,0.6,1,0.7,0.5,0.7,0.5,0.5]
        total_steps_for_stage = int(cfg.vla.epochs_per_stage * steps_per_epoch )
        # total_steps_for_stage = 1000# for test
        if scheduled_stage == 0:
            total_steps_for_stage = 500
        else:
            total_steps_for_stage = int(total_steps_for_stage * ratio_per_stage[scheduled_stage-1])
        with tqdm(
            total=total_steps_for_stage,
            desc=f"{status} (Stage {scheduled_stage})",
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            self.vla.train()

            # Zero Gradients (just in case)
            self.optimizer.zero_grad()

            # Run for epochs_per_stage epochs
            step_count = 0
            final_first_thinking_pos = 0
            for batch in dataloader:
                if step_count >= total_steps_for_stage:
                    sleep(30)
                    break
                # input_ids = batch["input_ids"]
                # thinking_token_mask = (input_ids == latent_token_id)
                # first_thinking_pos = thinking_token_mask.nonzero(as_tuple=True)[1][0].item()
                # final_first_thinking_pos = max(final_first_thinking_pos, first_thinking_pos)
                # rank = dist.get_rank()
                # # print("[Rank {rank}] rank",rank)
                print("step_count",step_count,scheduled_stage)
                # print("[Rank {rank}] final_first_thinking_pos",final_first_thinking_pos)
                # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
                #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!
                with torch.autocast(
                    "cuda", dtype=self.mixed_precision_dtype, enabled=self.enable_mixed_precision_training
                ):
                    # [Contract] self.vlm.forward() must automatically compute `loss` and return!
                    # Pass scheduled_stage to the model for stage-aware processing
                    output: CausalLMOutputWithPast = self.vla(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        pixel_values=batch["pixel_values"],
                        labels=batch["labels"],
                        actions=batch.get("actions", None),  # ⭐ For Action Head (if enabled)
                        scheduled_stage=scheduled_stage,  #     Pass stage to model
                        start_thinking_id=start_thinking_id,
                        end_thinking_id=end_thinking_id,
                        latent_token_id=latent_token_id,
                        cfg=cfg,
                    )
                    loss = output.loss

                # Commit Loss =>> Backward!
                metrics.commit(loss=loss)
                loss.backward()

                # === Compute Action Token Accuracy & L1 Loss ===

                # To compute action token accuracy, we need to identify the locations of the action tokens
                # in both `output.logits` and `batch["labels"]`. We know that when "right" padding, we
                # insert `self.vlm.vision_backbone.num_patches` at index 1.
                #
                # Computing `action_prediction_accuracy` is then pretty straightforward:
                #   1) Extract "aligned" predictions & labels
                #   2) Compute boolean "mask" where "labels > 2" (where 2 is ID for `EOS_TOKEN`)
                #           => If masking out EOS, then it's just "labels != -100 (IGNORE_INDEX)
                #   3) Compute masked accuracy as `(preds == logits) & mask` --> sum/divide by # unmasked!

                token_preds = output.logits[:, self.vlm.vision_backbone.num_patches : -1].argmax(dim=2)
                token_gt = batch["labels"][:, 1:].to(token_preds.device)
                action_mask = token_gt > action_tokenizer.action_token_begin_idx

                def get_masks(tokens, tags):
                    tag_tokens = dict()

                    for tag in tags:
                        encoded_tags = self.vlm.llm_backbone.tokenizer.encode_plus(tag, return_tensors="pt")
                        tag_ids = encoded_tags["input_ids"][0]
                        tag_tokens[tag] = tag_ids[1:].to(tokens.device)

                    tag_masks = dict()
                    prev_tag = None
                    prev_pos = 0

                    def make_mask(a, b):
                        mask = torch.zeros_like(tokens)
                        mask[a:b] = 1
                        return mask

                    for i in range(len(tokens) - 1):
                        for tag, tag_ids in tag_tokens.items():
                            if i + len(tag_ids) > len(tokens):
                                continue

                            if torch.all(tokens[i : i + len(tag_ids)] == tag_ids):
                                tag_masks[prev_tag] = make_mask(prev_pos, i)
                                prev_tag = tag
                                prev_pos = i + len(tag_ids)

                    tag_masks[prev_tag] = make_mask(prev_pos, len(tokens))

                    for tag in tags:
                        if tag not in tag_masks:
                            tag_masks[tag] = make_mask(0, 0)

                    return tag_masks

                prompt_tags = get_cot_tags_list()

                def get_final_masks(tokens, tags):
                    final_masks = {tag: [] for tag in tags}

                    for group in tokens:
                        group_masks = get_masks(group, tags)

                        for tag in tags:
                            final_masks[tag].append(group_masks[tag])

                    for tag in tags:
                        final_masks[tag] = torch.stack(final_masks[tag], dim=0)

                    return final_masks

                # Dense reasoning metrics
                if metrics.global_step % 100 == 0:
                    final_pred_masks = get_final_masks(token_preds, prompt_tags)
                    final_gt_masks = get_final_masks(token_gt, prompt_tags)

                    # Compute accuracy for each tag
                    for tag in prompt_tags:
                        correct_tags = [0, 0]

                        for reasoning_pred, mask_pred, reasoning_gt, mask_gt in zip(
                            token_preds, final_pred_masks[tag], token_gt, final_gt_masks[tag]
                        ):
                            tag_pred = torch.masked_select(reasoning_pred, mask_pred.bool())
                            tag_gt = torch.masked_select(reasoning_gt, mask_gt.bool())

                            max_size = max(len(tag_pred), len(tag_gt))
                            tag_pred = torch.nn.functional.pad(tag_pred, (0, max_size - len(tag_pred)))
                            tag_gt = torch.nn.functional.pad(tag_gt, (0, max_size - len(tag_gt)))

                            correct_tags[0] += (tag_pred == tag_gt).sum().float()
                            correct_tags[1] += len(tag_gt)

                        if correct_tags[1] > 0:
                            tag_accuracy = correct_tags[0] / correct_tags[1]
                            metrics.commit(**{f"{tag[:-1].lower()}_tag_accuracy": tag_accuracy})

                # Compute Accuracy
                correct_action_preds = (token_preds == token_gt) & action_mask
                action_accuracy = correct_action_preds.sum().float() / action_mask.sum().float()

                # Compute L1 Loss on Predicted (Continuous) Actions
                continuous_actions_pred = torch.tensor(
                    action_tokenizer.decode_token_ids_to_actions(token_preds[action_mask].cpu().numpy())
                )
                continuous_actions_gt = torch.tensor(
                    action_tokenizer.decode_token_ids_to_actions(token_gt[action_mask].cpu().numpy())
                )
                action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)
                
                # === Action Head Metrics (if enabled) ===
                if hasattr(output, 'l1_loss') and hasattr(output, 'token_loss'):
                    metrics.commit(token_loss_separate=output.token_loss if isinstance(output.token_loss, (int, float)) else output.token_loss.item())
                    metrics.commit(action_head_l1=output.l1_loss.item())

                # Compute Accuracy on non-action (ie CoT) tokens
                cot_mask = (token_gt > EOS_TOKEN) & ~action_mask
                correct_cot_preds = (token_preds == token_gt) & cot_mask
                cot_accuracy = correct_cot_preds.sum().float() / cot_mask.sum().float()

                # Commit Metrics with stage information
                metrics.commit(
                    action_accuracy=action_accuracy,
                    cot_accuracy=cot_accuracy,
                    l1_loss=action_l1_loss,
                    update_step_time=True,
                )

                # Compute metrics per dataset --> only on rank_zero since we don't log them on other workers anyway
                if overwatch.is_rank_zero():
                    datasets = set(batch["dataset_names"])
                    if len(datasets) > 1:
                        for ds in datasets:
                            ds_mask = torch.tensor([elem == ds for elem in batch["dataset_names"]])
                            action_accuracy_ds = (
                                correct_action_preds[ds_mask].sum().float() / action_mask[ds_mask].sum().float()
                            )
                            continuous_actions_pred_ds = torch.tensor(
                                action_tokenizer.decode_token_ids_to_actions(
                                    token_preds[ds_mask][action_mask[ds_mask]].cpu().numpy()
                                )
                            )
                            continuous_actions_gt_ds = torch.tensor(
                                action_tokenizer.decode_token_ids_to_actions(
                                    token_gt[ds_mask][action_mask[ds_mask]].cpu().numpy()
                                )
                            )
                            action_l1_loss_ds = torch.nn.functional.l1_loss(
                                continuous_actions_pred_ds, continuous_actions_gt_ds
                            )
                            metrics.commit_for_dataset(
                                dataset_name=ds.decode(), action_accuracy=action_accuracy_ds, l1_loss=action_l1_loss_ds
                            )

                # === Gradient Step ===

                # Clip Gradients --> this is custom, per-strategy because of DDP vs. FSDP locality assumptions
                self.clip_grad_norm()

                # Optimizer & LR Scheduler Step
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                # Compute epoch value using number of completed gradient steps
                epoch = (metrics.global_step + 1) // (len(vla_dataset) // self.global_batch_size)

                # Push Metrics
                metrics.commit(global_step=metrics.global_step + 1, epoch=epoch, lr=self.lr_scheduler.get_last_lr()[0])
                status = metrics.push()

                # Check for Save Interval or Max Steps & Save Checkpoint
                if (terminate := (self.max_steps is not None and metrics.global_step >= self.max_steps)) or (
                    (metrics.global_step % save_interval) == 0
                ):
                    self.save_checkpoint(
                        metrics.run_dir, metrics.global_step, epoch, loss.item(), only_trainable=not save_full_model
                    )
                    dist.barrier()

                    if terminate:
                        return

                # Update Progress Bar
                progress.update()
                progress.set_description(f"{status} (Stage {scheduled_stage})")
                
                # Increment step count
                step_count += 1
            