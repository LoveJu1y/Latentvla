#!/usr/bin/env bash
# ============================================================================
# ECoT Stage 2+ Training Script (8 GPUs)
# 使用方法：直接修改下面的配置，然后运行 bash scripts/run_ecot_stage2_8gpu.sh
# ============================================================================

set -e

# ============================================================================
# 环境变量
# ============================================================================
# NCCL 网络配置（单机多卡时，如果遇到 "no socket interface found" 错误，可以注释掉）
# 如果需要，请根据实际网络接口修改（使用 ifconfig 或 ip addr 查看）
# export NCCL_SOCKET_IFNAME=bond0
# export NCCL_IB_HCA=mlx5_2,mlx5_3

# export NCCL_BLOCKING_WAIT=1
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_TIMEOUT=1000
# export WANDB_MODE=offline
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="/share/project/lvjing/starVLA/qwen_cache"
export WANDB_API_KEY="a8989c35c0573184da807b8a781d72936fe7e379"
export TOKENIZERS_PARALLELISM="false"
export WANDB_BASE_URL="https://api.bandw.top"

# ============================================================================
# 配置参数 - 在这里修改配置
# ============================================================================

# 基础配置

scheduled_stage=4  # scheduled_stage 需要数字
max_train_steps=200000
num_gpus=8

RUN_ROOT_DIR="./outputs2"
RUN_ID="ecot_stage${scheduled_stage}_fianl_plus100k"
CONFIG_YAML="config/training/ecot_stage2_full.yaml"
OUTPUT_DIR=${RUN_ROOT_DIR}/${RUN_ID}
PRETRAINED_CHECKPOINT="/share/project/lvjing/starVLA/outputs2/ecot_stage4_fianl_plus60k/checkpoints/steps_20000_pytorch_model.pt"
mkdir -p ${OUTPUT_DIR}
# PRETRAINED_CHECKPOINT="/share/project/lvjing/starVLA/outputs/ecot_stage0_20251116_234857/checkpoints/steps_25000_pytorch_model.pt"
# 训练参数（使用 -- 格式）
TRAIN_CONFIG_ARGS=(
  --trainer.pretrained_checkpoint "${PRETRAINED_CHECKPOINT}"
  --trainer.max_train_steps ${max_train_steps}
  --trainer.save_interval 5000
  --trainer.logging_frequency 10
  --trainer.eval_interval 500
  --trainer.learning_rate.base 0.00001
  --trainer.gradient_accumulation_steps 1
  --framework.qwenvl.model_max_length 2048
  --framework.latent_reasoning.vlm_loss_weight 0.1
  --datasets.vla_data.per_device_batch_size 32
  --datasets.vla_data.ecot.scheduled_stage ${scheduled_stage}
  --datasets.vla_data.num_workers 0

)

# W&B 和数据配置（使用 -- 格式）
BASE_CONFIG_ARGS=(
  --wandb_project "Latent_qwengr00t_all_stage"
  --wandb_entity "lvj2114-beijing-academy-of-artificial-intelligence"
  --datasets.vla_data.ecot.data_root_dir "/share/project/emllm_mnt.1d/mnt/sfs/baaiei/jyShi/rt_newData"
  --datasets.vla_data.ecot.data_mix "bridge"
)


# ============================================================================
# 启动训练
# ============================================================================


echo "============================================================================"
echo "🚀 Starting ECoT Stage 2+ Training (8 GPUs)"
echo "============================================================================"
echo "Run ID: ${RUN_ID}"
echo "Output: ${OUTPUT_DIR}"
echo "============================================================================"
cp $0 ${OUTPUT_DIR}/
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${num_gpus} \
  starVLA/training/train_ecot.py \
  --config_yaml ${CONFIG_YAML} \
  --run_root_dir ${RUN_ROOT_DIR} \
  --run_id ${RUN_ID} \
  "${TRAIN_CONFIG_ARGS[@]}" \
  "${BASE_CONFIG_ARGS[@]}" 

echo ""
echo "✅ Training completed! Output: ${OUTPUT_DIR}"
