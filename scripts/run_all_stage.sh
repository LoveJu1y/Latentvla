#!/usr/bin/env bash
# ============================================================================
# ECoT Multi-Stage Training Script
# 依次运行 Stage 0 -> Stage 1 -> Stage 2 -> Stage 3 -> Stage 4
# 每个 stage 从前一个 stage 的指定 step checkpoint 继续训练
#
# Stage 说明：
#   - Stage 0: 全 CoT 阶段（完整推理链，无 thinking tokens）
#   - Stage 1-3: 逐步减少显式 CoT，增加隐式推理（thinking tokens）
#   - Stage 4: 最后阶段（最少或没有显式 CoT，主要隐式推理）
# ============================================================================

set -e

# ============================================================================
# 环境变量
# ============================================================================
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="/share/project/lvjing/starVLA/qwen_cache"
export WANDB_API_KEY="a8989c35c0573184da807b8a781d72936fe7e379"
export TOKENIZERS_PARALLELISM="false"
export WANDB_BASE_URL="https://api.bandw.top"

# ============================================================================
# 配置参数 - 在这里修改配置
# ============================================================================

# 基础配置
RUN_ROOT_DIR="./outputs1"
num_gpus=8
CONFIG_YAML="config/training/ecot_stage2_full.yaml"

# Stage 列表（按顺序执行）
STAGES=(0 1 2 3 4)

# Stage 配置：每个 stage 的训练步数和保存间隔
# Stage 0: 全 CoT 阶段（完整推理链）
# Stage 1-3: 逐步减少显式 CoT，增加隐式推理
# Stage 4: 最后一个阶段（最少或没有显式 CoT，主要隐式推理）
declare -A STAGE_STEPS=(
    [0]=30001   # Stage 0 训练步数（全 CoT）
    [1]=10001   # Stage 1 训练步数
    [2]=10001   # Stage 2 训练步数
    [3]=10001   # Stage 3 训练步数
    [4]=40000   # Stage 4 训练步数（最后阶段）
)

declare -A STAGE_SAVE_INTERVALS=(
    [0]=2500    # Stage 0 保存间隔
    [1]=2500    # Stage 1 保存间隔
    [2]=2500   # Stage 2 保存间隔
    [3]=2500    # Stage 3 保存间隔
    [4]=2500    # Stage 4 保存间隔
)

# 每个 stage 使用的 checkpoint step（从前一个 stage 加载）
# 格式：stage -> checkpoint_step（从前一个 stage 的哪个 step 加载）
declare -A STAGE_CHECKPOINT_STEPS=(
    [0]=""           # Stage 0 不使用 checkpoint（从头训练）
    [1]="25000"      # Stage 1 使用 Stage 0 的 steps_50000 checkpoint
    [2]="10000"      # Stage 2 使用 Stage 1 的 steps_10000 checkpoint
    [3]="10000"      # Stage 3 使用 Stage 2 的 steps_10000 checkpoint
    [4]="10000"      # Stage 4 使用 Stage 3 的 steps_10000 checkpoint
)

# ============================================================================
# 训练函数
# ============================================================================

run_stage() {
    local stage=$1
    local prev_checkpoint=$2  # 上一个 stage 的 checkpoint 路径（可选）
    
    echo ""
    echo "============================================================================"
    echo "🚀 Starting Stage ${stage} Training"
    echo "============================================================================"
    
    # 创建 stage 特定的 run ID（不使用日期，更简单直观）
    STAGE_RUN_ID="ecot_stage${stage}"
    STAGE_OUTPUT_DIR="${RUN_ROOT_DIR}/${STAGE_RUN_ID}"
    mkdir -p ${STAGE_OUTPUT_DIR}
    
    # 构建训练参数（与 run_ecot_8gpu.sh 保持一致）
    TRAIN_CONFIG_ARGS=(
        --trainer.max_train_steps ${STAGE_STEPS[$stage]}
        --trainer.save_interval ${STAGE_SAVE_INTERVALS[$stage]}
        --trainer.logging_frequency 10
        --trainer.eval_interval 500
        --trainer.learning_rate.base 0.00001
        --trainer.gradient_accumulation_steps 1
        --framework.qwenvl.model_max_length 2048
        --framework.latent_reasoning.vlm_loss_weight 0.2
        --datasets.vla_data.per_device_batch_size 16
        --datasets.vla_data.ecot.scheduled_stage ${stage}
        --datasets.vla_data.num_workers 0
    )
    
    # 如果提供了上一个 stage 的 checkpoint，添加预训练 checkpoint 参数
    if [ -n "${prev_checkpoint}" ] && [ -f "${prev_checkpoint}" ]; then
        TRAIN_CONFIG_ARGS+=(
            --trainer.pretrained_checkpoint "${prev_checkpoint}"
        )
        echo "📦 Loading checkpoint from previous stage: ${prev_checkpoint}"
    else
        echo "🆕 Starting from scratch (no previous checkpoint)"
    fi
    
    # W&B 和数据配置（与 run_ecot_8gpu.sh 保持一致）
    BASE_CONFIG_ARGS=(
        --wandb_project "Latent_qwengr00t_all_stage"
        --wandb_entity "lvj2114-beijing-academy-of-artificial-intelligence"
        --datasets.vla_data.ecot.data_root_dir "/share/project/emllm_mnt.1d/mnt/sfs/baaiei/jyShi/rt_newData"
        --datasets.vla_data.ecot.data_mix "bridge"
    )
    
    # 保存脚本副本
    cp $0 ${STAGE_OUTPUT_DIR}/
    
    # 启动训练
    accelerate launch \
        --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
        --num_processes ${num_gpus} \
        starVLA/training/train_ecot.py \
        --config_yaml ${CONFIG_YAML} \
        --run_root_dir ${RUN_ROOT_DIR} \
        --run_id ${STAGE_RUN_ID} \
        "${TRAIN_CONFIG_ARGS[@]}" \
        "${BASE_CONFIG_ARGS[@]}"
    
    # 检查训练是否成功
    TRAIN_EXIT_CODE=$?
    if [ ${TRAIN_EXIT_CODE} -ne 0 ]; then
        echo "❌ Stage ${stage} training failed!" >&2
        return ${TRAIN_EXIT_CODE}
    fi
    
    # 返回当前 stage 的输出目录，供下一个 stage 构建 checkpoint 路径
    # 使用特殊标记确保能被正确捕获
    echo "STAGE_OUTPUT_DIR:${STAGE_OUTPUT_DIR}"
}

# ============================================================================
# 主训练流程
# ============================================================================

echo "============================================================================"
echo "🎯 ECoT Multi-Stage Training Pipeline"
echo "============================================================================"
echo "Stages to run: ${STAGES[@]}"
echo "  - Stage 0: 全 CoT 阶段（完整推理链，无 thinking tokens）"
echo "  - Stage 1-3: 逐步减少显式 CoT，增加隐式推理（thinking tokens）"
echo "  - Stage 4: 最后阶段（最少或没有显式 CoT，主要隐式推理）"
echo "Number of GPUs: ${num_gpus}"
echo "Config: ${CONFIG_YAML}"
echo "Output directories: ${RUN_ROOT_DIR}/ecot_stage{0,1,2,3,4}"
echo "============================================================================"

# 记录开始时间
START_TIME=$(date +%s)

# 依次运行各个 stage
PREV_STAGE_OUTPUT_DIR=""
for i in "${!STAGES[@]}"; do
    stage=${STAGES[$i]}
    STAGE_START_TIME=$(date +%s)
    
    # 构建 checkpoint 路径（如果有上一个 stage）
    PREV_CHECKPOINT=""
    if [ -n "${PREV_STAGE_OUTPUT_DIR}" ] && [ -n "${STAGE_CHECKPOINT_STEPS[$stage]}" ]; then
        checkpoint_step=${STAGE_CHECKPOINT_STEPS[$stage]}
        PREV_CHECKPOINT="${PREV_STAGE_OUTPUT_DIR}/checkpoints/steps_${checkpoint_step}_pytorch_model.pt"
        
        # 检查 checkpoint 是否存在
        if [ ! -f "${PREV_CHECKPOINT}" ]; then
            echo "❌ Checkpoint not found: ${PREV_CHECKPOINT}"
            echo "   Please check if the previous stage completed successfully."
            exit 1
        fi
    fi
    
    # 运行当前 stage（将 stderr 重定向到 stdout，捕获带有特殊标记的输出目录）
    # 注意：run_stage 函数会在最后 echo "STAGE_OUTPUT_DIR:${STAGE_OUTPUT_DIR}" 到 stdout
    OUTPUT=$(run_stage ${stage} "${PREV_CHECKPOINT}" 2>&1)
    EXIT_CODE=$?
    
    # 从输出中提取输出目录（使用特殊标记）
    CURRENT_STAGE_OUTPUT_DIR=$(echo "${OUTPUT}" | grep "STAGE_OUTPUT_DIR:" | tail -1 | sed 's/.*STAGE_OUTPUT_DIR://')
    
    # 显示训练输出（除了最后一行）
    echo "${OUTPUT}" | grep -v "STAGE_OUTPUT_DIR:" || true
    
    # 检查训练是否成功
    if [ ${EXIT_CODE} -ne 0 ] || [ -z "${CURRENT_STAGE_OUTPUT_DIR}" ]; then
        echo "❌ Stage ${stage} training failed or output directory not found!"
        exit 1
    fi
    
    # 验证输出目录是否存在
    if [ ! -d "${CURRENT_STAGE_OUTPUT_DIR}" ]; then
        echo "❌ Stage ${stage} output directory not found: ${CURRENT_STAGE_OUTPUT_DIR}"
        exit 1
    fi
    
    # 更新上一个 stage 的输出目录
    PREV_STAGE_OUTPUT_DIR="${CURRENT_STAGE_OUTPUT_DIR}"
    
    # 计算 stage 耗时
    STAGE_END_TIME=$(date +%s)
    STAGE_DURATION=$((STAGE_END_TIME - STAGE_START_TIME))
    STAGE_HOURS=$((STAGE_DURATION / 3600))
    STAGE_MINS=$(((STAGE_DURATION % 3600) / 60))
    
    echo ""
    echo "⏱️  Stage ${stage} took ${STAGE_HOURS}h ${STAGE_MINS}m"
    echo "============================================================================"
done

# 计算总耗时
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
TOTAL_HOURS=$((TOTAL_DURATION / 3600))
TOTAL_MINS=$(((TOTAL_DURATION % 3600) / 60))

# 构建最终 checkpoint 路径（使用最后一个 stage 的最终训练步数）
FINAL_STAGE=${STAGES[-1]}
FINAL_CHECKPOINT_STEP=${STAGE_STEPS[$FINAL_STAGE]}
FINAL_CHECKPOINT="${PREV_STAGE_OUTPUT_DIR}/checkpoints/steps_${FINAL_CHECKPOINT_STEP}_pytorch_model.pt"

echo ""
echo "============================================================================"
echo "✅ All stages completed successfully!"
echo "============================================================================"
echo "Total training time: ${TOTAL_HOURS}h ${TOTAL_MINS}m"
echo "Final checkpoint: ${FINAL_CHECKPOINT}"
echo "============================================================================"
