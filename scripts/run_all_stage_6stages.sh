#!/usr/bin/env bash
# ============================================================================
# ECoT Multi-Stage Training Script (6 Stages)
# 依次运行 Stage 0 -> Stage 1 -> Stage 2 -> Stage 3 -> Stage 4 -> Stage 5 -> Stage 6
# 每个 stage 从前一个 stage 的指定 step checkpoint 继续训练
#
# Stage 说明：
#   - Stage 0-4: 纯推理训练阶段（reasoning_only）
#              - 逐步减少显式 CoT，增加隐式推理（thinking tokens）
#              - 只训练 VLM 的隐式推理能力
#              - Action head 参数冻结（requires_grad=False）
#              - Stage 0-3: batch_size=16
#              - Stage 4: batch_size=32，使用 Stage 4 数据配置
#   - Stage 5: 动作头专项训练（action_only）
#              - 使用 Stage 4 的数据配置（scheduled_stage=4）
#              - VLM 冻结，只训练 action head
#              - batch_size=32
#   - Stage 6: 最终联合微调（full）
#              - 使用 Stage 4 的数据配置（scheduled_stage=4）
#              - VLM + action head 一起微调
#              - batch_size=32
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
RUN_ROOT_DIR="./4B_train_vlm_only/outputs"
num_gpus=8
CONFIG_YAML="config/training/ecot_stage2_full.yaml"
WANDB_PROJECT="4B_Latent_qwengr00t_vlm_only1"
# ============================================================================
# 跳过 Stage 配置（可选）
# 如果想从某个 Stage 开始训练（跳过前面的 stages），可以设置：
# START_STAGE：从哪个 stage 开始训练（0-6）
# PREV_STAGE_CHECKPOINT：上一个 stage 的 checkpoint 路径
# 
# 例如：从 Stage 2 开始训练，跳过 Stage 0 和 Stage 1
# START_STAGE=2
# PREV_STAGE_CHECKPOINT="/path/to/stage1/checkpoint.pt"
# 
# 如果 START_STAGE=0，则从头开始训练所有 stages
# 请设置 START_STAGE > 0，并提供上一 Stage 的 checkpoint
# ============================================================================

START_STAGE=3
# 当 START_STAGE > 0 时，必须提供 PREV_STAGE_CHECKPOINT，用于初始化后续训练
PREV_STAGE_CHECKPOINT="/share/project/lvjing/starVLA/4B_train_vlm_only/outputs/ecot_stage2/checkpoints/steps_2500_pytorch_model.pt"
# 示例：从 Stage 2 开始训练，使用 Stage 1 的 checkpoint
# START_STAGE=2
# PREV_STAGE_CHECKPOINT="/share/project/lvjing/starVLA/train_6stages/outputs/ecot_stage1/checkpoints/steps_5000_pytorch_model.pt"

# ============================================================================
# 权重重载配置（架构变更时使用）
# ============================================================================
# 如果从旧架构（如 DiT-B）迁移到新架构（如 DiT-L），需要忽略形状不匹配的层
# 设置为 "qwen_vl_interface" 可只加载 VLM 权重；正常情况请留空 ""
RELOAD_MODULES=""

# 验证配置
ALL_STAGES=(0 1 2 3 4 5 6)
ALL_STAGES=(0 1 2 3 )
if [ ${START_STAGE} -lt 0 ] || [ ${START_STAGE} -gt 6 ]; then
    echo "❌ Error: START_STAGE must be between 0 and 6, got: ${START_STAGE}"
    exit 1
fi

if [ ${START_STAGE} -gt 0 ] && [ -z "${PREV_STAGE_CHECKPOINT}" ]; then
    echo "❌ Error: When START_STAGE > 0, PREV_STAGE_CHECKPOINT must be provided!"
    echo "   You want to start from Stage ${START_STAGE}, but no checkpoint is provided."
    echo "   Please set PREV_STAGE_CHECKPOINT to the checkpoint from Stage $((START_STAGE - 1))."
    exit 1
fi

# 构建需要训练的 Stage 列表
STAGES=()
for stage in "${ALL_STAGES[@]}"; do
    if [ ${stage} -ge ${START_STAGE} ]; then
        STAGES+=($stage)
    fi
done

if [ ${#STAGES[@]} -eq 0 ]; then
    echo "❌ Error: No stages to train!"
    exit 1
fi

echo "✅ Stages to train: ${STAGES[@]}"
if [ ${START_STAGE} -gt 0 ]; then
    echo "⏭️  Skipping stages 0-$((START_STAGE - 1))"
    echo "📦 Will use checkpoint: ${PREV_STAGE_CHECKPOINT}"
fi

# Stage 配置：每个 stage 的训练步数和保存间隔
# Stage 0-4: 纯推理训练（reasoning_only，只训练 VLM，逐步增加隐式推理）
#            Stage 0-3 使用 batch_size=16，Stage 4 使用 batch_size=32
# Stage 5: 动作头专项训练（action_only，VLM 冻结，只训练 action head，batch_size=32）
# Stage 6: 最终联合微调（full，VLM + action head 一起微调，batch_size=32）
declare -A STAGE_STEPS=(
    [0]=2500   # Stage 0 训练步数（纯推理训练，全 CoT）
    [1]=2500   # Stage 1 训练步数（纯推理训练）
    [2]=2500   # Stage 2 训练步数（纯推理训练）
    [3]=2500   # Stage 3 训练步数（纯推理训练）
    [4]=500   # Stage 4 训练步数（完整训练，VLM + action head）
    [5]=5000   # Stage 5 训练步数（action_only，只训练 action head）
    [6]=60000   # Stage 6 训练步数（最终联合微调）
)

declare -A STAGE_SAVE_INTERVALS=(
    [0]=2500    # Stage 0 保存间隔
    [1]=2500    # Stage 1 保存间隔
    [2]=2500    # Stage 2 保存间隔
    [3]=2500    # Stage 3 保存间隔
    [4]=500    # Stage 4 保存间隔
    [5]=2500    # Stage 5 保存间隔
    [6]=2500    # Stage 6 保存间隔
)

# 每个 stage 使用的 checkpoint step（从前一个 stage 加载）
# 格式：stage -> checkpoint_step（从前一个 stage 的哪个 step 加载）
declare -A STAGE_CHECKPOINT_STEPS=(
    [0]=""           # Stage 0 不使用 checkpoint（从头训练）
    [1]="2500"      # Stage 1 使用 Stage 0 的最终 checkpoint
    [2]="2500"      # Stage 2 使用 Stage 1 的最终 checkpoint
    [3]="2500"      # Stage 3 使用 Stage 2 的最终 checkpoint
    [4]="2500"      # Stage 4 使用 Stage 3 的最终 checkpoint
    [5]="500"      # Stage 5 使用 Stage 4 的最终 checkpoint
    [6]="60000"      # Stage 6 使用 Stage 5 的最终 checkpoint
)

# ============================================================================
# 训练函数
# ============================================================================

run_stage() {
    local stage=$1
    local prev_checkpoint=$2  # 上一个 stage 的 checkpoint 路径（可选）
    local reload_modules=$3   # [New] 仅重载指定模块（用于架构变更时的部分加载）
    
    echo ""
    echo "============================================================================"
    echo "🚀 Starting Stage ${stage} Training"
    echo "============================================================================"
    
    # 创建 stage 特定的 run ID（不使用日期，更简单直观）
    STAGE_RUN_ID="ecot_stage${stage}"
    STAGE_OUTPUT_DIR="${RUN_ROOT_DIR}/${STAGE_RUN_ID}"
    mkdir -p ${STAGE_OUTPUT_DIR}
    
    # 根据 stage 决定训练阶段模式和数据配置
    # Stage 0-4: 纯推理训练（只训练 VLM，冻结 action head）
    # Stage 5: action_only（VLM 冻结，只训练 action head，使用 stage 4 的数据）
    # Stage 6: 最终联合微调（VLM + action head，使用 stage 4 的数据）
    if [ ${stage} -lt 5 ]; then
        TRAINING_STAGE="reasoning_only"
        VLM_LOSS_WEIGHT=1.0  # Stage 0-3 纯推理，权重设为 1.0
        DATA_STAGE=${stage}   # 数据配置跟随实际 stage
        echo "🧠 [Stage ${stage}] Reasoning-only mode: Training VLM reasoning, action_model frozen"
    elif [ ${stage} -eq 5 ]; then
        TRAINING_STAGE="action_only"
        VLM_LOSS_WEIGHT=0.0  # action_only 模式不需要 vlm_loss
        DATA_STAGE=4  # 使用 Stage 4 的数据配置
        echo "🔧 [Stage ${stage}] Action-only mode: VLM frozen, training action_model only (using Stage 4 data)"
    else
        TRAINING_STAGE="full"
        VLM_LOSS_WEIGHT=0.1  # Stage 6 最终微调
        DATA_STAGE=4  # 使用 Stage 4 的数据配置
        echo "🎯 [Stage ${stage}] Final fine-tuning: VLM + action_model both trainable (using Stage 4 data)"
    fi
    
    # 根据 stage 决定 batch size
    # Stage 0-3: batch_size=12
    # Stage 4-6: batch_size=16
    if [ ${stage} -ge 3 ]; then
        BATCH_SIZE=16
    else
        BATCH_SIZE=12
    fi
    echo "📊 Using batch size: ${BATCH_SIZE} (Stage ${stage})"
    
    # 构建训练参数（与 run_ecot_8gpu.sh 保持一致）
    TRAIN_CONFIG_ARGS=(
        --trainer.max_train_steps ${STAGE_STEPS[$stage]}
        --trainer.save_interval ${STAGE_SAVE_INTERVALS[$stage]}
        --trainer.logging_frequency 10
        --trainer.eval_interval 60000000
        --trainer.learning_rate.base 3.0e-5
        --trainer.learning_rate.action_model 5.0e-5
        --trainer.gradient_accumulation_steps 1
        --framework.qwenvl.model_max_length 2048
        --framework.action_model.action_model_type DiT-L
        --framework.action_model.diffusion_model_cfg.num_layers 16
        --framework.training_stage ${TRAINING_STAGE}
        --framework.latent_reasoning.vlm_loss_weight ${VLM_LOSS_WEIGHT}
        --datasets.vla_data.per_device_batch_size ${BATCH_SIZE}
        --datasets.vla_data.ecot.scheduled_stage ${DATA_STAGE}
        --datasets.vla_data.num_workers 0
    )
    
    # 如果提供了上一个 stage 的 checkpoint，添加预训练 checkpoint 参数
    if [ -n "${prev_checkpoint}" ] && [ -f "${prev_checkpoint}" ]; then
        TRAIN_CONFIG_ARGS+=(
            --trainer.pretrained_checkpoint "${prev_checkpoint}"
        )
        # 如果指定了部分加载（例如只加载 VLM），添加参数
        if [ -n "${reload_modules}" ]; then
            TRAIN_CONFIG_ARGS+=( --trainer.reload_modules "${reload_modules}" )
            echo "⚠️  Partial load enabled: Only reloading modules: ${reload_modules}"
        fi
        echo "📦 Loading checkpoint from previous stage: ${prev_checkpoint}"
    else
        echo "🆕 Starting from scratch (no previous checkpoint)"
    fi
    
    # W&B 和数据配置（与 run_ecot_8gpu.sh 保持一致）
    BASE_CONFIG_ARGS=(
        --wandb_project ${WANDB_PROJECT}
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
    
    echo ""
    echo "✅ Stage ${stage} training completed successfully!"
    echo "📁 Output directory: ${STAGE_OUTPUT_DIR}"
}

# ============================================================================
# 主训练流程
# ============================================================================

echo "============================================================================"
echo "🎯 ECoT 6-Stage Training Pipeline"
echo "============================================================================"
echo "Stages to run: ${STAGES[@]}"
if [ ${START_STAGE} -gt 0 ]; then
    echo "  - Skipped stages: 0-$((START_STAGE - 1)) (using checkpoint from Stage $((START_STAGE - 1)))"
fi
echo "  - Stage 0-4: 纯推理训练（reasoning_only，只训练 VLM，action head 冻结）"
echo "               Stage 0-3: batch_size=16，Stage 4: batch_size=32"
echo "  - Stage 5: 动作头专项训练（action_only，VLM 冻结，batch_size=32）"
echo "  - Stage 6: 最终联合微调（full，VLM + action head，batch_size=32）"
echo "Number of GPUs: ${num_gpus}"
echo "Config: ${CONFIG_YAML}"
echo "Output directories: ${RUN_ROOT_DIR}/ecot_stage{${STAGES[@]}}"
echo "============================================================================"

# 记录开始时间
START_TIME=$(date +%s)

# 依次运行各个 stage
PREV_STAGE_OUTPUT_DIR=""
for i in "${!STAGES[@]}"; do
    stage=${STAGES[$i]}
    STAGE_START_TIME=$(date +%s)
    
    # 调试信息：显示当前正在处理的 stage
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📍 [Main Loop] Processing Stage ${stage} (index ${i})"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # 构建 checkpoint 路径（如果有上一个 stage）
    PREV_CHECKPOINT=""
    
    # 情况1：如果是第一个要训练的 stage，且 START_STAGE > 0，使用提供的 checkpoint
    if [ ${i} -eq 0 ] && [ ${START_STAGE} -gt 0 ] && [ -n "${PREV_STAGE_CHECKPOINT}" ]; then
        PREV_CHECKPOINT="${PREV_STAGE_CHECKPOINT}"
        echo "🔍 Using provided checkpoint from Stage $((START_STAGE - 1)): ${PREV_CHECKPOINT}"
        
        # 检查 checkpoint 是否存在
        if [ ! -f "${PREV_CHECKPOINT}" ]; then
            echo "❌ Checkpoint not found: ${PREV_CHECKPOINT}"
            echo "   Please check the checkpoint path."
            exit 1
        fi
        echo "✅ Checkpoint found"
    # 情况2：正常情况，从前一个 stage 的输出目录加载 checkpoint
    elif [ -n "${PREV_STAGE_OUTPUT_DIR}" ] && [ -n "${STAGE_CHECKPOINT_STEPS[$stage]}" ]; then
        checkpoint_step=${STAGE_CHECKPOINT_STEPS[$stage]}
        PREV_CHECKPOINT="${PREV_STAGE_OUTPUT_DIR}/checkpoints/steps_${checkpoint_step}_pytorch_model.pt"
        
        echo "🔍 Checking for checkpoint: ${PREV_CHECKPOINT}"
        
        # 检查 checkpoint 是否存在
        if [ ! -f "${PREV_CHECKPOINT}" ]; then
            echo "❌ Checkpoint not found: ${PREV_CHECKPOINT}"
            echo "   Please check if the previous stage completed successfully."
            exit 1
        fi
        echo "✅ Checkpoint found"
    else
        echo "ℹ️  No previous checkpoint (starting from scratch)"
    fi
    
    echo "🚀 Now calling run_stage function..."
    
    # 判断是否需要 reload_modules (仅在第一个运行的 stage 且有外部 checkpoint 时使用)
    RELOAD_OPT=""
    if [ "${stage}" == "${STAGES[0]}" ] && [ "${START_STAGE}" -gt 0 ]; then
        RELOAD_OPT="${RELOAD_MODULES}"
    fi

    # 直接运行 run_stage，实时显示输出（不捕获）
    run_stage ${stage} "${PREV_CHECKPOINT}" "${RELOAD_OPT}"
    EXIT_CODE=$?
    
    # 检查训练是否成功
    if [ ${EXIT_CODE} -ne 0 ]; then
        echo "❌ Stage ${stage} training failed with exit code ${EXIT_CODE}!"
        exit 1
    fi
    
    # 构建当前 stage 的输出目录（已知路径）
    CURRENT_STAGE_OUTPUT_DIR="${RUN_ROOT_DIR}/ecot_stage${stage}"
    
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
echo "✅ All 6 stages completed successfully!"
echo "============================================================================"
echo "Total training time: ${TOTAL_HOURS}h ${TOTAL_MINS}m"
echo "Final checkpoint: ${FINAL_CHECKPOINT}"
echo ""
echo "📋 Training Summary:"
echo "  Stage 0-4: VLM reasoning training (reasoning_only)"
echo "             Stage 0-3: batch_size=16, Stage 4: batch_size=32"
echo "  Stage 5:   Action head specialization (action_only, batch_size=32)"
echo "  Stage 6:   Final joint fine-tuning (full, batch_size=32)"
echo "============================================================================"

