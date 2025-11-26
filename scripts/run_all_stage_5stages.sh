#!/usr/bin/env bash
# ============================================================================
# ECoT Multi-Stage Training Script
# 依次运行 Stage 0 -> Stage 1 -> Stage 2 -> Stage 3 -> Stage 4
# 每个 stage 从前一个 stage 的指定 step checkpoint 继续训练
#
# Stage 说明：
#   - Stage 0-3: 纯推理训练阶段（reasoning_only）
#              - 逐步减少显式 CoT，增加隐式推理（thinking tokens）
#              - 只训练 VLM 的隐式推理能力
#              - Action head 参数冻结（requires_grad=False）
#   - Stage 4: 完整训练阶段（full）
#              - 最少或没有显式 CoT，主要隐式推理
#              - VLM + action head 一起端到端训练
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
RUN_ROOT_DIR="./4B_train_5stages/outputs_3"
num_gpus=8
CONFIG_YAML="config/training/ecot_stage2_full.yaml"
WANDB_PROJECT="4B_Latent_qwengr00t_5stage_1"
# ============================================================================
# 跳过前若干 Stage 配置（可选）
# 如果希望从某个 stage 之后开始训练（例如跳过 Stage 0、1，从 Stage 2 开始），
# 请设置 START_STAGE > 0，并提供上一 Stage 的 checkpoint
# ============================================================================
START_STAGE=4
# 当 START_STAGE > 0 时，必须提供 PREV_STAGE_CHECKPOINT，用于初始化后续训练
PREV_STAGE_CHECKPOINT="/share/project/lvjing/starVLA/4B_train_vlm_only/outputs/ecot_stage3/checkpoints/steps_2500_pytorch_model.pt"
# 示例：从 Stage 2 开始训练，使用 Stage 1 的 checkpoint
# START_STAGE=2
# PREV_STAGE_CHECKPOINT="/share/project/lvjing/starVLA/train_2shcedule/outputs1/ecot_stage1/checkpoints/steps_5000_pytorch_model.pt"

# ============================================================================
# 权重重载配置（架构变更时使用）
# ============================================================================
# 如果从旧架构（如 DiT-B）迁移到新架构（如 DiT-L），需要忽略形状不匹配的层
# 设置为 "qwen_vl_interface" 可只加载 VLM 权重；正常情况请留空 ""
# RELOAD_MODULES="qwen_vl_interface"
RELOAD_MODULES=""
ALL_STAGES=(0 1 2 3 4)
STAGES=()
for stage in "${ALL_STAGES[@]}"; do
    if [ ${stage} -ge ${START_STAGE} ]; then
        STAGES+=($stage)
    fi
done

if [ ${#STAGES[@]} -eq 0 ]; then
    echo "❌ Error: START_STAGE (${START_STAGE}) exceeds available stages!"
    exit 1
fi

if [ ${START_STAGE} -gt 0 ] && [ -z "${PREV_STAGE_CHECKPOINT}" ]; then
    echo "❌ Error: START_STAGE=${START_STAGE} but PREV_STAGE_CHECKPOINT is empty."
    echo "   Please provide the checkpoint of Stage $((START_STAGE-1))."
    exit 1
fi

# Stage 配置：每个 stage 的训练步数和保存间隔
# Stage 0-3: 纯推理训练（reasoning_only，只训练 VLM，逐步增加隐式推理）
# Stage 4: 完整训练（full，VLM + action head 端到端训练）
declare -A STAGE_STEPS=(
    [0]=10000   # Stage 0 训练步数（纯推理训练，全 CoT）
    [1]=8000   # Stage 1 训练步数（纯推理训练）
    [2]=8000  # Stage 2 训练步数（纯推理训练）
    [3]=8000   # Stage 3 训练步数（纯推理训练）
    [4]=60000   # Stage 4 训练步数（完整训练，VLM + action head）
)

declare -A STAGE_SAVE_INTERVALS=(
    [0]=5000    # Stage 0 保存间隔
    [1]=4000    # Stage 1 保存间隔
    [2]=4000   # Stage 2 保存间隔
    [3]=4000    # Stage 3 保存间隔
    [4]=2500    # Stage 4 保存间隔
)

# 每个 stage 使用的 checkpoint step（从前一个 stage 加载）
# 格式：stage -> checkpoint_step（从前一个 stage 的哪个 step 加载）
declare -A STAGE_CHECKPOINT_STEPS=(
    [0]=""           # Stage 0 不使用 checkpoint（从头训练）
    [1]="10000"      # Stage 1 使用 Stage 0 的 steps_50000 checkpoint
    [2]="8000"      # Stage 2 使用 Stage 1 的 steps_10000 checkpoint
    [3]="8000"      # Stage 3 使用 Stage 2 的 steps_10000 checkpoint
    [4]="8000"      # Stage 4 使用 Stage 3 的 steps_10000 checkpoint
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
    
    # 根据 stage 决定训练阶段模式
    # Stage 0-3: 纯推理训练（只训练 VLM，冻结 action head）
    # Stage 4: 完整训练（VLM + action head 一起训练）
    if [ ${stage} -lt 4 ]; then
        TRAINING_STAGE="reasoning_only"
        VLM_LOSS_WEIGHT=1.0  # Stage 0-3 纯推理，权重设为 1.0
        echo "🧠 [Stage ${stage}] Reasoning-only mode: Training VLM reasoning, action_model frozen"
    else
        TRAINING_STAGE="full"
        VLM_LOSS_WEIGHT=0.1  # Stage 4 完整训练，平衡 action_loss 和 vlm_loss
        echo "🎯 [Stage ${stage}] Full training mode: VLM + action_model both trainable"
    fi
    
    # 根据 stage 设置 batch size
    # Stage 0-3: batch_size=12
    # Stage 4: batch_size=16
    if [ ${stage} -ge 3 ]; then
        BATCH_SIZE=16
    else
        BATCH_SIZE=12
    fi
    
    # 构建训练参数（与 run_ecot_8gpu.sh 保持一致）
    TRAIN_CONFIG_ARGS=(
        --trainer.max_train_steps ${STAGE_STEPS[$stage]}
        --trainer.save_interval ${STAGE_SAVE_INTERVALS[$stage]}
        --trainer.logging_frequency 10
        --trainer.eval_interval 50000000
        --trainer.learning_rate.base 3.0e-5
        --trainer.gradient_accumulation_steps 1
        --framework.qwenvl.model_max_length 2048
        --framework.action_model.action_model_type DiT-L
        --framework.action_model.diffusion_model_cfg.num_layers 16
        --framework.training_stage ${TRAINING_STAGE}
        --framework.latent_reasoning.vlm_loss_weight ${VLM_LOSS_WEIGHT}
        --datasets.vla_data.per_device_batch_size ${BATCH_SIZE}
        --datasets.vla_data.ecot.scheduled_stage ${stage}
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
        --wandb_project "${WANDB_PROJECT}"
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
echo "🎯 ECoT Multi-Stage Training Pipeline"
echo "============================================================================"
echo "Stages to run: ${STAGES[@]}"
if [ ${START_STAGE} -gt 0 ]; then
    echo "  - Stages 0~$((START_STAGE-1)): 跳过（使用提供的 checkpoint: ${PREV_STAGE_CHECKPOINT}）"
    echo "  - Stage ${START_STAGE} 起：按配置训练"
else
    echo "  - Stage 0-3: 纯推理训练（reasoning_only，只训练 VLM，action head 冻结）"
    echo "  - Stage 4: 完整训练（full，VLM + action head 端到端训练）"
fi
echo "Number of GPUs: ${num_gpus}"
echo "Config: ${CONFIG_YAML}"
echo "Output directories: ${RUN_ROOT_DIR}/ecot_stage{${STAGES[@]}}"
echo "============================================================================"

# 记录开始时间
START_TIME=$(date +%s)

# 依次运行各个 stage
PREV_STAGE_OUTPUT_DIR=""
FIRST_STAGE=${STAGES[0]}
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
    
    if [ ${stage} -eq ${FIRST_STAGE} ]; then
        if [ ${stage} -eq 0 ]; then
            echo "ℹ️  Starting Stage 0 from scratch (no previous checkpoint)"
        else
            PREV_CHECKPOINT="${PREV_STAGE_CHECKPOINT}"
            echo "🔍 Using provided checkpoint from Stage $((stage-1)): ${PREV_CHECKPOINT}"
            if [ ! -f "${PREV_CHECKPOINT}" ]; then
                echo "❌ Provided checkpoint not found: ${PREV_CHECKPOINT}"
                exit 1
            fi
            echo "✅ Checkpoint found"
        fi
    else
        if [ -n "${PREV_STAGE_OUTPUT_DIR}" ] && [ -n "${STAGE_CHECKPOINT_STEPS[$stage]}" ]; then
            checkpoint_step=${STAGE_CHECKPOINT_STEPS[$stage]}
            PREV_CHECKPOINT="${PREV_STAGE_OUTPUT_DIR}/checkpoints/steps_${checkpoint_step}_pytorch_model.pt"
            echo "🔍 Checking for checkpoint from Stage $((stage-1)): ${PREV_CHECKPOINT}"
            if [ ! -f "${PREV_CHECKPOINT}" ]; then
                echo "❌ Checkpoint not found: ${PREV_CHECKPOINT}"
                echo "   Please check if Stage $((stage-1)) completed successfully."
                exit 1
            fi
            echo "✅ Checkpoint found"
        else
            echo "ℹ️  No previous checkpoint available; starting from scratch"
        fi
    fi
    
    echo "🚀 Now calling run_stage function..."
    # 直接运行 run_stage，实时显示输出（不捕获）
    # 判断是否需要 reload_modules (仅在第一个运行的 stage 且有外部 checkpoint 时使用)
    RELOAD_OPT=""
    if [ "${stage}" == "${STAGES[0]}" ] && [ "${START_STAGE}" -gt 0 ]; then
        RELOAD_OPT="${RELOAD_MODULES}"
    fi
    
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
echo "✅ All stages completed successfully!"
echo "============================================================================"
echo "Total training time: ${TOTAL_HOURS}h ${TOTAL_MINS}m"
echo "Final checkpoint: ${FINAL_CHECKPOINT}"
echo "============================================================================"
