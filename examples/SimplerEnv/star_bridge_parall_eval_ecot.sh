#!/bin/bash

# ECOT版本的SimplerEnv评测脚本（并行执行版本）
# 使用方法: bash star_bridge_parall_eval_ecot.sh
# 
# 配置说明: 直接修改下面的配置变量即可
#
# 并行执行特性:
#   - 所有任务并行启动，大幅提升评测速度
#   - 支持多 GPU 并行（通过 CUDA_VISIBLE_DEVICES 环境变量）
#   - 自动在所有可见 GPU 上分配任务
#
# 日志目录自动设置:
#   - 默认情况下，日志会保存在模型文件所在目录下的同名目录（去掉.pt后缀）
#   - 例如: /path/to/steps_14000_pytorch_model.pt 
#         → /path/to/steps_14000_pytorch_model/
#   - 如需自定义，可以在脚本中设置 LOG_DIR 变量
#
# 并行运行多个评测实例:
#   1. 为每个评测任务设置不同的 BASE_PORT (例如: 10068, 10100, 10200)
#   2. 每个模型会自动使用对应的日志目录（无需手动设置）
#   3. 在不同终端运行多个脚本实例即可
#   脚本只会清理自己使用的端口范围，不会影响其他评测任务

# ==================== 用户配置区 ====================
# 模型配置

MODEL_PATH="/share/project/lvjing/starVLA/train_6stages/outputs_1/ecot_stage6/checkpoints/steps_29000_pytorch_model.pt"
THINKING_TOKEN_COUNT=4  # thinking token 数量 (必须与训练时一致)

# 日志配置
# 日志目录会自动设置为模型文件所在目录下的同名目录（去掉.pt后缀）
# 例如: /path/to/steps_30000_pytorch_model.pt 
#      → /path/to/steps_30000_pytorch_model/
# 如果不需要自动设置，可以取消下面的注释并手动指定 LOG_DIR
# LOG_DIR="./29000_pytorch_model3"  # 手动指定日志目录（可选）

# 评测配置
TSET_NUM=1  # 每个任务重复次数 (1=快速测试, 4=完整评测)
NUM_EPISODES=24  # 每个任务测试的 episode 数量 (SimplerEnv 标准是 24)

# 网络配置
BASE_PORT=10100  # 起始端口号
# GPU 配置: 支持多 GPU 并行，例如 "0,1,2,3" 或 "0"
# 如果不设置 CUDA_VISIBLE_DEVICES，将使用 GPU_ID
# 如果设置了 CUDA_VISIBLE_DEVICES，将自动在所有可见 GPU 上并行执行
GPU_ID=0  # 单个 GPU 模式时使用的 GPU ID

# ==================== 环境配置 ====================
cd "$(dirname "$0")/../.."  # 回到项目根目录
export star_vla_python=/share/project/lvjing/miniconda3/envs/starVLA/bin/python
export sim_python=/share/project/lvjing/miniconda3/envs/simpler_env/bin/python
export SimplerEnv_PATH=/share/project/lvjing/SimplerEnv
export PYTHONPATH=$(pwd):${PYTHONPATH}
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/share/project/lvjing/starVLA/qwen_cache

# 检查模型路径
if [ ! -f "$MODEL_PATH" ]; then
  echo "❌ 错误: 模型文件不存在: $MODEL_PATH"
  echo "💡 请在脚本开头的配置区修改 MODEL_PATH"
  exit 1
fi

ckpt_path="$MODEL_PATH"
ckpt_name=$(basename "${ckpt_path%.*}")

# 自动设置日志目录为模型文件所在目录下的同名目录
if [ -z "$LOG_DIR" ]; then
  # 如果 LOG_DIR 未设置，自动生成：模型目录/模型文件名（去掉.pt后缀）
  LOG_DIR="$(dirname "${ckpt_path}")/${ckpt_name}"
fi

# 创建日志目录
mkdir -p "$LOG_DIR"
LOG_DIR=$(cd "$LOG_DIR" && pwd)

# ==================== 配置信息 ====================
echo "======================================================"
echo "📊 ECOT 评测配置"
echo "======================================================"
echo "模型路径: ${ckpt_path}"
echo "日志目录: ${LOG_DIR}"
echo "Thinking Token 数量: ${THINKING_TOKEN_COUNT}"
echo "每个任务 Episodes: ${NUM_EPISODES}"
echo "任务重复次数: ${TSET_NUM}"
echo "起始端口: ${BASE_PORT}"
echo "GPU: ${GPU_ID}"
echo "======================================================"
echo ""

# ==================== GPU 配置 ====================
# 获取 CUDA_VISIBLE_DEVICES 列表（支持多 GPU 并行）
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
  # 如果没有设置，使用单个 GPU_ID
  CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi
IFS=',' read -r -a CUDA_DEVICES <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS=${#CUDA_DEVICES[@]}

echo "GPU 配置: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "可用 GPU 数量: ${NUM_GPUS}"
echo "GPU 列表: ${CUDA_DEVICES[@]}"
echo ""

# ==================== PID 管理 ====================
policyserver_pids=()  # 所有服务器进程 PID
eval_pids=()          # 所有评测任务进程 PID
server_ports=()       # 所有服务器端口（用于清理）

# ==================== 函数定义 ====================

# 清理当前评测使用的端口范围（不影响其他端口的服务器）
cleanup_old_servers() {
  echo "🧹 清理端口范围内的旧服务器..."
  
  # 计算需要的端口数量（V1任务数 + V2任务数）× 重复次数
  local num_tasks=$((${#TASKS_V1[@]} + ${#TASKS_V2[@]}))
  local total_ports=$((num_tasks * TSET_NUM))
  local end_port=$((BASE_PORT + total_ports - 1))
  
  echo "   目标端口范围: ${BASE_PORT}-${end_port}"
  
  # 只清理指定端口范围的服务器进程
  # 查找所有 server_policy.py 进程
  local all_server_pids=$(ps aux | grep "server_policy.py" | grep -v grep | awk '{print $2}')
  
  for pid in $all_server_pids; do
    # 获取该进程使用的端口
    local proc_port=$(ps -p ${pid} -o args= 2>/dev/null | grep -oP '(?<=--port )\d+')
    
    if [ -n "$proc_port" ]; then
      # 检查该端口是否在我们的清理范围内
      if [ "$proc_port" -ge "${BASE_PORT}" ] && [ "$proc_port" -le "${end_port}" ]; then
        echo "   端口 ${proc_port}: 发现旧服务器 (PID: ${pid})，正在清理..."
        kill ${pid} 2>/dev/null
        sleep 0.5
        
        # 如果进程还在，强制结束
        if kill -0 ${pid} 2>/dev/null; then
          kill -9 ${pid} 2>/dev/null
          sleep 0.5
        fi
      fi
    fi
  done
  
  echo "✅ 清理完成，其他端口的服务器不受影响"
  echo ""
}

# 启动策略服务器（并行模式：不等待任务完成）
start_policy_server() {
  local gpu_id=$1
  local port=$2
  local server_log_dir="${LOG_DIR}/server_logs"
  local svc_log="${server_log_dir}/${ckpt_name}_policy_server_${port}.log"
  
  mkdir -p "${server_log_dir}"
  
  # 确保端口可用（清理占用该端口的旧进程）
  local old_pids=$(ps aux | grep "server_policy.py.*--port ${port}" | grep -v grep | awk '{print $2}')
  if [ -n "$old_pids" ]; then
    echo "   清理端口 ${port} 上的旧服务器: $old_pids"
    kill -9 $old_pids 2>/dev/null
    sleep 1
  fi
  
  echo "▶️  启动策略服务器 (GPU ${gpu_id}, 端口 ${port})..."
  
  CUDA_VISIBLE_DEVICES=${gpu_id} ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path "${ckpt_path}" \
    --port ${port} \
    --use_bf16 \
    > "${svc_log}" 2>&1 &
  
  local pid=$!
  policyserver_pids+=($pid)
  server_ports+=($port)
  echo "   服务器 PID: ${pid}"
  sleep 8  # 等待服务器启动
  
  # 验证服务器是否成功启动
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "   ⚠️  警告: 服务器进程可能启动失败，请检查日志: ${svc_log}"
  fi
}

# 停止所有服务器（并行模式：统一清理）
stop_all_servers() {
  echo ""
  echo "⏹️  停止所有策略服务器..."
  
  # 等待所有评测任务完成
  if [ "${#eval_pids[@]}" -gt 0 ]; then
    echo "⏳ 等待所有评测任务完成..."
    for pid in "${eval_pids[@]}"; do
      if ps -p "$pid" > /dev/null 2>&1; then
        wait "$pid"
        status=$?
        if [ $status -ne 0 ]; then
          echo "   ⚠️  警告: 评测任务 $pid 异常退出 (状态码: $status)"
        fi
      fi
    done
    echo "✅ 所有评测任务已完成"
  fi
  
  # 停止所有服务器
  if [ "${#policyserver_pids[@]}" -gt 0 ]; then
    for pid in "${policyserver_pids[@]}"; do
      if ps -p "$pid" > /dev/null 2>&1; then
        echo "   停止服务器 (PID: ${pid})"
        kill "$pid" 2>/dev/null
      fi
    done
    sleep 2
    
    # 强制停止仍在运行的服务器
    for pid in "${policyserver_pids[@]}"; do
      if ps -p "$pid" > /dev/null 2>&1; then
        echo "   强制停止服务器 (PID: ${pid})"
        kill -9 "$pid" 2>/dev/null
      fi
    done
    
    # 清理所有端口上的残留进程
    for port in "${server_ports[@]}"; do
      local remaining_pids=$(ps aux | grep "server_policy.py.*--port ${port}" | grep -v grep | awk '{print $2}')
      if [ -n "$remaining_pids" ]; then
        echo "   清理端口 ${port} 上的残留进程: $remaining_pids"
        kill -9 $remaining_pids 2>/dev/null
      fi
    done
  fi
  
  echo "✅ 所有服务器已停止"
}

# 运行单个任务（并行模式：后台执行）
run_task() {
  local env_name=$1
  local scene_name=$2
  local robot=$3
  local rgb_overlay=$4
  local robot_x=$5
  local robot_y=$6
  local run_idx=$7
  local port=$8
  local gpu_id=$9
  
  local tag="run${run_idx}"
  local task_log="${LOG_DIR}/${ckpt_name}_ecot_think${THINKING_TOKEN_COUNT}_infer_${env_name}.log.${tag}"
  
  echo "▶️  [任务 ${env_name}] 第 ${run_idx}/${TSET_NUM} 次运行 (GPU ${gpu_id}, 端口 ${port})"
  echo "   日志: ${task_log}"
  
  # 取消 WORLD_SIZE 避免 accelerate 干扰
  unset WORLD_SIZE
  
  CUDA_VISIBLE_DEVICES=${gpu_id} ${sim_python} examples/SimplerEnv/start_simpler_env.py \
    --port ${port} \
    --ckpt-path "${ckpt_path}" \
    --robot ${robot} \
    --policy-setup widowx_bridge \
    --control-freq 5 \
    --sim-freq 500 \
    --max-episode-steps 120 \
    --env-name "${env_name}" \
    --scene-name ${scene_name} \
    --rgb-overlay-path ${rgb_overlay} \
    --robot-init-x ${robot_x} ${robot_x} 1 \
    --robot-init-y ${robot_y} ${robot_y} 1 \
    --obj-variation-mode episode \
    --obj-episode-range 0 ${NUM_EPISODES} \
    --robot-init-rot-quat-center 0 0 0 1 \
    --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
    --enable-latent-reasoning \
    --thinking-token-count ${THINKING_TOKEN_COUNT} \
    --logging-dir "${LOG_DIR}" \
    > "${task_log}" 2>&1 &
  
  local task_pid=$!
  eval_pids+=($task_pid)
  echo "   任务 PID: ${task_pid}"
}

# ==================== 任务配置 ====================

# V1 场景配置
declare -a TASKS_V1=(
  "StackGreenCubeOnYellowCubeBakedTexInScene-v0"
  "PutCarrotOnPlateInScene-v0"
  "PutSpoonOnTableClothInScene-v0"
)
V1_SCENE="bridge_table_1_v1"
V1_ROBOT="widowx"
V1_RGB="${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png"
V1_INIT_X=0.147
V1_INIT_Y=0.028

# V2 场景配置
declare -a TASKS_V2=(
  "PutEggplantInBasketScene-v0"
)
V2_SCENE="bridge_table_1_v2"
V2_ROBOT="widowx_sink_camera_setup"
V2_RGB="${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png"
V2_INIT_X=0.127
V2_INIT_Y=0.06

# ==================== 执行评测 ====================

# 清理所有旧服务器（在开始评测前执行一次）
cleanup_old_servers

task_count=0

# 执行 V1 场景任务（并行启动所有任务）
echo "🚀 启动 V1 场景任务（并行模式）..."
for env in "${TASKS_V1[@]}"; do
  for ((run_idx=1; run_idx<=TSET_NUM; run_idx++)); do
    port=$((BASE_PORT + task_count))
    gpu_id=${CUDA_DEVICES[$((task_count % NUM_GPUS))]}
    
    # 启动服务器
    start_policy_server ${gpu_id} ${port}
    
    # 运行任务（后台执行）
    run_task "$env" "$V1_SCENE" "$V1_ROBOT" "$V1_RGB" \
             "$V1_INIT_X" "$V1_INIT_Y" "$run_idx" "$port" "$gpu_id"
    
    task_count=$((task_count + 1))
  done
done

# 执行 V2 场景任务（并行启动所有任务）
echo ""
echo "🚀 启动 V2 场景任务（并行模式）..."
for env in "${TASKS_V2[@]}"; do
  for ((run_idx=1; run_idx<=TSET_NUM; run_idx++)); do
    port=$((BASE_PORT + task_count))
    gpu_id=${CUDA_DEVICES[$((task_count % NUM_GPUS))]}
    
    # 启动服务器
    start_policy_server ${gpu_id} ${port}
    
    # 运行任务（后台执行）
    run_task "$env" "$V2_SCENE" "$V2_ROBOT" "$V2_RGB" \
             "$V2_INIT_X" "$V2_INIT_Y" "$run_idx" "$port" "$gpu_id"
    
    task_count=$((task_count + 1))
  done
done

echo ""
echo "✅ 已启动 ${task_count} 个并行任务"
echo "⏳ 等待所有任务完成..."
echo ""

# 统一等待所有任务完成并停止所有服务器
stop_all_servers

# ==================== 结果汇总 ====================
echo ""
echo "======================================================"
echo "📊 评测完成 - 最终统计"
echo "======================================================"
echo "总任务数: ${task_count}"
echo ""

# 创建结果文件路径
RESULT_FILE="${LOG_DIR}/evaluation_results.txt"

# 将结果同时输出到终端和文件
{
  echo "======================================================"
  echo "📊 评测完成 - 最终统计"
  echo "======================================================"
  echo "模型路径: ${ckpt_path}"
  echo "日志目录: ${LOG_DIR}"
  echo "Thinking Token 数量: ${THINKING_TOKEN_COUNT}"
  echo "每个任务 Episodes: ${NUM_EPISODES}"
  echo "任务重复次数: ${TSET_NUM}"
  echo "总任务数: ${task_count}"
  echo "生成时间: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "======================================================"
  echo ""
  
  if ls ${LOG_DIR}/*_ecot_think${THINKING_TOKEN_COUNT}_*.log.* 1> /dev/null 2>&1; then
    echo "各任务成功率:"
    echo "----------------------------------------"
    # 从日志文件名中提取任务名称，并匹配对应的成功率
    for log_file in ${LOG_DIR}/*_ecot_think${THINKING_TOKEN_COUNT}_*.log.*; do
      # 从文件名中提取任务名称
      # 格式: ${ckpt_name}_ecot_think${THINKING_TOKEN_COUNT}_infer_${env_name}.log.${tag}
      filename=$(basename "$log_file")
      task_name=$(echo "$filename" | sed -n "s/.*_infer_\([^.]*\)\.log\..*/\1/p")
      run_tag=$(echo "$filename" | sed -n "s/.*\.log\.\(.*\)/\1/p")
      
      # 从日志文件中提取成功率
      success_rate=$(grep "Average success" "$log_file" | awk '{print $3}')
      
      if [ -n "$success_rate" ]; then
        printf "   %s (run%s): %s\n" "$task_name" "$run_tag" "$success_rate"
      fi
    done | sort
    echo ""
    echo "----------------------------------------"
    
    # 计算平均成功率
    avg_success=$(grep -h "Average success" ${LOG_DIR}/*_ecot_think${THINKING_TOKEN_COUNT}_*.log.* | \
      awk '{sum+=$3; count++} END {if(count>0) printf "%.6f", sum/count; else printf "0.000000"}')
    echo "平均成功率: ${avg_success}"
    echo "======================================================"
  else
    echo "⚠️  未找到日志文件"
    echo "======================================================"
  fi
} | tee "${RESULT_FILE}"

echo ""
echo "✅ 所有评测任务已完成！"
echo "📁 结果保存在: ${LOG_DIR}"
echo "📄 统计结果已保存到: ${RESULT_FILE}"
echo ""
