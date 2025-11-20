#!/bin/bash

# ECOT版本的SimplerEnv评测脚本
# 使用方法: bash star_bridge_parall_eval_ecot.sh
# 
# 配置说明: 直接修改下面的配置变量即可
#
# 并行运行多个评测:
#   1. 为每个评测任务设置不同的 BASE_PORT (例如: 10068, 10100, 10200)
#   2. 设置不同的 LOG_DIR (例如: ./logs_model1, ./logs_model2)
#   3. 在不同终端运行多个脚本实例即可
#   脚本只会清理自己使用的端口范围，不会影响其他评测任务

# ==================== 用户配置区 ====================
# 模型配置
MODEL_PATH="/share/project/lvjing/starVLA/outputs2/ecot_stage4_fianl_plus60k/checkpoints/steps_22500_pytorch_model.pt"
THINKING_TOKEN_COUNT=4  # thinking token 数量 (必须与训练时一致)

# 日志配置
LOG_DIR="./50+22500_logs_semble7"  # 日志和视频保存目录

# 评测配置
TSET_NUM=1  # 每个任务重复次数 (1=快速测试, 4=完整评测)
NUM_EPISODES=20  # 每个任务测试的 episode 数量 (SimplerEnv 标准是 24)

# 网络配置
BASE_PORT=10100  # 起始端口号
GPU_ID=0  # 使用的 GPU ID

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

# 启动策略服务器
start_policy_server() {
  local port=$1
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
  
  echo "▶️  启动策略服务器 (端口 ${port})..."
  
  CUDA_VISIBLE_DEVICES=${GPU_ID} ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path "${ckpt_path}" \
    --port ${port} \
    --use_bf16 \
    > "${svc_log}" 2>&1 &
  
  local pid=$!
  echo "   服务器 PID: ${pid}"
  sleep 8  # 等待服务器启动
  
  # 验证服务器是否成功启动
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "   ⚠️  警告: 服务器进程可能启动失败，请检查日志: ${svc_log}"
  fi
  
  echo "$pid"
}

# 停止策略服务器
stop_policy_server() {
  local pid=$1
  local port=$2
  
  # 尝试优雅关闭
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    echo "⏹️  停止策略服务器 (PID: ${pid})"
    kill "$pid" 2>/dev/null
    sleep 2
  fi
  
  # 强制关闭进程
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    echo "   强制停止进程..."
    kill -9 "$pid" 2>/dev/null
    sleep 1
  fi
  
  # 再次检查端口上是否还有残留进程
  local remaining_pids=$(ps aux | grep "server_policy.py.*--port ${port}" | grep -v grep | awk '{print $2}')
  if [ -n "$remaining_pids" ]; then
    echo "   清理端口 ${port} 上的残留进程: $remaining_pids"
    kill -9 $remaining_pids 2>/dev/null
    sleep 1
  fi
}

# 运行单个任务
run_task() {
  local env_name=$1
  local scene_name=$2
  local robot=$3
  local rgb_overlay=$4
  local robot_x=$5
  local robot_y=$6
  local run_idx=$7
  local port=$8
  
  local tag="run${run_idx}"
  local task_log="${LOG_DIR}/${ckpt_name}_ecot_think${THINKING_TOKEN_COUNT}_infer_${env_name}.log.${tag}"
  
  echo ""
  echo "▶️  [任务 ${env_name}] 第 ${run_idx}/${TSET_NUM} 次运行"
  echo "   日志: ${task_log}"
  
  # 取消 WORLD_SIZE 避免 accelerate 干扰
  unset WORLD_SIZE
  
  CUDA_VISIBLE_DEVICES=${GPU_ID} ${sim_python} examples/SimplerEnv/start_simpler_env.py \
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
    > "${task_log}" 2>&1
  
  echo "✅  任务完成"
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

# 执行 V1 场景任务
for env in "${TASKS_V1[@]}"; do
  for ((run_idx=1; run_idx<=TSET_NUM; run_idx++)); do
    port=$((BASE_PORT + task_count))
    
    # 启动服务器
    server_pid=$(start_policy_server ${port})
    
    # 运行任务
    run_task "$env" "$V1_SCENE" "$V1_ROBOT" "$V1_RGB" \
             "$V1_INIT_X" "$V1_INIT_Y" "$run_idx" "$port"
    
    # 停止服务器（传入 PID 和端口）
    stop_policy_server "$server_pid" "$port"
    
    task_count=$((task_count + 1))
  done
done

# 执行 V2 场景任务
for env in "${TASKS_V2[@]}"; do
  for ((run_idx=1; run_idx<=TSET_NUM; run_idx++)); do
    port=$((BASE_PORT + task_count))
    
    # 启动服务器
    server_pid=$(start_policy_server ${port})
    
    # 运行任务
    run_task "$env" "$V2_SCENE" "$V2_ROBOT" "$V2_RGB" \
             "$V2_INIT_X" "$V2_INIT_Y" "$run_idx" "$port"
    
    # 停止服务器（传入 PID 和端口）
    stop_policy_server "$server_pid" "$port"
    
    task_count=$((task_count + 1))
  done
done

# ==================== 结果汇总 ====================
echo ""
echo "======================================================"
echo "📊 评测完成 - 最终统计"
echo "======================================================"
echo "总任务数: ${task_count}"
echo ""

if ls ${LOG_DIR}/*_ecot_think${THINKING_TOKEN_COUNT}_*.log.* 1> /dev/null 2>&1; then
  grep -h "Average success" ${LOG_DIR}/*_ecot_think${THINKING_TOKEN_COUNT}_*.log.* | \
    awk '{print "   " $0}'
else
  echo "   ⚠️  未找到日志文件"
fi

echo "======================================================"
echo ""
echo "✅ 所有评测任务已完成！"
echo "📁 结果保存在: ${LOG_DIR}"
echo ""
