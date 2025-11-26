#!/bin/bash

# ECOT版本的SimplerEnv批量评测脚本
# 
# 使用方法:
#   bash star_bridge_parall_eval_ecot_batch.sh              # 使用脚本中的 ACTIVE_GROUPS 配置
#   bash star_bridge_parall_eval_ecot_batch.sh 1           # 执行配置组 1
#   bash star_bridge_parall_eval_ecot_batch.sh 1 2         # 执行配置组 1 和 2
#   bash star_bridge_parall_eval_ecot_batch.sh all         # 执行所有配置组
# 
# 功能说明:
#   - 支持评测多个 checkpoint，依次顺序执行
#   - 每个 checkpoint 内部的任务并行执行
#   - 日志自动保存在每个 checkpoint 所在目录下的同名目录
#   - 支持多 GPU 并行（通过 CUDA_VISIBLE_DEVICES 环境变量）
#   - 支持通过命令行参数选择配置组

# ==================== 用户配置区 ====================

# ==================== Checkpoint 配置组 ====================
# 使用配置组管理多个训练运行（推荐）

declare -A CHECKPOINT_GROUP_1=(
    [name]="ecot_stage4_outputs_2"
    [base_dir]="/share/project/lvjing/starVLA/4B_train_5stages/outputs_2/ecot_stage4/checkpoints"
    [pattern]="steps_*_pytorch_model.pt"
    [min_steps]=25000
    [max_steps]=47500
    [step_interval]=2500  # 可选：只选择特定间隔的checkpoint
)

declare -A CHECKPOINT_GROUP_2=(
    [name]="ecot_stage4_outputs_1"
    [base_dir]="/share/project/lvjing/starVLA/4B_train_5stages/outputs_1/ecot_stage4/checkpoints"
    [pattern]="steps_*_pytorch_model.pt"
    [min_steps]=22500
    [max_steps]=25000
    [step_interval]=2500
)

declare -A CHECKPOINT_GROUP_3=(
    [name]="ecot_stage6"
    [base_dir]="/share/project/lvjing/starVLA/4B_train_6stages/outputs_3/ecot_stage6/checkpoints"
    [pattern]="steps_*_pytorch_model.pt"
    [min_steps]=20000
    [max_steps]=22500
    [step_interval]=2500
)
declare -A CHECKPOINT_GROUP_4=(
    [name]="ecot_stage6"
    [base_dir]="/share/project/lvjing/starVLA/train_6stages/outputs_2/ecot_stage6/checkpoints"
    [pattern]="steps_*_pytorch_model.pt"
    [min_steps]=20000
    [max_steps]=60000
    [step_interval]=2000
)
# 选择要执行的配置组（"all" 表示执行所有组）
# 注意: 也可以通过命令行参数指定，如: bash script.sh 1 2 或 bash script.sh all
ACTIVE_GROUPS=("CHECKPOINT_GROUP_1")

# ==================== 手动列表方式（备用）====================
declare -a CHECKPOINT_LIST=()  # 如果不想使用配置组，可以手动指定checkpoint路径
# 模型配置
THINKING_TOKEN_COUNT=4  # thinking token 数量 (必须与训练时一致)

# 评测配置
TSET_NUM=1  # 每个任务重复次数 (1=快速测试, 4=完整评测)
NUM_EPISODES=24  # 每个任务测试的 episode 数量
# 注意: TSET_NUM=2, NUM_EPISODES=24 会生成2个独立日志文件，可评估稳定性
#       TSET_NUM=1, NUM_EPISODES=48 只生成1个日志文件，更简单

# 网络配置
BASE_PORT=10100  # 起始端口号

# GPU 配置（支持多 GPU 并行，如 "0,1,2,3"）
GPU_ID=0  # 单个 GPU 模式时使用

# ==================== 环境配置 ====================
cd "$(dirname "$0")/../.."  # 回到项目根目录
export star_vla_python=/share/project/lvjing/miniconda3/envs/starVLA/bin/python
export sim_python=/share/project/lvjing/miniconda3/envs/simpler_env/bin/python
export SimplerEnv_PATH=/share/project/lvjing/SimplerEnv
export PYTHONPATH=$(pwd):${PYTHONPATH}
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/share/project/lvjing/starVLA/qwen_cache

# ==================== GPU 配置 ====================
# 获取 CUDA_VISIBLE_DEVICES 列表（支持多 GPU 并行）
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
  # 如果没有设置，使用单个 GPU_ID
  CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi
IFS=',' read -r -a CUDA_DEVICES <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS=${#CUDA_DEVICES[@]}

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

# ==================== 函数定义 ====================

# PID 管理数组（每个 checkpoint 评测时重置）
policyserver_pids=()
eval_pids=()
server_ports=()

# 信号处理：脚本被中断时清理所有进程
cleanup_on_exit() {
  echo ""
  echo "⚠️  检测到中断信号，正在清理所有进程..."
  stop_all_servers
  exit 1
}

# 注册信号处理
trap cleanup_on_exit INT TERM

# 检查服务器是否就绪（端口是否在监听）
check_server_ready() {
  local port=$1
  local max_attempts=70  # 最多等待 30 次（约 30 秒）
  local attempt=0
  
  while [ $attempt -lt $max_attempts ]; do
    # 检查端口是否在监听（使用 netcat 或 /proc/net/tcp）
    if command -v nc >/dev/null 2>&1; then
      if nc -z localhost ${port} >/dev/null 2>&1; then
        return 0  # 服务器就绪
      fi
    else
      # 备用方法：检查 /proc/net/tcp（Linux）
      if grep -q ":$(printf '%04X' ${port}) " /proc/net/tcp 2>/dev/null; then
        return 0  # 服务器就绪
      fi
    fi
    
    attempt=$((attempt + 1))
    sleep 1
  done
  
  return 1  # 服务器未就绪
}

# 清理当前评测使用的端口范围（不影响其他端口的服务器）
cleanup_old_servers() {
  local base_port=$1
  local num_tasks=$((${#TASKS_V1[@]} + ${#TASKS_V2[@]}))
  local total_ports=$((num_tasks * TSET_NUM))
  local end_port=$((base_port + total_ports - 1))
  
  echo "🧹 清理端口范围内的旧服务器..."
  echo "   目标端口范围: ${base_port}-${end_port}"
  
  # 只清理指定端口范围的服务器进程
  local all_server_pids=$(ps aux | grep "server_policy.py" | grep -v grep | awk '{print $2}')
  
  for pid in $all_server_pids; do
    # 使用更兼容的方式提取端口（不使用 Perl 正则）
    local proc_args=$(ps -p ${pid} -o args= 2>/dev/null)
    local proc_port=$(echo "$proc_args" | sed -n 's/.*--port[[:space:]]*\([0-9]*\).*/\1/p')
    
    if [ -n "$proc_port" ] && [ "$proc_port" -ge "${base_port}" ] && [ "$proc_port" -le "${end_port}" ]; then
      echo "   端口 ${proc_port}: 发现旧服务器 (PID: ${pid})，正在清理..."
      kill ${pid} 2>/dev/null
      sleep 0.5
      
      if kill -0 ${pid} 2>/dev/null; then
        kill -9 ${pid} 2>/dev/null
        sleep 0.5
      fi
    fi
  done
  
  echo "✅ 清理完成"
  echo ""
}

# 启动策略服务器（并行模式：不等待任务完成）
start_policy_server() {
  local gpu_id=$1
  local port=$2
  local ckpt_path=$3
  local log_dir=$4
  local ckpt_name=$5
  
  local server_log_dir="${log_dir}/server_logs"
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
  
  # 验证服务器进程是否启动
  sleep 2
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "   ❌ 错误: 服务器进程启动失败，请检查日志: ${svc_log}"
    return 1
  fi
  
  # 等待服务器就绪（端口开始监听）
  echo "   ⏳ 等待服务器就绪（端口 ${port}）..."
  if check_server_ready ${port}; then
    echo "   ✅ 服务器已就绪"
  else
    echo "   ⚠️  警告: 服务器可能未完全就绪，但继续执行任务"
    echo "      请检查日志: ${svc_log}"
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
    
    # 额外清理：查找所有可能的残留进程
    local all_remaining=$(ps aux | grep "server_policy.py" | grep -v grep | awk '{print $2}')
    if [ -n "$all_remaining" ]; then
      echo "   清理所有残留的服务器进程: $all_remaining"
      kill -9 $all_remaining 2>/dev/null
    fi
  fi
  
  echo "✅ 所有服务器已停止"
  
  # 清空 PID 数组
  policyserver_pids=()
  eval_pids=()
  server_ports=()
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
  local ckpt_path=${10}
  local log_dir=${11}
  local ckpt_name=${12}
  local thinking_token_count=${13}
  
  local tag="run${run_idx}"
  local task_log="${log_dir}/${ckpt_name}_ecot_think${thinking_token_count}_infer_${env_name}.log.${tag}"
  
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
    --thinking-token-count ${thinking_token_count} \
    --logging-dir "${log_dir}" \
    > "${task_log}" 2>&1 &
  
  local task_pid=$!
  eval_pids+=($task_pid)
  echo "   任务 PID: ${task_pid}"
}

# 评测单个 checkpoint
evaluate_checkpoint() {
  local ckpt_path=$1
  local base_port=$2
  local ckpt_index=$3
  local total_ckpts=$4
  
  echo ""
  echo "======================================================"
  echo "📊 开始评测 Checkpoint [${ckpt_index}/${total_ckpts}]"
  echo "======================================================"
  
  # 检查模型文件是否存在
  if [ ! -f "$ckpt_path" ]; then
    echo "❌ 错误: 模型文件不存在: $ckpt_path"
    echo "⏭️  跳过该 checkpoint"
    return 1
  fi
  
  local ckpt_name=$(basename "${ckpt_path%.*}")
  local log_dir="$(dirname "${ckpt_path}")/${ckpt_name}"
  
  # 创建日志目录
  mkdir -p "$log_dir"
  log_dir=$(cd "$log_dir" && pwd)
  
  echo "模型路径: ${ckpt_path}"
  echo "日志目录: ${log_dir}"
  echo "Thinking Token 数量: ${THINKING_TOKEN_COUNT}"
  echo "每个任务 Episodes: ${NUM_EPISODES}"
  echo "任务重复次数: ${TSET_NUM}"
  echo "起始端口: ${base_port}"
  echo "======================================================"
  echo ""
  
  # 清理旧服务器
  cleanup_old_servers ${base_port}
  
  # 重置 PID 数组
  policyserver_pids=()
  eval_pids=()
  server_ports=()
  
  task_count=0
  
  # 执行 V1 场景任务（并行启动所有任务）
  echo "🚀 启动 V1 场景任务（并行模式）..."
  for env in "${TASKS_V1[@]}"; do
    for ((run_idx=1; run_idx<=TSET_NUM; run_idx++)); do
      port=$((base_port + task_count))
      gpu_id=${CUDA_DEVICES[$((task_count % NUM_GPUS))]}
      
      # 启动服务器
      if ! start_policy_server ${gpu_id} ${port} "${ckpt_path}" "${log_dir}" "${ckpt_name}"; then
        echo "   ❌ 服务器启动失败，跳过该任务"
        task_count=$((task_count + 1))
        continue
      fi
      
      # 运行任务（后台执行）
      run_task "$env" "$V1_SCENE" "$V1_ROBOT" "$V1_RGB" \
               "$V1_INIT_X" "$V1_INIT_Y" "$run_idx" "$port" "$gpu_id" \
               "${ckpt_path}" "${log_dir}" "${ckpt_name}" "${THINKING_TOKEN_COUNT}"
      
      task_count=$((task_count + 1))
    done
  done
  
  # 执行 V2 场景任务（并行启动所有任务）
  echo ""
  echo "🚀 启动 V2 场景任务（并行模式）..."
  for env in "${TASKS_V2[@]}"; do
    for ((run_idx=1; run_idx<=TSET_NUM; run_idx++)); do
      port=$((base_port + task_count))
      gpu_id=${CUDA_DEVICES[$((task_count % NUM_GPUS))]}
      
      # 启动服务器
      if ! start_policy_server ${gpu_id} ${port} "${ckpt_path}" "${log_dir}" "${ckpt_name}"; then
        echo "   ❌ 服务器启动失败，跳过该任务"
        task_count=$((task_count + 1))
        continue
      fi
      
      # 运行任务（后台执行）
      run_task "$env" "$V2_SCENE" "$V2_ROBOT" "$V2_RGB" \
               "$V2_INIT_X" "$V2_INIT_Y" "$run_idx" "$port" "$gpu_id" \
               "${ckpt_path}" "${log_dir}" "${ckpt_name}" "${THINKING_TOKEN_COUNT}"
      
      task_count=$((task_count + 1))
    done
  done
  
  echo ""
  echo "✅ 已启动 ${task_count} 个并行任务"
  echo "⏳ 等待所有任务完成..."
  echo ""
  
  # 统一等待所有任务完成并停止所有服务器
  stop_all_servers
  
  # 生成结果文件
  generate_result_file "${ckpt_path}" "${log_dir}" "${ckpt_name}" "${task_count}"
  
  echo ""
  echo "✅ Checkpoint [${ckpt_index}/${total_ckpts}] 评测完成！"
  echo "📁 结果保存在: ${log_dir}"
  echo ""
}

# 生成结果文件
generate_result_file() {
  local ckpt_path=$1
  local log_dir=$2
  local ckpt_name=$3
  local task_count=$4
  
  local result_file="${log_dir}/evaluation_results.txt"
  
  {
    echo "======================================================"
    echo "📊 评测完成 - 最终统计"
    echo "======================================================"
    echo "模型路径: ${ckpt_path}"
    echo "日志目录: ${log_dir}"
    echo "Thinking Token 数量: ${THINKING_TOKEN_COUNT}"
    echo "每个任务 Episodes: ${NUM_EPISODES}"
    echo "任务重复次数: ${TSET_NUM}"
    echo "总任务数: ${task_count}"
    echo "生成时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "======================================================"
    echo ""
    
    if ls ${log_dir}/*_ecot_think${THINKING_TOKEN_COUNT}_*.log.* 1> /dev/null 2>&1; then
      echo "各任务成功率:"
      echo "----------------------------------------"
      # 从日志文件名中提取任务名称，并匹配对应的成功率
      for log_file in ${log_dir}/*_ecot_think${THINKING_TOKEN_COUNT}_*.log.*; do
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
      avg_success=$(grep -h "Average success" ${log_dir}/*_ecot_think${THINKING_TOKEN_COUNT}_*.log.* | \
        awk '{sum+=$3; count++} END {if(count>0) printf "%.6f", sum/count; else printf "0.000000"}')
      echo "平均成功率: ${avg_success}"
      echo "======================================================"
    else
      echo "⚠️  未找到日志文件"
      echo "======================================================"
    fi
  } | tee "${result_file}"
  
  echo "📄 统计结果已保存到: ${result_file}"
}

# ==================== 配置组解析函数 ====================

# 从文件名提取步数
extract_steps() {
  echo "$1" | sed -n 's/.*steps_\([0-9]*\)_pytorch_model\.pt/\1/p'
}

# 解析配置组并查找checkpoint
parse_checkpoint_groups() {
  local all_groups=($(declare -p 2>/dev/null | grep -oE 'CHECKPOINT_GROUP_[0-9]+' | sort -V))
  local selected_groups=()
  
  # 确定要执行的组
  if [ "${#ACTIVE_GROUPS[@]}" -eq 0 ] || [ "${ACTIVE_GROUPS[0]}" = "all" ]; then
    selected_groups=("${all_groups[@]}")
    echo "📋 执行所有配置组: ${#selected_groups[@]} 个"
  else
    for group in "${ACTIVE_GROUPS[@]}"; do
      [[ " ${all_groups[@]} " =~ " ${group} " ]] && selected_groups+=("$group") || echo "⚠️  警告: 配置组 ${group} 不存在，跳过"
    done
    echo "📋 执行选定的配置组: ${#selected_groups[@]} 个"
  fi
  
  [ ${#selected_groups[@]} -eq 0 ] && { echo "❌ 错误: 没有找到有效的配置组"; return 1; }
  
  echo ""
  local total_found=0
  
  # 解析每个配置组
  for group_name in "${selected_groups[@]}"; do
    # 获取配置组的值
    local name=$(eval echo "\${${group_name}[name]}")
    local base_dir=$(eval echo "\${${group_name}[base_dir]}")
    local pattern=$(eval echo "\${${group_name}[pattern]}")
    local min_steps=$(eval echo "\${${group_name}[min_steps]}")
    local max_steps=$(eval echo "\${${group_name}[max_steps]}")
    local step_interval=$(eval echo "\${${group_name}[step_interval]}")
    
    echo "----------------------------------------"
    echo "配置组: ${name}"
    echo "  目录: ${base_dir}"
    [ -n "$min_steps" ] && [ -n "$max_steps" ] && echo "  步数范围: ${min_steps} - ${max_steps}"
    [ -n "$step_interval" ] && echo "  步数间隔: ${step_interval}"
    
    [ ! -d "$base_dir" ] && { echo "  ⚠️  警告: 目录不存在，跳过"; echo ""; continue; }
    
    # 查找并过滤文件
    local filtered_files=()
    while IFS= read -r -d '' file; do
      local steps=$(extract_steps "$(basename "$file")")
      [ -z "$steps" ] && continue
      [ -n "$min_steps" ] && [ "$steps" -lt "$min_steps" ] && continue
      [ -n "$max_steps" ] && [ "$steps" -gt "$max_steps" ] && continue
      [ -n "$step_interval" ] && [ "$step_interval" -gt 0 ] && [ $((steps % step_interval)) -ne 0 ] && continue
      filtered_files+=("$file")
    done < <(find "$base_dir" -maxdepth 1 -name "$pattern" -type f -print0 2>/dev/null | sort -zV)
    
    if [ ${#filtered_files[@]} -eq 0 ]; then
      echo "  ⚠️  警告: 未找到匹配的checkpoint文件"
      echo ""
      continue
    fi
    
    echo "  找到 ${#filtered_files[@]} 个 checkpoint:"
    for file in "${filtered_files[@]}"; do
      local filename=$(basename "$file")
      local steps=$(extract_steps "$filename")
      echo "    - ${filename} (steps: ${steps})"
      CHECKPOINT_LIST+=("$file")
      total_found=$((total_found + 1))
    done
    echo ""
  done
  
  echo "======================================================"
  echo "✅ 总共找到 ${total_found} 个 checkpoint"
  echo "======================================================"
  echo ""
}

# ==================== 命令行参数解析 ====================
# 支持通过命令行参数选择配置组
# 用法: bash script.sh [group_num1] [group_num2] ... [all]
# 示例: bash script.sh 1        # 执行 GROUP_1
#       bash script.sh 1 2      # 执行 GROUP_1 和 GROUP_2
#       bash script.sh all      # 执行所有组
#       无参数                  # 使用脚本中的 ACTIVE_GROUPS 配置

if [ $# -gt 0 ]; then
  ACTIVE_GROUPS=()
  for arg in "$@"; do
    if [ "$arg" = "all" ]; then
      ACTIVE_GROUPS=("all")
      break
    elif [[ "$arg" =~ ^[0-9]+$ ]]; then
      ACTIVE_GROUPS+=("CHECKPOINT_GROUP_${arg}")
    else
      echo "⚠️  警告: 无效的参数 '${arg}'，忽略"
    fi
  done
  
  if [ ${#ACTIVE_GROUPS[@]} -gt 0 ]; then
    echo "📋 命令行参数: 将执行配置组 ${ACTIVE_GROUPS[@]}"
    echo ""
  fi
fi

# ==================== 批量评测执行 ====================

# 如果 CHECKPOINT_LIST 为空，尝试从配置组解析
if [ ${#CHECKPOINT_LIST[@]} -eq 0 ]; then
  echo "======================================================"
  echo "🔍 使用配置组模式，正在查找 checkpoint..."
  echo "======================================================"
  echo ""
  
  if ! parse_checkpoint_groups; then
    echo "❌ 错误: 无法从配置组解析 checkpoint，请检查配置"
    exit 1
  fi
fi

# 检查 checkpoint 列表
if [ ${#CHECKPOINT_LIST[@]} -eq 0 ]; then
  echo "❌ 错误: CHECKPOINT_LIST 为空，请添加 checkpoint 路径或配置组"
  exit 1
fi

echo "======================================================"
echo "🚀 ECOT 批量评测脚本"
echo "======================================================"
echo "GPU 配置: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "可用 GPU 数量: ${NUM_GPUS}"
echo "GPU 列表: ${CUDA_DEVICES[@]}"
echo "Thinking Token 数量: ${THINKING_TOKEN_COUNT}"
echo "每个任务 Episodes: ${NUM_EPISODES}"
echo "任务重复次数: ${TSET_NUM}"
echo "总 Checkpoint 数量: ${#CHECKPOINT_LIST[@]}"
if [ ${#ACTIVE_GROUPS[@]} -gt 0 ] && [ "${ACTIVE_GROUPS[0]}" != "all" ]; then
  echo "执行的配置组: ${ACTIVE_GROUPS[@]}"
fi
echo "======================================================"
echo ""

# 计算每个 checkpoint 需要的端口数量
num_tasks_per_ckpt=$((${#TASKS_V1[@]} + ${#TASKS_V2[@]}))
ports_per_ckpt=$((num_tasks_per_ckpt * TSET_NUM))

# 依次评测每个 checkpoint
total_ckpts=${#CHECKPOINT_LIST[@]}
current_port=${BASE_PORT}

for ((i=0; i<${total_ckpts}; i++)); do
  ckpt_path="${CHECKPOINT_LIST[$i]}"
  ckpt_index=$((i + 1))
  
  # 评测当前 checkpoint
  evaluate_checkpoint "${ckpt_path}" "${current_port}" "${ckpt_index}" "${total_ckpts}"
  
  # 为下一个 checkpoint 更新端口（避免冲突）
  current_port=$((current_port + ports_per_ckpt + 10))  # 额外留出 10 个端口作为缓冲
done

# ==================== 最终汇总 ====================
echo ""
echo "======================================================"
echo "🎉 所有 Checkpoint 评测完成！"
echo "======================================================"
echo "总 Checkpoint 数量: ${total_ckpts}"
echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================"
echo ""
echo "📋 各 Checkpoint 结果文件位置:"
for ((i=0; i<${total_ckpts}; i++)); do
  ckpt_path="${CHECKPOINT_LIST[$i]}"
  if [ -f "$ckpt_path" ]; then
    ckpt_name=$(basename "${ckpt_path%.*}")
    log_dir="$(dirname "${ckpt_path}")/${ckpt_name}"
    result_file="${log_dir}/evaluation_results.txt"
    echo "   [$(($i + 1))] ${result_file}"
  fi
done
echo ""

