#!/bin/bash

# 同步离线 WandB 日志到云端
# 使用方法: bash scripts/sync_wandb.sh

cd "$(dirname "$0")/.."

echo "======================================================"
echo "📤 同步离线 WandB 日志"
echo "======================================================"

# WandB 配置
export WANDB_API_KEY="a8989c35c0573184da807b8a781d72936fe7e379"
export WANDB_BASE_URL="https://api.bandw.top"

# WandB 环境
WANDB_BIN=/share/project/lvjing/miniconda3/envs/starVLA/bin/wandb

# 查找所有离线运行目录
WANDB_DIR="outputs2/ecot_stage4_fianl/wandb/wandb"

if [ ! -d "$WANDB_DIR" ]; then
  echo "❌ 错误: WandB 目录不存在: $WANDB_DIR"
  exit 1
fi

echo ""
echo "正在查找离线运行..."
echo ""

# 查找所有 offline-run-* 目录
offline_runs=$(find "$WANDB_DIR" -maxdepth 1 -type d -name "offline-run-*")

if [ -z "$offline_runs" ]; then
  echo "⚠️  未找到离线运行"
  exit 0
fi

# 统计数量
num_runs=$(echo "$offline_runs" | wc -l)
echo "发现 ${num_runs} 个离线运行"
echo ""

# 逐个同步
count=0
for run_dir in $offline_runs; do
  count=$((count + 1))
  run_name=$(basename "$run_dir")
  
  echo "======================================================"
  echo "[$count/$num_runs] 同步: $run_name"
  echo "======================================================"
  
  ${WANDB_BIN} sync "$run_dir"
  
  if [ $? -eq 0 ]; then
    echo "✅ 同步成功: $run_name"
  else
    echo "❌ 同步失败: $run_name"
  fi
  
  echo ""
done

echo "======================================================"
echo "✅ 所有离线运行已处理完成"
echo "======================================================"

