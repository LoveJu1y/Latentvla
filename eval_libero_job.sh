#!/bin/bash

# cd /share/project/baishuanghao/code/starVLA && ./examples/LIBERO/eval_libero_job.sh

# ==== 启动 inference server（后台）====
source /share/project/baishuanghao/miniconda3/etc/profile.d/conda.sh
conda activate starVLA

ckpt_root=/share/project/baishuanghao/code/starVLA/pretrained_models/libero_object_2B_mee1e-2
ckpt_path=starvla_qwen_gr00t/final_model/pytorch_model.pt
run_id=$(echo "$ckpt_path" | cut -d'/' -f1)
your_ckpt="$ckpt_root/$ckpt_path"
log_path="$ckpt_root/$run_id"
task_suite_name=$(basename "$ckpt_root" | sed 's/_2B.*//')
echo $task_suite_name
base_port=10093

export CUDA_VISIBLE_DEVICES=0
python deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${base_port} \
    --use_bf16 \
    > server.log 2>&1 &

SERVER_PID=$!
echo ">>> server started, pid = $SERVER_PID"

sleep 15   # 等服务启动

# ==== 启动 eval ====
source /share/project/baishuanghao/miniconda3/etc/profile.d/conda.sh
conda activate starVLA

export LIBERO_HOME=/share/project/baishuanghao/code/LIBERO
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME} # let eval_libero find the LIBERO tools
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo

num_trials_per_task=50
video_out_path="results/${task_suite_name}/${folder_name}"

host="127.0.0.1"
base_port=10093
unnorm_key="franka"

LOG_DIR="logs/$(date +"%Y%m%d_%H%M%S")"
mkdir -p ${LOG_DIR}

python ./examples/LIBERO/eval_libero.py \
    --args.pretrained-path ${your_ckpt} \
    --args.host "$host" \
    --args.port $base_port \
    --args.task-suite-name "$task_suite_name" \
    --args.num-trials-per-task "$num_trials_per_task" \
    --args.video-out-path "$video_out_path", \
    --args.log_path ${log_path}

# ==== eval 完成，自动 kill server ====
echo ">>> eval finished, killing server..."
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null

echo ">>> job finished"
