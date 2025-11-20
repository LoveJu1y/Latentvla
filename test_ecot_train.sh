export CUDA_VISIBLE_DEVICES=0
export HF_HOME=./qwen
export HF_ENDPOINT=https://hf-mirror.com


python test_ecot_training.py --config_yaml config/test_ecot_stage2.yaml