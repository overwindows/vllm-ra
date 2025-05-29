# MODEL_PATH=/nvmedata/hf_checkpoints/Qwen3-32B/
MODEL_PATH=/nvmedata/hf_checkpoints/Qwen3-8B/

CUDA_VISIBLE_DEVICES=3 python3 -m vllm.entrypoints.openai.api_server --model $MODEL_PATH --port 8000