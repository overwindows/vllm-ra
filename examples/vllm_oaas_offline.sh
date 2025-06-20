# CUDA_VISIBLE_DEVICES=3 python3 vllm_oaas.py
# CUDA_VISIBLE_DEVICES=3 python3 vllm_oaas_async.py
# CUDA_VISIBLE_DEVICES=3 python3 llm_analyzer_vllm_oaas.py
# CUDA_VISIBLE_DEVICES=3 python3 llm_analyzer_vllm_oaas_async.py \

# Set logging levels
export PYTHONWARNINGS="ignore"
export LOGURU_LEVEL="ERROR"
export VLLM_LOG_LEVEL="ERROR"
export PYTHONUNBUFFERED=1

# MODEL_PATH=/nvmedata/hf_checkpoints/Qwen3-8B/
MODEL_PATH=/nvmedata/hf_checkpoints/Llama-2-7b-chat-hf-bf16

CUDA_VISIBLE_DEVICES=3 python3 llm_analyzer_vllm_oaas_async_ra.py \
    --input_path /nvmedata/chenw/genz/genz_users_20k_format.tsv \
    --output_path /nvmedata/chenw/genz/genz_users_interests_vllm_oaas_async.jsonl \
    --model_path $MODEL_PATH \
    --batch_size 32