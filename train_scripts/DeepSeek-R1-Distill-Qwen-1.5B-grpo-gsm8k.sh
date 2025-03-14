CUDA_VISIBLE_DEVICES=2,4 ACCELERATE_LOG_LEVEL=info accelerate launch \
	--config_file recipes/accelerate_configs/zero3.yaml \
	--main_process_port=29501 \
    --num_processes=1 src/open_r1/grpo.py \
    --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_gsm8k.yaml
