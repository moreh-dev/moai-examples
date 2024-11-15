#!/bin/bash

START_TIME=$(TZ="Asia/Seoul" date)
current_time=$(date +"%y%m%d_%H%M%S")

TRANSFORMERS_VERBOSITY=info accelerate launch \
    --config_file finetuning_codes/config.yaml \
    finetuning_codes/train.py \
    --model Qwen/Qwen-14B \
    --dataset alespalla/chatbot_instruction_prompts \
    --lr 0.0001 \
    --use-lora \
    --train-batch-size 8 \
    --eval-batch-size 1 \
    --num-epochs 5 \
    --max-steps 100 \
    --log-interval 2 \
    --save-path /root/poc/checkpoints/qwen_finetuned_$current_time \
    |& tee /root/poc/finetuning_codes/logs/qwen_finetune_$current_time.log

echo "Start: $START_TIME"
echo "End: $(TZ="Asia/Seoul" date)"
