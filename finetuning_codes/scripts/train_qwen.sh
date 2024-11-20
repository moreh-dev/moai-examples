#!/bin/bash

export HF_HOME=/nas/team_cx/jingee/datasets

START_TIME=$(TZ="Asia/Seoul" date)
current_time=$(date +"%y%m%d_%H%M%S")

TRANSFORMERS_VERBOSITY=info accelerate launch \
    --config_file finetuning_codes/config.yaml \
    finetuning_codes/train.py \
    --model Qwen/Qwen-14B \
    --dataset alespalla/chatbot_instruction_prompts \
    --lr 0.0001 \
    --use-lora \
    --train-batch-size 64 \
    --eval-batch-size 1 \
    --num-epochs 5 \
    --max-steps 100 \
    --log-interval 20 \
    --save-path /nas/team_cx/jingee/checkpoints/qwen_finetuned_$current_time \
    |& tee /nas/team_cx/jingee/finetuning_codes/logs/qwen_finetune_$current_time.log

echo "Start: $START_TIME"
echo "End: $(TZ="Asia/Seoul" date)"
