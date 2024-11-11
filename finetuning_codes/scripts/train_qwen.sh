#!/bin/bash

START_TIME=$(TZ="Asia/Seoul" date)
current_time=$(date +"%y%m%d_%H%M%S")

python /root/poc/finetuning_codes/train_check_data.py \
  --epochs 1 \
  --train-batch-size 16 \
  --log-interval 20 \
  --block-size 1024 \
  --model-name-or-path Qwen/Qwen-14B \
  --dataset-name-or-path alespalla/chatbot_instruction_prompts \
  --save-path /root/poc/checkpoints/qwen-finetuned \
  --max-step 20 \
  |& tee /root/poc/finetuning_codes/logs/qwen_finetune_$current_time.log

echo "Start: $START_TIME"
echo "End: $(TZ="Asia/Seoul" date)"
