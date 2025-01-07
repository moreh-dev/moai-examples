#!/bin/bash

START_TIME=$(TZ="Asia/Seoul" date)
current_time=$(date +"%y%m%d_%H%M%S")

TOKENIZERS_PARALLELISM=false TRANSFORMERS_VERBOSITY=info accelerate launch \
    --config_file $CONFIG_PATH \
    train.py \
    --model Qwen/Qwen-72B \
    --dataset alespalla/chatbot_instruction_prompts \
    --lr 0.0001 \
    --train-batch-size 256 \
    --eval-batch-size 8 \
    --num-epochs 5 \
    --max-steps -1 \
    --log-interval 20 \
    --save-path $SAVE_DIR \
    |& tee $LOG_DIR

echo "Start: $START_TIME"
echo "End: $(TZ="Asia/Seoul" date)"
