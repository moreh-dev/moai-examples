#!/bin/bash

START_TIME=$(TZ="Asia/Seoul" date)
current_time=$(date +"%y%m%d_%H%M%S")

TOKENIZERS_PARALLELISM=false TRANSFORMERS_VERBOSITY=info accelerate launch \
    --config_file $CONFIG_PATH \
    train.py \
    --model Qwen/Qwen2-72B-Instruct \
    --dataset bitext/Bitext-customer-support-llm-chatbot-training-dataset \
    --lr 0.000001 \
    --train-batch-size 32 \
    --eval-batch-size 32 \
    --num-epochs 3 \
    --max-steps 20 \
    --log-interval 10 \
    --save-path $SAVE_DIR \
    |& tee $LOG_DIR

echo "Start: $START_TIME"
echo "End: $(TZ="Asia/Seoul" date)"
