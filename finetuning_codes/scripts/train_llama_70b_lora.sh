#!/bin/bash

START_TIME=$(TZ="Asia/Seoul" date)
current_time=$(date +"%y%m%d_%H%M%S")

TOKENIZERS_PARALLELISM=false TRANSFORMERS_VERBOSITY=info accelerate launch \
    --config_file $CONFIG_PATH \
    train.py \
    --model meta-llama/Meta-Llama-3-70B-Instruct \
    --dataset bitext/Bitext-customer-support-llm-chatbot-training-dataset \
    --lr 0.0001 \
    --use-lora \
    --train-batch-size 16 \
    --eval-batch-size 16 \
    --block-size 1024 \
    --num-epochs 3 \
    --max-steps -1 \
    --log-interval 20 \
    --save-path $SAVE_DIR \
    |& tee $LOG_DIR

echo "Start: $START_TIME"
echo "End: $(TZ="Asia/Seoul" date)"
