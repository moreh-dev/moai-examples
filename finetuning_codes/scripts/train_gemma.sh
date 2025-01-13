#!/bin/bash

START_TIME=$(TZ="Asia/Seoul" date)
current_time=$(date +"%y%m%d_%H%M%S")

TOKENIZERS_PARALLELISM=false TRANSFORMERS_VERBOSITY=info accelerate launch \
    --config_file $CONFIG_PATH \
    train.py \
    --model gemma/gemma-2-27b-it \
    --dataset bitext/Bitext-customer-support-llm-chatbot-training-dataset \
    --lr 0.000001 \
    --train-batch-size 64 \
    --eval-batch-size 32 \
    --block-size 1024 \
    --num-epochs 3 \
    --max-steps 4 \
    --log-interval 2 \
    --save-path $SAVE_DIR \
    |& tee $LOG_DIR

echo "Start: $START_TIME"
echo "End: $(TZ="Asia/Seoul" date)"
