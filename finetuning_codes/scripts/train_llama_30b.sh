#!/bin/bash

START_TIME=$(TZ="Asia/Seoul" date)
CURR_TIME=$(date +"%y%m%d_%H%M%S")

CONFIG_PATH=config.yaml
MODEL=huggyllama/llama-30b
SAVE_DIR=../checkpoints/llama-30b
LOG_DIR=logs

mkdir -p $SAVE_DIR $LOG_DIR

export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=info
export ACCELERATOR_PLATFORM_FLAVOR=flavor-default-16

uv run accelerate launch \
    --config_file $CONFIG_PATH \
    train.py \
    --model $MODEL \
    --dataset bitext/Bitext-customer-support-llm-chatbot-training-dataset \
    --lr 0.00001 \
    --train-batch-size 32 \
    --eval-batch-size 16 \
    --block-size 1024 \
    --num-epochs 5 \
    --max-steps -1 \
    --log-interval 20 \
    --save-path $SAVE_DIR \
    |& tee $LOG_DIR/$CURR_TIME.log

echo "Start: $START_TIME"
echo "End: $(TZ="Asia/Seoul" date)"
