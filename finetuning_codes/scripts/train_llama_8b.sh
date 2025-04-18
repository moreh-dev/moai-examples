#!/bin/bash

START_TIME=$(TZ="Asia/Seoul" date)
CURR_TIME=$(date +"%y%m%d_%H%M%S")

CONFIG_PATH=config.yaml
MODEL=meta-llama/Meta-Llama-3-8B
SAVE_DIR=../checkpoints/llama3_8b
LOG_DIR=logs

mkdir -p $SAVE_DIR $LOG_DIR

export ACCELERATOR_PLATFORM_FLAVOR=flavor-default-8
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=info

accelerate launch \
    --config_file $CONFIG_PATH \
    train.py \
    --model $MODEL \
    --dataset bitext/Bitext-customer-support-llm-chatbot-training-dataset \
    --lr 0.00001 \
    --train-batch-size 64 \
    --eval-batch-size 32 \
    --block-size 1024 \
    --num-epochs 5 \
    --max-steps -1 \
    --log-interval 20 \
    --save-path $SAVE_DIR \
    |& tee $LOG_DIR/$CURR_TIME.log

echo "Start: $START_TIME"
echo "End: $(TZ="Asia/Seoul" date)"
