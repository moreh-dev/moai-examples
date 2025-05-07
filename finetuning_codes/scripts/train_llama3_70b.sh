#!/bin/bash

START_TIME=$(TZ="Asia/Seoul" date)
CURR_TIME=$(date +"%y%m%d_%H%M%S")

CONFIG_PATH=config.yaml
#MODEL=meta-llama/Meta-Llama-3-70B-Instuct
MODEL=/root/models/llama3-70b-instruct
SAVE_DIR=../checkpoints/llama3-70b-instruct
LOG_DIR=logs

mkdir -p $SAVE_DIR $LOG_DIR

export ACCELERATOR_PLATFORM_FLAVOR=flavor-default-32
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=info

accelerate launch \
    --config_file $CONFIG_PATH \
    train.py \
    --model $MODEL \
    --dataset bitext/Bitext-customer-support-llm-chatbot-training-dataset \
    --lr 0.0001 \
    --train-batch-size 256 \
    --eval-batch-size 64 \
    --block-size 1024 \
    --num-epochs 3 \
    --max-steps -1 \
    --log-interval 20 \
    --save-path $SAVE_DIR \
    |& tee $LOG_DIR/$CURR_TIME.log

echo "Start: $START_TIME"
echo "End: $(TZ="Asia/Seoul" date)"
