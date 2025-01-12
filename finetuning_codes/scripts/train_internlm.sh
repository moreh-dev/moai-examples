#!/bin/bash

START_TIME=$(TZ="Asia/Seoul" date)
current_time=$(date +"%y%m%d_%H%M%S")

TOKENIZERS_PARALLELISM=false TRANSFORMERS_VERBOSITY=info accelerate launch \
    --config_file $CONFIG_PATH \
    train.py \
    --model internlm/internlm2_5-20b-chat \
    --dataset agileloop/izaz-sequence-of-actions-prediction-dataset-llama2-7b-32k  \
    --lr 0.0001 \
    --train-batch-size 64 \
    --eval-batch-size 16 \
    --block-size 1024 \
    --num-epochs 1 \
    --max-steps -1 \
    --log-interval 20 \
    --save-path $SAVE_DIR \
    |& tee $LOG_DIR

echo "Start: $START_TIME"
echo "End: $(TZ="Asia/Seoul" date)"
