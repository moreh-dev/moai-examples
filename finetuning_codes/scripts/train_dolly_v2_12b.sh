#!/bin/bash

START_TIME=$(TZ="Asia/Seoul" date)
current_time=$(date +"%y%m%d_%H%M%S")

export SAVE_DIR='/root/poc/checkpoints/dolly_finetuned_new_run'
export LOG_DIR='/root/poc/logs/dolly_finetune_new.log'

TOKENIZERS_PARALLELISM=false TRANSFORMERS_VERBOSITY=info accelerate launch \
    --config_file config.yaml \
    train.py \
    --model-name-or-path databricks/dolly-v2-12b \
    --dataset-name-or-path fawern/Text-to-sql-query-generation \
    --lr 0.0001 \
    --train-batch-size 64 \
    --eval-batch-size 64 \
    --block-size 1024 \
    --num-epochs 10 \
    --max-steps -1 \
    --log-interval 20 \
    --save-path $SAVE_DIR \
    |& tee $LOG_DIR

echo "Start: $START_TIME"
echo "End: $(TZ="Asia/Seoul" date)"
