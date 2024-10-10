#!/bin/bash

START_TIME=$(TZ="Asia/Seoul" date)
current_time=$(date +"%y%m%d_%H%M%S")

TRANSFORMERS_VERBOSITY=info accelerate launch \
    --config_file config.yaml \
    trainer.py \
    --model /nas/team_cx/checkpoints/internlm2_5-20b-chat-unfused/ \
    --dataset agileloop/izaz-sequence-of-actions-prediction-dataset-llama2-7b-32k \
    --lr 0.0001 \
    --train-batch-size 16 \
    --eval-batch-size 1 \
    --block-size 32768 \
    --num-epochs 1 \
    --max-steps 20 \
    --log-interval 20 \
    --output-dir internlm_finetuned_$current_time \
    |& tee logs/internlm_$current_time.log

echo "Start: $START_TIME"
echo "End: $(TZ="Asia/Seoul" date)"
