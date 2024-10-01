#!/bin/bash

START_TIME=$(TZ="Asia/Seoul" date)
current_time=$(date +"%y%m%d_%H%M%S")

TRANSFORMERS_VERBOSITY=info accelerate launch \
    --config_file config.yaml \
    train.py \
    --model /nas/team_cx/checkpoints/internlm2_5-20b-chat-unfused/ \
    --dataset bitext/Bitext-customer-support-llm-chatbot-training-dataset \
    --lr 0.00005 \
    --train-batch-size 16 \
    --eval-batch-size 16 \
    --block-size 32768 \
    --num-epochs 1 \
    --max-steps 20 \
    --log-interval 1 \
    --output-dir internlm_finetuned_$current_time \
    |& tee logs/internlm_$current_time.log

echo "Start: $START_TIME"
echo "End: $(TZ="Asia/Seoul" date)"
