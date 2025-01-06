#!/bin/bash

START_TIME=$(TZ="Asia/Seoul" date)
current_time=$(date +"%y%m%d_%H%M%S")

TRANSFORMERS_VERBOSITY=info accelerate launch \
    --config_file /root/poc/finetuning_codes/config.yaml \
    /root/poc/finetuning_codes/train.py \
    --model /model/gemma-2-27b-it \
    --dataset bitext/Bitext-customer-support-llm-chatbot-training-dataset \
    --lr 0.000001 \
    --train-batch-size 64 \
    --eval-batch-size 32 \
    --block-size 1024 \
    --num-epochs 3 \
    --max-steps 4 \
    --log-interval 2 \
    --save-path /root/poc/checkpoints/gemma_finetuned_$current_time \
    |& tee /root/poc/finetuning_codes/logs/gemma2-27b_$current_time.log

echo "Start: $START_TIME"
echo "End: $(TZ="Asia/Seoul" date)"
