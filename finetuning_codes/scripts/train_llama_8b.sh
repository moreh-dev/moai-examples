#!/bin/bash

START_TIME=$(TZ="Asia/Seoul" date)
current_time=$(date +"%y%m%d_%H%M%S")

TRANSFORMERS_VERBOSITY=info accelerate launch \
    --config_file config.yaml \
    train.py \
    --model /root/poc/pretrained_models/llama3-8b-instruct/ \
    --dataset bitext/Bitext-customer-support-llm-chatbot-training-dataset \
    --lr 0.0001 \
    --train-batch-size 8 \
    --eval-batch-size 8 \
    --block-size 1024 \
    --num-epochs 5 \
    --max-steps 20 \
    --log-interval 20 \
    --save-path /root/poc/checkpoints/llama_lora_finetuned_$current_time \
    |& tee /root/poc/finetuning_codes/logs/llama_lora_$current_time.log

echo "Start: $START_TIME"
echo "End: $(TZ="Asia/Seoul" date)"
