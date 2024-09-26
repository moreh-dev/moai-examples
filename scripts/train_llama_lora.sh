#!/bin/bash

START_TIME=$(TZ="Asia/Seoul" date)
current_time=$(date +"%y%m%d_%H%M%S")

TRANSFORMERS_VERBOSITY=info accelerate launch \
    --config_file config.yaml \
    train.py \
    --model /nas/team_cx/checkpoints/llama3-70b-instruct/ \
    --dataset bitext/Bitext-customer-support-llm-chatbot-training-dataset \
    --lr 0.0001 \
    --lora \
    --train-batch-size 256 \
    --eval-batch-size 32 \
    --num-epochs 5 \
    --max-steps 20 \
    --log-interval 20 \
    --output-dir llama_lora_finetuned_$current_time \
    |& tee llama_lora_$current_time.log

echo "Start: $START_TIME"
echo "End: $(TZ="Asia/Seoul" date)"
