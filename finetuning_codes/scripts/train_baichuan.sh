#!/bin/bash

START_TIME=$(TZ="Asia/Seoul" date)
current_time=$(date +"%y%m%d_%H%M%S")

TRANSFORMERS_VERBOSITY=info accelerate launch \
    --config_file /root/ambre-models/finetuning_codes/config.yaml \
    /root/ambre-models/finetuning_codes/train.py \
    --model /model/Baichuan-13B-Base  \
    --dataset bitext/Bitext-customer-support-llm-chatbot-training-dataset \
    --lr 0.000001 \
    --train-batch-size 64 \
    --eval-batch-size 32 \
    --block-size 1024 \
    --num-epochs 3 \
    --max-steps 20 \
    --log-interval 10 \
    --save-path /root/ambre-models/checkpoints/baichuan_finetuned_$current_time \
    |& tee baichuan_$current_time.log

echo "Start: $START_TIME"
echo "End: $(TZ="Asia/Seoul" date)"
