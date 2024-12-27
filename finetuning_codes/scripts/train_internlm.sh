#!/bin/bash

START_TIME=$(TZ="Asia/Seoul" date)
current_time=$(date +"%y%m%d_%H%M%S")

TRANSFORMERS_VERBOSITY=info accelerate launch \
    --config_file /root/poc/finetuning_codes/config.yaml \
    /root/poc/finetuning_codes/train.py \
    --model /root/poc/pretrained_models/internlm2_5-20b-chat \
    --dataset agileloop/izaz-sequence-of-actions-prediction-dataset-llama2-7b-32k  \
    --lr 0.0001 \
    --train-batch-size 16 \
    --eval-batch-size 8 \
    --block-size 32768 \
    --num-epochs 1 \
    --max-steps -1 \
    --log-interval 20 \
    --save-path /root/poccheckpoints/internlm_finetuned_$current_time \
    |& tee /root/poc/finetuning_codes/logs/internlm_finetune_$current_time.log

echo "Start: $START_TIME"
echo "End: $(TZ="Asia/Seoul" date)"
