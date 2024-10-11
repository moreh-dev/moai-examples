#!/bin/bash

START_TIME=$(TZ="Asia/Seoul" date)
current_time=$(date +"%y%m%d_%H%M%S")

python train_internlm.py \
  --epochs 1 \
  --train-batch-size 16 \
  --log-interval 5 \
  --block-size 32768 \
  --model-name-or-path /root/poc_2_finetune//pretrained_models/internlm2_5-20b-chat \
  --dataset-name-or-path agileloop/izaz-sequence-of-actions-prediction-dataset-llama2-7b-32k \
  --save-path /root/poc/checkpoints/internlm2_5-20b-chat-finetuned \
  --max-step 20 \
  |& tee logs/internlm_finetune_$current_time.log

echo "Start: $START_TIME"
echo "End: $(TZ="Asia/Seoul" date)"
