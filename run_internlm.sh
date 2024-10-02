#!/bin/bash

python train_th.py \
  --epochs 1 \
  --batch-size 16 \
  --log-interval 5 \
  --block-size 32768 \
  --model-name-or-path /nas/team_cx/checkpoints/internlm2_5-20b-chat-unfused \
  --dataset-name-or-path agileloop/izaz-sequence-of-actions-prediction-dataset-llama2-7b-32k \
  --save-path /nas/team_cx/checkpoints/internlm2_5-20b-chat-finetuned
