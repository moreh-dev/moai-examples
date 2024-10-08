#!/bin/bash

python train_internlm.py \
  --epochs 1 \
  --train-batch-size 16 \
  --log-interval 5 \
  --block-size 32768 \
  --model-name-or-path /nas/team_cx/checkpoints/internlm2_5-20b-chat \
  --dataset-name-or-path agileloop/izaz-sequence-of-actions-prediction-dataset-llama2-7b-32k \
  --save-path /nas/team_cx/checkpoints/internlm2_5-20b-chat-finetuned
