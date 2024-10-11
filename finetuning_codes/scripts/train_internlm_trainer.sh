#!/bin/bash

START_TIME=$(TZ="Asia/Seoul" date)
current_time=$(date +"%y%m%d_%H%M%S")

MOREH_ADVANCED_PARALLELIZAION_DETAILED_LOG=2 MOREH_DUMP_PULL_TENSOR=true TRANSFORMERS_VERBOSITY=info accelerate launch \
    --config_file config.yaml \
    train.py \
    --model /nas/team_cx/checkpoints/internlm2_5-20b-chat/ \
    --dataset agileloop/izaz-sequence-of-actions-prediction-dataset-llama2-7b-32k \
    --lr 0.0001 \
    --train-batch-size 16 \
    --eval-batch-size 8 \
    --block-size 32768 \
    --num-epochs 1 \
    --max-steps 20 \
    --log-interval 5 \
    --output-dir /nas/team_cx/jaeyoung/internlm_$current_time \
    |& tee internlm_$current_time.log

echo "Start: $START_TIME"
echo "End: $(TZ="Asia/Seoul" date)"
