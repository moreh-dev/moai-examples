#!/bin/bash

START_TIME=$(TZ="Asia/Seoul" date)
current_time=$(date +"%y%m%d_%H%M%S")
CONFIG_PATH=/root/glm/moai-examples/finetuning_codes/config.yaml
SAVE_DIR=/root/glm/moai-examples/checkpoints/chatglm_$current_time
LOG_DIR=/root/glm/moai-examples/finetuning_codes/logs/chatglm_$current_time.log

TOKENIZERS_PARALLELISM=false TRANSFORMERS_VERBOSITY=info accelerate launch \
    --config_file $CONFIG_PATH \
    train.py \
    --model /root/pretrained_models/chatglm3-6b \
    --dataset bitext/Bitext-customer-support-llm-chatbot-training-dataset \
    --lr 0.00001 \
    --train-batch-size 64 \
    --eval-batch-size 16 \
    --num-epochs 5 \
    --max-steps -1 \
    --log-interval 20 \
    --save-path $SAVE_DIR \
    |& tee $LOG_DIR

echo "Start: $START_TIME"
echo "End: $(TZ="Asia/Seoul" date)"
