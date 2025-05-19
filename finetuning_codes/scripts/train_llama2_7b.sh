#!/bin/bash

START_TIME=$(TZ="Asia/Seoul" date)
CURR_TIME=$(date +"%y%m%d_%H%M%S")

CONFIG_PATH=config.yaml
MODEL=meta-llama/Llama-2-7b-chat-hf
SAVE_DIR=../checkpoints/llama2-7b-instruct
LOG_DIR=logs

mkdir -p $SAVE_DIR $LOG_DIR

export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=info
export ACCELERATOR_PLATFORM_FLAVOR=flavor-default-8

VENV_ROOT=$(command -v uv >/dev/null 2>&1 && uv run python -c 'import sys, os; print(os.path.dirname(os.path.dirname(sys.executable)))' 2>/dev/null)

if [ -n "$VENV_ROOT" ]; then
	EXEC_CMD="uv run accelerate"
else
	EXEC_CMD="accelerate"
fi

export LD_LIBRARY_PATH="${VENV_ROOT}/lib:${LD_LIBRARY_PATH}"

$EXEC_CMD launch \
	--config_file $CONFIG_PATH \
	train.py \
	--model $MODEL \
	--dataset bitext/Bitext-customer-support-llm-chatbot-training-dataset \
	--lr 0.00001 \
	--train-batch-size 64 \
	--eval-batch-size 32 \
	--block-size 1024 \
	--num-epochs 5 \
	--max-steps -1 \
	--log-interval 20 \
	--save-path $SAVE_DIR \
	|& tee $LOG_DIR/$CURR_TIME.log
echo "Start: $START_TIME"
echo "End: $(TZ="Asia/Seoul" date)"
