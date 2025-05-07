<div align="center">
  <h1 style="font-size: 2.5em; color: #3498db;">MoAI Platform</h1>
</div>


<p align="center" style="font-size: 1.2em;">
  <strong>A full-stack infrastructure software from PyTorch to GPUs for the LLM era.</strong><br/>
  <em>Decouple AI infrastructure from specific hardware vendors.</em><br/>
  <em>Virtualize of all GPU/NPUs in a cluster for higher utilization and failover.</em><br/>
  <em>Scale to thousands of GPUs/NPUs with automatic parallelization and optimization.</em><br/>
  <em>Supports any multi-billion or multi-trillion parameter model for training, and serving.</em><br/>
</p>


<hr/>

<p align="center" style="font-size: 1.1em; color: #2ecc71;">
  <strong>🚀 Designed to unlock the full potential of your AI infrastructure!</strong>
</p>


![overview_01](https://github.com/user-attachments/assets/a1d7b9b5-83f6-4844-8f16-fb6a288f54b3)

## QuickStart

The **moai-examples** repository is designed to work with a cluster where the MoAI Platform is installed.  
To test these scripts, please contact us.

**Recommended Specifications**

The optimized versions of MAF, Torch, and Flavor for each model are as follows:

<div align="center">



|                            Model                             | MAF Version | Torch Version | Python Version |       Flavor        | Train Batch | Eval Batch |
| :----------------------------------------------------------: | :---------: | :-----------: | -------------- | :-----------------: | :---------: | :--------: |
| [hugglyllama/llama-30](https://huggingface.co/huggyllama/llama-30b) | `25.4.3016` |    `2.1.0`    | `3.10`         | `flavor-default-16` |     32      |     16     |
| [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) | `25.4.3016` |    `2.1.0`    | `3.10`         | `flavor-default-8`  |     64      |     32     |
| [meta-llama/Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) | `25.4.3016` |    `2.1.0`    | `3.10`         | `flavor-default-8`  |     32      |     8      |
| [meta-llama/Meta-Llama-2-70b-hf](https://huggingface.co/meta-llama/Llama-2-70b-hf) | `25.4.3016` |    `2.1.0`    | `3.10`         | `flavor-default-32` |     256     |     64     |
| [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) | `25.4.3016` |    `2.1.0`    | `3.10`         | `flavor-default-8`  |     64      |     32     |
| [meta-llama/Meta-Llama-3-70B-Instuct](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct) | `25.4.3016` |    `2.1.0`    | `3.10`         | `flavor-default-32` |     256     |     64     |
| [Qwen/Qwen2-72B-Instruct](https://huggingface.co/Qwen/Qwen2-72B-Instruct) | `25.4.3016` |    `2.1.0`    | `3.10`         | `flavor-default-32` |     32      |     32     |

</div>

### Install

[![python](https://img.shields.io/badge/Python-3.10-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)

You can install another version of MAF using the command below.

```bash
pip install torch==2.1.0+moreh25.4.3016 torchvision==0.16.0+cpu
```

### MoAI Accelerator

You can check the current moai version and flavor through `ap-smi`.

```bash
ap-smi

+-----------------------------------------------------------------------------+
|                                     Accelerator platform Version : v0.0.30  |
|              MoAI Framework Version : 25.4.3016 Latest Version : 25.4.3016  |
+---------+---------+------------------------+----------------+---------------+
|      #  |  NAME   |  FLAVOR                |  MEM USAGE(%)  |  GPU UTIL(%)  |
+---------+---------+------------------------+----------------+---------------+
|      0  |  1779   |  flavor-default-8      |  43            |  100          |
+---------+---------+------------------------+----------------+---------------+
```


### Training

To fine-tune the model, run the training script as follows:

```bash
cd moai-examples/finetuning_codes
pip install -r requirments.txt
bash scripts/train_{model}.sh
```

> For training `qwen_14b`, `qwen_72b`, additional environment setup is required using the following command:
>
> ```bash
> pip install -r requirements/requirements_qwen.txt
> ```


By specifying one of the models listed under **example model names** in `{model}`, you can also run other examples.  

<div align="center" style="margin-top: 1rem;">



| **List of Example Models**                                   | Name in `{model}` |
| :----------------------------------------------------------- | :---------------- |
| [hugglyllama/llama-30](https://huggingface.co/huggyllama/llama-30b) | `llama_30b`       |
| [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) | `llama2_7b`       |
| [meta-llama/Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) | `llama2_13b`      |
| [meta-llama/Meta-Llama-2-70b-hf](https://huggingface.co/meta-llama/Llama-2-70b-hf) | `llama2_70b`      |
| [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) | `llama3_8b`       |
| [meta-llama/Meta-Llama-3-70B-Instuct](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct) | `llama3_70b`      |
| [meta-llama/Meta-Llama-3-70B-Instuct](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct) (with LoRA) | `llama3_70b_lora` |
| [Qwen/Qwen2-72B-Instruct](https://huggingface.co/Qwen/Qwen2-72B-Instruct) | `qwen2_72b`       |



</div>

The scripts are as follows:

```bash
#!/bin/bash
# example of llama3_8b

START_TIME=$(TZ="Asia/Seoul" date)
CURR_TIME=$(date +"%y%m%d_%H%M%S")

CONFIG_PATH=config.yaml
MODEL=meta-llama/Meta-Llama-3-8B-Instruct
SAVE_DIR=../checkpoints/llama3_8b
LOG_DIR=logs

mkdir -p $SAVE_DIR $LOG_DIR

export ACCELERATOR_PLATFORM_FLAVOR=flavor-default-8
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=info

accelerate launch \
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
```

The above script is based on execution from the `moai-examples/finetuning_codes` directory.  
If modifications are required, please adjust it to fit the client or platform specifications.   
Additionally, paths such as `CONFIG_PATH` , `SAVE_DIR` and `LOG_DIR` should be updated to match the context of the container in use.

### Inference


Please refer to the [inference_codes/README.md](inference_codes/README.md)



## **Directory and Code Details**

### Repo Structure

The structure of the entire repository is as follows:

```bash
moai-examples
├── README.md                 # Project overview and instructions
├── checkpoints               # Directory to store model checkpoints during finetuning
├── finetuning_codes          # Code related to model fine-tuning
├── git-hooks                 # Git hooks directory for code formatting and other pre/post-commit tasks
├── inference_codes           # Code for running inference with the trained model
└── pretrained_models         # Pretrained weights obtained from Huggingface
```



### `finetuning_codes`

 `finetuning_codes` directory contains train codes, model configs and scripts necessary for fine-tuning.

```bash
finetuning_codes
├── config.yaml                   # Config file for accelerate
├── model                         # Directory containing model-related files
├── requirements                  # Folder for additional dependencies or packages required for fine-tuning
├── scripts                       # Directory containing shell scripts for different fine-tuning setups
├── train.py                      # Main Python script for initiating the fine-tuning process
└── utils.py                      # Utility functions for train.py/train_internlm.py
```




## Learn More

| **Section**                                 | **Description**                          |
| ------------------------------------------- | ---------------------------------------- |
| **[Portal](https://moreh.io/)**             | Overview of technologies and company     |
| **[ModelHub](https://model-hub.moreh.io/)** | Chatbot using the MoAI Platform solution |


---
