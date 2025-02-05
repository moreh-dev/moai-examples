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


|                            Model                             | MAF Version | Torch Version | Python Version |      Flavor      | Train Batch | Eval Batch |
| :----------------------------------------------------------: | :---------: | :-----------: | -------------- | :--------------: | :---------: | :--------: |
|    [Qwen/Qwen-14B](https://huggingface.co/Qwen/Qwen-14B)     | `25.1.202`  |    `2.1.0`    | `3.10`         |  `xLarge.512GB`  |     64      |     16     |
|    [Qwen/Qwen-72B](https://huggingface.co/Qwen/Qwen-72B)     | `25.1.202`  |    `2.1.0`    | `3.10`         | `4xLarge.2048GB` |     256     |     8      |
| [Qwen/Qwen2-72B-Instruct](https://huggingface.co/Qwen/Qwen2-72B-Instruct) | `25.1.202`  |    `2.1.0`    | `3.10`         | `4xLarge.2048GB` |     32      |     32     |
| [baichuan-inc/Baichuan-13B-Chat](https://huggingface.co/baichuan-inc/Baichuan-13B-Chat) | `25.1.202`  |    `2.1.0`    | `3.10`         |  `xLarge.512GB`  |     64      |     16     |
| [internlm/internlm2_5-20b-chat](https://huggingface.co/internlm/internlm2_5-20b-chat) | `25.1.202`  |    `2.1.0`    | `3.10`         | `2xLarge.1024GB` |     64      |     16     |
| [meta-llama/Meta-Llama-3-8B ](https://huggingface.co/meta-llama/Meta-Llama-3-8B) | `25.1.202`  |    `2.1.0`    | `3.10`         |  `xLarge.512GB`  |     64      |     32     |
| [meta-llama/Meta-Llama-3-70B-Instuct ](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct) | `25.1.202`  |    `2.1.0`    | `3.10`         | `4xLarge.2048GB` |     256     |     64     |
| [google/gemma-2-27b-it](https://huggingface.co/google/gemma-2-27b-it) | `25.1.202`  |    `2.1.0`    | `3.10`         | `2xLarge.1024GB` |     64      |     32     |

</div>

### Install

[![python](https://img.shields.io/badge/Python-3.10-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)

```bash
pip install torch==2.1.0+moreh25.1.202 torchvision==0.16.0 sympy
```

### MoAI Accelerator

You can check the current moai version and flavor through `moreh-smi`.
```bash
moreh-smi

+-----------------------------------------------------------------------------------------------+
|                                          Current Version: 25.1.202  Latest Version: 25.1.202  |
+-----------------------------------------------------------------------------------------------+
|  Device  |          Name          |  Model  |  Memory Usage  |  Total Memory  |  Utilization  |
+===============================================================================================+
|  * 0     |  Ambre AI Accelerator  |  micro  |  -             |  -             |  -            |
+-----------------------------------------------------------------------------------------------+
```
If they are set differently, please refer to the following links to adjust the torch version and flavor accordingly:
- [How to check pytorch installation on MoAI Platform](https://docs.moreh.io/tutorials/llama3_8b_tutorial/1_prepare_fine-tuning/#checking-pytorch-installation)
- [How to set MoAI accelerator flavor](https://docs.moreh.io/tutorials/llama3_8b_tutorial/1_prepare_fine-tuning/#checking-pytorch-installation)

### Training
To fine-tune the model, run the training script as follows:

```bash
cd moai-examples/finetuning_codes
pip install -r requirments.txt
bash scripts/train_{model}.sh
```
> For training `qwen_14b`, `qwen_72b`, additional environment setup is required using the following command:
> ```bash
> pip install -r requirements/requirements_qwen.txt
> ```


By specifying one of the models listed under **example model names** in `{model}`, you can also run other examples.  

<div align="center" style="margin-top: 1rem;">

|  **List of Example Models**                                                             |  Name in `{model}`|
|-----------------------------------------------------------------------------------------|---------------|
| [Qwen/Qwen-14B](https://huggingface.co/Qwen/Qwen-14B)                                   | `qwen_14b`    |
| [Qwen/Qwen-72B](https://huggingface.co/Qwen/Qwen-72B)                                   | `qwen_72b`    |
| [Qwen/Qwen2-72B-Instruct](https://huggingface.co/Qwen/Qwen2-72B-Instruct)               | `qwen2_72b`   |
| [baichuan-inc/Baichuan-13B-Chat](https://huggingface.co/baichuan-inc/Baichuan-13B-Chat) | `baichuan`    |
| [internlm/internlm2_5-20b-chat](https://huggingface.co/internlm/internlm2_5-20b-chat)   | `internlm`    |
| [meta-llama/Meta-Llama-3-8B ](https://huggingface.co/meta-llama/Meta-Llama-3-8B)        | `llama_8b`    |
| [meta-llama/Meta-Llama-3-70B-Instuct ](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct) | `llama_70b` |
| [google/gemma-2-27b-it](https://huggingface.co/google/gemma-2-27b-it)                   | `gemma`       |

</div>

The scripts are as follows:

```bash
#!/bin/bash
# example of train_qwen_14b.sh
START_TIME=$(TZ="Asia/Seoul" date)
current_time=$(date +"%y%m%d_%H%M%S")

TRANSFORMERS_VERBOSITY=info accelerate launch \
    --config_file $CONFIG_PATH \
    train.py \
    --model Qwen/Qwen-14B \
    --dataset alespalla/chatbot_instruction_prompts \
    --lr 0.0001 \
    --train-batch-size 64 \
    --eval-batch-size 16 \
    --num-epochs 5 \
    --max-steps -1 \
    --log-interval 20 \
    --save-path $SAVE_DIR \
    |& tee $LOG_DIR

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

### `inference_codes`

 `inference_codes` directory contains scripts for model inference. 

```bash
finetuning_codes
├── agent_client.py            # Python script for model loading
├── benchmark_client.py        # Python script to evaluate inference performance 
├── requirements.txt           # Requirements for inference 
├── chat.py                    # Python script for human evaluation of loaded model
└── client_utils.py            # Utility functions for chat.py/benchmark_client.py/agent_client.py
```




## Learn More

| **Section**       | **Description**                                 |
|-------------------|-------------------------------------------------|
| **[Portal](https://moreh.io/)**        | Overview of technologies and company            |
| **[Documentation](https://docs.moreh.io/)** | Detailed explanation of technology and tutorial |
| **[ModelHub](https://model-hub.moreh.io/)**     | Chatbot using the MoAI Platform solution        |


---
