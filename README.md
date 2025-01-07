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

The **moai-examples** repository is designed to work with a cluster where the MoAI Platform is installed. To test these scripts, please contact us.

### Checking Pytorch Installation
After connecting to the container via SSH, run the following command to check if PyTorch is installed in the current conda environment:

```bash
conda list torch
...
# Name                    Version                   Build  Channel
torch                     1.13.1+cu116.moreh24.9.211          pypi_0    pypi
...
```
The version name includes both the PyTorch version and the MoAI version required to run it.
In the example above, it indicates that PyTorch `1.13.1+cu116` is installed with MoAI version `24.9.211`.

If the moreh version is not `24.9.211` but a different version, please execute the following code.

```bash
$ update-moreh --target 24.9.211 --torch 1.13.1
Currently installed: 24.8.0
Possible upgrading version: 24.9.211
  
Do you want to upgrade? (y/n, default:n)
y
```
You can follow the same procedure when installing a different version.

### Setting MoAI Accelerator

Use `moreh-smi` to check the current flavor. If the torch installation was successful, you should see the following output:
```bash
moreh-smi

+-----------------------------------------------------------------------------------------------+
|                                          Current Version: 24.9.211  Latest Version: 25.1.201  |
+-----------------------------------------------------------------------------------------------+
|  Device  |          Name          |  Model  |  Memory Usage  |  Total Memory  |  Utilization  |
+===============================================================================================+
|  * 0     |  Ambre AI Accelerator  |  micro  |  -             |  -             |  -            |
+-----------------------------------------------------------------------------------------------+
```
If the current version is displayed differently, please repeat the steps in the Checking PyTorch Installation section.

The current MoAI Accelerator in use is `micro`.

You can utilize the `moreh-switch-model` command to review the available accelerator flavors on the current system. For seamless model training, consider using the moreh-switch-modelcommand to switch to a MoAI Accelerator with larger memory capacity.

```bash
moreh-switch-model
Current Ambre AI Accelerator: micro

1. micro  *
2. small
3. large
4. xlarge
5. 2xlarge
6. 3xlarge
7. 4xlarge
8. 8xlarge

Selection (1-8, q, Q):
```

You can enter the number to switch to a different flavor.

As an example, we will train `qwen_14b`. For this, let’s select the `xlarge`-sized MoAI Accelerator.  
Enter 4 to use `xlarge`

```bash
Selection (1-8, q, Q): 4
The Ambre AI Accelerator model is successfully switched to  "xlarge".

1. micro
2. small
3. large
4. xlarge  *
5. 2xlarge
6. 3xlarge
7. 4xlarge
8. 8xlarge

Selection (1-8, q, Q):
```
Enter `q` to complete the change.

To confirm that the changes have been successfully applied, use the moreh-smi command again to check the currently used MoAI Accelerator.
```bash
moreh-smi

+------------------------------------------------------------------------------------------------+
|                                           Current Version: 24.9.211  Latest Version: 25.1.201  |
+------------------------------------------------------------------------------------------------+
|  Device  |          Name          |   Model  |  Memory Usage  |  Total Memory  |  Utilization  |
+================================================================================================+
|  * 0     |  Ambre AI Accelerator  |  xlarge  |  -             |  -             |  -            |
+------------------------------------------------------------------------------------------------+
```


### Training

To fine-tune the model, run the training script as follows:

```
cd moai-examples/finetuning_codes
pip install -r requirments.txt
bash scripts/train_{model}.sh
```
By specifying one of the models listed under **supported models** in {model}, you can also experiment with other examples.

**CURRENTLY SUPPORTED MODELS:**

- `qwen_14b`
- `qwen_72b`
- `baichuan`
- `internlm`
- `llama_8b`

> For training Qwen, additional environment setup is required using the following command:
> ```bash
> pip install -r requirements/requirements_qwen.txt
> ```


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



**Recommended Specifications**

The optimized versions of MAF, Torch, and Flavor for each model are as follows:

|      model       | MAF Version | Torch Version |      Flavor      | Train Batch | Eval Batch |
| :--------------: | :---------: | :-----------: | :--------------: | :---------: | :--------: |
|    `qwen_14b`    |  `24.9.211` |   `1.13.1`    |  `xLarge.512GB`  |     64      |     16     |
|    `qwen_72b`    |  `24.9.211` |   `1.13.1`    | `4xLarge.2048GB` |     256     |     8      |
|    `baichuan`    |  `24.9.211` |   `1.13.1`    |  `xLarge.512GB`  |     64      |     16     |
|    `internlm`    |  `24.9.212` |   `1.13.1`    | `2xLarge.1024GB` |     64      |     16     |
|    `llama_8b`    |  `24.9.211` |   `1.13.1`    |  `xLarge.512GB`  |     64      |     16     |

The detailed fine-tuning parameters are included in the script.


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
├── train_internlm.py             # Fine-tuning code for InternLM training
└── utils.py                      # Utility functions for train.py/train_internlm.py
```


## Learn More

| **Section**       | **Description**                                 |
|-------------------|-------------------------------------------------|
| **[Portal](https://moreh.io/)**        | Overview of technologies and company            |
| **[Documentation](https://docs.moreh.io/)** | Detailed explanation of technology and tutorial |
| **[ModelHub](https://model-hub.moreh.io/)**     | Chatbot using the MoAI Platform solution        |


---
