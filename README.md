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

### Training

To fine-tune the model, run the training script as follows:

```
cd moai-examples
bash finetuning_codes/scripts/train_{model}.sh
```
By specifying one of the models listed under **supported models** in {model}, you can also experiment with other examples.

**CURRENTLY SUPPORTED MODELS:**

- `baichuan`
- `qwen_14b`
- `qwen_72b`
- `internlm`
- `llama_8b`

> For training Qwen, additional environment setup is required using the following command:
> ```bash
> pip install -r requirements/requirements_qwen.txt
> ```

The scripts are as follows:

```bash
#!/bin/bash

# example of train_qwen.sh

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
|    `baichuan`    |  `24.9.211` |   `1.13.1`    | `4xLarge.2048GB` |     64      |     16     |
|    `qwen_14b`    |  `24.9.211` |   `1.13.1`    |  `xLarge.512GB`  |     64      |     16     |
|    `qwen_72b`    |  `24.9.211` |   `1.13.1`    | `4xLarge.2048GB` |     256     |     8      |
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
