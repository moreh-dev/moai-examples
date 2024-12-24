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
- `internlm`
- `llama_70b_lora`
- `llama_8b`
- `qwen`

> For training Qwen, additional environment setup is required using the following command:
> ```bash
> pip install -r requirements/requirements_qwen.txt
> ```

### Inference

The MoAI Platform also supports deploying inference servers for your model.

1. Run the script to deploy the model:

    ```
    bash inference_codes/scripts/change_model.sh
    ```

2. Select the model number:

    ```
    Checking agent server status...
    Agent server is normal
    
    ┌───── Current Server Info ────┐
    │ Model : internlm2_5-20b-chat │
    │ LoRA : False                 │
    │ Checkpoint :                 │
    │ Server Status : NORMAL       │
    └──────────────────────────────┘

    ========== Supported Model List ==========
     1. Meta-Llama-3-70B-Instruct
     2. internlm2_5-20b-chat
    ==========================================
    
    Select Model Number [1-2/q/Q]:
    ```

3. Check the server status:

    ```bash
    bash inference_codes/scripts/check_server.sh
    ```

    Example output:

    ```
    2024-10-15 15:34:28.736 | INFO | __main__:check_server:38 - Checking agent server status...
    2024-10-15 15:34:28.754 | INFO | __main__:check_server:41 - Agent server is normal
    
    ┌───── Current Server Info ────┐
    │ Model : internlm2_5-20b-chat │
    │ LoRA : False                 │
    │ Checkpoint :                 │
    │ Server Status : NORMAL       │
    └──────────────────────────────┘
    ```

4. Chat with your model locally or build a chat platform using the API:

    ```bash
    bash inference_codes/scripts/chat.sh
    ```

    Example chat:

    ```
    [INFO] Type 'quit' to exit
    Prompt: hello
    ================================================================================
    Assistant:
    Hello! How can I assist you today?
    ```



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
├── config.yaml                   # Config file for accelerate.
├── model                         # Directory containing model-related files
├── requirements                  # Folder for additional dependencies or packages required for fine-tuning. 
├── scripts                       # Directory containing shell scripts for different fine-tuning setups.     
├── train.py                      # Main Python script for initiating the fine-tuning process.
├── train_internlm.py             # Fine-tuning code for InternLM training.
└── utils.py                      # Utility functions for train.py/train_internlm.py
```



### `inference_codes`

 `inference_codes` directory contains scripts, utilities, and configs required for running inference tasks, benchmarking, and interacting with the server.

```bash
inference_codes
├── agent_client.py              # Contains code for calling server agent.
├── benchmark_client.py          # Code for benchmarking online servering performance.
├── benchmark_result             # Directory to store the results of benchmarking tests.
├── chat.py                      # Code for handling chat-based interactions with the model.
├── client_utils.py              # Utility functions for agent_client.py.
├── prompt.txt                   # Text file containing predefined prompts used during inference.
├── requirements.txt             # Lists the dependencies required to run the inference code.
└── scripts                      # Directory for additional scripts related to inference tasks.
```



## Learn More

| **Section**       | **Description**                                 |
|-------------------|-------------------------------------------------|
| **[Portal](https://moreh.io/)**        | Overview of technologies and company            |
| **[Documentation](https://docs.moreh.io/)** | Detailed explanation of technology and tutorial |
| **[ModelHub](https://model-hub.moreh.io/)**     | Chatbot using the MoAI Platform solution        |


---
