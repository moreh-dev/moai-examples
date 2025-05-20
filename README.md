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

</div>

# QuickStart

The **moai-examples** repository is designed to work with a cluster where the MoAI Platform is installed.  
To test these scripts, please contact us.

## 🔷 Training

### 🔸 Supported Models

**Recommended Specifications** [![python](https://img.shields.io/badge/Python-3.10-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)

The optimized versions of MAF, Torch, and Flavor for each model are as follows:

|                            Model                             | MAF Version | Torch Version | Python Version |       Flavor        | Train Batch | Eval Batch | Name in `{model}` |
| :----------------------------------------------------------: | :---------: | :-----------: | -------------- | :-----------------: | :---------: | :--------: | ----------------- |
| [hugglyllama/llama-30b](https://huggingface.co/huggyllama/llama-30b) | `25.5.3005` |    `2.1.0`    | `3.10`         | `flavor-default-16` |     32      |     16     | `llama_30b`       |
| [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) | `25.5.3005` |    `2.1.0`    | `3.10`         | `flavor-default-8`  |     64      |     32     | `llama2_7b`       |
| [meta-llama/Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) | `25.5.3005` |    `2.1.0`    | `3.10`         | `flavor-default-8`  |     32      |     8      | `llama2_13b`      |
| [meta-llama/Meta-Llama-2-70b-hf](https://huggingface.co/meta-llama/Llama-2-70b-hf) | `25.5.3005` |    `2.1.0`    | `3.10`         | `flavor-default-32` |     256     |     64     | `llama2_70b`      |
| [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) | `25.5.3005` |    `2.1.0`    | `3.10`         | `flavor-default-8`  |     64      |     32     | `llama3_8b`       |
| [meta-llama/Meta-Llama-3-70B-Instuct](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct) | `25.5.3005` |    `2.1.0`    | `3.10`         | `flavor-default-32` |     256     |     64     | `llama3_70b`      |
| [meta-llama/Meta-Llama-3-70B-Instuct](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct) (with LoRA) | `25.5.3005` |    `2.1.0`    | `3.10`         | `flavor-default-8`  |     16      |     16     | `llama3_70b_lora` |
| [Qwen/Qwen2-72B-Instruct](https://huggingface.co/Qwen/Qwen2-72B-Instruct) | `25.5.3005` |    `2.1.0`    | `3.10`         | `flavor-default-32` |     32      |     32     | `qwen2_72b`       |

</div>

### 🔸 Environment Setup

You can simply set up fine-tuning with the command below.

```bash
cd moai-examples/
uv sync
```

**💡 Note:** This repository uses version `25.5.3005` for set A (recommended).

### 🔸 Run a fine-tuning script

To fine-tune the model, run the training script as follows:

```bash
cd finetuning_codes
bash scripts/train_{model}.sh
```


By specifying one of the models listed under **example model names** in `{model}`, you can also run other examples.  

### 🔸 Example: `train_llama3_8b.sh`

```bash
#!/bin/bash
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=info
export ACCELERATOR_PLATFORM_FLAVOR=flavor-default-8

uv run accelerate launch \
        --config_file config.yaml \
        train.py \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --dataset bitext/Bitext-customer-support-llm-chatbot-training-dataset \
        --lr 0.00001 \
        --train-batch-size 64 \
        --eval-batch-size 32 \
        --block-size 1024 \
        --num-epochs 1 \
        --max-steps -1 \
        --log-interval 20 \
        --save-path $SAVE_DIR
```

The above script is based on execution from the `moai-examples/finetuning_codes` directory.  

</div>

## 🔷 Inference

Please contact the owner of the MoAI platform you wish to use for instructions on how to create an endpoint.



### 🔸 Supported Models

<div align="center" style="margin-top: 1rem;">
<table>
  <thead>
    <tr>
      <th rowspan="2" style="text-align: center;">Supported Models</th>
      <th rowspan="2" style="text-align: center;">Model Max Length</th>
      <th colspan="2" style="text-align: center;">TP Size</th>
    </tr>
    <tr>
      <th style="text-align: center;">MI250</th>
      <th style="text-align: center;">MI308x</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: center;"><a href="https://huggingface.co/huggyllama/llama-30b">huggyllama/llama-30b</a></td>
      <td style="text-align: center;">2048</td>
      <td style="text-align: center;">1</td>
      <td style="text-align: center;">1</td>
    </tr>
    <tr>
      <td style="text-align: center;"><a href="https://huggingface.co/meta-llama/Llama-2-7b-chat-hf">meta-llama/Llama-2-7b-chat-hf</a></td>
      <td style="text-align: center;">4096</td>
      <td style="text-align: center;">1</td>
      <td style="text-align: center;">1</td>
    </tr>
    <tr>
      <td style="text-align: center;"><a href="https://huggingface.co/meta-llama/Llama-2-13b-chat-hf">meta-llama/Llama-2-13b-chat-hf</a></td>
      <td style="text-align: center;">4096</td>
      <td style="text-align: center;">1</td>
      <td style="text-align: center;">1</td>
    </tr>
    <tr>
      <td style="text-align: center;"><a href="https://huggingface.co/meta-llama/Meta-Llama-2-70b-hf">meta-llama/Meta-Llama-2-70b-hf</a></td>
      <td style="text-align: center;">4096</td>
      <td style="text-align: center;">4</td>
      <td style="text-align: center;">2</td>
    </tr>
    <tr>
      <td style="text-align: center;"><a href="https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct">meta-llama/Meta-Llama-3-8B-Instruct</a></td>
      <td style="text-align: center;">8192</td>
      <td style="text-align: center;">1</td>
      <td style="text-align: center;">1</td>
    </tr>
    <tr>
      <td style="text-align: center;"><a href="https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instuct">meta-llama/Meta-Llama-3-70B-Instuct</a></td>
      <td style="text-align: center;">8192</td>
      <td style="text-align: center;">4</td>
      <td style="text-align: center;">2</td>
    </tr>
    <tr>
      <td style="text-align: center;"><a href="https://huggingface.co/Qwen/Qwen2-72B-Instruct">Qwen/Qwen2-72B-Instruct</a></td>
      <td style="text-align: center;">32768</td>
      <td style="text-align: center;">4</td>
      <td style="text-align: center;">2</td>
    </tr>
    <tr>
      <td style="text-align: center;"><a href="https://huggingface.co/Qwen/QwQ-32B">Qwen/QwQ-32B</a></td>
      <td style="text-align: center;">40960</td>
      <td style="text-align: center;">4</td>
      <td style="text-align: center;">2</td>
    </tr>
    <tr>
      <td style="text-align: center;"><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1">deepseek-ai/DeepSeek-R1</a></td>
      <td style="text-align: center;">163840</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">8</td>
    </tr>
  </tbody>
</table>
</div>




### 🔸 Endpoint Command

The following is the command to create an endpoint. Most models use the same command, but some models may require additional settings.  
The command includes several environment variables:

- `$MODEL`: the name or path of the model
- `$MODEL_MAX_LEN`: the maximum context length supported by the model
- `$TP_SIZE`: tensor parallel size, must match the number of GPUs used

Please note each models may require different values for these variables. Be sure to adjust them accordingly depending on the model you are using.

```bash
vllm serve $MODEL --max-model-len $MODEL_MAX_LEN --trust-remote-code --tensor-parallel-size $TP_SIZE --gpu-memory-utilization 0.95 --quantization None --block-size 16 --max-num-batched-tokens $MODEL_MAX_LEN --enable-chunked-prefill False
```


Commands for these specific models are described separately below:  

- `deepseek-ai/DeepSeek-R1` : Some arguments are unnecessary for DeepSeek.

```bash
vllm serve deepseek-ai/DeepSeek-R1 --max-model-len 163840 --trust-remote-code --tensor-parallel-size 8 --gpu-memory-utilization 0.95 --quantization None
```

### 🔸 Request Command

The following is an example of a request command using `curl`. This format is compatible with the OpenAI Chat Completions API.

```bash
curl $ENDPOINT_URL/v1/chat/completions -H "Content-Type: application/json" -d '{
    "model": "$MODEL",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"}
    ],
    "max_tokens": 200
}'
```

If the request is sent correctly, you should receive a response similar to the one below.  
Note that the exact response content may vary depending on the model and its configuration.

```json
{
  "id":"chatcmpl-abc123",
  "object":"chat.completion",
  "created":1744950029,
  "model":"$MODEL",
  "choices":[
    {
      "index":0,
      "message":{
        "role":"assistant",
        "reasoning_content":null,
        "content":" The winning team of the 2020 World Series was the Los Angeles Dodgers. They beat the Tampa Bay Rays in the seven-game series, which was held in stringent COVID-19 protocols to ensure the safety of players, staff, and fans.",
        "tool_calls":[]
      },
      "logprobs":null,
      "finish_reason":"stop",
      "stop_reason":null
    }
  ],
  "usage":{
    "prompt_tokens":35,
    "total_tokens":91,
    "completion_tokens":56,
    "prompt_tokens_details":null
  },
  "prompt_logprobs":null
}
```

</div>

## 🔷 MoAI Accelerator

You can check the current moai version and flavor through `ap-smi`.

```bash
ap-smi

+-----------------------------------------------------------------------------+
|                                     Accelerator platform Version : v0.0.30  |
|              MoAI Framework Version : 25.5.3005 Latest Version : 25.5.3005  |
+---------+---------+------------------------+----------------+---------------+
|      #  |  NAME   |  FLAVOR                |  MEM USAGE(%)  |  GPU UTIL(%)  |
+---------+---------+------------------------+----------------+---------------+
|      0  |  1779   |  flavor-default-8      |  43            |  100          |
+---------+---------+------------------------+----------------+---------------+
```

</div>

## 🔷 **Directory and Code Details**

### 🔸 Repo Structure

The structure of the entire repository is as follows:

```bash
moai-examples
├── README.md                 # Project overview and instructions
├── checkpoints               # Directory to store model checkpoints during finetuning
├── finetuning_codes          # Code related to model fine-tuning
├── git-hooks                 # Git hooks directory for code formatting and other pre/post-commit tasks
├── pyproject.toml						# Project metadata
└── uv.lock         					# Lockfile that contains exact information about the proejct's dependencies
```



### 🔸  `finetuning_codes`

 `finetuning_codes` directory contains train codes, model configs and scripts necessary for fine-tuning.

```bash
finetuning_codes
├── config.yaml                   # Config file for accelerate
├── logs 													# Directory for training logs
├── scripts                       # Directory containing shell scripts for different fine-tuning setups
├── train.py                      # Main Python script for initiating the fine-tuning process
└── utils.py                      # Utility functions for train.py
```



## Learn More

| **Section**                                 | **Description**                          |
| ------------------------------------------- | ---------------------------------------- |
| **[Portal](https://moreh.io/)**             | Overview of technologies and company     |
| **[ModelHub](https://model-hub.moreh.io/)** | Chatbot using the MoAI Platform solution |


---
