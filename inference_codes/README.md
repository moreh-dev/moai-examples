# Inference 
## Prerequisite
```
pip install -r requirements.txt
```
## Supported Models

<div align="center" style="margin-top: 1rem;">

| Supported Models                      |
|-------------------------------------|
| [Qwen/Qwen-14B](https://huggingface.co/Qwen/Qwen-14B)                          |
| [Qwen/Qwen-72B](https://huggingface.co/Qwen/Qwen-72B)                       |
| [Qwen/Qwen2-72B-Instruct](https://huggingface.co/Qwen/Qwen2-72B-Instruct)               |
| [baichuan-inc/Baichuan-13B-Chat](https://huggingface.co/baichuan-inc/Baichuan-13B-Chat)      |
| [internlm/internlm2_5-20b-chat](https://huggingface.co/internlm/internlm2_5-20b-chat)      |
| [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)          |
| [google/gemma-2-27b-it](https://huggingface.co/google/gemma-2-27b-it)               |
| [THUDM/chatglm3-6b-32k](https://huggingface.co/THUDM/chatglm3-6b-32k)               |
| [Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8) |

</div>


## Model Load 

Below is the code for loading the model onto the inference server. When you run the code, you first choose which of the two model architectures to use.
```bash
(moreh) root@container:~/poc/inference_codes# python agent_client.py 

┌─ Current Server Info. ─┐
│ Model :                │
│ LoRA : False           │
│ Checkpoint :           │
│ Server Status : Idle   │
└────────────────────────┘


========== Supported Model List ==========
 1. Qwen-14B
==========================================

Select Model Number [1-1/q/Q] : 1
Selected Model : Qwen-14B
```

You can select the option by entering a number(ex. {MODEL_NUMBER}). To stop the process, you may simply enter q or Q.

```bash
Select Model Number [1-1/q/Q] : 1
Selected Model : Qwen-14B


========== Select Checkpoints ============
 1. Use Pretrained Model (default model)
 2. Use Your Checkpoint
==========================================

Select Option [1-2/q/Q] : 
```

If the model is selected correctly, the next step is to choose whether to use the pretrained model checkpoint or the fine-tuned model checkpoint. If you have a fine-tuned model, select option 2. If not, choose option 1 to use the checkpoint of the pre-trained model.

### Using Pretrained model checkpoint

If you select option 1, it automatically loads selected pre-trained model checkpoint on your inference server.

```bash
Select Option [1-2/q/Q] : 1

Request has been sent.
 Loading .....
Inference server has been successfully LOADED
```


### Using Fine-tuned model checkpoint

If you select option 2, you will be asked to put the path of your fine-tuned model checkpoint.

```bash
Select Option [1-2/q/Q] : 2

Checkpoint path : /root/poc/checkpoints/
```
For example, if your fine-tuned checkpoint is saved in `/root/poc/checkpoint/qwen_finetuned`, give the checkpoint path as follows then press enter to load your fine-tuned checkpoint model.

```bash
Checkpoint path : /root/poc/checkpoints/qwen_finetuned
Request has been sent.
 Loading .....
Inference server has been successfully LOADED
```

## Human Evaluation by chatting

Once the model has been successfully loaded, you can start a chat by running the client code. Execute the following script to initiate a conversation with the loaded model. This script connects the client interface to the model, allowing you to interact with it through text inputs and receive responses in real time.


```bash
(moreh) root@container:~/poc/inference_codes# python chat.py
[INFO] Type 'quit' to exit
Prompt : Hi
================================================================================
Assistant : 
Hello! How can I assist you today?
```

## Measuring the Inference Performance

If you want to measure the inference performance , you can use `benchmark_client.py` as following: 

```bash
python benchmark_client.py \
--input-len {input_len} \ # Length of the input tokens 
--output-len {output_len} \ # Length of the output tokens
--num-prompts {num_conc_req} \ # Number of requests that run concurrently in a single trial
--num-trial {num_run} \ # Number of trials 
--save-result \ # Whether to save result in .json file 
--result-dir benchmark_result # Path where result file saved 
```

When the script is executed, the results are out as a log as shown below.

```bash
warmup
0th Experiments
100%|_________________________________________________________| 1/1 [01:11<00:00,  11.11s/it]
1th Experiments
100%|_________________________________________________________| 1/1 [01:11<00:00,  11.11s/it]
2th Experiments
100%|_________________________________________________________| 1/1 [01:11<00:00,  11.11s/it]
============ Serving Benchmark Result ============
Successful requests:                     11         
Benchmark duration (s):                  11.11      
Single input token length:               111     
Single output token length:              111       
Total input tokens:                      111       
Total generated tokens:                  111       
Max. generation throughput (tok/s):      11.11     
Max. generation throughput/GPU (tok/s):  11.11     
Max. running requests:                   1      
---------------Time to First Token----------------
Mean TTFT (ms):                          11.11     
Median TTFT (ms):                        11.11     
P99 TTFT (ms):                           11.11     
==================================================
```