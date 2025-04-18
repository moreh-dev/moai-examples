# Inference 

## Prerequisite

Please contact the owner of the MoAI platform you wish to use for instructions on how to create an endpoint.

## Supported Models

<div align="center" style="margin-top: 1rem;">


| Supported Models                                             | Model Max Length | TP Size |
| ------------------------------------------------------------ | ---------------: | :-----: |
| [Qwen/Qwen-14B-Chat](https://huggingface.co/Qwen/Qwen-14B-Chat) |             2048 |    1    |
| [Qwen/Qwen-72B-Chat](https://huggingface.co/Qwen/Qwen-72B-Chat) |             2048 |    4    |
| [Qwen/Qwen2-72B-Instruct](https://huggingface.co/Qwen/Qwen2-72B-Instruct) |            32768 |    4    |
| [Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8) |            32768 |    1    |
| [baichuan-inc/Baichuan-13B-Chat](https://huggingface.co/baichuan-inc/Baichuan-13B-Chat) |             4096 |    1    |
| [baichuan-inc/Baichuan2-13B-Chat](https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat) |             4096 |    1    |
| [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) |             8192 |    1    |
| [meta-llama/Meta-Llama-3-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct) |             8192 |    4    |
| [internlm/internlm2_5-20b-chat](https://huggingface.co/internlm/internlm2_5-20b-chat) |            32768 |    1    |
| [google/gemma-2-27b-it](https://huggingface.co/google/gemma-2-27b-it) |             4096 |    2    |
| [THUDM/chatglm3-6b](https://huggingface.co/THUDM/chatglm3-6b) |             8192 |    1    |
| [THUDM/chatglm3-6b-32k](https://huggingface.co/THUDM/chatglm3-6b-32k) |            32768 |    1    |
| [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) |            32768 |    1    |

</div>


## Endpoint Command

The following is the command to create an endpoint. Most models use the same command, but some models may require additional settings.  
The command includes several environment variables:

- `$MODEL`: the name or path of the model
- `$MODEL_MAX_LEN`: the maximum context length supported by the model
- `$TP_SIZE`: tensor parallel size, must match the number of GPUs used

Please note each models may require different values for these variables. Be sure to adjust them accordingly depending on the model you are using.

```
python -m vllm.entrypoints.openai.api_server --model $MODEL --max-model-len $MODEL_MAX_LEN --trust-remote-code --tensor-parallel-size $TP_SIZE --gpu-memory-utilization 0.95 --quantization None --block-size 16 --max-num-batched-tokens $MODEL_MAX_LEN --enable-chunked-prefill False
```


Commands for these specific models are described separately below:  

- `baichuan-inc/Baichuan-13B-Chat`, `baichuan-inc/Baichuan2-13B-Chat` : Baichuan models require a separate chat template when deployed.

```
python -m vllm.entrypoints.openai.api_server --model $MODEL --max-model-len $MODEL_MAX_LEN --trust-remote-code --tensor-parallel-size $TP_SIZE --gpu-memory-utilization 0.95 --quantization None --block-size 16 --max-num-batched-tokens $MODEL_MAX_LEN --enable-chunked-prefill False --chat-template /app/vllm/examples/template_baichuan.jinja
```

## Request Command

The following is an example of a request command using `curl`. This format is compatible with the OpenAI Chat Completions API.

```
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

```
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

Request formats for specific models are described separately below:

 - `google/gemma2-27b-it` : Gemma does not support a “system” role in the chat template. Including it may raise the `jinja2.exceptions.TemplateError: System role not supported` error. Make sure to remove the system message from your request as below

```
...
"model": "google/gemma-2-27b-it",
    "messages": [
        {"role": "user", "content": "Who won the world series in 2020?"}
    ],
    "max_tokens": 200
...
```

- `Qwen/Qwen-14B-Chat`, `Qwen/Qwen-72B-Chat`: Initial versions of Qwen models require `top_k` to be included in the request.

```
curl $ENDPOINT_URL/v1/chat/completions -H "Content-Type: application/json"   -d '{
    "model": "/model/Qwen-14B-Chat",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"}
    ],
    "max_tokens": 200,
    "top_k": 1
}'
```