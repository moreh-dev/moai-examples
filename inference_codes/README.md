# Inference 

## Prerequisite

Please contact the owner of the MoAI platform you wish to use for instructions on how to create an endpoint.

## Supported Models



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





## Endpoint Command

The following is the command to create an endpoint. Most models use the same command, but some models may require additional settings.  
The command includes several environment variables:

- `$MODEL`: the name or path of the model
- `$MODEL_MAX_LEN`: the maximum context length supported by the model
- `$TP_SIZE`: tensor parallel size, must match the number of GPUs used

Please note each models may require different values for these variables. Be sure to adjust them accordingly depending on the model you are using.

```bash
python -m vllm.entrypoints.openai.api_server --model $MODEL --max-model-len $MODEL_MAX_LEN --trust-remote-code --tensor-parallel-size $TP_SIZE --gpu-memory-utilization 0.95 --quantization None --block-size 16 --max-num-batched-tokens $MODEL_MAX_LEN --enable-chunked-prefill False
```


Commands for these specific models are described separately below:  

- `deepseek-ai/DeepSeek-R1` : Some arguments are unnecessary for DeepSeek.

```bash
python -m vllm.entrypoints.openai.api_server --model deepseek-ai/DeepSeek-R1 --max-model-len 163840 --trust-remote-code --tensor-parallel-size 8 --gpu-memory-utilization 0.95 --quantization None
```

## Request Command

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
