from dataclasses import dataclass, field
import argparse
import asyncio
import json
import os
import random
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
import traceback
import sys
import numpy as np
import re
import io
import pandas as pd
import requests
from tqdm.asyncio import tqdm
import aiohttp
import logging
from client_utils import get_server_config, get_model_config

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

logger = logging.getLogger(__name__)

@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    best_of: int = 1
    use_beam_search: bool = False

@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0  # Time to first token
    itl: List[float] = field(
        default_factory=list)  # List of inter-token latencies
    prompt_len: int = 0
    error: str = ""

def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def remove_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

async def get_metric(url, queue : asyncio.Queue):
    running_pattern = r'vllm:num_requests_running{model_name="[^"]*"}\s*([\d.]+)'
    waiting_pattern = r'vllm:num_requests_waiting{model_name="[^"]*"}\s*([\d.]+)'

    prompt_throughput_patterm = r'vllm:avg_prompt_throughput_toks_per_s{model_name="[^"]*"}\s*([\d.]+)'
    generation_throughput_pattern = r'vllm:avg_generation_throughput_toks_per_s{model_name="[^"]*"}\s*([\d.]+)'
    string = "Running,Waiting,prompt_tps,generation_tps\n"
    while True:
        try:
            response = requests.get(url)
            data = response.content.decode()
            running = re.search(running_pattern, data).group(1)
            pending = re.search(waiting_pattern, data).group(1)
            prompt_throughput = re.search(prompt_throughput_patterm, data).group(1)
            generation_throughput = re.search(generation_throughput_pattern, data).group(1)
            string += f"{running},{pending},{prompt_throughput},{generation_throughput}\n"
            if not queue.empty():
                return string
            else:
                await asyncio.sleep(1)
        except:
             await asyncio.sleep(1)

        
async def async_request_openai_completions(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith(
        "completions"
    ), "OpenAI Completions API URL must end with 'completions'."

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        assert not request_func_input.use_beam_search
        payload = {
            "model": request_func_input.model,
            "prompt": request_func_input.prompt,
            "temperature": 0.0,
            "best_of": request_func_input.best_of,
            "max_tokens": request_func_input.output_len,
            "stream": True,
            "ignore_eos" : True 
        }
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
        }

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(url=api_url, json=payload,
                                    headers=headers) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = remove_prefix(chunk_bytes.decode("utf-8"),
                                              "data: ")
                        if chunk == "[DONE]":
                            latency = time.perf_counter() - st
                        else:
                            data = json.loads(chunk)

                            # NOTE: Some completion API might have a last
                            # usage summary response without a token so we
                            # want to check a token was generated
                            if data["choices"][0]["text"]:
                                timestamp = time.perf_counter()
                                # First token
                                if ttft == 0.0:
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft

                                # Decoding phase
                                output.itl.append(timestamp -
                                                  most_recent_timestamp)

                                most_recent_timestamp = timestamp
                                generated_text += data["choices"][0]["text"]

                    output.generated_text = generated_text
                    output.success = True
                    output.latency = latency
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output


@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    input_throughput: float
    output_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    p99_ttft_ms: float
    mean_tpot_ms: float
    median_tpot_ms: float
    p99_tpot_ms: float
    mean_itl_ms: float
    median_itl_ms: float
    p99_itl_ms: float

def sample_hi_requests(
    num_requests: int,
    prompt_len : int,
    word : str,
    fixed_output_len: Optional[int] = None,
    
) -> List[Tuple[str, int, int]]:
    filtered_dataset=[]
    for i in range(num_requests):
        prompt = word * prompt_len

        filtered_dataset.append((prompt, prompt_len, fixed_output_len))

    return filtered_dataset

async def get_request(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    input_requests = iter(input_requests)
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue
        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


def calculate_metrics(
    input_requests: List[Tuple[str, int, int]],
    outputs: List[RequestFuncOutput],
    dur_s: float,
    output_len : int
) -> Tuple[BenchmarkMetrics, List[int]]:
    actual_output_lens: List[int] = []
    total_input = 0
    completed = 0
    itls: List[float] = []
    tpots: List[float] = []
    ttfts: List[float] = []
    for i in range(len(outputs)):
        if outputs[i].success:
            # We use the tokenizer to count the number of output tokens for all
            # serving backends instead of looking at len(outputs[i].itl) since
            # multiple output tokens may be bundled together
            # Note: this may inflate the output token count slightly
            actual_output_lens.append(output_len)
            total_input += input_requests[i][1]
            if output_len > 1:
                tpots.append(
                    (outputs[i].latency - outputs[i].ttft) / (output_len - 1))
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)
            completed += 1
        else:
            actual_output_lens.append(0)

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2)
    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=sum(actual_output_lens),
        request_throughput=completed / dur_s,
        input_throughput=total_input / dur_s,
        output_throughput=sum(actual_output_lens) / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0) *
        1000,  # ttfts is empty if streaming is not supported by backend
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        p99_ttft_ms=np.percentile(ttfts or 0, 99) * 1000,
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        p99_tpot_ms=np.percentile(tpots or 0, 99) * 1000,
        mean_itl_ms=np.mean(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        p99_itl_ms=np.percentile(itls or 0, 99) * 1000,
    )

    return metrics, actual_output_lens


async def benchmark(
    backend: str,
    api_url: str,
    metric_url :str,
    model_id: str,
    input_requests: List[Tuple[str, int, int]],
    best_of: int,
    use_beam_search: bool,
    request_rate: float,
    disable_tqdm: bool,
    output_len : int
):
    request_func = async_request_openai_completions

    pbar = None if disable_tqdm else tqdm(total=len(input_requests))

    benchmark_start_time = time.perf_counter()
    tasks: List[asyncio.Task] = []
    queue = asyncio.Queue()
    metric_logger = asyncio.create_task(get_metric(metric_url, queue))
    async for request in get_request(input_requests, request_rate):
        prompt, prompt_len, output_len = request
        request_func_input = RequestFuncInput(
            model=model_id,
            prompt=prompt,
            api_url=api_url,
            prompt_len=prompt_len,
            output_len=output_len,
            best_of=best_of,
            use_beam_search=use_beam_search,
        )
        tasks.append(
            asyncio.create_task(
                request_func(request_func_input=request_func_input,
                             pbar=pbar)))
    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)

    if pbar is not None:
        pbar.close()


    benchmark_duration = time.perf_counter() - benchmark_start_time
    await queue.put("done")
    df_string = await metric_logger
    df = pd.read_csv(io.StringIO(df_string))
    return df, outputs, benchmark_duration


def tokenize(model_id, tokenize_url, prompt):
    tokenize_request = { "model" : model_id, "prompt" : prompt}
    tokenize_response = requests.post(tokenize_url, json.dumps(tokenize_request))
    tokenized = json.loads(tokenize_response.content)
    return tokenized['tokens']

def detokenize(model_id, detokenize_url, tokens):
    payload = {
        "model": model_id,
        "tokens": tokens
    }

    response = requests.post(detokenize_url, data = json.dumps(payload))
    return json.loads(bytes.decode(response.content))['prompt']

def main(args: argparse.Namespace):
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    base_url = get_server_config()
    model_id = get_model_config()
    if base_url == False:
        if args.base_url is not None:
            base_url = args.base_url
        else:
            base_url = f"http://{args.host}:{args.port}"
    model_id = model_id if model_id else args.model
    api_url = f"{base_url}{args.endpoint}"
    metric_url = f"{base_url}/metrics"
    detokenize_url = f"{base_url}/detokenize"
    tokenize_url = f"{base_url}/tokenize"

    one_size_word = detokenize(model_id, detokenize_url, [args.default_token])
    target_len_word = one_size_word * (args.input_len)
    raw_tokens = tokenize(model_id, tokenize_url, target_len_word)

    if len(raw_tokens) >  args.input_len:
        over = len(raw_tokens) - args.input_len
        target_len_word = one_size_word * (args.input_len - over)

    current_len = len(tokenize(model_id, tokenize_url, target_len_word)) 
    assert current_len ==  args.input_len, f"current token number {args.default_token} is not suitable for test. try other token number."
    input_requests=[]
    for i in range(args.num_prompts):
        input_requests.append((target_len_word, args.input_len, args.output_len))
    max_generation_tps_list = []
    max_running_request_list = []
    output_list = []
    duration_list = []
    if args.num_prompts == 1:
        # Warmup
        print("warmup")
        asyncio.run(
            benchmark(
                backend='openai',
                api_url=api_url,
                metric_url = metric_url,
                model_id=model_id,
                input_requests=input_requests,
                best_of=args.best_of,
                use_beam_search=args.use_beam_search,
                request_rate=args.request_rate,
                disable_tqdm=True,
                output_len = args.output_len
            ))

    for j in range(args.num_trial):
        print(f"{j}th Experiments")
        df, outputs, benchmark_duration = asyncio.run(
            benchmark(
                backend='openai',
                api_url=api_url,
                metric_url = metric_url,
                model_id=model_id,
                input_requests=input_requests,
                best_of=args.best_of,
                use_beam_search=args.use_beam_search,
                request_rate=args.request_rate,
                disable_tqdm=args.disable_tqdm,
                output_len = args.output_len
            ))
        output_list += outputs
        duration_list.append(benchmark_duration)
        if args.num_prompts == 1:
            maximum_generation_tps = df.sort_values(by = ["Running", "generation_tps"], ascending=False).iloc[0]["generation_tps"]
        else:
            if (df["prompt_tps"] == 0).any():
                maximum_generation_tps = df[(df["prompt_tps"] == 0)].sort_values(by = ["Running", "generation_tps"], ascending=False).iloc[0]["generation_tps"]
            else:
                maximum_generation_tps = df["generation_tps"].max()
        max_generation_tps_list.append(maximum_generation_tps)
        maximum_running_req = df["Running"].max()
        max_running_request_list.append(maximum_running_req)
    max_generation_tps_list = np.array(max_generation_tps_list)
    metrics, actual_output_lens = calculate_metrics(
        input_requests=input_requests * args.num_trial,
        outputs=output_list,
        dur_s=np.array(duration_list).mean(),
        output_len=args.output_len,
    )
    print("{s:{c}^{n}}".format(s=' Serving Benchmark Result ', n=50, c='='))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):",
                                    np.array(duration_list).mean()))
    print("{:<40} {:<10}".format("Single input token length:", args.input_len))
    print("{:<40} {:<10}".format("Single output token length:",
                                 args.output_len))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total generated tokens:",
                                 metrics.total_output))
    print("{:<40} {:<10.2f}".format("Max. generation throughput (tok/s):",
                                    max_generation_tps_list.mean()))
    print("{:<40} {:<10.2f}".format("Max. generation throughput/GPU (tok/s):",
                                    (max_generation_tps_list.mean()/2)))
    print("{:<40} {:<10.2f}".format("Max. running requests:",
                                    np.array(max_running_request_list).mean()))
    # if len(input_requests) == 1:
    print("{s:{c}^{n}}".format(s='Time to First Token', n=50, c='-'))
    print("{:<40} {:<10.2f}".format("Mean TTFT (ms):", metrics.mean_ttft_ms))
    print("{:<40} {:<10.2f}".format("Median TTFT (ms):",
                                        metrics.median_ttft_ms))
    print("{:<40} {:<10.2f}".format("P99 TTFT (ms):", metrics.p99_ttft_ms))

    print("=" * 50)
    benchmark_result = {
        "input_lens": args.input_len,
        "output_lens": args.output_len,
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "average_max_generation_token_per_sec" :max_generation_tps_list.mean(),
        "maximum_running_request" : np.array(max_running_request_list).mean(),
        "max_generation_tps": max_generation_tps_list.tolist(),
        "mean_ttft_ms": metrics.mean_ttft_ms,
        "median_ttft_ms": metrics.median_ttft_ms,
        "p99_ttft_ms": metrics.p99_ttft_ms,
        "ttfts": [output.ttft for output in outputs],
    }
    # Save config and results to json
    if args.save_result:
        result_json: Dict[str, Any] = {}

        # Setup
        current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
        result_json["date"] = current_dt
        result_json["backend"] = "openai"
        result_json["model_id"] = model_id
        result_json["best_of"] = args.best_of
        result_json["use_beam_search"] = args.use_beam_search
        result_json["num_prompts"] = args.num_prompts

        # Metadata
        if args.metadata:
            for item in args.metadata:
                if "=" in item:
                    kvstring = item.split("=")
                    result_json[kvstring[0].strip()] = kvstring[1].strip()
                else:
                    raise ValueError(
                        "Invalid metadata format. Please use KEY=VALUE format."
                    )

        # Traffic
        result_json["request_rate"] = (
            args.request_rate if args.request_rate < float("inf") else "inf")

        # Merge with benchmark result
        result_json = {**result_json, **benchmark_result}

        # Save to file
        base_model_id = model_id.split("/")[-1]
        file_name = f"ambre_TCO_{base_model_id}_{args.input_len}_{args.num_prompts}.json"  #noqa
        if args.result_filename:
            file_name = args.result_filename
        if args.result_dir:
            file_name = os.path.join(args.result_dir, file_name)
        with open(file_name, "w") as outfile:
            json.dump(result_json, outfile, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base url if not using http host and port.",
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/completions",
        help="API endpoint.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Name of the model.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help=
        "Name or path of the tokenizer, if not using the default tokenizer.",
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=1,
        help="Generates `best_of` sequences per prompt and "
        "returns the best one.",
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=128,
        help="Number of prompts to process.",
    )
    parser.add_argument(
        "--num-trial",
        type=int,
        default=1,
        help="Number of trials to get average result.",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=200,
        help="Desired output length of the results.")

    parser.add_argument(
        "--input-len",
        type=int,
        default=None,
        help="Length of the input prompt."
    )
    parser.add_argument(
        "--default-token",
        type=int,
        default=696,
        help="Default token to use."
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize "
        "the request arrival times.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from huggingface",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Specify to disable tqdm progress bar.",
    )
    parser.add_argument(
        "--save-result",
        action="store_true",
        help="Specify to save benchmark results to a json file",
    )
    parser.add_argument(
        "--metadata",
        metavar="KEY=VALUE",
        nargs="*",
        help="Key-value pairs (e.g, --metadata version=0.3.3 tp=1) "
        "for metadata of this run to be saved in the result JSON file "
        "for record keeping purposes.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="Specify directory to save benchmark json results."
        "If not specified, results are saved in the current directory.",
    )
    parser.add_argument(
        "--result-filename",
        type=str,
        default=None,
        help="Specify the filename to save benchmark json results."
        "If not specified, results will be saved in "
        "{backend}-{args.request_rate}qps-{base_model_id}-{current_dt}.json"
        " format.",
    )

    args = parser.parse_args()
    main(args)
