"""Example Python client for vllm.entrypoints.api_server
NOTE: The API server is used only for demonstration and simple performance
benchmarks. It is not intended for production use.
For production use, we recommend vllm.entrypoints.openai.api_server
and the OpenAI client API
"""

import argparse
import time
import argparse
import asyncio
import time

import yaml
import logging
import asyncio
import logging
from typing import Any

from openai import OpenAI
from tenacity import retry, stop_after_attempt
from client_utils import get_server_config, get_model_config
import openai

logger = logging.getLogger(__name__)

def log_retry(retry_state):
    logging.info("Retrying %s: attempt #%s ended with: %s", retry_state.fn, retry_state.attempt_number, retry_state.outcome)

class Llm:
    def __init__(self, flags):
        self.max_retries = flags.max_retry
        self.flags = flags
        self.timeout = flags.stream_timeout
        base_url = get_server_config()
        model = get_model_config()
        if base_url == False:
            base_url = self.flags.url
        self.model = model if model else flags.model
        self.url = base_url
        self.model_post_init(None)
        self.result = {}
        # Test Response
        self._client.completions.create(
            prompt="hi", model=self.model, max_tokens=10
        )


    def model_post_init(self, __context: Any) -> None:
        """creates openai client after class is instantiated"""
        self._client = OpenAI(max_retries=self.max_retries, base_url= self.url + "/v1", api_key=self.flags.api_token)
    
    def chat(self):
        message_list = []
        seperator = '=' * 80
        while True:
            print("[INFO] Type 'quit' to exit")
            user_prompt = input("Prompt : ")
            if user_prompt.lower() == 'quit':
                break
            message_list = [{"role": "user", "content": user_prompt}]
            print(seperator)
            print("Assistant : ")
            response = self.run_completions(message_list, stream=True)
            print()
            print(seperator)


    def run_completions(self, messages, **kwargs) -> list:
        """runs completions synchronously"""
        response=self._client.chat.completions.create(
            messages=messages, model=self.model, **kwargs
        )
        if "stream" in kwargs and kwargs["stream"]:
            response = self._parse_stream(response)
        return response
    
    def _parse_stream(self, stream):
        """parses stream response from completions"""
        response = {"role": "assistant", "content": None, "tool_calls": None}
        for chunk in stream:
            choice = chunk.choices[0]
            if choice.delta and choice.delta.content:
                self._parse_delta_content(choice.delta, response)
            elif choice.delta and choice.delta.tool_calls:
                self._parse_delta_tools(choice.delta, response)
        return response
    
    
    @retry(stop=stop_after_attempt(3), after=log_retry)
    async def _run_async_completions(self, client, messages, id, **kwargs):
        """runs completions asynchronously"""
        response = await client.completions.create(
            prompt=messages, model=self.flags.model, **kwargs
        )
        if "stream" in kwargs and kwargs["stream"]:
            response = await self._parse_async_stream(response, id)
        return response

    async def _parse_async_stream(self, stream, id, **kwargs):
        """parses stream response from async completions"""
        response = {"role": "assistant", "content": None, "tool_calls": None, "first_response_time" : None, "id" : id}
        async for chunk in stream:
            choice = chunk.choices[0]
            # print(choice)
            if choice.delta and choice.delta.content:
                self._parse_delta_content(choice.delta, response, **kwargs)
            elif choice.delta and choice.delta.tool_calls:
                self._parse_delta_tools(choice.delta, response, **kwargs)
        return response
    
    async def _run_batches(self, coroutines: list):
        for batch in self._batches(coroutines, self.batch_size):
            yield await asyncio.gather(*batch)
    
    def _batches(self, items, batch_size):
        for i in range(0, len(items), batch_size):
            yield items[i : i + batch_size]

    def _parse_delta_content(self, delta, response, **kwargs):
        if response["content"] is None:
            response["content"] = ""
            response["first_response_time"] = time.perf_counter()
        
        response["content"] += delta.content
        print(delta.content, end="", flush=True)
            
    def _parse_delta_tools(self, delta, response,  **kwargs):
        if response["tool_calls"] is None:
            response["tool_calls"] = []
        
        for tchunk in delta.tool_calls:
            if len(response["tool_calls"]) <= tchunk.index:
                response["tool_calls"].append(
                    {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }
                )
            if tchunk.id:
                response["tool_calls"][tchunk.index]["id"] += tchunk.id
            if tchunk.function.name:
                response["tool_calls"][tchunk.index]["function"][
                    "name"
                ] += tchunk.function.name
            if tchunk.function.arguments:
                response["tool_calls"][tchunk.index]["function"][
                    "arguments"
                ] += tchunk.function.arguments



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-token", type=str, default = "ffff")
    parser.add_argument("--model", type=str)
    parser.add_argument("--stream-timeout", type=float, default=None)
    parser.add_argument("--url", type=str, default = None)
    parser.add_argument("--max_retry", type=int, default = 5)
    FLAGS = parser.parse_args()
    llm = Llm(FLAGS)
    llm.chat()
