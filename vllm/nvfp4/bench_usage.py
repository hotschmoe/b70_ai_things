#!/usr/bin/env python3
"""Small decode, TTFT, and prefill probe for an OpenAI-compatible endpoint."""

import json
import os
import sys
import time
import urllib.request


EP = os.environ.get("B70_ENDPOINT", "http://127.0.0.1:18080").rstrip("/")
API_KEY = os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY", "")
LABEL = sys.argv[1] if len(sys.argv) > 1 else "cfg"


def headers():
    result = {"Content-Type": "application/json"}
    if API_KEY:
        result["Authorization"] = "Bearer " + API_KEY
    return result


def model_id():
    req = urllib.request.Request(EP + "/v1/models", headers=headers())
    with urllib.request.urlopen(req, timeout=15) as response:
        return json.load(response)["data"][0]["id"]


MID = model_id()


def post(path, body, timeout=180):
    req = urllib.request.Request(
        EP + path,
        data=json.dumps(body).encode(),
        headers=headers(),
    )
    return urllib.request.urlopen(req, timeout=timeout)


def decode_ts(prompt, n=400):
    start = time.time()
    response = post(
        "/v1/completions",
        {
            "model": MID,
            "prompt": prompt,
            "max_tokens": n,
            "temperature": 0,
            "ignore_eos": True,
        },
    )
    data = json.load(response)
    elapsed = time.time() - start
    tokens = data["usage"]["completion_tokens"]
    return tokens / elapsed, tokens, elapsed


def ttft(prompt):
    start = time.time()
    response = post(
        "/v1/completions",
        {
            "model": MID,
            "prompt": prompt,
            "max_tokens": 16,
            "temperature": 0,
            "stream": True,
        },
    )
    for line in response:
        text = line.decode("utf-8", "ignore").strip()
        if text.startswith("data:") and text != "data: [DONE]":
            try:
                data = json.loads(text[5:])
                if data["choices"][0].get("text"):
                    return time.time() - start
            except (KeyError, json.JSONDecodeError):
                pass
    return time.time() - start


def prefill_pp(prompt):
    start = time.time()
    response = post(
        "/v1/completions",
        {"model": MID, "prompt": prompt, "max_tokens": 1, "temperature": 0},
    )
    data = json.load(response)
    elapsed = time.time() - start
    prompt_tokens = data["usage"]["prompt_tokens"]
    return prompt_tokens / elapsed, prompt_tokens, elapsed


nonce = str(os.getpid())
short = (
    "<|im_start|>user\n"
    + nonce
    + " Hello.<|im_end|>\n<|im_start|>assistant\n"
)
code_prompt = (
    "<|im_start|>user\n"
    + nonce
    + " Write a Python function that merges two sorted linked lists, with a "
    "docstring and type hints. Then explain the time complexity."
    "<|im_end|>\n<|im_start|>assistant\n"
)
long_prompt = (
    "<|im_start|>user\n"
    + (nonce + " The quick brown fox jumps over the lazy dog. " * 1400)
    + "\nSummarize.<|im_end|>\n<|im_start|>assistant\n"
)

print(f"### {LABEL}  served={MID}", flush=True)
decode_ts(short, 32)
decode_tps, count, elapsed = decode_ts(short, 400)
print(
    f"decode_tg (generic, 400tok): {decode_tps:.1f} t/s   "
    f"({count} in {elapsed:.1f}s)",
    flush=True,
)
code_tps, count, elapsed = decode_ts(code_prompt, 400)
print(
    f"decode_tg (coding,  400tok): {code_tps:.1f} t/s   "
    f"({count} in {elapsed:.1f}s)",
    flush=True,
)
first_token = ttft(short)
print(f"TTFT (short prompt):        {first_token * 1000:.0f} ms", flush=True)
prefill, prompt_tokens, elapsed = prefill_pp(long_prompt)
print(
    f"prefill PP ({prompt_tokens} tok):     {prefill:.0f} tok/s  "
    f"(TTFT {elapsed:.2f}s)",
    flush=True,
)
