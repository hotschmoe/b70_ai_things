#!/usr/bin/env python3
"""Live API checks for a real hotschmoe-dd endpoint (the GPU window). Adds to dd_functional.py: streaming tool calls,
reasoning_content in stream and non-stream, thinking-budget override, stop/length finish reasons, queueing past --parallel,
context overflow shape, cached_tokens, /v1/models and /health. Standard library only; prints one JSON result.

    live_checks.py --base-url http://127.0.0.1:18080/v1 [--api-key-file F | --no-api-key] [--window 200000] [--queue-n 8]

Exit status 1 when a gate fails. No prompt or response text is printed beyond short heads of fixed test strings.
"""
import argparse
import concurrent.futures as cf
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--base-url", default="http://127.0.0.1:18080/v1")
ap.add_argument("--api-key-file", default="/mnt/vm_8tb/b70/secrets/dd_api_key")
ap.add_argument("--no-api-key", action="store_true")
ap.add_argument("--queue-n", type=int, default=8)
ap.add_argument("--window", type=int, default=0, help="context window to overflow in the shape check (0: skip)")
a = ap.parse_args()
BASE = a.base_url.rstrip("/")
ROOT = BASE[:-3] if BASE.endswith("/v1") else BASE
KEY = "" if a.no_api_key else Path(a.api_key_file).read_text().strip()
TOOLS = [{"type": "function", "function": {"name": "write_file", "description": "Write a file",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                         "required": ["path", "content"]}}},
         {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
NOTHINK = {"enable_thinking": False}


def req(path, body=None, stream=False, timeout=900):
    h = {"Content-Type": "application/json"}
    if KEY:
        h["Authorization"] = "Bearer " + KEY
    r = urllib.request.Request((BASE if path.startswith("/chat") else ROOT) + path, json.dumps(body).encode() if body is not None else None, h)
    try:
        resp = urllib.request.urlopen(r, timeout=timeout)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")
    with resp:
        if not stream:
            return resp.status, json.load(resp)
        events, t0, last, gaps = [], time.time(), None, []
        for line in resp:
            now = time.time()
            if last is not None:
                gaps.append(now - last)
            last = now
            if line.startswith(b"data: "):
                payload = line[6:].strip()
                events.append("[DONE]" if payload == b"[DONE]" else json.loads(payload))
        return resp.status, {"events": events, "max_gap_s": max(gaps) if gaps else 0, "wall_s": time.time() - t0}


def stream_view(events):
    kinds, reasoning, content, calls, finish, usage = [], [], [], {}, None, None
    for ev in events:
        if ev == "[DONE]":
            kinds.append("done")
            continue
        if ev.get("usage"):
            usage = ev["usage"]
        for ch in ev.get("choices", []):
            d = ch.get("delta", {})
            if d.get("reasoning_content"):
                reasoning.append(d["reasoning_content"])
                kinds.append("r")
            if d.get("content"):
                content.append(d["content"])
                kinds.append("c")
            for tc in d.get("tool_calls") or []:
                c = calls.setdefault(tc.get("index", 0), {"name": "", "args": "", "n_args": 0})
                f = tc.get("function", {})
                c["name"] += f.get("name") or ""
                if f.get("arguments"):
                    c["args"] += f["arguments"]
                    c["n_args"] += 1
                kinds.append("t")
            finish = ch.get("finish_reason") or finish
    return {"kinds": kinds, "reasoning": "".join(reasoning), "content": "".join(content), "calls": calls,
            "finish": finish, "usage": usage, "done": bool(kinds) and kinds[-1] == "done"}


res, gates = {}, {}
code, models = req("/models")
ids = [m["id"] for m in models.get("data", [])]
res["models"] = ids
gates["models"] = code == 200 and ids[:1] == ["hotschmoe-dd"] and any("strata-0216" in i for i in ids)
code, _ = req("/health")
gates["health"] = code == 200

# reasoning_content, non-stream and stream (thinking on, the server default)
q = "A train leaves at 9:40 and arrives at 13:05. How many minutes is the trip? Answer with the number only at the end."
code, r = req("/chat/completions", {"model": "hotschmoe-dd", "max_tokens": 2048, "temperature": 0, "messages": [{"role": "user", "content": q}]})
m = r["choices"][0]["message"]
gates["reasoning_content_nonstream"] = bool(m.get("reasoning_content")) and "205" in (m.get("content") or "")
code, r = req("/chat/completions", {"model": "hotschmoe-dd", "max_tokens": 2048, "temperature": 0, "stream": True,
                                    "stream_options": {"include_usage": True}, "messages": [{"role": "user", "content": q}]}, stream=True)
v = stream_view(r["events"])
first_r, first_c = (v["kinds"].index("r") if "r" in v["kinds"] else -1), (v["kinds"].index("c") if "c" in v["kinds"] else -1)
gates["reasoning_content_stream"] = first_r >= 0 and first_c > first_r and "205" in v["content"] and v["done"] and bool(v["usage"]) \
    and v["finish"] == "stop" and (v["usage"] or {}).get("completion_tokens", 0) > 0
res["stream_reasoning_chars"] = len(v["reasoning"])

# thinking budget override: a tiny budget must still answer, with a short reasoning block
code, r = req("/chat/completions", {"model": "hotschmoe-dd", "max_tokens": 1024, "temperature": 0, "reasoning_budget_tokens": 48,
                                    "messages": [{"role": "user", "content": q}]})
m = r["choices"][0]["message"]
res["budget48_reasoning_chars"] = len(m.get("reasoning_content") or "")
gates["thinking_budget_override"] = code == 200 and len(m.get("reasoning_content") or "") < 1200 and bool(m.get("content"))

# thinking off by kwargs and by effort none
code, r = req("/chat/completions", {"model": "hotschmoe-dd", "max_tokens": 64, "temperature": 0, "chat_template_kwargs": NOTHINK,
                                    "messages": [{"role": "user", "content": "Reply with the word ok."}]})
gates["thinking_off_kwargs"] = code == 200 and not r["choices"][0]["message"].get("reasoning_content")

# streaming tool call: name first, then incremental argument deltas that assemble to JSON, finish_reason tool_calls
body = "\n".join(f"line {i:02d}: the quick brown fox jumps over the lazy dog" for i in range(30))
code, r = req("/chat/completions", {"model": "hotschmoe-dd", "max_tokens": 3000, "temperature": 0, "stream": True, "tools": TOOLS,
                                    "tool_choice": "auto", "chat_template_kwargs": NOTHINK, "messages": [{"role": "user", "content":
                                    "Call write_file with path notes.txt and content exactly these 30 lines:\n" + body}]}, stream=True)
v = stream_view(r["events"])
call = v["calls"].get(0, {})
try:
    parsed = json.loads(call.get("args", ""))
except ValueError:
    parsed = None
res["stream_tool"] = {"name": call.get("name"), "arg_deltas": call.get("n_args"), "max_gap_s": round(r["max_gap_s"], 2), "finish": v["finish"]}
gates["stream_tool_call"] = call.get("name") == "write_file" and isinstance(parsed, dict) and parsed.get("path") == "notes.txt" \
    and call.get("n_args", 0) >= 3 and v["finish"] == "tool_calls" and v["done"] and r["max_gap_s"] < 30
code, r2 = req("/chat/completions", {"model": "hotschmoe-dd", "max_tokens": 1024, "temperature": 0, "tools": TOOLS, "tool_choice": "auto",
                                     "chat_template_kwargs": NOTHINK, "messages": [{"role": "user", "content": "Weather in Reno? Use the tool."}]})
ch = r2["choices"][0]
gates["tool_call_finish_reason"] = ch["finish_reason"] == "tool_calls" and ch["message"]["tool_calls"][0]["function"]["name"] == "get_weather"

# finish reasons: length and stop sequence
code, r = req("/chat/completions", {"model": "hotschmoe-dd", "max_tokens": 8, "temperature": 0, "chat_template_kwargs": NOTHINK,
                                    "messages": [{"role": "user", "content": "Count from 1 to 100 separated by commas."}]})
gates["finish_length"] = r["choices"][0]["finish_reason"] == "length" and r["usage"]["completion_tokens"] <= 9
code, r = req("/chat/completions", {"model": "hotschmoe-dd", "max_tokens": 200, "temperature": 0, "stop": ["5"], "chat_template_kwargs": NOTHINK,
                                    "messages": [{"role": "user", "content": "Count from 1 to 20 separated by commas."}]})
gates["stop_sequence"] = r["choices"][0]["finish_reason"] == "stop" and "5" not in (r["choices"][0]["message"].get("content") or "")

# queueing: more concurrent requests than --parallel all complete (no 429/503), none is refused
def one(i):
    t0 = time.time()
    c, rr = req("/chat/completions", {"model": "hotschmoe-dd", "max_tokens": 24, "temperature": 0, "chat_template_kwargs": NOTHINK,
                                      "messages": [{"role": "user", "content": f"Return only the integer: {i + 11} * 7."}]})
    return c, time.time() - t0, (rr.get("choices") or [{}])[0].get("message", {}).get("content", "").strip() if c == 200 else None
with cf.ThreadPoolExecutor(a.queue_n) as ex:
    outs = list(ex.map(one, range(a.queue_n)))
res["queue"] = {"codes": sorted(c for c, _, _ in outs), "max_s": round(max(t for _, t, _ in outs), 1)}
gates["queue_not_reject"] = all(c == 200 for c, _, _ in outs) and all(o == str((i + 11) * 7) for i, (_, _, o) in enumerate(outs))

# cached_tokens present in usage (the value is informational unless a prompt cache is configured)
code, r = req("/chat/completions", {"model": "hotschmoe-dd", "max_tokens": 8, "temperature": 0, "chat_template_kwargs": NOTHINK,
                                    "messages": [{"role": "user", "content": "Say hi."}]})
gates["usage_cached_tokens_key"] = "cached_tokens" in (r.get("usage", {}).get("prompt_tokens_details") or {})
res["cached_tokens"] = (r.get("usage", {}).get("prompt_tokens_details") or {}).get("cached_tokens")

# context overflow shape: 400 whose message names the context window (clients match on it)
if a.window:
    code, r = req("/chat/completions", {"model": "hotschmoe-dd", "max_tokens": 16, "chat_template_kwargs": NOTHINK,
                                        "messages": [{"role": "user", "content": "word " * (a.window + 2000)}]})
    msg = (r.get("error") or {}).get("message", "")
    gates["context_overflow_400"] = code == 400 and "context window" in msg
res["gates"] = gates
print(json.dumps(res, indent=1))
bad = [k for k, ok in gates.items() if not ok]
print("FAILED GATES:", bad if bad else "none")
raise SystemExit(1 if bad else 0)
