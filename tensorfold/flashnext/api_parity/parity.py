#!/usr/bin/env python3
"""API parity: one request set against Strata's server.py (mock engine) and TensorFold's fake_serve; diff the replies.

    parity.py --strata http://127.0.0.1:P1 --tf http://127.0.0.1:P2 [--only a,b] [--json out.json] [--quiet]

Both servers run on CPU over the scripts in scenarios.py (no GPU, no weights, never a production port). A case's text
carries "@script=NAME" to pick the scripted model output. Each reply is reduced to a "view" (status, content type, key
shapes, assembled values, SSE delta kinds) and the views are diffed. Differences listed in KNOWN are documented and
accepted; anything else is a gap and makes the exit status 1 (with --strict).
"""
import argparse
import concurrent.futures as cf
import http.client
import json
import sys
import time
import urllib.parse

TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
     "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "units": {"type": "string"}},
                    "required": ["city"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Write a file",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["path", "content"]}}},
]
TYPED_TOOL = [{"type": "function", "function": {"name": "configure", "description": "Configure",
              "parameters": {"type": "object", "properties": {
                  "count": {"type": "integer"}, "ratio": {"type": "number"}, "enabled": {"type": "boolean"},
                  "tags": {"type": "array", "items": {"type": "string"}}, "options": {"type": "object"}, "note": {"type": "string"}},
                  "required": ["count"]}}}]
NOTHINK = {"enable_thinking": False}


def chat(text, script, **kw):
    body = {"model": "hotschmoe-dd", "messages": [{"role": "user", "content": f"{text} @script={script}"}]}
    body.update(kw)
    return {"method": "POST", "path": "/v1/chat/completions", "body": body, "stream": bool(kw.get("stream"))}


def msgs(text, script, **kw):
    body = {"model": "hotschmoe-dd", "max_tokens": 512, "messages": [{"role": "user", "content": f"{text} @script={script}"}]}
    body.update(kw)
    return {"method": "POST", "path": "/v1/messages", "body": body, "stream": bool(kw.get("stream"))}


CASES = {
    "models": {"method": "GET", "path": "/v1/models"},
    "models_one": {"method": "GET", "path": "/v1/models/hotschmoe-dd"},
    "health": {"method": "GET", "path": "/health"},
    "think_default": chat("hi", "think"),
    "think_stream": chat("hi", "think", stream=True, stream_options={"include_usage": True}),
    "think_stream_nousage": chat("hi", "think", stream=True),
    "think_off_kwargs": chat("hi", "plain", chat_template_kwargs=NOTHINK),
    "think_off_effort_none": chat("hi", "plain", reasoning_effort="none"),
    "think_effort_low": chat("hi", "think", reasoning_effort="low"),
    "think_effort_high": chat("hi", "think", reasoning_effort="high"),
    "think_effort_bogus": chat("hi", "think", reasoning_effort="bogus"),
    "budget_reasoning_budget_tokens": chat("hi", "think", reasoning_budget_tokens=100),
    "budget_thinking_budget_tokens": chat("hi", "think", thinking_budget_tokens=100),
    "budget_thinking_budget": chat("hi", "think", thinking_budget=100),
    "budget_zero": chat("hi", "think", reasoning_budget_tokens=0),
    "tool_nothink": chat("weather in Boise?", "tool_nothink", tools=TOOLS, chat_template_kwargs=NOTHINK),
    "tool_nothink_stream": chat("weather in Boise?", "tool_nothink", tools=TOOLS, chat_template_kwargs=NOTHINK, stream=True,
                                stream_options={"include_usage": True}),
    "tool_think": chat("weather in Boise?", "tool_think", tools=TOOLS),
    "tool_think_stream": chat("weather in Boise?", "tool_think", tools=TOOLS, stream=True),
    "tool_text_then_call": chat("weather in Boise?", "tool_text_then_call", tools=TOOLS, chat_template_kwargs=NOTHINK),
    "tool_two": chat("weather in Boise and Reno?", "tool_two", tools=TOOLS, chat_template_kwargs=NOTHINK),
    "tool_two_stream": chat("weather in Boise and Reno?", "tool_two", tools=TOOLS, chat_template_kwargs=NOTHINK, stream=True),
    "tool_long_stream": chat("write notes", "tool_long", tools=TOOLS, chat_template_kwargs=NOTHINK, stream=True),
    "tool_long": chat("write notes", "tool_long", tools=TOOLS, chat_template_kwargs=NOTHINK),
    "tool_typed": chat("configure", "tool_typed", tools=TYPED_TOOL, chat_template_kwargs=NOTHINK),
    "tool_typed_stream": chat("configure", "tool_typed", tools=TYPED_TOOL, chat_template_kwargs=NOTHINK, stream=True),
    "tool_choice_auto": chat("weather?", "tool_nothink", tools=TOOLS, tool_choice="auto", chat_template_kwargs=NOTHINK),
    "tool_choice_none": chat("weather?", "plain", tools=TOOLS, tool_choice="none", chat_template_kwargs=NOTHINK),
    "tool_choice_required": chat("weather?", "tool_after_open", tools=TOOLS, tool_choice="required", chat_template_kwargs=NOTHINK),
    "tool_choice_required_think": chat("weather?", "tool_after_open", tools=TOOLS, tool_choice="required"),
    "tool_choice_required_stream": chat("weather?", "tool_after_open", tools=TOOLS, tool_choice="required",
                                        chat_template_kwargs=NOTHINK, stream=True),
    "tool_choice_required_one_tool": chat("weather?", "tool_after_named", tools=TOOLS[:1], tool_choice="required",
                                          chat_template_kwargs=NOTHINK),
    "tool_choice_named": chat("weather?", "tool_after_named", tools=TOOLS, chat_template_kwargs=NOTHINK,
                              tool_choice={"type": "function", "function": {"name": "get_weather"}}),
    "tool_choice_named_stream": chat("weather?", "tool_after_named", tools=TOOLS, chat_template_kwargs=NOTHINK, stream=True,
                                     tool_choice={"type": "function", "function": {"name": "get_weather"}}),
    "tool_choice_named_unknown": chat("weather?", "tool_nothink", tools=TOOLS, chat_template_kwargs=NOTHINK,
                                      tool_choice={"type": "function", "function": {"name": "nope"}}),
    "tool_choice_required_no_tools": chat("hi", "plain", tool_choice="required", chat_template_kwargs=NOTHINK),
    "tool_roundtrip": {"method": "POST", "path": "/v1/chat/completions", "body": {
        "model": "hotschmoe-dd", "tools": TOOLS, "chat_template_kwargs": NOTHINK, "messages": [
            {"role": "user", "content": "weather in Boise? @script=answer_after_tool"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {
                "name": "get_weather", "arguments": json.dumps({"city": "Boise"})}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": json.dumps({"temp_f": 58, "sky": "clear"})}]}},
    "format_json_object": chat("json", "json_object", response_format={"type": "json_object"}, chat_template_kwargs=NOTHINK),
    "format_json_schema": chat("json", "json_object", chat_template_kwargs=NOTHINK, response_format={
        "type": "json_schema", "json_schema": {"name": "w", "strict": True, "schema": {
            "type": "object", "properties": {"city": {"type": "string"}, "temp_f": {"type": "number"}},
            "required": ["city", "temp_f"], "additionalProperties": False}}}),
    "format_json_schema_stream": chat("json", "json_object", chat_template_kwargs=NOTHINK, stream=True, response_format={
        "type": "json_schema", "json_schema": {"name": "w", "schema": {"type": "object"}}}),
    "format_json_bad_output": chat("json", "json_bad", response_format={"type": "json_object"}, chat_template_kwargs=NOTHINK),
    "format_text": chat("hi", "plain", response_format={"type": "text"}, chat_template_kwargs=NOTHINK),
    "stop_string": chat("hi", "stop_words", stop="STOP", chat_template_kwargs=NOTHINK),
    "stop_list": chat("hi", "stop_words", stop=["zzz", "STOP"], chat_template_kwargs=NOTHINK),
    "stop_stream": chat("hi", "stop_words", stop="STOP", stream=True, chat_template_kwargs=NOTHINK),
    "max_tokens_3": chat("hi", "long_text", max_tokens=3, chat_template_kwargs=NOTHINK),
    "max_completion_tokens_3": chat("hi", "long_text", max_completion_tokens=3, chat_template_kwargs=NOTHINK),
    "max_tokens_stream_usage": chat("hi", "long_text", max_tokens=3, stream=True, stream_options={"include_usage": True},
                                    chat_template_kwargs=NOTHINK),
    "n_2": chat("hi", "plain", n=2, chat_template_kwargs=NOTHINK),
    "usage_cached": chat("hi", "think_cached"),
    "ctx_overflow_prompt": chat("word " * 6000, "plain", chat_template_kwargs=NOTHINK),
    "ctx_overflow_max_tokens": chat("hi", "plain", max_tokens=100000, chat_template_kwargs=NOTHINK),
    "ctx_overflow_stream": chat("word " * 6000, "plain", stream=True, chat_template_kwargs=NOTHINK),
    "unknown_model": {"method": "POST", "path": "/v1/chat/completions", "body": {
        "model": "no-such-model", "chat_template_kwargs": NOTHINK, "messages": [{"role": "user", "content": "hi @script=plain"}]}},
    "bad_json": {"method": "POST", "path": "/v1/chat/completions", "raw": b"{not json"},
    "no_messages": {"method": "POST", "path": "/v1/chat/completions", "body": {"model": "hotschmoe-dd"}},
    "empty_messages": {"method": "POST", "path": "/v1/chat/completions", "body": {"model": "hotschmoe-dd", "messages": []}},
    "unicode": chat("hi", "unicode", chat_template_kwargs=NOTHINK),
    "system_multi_turn": {"method": "POST", "path": "/v1/chat/completions", "body": {
        "model": "hotschmoe-dd", "chat_template_kwargs": NOTHINK, "messages": [
            {"role": "system", "content": "You are terse."}, {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"}, {"role": "user", "content": "again @script=plain"}]}},
    "sampled_temp": chat("hi", "plain", temperature=0.7, top_p=0.9, chat_template_kwargs=NOTHINK),
    "sampled_zero": chat("hi", "plain", temperature=0, chat_template_kwargs=NOTHINK),
    "penalty_presence": chat("hi", "plain", presence_penalty=0.5, chat_template_kwargs=NOTHINK),
    "logprobs": chat("hi", "plain", logprobs=True, top_logprobs=2, chat_template_kwargs=NOTHINK),
    "messages_think": msgs("hi", "think", thinking={"type": "enabled", "budget_tokens": 256}),
    "messages_plain": msgs("hi", "plain", thinking={"type": "disabled"}),
    "messages_stream": msgs("hi", "think", stream=True, thinking={"type": "enabled", "budget_tokens": 256}),
    "messages_tool": msgs("weather?", "tool_nothink", tools=[{"name": "get_weather", "description": "w", "input_schema": {
        "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}]),
    "slots": {"method": "GET", "path": "/slots"},
    "props": {"method": "GET", "path": "/props"},
    "completions_raw": {"method": "POST", "path": "/v1/completions", "body": {"model": "hotschmoe-dd", "prompt": "Say hi @script=plain", "max_tokens": 16}},
    "echo_defaults": chat("hi", "echo"),
    "echo_override": chat("hi", "echo", reasoning_budget_tokens=77, max_tokens=99),
    "echo_thinking_budget": chat("hi", "echo", thinking_budget=55),
    "echo_thinking_budget_tokens": chat("hi", "echo", thinking_budget_tokens=33),
    "echo_budget_zero": chat("hi", "echo", reasoning_budget_tokens=0),
    "echo_off": chat("hi", "echo", chat_template_kwargs=NOTHINK),
    "echo_max_completion_tokens": chat("hi", "echo", max_completion_tokens=41),
}
# cases whose assembled text is the mock's echo or otherwise engine-specific: compare status and shape, not values
SHAPE_ONLY = {"max_tokens_3", "max_completion_tokens_3", "max_tokens_stream_usage", "echo_defaults", "echo_override", "echo_thinking_budget",
              "echo_thinking_budget_tokens", "echo_budget_zero", "echo_off", "echo_max_completion_tokens", "models", "models_one", "health", "ctx_overflow_prompt", "ctx_overflow_stream"}
CONCURRENCY = {"n": 8, "case": chat("hi", "slow", chat_template_kwargs=NOTHINK)}


def call(base, case, timeout=60):
    u = urllib.parse.urlparse(base)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=timeout)
    body = case.get("raw") if "raw" in case else (json.dumps(case["body"]).encode() if "body" in case else None)
    headers = {"Content-Type": "application/json"} if body is not None else {}
    t0 = time.time()
    try:
        conn.request(case["method"], case["path"], body, headers)
        r = conn.getresponse()
        data = r.read()
        return {"status": r.status, "ctype": (r.getheader("Content-Type") or "").split(";")[0], "data": data,
                "elapsed": time.time() - t0}
    except Exception as e:  # noqa: BLE001
        return {"status": "EXC", "ctype": "", "data": repr(e).encode(), "elapsed": time.time() - t0}
    finally:
        conn.close()


def sse(data):
    chunks, done, other = [], False, []
    for line in data.decode("utf-8", "replace").splitlines():
        if line.startswith("data: "):
            payload = line[6:]
            if payload.strip() == "[DONE]":
                done = True
            else:
                try:
                    chunks.append(json.loads(payload))
                except ValueError:
                    other.append(line)
        elif line.startswith("event: "):
            other.append(line)
        elif line.strip() and not line.startswith(":"):
            other.append(line)
    return chunks, done, other


def shape(v, depth=0):
    if isinstance(v, dict):
        return {k: shape(x, depth + 1) for k, x in sorted(v.items())}
    if isinstance(v, list):
        return [shape(v[0], depth + 1)] if v else []
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float)):
        return "num"
    return "str"


def keep(v):
    """Values worth comparing exactly inside a shape (enumerations), everything else is a type name."""
    return v


def tool_view(calls):
    out = []
    for c in calls or []:
        fn = c.get("function", {})
        try:
            args = json.loads(fn.get("arguments") or "")
        except ValueError:
            args = "<<not json>>" + str(fn.get("arguments"))[:60]
        out.append({"name": fn.get("name"), "args": args, "id_prefix": (c.get("id") or "")[:5], "type": c.get("type")})
    return out


def view_json(res, case_name):
    v = {"status": res["status"], "ctype": res["ctype"]}
    try:
        body = json.loads(res["data"])
    except ValueError:
        v["raw"] = res["data"][:200].decode("utf-8", "replace")
        return v
    v["shape"] = shape(body)
    if isinstance(body, dict):
        if "error" in body:
            e = body["error"]
            v["error"] = {k: e.get(k) for k in ("type", "code", "param")} if isinstance(e, dict) else {"str": "str"}
            v["error_message_head"] = (e.get("message", "") if isinstance(e, dict) else str(e))[:70]
        ch = (body.get("choices") or [{}])[0]
        m = ch.get("message") or {}
        v["finish_reason"] = ch.get("finish_reason")
        v["content"] = m.get("content")
        v["reasoning_content"] = m.get("reasoning_content")
        v["tool_calls"] = tool_view(m.get("tool_calls"))
        v["n_choices"] = len(body.get("choices") or [])
        u = body.get("usage")
        if isinstance(u, dict):
            v["usage_keys"] = sorted(u)
            v["cached_tokens"] = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
        v["object"] = body.get("object")
        v["model"] = body.get("model")
        if body.get("type") == "message":      # Anthropic
            v["anthropic"] = {"stop_reason": body.get("stop_reason"), "blocks": [b.get("type") for b in body.get("content", [])],
                              "text": "".join(b.get("text", "") for b in body.get("content", []) if b.get("type") == "text"),
                              "thinking": "".join(b.get("thinking", "") for b in body.get("content", []) if b.get("type") == "thinking"),
                              "tool_use": [(b.get("name"), b.get("input")) for b in body.get("content", []) if b.get("type") == "tool_use"],
                              "usage_keys": sorted((body.get("usage") or {}))}
        if "data" in body and isinstance(body["data"], list):
            v["model_ids"] = [d.get("id") for d in body["data"]]
        if "choices" in body and ch.get("text") is not None:
            v["text"] = ch.get("text")
    return v


def view_stream(res, case_name):
    v = {"status": res["status"], "ctype": res["ctype"]}
    if res["status"] != 200 or "event-stream" not in res["ctype"]:
        return view_json(res, case_name)
    chunks, done, other = sse(res["data"])
    kinds, content, reasoning, tools, usage, finish = [], [], [], {}, None, None
    keysets, tool_arg_chunks = set(), 0

    def push(k):
        if not kinds or kinds[-1] != k:
            kinds.append(k)

    anth = any(c.get("type") for c in chunks if "choices" not in c)
    if anth:
        v["events"] = [c.get("type") for c in chunks]
        v["done_marker"] = done
        v["other"] = other[:3]
        text = "".join(c.get("delta", {}).get("text", "") for c in chunks if c.get("type") == "content_block_delta")
        think = "".join(c.get("delta", {}).get("thinking", "") for c in chunks if c.get("type") == "content_block_delta")
        v["text"], v["thinking"] = text, think
        return v
    for c in chunks:
        keysets.add(tuple(sorted(c)))
        if c.get("usage"):
            usage = c["usage"]
            push("usage")
        for ch in c.get("choices", []):
            d = ch.get("delta", {})
            keysets.add(("delta",) + tuple(sorted(d)))
            if "role" in d:
                push("role")
            if d.get("reasoning_content"):
                reasoning.append(d["reasoning_content"])
                push("reasoning")
            if d.get("content"):
                content.append(d["content"])
                push("content")
            for tc in d.get("tool_calls") or []:
                i = tc.get("index", 0)
                t = tools.setdefault(i, {"name": "", "args": "", "id": "", "type": None, "deltas": 0})
                fn = tc.get("function", {})
                if fn.get("name"):
                    t["name"] += fn["name"]
                    t["id"] = tc.get("id", t["id"])
                    t["type"] = tc.get("type", t["type"])
                    push("tool_open")
                if fn.get("arguments"):
                    t["args"] += fn["arguments"]
                    t["deltas"] += 1
                    tool_arg_chunks += 1
                    push("tool_args")
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
                push("finish")
    v["kinds"] = kinds
    v["done_marker"] = done
    v["other"] = other[:3]
    v["finish_reason"] = finish
    v["content"] = "".join(content)
    v["reasoning_content"] = "".join(reasoning)
    v["tool_calls"] = tool_view([{"id": t["id"], "type": t["type"], "function": {"name": t["name"], "arguments": t["args"]}}
                                  for _, t in sorted(tools.items())])
    v["tool_arg_deltas"] = tool_arg_chunks
    v["usage_keys"] = sorted(usage) if usage else None
    v["cached_tokens"] = (usage or {}).get("prompt_tokens_details", {}).get("cached_tokens") if usage else None
    v["chunk_keysets"] = sorted(map(list, keysets))
    return v


def view(case_name, case, res):
    if res["status"] == "EXC":
        return {"status": "EXC", "raw": res["data"].decode()}
    return (view_stream if case.get("stream") else view_json)(res, case_name)


# Differences that are documented and accepted: (case, field) -> reason. field "*" covers the whole case.
KNOWN = {
    ("models", "*"): "listing shape differs by design: TF lists capabilities/context_length (vLLM style), Strata status/meta.n_ctx (llama.cpp "
                     "style); both put hotschmoe-dd first, carry the strata-0216 alias, object=list",
    ("models_one", "*"): "neither serves GET /v1/models/ID (both 404)",
    ("health", "*"): "health bodies differ by design; both 200 with status ok",
    ("n_2", "*"): "TF refuses n>1 with a 400 (honest); Strata silently returns one choice. n=1 is identical",
    ("ctx_overflow_stream", "*"): "TF streams a context-window refusal (role chunk, error event, [DONE]) by upstream design "
                                  "(stream_preflight_test.zig); Strata answers 400 before the stream. Both name the context window",
    ("tool_choice_named_unknown", "*"): "TF 400s a named tool the request does not offer; Strata treats it as auto",
    ("tool_choice_required_no_tools", "*"): "TF 400s required with no tools (OpenAI behaviour); Strata treats it as auto",
    ("logprobs", "*"): "engine capability: the XPU lane engine has no logprobs (Info.logprobs=false); 400 by design",
    ("completions_raw", "*"): "TF also serves /v1/completions; Strata has no raw-completion route (404). No client uses it",
    ("bad_json", "*"): "TF omits error.type for a malformed body; status and message are the same kind",
    ("no_messages", "*"): "message wording differs; both 400",
    ("empty_messages", "*"): "message wording differs; both 400",
    ("think_effort_bogus", "*"): "message wording differs (TF also accepts minimal/xhigh/max, which the template coerces); both 400",
    ("slots", "*"): "TF /slots lists one entry per lane (Strata: one); both carry id, n_ctx, is_processing",
    ("props", "*"): "TF /props is the subset clients read (n_ctx, total_slots, model_alias); Strata also embeds the chat template",
    ("tool_choice_required_think", "*"): "a forced call turn: TF renders it with thinking off and writes the call opening itself (no reasoning "
                                         "block that turn); the Strata mock puts the unclosed script in reasoning (real Strata thinks, then forces the call)",
    ("format_json_bad_output", "*"): "TF's prompt-side response_format does not validate the reply (Strata 502s output that is not JSON); "
                                     "the GPU window gates real compliance (live_checks.py format checks)",
}
# cases that depend on the SAMP item (the engine is greedy-only today): informational until SAMP merges
SAMP_CASES = {"sampled_temp", "penalty_presence"}
EXTRA_TOP_KEYS = {"exact_mode", "speculative", "tensorfold", "stop_sequence"}   # TF adds timing/spec blocks; additive
SKIP_ALWAYS = {"shape", "error_message_head", "chunk_keysets", "ctype"}


def norm(v):
    """Reduce a view to what a client could notice: additive TF extras and wording are dropped."""
    v = dict(v)
    for k in SKIP_ALWAYS:
        v.pop(k, None)
    if "kinds" in v:
        v["kinds"] = [k for k in v["kinds"] if k != "usage"]
    if "usage_keys" in v and v["usage_keys"] is not None:
        v["usage_keys"] = sorted(set(v["usage_keys"]) - {"completion_tokens_details", "output_tokens_details"})
    if isinstance(v.get("anthropic"), dict):
        a = dict(v["anthropic"])
        a["usage_keys"] = sorted(set(a.get("usage_keys", [])) - {"output_tokens_details"})
        v["anthropic"] = a
    if "other" in v:
        v.pop("other")
    if "events" in v:
        v["events"] = [e for i, e in enumerate(v["events"]) if i == 0 or e != v["events"][i - 1]]     # run-length
    if "tool_arg_deltas" in v:
        v["tool_arg_deltas"] = min(2, v["tool_arg_deltas"] or 0)       # incremental or not; the count is chunking
    if "error" in v and isinstance(v["error"], dict):
        v["error"] = {k: x for k, x in v["error"].items() if x is not None}
    return v


def diff(name, a, b, shape_only):
    a, b = norm(a), norm(b)
    out = []
    for k in sorted(set(a) | set(b)):
        if shape_only and k in ("content", "reasoning_content", "tool_calls", "text", "thinking", "anthropic", "cached_tokens", "finish_reason"):
            continue
        x, y = a.get(k), b.get(k)
        if k == "error":                               # TF may add code/param/type where Strata has none
            x, y = (x or {}), (y or {})
            if all(y.get(kk) == vv for kk, vv in x.items()):
                continue
        if k == "cached_tokens" and name == "usage_cached":
            continue                                   # the mock engine has no prompt cache; TF reports the scripted 7
        if k == "model_ids":
            if x and y and x[0] == y[0] and set(x) <= set(y):
                continue
        if k == "content" and name.startswith("format_json") and isinstance(x, str) and isinstance(y, str):
            try:                                       # Strata canonicalizes the JSON it returns; compare the value, not the spacing
                if json.loads(x) == json.loads(y):
                    continue
            except ValueError:
                pass
        if x != y:
            out.append((k, x, y))
    return out


def shape_delta(a, b, path=""):
    """Paths where Strata's shape has something TF's lacks, or a leaf differs. TF-only keys are additive and ignored."""
    res = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in b:
                res.append(f"{path}/{k}: only in Strata")
            elif k in a:
                res += shape_delta(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if a and b:
            res += shape_delta(a[0], b[0], path + "[]")
        elif a and not b:
            res.append(f"{path}: empty list in TF")
    elif a != b and not (a is None or b is None):
        res.append(f"{path}: {a!r} vs {b!r}")
    return res


def tf_expect(name, v):
    """Absolute expectations on TF alone (the request-side half the mock echo exposes), None when met."""
    def echo():
        text = (v.get("content") or "") or (v.get("reasoning_content") or "")
        try:
            return json.loads(text)
        except ValueError:
            return None
    want = {
        "echo_defaults": {"think_budget": 16384, "max_tokens": 4081},
        "echo_override": {"think_budget": 77, "max_tokens": 99},
        "echo_thinking_budget": {"think_budget": 55},
        "echo_thinking_budget_tokens": {"think_budget": 33},
        "echo_budget_zero": {"think_budget": 0},
        "echo_off": {"think_budget": 0},
        "echo_max_completion_tokens": {"max_tokens": 41},
    }.get(name)
    if want is None:
        return None
    text = (v.get("content") or "") or (v.get("reasoning_content") or "")
    import re
    e = {k: int(m.group(1)) for k in want for m in [re.search(r'"%s":(-?\d+)' % k, text)] if m}    # the echo may be cut by max_tokens
    if not e:
        return f"engine echo unreadable: {str(v)[:120]}"
    bad = {k: (e.get(k), x) for k, x in want.items() if e.get(k) != x}
    return f"engine saw {bad} (got, wanted)" if bad else None


def run_concurrency(base, n, case):
    t0 = time.time()
    with cf.ThreadPoolExecutor(n) as ex:
        rs = list(ex.map(lambda _: call(base, case, timeout=120), range(n)))
    return {"statuses": sorted(str(r["status"]) for r in rs), "wall": round(time.time() - t0, 2),
            "max_elapsed": round(max(r["elapsed"] for r in rs), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strata", required=True)
    ap.add_argument("--tf", required=True)
    ap.add_argument("--only")
    ap.add_argument("--json")
    ap.add_argument("--strict", action="store_true", help="exit 1 on any unexplained difference or failed TF expectation")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    names = [n for n in CASES if not a.only or n in a.only.split(",")]
    report, gaps, samp, expect_fail = {}, [], [], []
    for name in names:
        case = CASES[name]
        rs, rt = call(a.strata, case), call(a.tf, case)
        vs, vt = view(name, case, rs), view(name, case, rt)
        d = diff(name, vs, vt, name in SHAPE_ONLY)
        sd = shape_delta(vs.get("shape"), vt.get("shape")) if "shape" in vs and "shape" in vt else []
        known = KNOWN.get((name, "*"))
        exp = tf_expect(name, vt)
        report[name] = {"strata": vs, "tf": vt, "diff": d, "shape_delta": sd, "known": known, "tf_expect": exp}
        same = not (d or sd) and not exp
        if exp:
            expect_fail.append(name)
        if not same and not known and name in SAMP_CASES:
            samp.append(name)
        elif (d or sd) and not known:
            gaps.append(name)
        if not a.quiet:
            tag = "SAME " if same else ("KNOWN" if known and not exp else ("SAMP " if name in SAMP_CASES else "GAP  "))
            print(f"[{tag}] {name}  strata={vs.get('status')} tf={vt.get('status')}" + (f"  ({known})" if tag == "KNOWN" else ""))
            if tag in ("GAP  ", "SAMP ") or exp:
                for k, x, y in d:
                    print(f"        {k}: strata={json.dumps(x, ensure_ascii=False)[:200]}")
                    print(f"        {' ' * len(k)}  tf    ={json.dumps(y, ensure_ascii=False)[:200]}")
                for s in sd[:10]:
                    print(f"        shape {s}")
                if exp:
                    print(f"        TF expectation: {exp}")
    cc = {"strata": run_concurrency(a.strata, CONCURRENCY["n"], CONCURRENCY["case"]),
          "tf": run_concurrency(a.tf, CONCURRENCY["n"], CONCURRENCY["case"])}
    queued = all(s == "200" for s in cc["tf"]["statuses"]) and cc["tf"]["wall"] >= 1.0
    report["_concurrency8_slow"] = {**cc, "tf_queues_not_rejects": queued}
    print("concurrency x8 (0.6 s of scripted decode each, 4 lanes):", json.dumps(cc), "-> queued, none refused" if queued else "-> FAIL")
    print(f"cases={len(names)} gaps={len(gaps)} {gaps} samp={samp} tf_expect_failed={expect_fail} queue_ok={queued}")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1, ensure_ascii=False, default=str)
    return 1 if (a.strict and (gaps or expect_fail or not queued)) else 0


if __name__ == "__main__":
    sys.exit(main())
