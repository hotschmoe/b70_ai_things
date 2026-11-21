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
    "tool_choice_auto": chat("weather?", "tool_nothink", tools=TOOLS, tool_choice="auto", chat_template_kwargs=NOTHINK),
    "tool_choice_none": chat("weather?", "plain", tools=TOOLS, tool_choice="none", chat_template_kwargs=NOTHINK),
    "tool_choice_required": chat("weather?", "tool_nothink", tools=TOOLS, tool_choice="required", chat_template_kwargs=NOTHINK),
    "tool_choice_required_stream": chat("weather?", "tool_nothink", tools=TOOLS, tool_choice="required",
                                        chat_template_kwargs=NOTHINK, stream=True),
    "tool_choice_named": chat("weather?", "tool_nothink", tools=TOOLS, chat_template_kwargs=NOTHINK,
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
        "model": "no-such-model", "messages": [{"role": "user", "content": "hi @script=plain"}]}},
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
    "messages_plain": msgs("hi", "plain"),
    "messages_stream": msgs("hi", "think", stream=True, thinking={"type": "enabled", "budget_tokens": 256}),
    "messages_tool": msgs("weather?", "tool_nothink", tools=[{"name": "get_weather", "description": "w", "input_schema": {
        "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}]),
    "completions_raw": {"method": "POST", "path": "/v1/completions", "body": {"model": "hotschmoe-dd", "prompt": "Say hi @script=plain", "max_tokens": 16}},
    "echo_defaults": chat("hi", "echo"),
    "echo_override": chat("hi", "echo", reasoning_budget_tokens=77, max_tokens=99),
}
# cases whose assembled text is the mock's echo or otherwise engine-specific: compare status and shape, not values
SHAPE_ONLY = {"echo_defaults", "echo_override", "models", "models_one", "health", "ctx_overflow_prompt", "ctx_overflow_stream"}
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


# Differences that are documented and accepted. (case, field) -> reason. field "*" covers the whole case.
KNOWN = {
    ("models", "*"): "listing shape differs by design; both put hotschmoe-dd first and carry the alias (see ids line)",
    ("models_one", "*"): "listing shape differs by design",
    ("health", "*"): "health bodies differ by design; both 200 with status ok",
}


def diff(name, a, b, shape_only):
    out = []
    keys = sorted(set(a) | set(b))
    for k in keys:
        if shape_only and k in ("content", "reasoning_content", "tool_calls", "text", "thinking", "anthropic", "error_message_head"):
            continue
        if a.get(k) != b.get(k):
            out.append((k, a.get(k), b.get(k)))
    return out


def shape_delta(a, b, path=""):
    """Paths where two shapes differ: only-in-a, only-in-b, or different leaf."""
    res = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                res.append(f"{path}/{k}: only in TF")
            elif k not in b:
                res.append(f"{path}/{k}: only in Strata")
            else:
                res += shape_delta(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if a and b:
            res += shape_delta(a[0], b[0], path + "[]")
        elif bool(a) != bool(b):
            res.append(f"{path}: empty list on one side")
    elif a != b:
        res.append(f"{path}: {a!r} vs {b!r}")
    return res


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
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    names = [n for n in CASES if not a.only or n in a.only.split(",")]
    report, gaps = {}, 0
    for name in names:
        case = CASES[name]
        rs, rt = call(a.strata, case), call(a.tf, case)
        vs, vt = view(name, case, rs), view(name, case, rt)
        d = diff(name, vs, vt, name in SHAPE_ONLY)
        sd = shape_delta(vs.get("shape"), vt.get("shape")) if "shape" in vs and "shape" in vt else []
        known = KNOWN.get((name, "*"))
        report[name] = {"strata": vs, "tf": vt, "diff": [(k, x, y) for k, x, y in d if k != "shape"], "shape_delta": sd,
                        "known": known}
        if (d or sd) and not known:
            gaps += 1
        if not a.quiet:
            tag = "SAME " if not (d or sd) else ("KNOWN" if known else "DIFF ")
            print(f"[{tag}] {name}  strata={vs.get('status')} tf={vt.get('status')}")
            if (d or sd) and not known:
                for k, x, y in d:
                    if k == "shape":
                        continue
                    print(f"        {k}: strata={json.dumps(x, ensure_ascii=False)[:230]}")
                    print(f"        {' ' * len(k)}  tf    ={json.dumps(y, ensure_ascii=False)[:230]}")
                for s in sd[:12]:
                    print(f"        shape {s}")
    cc = {"strata": run_concurrency(a.strata, CONCURRENCY["n"], CONCURRENCY["case"]),
          "tf": run_concurrency(a.tf, CONCURRENCY["n"], CONCURRENCY["case"])}
    report["_concurrency8_slow"] = cc
    print("concurrency x8 (0.6 s of scripted decode each):", json.dumps(cc))
    print(f"cases={len(names)} gaps={gaps}")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1, ensure_ascii=False, default=str)
    return 1 if (a.strict and gaps) else 0


if __name__ == "__main__":
    sys.exit(main())
