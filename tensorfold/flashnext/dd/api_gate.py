#!/usr/bin/env python3
"""API gates for gpu_window.sh (worker API): user-facing parity checks against the window server (hotschmoe-dd).

  reasoning URL        reasoning_content (non-stream and stream, usage chunk, [DONE]), thinking on by default
  tools URL            streamed tool-call deltas (name first, incremental argument deltas, valid JSON), non-stream call, tool_choice none
  forced URL           tool_choice required / named (prompt-side), response_format json_schema (2 of 3 replies valid)
  budget URL           per-request thinking budget (reasoning_budget_tokens, thinking_budget_tokens) and effort none
  edge URL             /v1/models identity, /health, context overflow 400, max_tokens clamp, finish reasons, usage keys, queue past --parallel
  all URL              every gate above, one GATE line each

Each prints one GATE line and exits 0 on PASS, 1 on FAIL. Standard library only. No prompt or response text is printed beyond
the fixed test strings' short heads. URL is the server root (no /v1), like vis_gate.py.
"""
import concurrent.futures as cf
import json
import sys
import time
import urllib.error
import urllib.request

MODEL = "hotschmoe-dd"
TOOLS = [{"type": "function", "function": {"name": "write_file", "description": "Write a file",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                         "required": ["path", "content"]}}},
         {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
OFF = {"enable_thinking": False}
Q = "A train leaves at 9:40 and arrives at 13:05. How many minutes is the trip? Answer with the number only at the end."


def req(url, path, body=None, stream=False, timeout=170):
    r = urllib.request.Request(url.rstrip("/") + path, json.dumps(body).encode() if body is not None else None,
                               {"Content-Type": "application/json"})
    try:
        resp = urllib.request.urlopen(r, timeout=timeout)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")
    with resp:
        if not stream:
            return resp.status, json.load(resp)
        events, last, gaps, t0 = [], None, [], time.time()
        for line in resp:
            now = time.time()
            if last is not None:
                gaps.append(now - last)
            last = now
            if line.startswith(b"data: "):
                p = line[6:].strip()
                events.append("[DONE]" if p == b"[DONE]" else json.loads(p))
        return resp.status, {"events": events, "max_gap_s": max(gaps) if gaps else 0.0, "wall_s": time.time() - t0}


def chat(url, **kw):
    body = {"model": MODEL, "temperature": 0}
    body.update(kw)
    return req(url, "/v1/chat/completions", body, stream=bool(kw.get("stream")))


def sview(events):
    kinds, reasoning, content, calls, finish, usage = [], [], [], {}, None, None
    for ev in events:
        if ev == "[DONE]":
            kinds.append("done")
            continue
        usage = ev.get("usage") or usage
        for ch in ev.get("choices", []):
            d = ch.get("delta", {})
            if d.get("reasoning_content"):
                reasoning.append(d["reasoning_content"])
                kinds.append("r")
            if d.get("content"):
                content.append(d["content"])
                kinds.append("c")
            for tc in d.get("tool_calls") or []:
                c = calls.setdefault(tc.get("index", 0), {"name": "", "args": "", "n": 0, "id": ""})
                f = tc.get("function", {})
                c["name"] += f.get("name") or ""
                c["id"] = tc.get("id") or c["id"]
                if f.get("arguments"):
                    c["args"] += f["arguments"]
                    c["n"] += 1
                kinds.append("t")
            finish = ch.get("finish_reason") or finish
    return {"kinds": kinds, "reasoning": "".join(reasoning), "content": "".join(content), "calls": calls, "finish": finish,
            "usage": usage, "done": bool(kinds) and kinds[-1] == "done"}


def out(name, ok, detail):
    print(f"GATE api-{name} {'PASS' if ok else 'FAIL'} {json.dumps(detail)}", flush=True)
    return ok


def reasoning(url):
    d = {}
    code, r = chat(url, max_tokens=2048, messages=[{"role": "user", "content": Q}])
    m = r["choices"][0]["message"] if code == 200 else {}
    d["nonstream"] = {"code": code, "reasoning_chars": len(m.get("reasoning_content") or ""), "ok205": "205" in (m.get("content") or ""),
                      "cached_key": "cached_tokens" in ((r.get("usage") or {}).get("prompt_tokens_details") or {}),
                      "finish": (r.get("choices") or [{}])[0].get("finish_reason")}
    code, r = chat(url, max_tokens=2048, stream=True, stream_options={"include_usage": True}, messages=[{"role": "user", "content": Q}])
    v = sview(r["events"]) if code == 200 else {"kinds": [], "reasoning": "", "content": "", "done": False, "usage": None, "finish": None}
    k = v["kinds"]
    first_r, first_c = (k.index("r") if "r" in k else -1), (k.index("c") if "c" in k else -1)
    d["stream"] = {"code": code, "reasoning_chars": len(v["reasoning"]), "r_before_c": 0 <= first_r < first_c, "ok205": "205" in v["content"],
                   "done": v["done"], "usage": bool(v["usage"]), "finish": v["finish"]}
    ok = (d["nonstream"]["code"] == 200 and d["nonstream"]["reasoning_chars"] > 0 and d["nonstream"]["ok205"] and d["nonstream"]["cached_key"]
          and d["nonstream"]["finish"] == "stop" and d["stream"]["r_before_c"] and d["stream"]["ok205"] and d["stream"]["done"]
          and d["stream"]["usage"] and d["stream"]["finish"] == "stop")
    return out("reasoning", ok, d)


def tools(url):
    d = {}
    body = "\n".join(f"line {i:02d}: the quick brown fox jumps over the lazy dog" for i in range(30))
    code, r = chat(url, max_tokens=3000, stream=True, tools=TOOLS, tool_choice="auto", chat_template_kwargs=OFF,
                   messages=[{"role": "user", "content": "Call write_file with path notes.txt and content exactly these 30 lines:\n" + body}])
    v = sview(r["events"]) if code == 200 else {"calls": {}, "finish": None, "done": False}
    c = v["calls"].get(0, {})
    try:
        args = json.loads(c.get("args", ""))
    except ValueError:
        args = None
    lines = sum(1 for i in range(30) if isinstance(args, dict) and f"line {i:02d}" in (args.get("content") or ""))
    d["stream"] = {"code": code, "name": c.get("name"), "id": c.get("id", "")[:5], "arg_deltas": c.get("n"), "lines_ok": lines,
                   "finish": v["finish"], "done": v["done"], "max_gap_s": round(r.get("max_gap_s", 0), 2) if code == 200 else None}
    ok_stream = (code == 200 and c.get("name") == "write_file" and isinstance(args, dict) and args.get("path") == "notes.txt"
                 and c.get("n", 0) >= 3 and lines >= 28 and v["finish"] == "tool_calls" and v["done"] and r["max_gap_s"] < 20)
    code, r = chat(url, max_tokens=1024, tools=TOOLS, tool_choice="auto", chat_template_kwargs=OFF,
                   messages=[{"role": "user", "content": "What is the weather in Reno right now? Use the tool."}])
    ch = (r.get("choices") or [{}])[0] if code == 200 else {}
    tc = (ch.get("message", {}).get("tool_calls") or [{}])[0]
    d["nonstream"] = {"code": code, "name": tc.get("function", {}).get("name"), "finish": ch.get("finish_reason"), "id": (tc.get("id") or "")[:5]}
    ok_non = code == 200 and d["nonstream"]["name"] == "get_weather" and ch.get("finish_reason") == "tool_calls" and (tc.get("id") or "").startswith("call_")
    code, r = chat(url, max_tokens=200, tools=TOOLS, tool_choice="none", chat_template_kwargs=OFF,
                   messages=[{"role": "user", "content": "What is the weather in Reno right now? Use the tool if you can, else say you cannot."}])
    ch = (r.get("choices") or [{}])[0] if code == 200 else {}
    d["tool_choice_none"] = {"code": code, "calls": len(ch.get("message", {}).get("tool_calls") or []), "has_text": bool(ch.get("message", {}).get("content"))}
    ok_none = code == 200 and d["tool_choice_none"]["calls"] == 0 and d["tool_choice_none"]["has_text"]
    return out("tools", ok_stream and ok_non and ok_none, d)


def forced(url):
    d = {}
    code, r = chat(url, max_tokens=512, tools=TOOLS, tool_choice="required", messages=[{"role": "user", "content": "Hello there, how are you today?"}])
    ch = (r.get("choices") or [{}])[0] if code == 200 else {}
    tc = (ch.get("message", {}).get("tool_calls") or [{}])[0].get("function", {})
    d["required"] = {"code": code, "name": tc.get("name"), "finish": ch.get("finish_reason")}
    ok_req = code == 200 and tc.get("name") in ("write_file", "get_weather") and ch.get("finish_reason") == "tool_calls"
    code, r = chat(url, max_tokens=512, tools=TOOLS, tool_choice={"type": "function", "function": {"name": "get_weather"}},
                   messages=[{"role": "user", "content": "Tell me a joke."}])
    ch = (r.get("choices") or [{}])[0] if code == 200 else {}
    tc = (ch.get("message", {}).get("tool_calls") or [{}])[0].get("function", {})
    try:
        city = json.loads(tc.get("arguments", "")).get("city")
    except (ValueError, AttributeError):
        city = None
    d["named"] = {"code": code, "name": tc.get("name"), "city_str": isinstance(city, str)}
    ok_named = code == 200 and tc.get("name") == "get_weather" and isinstance(city, str)
    code, r = chat(url, max_tokens=512, tools=TOOLS, tool_choice="required", stream=True, messages=[{"role": "user", "content": "Say hello."}])
    v = sview(r["events"]) if code == 200 else {"calls": {}, "finish": None}
    d["required_stream"] = {"code": code, "calls": len(v["calls"]), "finish": v["finish"]}
    ok_rs = code == 200 and len(v["calls"]) >= 1 and v["finish"] == "tool_calls"
    good = 0
    for city_name in ("Boise", "Reno", "Spokane"):
        code, r = chat(url, max_tokens=300, chat_template_kwargs=OFF, messages=[{"role": "user", "content": f"{city_name}: give the city and a rough population."}],
                       response_format={"type": "json_schema", "json_schema": {"name": "city", "strict": True, "schema": {
                           "type": "object", "properties": {"city": {"type": "string"}, "population": {"type": "integer"}},
                           "required": ["city", "population"], "additionalProperties": False}}})
        try:
            obj = json.loads(r["choices"][0]["message"]["content"]) if code == 200 else None
            good += isinstance(obj, dict) and isinstance(obj.get("city"), str) and isinstance(obj.get("population"), int)
        except (ValueError, KeyError, TypeError):
            pass
    d["response_format_ok_of_3"] = good
    return out("forced", ok_req and ok_named and ok_rs and good >= 2, d)


def budget(url):
    d = {}
    ok = True
    for field in ("reasoning_budget_tokens", "thinking_budget_tokens", "thinking_budget"):
        code, r = chat(url, max_tokens=1024, messages=[{"role": "user", "content": Q}], **{field: 48})
        m = r["choices"][0]["message"] if code == 200 else {}
        n = len(m.get("reasoning_content") or "")
        d[field] = {"code": code, "reasoning_chars": n, "answer": bool(m.get("content"))}
        ok = ok and code == 200 and n < 1200 and bool(m.get("content"))
    code, r = chat(url, max_tokens=256, reasoning_effort="none", messages=[{"role": "user", "content": "Reply with the word ok."}])
    m = r["choices"][0]["message"] if code == 200 else {}
    d["effort_none"] = {"code": code, "reasoning": bool(m.get("reasoning_content"))}
    ok = ok and code == 200 and not m.get("reasoning_content")
    return out("budget", ok, d)


def edge(url):
    d = {}
    code, m = req(url, "/v1/models")
    data = m.get("data") or []
    ids = [x.get("id") for x in data]
    window = (data[0].get("context_length") or data[0].get("max_model_len") or 0) if data else 0
    d["models"] = {"code": code, "first": ids[:1], "aliases": [i for i in ids[1:]][:3], "window": window}
    ok = code == 200 and ids[:1] == [MODEL] and any("strata-0216" in i for i in ids) and window >= 100000
    code, _ = req(url, "/health")
    d["health"] = code
    ok = ok and code == 200
    if window:
        code, r = chat(url, max_tokens=16, chat_template_kwargs=OFF, messages=[{"role": "user", "content": "word " * (window + 2000)}])
        msg = (r.get("error") or {}).get("message", "")
        d["overflow"] = {"code": code, "names_window": "context window" in msg}
        ok = ok and code == 400 and "context window" in msg
        code, r = chat(url, max_tokens=window, chat_template_kwargs=OFF, messages=[{"role": "user", "content": "Reply with the word ok."}])
        fr = (r.get("choices") or [{}])[0].get("finish_reason") if code == 200 else None
        d["clamp_max_tokens"] = {"code": code, "finish": fr}
        ok = ok and code == 200 and fr == "stop"
    code, r = chat(url, max_tokens=8, chat_template_kwargs=OFF, messages=[{"role": "user", "content": "Count from 1 to 100 separated by commas."}])
    fl = (r.get("choices") or [{}])[0].get("finish_reason") if code == 200 else None
    d["finish_length"] = {"code": code, "finish": fl, "completion_tokens": (r.get("usage") or {}).get("completion_tokens")}
    ok = ok and fl == "length" and (r.get("usage") or {}).get("completion_tokens", 99) <= 9
    code, r = chat(url, max_tokens=200, stop=["5"], chat_template_kwargs=OFF, messages=[{"role": "user", "content": "Count from 1 to 20 separated by commas."}])
    ch = (r.get("choices") or [{}])[0] if code == 200 else {}
    d["stop_sequence"] = {"code": code, "finish": ch.get("finish_reason"), "leaked": "5" in (ch.get("message", {}).get("content") or "")}
    ok = ok and ch.get("finish_reason") == "stop" and not d["stop_sequence"]["leaked"]
    code, r = chat(url, max_tokens=8, chat_template_kwargs=OFF, messages=[{"role": "user", "content": "Say hi."}])
    u = r.get("usage") or {}
    d["usage"] = {"keys": sorted(u), "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens")}
    ok = ok and {"prompt_tokens", "completion_tokens", "total_tokens"} <= set(u) and "cached_tokens" in (u.get("prompt_tokens_details") or {})

    def one(i):
        t0 = time.time()
        c, rr = chat(url, max_tokens=24, chat_template_kwargs=OFF, messages=[{"role": "user", "content": f"Return only the integer: {i + 11} * 7."}])
        text = ((rr.get("choices") or [{}])[0].get("message", {}).get("content") or "").strip() if c == 200 else None
        return c, time.time() - t0, text
    with cf.ThreadPoolExecutor(8) as ex:
        res = list(ex.map(one, range(8)))
    d["queue8"] = {"codes": sorted(c for c, _, _ in res), "max_s": round(max(t for _, t, _ in res), 1)}
    ok = ok and all(c == 200 for c, _, _ in res) and all(t == str((i + 11) * 7) for i, (_, _, t) in enumerate(res))
    return out("edge", ok, d)


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ("reasoning", "tools", "forced", "budget", "edge", "all"):
        raise SystemExit(__doc__)
    which, url = sys.argv[1], sys.argv[2]
    fns = {"reasoning": reasoning, "tools": tools, "forced": forced, "budget": budget, "edge": edge}
    results = [fns[n](url) for n in (fns if which == "all" else [which])]
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
