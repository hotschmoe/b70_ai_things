#!/usr/bin/env python3
"""Scripted OpenAI-compatible chat server for testing the longrun harness with NO model.

  mock_openai.py PORT [--ctx-bloat N]
Behaviour per request (by number of assistant turns seen since the last user message):
  - summarization requests (system prompt mentions 'summar'): returns a short text summary
  - first user message / any continuation message: emits ROUNDS tool calls (bash writing file note_<k>.txt,
    returning a large output to bloat context), then a final text 'All done.'
  - env MOCK_STOP_AFTER_FIRST=1: later rounds (continuation) answer 'done' without tools (idle round)
"""
import json, os, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROUNDS = int(os.environ.get("MOCK_TOOL_TURNS", "4"))
BLOAT = int(os.environ.get("MOCK_BLOAT_CHARS", "6000"))
STOP_AFTER_FIRST = os.environ.get("MOCK_STOP_AFTER_FIRST", "0") == "1"
CALLS = json.loads(os.environ.get("MOCK_CALLS", "null"))   # scripted tool calls [{"name":..,"arguments":{..}}]
LOG = open(os.environ.get("MOCK_LOG", "/dev/null"), "a")


def sse(h, obj):
    h.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n"); h.wfile.flush()


ST = {"calls": 0}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_GET(self):
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(json.dumps({"data": [{"id": "hotschmoe-dd", "object": "model"}]}).encode())

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n))
        msgs = req["messages"]
        sysmsg = json.dumps(msgs[0]) if msgs else ""
        LOG.write(json.dumps({"n_msgs": len(msgs), "chars": len(json.dumps(msgs)), "tools": [t["function"]["name"] for t in req.get("tools", [])], "tools_chars": len(json.dumps(req.get("tools", []))), "skill_in_system": "reverse-engineer-anything" in sysmsg}) + "\n"); LOG.flush()
        last_user = max([i for i, m in enumerate(msgs) if m["role"] == "user"] or [0])
        users = [m for m in msgs if m["role"] == "user"]
        turns_since = sum(1 for m in msgs[last_user:] if m["role"] == "assistant")
        is_sum = "summar" in sysmsg.lower() or "summar" in json.dumps(msgs[-1]).lower()[:3000] and len(users) == 1 and msgs[-1]["role"] == "user" and "<conversation>" in json.dumps(msgs[-1])
        first_round = "Continue working until" not in json.dumps(users[-1]) if users else True
        if not is_sum and msgs and msgs[-1]["role"] == "user":
            ST["calls"] = 0       # new round / new user prompt
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        base = {"id": "x", "object": "chat.completion.chunk", "created": int(time.time()), "model": "hotschmoe-dd"}
        if is_sum:
            delta = {"content": "## Summary\nWork in progress; notes written."}
            sse(self, {**base, "choices": [{"index": 0, "delta": {"role": "assistant", **delta}, "finish_reason": None}]})
            sse(self, {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}})
        elif CALLS is not None and ST["calls"] < len(CALLS):
            c = CALLS[ST["calls"]]; ST["calls"] += 1
            tc = {"index": 0, "id": f"call{len(msgs)}", "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
            sse(self, {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "tool_calls": [tc]}, "finish_reason": None}]})
            sse(self, {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 0}})
        elif CALLS is None and (first_round or not STOP_AFTER_FIRST) and ST["calls"] < ROUNDS:
            k = ST["calls"]; ST["calls"] += 1
            cmd = f"echo step{k} >> note_{len(users)}_{k}.txt; head -c {BLOAT} /dev/zero | tr '\\0' 'x'"
            tc = {"index": 0, "id": f"call{len(msgs)}", "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": cmd})}}
            sse(self, {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "tool_calls": [tc]}, "finish_reason": None}]})
            sse(self, {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": len(json.dumps(msgs)) // 4, "completion_tokens": 20, "total_tokens": 0}})
        else:
            sse(self, {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "All done."}, "finish_reason": None}]})
            sse(self, {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": len(json.dumps(msgs)) // 4, "completion_tokens": 3, "total_tokens": 0}})
        self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()


if __name__ == "__main__":
    if os.environ.get("MOCK_DUMP_TOOLS"):
        pass
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
