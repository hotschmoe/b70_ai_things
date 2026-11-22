#!/usr/bin/env python3
"""Tiny OpenAI-compatible mock for release_traffic.py --dry-run (worker REL, 2026-11-22). CPU only, stdlib only.

/v1/models, /health, /metrics (Prometheus text, tensorfold: prefix, open) and /v1/chat/completions with Bearer auth,
SSE streaming (reasoning_content deltas, content deltas, incremental tool-call deltas, a usage chunk, [DONE]) and
non-streaming replies. prompt_tokens = request chars / 3.3 + 300 per image; cached_tokens = the longest message
prefix already seen (like a prompt cache). Tool calls are valid JSON for the offered tools. A needle codeword in the
prompt is answered. Delays: prefill 200K tok/s on uncached tokens, decode 800 tok/s.
--faults P (or serve_background(faults=P)) injects per request: HTTP 500, a truncated stream (no [DONE]), malformed
tool arguments, an empty answer, or an odd finish reason, to prove the harness gates fail.

Standalone: python3 release_traffic_mock.py --port 18999 --key-file KEYFILE [--faults 0.1]
"""
import argparse, hashlib, json, random, re, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATE = {"key": "", "faults": 0.0, "rng": random.Random(1), "seen": {}, "lock": threading.Lock(),
         "status": {}, "prompt": 0, "cached": 0, "gen": 0, "running": 0, "malformed": 0}
WORDS = ("the server keeps the prompt state in host memory and restores it on the next turn so the follow up "
         "prefill only covers new tokens while decode continues at the usual rate").split()


def words(rng, n):
    return " ".join(rng.choice(WORDS) for _ in range(n))


def tool_args(rng, name, prompt):
    paths = re.findall(r"read with read_file: ([^\n]+)\.", prompt)
    path = rng.choice(paths[-1].split(", ")) if paths else "README.md"
    return {"read_file": {"path": path}, "search_code": {"pattern": "XPU_FN_[A-Z_]+"}, "list_dir": {"path": "tensorfold"},
            "run_command": {"command": "pytest -q"}, "write_file": {"path": "notes.txt", "content": "ok"}}.get(name, {})


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *a):
        pass

    def send(self, code, obj, ctype="application/json"):
        body = obj.encode() if isinstance(obj, str) else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        with STATE["lock"]:
            STATE["status"][code] = STATE["status"].get(code, 0) + 1

    def authed(self):
        if self.headers.get("Authorization") != "Bearer " + STATE["key"]:
            self.send(401, {"error": {"message": "bad key"}})
            return False
        return True

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            return self.send(200, {"status": "ok"})
        if self.path == "/metrics":
            s = STATE
            lines = [f'tensorfold:requests_total{{key="-",status="{k}"}} {v}' for k, v in sorted(s["status"].items())]
            lines += [f"tensorfold:requests_running {s['running']}", "tensorfold:requests_waiting 0",
                      f"tensorfold:prompt_tokens_total {s['prompt']}", f"tensorfold:prompt_tokens_cached_total {s['cached']}",
                      f"tensorfold:generation_tokens_total {s['gen']}", f"tensorfold:kept_tool_calls_malformed_total {s['malformed']}",
                      "tensorfold:client_disconnections_total 0", f"tensorfold:process_resident_bytes {40 << 30}",
                      f"tensorfold:prompt_cache_bytes {min(2 << 30, s['cached'] * 1000)}"]
            return self.send(200, "\n".join(lines) + "\n", "text/plain")
        if not self.authed():
            return
        if self.path == "/v1/models":
            return self.send(200, {"object": "list", "data": [{"id": "hotschmoe-dd", "context_length": 200000}]})
        self.send(404, {"error": {"message": "not found"}})

    def do_POST(self):  # noqa: N802
        if not self.authed():
            return
        if self.path != "/v1/chat/completions":
            return self.send(404, {"error": {"message": "not found"}})
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        msgs, tools = body["messages"], body.get("tools")
        rng = random.Random(STATE["rng"].random())
        fault = rng.choice(["http500", "truncate", "badjson", "empty", "finish"]) if rng.random() < STATE["faults"] else None
        if fault == "http500":
            return self.send(500, {"error": {"message": "mock fault"}})
        # prompt accounting and prefix cache
        h, best, chars, images, text = hashlib.sha256(), 0, 0, 0, []
        with STATE["lock"]:
            for m in msgs:
                h.update(json.dumps(m, sort_keys=True).encode())
                c = m.get("content")
                parts = c if isinstance(c, list) else [{"text": c or ""}]
                for p in parts:
                    chars += len(p.get("text") or "")
                    images += p.get("type") == "image_url"
                    text.append(p.get("text") or "")
                chars += len(json.dumps(m.get("tool_calls") or "")) + len(m.get("reasoning_content") or "")
                tok = int(chars / 3.3) + 300 * images
                d = h.hexdigest()
                if d in STATE["seen"]:
                    best = tok
                STATE["seen"][d] = tok
            prompt_tokens = int((chars + len(json.dumps(tools or ""))) / 3.3) + 300 * images + 16 * len(msgs)
            cached = max(0, best - 64) if best else 0
            STATE["running"] += 1
        prompt = "\n".join(text)
        kw = body.get("chat_template_kwargs") or {}
        thinking = body.get("reasoning_effort") != "none" and kw.get("enable_thinking", True)
        last = msgs[-1]["role"]
        ntool = sum(m["role"] == "tool" for m in msgs)
        call = tools and body.get("tool_choice") != "none" and last in ("user", "tool") and ntool < 8 and rng.random() < 0.75
        reasoning = words(rng, rng.randint(20, 120)) if thinking else ""
        m = re.search(r"codeword for this bundle is (\S+?)\.", prompt)
        if images and not ("second image" in prompt or "background" in prompt[-200:]):
            content = "The circle is blue and it is on the left."
        elif "background" in prompt[-200:]:
            content = "White."
        elif m and "codeword" in prompt[-200:]:
            content = m.group(1)
        else:
            content = words(rng, rng.randint(20, 250))
        if fault == "empty":
            content = ""
        calls = []
        if call:
            name = rng.choice([t["function"]["name"] for t in tools])
            args = json.dumps(tool_args(rng, name, prompt))
            if fault == "badjson":
                args = args[:-1] + ',"x":'
            calls = [{"id": "call_" + hashlib.sha256(args.encode() + str(rng.random()).encode()).hexdigest()[:12], "name": name, "args": args}]
            content = ""
        ntok = len(reasoning.split()) + len(content.split()) + sum(len(c["args"]) // 3 for c in calls)
        cap = body.get("max_tokens") or 32768
        finish = "tool_calls" if calls else "stop"
        if ntok > cap:
            ntok, finish, content = cap, "length", ""
        if fault == "finish":
            finish = "abort"
        time.sleep((prompt_tokens - cached) / 200000)
        usage = {"prompt_tokens": prompt_tokens, "completion_tokens": ntok, "total_tokens": prompt_tokens + ntok,
                 "prompt_tokens_details": {"cached_tokens": cached}}
        with STATE["lock"]:
            STATE["prompt"] += prompt_tokens
            STATE["cached"] += cached
            STATE["gen"] += ntok
        try:
            if body.get("stream"):
                self.stream(body, reasoning, content, calls, finish, usage, fault, ntok)
            else:
                msg = {"role": "assistant", "content": content or None}
                if reasoning:
                    msg["reasoning_content"] = reasoning
                if calls:
                    msg["tool_calls"] = [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["args"]}} for c in calls]
                time.sleep(ntok / 800)
                self.send(200, {"id": "chatcmpl-mock", "object": "chat.completion", "model": "hotschmoe-dd",
                                "choices": [{"index": 0, "message": msg, "finish_reason": finish}], "usage": usage})
        finally:
            with STATE["lock"]:
                STATE["running"] -= 1

    def stream(self, body, reasoning, content, calls, finish, usage, fault, ntok):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        with STATE["lock"]:
            STATE["status"][200] = STATE["status"].get(200, 0) + 1

        def ev(delta=None, fin=None, use=None):
            obj = {"id": "chatcmpl-mock", "object": "chat.completion.chunk", "model": "hotschmoe-dd",
                   "choices": [] if use else [{"index": 0, "delta": delta or {}, "finish_reason": fin}]}
            if use:
                obj["usage"] = use
            self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
            self.wfile.flush()
        ev({"role": "assistant"})
        per = max(1, ntok) and 1 / 800
        for key, text in (("reasoning_content", reasoning), ("content", content)):
            ws = text.split(" ") if text else []
            for i in range(0, len(ws), 4):
                time.sleep(4 * per)
                ev({key: " ".join(ws[i:i + 4]) + " "})
                if fault == "truncate" and i > 8:
                    return
        for i, c in enumerate(calls):
            ev({"tool_calls": [{"index": i, "id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": ""}}]})
            a = c["args"]
            for j in range(0, len(a), max(1, len(a) // 3)):
                time.sleep(per)
                ev({"tool_calls": [{"index": i, "function": {"arguments": a[j:j + max(1, len(a) // 3)]}}]})
        if fault == "truncate":
            return
        ev({}, finish)
        if (body.get("stream_options") or {}).get("include_usage"):
            ev(use=usage)
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def serve_background(key, faults=0.0, seed=1, port=0):
    STATE.update(key=key, faults=faults, rng=random.Random(seed))
    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv.server_address[1]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=18999)
    p.add_argument("--key-file", required=True)
    p.add_argument("--faults", type=float, default=0.0)
    a = p.parse_args()
    print("mock on port", serve_background(open(a.key_file).read().strip(), a.faults, port=a.port), flush=True)
    threading.Event().wait()
