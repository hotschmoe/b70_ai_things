#!/usr/bin/env python3
"""Tiny fake OpenAI-compatible endpoint for harness plumbing tests (no model): every chat completion
answers with one short text message and no tool call. Usage: mock_endpoint.py PORT"""
import http.server, json, socketserver, sys, time

class H(http.server.BaseHTTPRequestHandler):
    def _j(self, obj, code=200):
        b = json.dumps(obj).encode(); self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        self._j({"object": "list", "data": [{"id": "hotschmoe-dd", "object": "model"}]})
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); body = json.loads(self.rfile.read(n) or b"{}")
        mk = lambda delta, fin=None, usage=None: "data: " + json.dumps({"id": "mock", "object": "chat.completion.chunk", "created": int(time.time()),
             "model": "hotschmoe-dd", "choices": [{"index": 0, "delta": delta, "finish_reason": fin}], **({"usage": usage} if usage else {})}) + "\n\n"
        if body.get("stream"):
            self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
            for part in (mk({"role": "assistant", "content": ""}), mk({"content": "I looked at it but this is a mock endpoint."}),
                         mk({}, "stop", {"prompt_tokens": 10, "completion_tokens": 9, "total_tokens": 19}), "data: [DONE]\n\n"):
                self.wfile.write(part.encode()); self.wfile.flush()
        else:
            self._j({"id": "mock", "object": "chat.completion", "choices": [{"index": 0, "message": {"role": "assistant", "content": "mock"}, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}})
    def log_message(self, *a): pass

if __name__ == "__main__":
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    socketserver.ThreadingTCPServer(("0.0.0.0", int(sys.argv[1])), H).serve_forever()
