#!/usr/bin/env python3
"""Capture native llama.cpp SSE diagnostics; never declares quality or speed wins.

The caller owns the GPU lease, server lifecycle, health and fault monitoring.
No generated code is executed. Receipt time is client SSE-event time, not a
GPU token-completion timestamp. Instrumented logprobs can perturb throughput.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=True) + "\n", encoding="ascii")


def snapshot(path, output, label):
    raw = path.read_bytes()
    save(output / (label + ".snapshot.json"), {"path": str(path.resolve()),
         "sha256": sha(raw), "raw_base64": base64.b64encode(raw).decode("ascii")})
    return sha(raw)


def request_json(base, route, payload, out, label, timeout):
    body = None if payload is None else json.dumps(payload, ensure_ascii=True).encode("ascii")
    save(out / (label + ".request.json"), {"url": base + route, "payload": payload})
    req = urllib.request.Request(base + route, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read(4 * 1024 * 1024 + 1)
    save(out / (label + ".response.json"), {"sha256": sha(raw),
         "raw_base64": base64.b64encode(raw).decode("ascii")})
    if len(raw) > 4 * 1024 * 1024:
        raise ValueError("JSON response exceeded capture bound")
    return json.loads(raw)


def stream(base, payload, out, label, timeout):
    save(out / (label + ".request.json"), {"url": base + "/completion", "payload": payload})
    body = json.dumps(payload, ensure_ascii=True).encode("ascii")
    req = urllib.request.Request(base + "/completion", data=body,
                                 headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
    start = time.monotonic()
    events = []
    data_lines = []
    total = 0
    final = None
    with (out / (label + ".wire.jsonl")).open("w", encoding="ascii") as wire, \
         (out / (label + ".events.jsonl")).open("w", encoding="ascii") as decoded, \
         urllib.request.urlopen(req, timeout=timeout) as response:
        if "text/event-stream" not in response.headers.get("Content-Type", ""):
            raise ValueError("Expected native SSE response")
        save(out / (label + ".headers.json"), dict(response.headers))
        while True:
            line = response.readline(1024 * 1024 + 1)
            elapsed = time.monotonic() - start
            if not line:
                if data_lines:
                    raise ValueError("Truncated SSE frame at EOF")
                break
            total += len(line)
            wire.write(json.dumps({"elapsed_seconds": elapsed, "raw_base64": base64.b64encode(line).decode("ascii")}) + "\n")
            wire.flush()
            if len(line) > 1024 * 1024 or total > 32 * 1024 * 1024 or elapsed > timeout:
                raise ValueError("Stream exceeded line, total byte or wall deadline")
            stripped = line.rstrip(b"\r\n")
            if stripped.startswith(b"data:"):
                data_lines.append(stripped[5:].lstrip(b" "))
            elif not stripped and data_lines:
                raw = b"\n".join(data_lines)
                data_lines = []
                if raw == b"[DONE]":
                    break
                event = json.loads(raw)
                record = {"client_event_elapsed_seconds": elapsed, "payload": event}
                decoded.write(json.dumps(record, ensure_ascii=True) + "\n")
                decoded.flush()
                events.append(record)
                if "error" in event:
                    raise ValueError("Server returned SSE error")
                if event.get("stop") is True:
                    final = event
                    break
    if final is None:
        raise ValueError("Missing terminal stop:true event")
    partials = [e for e in events if e["payload"].get("stop") is False and not e["payload"].get("prompt_progress")]
    tokens = [t for e in partials for t in e["payload"].get("tokens", [])]
    text = "".join(e["payload"].get("content", "") for e in partials)
    result = {"case": label, "capture_complete": True, "promoted": False,
              "client_wall_seconds": time.monotonic() - start,
              "client_sse_events": len(events), "partial_token_ids": tokens,
              "partial_token_ids_sha256": sha(json.dumps(tokens).encode("ascii")),
              "partial_text_sha256": sha(text.encode("utf-8")),
              "partial_token_event_sizes": [len(e["payload"].get("tokens", [])) for e in partials],
              "events_with_probabilities": sum(bool(e["payload"].get("completion_probabilities")) for e in partials),
              "final": final,
              "limitations": ["Receipt timing is SSE event timing, not GPU token timing",
                               "Probabilities are reported server values, not raw logits",
                               "Do not concatenate final cumulative token list onto partial tokens",
                               "No CPU/PCIe/cache attribution without separately aligned profiler evidence",
                               "No quality or speed verdict"]}
    save(out / (label + ".summary.json"), result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--endpoint", required=True)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--runtime-receipt", required=True, type=Path)
    p.add_argument("--screen-receipt", required=True, type=Path)
    p.add_argument("--model-lock", required=True, type=Path)
    p.add_argument("--suite", type=Path, default=Path(__file__).with_name("profile_suite.json"))
    p.add_argument("--timeout", type=float, default=600, help="Per-request idle timeout and checked stream wall deadline")
    args = p.parse_args()
    base = args.endpoint.rstrip("/")
    url = urllib.parse.urlparse(base)
    if url.scheme not in ("http", "https") or not url.netloc or url.path not in ("", "/v1") or url.query or url.fragment:
        p.error("Expected HTTP(S) server base URL, optionally ending /v1")
    if base.endswith("/v1"):
        base = base[:-3]
    if not 0 < args.timeout <= 1200:
        p.error("timeout must be >0 and <=1200")
    suite = json.loads(args.suite.read_text())
    screen = json.loads(args.screen_receipt.read_text())
    if len(screen.get("checks", [])) < 6:
        p.error("Six-case screen evidence required before profiling")
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output
    result = {"scope": "diagnostic_only", "quality_qualified": False, "promoted": False,
              "capture_complete": False, "screen_passed": screen.get("passed"), "cases": []}
    try:
        result["provenance"] = {label: snapshot(path, out, label) for label, path in
                                [("runtime-receipt", args.runtime_receipt), ("screen-receipt", args.screen_receipt),
                                 ("model-lock", args.model_lock), ("suite", args.suite), ("client", Path(__file__))]}
        identity = request_json(base, "/v1/models", None, out, "models-before", args.timeout)
        ids = [x["id"] for x in identity.get("data", [])]
        if not ids or ids[0] != "hotschmoe-dd":
            raise ValueError("Expected first model ID hotschmoe-dd")
        for case in suite["cases"]:
            template = request_json(base, "/apply-template", {"messages": [{"role": "user", "content": case["prompt"]}],
                                     "chat_template_kwargs": {"enable_thinking": False}}, out, case["id"] + ".template", args.timeout)
            payload = {"prompt": template["prompt"], "stream": True, "temperature": 0, "seed": 42,
                       "n_predict": suite["max_tokens"], "ignore_eos": False, "cache_prompt": False,
                       "return_tokens": True, "n_probs": 5, "post_sampling_probs": False,
                       "timings_per_token": True, "return_progress": True, "reasoning_budget_tokens": 0}
            row = stream(base, payload, out, case["id"], args.timeout)
            result["cases"].append({"id": case["id"], "capture_complete": row["capture_complete"]})
            save(out / "profile-result.json", result)
        after = request_json(base, "/v1/models", None, out, "models-after", args.timeout)
        if [x["id"] for x in after.get("data", [])] != ids:
            raise ValueError("Model ID list changed during capture")
        result["capture_complete"] = True
    except Exception as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        save(out / "profile-result.json", result)
    print(json.dumps({"capture_complete": result["capture_complete"], "promoted": False,
                      "result": str(out / "profile-result.json")}, ensure_ascii=True))
    return 0 if result["capture_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
