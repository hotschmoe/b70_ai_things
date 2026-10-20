#!/usr/bin/env python3
"""Drive T=1..4 decode steps against a D2 profile server, then stop.

The server is already up. This does not take a GPU lease. It opens one
long stream, then adds streams so a later step schedules 2, 3, and 4
tokens. It waits until the profile log says d2_profile_done.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.request


def post_stream(url: str, payload: dict, stop: threading.Event) -> None:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            while not stop.is_set():
                chunk = resp.readline()
                if not chunk:
                    break
    except Exception as exc:
        print(f"stream_end {exc!r}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:18081")
    ap.add_argument(
        "--log",
        default="/mnt/vm_8tb/b70/build/vllm-flashnext/logs/d2-profile.log",
    )
    ap.add_argument(
        "--done",
        default="/mnt/vm_8tb/b70/build/vllm-flashnext/logs/d2-profile.done",
    )
    ap.add_argument("--model", default="hotschmoe-dev")
    args = ap.parse_args()
    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    stop = threading.Event()
    prompt = (
        "The capital of France is Paris. Continue with short facts, "
        "one per line, and do not stop early."
    )

    def launch(i: int) -> None:
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt + f" stream {i}."}],
            "max_tokens": 256,
            "temperature": 0,
            "stream": True,
        }
        threading.Thread(
            target=post_stream, args=(url, payload, stop), daemon=True
        ).start()
        print(f"launched stream {i}", flush=True)

    # Launch the next stream when the previous T is armed, while that step
    # is still inside its region bench. The new request then joins the next
    # decode step instead of arriving after the earlier streams have finished.
    marks = (
        "d2_step_armed T=1",
        "d2_step_armed T=2",
        "d2_step_armed T=3",
        "d2_profile_done",
    )
    for i, mark in enumerate(marks):
        launch(i)
        deadline_i = time.time() + 240
        while time.time() < deadline_i:
            try:
                text = open(args.log, encoding="utf-8", errors="replace").read()
            except OSError:
                text = ""
            if mark in text or "d2_profile_done" in text:
                print(f"saw {mark}", flush=True)
                break
            time.sleep(2)
        else:
            print(f"timeout waiting for {mark}", flush=True)
            break
        if "d2_profile_done" in open(args.log, encoding="utf-8", errors="replace").read():
            break

    deadline = time.time() + 300
    while time.time() < deadline:
        try:
            text = open(args.log, encoding="utf-8", errors="replace").read()
        except OSError:
            text = ""
        if "d2_profile_done" in text:
            print("profile done", flush=True)
            break
        time.sleep(2)
    else:
        print("profile timeout", flush=True)
    stop.set()


if __name__ == "__main__":
    main()
