#!/usr/bin/env python3
"""Gate OpenAI tool-call compatibility for the live Ornith endpoint."""

import argparse
import json
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://localhost:18080/v1")
    parser.add_argument("--model", default="ornith-1.5-35b-a3b-W8A8-rtn-mtp-shisa")
    args = parser.parse_args()

    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": "Call b70_add with a=19 and b=23. Do not answer directly.",
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "b70_add",
                    "description": "Add two integers.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "integer"},
                            "b": {"type": "integer"},
                        },
                        "required": ["a", "b"],
                    },
                },
            }
        ],
        "tool_choice": "auto",
        "temperature": 0,
        "max_tokens": 128,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        args.endpoint + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        result = json.load(response)

    message = result["choices"][0]["message"]
    calls = message.get("tool_calls") or []
    summary = {"finish_reason": result["choices"][0].get("finish_reason"), "tool_calls": calls}
    print(json.dumps(summary, indent=2))
    if len(calls) != 1:
        raise SystemExit("expected exactly one tool call")
    function = calls[0].get("function") or {}
    arguments = json.loads(function.get("arguments") or "{}")
    if function.get("name") != "b70_add" or arguments != {"a": 19, "b": 23}:
        raise SystemExit("tool call did not match the requested function and arguments")


if __name__ == "__main__":
    main()
