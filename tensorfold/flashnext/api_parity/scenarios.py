#!/usr/bin/env python3
"""Shared scripted model outputs for the API parity harness (CPU only, no GPU, no weights).

Each script is a list of chunks; a chunk is a list of text pieces the mock engine commits together.
  - TensorFold's fake_serve (zig/tests/server) replays the chunks as tokens (TF_FAKE_SCRIPTS=scripts.json).
  - Strata's serve/server.py MockEngine replays the joined text (strata_mock_launch.py picks by @script=NAME).
A request selects its script with "@script=NAME" anywhere in its text.
"""
import json
import sys

LONG_BODY = [f"line {i:02d}: the quick brown fox jumps over the lazy dog" for i in range(40)]


def tool(name, params):
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


def pieces(text, size):
    return [[text[i:i + size]] for i in range(0, len(text), size)]


SCRIPTS = {
    "default": {"chunks": [["Hello"], [" world"], ["<EOS>"]]},
    "think": {"chunks": [["Let me think"], [" about it"], ["</think>"], ["\n\nThe answer is 42."], ["<EOS>"]]},
    "think_cached": {"cached": 7, "chunks": [["Let me think"], ["</think>"], ["\n\nCached answer."], ["<EOS>"]]},
    "plain": {"chunks": [["Hello"], [" world"], ["!"], ["<EOS>"]]},
    "tool_nothink": {"chunks": [["<tool_call>\n<function=get_weather>\n"], ["<parameter=city>\nBoise\n</parameter>\n"],
                                ["<parameter=units>\ncelsius\n</parameter>\n</function>\n</tool_call>"], ["<EOS>"]]},
    "tool_think": {"chunks": [["Need the weather."], ["</think>"], ["\n\n"], [tool("get_weather", {"city": "Boise"})], ["<EOS>"]]},
    "tool_text_then_call": {"chunks": [["Checking."], ["\n\n"], [tool("get_weather", {"city": "Boise"})], ["<EOS>"]]},
    "tool_two": {"chunks": [[tool("get_weather", {"city": "Boise"})], ["\n\n"], [tool("get_weather", {"city": "Reno"})], ["<EOS>"]]},
    "tool_long": {"chunks": [["<tool_call>\n<function=write_file>\n<parameter=path>\nnotes.txt\n</parameter>\n<parameter=content>\n"]]
                  + pieces("\n".join(LONG_BODY), 120) + [["\n</parameter>\n</function>\n</tool_call>"], ["<EOS>"]]},
    # what the model writes AFTER a forced call opening (tool_choice required / named on an engine with no grammar): the server
    # has already written "<tool_call>\n" (several tools) or "<tool_call>\n<function=NAME>\n" (one tool) at the end of the prompt
    "tool_after_open": {"chunks": [["<function=get_weather>\n<parameter=city>\nBoise\n</parameter>\n</function>\n</tool_call>"], ["<EOS>"]]},
    "tool_after_named": {"chunks": [["<parameter=city>\nBoise\n</parameter>\n", "<parameter=units>\ncelsius\n</parameter>\n</function>\n</tool_call>"], ["<EOS>"]]},
    "tool_typed": {"chunks": [["<tool_call>\n<function=configure>\n<parameter=count>\n5\n</parameter>\n<parameter=ratio>\n0.25\n</parameter>\n"],
                              ["<parameter=enabled>\ntrue\n</parameter>\n<parameter=tags>\n[\"a\", \"b\"]\n</parameter>\n"],
                              ["<parameter=options>\n{\"depth\": 2, \"mode\": \"fast\"}\n</parameter>\n<parameter=note>\nhello\n</parameter>\n</function>\n</tool_call>"], ["<EOS>"]]},
    "answer_after_tool": {"chunks": [["It is 58 F and clear in Boise."], ["<EOS>"]]},
    "json_object": {"chunks": [['{"city": "Boise", "temp_f": 58}'], ["<EOS>"]]},
    "json_bad": {"chunks": [["this is not json"], ["<EOS>"]]},
    "stop_words": {"chunks": [["Hello"], [" wor"], ["ld"], [" STOP"], [" there"], ["<EOS>"]]},
    "long_text": {"chunks": pieces("word " * 200, 8) + [["<EOS>"]]},
    "slow": {"delay_ms": 150, "chunks": [["a"], ["b"], ["c"], ["d"], ["<EOS>"]]},
    "unicode": {"chunks": [["café "], ["日本語"], [" done"], ["<EOS>"]]},
    "echo": {"echo_request": True, "chunks": [["<EOS>"]]},
}


def strata_text(name):
    """The joined text of a script (no <EOS>), what Strata's MockEngine replays before its own end-of-turn token."""
    return "".join(p for chunk in SCRIPTS[name]["chunks"] for p in chunk if p != "<EOS>")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "scripts.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(SCRIPTS, f, ensure_ascii=False, indent=1)
    print("wrote", out, len(SCRIPTS), "scripts")
