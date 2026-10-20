#!/usr/bin/env python3
"""Read-only HTTP promotion gate. No server lifecycle or GPU access."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from bench_latency import COHERENCE_PROMPT, get_json, post_json

# Same four raw-completion prompts and sampling settings as x1_compare_greedy.py.
REPEAT_PROMPTS = (
    COHERENCE_PROMPT,
    'def fibonacci(n):\n    """Return the first n Fibonacci numbers."""\n',
    "Count the integers in order, separated by commas: 1, 2, 3,",
    "Explain how a hash table resolves collisions in three sentences.",
)
C4_PARIS = " Paris.\nThe capital of France is Paris.\nThe capital of France is"


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def identity_signature(body):
    # vLLM constructs a new ModelCard.created timestamp for every GET. It
    # describes response construction, not a model reload or identity change.
    return [{key: row.get(key) for key in ("id", "root", "parent", "owned_by", "max_model_len")}
            for row in body.get("data", [])]


def token_bytes(tokens):
    if not isinstance(tokens, list) or not tokens:
        raise ValueError("response is missing token_ids")
    if any(type(t) is not int or not 0 <= t < 2**32 for t in tokens):
        raise ValueError("invalid token ID")
    return struct.pack(f"<{len(tokens)}I", *tokens)


def mismatch(a, b):
    return next((i for i in range(max(len(a), len(b)))
                 if i >= len(a) or i >= len(b) or a[i] != b[i]), None)


class Gate:
    def __init__(self, args):
        self.args = args
        self.base = args.base_url.rstrip("/").removesuffix("/v1")
        self.model = None
        self.result = dict(schema_version=1, agent="X4",
                           started_at=datetime.now(timezone.utc).isoformat(),
                           base_url=self.base, profile=args.profile, checks={})

    def post(self, path, body):
        return post_json(self.base + path, body, self.args.timeout)

    def check(self, name, fn):
        try:
            row = fn()
            if type(row.get("pass")) is not bool:
                raise ValueError("check did not return a boolean pass")
        except Exception as exc:
            row = {"pass": False, "error": ascii(str(exc))}
        self.result["checks"][name] = row
        # Save partial evidence too, including after an HTTP failure.
        self.save()
        print(json.dumps({"check": name, **{k: v for k, v in row.items()
                                          if k in ("pass", "error")}}, ensure_ascii=True), flush=True)
        return row["pass"]

    def save(self):
        out = Path(self.args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        temp = out.with_suffix(out.suffix + ".tmp")
        temp.write_text(json.dumps(self.result, indent=2, ensure_ascii=True) + "\n", encoding="ascii")
        temp.replace(out)

    def identity(self):
        body = get_json(self.base + "/v1/models", self.args.timeout)
        ids = [row["id"] for row in body.get("data", [])]
        self.model = ids[0] if ids else None
        registry = Path(__file__).resolve().parents[3] / "evals/configs/models.yaml"
        registered = set(re.findall(r"^\s+served_model_id: (.+)$", registry.read_text(), re.MULTILINE))
        aliases = [x for x in ids[1:] if x in registered and x not in ("hotschmoe-dev", "hotschmoe-dd")]
        ok = self.model in ("hotschmoe-dev", "hotschmoe-dd") and bool(aliases)
        if self.args.expect_alias:
            ok = ok and self.args.expect_alias in ids[1:] and self.args.expect_alias in registered
        return {"pass": ok, "response": body, "registered_aliases": aliases,
                "expect_alias": self.args.expect_alias, "registry": str(registry)}

    def complete(self, prompt, count, forced=True):
        body = dict(model=self.model, prompt=prompt, temperature=0, seed=0,
                    max_tokens=count, ignore_eos=forced, return_token_ids=True)
        response = self.post("/v1/completions", body)
        choice = response["choices"][0]
        tokens = choice.get("token_ids")
        binary = token_bytes(tokens)
        if forced and len(tokens) != count:
            raise ValueError(f"forced {count} tokens, got {len(tokens)}")
        return dict(text=choice["text"], token_ids=tokens,
                    token_bytes_sha256=hashlib.sha256(binary).hexdigest(),
                    text_sha256=sha(choice["text"]), usage=response.get("usage"),
                    finish_reason=choice.get("finish_reason"), response_id=response.get("id"))

    def greedy(self, a, b):
        prompt = f"Return only the integer: {a} * {b}."
        body = dict(model=self.model, messages=[{"role": "user", "content": prompt}],
                    temperature=0, seed=0, max_tokens=32,
                    chat_template_kwargs={"enable_thinking": False})
        response = self.post("/v1/chat/completions", body)
        text = response["choices"][0]["message"].get("content") or ""
        return {"pass": text.strip() == str(a * b), "expected": str(a * b),
                "actual": text, "request": body, "response": response}

    def paris(self):
        row = self.complete(COHERENCE_PROMPT, 16)
        return {"pass": row["text"] == C4_PARIS, "expected": C4_PARIS, **row}

    def repeat(self):
        rounds = {}
        # Two serial passes, followed by two passes of pairs. Every prompt is
        # compared against its own first serial result, including C2 results.
        for mode in ("c1", "c2"):
            for repeat in range(2):
                rows = []
                for offset in range(0, 4, 1 if mode == "c1" else 2):
                    prompts = REPEAT_PROMPTS[offset:offset + (1 if mode == "c1" else 2)]
                    if mode == "c1":
                        rows.append(self.complete(prompts[0], 96))
                    else:
                        barrier = threading.Barrier(2)
                        def one(prompt):
                            barrier.wait(timeout=10)
                            return self.complete(prompt, 96)
                        with ThreadPoolExecutor(max_workers=2) as pool:
                            rows.extend(pool.map(one, prompts))
                rounds[f"{mode}_{repeat + 1}"] = rows
                self.result["repeat_rounds"] = rounds
                self.save()
        baseline = rounds["c1_1"]
        comparisons = []
        for label, rows in rounds.items():
            for i, (a, b) in enumerate(zip(baseline, rows, strict=True)):
                comparisons.append(dict(round=label, prompt_index=i,
                    pass_tokens=token_bytes(a["token_ids"]) == token_bytes(b["token_ids"]),
                    pass_text=a["text"].encode("utf-8") == b["text"].encode("utf-8"),
                    first_mismatch=mismatch(a["token_ids"], b["token_ids"])))
        return {"pass": all(x["pass_tokens"] and x["pass_text"] for x in comparisons),
                "prompts": REPEAT_PROMPTS, "forced_tokens": 96, "temperature": 0,
                "token_serialization": "uint32 little endian", "comparisons": comparisons,
                "concurrency": "two client requests released by a barrier; engine overlap not inferred"}

    def tokenize(self, text):
        result = self.post("/tokenize", dict(model=self.model, prompt=text, add_special_tokens=False))
        tokens = result.get("tokens")
        token_bytes(tokens)
        return tokens

    def long(self, count):
        # Token-ID input allows exact lengths without a tokenizer dependency.
        # Three needles probe early, middle and late context, not just the tail.
        prefix = self.tokenize("<|im_start|>user\nRead this archive. Remember its three access words.\n"
                               "The first access word is violet.\n")
        middle = self.tokenize("\nThe second access word is cedar.\n")
        suffix = self.tokenize("\nThe third access word is copper.\n"
                               "Return the first, second, and third access words, separated by single spaces. "
                               "Return only those three words.\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        remaining = count - len(prefix) - len(middle) - len(suffix)
        if remaining <= 0:
            raise ValueError("long prompt too short for needles")
        filler = self.tokenize("Archive filler: no access words in this record.\n" * (count // 4))
        if len(filler) < remaining:
            raise ValueError("tokenizer returned insufficient filler")
        half = remaining // 2
        tokens = prefix + filler[:half] + middle + filler[half:remaining] + suffix
        assert len(tokens) == count
        row = self.complete(tokens, 32, forced=False)
        usage = row.get("usage") or {}
        return {"pass": row["text"].strip() == "violet cedar copper" and usage.get("prompt_tokens") == count,
                "expected": "violet cedar copper", "requested_prompt_tokens": count,
                "prompt_token_bytes_sha256": hashlib.sha256(token_bytes(tokens)).hexdigest(),
                "needle_offsets": [len(prefix), len(prefix) + half, count - len(suffix)], **row}

    def run(self):
        required = ["identity", "greedy_17x19", "greedy_23x29", "exact_repeat",
                    "long_8192", "long_32768", "identity_after", "identity_unchanged"]
        if self.args.profile == "proxy35b":
            required.append("c4_paris")
        self.result["checks"] = {name: {"pass": False, "error": "not run"} for name in required}
        if self.check("identity", self.identity):
            self.check("greedy_17x19", lambda: self.greedy(17, 19))
            self.check("greedy_23x29", lambda: self.greedy(23, 29))
            if self.args.profile == "proxy35b":
                self.check("c4_paris", self.paris)
            self.check("exact_repeat", self.repeat)
            for count in (8192, 32768):
                self.check(f"long_{count}", lambda count=count: self.long(count))
            self.check("identity_after", self.identity)
            before = self.result["checks"]["identity"].get("response")
            after = self.result["checks"]["identity_after"].get("response")
            self.check("identity_unchanged", lambda: {
                "pass": before is not None and after is not None
                        and identity_signature(before) == identity_signature(after),
                "fields": ["id", "root", "parent", "owned_by", "max_model_len"]})
        self.result["pass"] = all(row["pass"] for row in self.result["checks"].values())
        self.result["finished_at"] = datetime.now(timezone.utc).isoformat()
        self.save()
        return 0 if self.result["pass"] else 3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--profile", choices=("proxy35b", "flashnext"), required=True)
    parser.add_argument("--expect-alias")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--out", required=True)
    return Gate(parser.parse_args()).run()


if __name__ == "__main__":
    raise SystemExit(main())
