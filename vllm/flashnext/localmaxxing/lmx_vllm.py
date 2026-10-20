#!/usr/bin/env python3
"""Prepare dry-run plans or collect local evidence. There is no submission path."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import secrets
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "harness"))
from bench_latency import get_json, get_text  # noqa: E402
from promo_gate import identity_signature  # noqa: E402

COUNTERS = (
    "vllm:request_success_total", "vllm:generation_tokens_total",
    "vllm:prompt_tokens_total", "vllm:prefix_cache_hits_total",
    "vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
)
GAUGES = ("vllm:num_requests_running", "vllm:num_requests_waiting")


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="ascii")


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def public_config(value):
    if isinstance(value, dict):
        return {key: ("<redacted>" if key.lower() in
                      ("hf_token", "api_key", "access_token", "password") and item else public_config(item))
                for key, item in value.items()}
    if isinstance(value, list):
        return [public_config(item) for item in value]
    return value


def canonical(suite):
    manifest = json.loads((HERE / "prompts.json").read_text())
    row = manifest[suite]
    text = (HERE / row["file"]).read_text()
    if digest(text) != row["sha256"]:
        raise ValueError("canonical prompt hash mismatch")
    return text, row


def metric_values(text):
    result = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r'([^ {]+)(?:\{.*\})?\s+([^ ]+)(?:\s+\d+)?', line)
        if match and match[1] in COUNTERS + GAUGES:
            result[match[1]] = result.get(match[1], 0) + float(match[2])
    return result


def metrics(base, path):
    text = get_text(base + "/metrics")
    Path(path).write_text(text, encoding="utf-8")
    return metric_values(text)


def idle(values):
    return all(values.get(key) == 0 for key in GAUGES)


def deltas(before, after):
    return {key: after[key] - before.get(key, 0) for key in COUNTERS if key in after}


def parse_stream(response, started):
    events, pieces, arrivals = [], [], []
    usage = native_timings = None
    done = False
    response_id = None
    for raw in response:
        line = raw.decode("utf-8").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        now = time.perf_counter_ns()
        if data == "[DONE]":
            done = True
            break
        event = json.loads(data)
        if event.get("error"):
            raise ValueError(f"stream error: {event['error']}")
        events.append({"offset_ns": now - started, "event": event})
        response_id = event.get("id") or response_id
        usage = event.get("usage") or usage
        native_timings = event.get("timings") or native_timings
        choices = event.get("choices") or []
        delta = choices[0].get("delta", {}) if choices else {}
        # Match lmx generatedContent: prefer reasoning over reasoning_content.
        piece = (delta.get("reasoning") or delta.get("reasoning_content") or "") + (delta.get("content") or "")
        if piece:
            arrivals.append(now - started)
            pieces.append(piece)
    if not done or not usage or len(arrivals) < 2:
        raise ValueError("incomplete stream, missing usage, or fewer than two content events")
    n = usage.get("completion_tokens")
    if type(n) is not int or n < 2:
        raise ValueError("missing authoritative completion token count")
    elapsed = (arrivals[-1] - arrivals[0]) / 1e9
    if elapsed <= 0:
        raise ValueError("zero decode window")
    return dict(events=events, output="".join(pieces), usage=usage,
                response_id=response_id, engine_timings=native_timings,
                content_arrivals_ns=arrivals, content_chunks=len(arrivals),
                first_content_ns=arrivals[0], last_content_ns=arrivals[-1],
                tok_s_out=(n - 1) / elapsed, effective_itl_ms=elapsed * 1000 / (n - 1),
                ttft_ms=arrivals[0] / 1e6,
                chunk_intervals_ms=[(b - a) / 1e6 for a, b in zip(arrivals, arrivals[1:])])


def one(args, text, directory):
    directory.mkdir()
    before = metrics(args.base_url, directory / "metrics-before.txt")
    if not idle(before):
        raise ValueError("server is busy or request gauges are unavailable")
    body = dict(model=args.served_model,
                messages=[{"role": "user", "content":
                           f"[LocalMaxxing cache-bust nonce: {secrets.token_hex(16)}]\n" + text}],
                temperature=0, max_tokens=args.max_tokens, stream=True,
                stream_options={"include_usage": True})
    # Thinking stays at the server default, matching the Strata suites and CLI.
    dump(directory / "request.json", body)
    request = Request(args.base_url + "/v1/chat/completions", json.dumps(body).encode(),
                      {"Content-Type": "application/json", "Accept": "text/event-stream"})
    started = time.perf_counter_ns()
    with urlopen(request, timeout=args.timeout) as response:
        row = parse_stream(response, started)
    dump(directory / "stream.json", row)
    n = row["usage"]["completion_tokens"]
    # Metrics may be exported on a periodic engine tick. Bound that wait;
    # it is outside the measured HTTP interval and never changes the rate.
    deadline = time.monotonic() + args.metrics_wait
    while True:
        after = metrics(args.base_url, directory / "metrics-after.txt")
        diff = deltas(before, after)
        if (idle(after) and diff.get(COUNTERS[0]) == 1
                and diff.get(COUNTERS[1]) == n) or time.monotonic() >= deadline:
            break
        time.sleep(0.2)
    errors = []
    if not idle(after) or diff.get(COUNTERS[0]) != 1 or diff.get(COUNTERS[1]) != n:
        errors.append("request/token counter deltas do not isolate exactly one request")
    if any(value < 0 for value in diff.values()):
        errors.append("server counters reset")
    if diff.get("vllm:prefix_cache_hits_total", 0) != 0:
        errors.append("prefix-cache hits despite leading nonce; rerun with prefix cache disabled")
    drafts = diff.get("vllm:spec_decode_num_draft_tokens_total", 0)
    accepted = diff.get("vllm:spec_decode_num_accepted_tokens_total", 0)
    if args.mode == "no-spec" and (drafts != 0 or accepted != 0):
        errors.append("draft activity in a declared no-spec run")
    if args.mode == "mtp" and (drafts <= 0 or not 0 <= accepted <= drafts):
        errors.append("missing or inconsistent MTP counters")
    row.update(metrics_before=before, metrics_after=after, metric_deltas=diff,
               valid=not errors, issues=errors)
    dump(directory / "sample.json", row)
    return row


def measure(args, suites, hardware, command):
    identity = get_json(args.base_url + "/v1/models")
    ids = [row["id"] for row in identity["data"]]
    if ids[0] not in ("hotschmoe-dev", "hotschmoe-dd") or args.served_model != ids[0] or args.alias not in ids[1:]:
        raise ValueError("stable first model ID / expected research alias mismatch")
    registry = HERE.parents[2] / "evals/configs/models.yaml"
    registered = set(re.findall(r"^\s+served_model_id: (.+)$", registry.read_text(), re.MULTILINE))
    if args.alias not in registered or args.alias in ("hotschmoe-dev", "hotschmoe-dd"):
        raise ValueError("research alias is not registered in evals/configs/models.yaml")
    version = get_json(args.base_url + "/version")
    if version.get("version") != args.engine_version:
        raise ValueError("--engine-version differs from live /version (wheel metadata may differ from fork imports)")
    # A JSON server config proves no-spec versus MTP; absence is not proof of no-spec.
    info = get_json(args.base_url + "/server_info?config_format=json")
    config = info["vllm_config"]
    if not isinstance(config, dict) or "speculative_config" not in config:
        raise ValueError("server_info lacks structured speculative_config")
    spec = config["speculative_config"]
    if args.mode == "no-spec" and spec is not None:
        raise ValueError("no-spec requires speculative_config=null at server startup")
    if args.mode == "mtp" and (not isinstance(spec, dict) or str(spec.get("method")).lower() != "mtp"
                               or spec.get("num_speculative_tokens") != args.spec_tokens):
        raise ValueError("MTP method/depth does not match live server config")
    # Store only the config, not server environment variables (which may contain secrets).
    dump(Path(args.out) / "identity.json", identity)
    dump(Path(args.out) / "server-config.json", public_config(config))
    dump(Path(args.out) / "server-version.json", version)
    for suite in suites:
        text, canon = canonical(suite)
        out = Path(args.out) / f"{args.mode}-{suite}"
        out.mkdir()
        warmup = one(args, text, out / "warmup")
        if not warmup["valid"]:
            raise ValueError(f"{suite}: warmup isolation failed; see warmup/sample.json")
        rows = [one(args, text, out / f"iteration-{i + 1}") for i in range(3)]
        med = sorted(rows, key=lambda row: row["tok_s_out"])[1]
        diff = med["metric_deltas"]
        flags = dict(commandSnippet=command, concurrency=1, temperature=0, stream=True,
                     maxTokens=args.max_tokens, warmup=1, iterations=3, mode="remote",
                     servedModel=args.served_model, prefixCacheBust="leading_nonce_per_request",
                     promptSource=f"canonical:{suite}", specDecoding=args.mode == "mtp")
        if args.mode == "mtp":
            draft = diff.get("vllm:spec_decode_num_draft_tokens_total", 0)
            accept = diff.get("vllm:spec_decode_num_accepted_tokens_total", 0)
            passes = diff.get("vllm:spec_decode_num_drafts_total", 0)
            flags.update(specMethod="MTP", specNumTokens=args.spec_tokens,
                         specDraftTokens=draft, specAcceptedTokens=accept)
            if draft > 0:
                flags["specAcceptanceRate"] = accept / draft
            if passes > 0:
                flags["specMeanAcceptedLength"] = 1 + accept / passes
        engine_raw = dict(usage=med["usage"], nativeTimings=med["engine_timings"],
                          prometheusCounterDeltas=diff,
                          provenance="vLLM server usage and isolated global metric deltas; not per-request engine timings")
        payload = dict(engineName="vllm", backend="xpu", hardware=hardware, batchSize=1,
                       benchmarkMode="remote", hfId=args.hf_id, quantization=args.quantization,
                       engineVersion=args.engine_version, engineCommit=args.engine_commit,
                       engineBuild=args.engine_build, engineRepository="https://github.com/vllm-project/vllm",
                       engineFlags=flags, metricSource="remote_endpoint", timingSource="client_observed_http",
                       samples=[dict(iteration=i + 1, tokSOut=r["tok_s_out"], ttftMs=r["ttft_ms"],
                                     outputTokens=r["usage"]["completion_tokens"],
                                     promptTokens=r["usage"]["prompt_tokens"],
                                     draftN=r["metric_deltas"].get("vllm:spec_decode_num_draft_tokens_total"),
                                     draftAccepted=r["metric_deltas"].get("vllm:spec_decode_num_accepted_tokens_total"))
                                for i, r in enumerate(rows)],
                       tokSOut=round(med["tok_s_out"], 1), tokSOutSource="inter_token",
                       ttftMs=round(med["ttft_ms"], 2), ttftSource="stream_first_token",
                       outputTokens=med["usage"]["completion_tokens"],
                       promptTokens=med["usage"]["prompt_tokens"], prefillTokens=0,
                       prompt=text, promptSha256=canon["sha256"], promptSample=text[:2000],
                       outputSha256=digest(med["output"]), outputSample=med["output"][:4000],
                       engineTimingsRaw=engine_raw)
        dump(out / "payload.json", payload)
        issues = [issue for row in rows for issue in row["issues"]]
        if any(row["usage"]["completion_tokens"] < canon["min_output_tokens"] for row in rows):
            issues.append("output below canonical minimum")
        if identity_signature(identity) != identity_signature(get_json(args.base_url + "/v1/models")):
            issues.append("identity changed during suite")
        final_info = get_json(args.base_url + "/server_info?config_format=json")
        if config != final_info.get("vllm_config"):
            issues.append("server configuration changed during suite")
        report = dict(valid=not issues, issues=issues, submitted=False,
                      ranking_eligible=args.mode == "no-spec" and not issues,
                      median_iteration=rows.index(med) + 1,
                      cli_metric_medians=dict(tokSOut=statistics.median(r["tok_s_out"] for r in rows),
                                              ttftMs=statistics.median(r["ttft_ms"] for r in rows)),
                      verification="pending localmaxxing dry-run validation; no badge claimed")
        dump(out / "validation.json", report)
        if issues:
            raise ValueError(f"{suite}: invalid measurement; see validation.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--served-model", default="hotschmoe-dev")
    parser.add_argument("--alias", required=True)
    parser.add_argument("--hf-id", required=True)
    parser.add_argument("--quantization", default="IQ3_XXS")
    parser.add_argument("--hardware", required=True)
    parser.add_argument("--engine-version", required=True)
    parser.add_argument("--engine-commit", required=True)
    parser.add_argument("--engine-build", required=True)
    parser.add_argument("--command-file", required=True)
    parser.add_argument("--mode", choices=("no-spec", "mtp"), required=True)
    parser.add_argument("--spec-tokens", type=int)
    parser.add_argument("--suite", choices=("all", "reasoning-v1", "code-v1"), default="all")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--metrics-wait", type=float, default=20)
    parser.add_argument("--lmx", default=str(Path.home() / ".local/bin/lmx"))
    parser.add_argument("--out", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="default: write plans only")
    mode.add_argument("--measure-local", action="store_true", help="future orchestrator use: HTTP measurement, never submits")
    args = parser.parse_args()
    if args.mode == "mtp" and (args.spec_tokens is None or args.spec_tokens < 1):
        parser.error("MTP requires --spec-tokens >= 1")
    if args.mode == "no-spec" and args.spec_tokens is not None:
        parser.error("no-spec must not specify draft tokens")
    args.base_url = args.base_url.rstrip("/").removesuffix("/v1")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    hardware = json.loads(Path(args.hardware).read_text())
    command = Path(args.command_file).read_text().strip()
    if not command or len(command) > 4000:
        parser.error("command file must contain 1..4000 characters")
    suites = ("reasoning-v1", "code-v1") if args.suite == "all" else (args.suite,)
    if args.measure_local:
        measure(args, suites, hardware, command)
        return
    for suite in suites:
        text, canon = canonical(suite)
        prompt_path = HERE / canon["file"]
        argv = [args.lmx, "speed-test", "run", "vllm", "--mode", "remote", "--base-url", args.base_url,
                "--served-model", args.served_model, "--hf-id", args.hf_id,
                "--quantization", args.quantization, "--hardware", str(Path(args.hardware).resolve()),
                "--backend", "xpu", "--engine-version", args.engine_version,
                "--engine-commit", args.engine_commit, "--engine-build", args.engine_build,
                "--engine-repository", "https://github.com/vllm-project/vllm",
                "--prompt-file", str(prompt_path), "--max-tokens", str(args.max_tokens),
                "--temperature", "0", "--warmup", "1", "--iterations", "3",
                "--concurrency", "1", "--dry-run", "--json"]
        if args.mode == "mtp":
            argv += ["--spec-decoding", "--spec-method", "MTP", "--spec-num-tokens", str(args.spec_tokens)]
        result = subprocess.run(argv, capture_output=True, text=True)
        dump(out / f"{args.mode}-{suite}-plan.json",
             dict(dry_run=True, measured=False, submitted=False, backend="xpu", suite=suite,
                  prompt_sha256=digest(text), expected_alias=args.alias, command_snippet=command,
                  lmx_argv=argv, shell_command=shlex.join(argv),
                  lmx_exit=result.returncode, lmx_stdout=result.stdout, lmx_stderr=result.stderr))
        if result.returncode:
            raise RuntimeError(f"lmx dry-run failed: {ascii(result.stderr)}")
        print(json.dumps(dict(suite=suite, mode=args.mode, dry_run=True, submitted=False)))


if __name__ == "__main__":
    main()
