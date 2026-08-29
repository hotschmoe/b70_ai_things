#!/usr/bin/env python3
"""Fail-closed identity and lifecycle evidence for Terminal-Bench arms."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DTYPE_ALIASES = {
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "torch.bfloat16": "bfloat16",
    "fp16": "float16",
    "float16": "float16",
    "torch.float16": "float16",
}


def normalize_dtype(value: str | None) -> str | None:
    if value is None:
        return None
    return DTYPE_ALIASES.get(value.strip().strip("'\""), value.strip().strip("'\""))


def parse_runtime_log(text: str) -> dict[str, Any]:
    target: str | None = None
    configured_kv: str | None = None
    served: str | None = None

    sglang = re.search(r"server_args=ServerArgs\((.+)", text)
    if sglang:
        line = sglang.group(1)
        match = re.search(r"(?:^|, )dtype='([^']+)'", line)
        target = normalize_dtype(match.group(1) if match else None)
        match = re.search(r"(?:^|, )kv_cache_dtype='([^']+)'", line)
        configured_kv = normalize_dtype(match.group(1) if match else None)
        match = re.search(r"(?:^|, )served_model_name='([^']+)'", line)
        served = match.group(1) if match else None

    vllm = re.search(r"Initializing a V1 LLM engine .*? with config: (.+)", text)
    if vllm:
        line = vllm.group(1)
        match = re.search(r"(?:^|, )dtype=([^,]+)", line)
        target = normalize_dtype(match.group(1) if match else None)
        match = re.search(r"(?:^|, )kv_cache_dtype=([^,]+)", line)
        configured_kv = normalize_dtype(match.group(1) if match else None)
        match = re.search(r"(?:^|, )served_model_name=([^,]+)", line)
        served = match.group(1).strip() if match else None

    observed_kv = {
        normalize_dtype(value)
        for value in re.findall(r"KV Cache is allocated\. dtype: ([^,\s]+)", text)
    }
    observed_kv.discard(None)
    kv_dtype = next(iter(observed_kv)) if len(observed_kv) == 1 else None
    return {
        "served_model_log": served,
        "target_dtype": target,
        "configured_kv_dtype": configured_kv,
        "observed_kv_dtype": kv_dtype,
        "observed_kv_dtypes": sorted(observed_kv),
    }


def validate_identity(
    models: dict[str, Any],
    runtime_log: str,
    *,
    expected_model: str,
    expected_target_dtype: str,
    expected_kv_dtype: str,
) -> dict[str, Any]:
    ids = [item.get("id") for item in models.get("data", []) if isinstance(item, dict)]
    parsed = parse_runtime_log(runtime_log)
    errors: list[str] = []
    if ids != [expected_model]:
        errors.append(f"/v1/models ids {ids!r} != [{expected_model!r}]")
    if parsed["served_model_log"] != expected_model:
        errors.append(
            f"runtime served model {parsed['served_model_log']!r} != {expected_model!r}"
        )
    target = normalize_dtype(expected_target_dtype)
    kv = normalize_dtype(expected_kv_dtype)
    if parsed["target_dtype"] != target:
        errors.append(f"target dtype {parsed['target_dtype']!r} != {target!r}")
    if parsed["observed_kv_dtype"] != kv:
        errors.append(
            f"observed KV dtype {parsed['observed_kv_dtype']!r} != {kv!r}"
        )
    result = {
        "expected_model": expected_model,
        "models_endpoint_ids": ids,
        **parsed,
        "valid": not errors,
        "errors": errors,
    }
    if errors:
        raise ValueError("; ".join(errors))
    return result


def epoch(value: str | int | None) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def build_lifecycle_record(
    *,
    arm: str,
    served_model: str,
    exit_code: int,
    machine_start: int,
    prehealth_end: int | None,
    server_start: int | None,
    server_ready: int | None,
    harbor_end: int | None,
    preteardown_check: int | None,
    teardown_end: int | None,
    posthealth_end: int,
    endpoint_healthy_before_teardown: bool | None,
    endpoint_down_after_teardown: bool | None,
    pre_card_health: bool,
    pre_collective_health: bool,
    post_card_health: bool,
    post_collective_health: bool,
    fatal_server_markers: list[str],
) -> dict[str, Any]:
    ordered = [
        value
        for value in (
            machine_start,
            prehealth_end,
            server_start,
            server_ready,
            harbor_end,
            preteardown_check,
            teardown_end,
            posthealth_end,
        )
        if value is not None
    ]
    if ordered != sorted(ordered):
        raise ValueError(f"lifecycle epochs are not monotonic: {ordered}")
    data: dict[str, Any] = {
        "arm": arm,
        "served_model": served_model,
        "exit_code": exit_code,
        "machine_start_epoch": machine_start,
        "prehealth_end_epoch": prehealth_end,
        "server_start_epoch": server_start,
        "server_ready_epoch": server_ready,
        "harbor_end_epoch": harbor_end,
        "preteardown_check_epoch": preteardown_check,
        "teardown_end_epoch": teardown_end,
        "posthealth_end_epoch": posthealth_end,
        "endpoint_healthy_before_teardown": endpoint_healthy_before_teardown,
        "endpoint_down_after_teardown": endpoint_down_after_teardown,
        "pre_card_health": pre_card_health,
        "pre_collective_health": pre_collective_health,
        "post_card_health": post_card_health,
        "post_collective_health": post_collective_health,
        "fatal_server_markers": fatal_server_markers,
        "full_machine_seconds": posthealth_end - machine_start,
    }
    if server_start is not None and server_ready is not None:
        data["startup_seconds"] = server_ready - server_start
    if server_ready is not None and harbor_end is not None:
        data["harbor_seconds"] = harbor_end - server_ready
    if server_start is not None and harbor_end is not None:
        data["server_start_through_harbor_seconds"] = harbor_end - server_start
    endpoint_contract = (
        True
        if server_start is None
        else endpoint_healthy_before_teardown is True
        and endpoint_down_after_teardown is True
    )
    data["health_contract_passed"] = all(
        (
            pre_card_health,
            pre_collective_health,
            post_card_health,
            post_collective_health,
            endpoint_contract,
            not fatal_server_markers,
        )
    )
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="ascii")


def identity_command(args: argparse.Namespace) -> None:
    models = json.loads(args.models_json.read_text(encoding="utf-8"))
    log = args.server_log.read_text(encoding="utf-8", errors="replace")
    result = validate_identity(
        models,
        log,
        expected_model=args.expected_model,
        expected_target_dtype=args.expected_target_dtype,
        expected_kv_dtype=args.expected_kv_dtype,
    )
    write_json(args.output, result)
    print(json.dumps(result, sort_keys=True))


def lifecycle_command(args: argparse.Namespace) -> None:
    markers = json.loads(args.fatal_server_markers)
    record = build_lifecycle_record(
        arm=args.arm,
        served_model=args.served_model,
        exit_code=args.exit_code,
        machine_start=args.machine_start,
        prehealth_end=epoch(args.prehealth_end),
        server_start=epoch(args.server_start),
        server_ready=epoch(args.server_ready),
        harbor_end=epoch(args.harbor_end),
        preteardown_check=epoch(args.preteardown_check),
        teardown_end=epoch(args.teardown_end),
        posthealth_end=args.posthealth_end,
        endpoint_healthy_before_teardown=args.endpoint_healthy_before_teardown,
        endpoint_down_after_teardown=args.endpoint_down_after_teardown,
        pre_card_health=args.pre_card_health,
        pre_collective_health=args.pre_collective_health,
        post_card_health=args.post_card_health,
        post_collective_health=args.post_collective_health,
        fatal_server_markers=markers,
    )
    write_json(args.output, record)
    print(json.dumps(record, sort_keys=True))


def boolean(value: str) -> bool | None:
    if value == "null":
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    raise argparse.ArgumentTypeError("expected true, false, or null")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    identity = commands.add_parser("identity")
    identity.add_argument("--models-json", required=True, type=Path)
    identity.add_argument("--server-log", required=True, type=Path)
    identity.add_argument("--expected-model", required=True)
    identity.add_argument("--expected-target-dtype", required=True)
    identity.add_argument("--expected-kv-dtype", required=True)
    identity.add_argument("--output", required=True, type=Path)
    identity.set_defaults(func=identity_command)

    lifecycle = commands.add_parser("lifecycle")
    lifecycle.add_argument("--output", required=True, type=Path)
    lifecycle.add_argument("--arm", required=True)
    lifecycle.add_argument("--served-model", required=True)
    lifecycle.add_argument("--exit-code", required=True, type=int)
    lifecycle.add_argument("--machine-start", required=True, type=int)
    for name in (
        "prehealth-end",
        "server-start",
        "server-ready",
        "harbor-end",
        "preteardown-check",
        "teardown-end",
    ):
        lifecycle.add_argument(f"--{name}", default="")
    lifecycle.add_argument("--posthealth-end", required=True, type=int)
    lifecycle.add_argument("--endpoint-healthy-before-teardown", type=boolean)
    lifecycle.add_argument("--endpoint-down-after-teardown", type=boolean)
    lifecycle.add_argument("--pre-card-health", required=True, type=boolean)
    lifecycle.add_argument("--pre-collective-health", required=True, type=boolean)
    lifecycle.add_argument("--post-card-health", required=True, type=boolean)
    lifecycle.add_argument("--post-collective-health", required=True, type=boolean)
    lifecycle.add_argument("--fatal-server-markers", default="[]")
    lifecycle.set_defaults(func=lifecycle_command)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
