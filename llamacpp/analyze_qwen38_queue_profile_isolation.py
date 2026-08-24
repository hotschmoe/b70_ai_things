#!/usr/bin/env python3
"""Fail-closed analyzer for the queue-profiling isolation startup pair."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ARMS = ("queue_profile_off", "queue_profile_on")
FATAL_RE = re.compile(r"device_lost|out_of_resources|ur_result_error", re.I)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="ascii"))


def inspect_arm(root: Path, arm: str, manifest: dict) -> dict:
    directory = root / arm
    server = (directory / "server.log").read_text(encoding="utf-8", errors="replace")
    start_rc = int((directory / "start.rc").read_text(encoding="ascii").strip())
    inspect_items = load_json(directory / "container_inspect.json")
    if len(inspect_items) != 1:
        raise ValueError(f"{arm}: expected exactly one inspect record")
    inspect = inspect_items[0]
    env = {}
    for item in inspect.get("Config", {}).get("Env", []):
        if "=" in item:
            key, value = item.split("=", 1)
            env[key] = value
    expected_profile = "0" if arm == "queue_profile_off" else "1"
    restart = inspect.get("HostConfig", {}).get("RestartPolicy", {}).get("Name")
    restart_count = inspect.get("RestartCount")
    checks = {
        "image": inspect.get("Image") == manifest["image_id"],
        "gpu_count": env.get("GPU_COUNT") == str(manifest["gpu_count"]),
        "timing_period": env.get("GGML_SYCL_QUANT_TIMING_SAMPLE") == "64",
        "timing_skip_unreachable": env.get("GGML_SYCL_QUANT_TIMING_SKIP")
        == "18446744073709551615",
        "queue_profile": env.get("GGML_SYCL_QUANT_TIMING_QUEUE_PROFILE")
        == expected_profile,
        "legacy_profile_off": env.get("GGML_SYCL_PROFILE", "0") == "0",
        "restart_disabled": restart == "no",
        "never_restarted": restart_count == 0,
        "no_timing_events": "[QUANT-TIMING]" not in server,
    }
    fatal = bool(FATAL_RE.search(server))
    return {
        "checks": checks,
        "start_rc": start_rc,
        "fatal": fatal,
        "device_lost": bool(re.search(r"device_lost", server, re.I)),
        "mul_mat_failure": "Error OP MUL_MAT" in server,
        "restart_policy": restart,
        "restart_count": restart_count,
    }


def analyze(root: Path) -> dict:
    manifest = load_json(root / "manifest.json")
    endpoint = load_json(root / "endpoint_down.json")
    rows = {arm: inspect_arm(root, arm, manifest) for arm in ARMS}
    cards = "0" if manifest["gpu_count"] == 1 else "0 1"
    health = {}
    for label in ("pre", "after_off", "final"):
        health[label] = (
            int((root / f"health_{label}.rc").read_text(encoding="ascii").strip())
            == 0
            and f"xpu-health: HEALTHY (cards {cards})"
            in (root / f"health_{label}.log").read_text(
                encoding="utf-8", errors="replace"
            )
        )
    code_hashes_stable = (
        int((root / "code_sha256_check.rc").read_text(encoding="ascii").strip())
        == 0
    )
    off = rows["queue_profile_off"]
    on = rows["queue_profile_on"]
    common_identity = all(all(row["checks"].values()) for row in rows.values())
    off_clean = off["start_rc"] == 0 and not off["fatal"]
    if on["start_rc"] == 0 and not on["fatal"]:
        classification = "queue_property_not_root_cause"
        on_decisive = True
    elif on["start_rc"] != 0 and on["device_lost"] and on["mul_mat_failure"]:
        classification = "queue_property_root_cause"
        on_decisive = True
    else:
        classification = "ambiguous"
        on_decisive = False
    hard_gates = {
        "identity_and_no_barriers": common_identity,
        "profile_off_starts_cleanly": off_clean,
        "profile_on_is_decisive": on_decisive,
        "gpu_health_pre_and_post": all(health.values()),
        "code_hashes_stable": code_hashes_stable,
        "endpoint_down": endpoint.get("passed") is True,
    }
    return {
        "passed": all(hard_gates.values()),
        "classification": classification,
        "hard_gates": hard_gates,
        "health": health,
        "arms": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--write", type=Path)
    args = parser.parse_args()
    result = analyze(args.root)
    payload = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    if args.write is not None:
        args.write.write_text(payload, encoding="ascii")
    print(payload, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
