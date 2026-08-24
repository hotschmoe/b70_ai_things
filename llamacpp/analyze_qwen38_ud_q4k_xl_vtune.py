#!/usr/bin/env python3
"""Fail-closed mechanism gate for the XL VTune GPU-offload trial."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


FAMILIES = {"q3_K", "q4_K", "q5_K", "q6_K", "q8_0", "iq3_s", "iq4_nl", "iq4_xs"}
DOMINANT = {"q5_K", "q8_0", "iq4_xs", "q4_K"}
FATAL = re.compile(
    r"device_lost|out_of_resources|ur_result_error|"
    r"(^|[^a-z])nan([^a-z]|$)|uncaught exception|segmentation fault|"
    r"data.?limit.*(reach|exceed)|incomplete finalization|"
    r"collection failed|vtune:\s*error|internal error",
    re.IGNORECASE | re.MULTILINE,
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_env(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--write", type=Path)
    args = parser.parse_args()
    out = args.out_dir
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}

    required = [
        out / "manifest.json",
        out / "endpoint_down.json",
        out / "reference" / "container_inspect.json",
        out / "reference" / "container_env.txt",
        out / "reference" / "models.json",
        out / "reference" / "warmup.json",
        out / "reference" / "measure.json",
        out / "reference" / "server.log",
        out / "vtune" / "container_inspect.json",
        out / "vtune" / "container_env.txt",
        out / "vtune" / "models.json",
        out / "vtune" / "warmup.json",
        out / "vtune" / "measure.json",
        out / "vtune" / "server.log",
        out / "vtune" / "stop_contract.json",
        out / "vtune" / "tasks.csv",
        out / "vtune" / "tasks.json",
        out / "vtune" / "summary.csv",
        out / "vtune" / "profile" / "vtune_version.txt",
        out / "vtune" / "census.json",
        out / "health_pre.log",
        out / "reference" / "health_post.log",
        out / "vtune" / "health_post.log",
    ]
    missing = [str(path.relative_to(out)) for path in required if not path.is_file()]
    checks["required_artifacts"] = not missing
    details["missing_artifacts"] = missing
    if missing:
        result = {"passed": False, "checks": checks, "details": details}
        text = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
        if args.write:
            args.write.write_text(text, encoding="ascii")
        print(text, end="")
        return 1

    manifest = load_json(out / "manifest.json")
    endpoint = load_json(out / "endpoint_down.json")
    reference = load_json(out / "reference" / "measure.json")
    traced = load_json(out / "vtune" / "measure.json")
    ref_warmup = load_json(out / "reference" / "warmup.json")
    trace_warmup = load_json(out / "vtune" / "warmup.json")
    tasks = load_json(out / "vtune" / "tasks.json")
    census = load_json(out / "vtune" / "census.json")
    stop_contract = load_json(out / "vtune" / "stop_contract.json")
    reference_env = load_env(out / "reference" / "container_env.txt")
    traced_env = load_env(out / "vtune" / "container_env.txt")

    reference_inspect = load_json(out / "reference" / "container_inspect.json")[0]
    traced_inspect = load_json(out / "vtune" / "container_inspect.json")[0]
    reference_models = load_json(out / "reference" / "models.json")
    traced_models = load_json(out / "vtune" / "models.json")
    served = manifest["config"]["served"]
    checks["served_identity"] = all(
        served in {item.get("id") for item in models.get("data", [])}
        for models in (reference_models, traced_models)
    )
    checks["image_identity"] = all(
        inspect.get("Image") == manifest["image"]["id"]
        for inspect in (reference_inspect, traced_inspect)
    )
    checks["restart_disabled"] = all(
        inspect.get("HostConfig", {}).get("RestartPolicy", {}).get("Name") == "no"
        for inspect in (reference_inspect, traced_inspect)
    )
    checks["vtune_version"] = "2025.10" in (
        out / "vtune" / "profile" / "vtune_version.txt"
    ).read_text(encoding="utf-8", errors="replace")
    collection_mode = manifest.get("config", {}).get("collection_mode", "launch_under")
    checks["known_collection_mode"] = collection_mode in {"launch_under", "attach_after_load"}
    if collection_mode == "attach_after_load":
        trace_host = traced_inspect.get("HostConfig", {})
        cap_add = trace_host.get("CapAdd") or []
        security_opt = trace_host.get("SecurityOpt") or []
        reference_cap_add = reference_inspect.get("HostConfig", {}).get("CapAdd") or []
        checks["attach_capability_scoped"] = (
            sorted(cap_add) == ["SYS_PTRACE"] and not reference_cap_add
        )
        checks["attach_security_scoped"] = (
            not bool(trace_host.get("Privileged"))
            and not bool(trace_host.get("PidMode"))
            and not any("unconfined" in item.lower() for item in security_opt)
        )
        checks["attach_mode_env"] = (
            reference_env.get("VTUNE_ATTACH_MODE") == "0"
            and traced_env.get("VTUNE_ATTACH_MODE") == "1"
        )
        checks["final_health_artifact"] = (out / "health_final.log").is_file()

    checks["endpoint_down"] = bool(endpoint.get("passed"))
    checks["requests_passed"] = all(
        bool(item.get("passed"))
        for item in (reference, traced, ref_warmup, trace_warmup)
    )
    ref_result = reference.get("result", {})
    trace_result = traced.get("result", {})
    checks["exact_512_tokens"] = (
        ref_result.get("completion_tokens") == 512
        and trace_result.get("completion_tokens") == 512
    )
    checks["deterministic_equal"] = (
        bool(ref_result.get("text_sha256"))
        and ref_result.get("text_sha256") == trace_result.get("text_sha256")
    )
    ref_speed = float(ref_result.get("post_first_tok_s") or 0)
    trace_speed = float(trace_result.get("post_first_tok_s") or 0)
    ref_ttft = float(ref_result.get("ttft_s") or 0)
    trace_ttft = float(trace_result.get("ttft_s") or 0)
    speed_ratio = trace_speed / ref_speed if ref_speed > 0 else 0
    ttft_ratio = trace_ttft / ref_ttft if ref_ttft > 0 else float("inf")
    checks["trace_speed_at_least_85pct"] = speed_ratio >= 0.85
    checks["trace_ttft_at_most_1_25x"] = ttft_ratio <= 1.25
    details["performance"] = {
        "reference_post_first_tok_s": ref_speed,
        "traced_post_first_tok_s": trace_speed,
        "speed_ratio": speed_ratio,
        "reference_ttft_s": ref_ttft,
        "traced_ttft_s": trace_ttft,
        "ttft_ratio": ttft_ratio,
    }

    expected_env = {
        "MODEL_FILE": manifest["model"]["file"],
        "MODEL_SHA256": manifest["model"]["sha256"],
        "GPU_COUNT": "2",
        "ENABLE_MTP": "0",
        "LAB_DOORS": "0",
        "CCL_TOPO_P2P_ACCESS": "0",
        "GGML_SYCL_QUANT_CENSUS": "1",
        "GGML_SYCL_QUANT_TIMING_SAMPLE": "0",
        "GGML_SYCL_PROFILE": "0",
        "GGML_SYCL_DEBUG": "0",
        "PROFILE_VERBOSE": "0",
        "PROFILE_STATS": "0",
    }
    checks["reference_env_exact"] = all(
        reference_env.get(key) == value for key, value in expected_env.items()
    ) and reference_env.get("VTUNE_GPU_OFFLOAD") == "0"
    expected_vtune_env = "0" if collection_mode == "attach_after_load" else "1"
    checks["vtune_env_exact"] = all(
        traced_env.get(key) == value for key, value in expected_env.items()
    ) and traced_env.get("VTUNE_GPU_OFFLOAD") == expected_vtune_env

    checks["vtune_stop_contract"] = bool(stop_contract.get("passed"))
    checks["task_parser_passed"] = bool(tasks.get("passed"))
    adapters = set(tasks.get("adapters", []))
    checks["exact_two_adapters"] = len(adapters) == 2
    found_families = set(tasks.get("families", []))
    checks["all_quant_families"] = FAMILIES <= found_families
    adapter_family = {
        (row.get("adapter"), row.get("family"))
        for row in tasks.get("by_adapter_family", [])
        if float(row.get("total_time_s") or 0) > 0
    }
    checks["dominant_families_on_both_adapters"] = all(
        all((adapter, family) in adapter_family for adapter in adapters)
        for family in DOMINANT
    )
    details["vtune"] = {
        "adapters": sorted(adapters),
        "families": sorted(found_families),
        "total_task_time_s": tasks.get("total_task_time_s"),
        "classified_quant_task_time_s": tasks.get("classified_quant_task_time_s"),
        "unknown_task_count": len(tasks.get("unknown_tasks", [])),
    }

    census_types = {
        row.get("type")
        for row in census.get("records", [])
        if row.get("kind") == "logical" and int(row.get("calls", 0)) > 0
    }
    checks["census_all_quant_families"] = FAMILIES <= census_types
    checks["census_nonempty"] = (
        int(census.get("computed_totals", {}).get("logical_total", 0)) > 0
        and int(census.get("computed_totals", {}).get("actual_total", 0)) > 0
    )

    scanned = []
    fatal_matches = []
    for path in out.rglob("*.log"):
        scanned.append(str(path.relative_to(out)))
        text = path.read_text(encoding="utf-8", errors="replace")
        match = FATAL.search(text)
        if match:
            fatal_matches.append(
                {"file": str(path.relative_to(out)), "marker": match.group(0)}
            )
    checks["no_fatal_markers"] = not fatal_matches
    checks["no_queue_timing_marker"] = "[QUANT-TIMING] enabled" not in (
        out / "vtune" / "server.log"
    ).read_text(encoding="utf-8", errors="replace")
    details["fatal_matches"] = fatal_matches
    details["scanned_logs"] = sorted(scanned)

    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "details": details,
        "verdict": (
            "VTune gpu-offload mechanism accepted; repeatability qualification remains"
            if all(checks.values())
            else "VTune gpu-offload mechanism rejected; do not draw timing conclusions"
        ),
    }
    text = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    if args.write:
        args.write.write_text(text, encoding="ascii")
    print(text, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
