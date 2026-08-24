#!/usr/bin/env python3
"""Fail-closed analyzer for the XL candidate quant timing GPU campaign."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


ARMS = ("current_off", "candidate_off", "counts_only", "timing_64", "timing_128", "timing_256")
TIMING_ARMS = ("timing_64", "timing_128", "timing_256")
KEY_FIELDS = ("algo", "device", "type", "reordered", "split", "width", "K", "rows")


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def profile_speeds(directory: Path) -> list[float]:
    data = load_json(directory / "decode_profile.json")
    if not data.get("passed"):
        raise ValueError(f"profile failed: {directory}")
    method = data.get("methodology", {})
    if not (
        method.get("temperature") == 0
        and method.get("ignore_eos") is True
        and int(method.get("gen_tokens", -1)) == 256
        and int(method.get("warmup_tokens", -1)) == 32
    ):
        raise ValueError(f"profile methodology mismatch: {directory}")
    warmup = data.get("warmup", {})
    runs = data.get("runs", [])
    if int(warmup.get("completion_tokens", -1)) != 32 or any(
        int(run.get("completion_tokens", -1)) != 256 for run in runs
    ):
        raise ValueError(f"profile token count mismatch: {directory}")
    speeds = [run.get("post_first_tok_s") for run in runs]
    if not speeds or any(value is None or not math.isfinite(value) or value <= 0 for value in speeds):
        raise ValueError(f"profile has invalid speeds: {directory}")
    return [float(value) for value in speeds]


def ratio_ci(candidate: list[float], baseline: list[float]) -> dict:
    if len(candidate) != len(baseline) or len(candidate) < 2:
        raise ValueError("matched inert profiles require at least two paired repetitions")
    logs = [math.log(c / b) for c, b in zip(candidate, baseline, strict=True)]
    mean = statistics.mean(logs)
    se = statistics.stdev(logs) / math.sqrt(len(logs))
    # Two-sided 90% Student-t critical values for df 1..9; n=5 uses df=4.
    t90 = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015,
           6: 1.943, 7: 1.895, 8: 1.860, 9: 1.833}.get(len(logs) - 1, 1.645)
    return {
        "paired_geomean_ratio": math.exp(mean),
        "paired_90pct_lower": math.exp(mean - t90 * se),
        "paired_90pct_upper": math.exp(mean + t90 * se),
        "median_ratio": statistics.median(candidate) / statistics.median(baseline),
    }


def record_key(record: dict) -> tuple:
    return tuple(record[field] for field in KEY_FIELDS)


def census_actual_map(directory: Path) -> tuple[dict[tuple, int], dict]:
    data = load_json(directory / "census.json")
    records = [record for record in data.get("records", []) if record.get("kind") == "actual"]
    if not records:
        raise ValueError(f"no actual census records: {directory}")
    result = {}
    for record in records:
        key = (
            record["algo"], record["device"], record["type"], record["reordered"],
            record["split"], record["width"], record["K"], record["rows"],
        )
        result[key] = result.get(key, 0) + int(record["calls"])
    return result, data


def timing_data(directory: Path) -> tuple[dict, dict[tuple, dict]]:
    data = load_json(directory / "timing.json")
    records = data.get("records", [])
    if not records:
        raise ValueError(f"no timing records: {directory}")
    return data, {record_key(record): record for record in records}


def deterministic_exact(current_dir: Path, candidate_dir: Path) -> dict:
    current = load_json(current_dir / "deterministic.json")
    candidate = load_json(candidate_dir / "deterministic.json")
    current_results = current.get("results", [])
    candidate_results = candidate.get("results", [])
    return {
        # The established XL artifact is 6/7 on this small coherence battery
        # because its modular-arithmetic answer is wrong. This campaign is an
        # instrumentation-inertness gate, not a model-quality promotion gate.
        "current_nonempty": all(item.get("text", "").strip() for item in current_results),
        "current_coherent_at_least_6_of_7": sum(
            item.get("coherent") is True for item in current_results
        ) >= 6,
        "candidate_passed": bool(candidate.get("passed")),
        "seven_cases": len(current_results) == 7 and len(candidate_results) == 7,
        "candidate_reference_exact": all(
            result.get("exact_reference") is True for result in candidate_results
        ),
    }


def analyze(out_dir: Path) -> dict:
    gates = {}
    identities = {arm: load_json(out_dir / arm / "identity.json") for arm in ARMS}
    stops = {arm: load_json(out_dir / arm / "graceful_stop.json") for arm in ARMS}
    gates["all_arm_identities"] = all(item.get("passed") is True for item in identities.values())
    gates["all_graceful_stops"] = all(item.get("passed") is True for item in stops.values())
    image_ids = {arm: identities[arm].get("actual", {}).get("image_id") for arm in ARMS}
    candidate_ids = {image_ids[arm] for arm in ARMS if arm != "current_off"}
    gates["candidate_image_constant"] = len(candidate_ids) == 1 and None not in candidate_ids
    gates["candidate_distinct_from_current"] = (
        bool(image_ids["current_off"])
        and gates["candidate_image_constant"]
        and image_ids["current_off"] not in candidate_ids
    )
    endpoint = load_json(out_dir / "endpoint_down.json")
    gates["endpoint_left_down"] = endpoint.get("passed") is True

    deterministic = deterministic_exact(out_dir / "current_off", out_dir / "candidate_off")
    gates.update({f"deterministic_{key}": value for key, value in deterministic.items()})

    current_speeds = profile_speeds(out_dir / "current_off")
    candidate_speeds = profile_speeds(out_dir / "candidate_off")
    gates["inert_profiles_are_matched_5x256"] = (
        len(current_speeds) == 5 and len(candidate_speeds) == 5
    )
    inert = ratio_ci(candidate_speeds, current_speeds)
    gates["candidate_inert_median_delta_le_2pct"] = 0.98 <= inert["median_ratio"] <= 1.02
    gates["candidate_inert_paired_90pct_interval_reasonable"] = (
        inert["paired_90pct_lower"] >= 0.95 and inert["paired_90pct_upper"] <= 1.05
    )

    counts_map, counts_payload = census_actual_map(out_dir / "counts_only")
    gates["counts_actual_cells_nonempty"] = bool(counts_map)
    gates["counts_actual_cells_both_devices"] = {key[1] for key in counts_map} == {0, 1}
    counts_totals = counts_payload.get("computed_totals", {})
    counts_ends = counts_payload.get("declared_ends", [])
    gates["counts_census_complete"] = (
        counts_payload.get("versions") == ["1"]
        and len(counts_ends) == 1
        and int(counts_ends[0].get("actual_total", -1))
        == int(counts_totals.get("actual_total", -2))
        and int(counts_totals.get("actual_total", 0)) > 0
    )

    timing_payloads = {}
    timing_maps = {}
    timing_census_maps = {}
    for arm in TIMING_ARMS:
        payload, records = timing_data(out_dir / arm)
        timing_payloads[arm] = payload
        timing_maps[arm] = records
        timing_census_maps[arm], arm_census = census_actual_map(out_dir / arm)
        summary = payload["summary"]
        headers = payload.get("headers", [])
        expected_period = int(arm.rsplit("_", 1)[1])
        dropped = sum(int(header.get("dropped", 0)) for header in headers)
        sampled_devices = {
            record["device"] for record in records.values() if int(record.get("samples", 0)) > 0
        }
        gates[f"{arm}_both_devices_sampled"] = sampled_devices == {0, 1}
        gates[f"{arm}_timestamps_valid"] = (
            int(summary.get("incomplete", -1)) == 0 and int(summary.get("invalid", -1)) == 0
        )
        gates[f"{arm}_no_dropped_samples"] = dropped == 0
        gates[f"{arm}_header_exact"] = (
            len(headers) == 1
            and headers[0].get("scope") == "standard_mul_mat"
            and int(headers[0].get("sample_period", -1)) == expected_period
        )
        calls_seen = {key: int(record["calls_seen"]) for key, record in records.items()}
        declared = payload.get("declared_ends", [])
        gates[f"{arm}_timing_dump_complete"] = (
            len(declared) == 1
            and int(declared[0].get("samples", -1)) == int(summary.get("samples", -2))
            and int(declared[0].get("device_ns", -1))
            == int(summary.get("sampled_device_ns", -2))
            and int(summary.get("samples", 0)) > 0
        )
        arm_totals = arm_census.get("computed_totals", {})
        arm_ends = arm_census.get("declared_ends", [])
        gates[f"{arm}_census_dump_complete"] = (
            arm_census.get("versions") == ["1"]
            and len(arm_ends) == 1
            and int(arm_ends[0].get("actual_total", -1))
            == int(arm_totals.get("actual_total", -2))
        )
        gates[f"{arm}_timing_calls_match_own_census"] = calls_seen == timing_census_maps[arm]
        gates[f"{arm}_request_counts_match_counts_only"] = timing_census_maps[arm] == counts_map

    evidence_speeds = {
        arm: profile_speeds(out_dir / arm) for arm in ("counts_only",) + TIMING_ARMS
    }
    if any(len(speeds) != 1 for speeds in evidence_speeds.values()):
        raise ValueError("counts and timing evidence arms require exactly one measured request")
    timing_128_speed = evidence_speeds["timing_128"]
    counts_speed = evidence_speeds["counts_only"]
    perturbation_ratio = timing_128_speed[0] / counts_speed[0]
    gates["timing_128_throughput_perturbation_le_3pct"] = (
        abs(perturbation_ratio - 1.0) <= 0.03
    )

    ranked = {
        arm: sorted(
            records,
            key=lambda key: float(timing_maps[arm][key].get("projected_device_ns") or 0),
            reverse=True,
        )
        for arm, records in timing_maps.items()
    }
    top128 = ranked["timing_128"][:5]
    top_overlap = {
        arm: len(set(top128) & set(ranked[arm][:5])) / max(len(top128), 1)
        for arm in ("timing_64", "timing_256")
    }
    gates["dominant_top5_overlap_ge_80pct"] = all(value >= 0.8 for value in top_overlap.values())

    dominant = []
    cumulative = 0.0
    for key in ranked["timing_128"]:
        dominant.append(key)
        cumulative += float(timing_maps["timing_128"][key].get("projected_share") or 0)
        if cumulative >= 0.8 and len(dominant) >= 3:
            break
        if len(dominant) >= 10:
            break
    dominant_deltas = {}
    dominant_ok = bool(dominant)
    for key in dominant:
        base_mean = timing_maps["timing_128"][key].get("mean_ns")
        key_name = "|".join(str(value) for value in key)
        dominant_deltas[key_name] = {}
        if base_mean in (None, 0):
            dominant_ok = False
            continue
        for arm in ("timing_64", "timing_256"):
            other = timing_maps[arm].get(key)
            if other is None or other.get("mean_ns") in (None, 0):
                dominant_ok = False
                dominant_deltas[key_name][arm] = None
                continue
            delta = abs(float(other["mean_ns"]) / float(base_mean) - 1.0)
            dominant_deltas[key_name][arm] = delta
            dominant_ok = dominant_ok and delta <= 0.15
    gates["dominant_mean_time_stable_within_15pct"] = dominant_ok

    timing_128_ranked = []
    rollup = {}
    for key in ranked["timing_128"]:
        row = timing_maps["timing_128"][key]
        timing_128_ranked.append({
            **{field: row[field] for field in KEY_FIELDS},
            "calls_seen": int(row["calls_seen"]),
            "samples": int(row["samples"]),
            "mean_ns": row.get("mean_ns"),
            "projected_device_ns": row.get("projected_device_ns"),
            "projected_share": row.get("projected_share"),
        })
        group = f"{row['algo']}|{row['type']}"
        rollup[group] = rollup.get(group, 0.0) + float(row.get("projected_device_ns") or 0)
    rollup_total = sum(rollup.values())
    timing_128_rollup = [
        {
            "algo_type": group,
            "projected_device_ns": value,
            "projected_share": value / rollup_total if rollup_total else None,
        }
        for group, value in sorted(rollup.items(), key=lambda item: item[1], reverse=True)
    ]

    return {
        "passed": all(gates.values()),
        "hard_gates": gates,
        "inert": inert,
        "timing_128_perturbation_ratio": perturbation_ratio,
        "top5_overlap": top_overlap,
        "dominant_keys": ["|".join(str(value) for value in key) for key in dominant],
        "dominant_mean_abs_deltas": dominant_deltas,
        "timing_128_ranked_cells": timing_128_ranked,
        "timing_128_algo_type_rollup": timing_128_rollup,
        "counts_actual_cells": len(counts_map),
        "image_ids": image_ids,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--write", type=Path)
    args = parser.parse_args()
    try:
        result = analyze(args.out_dir)
    except Exception as exc:  # noqa: BLE001 - missing/malformed evidence is a failed gate
        result = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
    output = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    if args.write:
        args.write.write_text(output, encoding="ascii")
    print(output, end="")
    return 0 if result.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
