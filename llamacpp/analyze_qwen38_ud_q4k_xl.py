#!/usr/bin/env python3
"""Analyze the matched Q4_K_M, UD-Q4_K_XL, and optional XL MTP campaign."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def pct(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline in (None, 0):
        return None
    return 100.0 * (candidate / baseline - 1.0)


def profile_metrics(directory: Path) -> dict:
    profile = load_json(directory / "profile.json")
    result = {"passed": bool(profile.get("passed")), "summary": profile["summary"]}
    return result


def deterministic_metrics(directory: Path) -> dict:
    data = load_json(directory / "deterministic.json")
    results = data.get("results", [])
    return {
        "passed": bool(data.get("passed")),
        "expectation_passes": sum(bool(item.get("coherent")) for item in results),
        "total": len(results),
        "failures": [item.get("id") for item in results if not item.get("coherent")],
        "nonempty": all(bool((item.get("text") or "").strip()) for item in results),
        "reference_exact": all(item.get("exact_reference") is True for item in results)
        if data.get("reference")
        else None,
    }


def kernel_evidence(directory: Path) -> dict:
    log_paths = [directory / "server.log", directory / "evidence_server.log"]
    text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in log_paths
        if path.exists()
    )
    patterns = {
        "type_count_lines": r"- type\s+\S+:\s+\d+ tensors",
        "sycl_door_lines": r"GGML_SYCL_[A-Z0-9_]+:\s+[-0-9]+",
        "allreduce_census_lines": r"allreduce=\d+ fused_allreduce_add=\d+",
        "fusion_exit_lines": r"\[FUSE-EXT\]",
        "q8_exit_lines": r"\[Q8-DEDUP\]",
        "server_timing_lines": r"prompt eval time|eval time",
    }
    return {name: len(re.findall(pattern, text)) for name, pattern in patterns.items()}


def heplus_metrics(directory: Path) -> dict | None:
    marker = directory / "heplus_summary_path.txt"
    if not marker.exists():
        return None
    path = Path(marker.read_text(encoding="utf-8").strip())
    if not path.is_file():
        return {"error": f"missing summary {path}"}
    data = load_json(path)
    scores = data["tiers"]["1"]["pass@1"]
    return {"summary": str(path), "base": scores["base"], "plus": scores["plus"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--run-mtp", action="store_true")
    parser.add_argument("--run-heplus", action="store_true")
    parser.add_argument("--write", type=Path)
    args = parser.parse_args()

    q4_dir = args.out_dir / "q4km_mtp0"
    xl_dir = args.out_dir / "xl_mtp0"
    q4_profile = profile_metrics(q4_dir)
    xl_profile = profile_metrics(xl_dir)
    q4_det = deterministic_metrics(q4_dir)
    xl_det = deterministic_metrics(xl_dir)

    deltas = {}
    for regime in ("decode", "coding", "prefill"):
        baseline = q4_profile["summary"][regime]
        candidate = xl_profile["summary"][regime]
        deltas[regime] = {
            "post_first_tok_s_pct": pct(
                candidate["median_post_first_tok_s"], baseline["median_post_first_tok_s"]
            ),
            "ttft_pct": pct(candidate["median_ttft_s"], baseline["median_ttft_s"]),
            "prefill_proxy_tok_s_pct": pct(
                candidate["median_prefill_proxy_tok_s"],
                baseline["median_prefill_proxy_tok_s"],
            ),
        }

    hard_gates = {
        "q4_profile_complete": q4_profile["passed"],
        "xl_profile_complete": xl_profile["passed"],
        "q4_deterministic_nonempty": q4_det["nonempty"],
        "xl_deterministic_nonempty": xl_det["nonempty"],
        "xl_quality_canaries_at_least_6_of_7": (
            xl_det["total"] == 7 and xl_det["expectation_passes"] >= 6
        ),
    }

    mtp = None
    heplus = None
    if args.run_mtp:
        mtp_dir = args.out_dir / "xl_mtp1"
        mtp_profile = profile_metrics(mtp_dir)
        mtp_det = deterministic_metrics(mtp_dir)
        mtp = {
            "profile": mtp_profile,
            "deterministic": mtp_det,
            "kernel_evidence": kernel_evidence(mtp_dir),
            "decode_post_first_tok_s_pct_vs_mtp0": pct(
                mtp_profile["summary"]["decode"]["median_post_first_tok_s"],
                xl_profile["summary"]["decode"]["median_post_first_tok_s"],
            ),
            "coding_post_first_tok_s_pct_vs_mtp0": pct(
                mtp_profile["summary"]["coding"]["median_post_first_tok_s"],
                xl_profile["summary"]["coding"]["median_post_first_tok_s"],
            ),
        }
        hard_gates["mtp_profile_complete"] = mtp_profile["passed"]
        hard_gates["mtp_greedy_reference_exact"] = mtp_det["reference_exact"] is True
        heplus = heplus_metrics(mtp_dir)

    if args.run_heplus:
        hard_gates["xl_mtp3_heplus_result_present"] = bool(
            heplus and "error" not in heplus
        )
        if heplus and "error" not in heplus:
            # At most one base and two plus problems below the pinned 0.970/0.927 Q4_K_M run.
            hard_gates["xl_mtp3_heplus_quality_band"] = (
                heplus["base"] >= 0.963 and heplus["plus"] >= 0.915
            )
        if mtp is not None:
            mtp["heplus"] = heplus

    result = {
        "passed": all(hard_gates.values()),
        "hard_gates": hard_gates,
        "q4km_mtp0": {
            "profile": q4_profile,
            "deterministic": q4_det,
            "kernel_evidence": kernel_evidence(q4_dir),
        },
        "xl_mtp0": {
            "profile": xl_profile,
            "deterministic": xl_det,
            "kernel_evidence": kernel_evidence(xl_dir),
        },
        "xl_vs_q4km_pct": deltas,
        "xl_mtp1": mtp,
        "notes": {
            "quality_canary_policy": (
                "Six of seven exact deterministic checks is a coherence floor, not a full quality proof."
            ),
            "heplus_quality_band": (
                "For embedded MTP3, require base >=0.963 and plus >=0.915 versus pinned 0.970/0.927."
            ),
        },
    }
    output = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    if args.write:
        args.write.write_text(output, encoding="ascii")
    print(output, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
