#!/usr/bin/env python3
"""Off-device guard for June-compatible no-spec PIECEWISE capture keys."""

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace

from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


CAPTURE_SIZES = [1, 2, 4, 8, 16, 24, 32, 40, 48]
MAX_NUM_SEQS = 24


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # The June e190923b dispatcher had no ordinary-decode-specific PIECEWISE
    # key. Ordinary decode reused the relaxed non-uniform key. August added a
    # speculative-decode-only helper; it must continue to return None when no
    # speculative model is configured.
    owner = SimpleNamespace(
        vllm_config=SimpleNamespace(speculative_config=None),
        uniform_decode_query_len=1,
        _bs_to_padded_graph_size=list(range(max(CAPTURE_SIZES) + 1)),
    )
    method = CudagraphDispatcher._create_piecewise_uniform_batch_descriptor
    results = {}
    for num_tokens in CAPTURE_SIZES:
        descriptor = method(owner, num_tokens, False, 0)
        assert descriptor is None, (
            "ordinary decode gained an August uniform PIECEWISE key at "
            f"capture size {num_tokens}"
        )

        # Mirror gpu_model_runner._dummy_run's general, non-uniform branch.
        num_reqs = min(num_tokens, MAX_NUM_SEQS)
        min_tokens_per_req = num_tokens // num_reqs
        schedule = [min_tokens_per_req] * num_reqs
        schedule[-1] += num_tokens % num_reqs
        assert len(schedule) == num_reqs
        assert sum(schedule) == num_tokens
        results[str(num_tokens)] = {
            "descriptor": None,
            "num_reqs": num_reqs,
            "scheduled_tokens": schedule,
        }

    # This is the exact failure boundary caught after reboot: a uniform
    # one-token schedule cannot represent sizes above max_num_seqs.
    rejected_uniform_sizes = [
        size
        for size in CAPTURE_SIZES
        if sum([1] * min(size, MAX_NUM_SEQS)) != size
    ]
    assert rejected_uniform_sizes == [32, 40, 48]

    # June disabled only non-uniform prefill replay. It still captured these
    # general descriptors because ordinary decode reused them. August later
    # made the same setting suppress capture as well. sitecustomize must
    # restore June's capture semantics without consuming the setting needed by
    # runtime dispatch.
    assert getattr(
        GPUModelRunner._xpu_filter_cudagraph_capture_descs,
        "_qwen36_june_contract",
        False,
    )
    fake_runner = SimpleNamespace(
        device=SimpleNamespace(type="xpu"),
        num_spec_tokens=0,
    )
    sentinel_capture_descs = [("PIECEWISE", [object() for _ in CAPTURE_SIZES])]
    setting = "VLLM_XPU_DISABLE_PREFILL_CUDAGRAPH_REPLAY"
    previous = os.environ.get(setting)
    os.environ[setting] = "1"
    try:
        retained = GPUModelRunner._xpu_filter_cudagraph_capture_descs(
            fake_runner, sentinel_capture_descs
        )
        assert retained is sentinel_capture_descs
        assert os.environ.get(setting) == "1"
    finally:
        if previous is None:
            os.environ.pop(setting, None)
        else:
            os.environ[setting] = previous

    record = {
        "protocol": "qwen36-june-piecewise-capture-contract-v2",
        "source_module": method.__module__,
        "source_file": inspect.getsourcefile(method),
        "max_num_seqs": MAX_NUM_SEQS,
        "capture_sizes": CAPTURE_SIZES,
        "ordinary_decode_specific_descriptors": 0,
        "general_piecewise_capture_descriptors_retained": len(CAPTURE_SIZES),
        "prefill_replay_setting_preserved": True,
        "rejected_uniform_sizes": rejected_uniform_sizes,
        "general_piecewise_schedules": results,
        "verdict": "pass",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
