#!/usr/bin/env python3
"""Summarize X2 JSON evidence without importing torch or touching a GPU."""
from collections import defaultdict
import json
from pathlib import Path
import sys


def category(name):
    if "routed_experts" in name and (
        "_parameters" in name or ".quant_method._c1_" in name
    ):
        return "routed_experts"
    if "indexer" in name:
        return "qsa_indexer"
    if "hyper_connection" in name:
        return "hyper_connections"
    if "lm_head" in name:
        return "lm_head"
    if "embed_tokens" in name:
        return "embedding"
    if "shared_expert" in name:
        return "shared_experts"
    if ".ple." in name:
        return "ple"
    if ".mlp.gate." in name:
        return "routers"
    if "linear_attn" in name or "self_attn" in name:
        return "attention_gdn"
    return "other_buffers"


def main():
    root = Path(sys.argv[1])
    result = {}
    for path in sorted(root.glob("*-weights.json")):
        pid = path.name.split("-")[0]
        data = json.loads(path.read_text())
        groups = defaultdict(int)
        owners = {}
        for ref in data["references"]:
            owners.setdefault(ref["storage"], ref["name"])
        for key, n in data["storage_bytes"].items():
            device = key.rsplit(":", 1)[0]
            groups[f"{device}/{category(owners[key])}"] += n
        result[pid] = dict(weights=dict(groups), total_owned_storage=data["unique_bytes"])
        for stage in ("budget", "kv-initialized", "devices-after-profile"):
            evidence = root / f"{pid}-{stage}.json"
            if evidence.exists():
                result[pid][stage] = json.loads(evidence.read_text())
        profile_path = root / f"{pid}-profile.json"
        if profile_path.exists():
            profile = json.loads(profile_path.read_text())
            after = profile["after_profile"]
            result[pid]["profile"] = {k: v for k, v in profile.items() if isinstance(v, (int, float))}
            result[pid]["after_profile"] = after
            slack = after["torch_memory"]-after["torch_allocated"]
            result[pid]["reserved_slack"] = slack
            if "reserved_peak_bytes" in profile:
                result[pid]["physical_peak_headroom"] = max(
                    0, profile["reserved_peak_bytes"]-after["torch_memory"])
        op_path = root / f"{pid}-operations.json"
        if op_path.exists():
            ops = json.loads(op_path.read_text())
            deltas = []
            for before, after in zip(ops, ops[1:]):
                previous_raw = before["total"]-before["free"]-before["reserved"]
                raw = after["total"]-after["free"]-after["reserved"]
                deltas.append(dict(name=after["name"], non_torch_growth=raw-previous_raw,
                                   peak_growth=after["peak"]-before["peak"]))
            result[pid]["top_non_torch_growth"] = sorted(deltas, key=lambda x: x["non_torch_growth"], reverse=True)[:12]
            result[pid]["top_peak_growth"] = sorted(deltas, key=lambda x: x["peak_growth"], reverse=True)[:12]
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
