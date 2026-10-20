#!/usr/bin/env python3
"""Header-only Flash-Next weight/KV accounting. Never reads tensor payloads."""
import argparse
from collections import defaultdict
import json
import math
import sys


def category(name):
    if name == "per_layer_token_embd.weight":
        return "ple_host_table"
    if "_exps.weight" in name:
        return "routed_experts"
    if "_shexp.weight" in name:
        return "shared_experts"
    if name == "token_embd.weight":
        return "embedding"
    if name == "output.weight":
        return "lm_head"
    if "indexer." in name:
        return "qsa_indexer"
    if "ple_" in name:
        return "ple_projections"
    if "hc_" in name:
        return "hyper_connections"
    if "ffn_gate_inp" in name:
        return "router"
    return "attention_gdn_norms"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("shards", nargs="+")
    p.add_argument("--gguf-py", default="/mnt/vm_8tb/b70/build/strata-ab/llama-pinned/gguf-py")
    args = p.parse_args()
    sys.path.insert(0, args.gguf_py)
    import gguf

    totals = defaultdict(int)
    counts = defaultdict(int)
    tensors = []
    dq = {}
    full_dense_duplicates = 0
    merged_dense_placeholders = 0
    for path in args.shards:
        reader = gguf.GGUFReader(path, "r")
        for t in reader.tensors:
            name, n = t.name, int(t.n_bytes)
            shape = [int(x) for x in t.shape]
            cat = category(name)
            totals[cat] += n
            counts[cat] += 1
            tensors.append(dict(name=name, category=cat, qtype=t.tensor_type.name,
                                shape=shape, bytes=n))
            if len(shape) == 2 and int(t.tensor_type) not in (0, 1, 30) and cat not in ("embedding", "ple_host_table"):
                # dense_dq's old cache retains one allocation per type/shape.
                dq[f"{int(t.tensor_type)}:{shape[1]}:{shape[0]}"] = math.prod(shape) * 2
            if int(t.tensor_type) == 30 and len(shape) == 2:
                if "indexer." in name or "ssm_alpha" in name or "ssm_beta" in name:
                    merged_dense_placeholders += n
                elif "ffn_gate_inp" in name:
                    full_dense_duplicates += n
            # HC and PLE projection mats are plain _WeightContainer parameters,
            # not B70GGUFLinearMethod shards, so they have no second alias.
            if "ffn_gate_inp_shexp.weight" in name:
                full_dense_duplicates += math.prod(shape) * 2
            if "ssm_conv1d.weight" in name:
                full_dense_duplicates += math.prod(shape) * 2
    device = sum(v for k, v in totals.items() if k != "ple_host_table")
    experts = totals["routed_experts"]
    trunk = device - experts
    gdn_state = 36 * (48 * 128 * 128 * 4 + 10240 * 3 * 2)
    result = dict(categories=dict(sorted(totals.items())), counts=dict(sorted(counts.items())),
                  shard_device_bytes=device, trunk_bytes=trunk,
                  dp2_weight_bytes_per_card=experts // 2 + trunk,
                  ep_weight_bytes=[experts * 138 // 512 + trunk, experts * 374 // 512],
                  # GateLinear/conv and HC/PLE containers bypass the GGUF
                  # linear method in the measured Flash-Next XPU model.
                  old_full_dense_duplicate_bytes=0,
                  full_dense_if_quantized_upper_bound_bytes=full_dense_duplicates,
                  old_merged_dense_placeholder_bytes=merged_dense_placeholders,
                  dense_dq_cache_bytes=sum(dq.values()),
                  dense_dq_shared_max_bytes=max(dq.values()), dense_dq_keys=dq,
                  kv_bytes_per_token=dict(bf16=12*2*2*256*2,
                                          fp8=12*2*2*256,
                                          strata_int8=12*2*2*(256+4*2),
                                          qsa_indexer_bf16=12*128*2//4),
                  strata_indexer_fp32_bytes_per_token=12*128*4//4,
                  gdn_fixed_bytes_per_slot=gdn_state, tensors=tensors)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
