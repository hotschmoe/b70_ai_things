"""Compare Flash-Next CPU references to the GGUF and to the vLLM fork.

Stages:
  self   pure-numpy properties (no weights)
  gguf   header, norm means, IQ4_NL rows, HC mix, PLE block on a fixed prompt
  fork   the same formulas through the XPU modules, if they import on CPU

Writes flashnext_ref/last_compare.json and a text config skeleton.
CPU only. Heavy stages belong under bin/cpu-run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

from flashnext_ref.geometry import FLASHNEXT, artifact_ple_consts
from flashnext_ref.hc import gr_read, gr_write
from flashnext_ref.mtp import fc_gguf_eh, fc_wide
from flashnext_ref.ple import (
    iq4nl_dequant_row,
    ngram_rows,
    ple_block,
    strata_sequence,
    xpu_sequence,
)
from flashnext_ref.qsa import (
    attend,
    gate_apply,
    indexer_scores,
    pool_indexer_keys,
    rope_neox,
    selection_width,
    topk_ids,
)

G = FLASHNEXT
HERE = Path(__file__).resolve().parent
# The pack dir is a llama.cpp tokenizer dump (no HF tokenizer.json).
# The HF tree matches the daily-driver chat template: the math prompt is 25 tokens.
TOKENIZER = os.environ.get(
    "FLASHNEXT_TOKENIZER",
    "/mnt/vm_8tb/b70/build/backend-decision/hf-flashnext",
)
PROMPTS = (
    "Return only the integer: 17 * 19.",
    "What is the capital of France? Reply only with the city name.",
)
MASK64 = (1 << 64) - 1


def _stats(a: np.ndarray, b: np.ndarray | None = None) -> dict:
    x = np.asarray(a, dtype=np.float64)
    if b is None:
        if not x.size:
            return {"mean": None, "min": None, "max": None, "std": None}
        return {
            "mean": float(x.mean()),
            "min": float(x.min()),
            "max": float(x.max()),
            "std": float(x.std()),
        }
    d = np.abs(x - np.asarray(b, dtype=np.float64))
    if not d.size:
        return {"max_abs": None, "mean_abs": None}
    return {"max_abs": float(d.max()), "mean_abs": float(d.mean())}


def _row_diff(a: np.ndarray, b: np.ndarray) -> dict:
    aa = np.asarray(a)
    bb = np.asarray(b)
    if aa.shape != bb.shape:
        return {"shape_a": list(aa.shape), "shape_b": list(bb.shape), "equal": False}
    neq = aa != bb
    n = int(neq.sum())
    first = None
    if n:
        idx = np.argwhere(neq)[0]
        first = [int(v) for v in idx]
    return {"n": int(aa.size), "n_diff": n, "equal": n == 0, "first": first}


def _signed_mod(mixed_u64: int, vocab: int) -> int:
    """torch.remainder on an int64 reinterpretation of a uint64 hash."""
    signed = mixed_u64 - (1 << 64) if mixed_u64 >= (1 << 63) else mixed_u64
    return int(signed % vocab)


def nvidia_i64_sequence(tokens: list[int], consts, eos: int) -> np.ndarray:
    """Same window as xpu_sequence, but the product is signed-int64 then remainder.

    This is the CPU path in nvidia/ngram_embedding.py (torch.bitwise_xor of
    int64 products, then torch.remainder). It is not the CUDA kernel.
    """
    from flashnext_ref.ple import mul_u64, ngram_mixed

    prev1 = eos
    prev2 = eos
    rows = np.empty((len(tokens), G.ple_n_heads), dtype=np.uint32)
    for i, c0 in enumerate(tokens):
        c1 = prev1
        c2 = eos if prev1 == eos else prev2
        ctx = [int(c0), int(c1), int(c2)]
        for n, base_n in ((2, 0), (3, G.heads_per_ngram)):
            mixed = ngram_mixed(ctx, consts.mult, n)
            for g in range(G.heads_per_ngram):
                h = base_n + g
                rows[i, h] = (_signed_mod(mixed, int(consts.vocab[h])) + int(consts.offset[h])) & 0xFFFFFFFF
        prev2 = prev1
        prev1 = int(c0)
    # mul_u64 is the uint64 product. The signed path above reinterprets the
    # already-wrapped mixed value, which is what int64 multiply produces.
    del mul_u64
    return rows


def stage_self() -> dict:
    consts = artifact_ple_consts()
    rng = np.random.default_rng(0)
    out: dict = {"ple_eos_constant": G.ple_eos, "generation_eos": G.eos}

    fresh = [1, 0, 7, G.ple_eos, 20]
    strata_rows = strata_sequence(fresh, consts)
    xpu_rows = xpu_sequence(fresh, consts, eos=G.ple_eos)
    xpu_gen_eos = xpu_sequence(fresh, consts, eos=G.eos)
    out["fresh_strata_vs_xpu_ple_eos"] = _row_diff(strata_rows, xpu_rows)
    out["fresh_xpu_ple_eos_vs_generation_eos"] = _row_diff(xpu_rows, xpu_gen_eos)

    # Latent: a stored predecessor of -1. XPU multiplies it; Strata rewrites
    # that slot and every older slot to PLE_EOS before the multiply.
    prev = np.array([[-1, 9]], dtype=np.int64)
    strata_null = ngram_rows(np.array([3]), prev, consts)
    # xpu with has_init and prev_newest=9, prev_older=-1
    xpu_null = xpu_sequence([3], consts, has_init=True, prev_newest=9, prev_older=-1)
    out["token_null_predecessor"] = _row_diff(strata_null, xpu_null)

    # Token 0 is a real id. Treating it as NULL must change the rows.
    real0 = ngram_rows(np.array([5]), np.array([[0, 4]]), consts)
    as_null = ngram_rows(np.array([5]), np.array([[G.token_null, 4]]), consts)
    out["token_0_is_not_null"] = _row_diff(real0, as_null)

    # Window direction: oldest-first vs newest-first on a 2-token history.
    oldest_first = ngram_rows(np.array([11]), np.array([[2, 8]]), consts)
    newest_first = ngram_rows(np.array([11]), np.array([[8, 2]]), consts)
    out["window_direction_differs"] = _row_diff(oldest_first, newest_first)

    width_short = selection_width(8)
    width_long = selection_width(200_000)
    out["selection_width"] = {"n_kv_8": width_short, "n_kv_200k": width_long}

    # Ties break toward the smaller index, and the returned ids are sorted.
    scores = np.array([1.0, 5.0, 5.0, 0.0, 5.0], dtype=np.float32)
    chosen = topk_ids(scores, 3)
    out["topk_ties"] = {"chosen": chosen.tolist(), "expect": [1, 2, 4]}

    # Relu per head is not relu of the sum. One positive and one negative dot.
    q = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
    keys = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    # n_bid = 1 (keys has the spare slot too? indexer_scores uses keys as pooled
    # including spare at the end). Use n_kv=1, block=4 so only cell 0, n_bid from
    # keys.shape[0]-1. Pass pooled of 2 rows and n_kv=1, n_bid=1.
    per_head = indexer_scores(keys, q, n_kv=1, n_bid=1, block=4)
    dot_sum = (q @ keys.T).sum(axis=0)
    relu_of_sum = np.maximum(dot_sum, 0.0)
    out["relu_per_head"] = {
        "per_head_sum": float(per_head[0]),
        "relu_of_sum": float(relu_of_sum[0]),
        "differ": bool(per_head[0] != relu_of_sum[0]),
    }

    # Tail bonus: n_kv=5, block=4 -> n_bid=1, cell 4 is the incomplete tail.
    raw = rng.normal(size=(5, 8)).astype(np.float32)
    gamma = np.ones(8, dtype=np.float32)
    pooled, n_bid = pool_indexer_keys(raw, gamma, block=4, rotary_dim=4)
    q_idx = rng.normal(size=(2, 8)).astype(np.float32)
    sc = indexer_scores(pooled, q_idx, n_kv=5, n_bid=n_bid, block=4)
    out["tail_bonus"] = {
        "n_bid": int(n_bid),
        "score0": float(sc[0]),
        "score4": float(sc[4]),
        "tail_is_huge": bool(sc[4] > 1e8),
    }
    ident = rope_neox(raw[:1], np.array([0]), rotary_dim=4)
    out["rope_pos0_identity"] = _stats(ident, raw[:1])

    # GQA integer division: head 12 of 24 maps to kv head 1, not head 12 % 2.
    n_q, dim, n_ids = 24, 4, 3
    qq = rng.normal(size=(n_q, dim)).astype(np.float32)
    kk = rng.normal(size=(n_ids, 2, dim)).astype(np.float32)
    vv = np.zeros((n_ids, 2, dim), dtype=np.float32)
    vv[:, 0, :] = 1.0
    vv[:, 1, :] = -1.0
    attn, _w = attend(qq, kk, vv)
    out["gqa_split"] = {
        "head0_mean": float(attn[0].mean()),
        "head12_mean": float(attn[12].mean()),
        "head0_uses_kv0": bool(attn[0].mean() > 0),
        "head12_uses_kv1": bool(attn[12].mean() < 0),
    }

    q_full = np.concatenate([qq, np.zeros_like(qq)], axis=-1)
    q_full[0, dim:] = 0.0  # sigmoid(0) = 0.5
    gated = gate_apply(attn, q_full)
    out["gate_half"] = {
        "max_abs_vs_half_attn": float(np.max(np.abs(gated[0] - 0.5 * attn[0])))
    }

    # Below the selection cap, top-k of a strictly increasing score is the
    # identity (every cell). Above it, dense attention is a different set.
    rising = np.arange(6, dtype=np.float32)
    out["identity_below_cap"] = {
        "chosen": topk_ids(rising, selection_width(6, top_k=4, block=4)).tolist()
    }

    residual = rng.normal(size=(2, G.hc, G.n_embd)).astype(np.float32)
    block = rng.normal(size=(2, G.n_embd)).astype(np.float32)
    inject0 = np.zeros((2, G.hc), dtype=np.float32)
    written = gr_write(residual, block, inject0)
    expect = residual + block[:, None, :]
    out["zero_inject_is_residual_add"] = _stats(written, expect)

    # MTP layouts disagree on a random wide residual.
    embed = rng.normal(size=(2, G.n_embd)).astype(np.float32)
    wide = rng.normal(size=(2, G.hc, G.n_embd)).astype(np.float32)
    ge = np.ones(G.n_embd, dtype=np.float32)
    gh = np.ones(G.hc * G.n_embd, dtype=np.float32)
    w_emb = rng.normal(size=(G.n_embd, G.n_embd)).astype(np.float32) * 0.01
    w_hid = rng.normal(size=(G.n_embd, G.n_embd)).astype(np.float32) * 0.01
    # eh_weight (H, 2H) is not the concatenation of the two square mats.
    eh = rng.normal(size=(G.n_embd, 2 * G.n_embd)).astype(np.float32) * 0.01
    wide_out = fc_wide(embed, wide, ge, gh, w_emb, w_hid, per_stream_norm=False)
    stream_out = fc_wide(embed, wide, ge, gh.reshape(G.hc, G.n_embd), w_emb, w_hid, per_stream_norm=True)
    eh_out = fc_gguf_eh(embed, wide, ge, gh, eh)
    out["mtp_layouts"] = {
        "wide_vs_per_stream": _stats(wide_out, stream_out),
        "wide_vs_eh_proj": _stats(wide_out, eh_out),
        "wide_shape": list(wide_out.shape),
        "eh_streams_identical": bool(np.allclose(eh_out[:, 0], eh_out[:, 1])),
        "wide_streams_identical": bool(np.allclose(wide_out[:, 0], wide_out[:, 1])),
    }
    return out


def _tokenize() -> dict:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)
    prompts = {}
    for text in PROMPTS:
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        encoded = tok.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if hasattr(encoded, "input_ids"):
            ids = list(encoded["input_ids"])
        else:
            ids = list(encoded)
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        prompts[text] = {
            "n_tokens": len(ids),
            "ids": [int(v) for v in ids],
            "rendered_prefix": rendered[:180],
            "tokenizer": TOKENIZER,
        }
    return prompts


def _load_named(index: dict, name: str) -> np.ndarray:
    from flashnext_ref.gguf_io import to_f32

    if name not in index:
        raise KeyError(name)
    return to_f32(index[name])


def stage_gguf(report: dict) -> dict:
    from flashnext_ref import gguf_io

    out: dict = {"shard1": gguf_io.SHARD1, "shard2": gguf_io.SHARD2}
    reader = gguf_io.open_reader(gguf_io.SHARD1)
    index = gguf_io.index_tensors(reader)
    # tokenizer.ggml.tokens is the whole vocab. Do not materialise it.
    skip_huge = ("tokenizer.ggml.tokens", "tokenizer.ggml.merges", "tokenizer.ggml.scores")
    keys = [
        k
        for k in reader.fields
        if (k.startswith("qwen4exp.") or k.startswith("tokenizer.ggml.")) and k not in skip_huge
    ]
    meta = {}
    for key in keys:
        value = gguf_io.field_value(reader, key)
        if isinstance(value, str) and len(value) > 200:
            meta[key] = value[:200]
            continue
        if isinstance(value, list) and len(value) > 32:
            meta[key] = {"len": len(value), "head": [int(v) for v in value[:8]]}
        else:
            meta[key] = value
    out["metadata"] = meta

    consts = artifact_ple_consts()
    mult = gguf_io.field_ints(reader, "qwen4exp.ple.layer_multipliers")
    vocab = gguf_io.field_ints(reader, "qwen4exp.ple.head_vocab_sizes")
    offset = gguf_io.field_ints(reader, "qwen4exp.ple.head_offsets")
    layers = gguf_io.field_ints(reader, "qwen4exp.ple.layers")
    out["ple_header"] = {
        "mult_match": mult == list(consts.mult),
        "vocab_match": vocab == list(consts.vocab),
        "offset_match": offset == list(consts.offset),
        "layers": layers,
        "mult": mult,
        "vocab_sum": None if vocab is None else int(sum(vocab)),
        "offset_is_prefix": None
        if vocab is None or offset is None
        else offset == [sum(vocab[:i]) for i in range(len(vocab))],
    }
    if not (mult and vocab and offset):
        out["error"] = "PLE hash metadata missing"
        return out
    from flashnext_ref.geometry import PleConsts

    live = PleConsts(mult=tuple(mult), vocab=tuple(vocab), offset=tuple(offset))

    prompts = report.get("prompts") or {}
    math_ids = prompts.get(PROMPTS[0], {}).get("ids")
    if not math_ids:
        out["prompt_hash"] = "no token ids"
    else:
        strata_rows = strata_sequence(math_ids, live, eos=G.ple_eos)
        xpu_rows = xpu_sequence(math_ids, live, eos=G.ple_eos)
        xpu_gen = xpu_sequence(math_ids, live, eos=G.eos)
        nvidia_rows = nvidia_i64_sequence(math_ids, live, eos=G.ple_eos)
        out["prompt_hash"] = {
            "n_tokens": len(math_ids),
            "strata_vs_xpu_248044": _row_diff(strata_rows, xpu_rows),
            "xpu_248044_vs_248046": _row_diff(xpu_rows, xpu_gen),
            "strata_vs_nvidia_i64_remainder": _row_diff(strata_rows, nvidia_rows),
            "first_token_rows": strata_rows[0].tolist(),
        }
        report["_strata_rows"] = strata_rows

    # Norm means decide the (1+w) question.
    norm_names = [
        "blk.0.hc_attn_norm.weight",
        "blk.0.hc_ffn_norm.weight",
        "blk.1.hc_attn_norm.weight",
        "blk.1.ple_norm_key.weight",
        "blk.1.ple_norm_query.weight",
        "blk.1.ple_norm_conv.weight",
        "blk.2.hc_attn_norm.weight",
        "output_norm.weight",
        "output_hc_norm.weight",
    ]
    norms = {}
    for name in norm_names:
        if name not in index:
            norms[name] = None
            continue
        weight = _load_named(index, name)
        folded = 1.0 + (weight - 1.0)
        norms[name] = {
            "meta": _meta(index[name]),
            "values": _stats(weight),
            "roundtrip_1_plus_w_minus_1_max_abs": float(np.max(np.abs(folded - weight))),
            "closer_to_one_than_zero": bool(abs(weight.mean() - 1.0) < abs(weight.mean())),
        }
    out["norms"] = norms

    # HC mix on real layer-0 BF16 weights, fp32 activations.
    rng = np.random.default_rng(1)
    residual = rng.normal(size=(2, G.hc, G.n_embd)).astype(np.float32)
    down = _load_named(index, "blk.0.hc_attn_down.weight")
    up = _load_named(index, "blk.0.hc_attn_up.weight")
    inject = _load_named(index, "blk.0.hc_attn_inject.weight")
    gamma = _load_named(index, "blk.0.hc_attn_norm.weight")
    mixed, inj = gr_read(residual, gamma, down, up, inject)
    mixed_dev, _ = gr_read(residual, 1.0 + gamma, down, up, inject)
    out["hc_layer0"] = {
        "down": _meta(index["blk.0.hc_attn_down.weight"]),
        "up": _meta(index["blk.0.hc_attn_up.weight"]),
        "inject": _meta(index["blk.0.hc_attn_inject.weight"]),
        "numpy_shapes": {
            "down": list(down.shape),
            "up": list(up.shape),
            "inject": list(inject.shape),
            "gamma": list(gamma.shape),
        },
        "mixed_mean": float(mixed.mean()),
        "inject_mean": None if inj is None else float(inj.mean()),
        "stored_vs_one_plus_stored_max_abs": float(np.max(np.abs(mixed - mixed_dev))),
    }
    report["_hc"] = {
        "residual": residual,
        "down": down,
        "up": up,
        "inject": inject,
        "gamma": gamma,
        "mixed": mixed,
        "inj": inj,
    }

    # Shapes that fix the head counts. Do not dequant the expert pools.
    shape_names = [
        "blk.0.attn_qkv.weight",
        "blk.0.attn_gate.weight",
        "blk.3.attn_q.weight",
        "blk.3.attn_k.weight",
        "blk.3.attn_v.weight",
        "blk.3.attn_output.weight",
    ]
    shapes = {}
    for name in shape_names:
        shapes[name] = None if name not in index else _meta(index[name])
    indexer = sorted(n for n in index if n.startswith("blk.3.") and "indexer" in n)
    shapes["blk.3.indexer"] = [_meta(index[n]) for n in indexer]
    router = "blk.0.ffn_gate_inp.weight"
    if router not in index:
        router = next((n for n in index if n.startswith("blk.0.") and "gate_inp" in n), None)
    shapes["router"] = None if router is None else _meta(index[router])
    out["shapes"] = shapes

    # PLE table rows for the first prompt token, compared to gguf-py.
    if "_strata_rows" not in report:
        out["iq4"] = "skipped, no prompt rows"
        return out
    table_reader = gguf_io.open_reader(gguf_io.SHARD2)
    table_index = gguf_io.index_tensors(table_reader)
    table_name = next(
        (n for n in table_index if "per_layer_token_embd" in n or n.endswith("ple.weight")),
        None,
    )
    if table_name is None:
        # The table tensor name varies. Take the only IQ4_NL tensor if it is unique.
        from gguf import GGMLQuantizationType

        iq4 = [n for n, t in table_index.items() if t.tensor_type == GGMLQuantizationType.IQ4_NL]
        table_name = iq4[0] if len(iq4) == 1 else None
        out["iq4_candidates"] = iq4[:8]
    if table_name is None:
        out["iq4"] = "PLE table tensor not found"
        return out
    table = table_index[table_name]
    rows = report["_strata_rows"][0]
    from gguf import GGMLQuantizationType
    from gguf.quants import dequantize

    ours = []
    theirs = []
    raw_rows = []
    for row in rows.tolist():
        raw = gguf_io.iq4_row_bytes(table, int(row))
        raw_rows.append(raw)
        ours.append(iq4nl_dequant_row(raw))
        packed = np.ascontiguousarray(raw).reshape(1, -1)
        dense = dequantize(packed, GGMLQuantizationType.IQ4_NL)
        theirs.append(np.asarray(dense, dtype=np.float32).reshape(-1))
    ours_a = np.stack(ours)
    theirs_a = np.stack(theirs)
    bit_mismatch = int(np.sum(ours_a.view(np.uint32) != theirs_a.view(np.uint32)))
    out["iq4"] = {
        "tensor": _meta(table),
        "n_rows_compared": int(rows.shape[0]),
        "row_ids": [int(v) for v in rows.tolist()],
        "bit_mismatches": bit_mismatch,
        "max_abs": float(np.max(np.abs(ours_a - theirs_a))),
        "our_row0_mean": float(ours_a[0].mean()),
    }

    key_w = _load_named(index, "blk.1.ple_key.weight")
    value_w = _load_named(index, "blk.1.ple_value.weight")
    conv_w = _load_named(index, "blk.1.ple_conv1d.weight")
    if conv_w.shape == (G.ple_conv_kernel, G.hc_dim):
        # ne0 was the channel axis. The kernel wants (hc_dim, kern), k fastest
        # inside each channel, which is the transpose of this layout.
        conv_w = np.ascontiguousarray(conv_w.T)
        out["ple_conv_transposed"] = True
    elif conv_w.shape != (G.hc_dim, G.ple_conv_kernel):
        out["ple_block"] = {"conv_shape": list(conv_w.shape), "error": "unexpected conv shape"}
        return out
    nk = _load_named(index, "blk.1.ple_norm_key.weight")
    nq = _load_named(index, "blk.1.ple_norm_query.weight")
    nc = _load_named(index, "blk.1.ple_norm_conv.weight")
    hidden = rng.normal(size=(1, G.hc, G.n_embd)).astype(np.float32)
    block = ple_block(ours_a.reshape(1, -1), hidden, None, key_w, value_w, nk, nq, nc, conv_w)
    # What the XPU forward computes if the loader subtracted 1 and the module adds 1
    # is the same gamma. What it computes if the file is passed through with an
    # extra +1 is a different block. Record the gap.
    block_extra = ple_block(
        ours_a.reshape(1, -1),
        hidden,
        None,
        key_w,
        value_w,
        1.0 + nk,
        1.0 + nq,
        1.0 + nc,
        conv_w,
    )
    out["ple_block"] = {
        "key": _meta(index["blk.1.ple_key.weight"]),
        "value": _meta(index["blk.1.ple_value.weight"]),
        "conv": _meta(index["blk.1.ple_conv1d.weight"]),
        "numpy_shapes": {
            "key": list(key_w.shape),
            "value": list(value_w.shape),
            "conv": list(conv_w.shape),
        },
        "out_mean": float(block["out"].mean()),
        "gate": block["gate"].reshape(-1).tolist(),
        "stored_gamma_vs_one_plus_stored_max_abs": float(
            np.max(np.abs(block["out"] - block_extra["out"]))
        ),
    }
    report["_ple"] = {
        "emb": ours_a.reshape(1, -1),
        "hidden": hidden,
        "key_w": key_w,
        "value_w": value_w,
        "conv_w": conv_w,
        "nk": nk,
        "nq": nq,
        "nc": nc,
        "out": block["out"],
    }
    return out


def _meta(tensor) -> dict:
    from flashnext_ref.gguf_io import tensor_meta

    return tensor_meta(tensor)


def _text_config(meta: dict, shapes: dict) -> dict:
    """Skeleton for C7. Fields that the header did not carry are marked."""
    def m(key, default, source):
        if key in meta and meta[key] is not None:
            return meta[key], "gguf"
        return default, source

    hidden, hs = m("qwen4exp.embedding_length", G.n_embd, "report-02")
    n_layer, ls = m("qwen4exp.block_count", G.n_layer, "report-02")
    vocab, vs = m("qwen4exp.vocab_size", G.vocab, "report-02")
    ctx, cs = m("qwen4exp.context_length", 262144, "report-02")
    n_q = G.n_q_heads
    n_q_src = "geometry"
    aq = (shapes or {}).get("blk.3.attn_q.weight")
    if aq is not None:
        ne = aq["ggml_ne"]
        wide = max(ne)
        if wide % (2 * G.head_dim) == 0:
            n_q = wide // (2 * G.head_dim)
            n_q_src = "blk.3.attn_q.weight"
    cfg = {
        "model_type": "qwen4_exp_text",
        "vocab_size": vocab,
        "hidden_size": hidden,
        "num_hidden_layers": n_layer,
        "num_attention_heads": n_q,
        "num_key_value_heads": G.n_kv_heads,
        "head_dim": G.head_dim,
        "max_position_embeddings": ctx,
        "rms_norm_eps": G.rms_eps,
        "bos_token_id": G.bos,
        "pad_token_id": G.bos,
        # Text-config eos is the PLE cut (248044). Tokenizer eos is 248046
        # and is a stop id, not the n-gram fill. See the HF text_config.
        "eos_token_id": G.ple_eos,
        "ple_eos_token_id_strata": G.ple_eos,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "moe_intermediate_size": G.expert_ffn,
        "shared_expert_intermediate_size": G.expert_ffn,
        "num_experts": G.n_routed,
        "num_experts_per_tok": G.top_k,
        "hc_count": G.hc,
        "hc_lowrank": G.hc_lr,
        "full_attention_interval": G.full_every,
        "ple_layer_ids": [2],
        "ple_embed_dim": G.n_embd,
        "ple_conv_kernel_size": G.ple_conv_kernel,
        "ngram_size": G.ngram,
        "heads_per_ngram": G.heads_per_ngram,
        "indexer_n_heads": G.idx_n_head,
        "indexer_kv_heads": 1,
        "indexer_head_dim": G.idx_dim,
        "indexer_budget": G.idx_top_k,
        "indexer_compress_ratio": G.idx_block,
        "output_gate_type": "sigmoid",
        "attn_output_gate": True,
        "rope_parameters": {
            "rope_theta": meta.get("qwen4exp.rope.freq_base", G.rope_base),
            "partial_rotary_factor": G.rope_dim / G.head_dim,
            "mrope_section": [11, 11, 10],
            "mrope_interleaved": True,
            "rope_type": "default",
        },
        "hf_config_dir": "/mnt/vm_8tb/b70/build/backend-decision/hf-flashnext",
        "_field_source": {
            "hidden_size": hs,
            "num_hidden_layers": ls,
            "vocab_size": vs,
            "max_position_embeddings": cs,
            "num_attention_heads": n_q_src,
            "ple_layer_ids": "HF 1-based; GGUF qwen4exp.ple.layers is the 0-based block",
            "eos_token_id": "HF text_config eos 248044, same as PLE_EOS. Tokenizer eos is 248046.",
            "rope_parameters": "gguf qwen4exp.rope.freq_base and dimension_sections; partial_rotary_factor 64/256",
        },
    }
    return cfg


def stage_fork(report: dict) -> dict:
    os.environ.setdefault("ONEAPI_DEVICE_SELECTOR", "opencl:cpu")
    os.environ.setdefault("ZE_AFFINITY_MASK", "")
    os.environ.setdefault("VLLM_TARGET_DEVICE", "cpu")
    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
    out: dict = {}
    try:
        import torch
        from vllm.models.qwen4_exp.xpu.model import HyperConnection, Qwen4ExpPLE
    except Exception as exc:  # noqa: BLE001 - record the import wall
        out["import_error"] = f"{type(exc).__name__}: {exc}"
        out["traceback"] = traceback.format_exc()[-2000:]
        return out

    try:
        return _fork_modules(report, torch, HyperConnection, Qwen4ExpPLE)
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["traceback"] = traceback.format_exc()[-2000:]
        return out


def _fork_modules(report, torch, HyperConnection, Qwen4ExpPLE) -> dict:
    out: dict = {"imported": True}
    class Cfg:
        hc_count = G.hc
        hidden_size = G.n_embd
        hc_lowrank = G.hc_lr
        rms_norm_eps = G.rms_eps
        ngram_size = G.ngram
        heads_per_ngram = G.heads_per_ngram
        ple_embed_dim = G.n_embd
        ple_conv_kernel_size = G.ple_conv_kernel
        eos_token_id = G.ple_eos
        image_token_id = G.ple_image_token

    hc_pack = report.get("_hc")
    if not hc_pack:
        out["hc"] = "no layer-0 weights in this process"
    else:
        module = HyperConnection(Cfg(), has_inject=True)
        gamma = hc_pack["gamma"]
        with torch.no_grad():
            module.hc_norm.weight.copy_(torch.from_numpy(gamma - 1.0))
            module.input_mix_weight_down.weight.copy_(torch.from_numpy(hc_pack["down"]))
            module.input_mix_weight_up.weight.copy_(torch.from_numpy(hc_pack["up"]))
            module.block_inject_weight.weight.copy_(torch.from_numpy(hc_pack["inject"]))
            x = torch.from_numpy(hc_pack["residual"])
            mixed, inj = module.mix(x)
        ref = hc_pack["mixed"]
        out["hc_loader_then_mix"] = _stats(mixed.float().numpy(), ref)
        with torch.no_grad():
            module.hc_norm.weight.copy_(torch.from_numpy(gamma))
            mixed_skip, _ = module.mix(x)
        out["hc_skip_deviation_transform"] = _stats(mixed_skip.float().numpy(), ref)
        if inj is not None and hc_pack["inj"] is not None:
            # Recompute inject on the loader path. The skip path overwrote weights.
            with torch.no_grad():
                module.hc_norm.weight.copy_(torch.from_numpy(gamma - 1.0))
                _mixed2, inj2 = module.mix(x)
            out["hc_inject"] = _stats(inj2.float().numpy(), hc_pack["inj"])

    ple_pack = report.get("_ple")
    prompts = report.get("prompts") or {}
    math_ids = prompts.get(PROMPTS[0], {}).get("ids")
    if ple_pack and math_ids:
        ple = Qwen4ExpPLE(Cfg())
        live_mult = report.get("_live_mult")
        # Constants were checked against the header. Use the artifact if the
        # stage_gguf stash is absent; the rows were computed from the header.
        from flashnext_ref.geometry import artifact_ple_consts as apc

        consts = apc()
        if live_mult:
            consts_mult = live_mult
        else:
            consts_mult = list(consts.mult)
        with torch.no_grad():
            ple.ple_embedding.layer_multipliers.copy_(torch.tensor(consts_mult, dtype=torch.int64))
            ple.ple_embedding.ngram_heads_offsets.copy_(
                torch.tensor(consts.offset, dtype=torch.int64)
            )
            ple.ple_embedding.ngram_heads_vocab_sizes.copy_(
                torch.tensor(consts.vocab, dtype=torch.int64)
            )
            ple._ensure_state(1, torch.device("cpu"))
            ids = torch.tensor(math_ids[:4], dtype=torch.int64)
            qsl = torch.tensor([0, ids.shape[0]], dtype=torch.int32)
            slot = torch.tensor([0], dtype=torch.int32)
            has_init = torch.tensor([False])
            rows = ple._compute_rows(ids, qsl, slot, has_init)
        ref_rows = xpu_sequence(math_ids[:4], consts, eos=G.ple_eos)
        out["ple_compute_rows"] = _row_diff(rows.numpy().astype(np.uint32), ref_rows)

        with torch.no_grad():
            ple.key_proj.weight.copy_(torch.from_numpy(ple_pack["key_w"]))
            ple.value_proj.weight.copy_(torch.from_numpy(ple_pack["value_w"]))
            # conv container is (hc_dim, 1, kern); numpy weight is (hc_dim, kern)
            conv = ple_pack["conv_w"]
            if conv.ndim == 2:
                conv_t = torch.from_numpy(conv).unsqueeze(1)
            else:
                conv_t = torch.from_numpy(conv)
            ple.conv1d.weight.copy_(conv_t)
            ple.norm_key.weight.copy_(torch.from_numpy(ple_pack["nk"] - 1.0))
            ple.norm_query.weight.copy_(torch.from_numpy(ple_pack["nq"] - 1.0))
            ple.norm_conv.weight.copy_(torch.from_numpy(ple_pack["nc"] - 1.0))
            emb = torch.from_numpy(ple_pack["emb"])
            hidden = torch.from_numpy(ple_pack["hidden"])
            key = (emb @ ple.key_proj.weight.t()).reshape(1, G.hc, G.n_embd)
            value = emb @ ple.value_proj.weight.t()
            key_n = ple._grouped_norm(key, ple.norm_key.weight)
            query_n = ple._grouped_norm(hidden, ple.norm_query.weight)
            score = (key_n.float() * query_n.float()).sum(dim=-1) / (G.n_embd ** 0.5)
            gate = torch.sigmoid(torch.sign(score) * torch.sqrt(torch.clamp(score.abs(), min=1e-6)))
            gated = value.unsqueeze(1) * gate.unsqueeze(-1)
            normalized = ple._grouped_norm(gated, ple.norm_conv.weight)
            qsl = torch.tensor([0, 1], dtype=torch.int32)
            slot = torch.tensor([0], dtype=torch.int32)
            has_init = torch.tensor([False])
            conv_out = ple._dilated_conv(normalized, qsl, slot, has_init)
            got = hidden + gated + conv_out.reshape(1, G.hc, G.n_embd)
        out["ple_block_loader_then_forward"] = _stats(got.float().numpy(), ple_pack["out"])
    else:
        out["ple"] = "no ple pack or prompt"
    return out


def _try_config_class(cfg: dict) -> str:
    try:
        from transformers import Qwen4ExpTextConfig
    except Exception as exc:  # noqa: BLE001
        return f"import failed: {type(exc).__name__}: {exc}"
    payload = {k: v for k, v in cfg.items() if not k.startswith("_") and k != "ple_eos_token_id_strata"}
    payload.pop("model_type", None)
    try:
        obj = Qwen4ExpTextConfig(**payload)
    except Exception as exc:  # noqa: BLE001
        return f"construct failed: {type(exc).__name__}: {exc}"
    ple_ok = list(obj.ple_layer_ids) == [2]
    layer1 = obj.layer_types[1] if obj.layer_types else None
    return (
        f"ok ple_layer_ids={list(obj.ple_layer_ids)} ple_ok={ple_ok} "
        f"layer1={layer1} heads={obj.num_attention_heads} "
        f"value_heads={obj.linear_num_value_heads} embed_dim={obj.ple_embed_dim}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("self", "gguf", "all"), default="all")
    args = parser.parse_args()
    report: dict = {"stage": args.stage}
    report["self"] = stage_self()
    if args.stage in ("gguf", "all"):
        try:
            report["prompts"] = _tokenize()
        except Exception as exc:  # noqa: BLE001
            report["prompts_error"] = f"{type(exc).__name__}: {exc}"
        try:
            report["gguf"] = stage_gguf(report)
        except Exception as exc:  # noqa: BLE001
            report["gguf_error"] = f"{type(exc).__name__}: {exc}"
            report["gguf_trace"] = traceback.format_exc()[-3000:]
        if args.stage == "all":
            report["fork"] = stage_fork(report)
        meta = (report.get("gguf") or {}).get("metadata") or {}
        shapes = (report.get("gguf") or {}).get("shapes") or {}
        cfg = _text_config(meta, shapes)
        cfg["_construct"] = _try_config_class(cfg)
        cfg_path = HERE / "qwen4exp_text_config.json"
        cfg_path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="ascii")
        report["config_path"] = str(cfg_path)
        report["config_construct"] = cfg["_construct"]

    # Drop numpy stashes before serialising.
    for key in list(report):
        if key.startswith("_"):
            del report[key]
    dest = HERE / "last_compare.json"
    dest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(f"wrote {dest}")
    self_keys = (
        "fresh_strata_vs_xpu_ple_eos",
        "fresh_xpu_ple_eos_vs_generation_eos",
        "token_null_predecessor",
        "selection_width",
        "relu_per_head",
        "tail_bonus",
        "gqa_split",
        "mtp_layouts",
    )
    for key in self_keys:
        print(f"self.{key}={json.dumps(report['self'].get(key), sort_keys=True)}")
    if "prompts" in report:
        for text, info in report["prompts"].items():
            print(f"prompt n={info['n_tokens']} {text[:48]!r}")
    gguf = report.get("gguf") or {}
    if gguf:
        header = gguf.get("ple_header")
        print(f"ple_header={json.dumps(header, sort_keys=True)}")
        print(f"prompt_hash={json.dumps(gguf.get('prompt_hash'), sort_keys=True)}")
        norms = gguf.get("norms") or {}
        for name, info in norms.items():
            if not info:
                print(f"norm {name}=MISSING")
                continue
            print(
                f"norm {name} mean={info['values']['mean']:.6f} "
                f"roundtrip={info['roundtrip_1_plus_w_minus_1_max_abs']:.3e} "
                f"closer_to_one={info['closer_to_one_than_zero']}"
            )
        print(f"hc_layer0={json.dumps(gguf.get('hc_layer0'), sort_keys=True)}")
        print(f"iq4={json.dumps(gguf.get('iq4'), sort_keys=True)}")
        print(f"ple_block={json.dumps(gguf.get('ple_block'), sort_keys=True)}")
    if "fork" in report:
        fork = {k: v for k, v in report["fork"].items() if k != "traceback"}
        print(f"fork={json.dumps(fork, sort_keys=True)}")
    if report.get("gguf_error"):
        print(report["gguf_error"], file=sys.stderr)
        print(report.get("gguf_trace", ""), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
