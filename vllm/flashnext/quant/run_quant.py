"""SignRound WNA16 export for the Flash-Next 35B-A3B dev model.

Three recipes, one trunk (int4 attention + shared expert) reused across them:

  (a) routed experts int3 g128, trunk int4 g128
  (b) routed experts int3 or int2 by FP routing mass, trunk int4
  (c) routed experts int4 g128, trunk int4  (the control)

Router, norms, conv, A_log, dt_bias, MTP, vision, embed, and lm_head stay bf16.
SignRound calls auto_round quant_tensor_sym + SignSGD. Empty experts stay RTN
and are counted. Packing is auto_round:auto_gptq (see plugins/LOWBIT_FORMAT.md).

Device: XPU under bin/gpu-run --card 1 with ZE_AFFINITY_MASK=1, else CPU.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from quant.pack_gptq import GROUP, dequant_linear, pack_linear
from quant.signround_batch import signround_experts_batched, signround_linear, signround_mlp

ORNITH = (
    "/mnt/vm_8tb/github/b70_ai_things/models/files/"
    "ornith-1.5-35b-a3b/bf16-mtp-shisa"
)
OFFICIAL = "/mnt/cache/b70/devmodel/Qwen/Qwen3.6-35B-A3B"
WIKI_TRAIN = "/mnt/cache/b70/devmodel/quant/_calib/train.parquet"
WIKI_TEST = "/mnt/cache/b70/devmodel/quant/_calib/test.parquet"

# First matching extra_config regex wins. Skips come before expert overrides.
SKIP_PATTERNS = [
    "embed_tokens",
    "lm_head",
    "input_layernorm",
    "post_attention_layernorm",
    r"self_attn\.q_norm",
    r"self_attn\.k_norm",
    r"linear_attn\.norm",
    r"shared_expert_gate",
    r"mlp\.gate$",
    "conv1d",
    "A_log",
    "dt_bias",
    "visual",
    r"^mtp\.",
    r"\.mtp\.",
]


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def official_ready(path: str) -> bool:
    root = Path(path)
    index = root / "model.safetensors.index.json"
    if not index.is_file():
        return False
    wm = json.loads(index.read_text())["weight_map"]
    shards = set(wm.values())
    return all((root / s).is_file() and (root / s).stat().st_size > 0 for s in shards)


def choose_model(explicit: str | None) -> tuple[str, str]:
    if explicit:
        name = "qwen36" if "Qwen3.6" in explicit or "qwen3.6" in explicit.lower() else "ornith"
        return explicit, name
    if official_ready(OFFICIAL):
        return OFFICIAL, "qwen36"
    return ORNITH, "ornith"


def load_index(model_dir: str) -> dict:
    return json.loads((Path(model_dir) / "model.safetensors.index.json").read_text())


def load_keys(model_dir: str, wanted: list[str]) -> dict[str, torch.Tensor]:
    if not wanted:
        return {}
    wm = load_index(model_dir)["weight_map"]
    by_shard: dict[str, list[str]] = {}
    for key in wanted:
        by_shard.setdefault(wm[key], []).append(key)
    out: dict[str, torch.Tensor] = {}
    for shard, keys in by_shard.items():
        from safetensors import safe_open

        with safe_open(os.path.join(model_dir, shard), framework="pt") as handle:
            for key in keys:
                out[key] = handle.get_tensor(key)
    return out


def keys_with_prefix(model_dir: str, prefix: str) -> list[str]:
    wm = load_index(model_dir)["weight_map"]
    return [k for k in wm if k.startswith(prefix)]


def text_config(model_dir: str):
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    text = cfg.text_config if getattr(cfg, "text_config", None) is not None else cfg
    text._attn_implementation = "eager"
    return cfg, text


def build_layer(text, layer_idx: int, tensors: dict[str, torch.Tensor], device: torch.device):
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer

    layer = Qwen3_5MoeDecoderLayer(text, layer_idx)
    pref = f"model.language_model.layers.{layer_idx}."
    sd = {k[len(pref):]: v for k, v in tensors.items() if k.startswith(pref)}
    missing, unexpected = layer.load_state_dict(sd, strict=False)
    bad = [m for m in missing if not m.endswith(".bias")]
    if bad or unexpected:
        raise RuntimeError(f"layer {layer_idx} load missing={bad} unexpected={unexpected}")
    layer.eval()
    return layer.to(device=device, dtype=torch.bfloat16)


def make_rotary(text, device: torch.device):
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextRotaryEmbedding

    rotary = Qwen3_5MoeTextRotaryEmbedding(config=text)
    return rotary.to(device)


def masks_and_pos(text, hidden: torch.Tensor, rotary):
    from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask

    batch, seqlen, _ = hidden.shape
    pos = torch.arange(seqlen, device=hidden.device)
    pos = pos.view(1, 1, -1).expand(4, batch, -1)
    text_pos = pos[0]
    rope_pos = pos[1:]
    position_embeddings = rotary(hidden, rope_pos)
    ones = torch.ones(batch, seqlen, device=hidden.device, dtype=torch.long)
    kwargs = {
        "config": text,
        "inputs_embeds": hidden,
        "attention_mask": ones,
        "past_key_values": None,
        "position_ids": text_pos,
    }
    mapping = {
        "full_attention": create_causal_mask(**kwargs),
        "linear_attention": create_recurrent_attention_mask(**kwargs),
    }
    return position_embeddings, text_pos, mapping


def forward_layer(layer, hidden, text, rotary, capture: bool):
    position_embeddings, text_pos, mapping = masks_and_pos(text, hidden, rotary)
    cap: dict = {}
    handles = []
    if capture:
        def pre(name):
            def fn(mod, args):
                cap[name] = args[0].detach()
            return fn

        if layer.block_type == "linear_attention":
            for name in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"):
                handles.append(getattr(layer.linear_attn, name).register_forward_pre_hook(pre(name)))
        else:
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                handles.append(getattr(layer.self_attn, name).register_forward_pre_hook(pre(name)))
        for name in ("gate_proj", "up_proj", "down_proj"):
            handles.append(getattr(layer.mlp.shared_expert, name).register_forward_pre_hook(pre("shared_" + name)))

        def router(mod, args, output):
            cap["mlp_in"] = args[0].detach()
            cap["router_idx"] = output[2].detach()

        handles.append(layer.mlp.gate.register_forward_hook(router))
    try:
        with torch.no_grad():
            out = layer(
                hidden,
                position_embeddings=position_embeddings,
                attention_mask=mapping[layer.block_type],
                position_ids=text_pos,
            )
    finally:
        for handle in handles:
            handle.remove()
    return out, cap


def bucket_tokens(mlp_in: torch.Tensor, router_idx: torch.Tensor, n_exp: int) -> list[torch.Tensor]:
    flat = mlp_in.reshape(-1, mlp_in.shape[-1]).contiguous().cpu()
    idx = router_idx.reshape(-1, router_idx.shape[-1]).cpu()
    out = []
    for e in range(n_exp):
        hit = (idx == e).any(dim=-1)
        out.append(flat[hit] if bool(hit.any()) else flat[:0])
    return out


def hit_counts(router_idx: torch.Tensor, n_exp: int) -> torch.Tensor:
    idx = router_idx.reshape(-1).cpu()
    return torch.bincount(idx, minlength=n_exp)[:n_exp].to(torch.int64)


def assign_bits(counts: torch.Tensor, mass: float) -> torch.Tensor:
    """Hottest experts whose hits cover `mass` of this layer stay int3. Tail is int2."""
    bits = torch.full((counts.numel(),), 2, dtype=torch.int8)
    total = int(counts.sum())
    if total <= 0:
        return bits
    order = torch.argsort(counts, descending=True, stable=True)
    acc = 0
    for ei in order.tolist():
        bits[ei] = 3
        acc += int(counts[ei])
        if acc / total >= mass:
            break
    return bits


def flatten_in(t: torch.Tensor, max_tokens: int) -> torch.Tensor:
    x = t.reshape(-1, t.shape[-1])
    if x.shape[0] > max_tokens:
        sel = torch.linspace(0, x.shape[0] - 1, max_tokens).long()
        x = x[sel]
    return x


def sync(device: torch.device) -> None:
    if device.type == "xpu":
        torch.xpu.synchronize()


def tune_linear(weight, inputs, bits, iters, group_size, device):
    w = weight.detach().to(device=device, dtype=torch.float32)
    x = inputs.detach().to(device=device, dtype=torch.float32)
    qdq, signed, scale_gn, best, rtn = signround_linear(w, x, bits, iters, group_size)
    rel = float((qdq - w).pow(2).sum().sqrt() / w.pow(2).sum().sqrt().clamp(min=1e-12))
    packed = pack_linear(signed.cpu(), scale_gn.float().cpu(), bits, group_size)
    return {
        "qdq": qdq.detach().to(torch.bfloat16).cpu(),
        "packed": packed,
        "best": best,
        "rtn": rtn,
        "rel_l2": rel,
    }


def tune_shared(layer, cap, bits, iters, group_size, device, max_tokens):
    gate = layer.mlp.shared_expert.gate_proj.weight
    up = layer.mlp.shared_expert.up_proj.weight
    down = layer.mlp.shared_expert.down_proj.weight
    x = flatten_in(cap["shared_gate_proj"], max_tokens).to(device=device, dtype=torch.float32)
    tuned = signround_mlp(
        gate.detach().float(),
        up.detach().float(),
        down.detach().float(),
        x,
        bits,
        iters,
        group_size,
    )
    out = {}
    for name, key, module in (
        ("gate_proj", "gate", gate),
        ("up_proj", "up", up),
        ("down_proj", "down", down),
    ):
        part = tuned[key]
        w = module.detach().float()
        rel = float((part["qdq"] - w).pow(2).sum().sqrt() / w.pow(2).sum().sqrt().clamp(min=1e-12))
        out[name] = {
            "qdq": part["qdq"].detach().to(torch.bfloat16).cpu(),
            "packed": pack_linear(part["signed"].cpu(), part["scale_gn"].float().cpu(), bits, group_size),
            "rel_l2": rel,
        }
    out["best"] = tuned["best_loss"]
    out["rtn"] = tuned["rtn_loss"]
    return out


def apply_trunk(layer, trunk: dict, device: torch.device) -> None:
    with torch.no_grad():
        for name, blob in trunk["linears"].items():
            if layer.block_type == "linear_attention":
                mod = getattr(layer.linear_attn, name)
            else:
                mod = getattr(layer.self_attn, name)
            mod.weight.copy_(blob["qdq"].to(device=device, dtype=torch.bfloat16))
        for name, blob in trunk["shared"].items():
            if name in ("best", "rtn"):
                continue
            getattr(layer.mlp.shared_expert, name).weight.copy_(blob["qdq"].to(device=device, dtype=torch.bfloat16))


def apply_experts(layer, qdq_gu, qdq_dn, device: torch.device) -> None:
    with torch.no_grad():
        layer.mlp.experts.gate_up_proj.copy_(qdq_gu.to(device=device, dtype=torch.bfloat16))
        layer.mlp.experts.down_proj.copy_(qdq_dn.to(device=device, dtype=torch.bfloat16))


def scatter_cold(full: dict, cold: list[int], part: dict) -> dict:
    """Overwrite cold-expert rows of an int3 result with an int2 result."""
    if not cold:
        return full
    idx = torch.tensor(cold, dtype=torch.long)
    out = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in full.items()}
    for key in (
        "qdq_gate_up",
        "qdq_down",
        "signed_gate",
        "signed_up",
        "signed_down",
        "scale_gate",
        "scale_up",
        "scale_down",
    ):
        out[key][idx] = part[key]
    return out


def pack_experts(result: dict, bits_per_expert: torch.Tensor, group_size: int) -> dict[str, torch.Tensor]:
    """bits_per_expert [E] int. Returns suffix -> tensor, suffix like '0.gate_proj.qweight'."""
    packed: dict[str, torch.Tensor] = {}
    pieces = (
        ("gate_proj", "signed_gate", "scale_gate"),
        ("up_proj", "signed_up", "scale_up"),
        ("down_proj", "signed_down", "scale_down"),
    )
    e = bits_per_expert.numel()
    for ei in range(e):
        bits = int(bits_per_expert[ei])
        for name, sk, sck in pieces:
            signed = result[sk][ei]
            scale = result[sck][ei].float()
            one = pack_linear(signed, scale, bits, group_size)
            for part, tensor in one.items():
                packed[f"{ei}.{name}.{part}"] = tensor
    return packed


def wiki_windows(tokenizer, parquet: str, n: int, seqlen: int) -> torch.Tensor:
    import pyarrow.parquet as pq

    texts = pq.read_table(parquet).column("text").to_pylist()
    ids: list[int] = []
    eos = tokenizer.eos_token_id
    for text in texts:
        if not text or not str(text).strip():
            continue
        ids.extend(tokenizer.encode(str(text), add_special_tokens=False))
        if eos is not None:
            ids.append(int(eos))
    nwin = len(ids) // seqlen
    if nwin < n:
        raise RuntimeError(f"{parquet} has {nwin} windows of {seqlen}, need {n}")
    arr = torch.tensor(ids[: nwin * seqlen], dtype=torch.long).view(nwin, seqlen)
    return arr[:n].contiguous()


def embed_rows(model_dir: str, ids: torch.Tensor, device: torch.device) -> torch.Tensor:
    weight = load_keys(model_dir, ["model.language_model.embed_tokens.weight"])[
        "model.language_model.embed_tokens.weight"
    ]
    weight = weight.to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        hidden = F.embedding(ids.to(device), weight)
    return hidden, weight


def recipe_dirs(out_root: str, tag: str) -> dict[str, Path]:
    return {
        "w3": Path(out_root) / f"{tag}-w3a16-g128-experts-w4a16-trunk",
        "mix": Path(out_root) / f"{tag}-mixed-int3-int2-experts-w4a16-trunk",
        "w4": Path(out_root) / f"{tag}-w4a16-g128",
    }


def write_config(src_model: str, dest: Path, extra: dict, default_bits: int) -> None:
    cfg = json.loads((Path(src_model) / "config.json").read_text())
    qcfg = {
        "quant_method": "auto-round",
        "bits": default_bits,
        "group_size": GROUP,
        "sym": True,
        "desc_act": False,
        "packing_format": "auto_round:auto_gptq",
        "data_type": "int",
        "backend": "auto",
        "extra_config": extra,
    }
    cfg["quantization_config"] = qcfg
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (dest / "quantize_config.json").write_text(json.dumps(qcfg, indent=2) + "\n")
    (dest / "quantization_config.json").write_text(json.dumps(qcfg, indent=2) + "\n")
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "video_preprocessor_config.json",
        "configuration.json",
    ):
        src = Path(src_model) / name
        if src.is_file():
            shutil.copy2(src, dest / name)


def extra_for(recipe: str, expert_bits: torch.Tensor | None) -> dict:
    extra = {pat: {"bits": 16, "group_size": -1, "sym": True} for pat in SKIP_PATTERNS}
    if recipe == "w4":
        return extra
    if recipe == "w3":
        extra[r"mlp\.experts\.\d+\.(gate|up|down)_proj"] = {"bits": 3, "group_size": GROUP, "sym": True}
        return extra
    # mix: exact int2 keys win over the int3 regex because exact match is first.
    assert expert_bits is not None
    n_layers, n_exp = expert_bits.shape
    for layer in range(n_layers):
        for e in range(n_exp):
            if int(expert_bits[layer, e]) != 2:
                continue
            for proj in ("gate_proj", "up_proj", "down_proj"):
                key = f"model.language_model.layers.{layer}.mlp.experts.{e}.{proj}"
                extra[key] = {"bits": 2, "group_size": GROUP, "sym": True}
    extra[r"mlp\.experts\.\d+\.(gate|up|down)_proj"] = {"bits": 3, "group_size": GROUP, "sym": True}
    return extra


def save_shard(path: Path, tensors: dict[str, torch.Tensor]) -> int:
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    cpu = {k: v.contiguous().cpu() for k, v in tensors.items()}
    save_file(cpu, str(tmp))
    tmp.replace(path)
    return sum(t.numel() * t.element_size() for t in cpu.values())


def update_index(dest: Path, weight_map: dict[str, str], total: int) -> None:
    payload = {"metadata": {"total_size": total}, "weight_map": weight_map}
    (dest / "model.safetensors.index.json").write_text(json.dumps(payload) + "\n")


def write_kept(model_dir: str, dirs: list[Path]) -> None:
    prefixes = (
        "model.language_model.embed_tokens.",
        "model.language_model.norm.",
        "lm_head.",
        "model.visual.",
        "mtp.",
    )
    wm = load_index(model_dir)["weight_map"]
    wanted = [k for k in wm if k.startswith(prefixes)]
    log(f"kept tensors {len(wanted)}")
    tensors = load_keys(model_dir, wanted)
    for dest in dirs:
        n = save_shard(dest / "model-kept.safetensors", tensors)
        index_path = dest / "model.safetensors.index.json"
        wmap = {}
        if index_path.is_file():
            wmap = json.loads(index_path.read_text())["weight_map"]
        for key in tensors:
            wmap[key] = "model-kept.safetensors"
        meta_path = dest / "_sizes.json"
        sizes = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
        sizes["kept"] = n
        meta_path.write_text(json.dumps(sizes) + "\n")
        update_index(dest, wmap, sum(sizes.values()))


def layer_tensor_dict(layer_idx: int, trunk: dict, expert_pack: dict, bf16: dict[str, torch.Tensor]) -> dict:
    pref = f"model.language_model.layers.{layer_idx}."
    out = {}
    block = "linear_attn" if any(k.startswith("in_proj_") for k in trunk["linears"]) else "self_attn"
    for name, blob in trunk["linears"].items():
        for part, tensor in blob["packed"].items():
            out[f"{pref}{block}.{name}.{part}"] = tensor
    for name, blob in trunk["shared"].items():
        if name in ("best", "rtn"):
            continue
        for part, tensor in blob["packed"].items():
            out[f"{pref}mlp.shared_expert.{name}.{part}"] = tensor
    for suffix, tensor in expert_pack.items():
        out[f"{pref}mlp.experts.{suffix}"] = tensor
    for key, tensor in bf16.items():
        out[key] = tensor
    return out


def bf16_layer_tensors(model_dir: str, layer_idx: int, block_type: str) -> dict[str, torch.Tensor]:
    pref = f"model.language_model.layers.{layer_idx}."
    keep_suffix = [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "mlp.gate.weight",
        "mlp.shared_expert_gate.weight",
    ]
    if block_type == "linear_attention":
        keep_suffix += [
            "linear_attn.norm.weight",
            "linear_attn.A_log",
            "linear_attn.conv1d.weight",
            "linear_attn.dt_bias",
        ]
    else:
        keep_suffix += ["self_attn.q_norm.weight", "self_attn.k_norm.weight"]
    keys = [pref + s for s in keep_suffix]
    return load_keys(model_dir, keys)


def micro_check(device: torch.device) -> None:
    g = torch.Generator().manual_seed(0)
    w = torch.randn(64, 256, generator=g)
    x = torch.randn(48, 256, generator=g)
    # 4 sign steps often miss the rounding boundary. 20 steps is enough for this shape.
    _qdq, _s, _sc, best, rtn = signround_linear(w.to(device), x.to(device), bits=3, iters=20)
    if not (best < rtn * 0.8):
        raise RuntimeError(f"SignRound did not beat RTN on the micro linear: best={best} rtn={rtn}")
    log(f"micro linear bits=3 best={best:.6e} rtn={rtn:.6e} ratio={best / rtn:.4f}")
    gu = torch.randn(4, 256, 128, generator=g)
    dn = torch.randn(4, 128, 128, generator=g)
    tokens = [torch.randn(16, 128, generator=g) for _ in range(3)] + [torch.zeros(0, 128)]
    result = signround_experts_batched(
        gu.to(device), dn.to(device), tokens, bits=3, iters=12, max_tokens=16, expert_batch=4
    )
    if result["n_rtn_empty"] != 1 or result["n_tuned"] != 3:
        raise RuntimeError(f"empty-expert count wrong: {result['n_tuned']} {result['n_rtn_empty']}")
    if result["mean_best_loss"] is None or not (result["mean_best_loss"] < result["mean_rtn_loss"] * 0.8):
        raise RuntimeError(
            f"batched SignRound did not beat RTN: {result['mean_best_loss']} vs {result['mean_rtn_loss']}"
        )
    # RTN expert (index 3) must match v=0 codes.
    from auto_round.data_type.int import quant_tensor_sym

    with torch.no_grad():
        flat = gu[3, :128].float().to(device)
        qdq, scale, _mq = quant_tensor_sym(flat, bits=3, group_size=128, v=0, scale_dtype=torch.float32)
        signed = torch.round(qdq.reshape(-1, 128) / scale).clamp(-4, 3).to(torch.int8).cpu()
    if not torch.equal(signed, result["signed_gate"][3]):
        raise RuntimeError("empty expert was not left at RTN codes")
    log(
        f"micro experts best={result['mean_best_loss']:.6e} rtn={result['mean_rtn_loss']:.6e} "
        f"rel_l2={result['rel_l2']:.4f} rtn_empty={result['n_rtn_empty']}"
    )


def run_pilot(args, model_dir: str, device: torch.device) -> None:
    micro_check(device)
    _cfg, text = text_config(model_dir)
    keys = keys_with_prefix(model_dir, "model.language_model.layers.0.")
    log(f"loading layer 0 ({len(keys)} tensors)")
    tensors = load_keys(model_dir, keys)
    layer = build_layer(text, 0, tensors, device)
    rotary = make_rotary(text, device)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    ids = wiki_windows(tok, WIKI_TRAIN, n=2, seqlen=128)
    hidden, _emb = embed_rows(model_dir, ids, device)
    sync(device)
    t0 = time.perf_counter()
    out, cap = forward_layer(layer, hidden, text, rotary, capture=True)
    sync(device)
    fwd = time.perf_counter() - t0
    log(f"forward layer0 {tuple(hidden.shape)} -> {tuple(out.shape)} in {fwd:.2f}s block={layer.block_type}")
    log(f"capture keys {sorted(cap)}")
    n_exp = int(text.num_experts)
    counts = hit_counts(cap["router_idx"], n_exp)
    bits = assign_bits(counts, args.mass)
    log(
        f"routing hits min={int(counts.min())} max={int(counts.max())} "
        f"mean={float(counts.float().mean()):.1f} int3={int((bits == 3).sum())} int2={int((bits == 2).sum())}"
    )
    tokens = bucket_tokens(cap["mlp_in"], cap["router_idx"], n_exp)
    # Time one expert chunk, then one attention linear.
    sync(device)
    t0 = time.perf_counter()
    part = signround_experts_batched(
        layer.mlp.experts.gate_up_proj[: args.expert_batch],
        layer.mlp.experts.down_proj[: args.expert_batch],
        tokens[: args.expert_batch],
        bits=3,
        iters=args.iters,
        max_tokens=args.max_expert_tokens,
        expert_batch=args.expert_batch,
    )
    sync(device)
    dt = time.perf_counter() - t0
    log(
        f"expert chunk E={args.expert_batch} iters={args.iters} {dt:.2f}s "
        f"({dt / args.iters:.3f}s/iter) best={part['mean_best_loss']} rtn={part['mean_rtn_loss']} "
        f"rel_l2={part['rel_l2']}"
    )
    if layer.block_type == "linear_attention":
        name = "in_proj_qkv"
        weight = layer.linear_attn.in_proj_qkv.weight
        inputs = cap[name]
    else:
        name = "q_proj"
        weight = layer.self_attn.q_proj.weight
        inputs = cap[name]
    x = flatten_in(inputs, args.attn_tokens)
    sync(device)
    t0 = time.perf_counter()
    tuned = tune_linear(weight, x, 4, args.iters, GROUP, device)
    sync(device)
    dt = time.perf_counter() - t0
    log(
        f"linear {name} {tuple(weight.shape)} tokens={x.shape[0]} iters={args.iters} {dt:.2f}s "
        f"best={tuned['best']:.6e} rtn={tuned['rtn']:.6e} rel_l2={tuned['rel_l2']:.4f}"
    )
    # Pack round-trip of the first tuned expert gate.
    signed = part["signed_gate"][0]
    scale = part["scale_gate"][0].float()
    packed = pack_linear(signed, scale, 3, GROUP)
    rec = dequant_linear(packed, 3, GROUP)
    ref = part["qdq_gate_up"][0, : signed.shape[0]].float()
    # qdq was cast to bf16; compare dequant to bf16 qdq, not to fp32 codes*scale.
    gap = float((rec - ref).abs().max())
    log(f"pack dequant vs bf16 qdq max_abs={gap:.6e} qweight={tuple(packed['qweight'].shape)}")
    if gap > 0.05:
        raise RuntimeError(f"pack/dequant gap too large: {gap}")


def quantize(args, model_dir: str, tag: str, device: torch.device) -> None:
    dirs = recipe_dirs(args.out_root, tag)
    work = Path(args.out_root) / "_work" / tag
    work.mkdir(parents=True, exist_ok=True)
    _cfg, text = text_config(model_dir)
    n_layers = int(text.num_hidden_layers)
    n_exp = int(text.num_experts)
    lo, hi = args.layer_lo, args.layer_hi
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    calib_path = work / f"calib_s{args.seqlen}_n{args.nsamples}.pt"
    if calib_path.is_file():
        ids = torch.load(calib_path)
    else:
        ids = wiki_windows(tok, WIKI_TRAIN, args.nsamples, args.seqlen)
        torch.save(ids, calib_path)
    ppl_path = work / f"ppl_s{args.seqlen}_n{args.ppl_windows}.pt"
    if not ppl_path.is_file():
        torch.save(wiki_windows(tok, WIKI_TEST, args.ppl_windows, args.seqlen), ppl_path)
    log(f"calib {tuple(ids.shape)} from wikitext-2-raw train prefix")
    hidden0, _emb = embed_rows(model_dir, ids, device)
    del _emb
    state_path = work / "streams.pt"
    if state_path.is_file():
        state = torch.load(state_path, weights_only=False)
        if state.get("fingerprint") != _fingerprint(args, model_dir):
            raise RuntimeError(f"fingerprint mismatch in {state_path}; delete _work to restart")
        next_layer = int(state["next_layer"])
        streams = {k: v.to(device) for k, v in state["streams"].items()}
        expert_bits = state["expert_bits"]
        log(f"resume at layer {next_layer}")
    else:
        streams = {k: hidden0.clone() for k in ("fp", "w4", "w3", "mix")}
        expert_bits = torch.full((n_layers, n_exp), -1, dtype=torch.int8)
        next_layer = lo
        if lo != 0:
            raise RuntimeError("--layers must start at 0 unless _work streams.pt is a resume of this fingerprint")
    rotary = make_rotary(text, device)
    # configs for uniform recipes can be written now. mix config is rewritten at the end.
    for name in ("w4", "w3", "mix"):
        if name != "mix":
            write_config(model_dir, dirs[name], extra_for(name, None), 4)
        else:
            dirs[name].mkdir(parents=True, exist_ok=True)
        meta = {
            "source_model": model_dir,
            "recipe": name,
            "iters": args.iters,
            "nsamples": args.nsamples,
            "seqlen": args.seqlen,
            "seed": args.seed,
            "mass": args.mass,
            "max_expert_tokens": args.max_expert_tokens,
            "attn_tokens": args.attn_tokens,
            "expert_batch": args.expert_batch,
            "group_size": GROUP,
            "sym": True,
            "algorithm": "auto_round.data_type.int.quant_tensor_sym + SignSGD",
            "lr": "1/iters with LinearLR 1 -> 0",
            "minmax_clamp": [0.0, 1.0],
            "calib": "wikitext-2-raw-v1 train, first nsamples non-overlapping windows",
            "routing": "FP prefix on that calib, per layer, hottest experts covering mass stay int3",
            "not_strata_residency_profile": True,
            "packing": "auto_round:auto_gptq bitstream, see plugins/LOWBIT_FORMAT.md",
        }
        (dirs[name] / "quant_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    if not (dirs["w4"] / "model-kept.safetensors").is_file():
        write_kept(model_dir, [dirs[k] for k in ("w4", "w3", "mix")])
        write_config(model_dir, dirs["mix"], extra_for("w3", None), 4)

    stats_path = work / "layer_stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.is_file() else []

    for layer_idx in range(next_layer, hi + 1):
        t_layer = time.perf_counter()
        pref = f"model.language_model.layers.{layer_idx}."
        log(f"layer {layer_idx} load")
        tensors = load_keys(model_dir, keys_with_prefix(model_dir, pref))
        layer = build_layer(text, layer_idx, tensors, device)
        fp_out, cap_fp = forward_layer(layer, streams["fp"], text, rotary, capture=True)
        counts = hit_counts(cap_fp["router_idx"], n_exp)
        bits = assign_bits(counts, args.mass)
        expert_bits[layer_idx] = bits
        caps = {"fp": cap_fp}
        for name in ("w4", "w3", "mix"):
            # Layer 0 starts from the same embedding for every recipe.
            if torch.equal(streams[name], streams["fp"]):
                caps[name] = cap_fp
                continue
            _out, caps[name] = forward_layer(layer, streams[name], text, rotary, capture=True)
        trunk = {"linears": {}, "shared": {}}
        if layer.block_type == "linear_attention":
            linear_names = ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj")
            parent = layer.linear_attn
        else:
            linear_names = ("q_proj", "k_proj", "v_proj", "o_proj")
            parent = layer.self_attn
        for name in linear_names:
            x = flatten_in(caps["w4"][name], args.attn_tokens)
            trunk["linears"][name] = tune_linear(getattr(parent, name).weight, x, 4, args.iters, GROUP, device)
            log(f"  trunk {name} rel_l2={trunk['linears'][name]['rel_l2']:.4f} best/rtn="
                f"{trunk['linears'][name]['best']:.4e}/{trunk['linears'][name]['rtn']:.4e}")
        trunk["shared"] = tune_shared(layer, caps["w4"], 4, args.iters, GROUP, device, args.attn_tokens)
        log(f"  shared best/rtn={trunk['shared']['best']:.4e}/{trunk['shared']['rtn']:.4e}")

        def _tokens(name):
            return bucket_tokens(caps[name]["mlp_in"], caps[name]["router_idx"], n_exp)

        log("  experts int4")
        exp4 = signround_experts_batched(
            layer.mlp.experts.gate_up_proj,
            layer.mlp.experts.down_proj,
            _tokens("w4"),
            4,
            args.iters,
            max_tokens=args.max_expert_tokens,
            expert_batch=args.expert_batch,
        )
        log(f"  int4 rel_l2={exp4['rel_l2']:.4f} best/rtn={exp4['mean_best_loss']}/{exp4['mean_rtn_loss']} "
            f"tuned={exp4['n_tuned']} rtn_empty={exp4['n_rtn_empty']}")
        log("  experts int3")
        exp3 = signround_experts_batched(
            layer.mlp.experts.gate_up_proj,
            layer.mlp.experts.down_proj,
            _tokens("w3"),
            3,
            args.iters,
            max_tokens=args.max_expert_tokens,
            expert_batch=args.expert_batch,
        )
        log(f"  int3 rel_l2={exp3['rel_l2']:.4f} best/rtn={exp3['mean_best_loss']}/{exp3['mean_rtn_loss']} "
            f"tuned={exp3['n_tuned']} rtn_empty={exp3['n_rtn_empty']}")
        cold = (bits == 2).nonzero().flatten().tolist()
        log(f"  experts int2 cold={len(cold)}")
        if cold:
            gu = layer.mlp.experts.gate_up_proj[cold]
            dn = layer.mlp.experts.down_proj[cold]
            tok = _tokens("mix")
            exp2 = signround_experts_batched(
                gu, dn, [tok[i] for i in cold], 2, args.iters,
                max_tokens=args.max_expert_tokens, expert_batch=args.expert_batch,
            )
            exp_mix = scatter_cold(exp3, cold, exp2)
            mix_rel = exp2["rel_l2"]
        else:
            exp_mix = exp3
            mix_rel = None
            exp2 = None
        log(f"  int2 rel_l2={mix_rel}")

        bf16 = bf16_layer_tensors(model_dir, layer_idx, layer.block_type)
        bits4 = torch.full((n_exp,), 4, dtype=torch.int8)
        bits3 = torch.full((n_exp,), 3, dtype=torch.int8)
        packs = {
            "w4": pack_experts(exp4, bits4, GROUP),
            "w3": pack_experts(exp3, bits3, GROUP),
            "mix": pack_experts(exp_mix, bits, GROUP),
        }
        qdq = {"w4": exp4, "w3": exp3, "mix": exp_mix}
        streams["fp"] = fp_out.detach()
        for name in ("w4", "w3", "mix"):
            apply_trunk(layer, trunk, device)
            apply_experts(layer, qdq[name]["qdq_gate_up"], qdq[name]["qdq_down"], device)
            nxt, _cap = forward_layer(layer, streams[name], text, rotary, capture=False)
            streams[name] = nxt.detach()
            blob = layer_tensor_dict(layer_idx, trunk, packs[name], bf16)
            fname = f"model-layer-{layer_idx:02d}.safetensors"
            nbytes = save_shard(dirs[name] / fname, blob)
            meta_path = dirs[name] / "_sizes.json"
            sizes = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
            sizes[fname] = nbytes
            meta_path.write_text(json.dumps(sizes) + "\n")
            wmap = {}
            index_path = dirs[name] / "model.safetensors.index.json"
            if index_path.is_file():
                wmap = json.loads(index_path.read_text())["weight_map"]
            for key in blob:
                wmap[key] = fname
            update_index(dirs[name], wmap, sum(sizes.values()))
        row = {
            "layer": layer_idx,
            "block": layer.block_type,
            "seconds": time.perf_counter() - t_layer,
            "hits_min": int(counts.min()),
            "hits_max": int(counts.max()),
            "hits_mean": float(counts.float().mean()),
            "n_int3": int((bits == 3).sum()),
            "n_int2": int((bits == 2).sum()),
            "int4_rel_l2": exp4["rel_l2"],
            "int3_rel_l2": exp3["rel_l2"],
            "int2_rel_l2": None if exp2 is None else exp2["rel_l2"],
            "int4_best": exp4["mean_best_loss"],
            "int4_rtn": exp4["mean_rtn_loss"],
            "int3_best": exp3["mean_best_loss"],
            "int3_rtn": exp3["mean_rtn_loss"],
            "int2_best": None if exp2 is None else exp2["mean_best_loss"],
            "int2_rtn": None if exp2 is None else exp2["mean_rtn_loss"],
            "int4_rtn_empty": exp4["n_rtn_empty"],
            "int3_rtn_empty": exp3["n_rtn_empty"],
            "trunk_rel_l2": {k: v["rel_l2"] for k, v in trunk["linears"].items()},
            "shared_rel_l2": {k: v["rel_l2"] for k, v in trunk["shared"].items() if k not in ("best", "rtn")},
            "shared_best": trunk["shared"]["best"],
            "shared_rtn": trunk["shared"]["rtn"],
        }
        stats = [s for s in stats if s["layer"] != layer_idx]
        stats.append(row)
        stats_path.write_text(json.dumps(stats, indent=2) + "\n")
        torch.save(expert_bits, work / "expert_bits.pt")
        torch.save(
            {
                "fingerprint": _fingerprint(args, model_dir),
                "next_layer": layer_idx + 1,
                "streams": {k: v.cpu() for k, v in streams.items()},
                "expert_bits": expert_bits,
            },
            state_path,
        )
        log(f"layer {layer_idx} done in {row['seconds']:.1f}s int3_rel={row['int3_rel_l2']:.4f}")
        del layer, tensors, caps, trunk, exp4, exp3, exp_mix, packs
        if device.type == "xpu":
            torch.xpu.empty_cache()

    # Final mix config now that every expert bit is known. Layers outside the
    # requested range stay -1; only write when the map is complete.
    if int((expert_bits < 0).sum()) == 0:
        write_config(model_dir, dirs["mix"], extra_for("mix", expert_bits), 4)
        (dirs["mix"] / "expert_bits.json").write_text(
            json.dumps({"mass": args.mass, "bits": expert_bits.tolist(),
                        "note": "int8[layer, expert]; 3 covers the hottest mass, 2 is the tail"}) + "\n"
        )
    (work / "expert_bits.json").write_text(json.dumps(expert_bits.tolist()) + "\n")
    log(f"quant finished layers {lo}-{hi}")


def _fingerprint(args, model_dir: str) -> dict:
    return {
        "model": model_dir,
        "iters": args.iters,
        "nsamples": args.nsamples,
        "seqlen": args.seqlen,
        "seed": args.seed,
        "mass": args.mass,
        "max_expert_tokens": args.max_expert_tokens,
        "attn_tokens": args.attn_tokens,
        "group": GROUP,
    }


def run_eval(args, model_dir: str, tag: str, device: torch.device) -> None:
    """Teacher-forced wikitext-2-raw test perplexity on the exported dequant."""
    dirs = recipe_dirs(args.out_root, tag)
    work = Path(args.out_root) / "_work" / tag
    _cfg, text = text_config(model_dir)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    ppl_path = work / f"ppl_s{args.seqlen}_n{args.ppl_windows}.pt"
    ids = torch.load(ppl_path) if ppl_path.is_file() else wiki_windows(tok, WIKI_TEST, args.ppl_windows, args.seqlen)
    rotary = make_rotary(text, device)
    norm_w = load_keys(model_dir, ["model.language_model.norm.weight"])["model.language_model.norm.weight"]
    head_w = load_keys(model_dir, ["lm_head.weight"])["lm_head.weight"]
    norm_w = norm_w.to(device=device, dtype=torch.float32)
    results = {}
    for name, dest in dirs.items():
        if not (dest / "model-layer-00.safetensors").is_file():
            log(f"skip eval {name}: no layer files")
            continue
        nll_sum = 0.0
        tok_sum = 0
        for bi in range(ids.shape[0]):
            hidden, _ = embed_rows(model_dir, ids[bi: bi + 1], device)
            for layer_idx in range(int(text.num_hidden_layers)):
                hidden = _eval_layer(model_dir, dest, text, layer_idx, hidden, rotary, device)
            var = hidden.float().pow(2).mean(dim=-1, keepdim=True)
            h = hidden.float() * torch.rsqrt(var + text.rms_norm_eps) * (1.0 + norm_w)
            # one sequence, chunk the vocab matmul to limit the logit tensor
            logits = F.linear(h, head_w.to(device=device, dtype=torch.float32))
            shift_logits = logits[:, :-1].reshape(-1, logits.shape[-1])
            shift_labels = ids[bi: bi + 1, 1:].to(device).reshape(-1)
            nll = F.cross_entropy(shift_logits, shift_labels, reduction="sum")
            nll_sum += float(nll)
            tok_sum += int(shift_labels.numel())
            del logits
        ppl = float(torch.exp(torch.tensor(nll_sum / tok_sum)))
        results[name] = {"nll": nll_sum / tok_sum, "ppl": ppl, "tokens": tok_sum}
        log(f"ppl {name} {ppl:.4f} nll={nll_sum / tok_sum:.4f} tokens={tok_sum}")
    (work / "ppl.json").write_text(json.dumps(results, indent=2) + "\n")


def _eval_layer(model_dir, dest, text, layer_idx, hidden, rotary, device):
    from safetensors import safe_open

    path = dest / f"model-layer-{layer_idx:02d}.safetensors"
    pref = f"model.language_model.layers.{layer_idx}."
    with safe_open(str(path), framework="pt") as handle:
        keys = list(handle.keys())
        # Build a fake state dict of bf16 weights by dequanting packed linears
        # and copying raw tensors. Then the ordinary decoder forward.
        raw = {}
        packed_prefix = {}
        for key in keys:
            if key.endswith(".qweight"):
                packed_prefix[key[: -len(".qweight")]] = True
            elif not any(key.endswith(s) for s in (".qzeros", ".scales", ".g_idx")):
                raw[key[len(pref):]] = handle.get_tensor(key)
        for prefix in packed_prefix:
            qw = handle.get_tensor(prefix + ".qweight")
            sc = handle.get_tensor(prefix + ".scales")
            gidx = handle.get_tensor(prefix + ".g_idx")
            k = int(gidx.numel())
            bits = int(qw.shape[0] * 32 // k)
            weight = dequant_linear(
                {"qweight": qw, "scales": sc, "qzeros": handle.get_tensor(prefix + ".qzeros"), "g_idx": gidx},
                bits,
                GROUP,
            )
            rel = prefix[len(pref):]
            if ".experts." in rel:
                # filled below
                raw.setdefault("_expert", {})[rel] = weight
            else:
                raw[rel + ".weight"] = weight.to(torch.bfloat16)
    # Assemble fused experts from per-expert gate/up/down.
    n_exp = int(text.num_experts)
    # infer shapes from text config
    h = int(text.hidden_size)
    inter = int(text.moe_intermediate_size)
    gu = torch.empty(n_exp, 2 * inter, h, dtype=torch.bfloat16)
    dn = torch.empty(n_exp, h, inter, dtype=torch.bfloat16)
    experts = raw.pop("_expert", {})
    for e in range(n_exp):
        g = experts[f"mlp.experts.{e}.gate_proj"]
        u = experts[f"mlp.experts.{e}.up_proj"]
        d = experts[f"mlp.experts.{e}.down_proj"]
        gu[e, :inter] = g.to(torch.bfloat16)
        gu[e, inter:] = u.to(torch.bfloat16)
        dn[e] = d.to(torch.bfloat16)
    raw["mlp.experts.gate_up_proj"] = gu
    raw["mlp.experts.down_proj"] = dn
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer

    layer = Qwen3_5MoeDecoderLayer(text, layer_idx)
    missing, unexpected = layer.load_state_dict(raw, strict=False)
    bad = [m for m in missing if not m.endswith(".bias")]
    if bad or unexpected:
        raise RuntimeError(f"eval layer {layer_idx} missing={bad} unexpected={unexpected}")
    layer.eval().to(device=device, dtype=torch.bfloat16)
    out, _ = forward_layer(layer, hidden, text, rotary, capture=False)
    del layer
    return out


def parse_layers(spec: str, n_layers: int) -> tuple[int, int]:
    if "-" not in spec:
        i = int(spec)
        return i, i
    a, b = spec.split("-", 1)
    lo = int(a) if a else 0
    hi = int(b) if b else n_layers - 1
    return lo, hi


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("pilot", "quant", "eval"), default="pilot")
    p.add_argument("--model", default=None)
    p.add_argument("--out-root", default="/mnt/cache/b70/devmodel/quant")
    p.add_argument("--device", choices=("xpu", "cpu"), default="xpu")
    p.add_argument("--iters", type=int, default=8)
    p.add_argument("--nsamples", type=int, default=8)
    p.add_argument("--seqlen", type=int, default=512)
    p.add_argument("--ppl-windows", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mass", type=float, default=0.8)
    p.add_argument("--max-expert-tokens", type=int, default=256)
    p.add_argument("--attn-tokens", type=int, default=4096)
    p.add_argument("--expert-batch", type=int, default=32)
    p.add_argument("--layers", default="0-39")
    args = p.parse_args()
    if args.iters < 1:
        raise SystemExit("iters must be >= 1")
    torch.manual_seed(args.seed)
    model_dir, tag = choose_model(args.model)
    if args.device == "xpu":
        if not torch.xpu.is_available():
            raise SystemExit("xpu requested but torch.xpu.is_available() is false")
        device = torch.device("xpu:0")
    else:
        device = torch.device("cpu")
    log(f"mode={args.mode} model={model_dir} tag={tag} device={device} visible={os.environ.get('ZE_AFFINITY_MASK', '')}")
    _cfg, text = text_config(model_dir)
    args.layer_lo, args.layer_hi = parse_layers(args.layers, int(text.num_hidden_layers))
    if args.mode == "pilot":
        run_pilot(args, model_dir, device)
    elif args.mode == "quant":
        quantize(args, model_dir, tag, device)
    else:
        run_eval(args, model_dir, tag, device)


if __name__ == "__main__":
    main()
