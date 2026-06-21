"""AutoRound W8A8 (int8 weight + dynamic per-token int8 activation) -> compressed-tensors.

Produces a checkpoint for our int8 W8A8 oneDNN XPU kernel (image vllm-xpu-env:int8/:int8g).
Driven entirely by env vars (see scripts/65_autoround_w8a8.sh). Works for dense (14B) and the
qwen3_5 VLM/MoE sources via a loader fallback chain. lm_head + IGN_REGEX modules stay 16-bit.

Refs: docs/kernel/15 sec 2 (the inline recipe this generalizes); QUANTS_TODO Q1/Q2/Q4/Q6.
"""
import os
import re
import sys
import time
import traceback

import torch
from transformers import AutoTokenizer

SRC = os.environ["SRC"]
OUT = os.environ["OUT"]
ITERS = int(os.environ.get("ITERS", "200"))
NSAMPLES = int(os.environ.get("NSAMPLES", "128"))
SEQLEN = int(os.environ.get("SEQLEN", "2048"))
DEVMAP = os.environ.get("DEVMAP", "xpu")          # "xpu" (1 card) | "0,1" (both) | "auto"
GROUP = int(os.environ.get("GROUP", "-1"))        # -1 = per-channel (W8A8 canonical)
LOWMEM = os.environ.get("LOWMEM", "1") == "1"     # keep model on CPU, stream blocks to GPU
# Modules to keep in bf16 (never quantize). Default covers qwen3_5 VLM + MTP + DeltaNet.
IGN_REGEX = os.environ.get("IGN_REGEX", r"(visual|\.mtp|mtp\.|linear_attn)")
# Extra ignore for MoE: keep the router/gate in bf16 (set IGN_MOE=1 from the shell for 35B).
if os.environ.get("IGN_MOE", "0") == "1":
    IGN_REGEX = IGN_REGEX[:-1] + r"|\.gate\.|mlp\.gate$|router)"

print("=== _autoround_w8a8 ENV ===", flush=True)
print("torch", torch.__version__, "xpu_avail", torch.xpu.is_available(),
      "count", torch.xpu.device_count(), flush=True)
for i in range(torch.xpu.device_count()):
    try:
        print(f"  xpu:{i} -> {torch.xpu.get_device_name(i)}", flush=True)
    except Exception as e:
        print(f"  xpu:{i} name err: {e}", flush=True)
print(f"SRC={SRC}\nOUT={OUT}", flush=True)
print(f"ITERS={ITERS} NSAMPLES={NSAMPLES} SEQLEN={SEQLEN} DEVMAP={DEVMAP} "
      f"GROUP={GROUP} LOWMEM={LOWMEM}", flush=True)
print(f"IGN_REGEX={IGN_REGEX}", flush=True)

import auto_round
from auto_round import AutoRound
from auto_round.schemes import QuantizationScheme
print("auto_round", getattr(auto_round, "__version__", "?"), flush=True)

# W8A8: per-channel sym int8 weights + dynamic per-token sym int8 activations.
W8A8 = QuantizationScheme(bits=8, group_size=GROUP, sym=True, data_type="int",
                          act_bits=8, act_dynamic=True, act_sym=True, act_data_type="int")

tok = AutoTokenizer.from_pretrained(SRC, trust_remote_code=True)


def _load():
    from transformers import AutoModelForCausalLM
    try:
        return AutoModelForCausalLM.from_pretrained(
            SRC, dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True)
    except Exception as e:
        print(f"[load] AutoModelForCausalLM failed ({type(e).__name__}: {str(e)[:300]})", flush=True)
    import transformers as T
    for cls_name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq", "AutoModel"):
        try:
            cls = getattr(T, cls_name)
            m = cls.from_pretrained(SRC, dtype=torch.bfloat16, trust_remote_code=True,
                                    low_cpu_mem_usage=True)
            print(f"[load] loaded via {cls_name}", flush=True)
            return m
        except Exception as e:
            print(f"[load] {cls_name} failed ({type(e).__name__}: {str(e)[:300]})", flush=True)
    raise SystemExit("FAIL: no loader could instantiate the model")


print(f"[load] {SRC} bf16 (low_cpu_mem_usage) ...", flush=True)
model = _load()

# Build layer_config: force lm_head + IGN_REGEX Linear modules to 16-bit; count what we quantize.
ign_re = re.compile(IGN_REGEX)
layer_config = {}
n_quant = 0
n_ign = 0
for name, mod in model.named_modules():
    if isinstance(mod, torch.nn.Linear):
        if name.endswith("lm_head") or ign_re.search(name):
            # MUST set act_bits=16 too: the llm_compressor exporter's check_to_quantized() gate is
            # `bits<=8 OR act_bits<=8` -- leaving act_bits at the scheme default (8) makes it try to
            # pack a never-quantized weight -> layer.scale is None -> AttributeError at export.py:152.
            layer_config[name] = {"bits": 16, "act_bits": 16}
            n_ign += 1
        else:
            n_quant += 1
print(f"[cfg] {n_ign} Linear modules forced 16-bit; {n_quant} Linear modules to quantize", flush=True)
if n_quant == 0:
    raise SystemExit("FAIL: 0 modules to quantize -- IGN_REGEX too broad?")

kw = dict(scheme=W8A8, layer_config=layer_config, iters=ITERS, nsamples=NSAMPLES,
          seqlen=SEQLEN, device_map=DEVMAP, low_gpu_mem_usage=LOWMEM, format="llm_compressor")
# Per-block tuning activation memory ~ batch_size x seqlen. At nsamples=128, seqlen=2048 the default
# batch_size OOMs one 32GB card (UR_RESULT_ERROR_OUT_OF_RESOURCES at layer 0). Cap batch_size and use
# gradient_accumulate_steps to keep the effective batch (and gradient quality) up at low peak memory.
BATCHSIZE = int(os.environ.get("BATCHSIZE", "0"))
GRADACC = int(os.environ.get("GRADACC", "0"))
if BATCHSIZE > 0:
    kw["batch_size"] = BATCHSIZE
if GRADACC > 0:
    kw["gradient_accumulate_steps"] = GRADACC
print(f"[cfg] batch_size={BATCHSIZE or 'default'} grad_accum={GRADACC or 'default'}", flush=True)
print(f"[quant] AutoRound(iters={ITERS}, nsamples={NSAMPLES}, device_map={DEVMAP}) ...", flush=True)
t0 = time.time()
try:
    ar = AutoRound(model, tok, **kw)
except TypeError as e:
    # Older AutoRound may not accept low_gpu_mem_usage / format in the ctor -- retry minimal.
    print(f"[quant] ctor TypeError ({e}); retry without low_gpu_mem_usage/format", flush=True)
    kw.pop("low_gpu_mem_usage", None)
    kw.pop("format", None)
    ar = AutoRound(model, tok, **kw)
ar.quantize_and_save(output_dir=OUT, format="llm_compressor")
print(f"[done] quant+save {time.time()-t0:.0f}s -> {OUT}", flush=True)
print("DONE_AUTOROUND_W8A8", OUT, flush=True)
