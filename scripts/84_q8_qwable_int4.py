"""QUANTS_TODO Q8 -- Qwable-5-27B-Coder -> INT4 W4A16 via AutoRound (inc-servable, like the Lorbus daily driver).

Runs inside vllm-xpu-env:v0230 across both B70s. Env: SRC, OUT, ITERS, NSAMPLES, SEQLEN, DEVMAP (default 0,1).
SMOKE first: ITERS=2 NSAMPLES=8 (validate toolchain + the text-only-calib dodge + export format) BEFORE the 4-8h full run.

Key gotcha (from QUANTS Q2/Q4): AutoRound's DEFAULT multimodal calibration BLOCKS on this qwen3_5 VLM. Dodge = TEXT-ONLY
calib (a list[str], no processor/images) + quant_nontext_module=False. Keep vision/MTP/DeltaNet/lm_head at bf16 (codex 2026-06-22).
"""
import os, re, sys, traceback
import torch

SRC = os.environ.get("SRC", "/models/DJLougen_Qwable-5-27B-Coder")
OUT = os.environ.get("OUT", "/models/Qwable-5-27B-Coder-int4-AutoRound")
ITERS = int(os.environ.get("ITERS", "200"))
NSAMPLES = int(os.environ.get("NSAMPLES", "128"))
SEQLEN = int(os.environ.get("SEQLEN", "2048"))
DEVMAP = os.environ.get("DEVMAP", "0,1")

print(f"=== Q8 AutoRound W4A16 :: SRC={SRC} OUT={OUT} iters={ITERS} nsamples={NSAMPLES} seqlen={SEQLEN} devmap={DEVMAP} ===")
print("torch", torch.__version__, "xpu", torch.xpu.is_available(), "count", torch.xpu.device_count())

import auto_round
from auto_round import AutoRound
print("auto_round", getattr(auto_round, "__version__", "?"))
import inspect
try:
    print("AutoRound signature:", inspect.signature(AutoRound))
except Exception as e:
    print("sig err", e)

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(SRC, trust_remote_code=True)

# ---- text-only calibration corpus (chat/code) -- avoids the multimodal calib path that blocked Q2/Q4 ----
_BASE = [
    "Write a Python function to compute the nth Fibonacci number iteratively, with a docstring.",
    "Explain the difference between a process and a thread, and when you would use each.",
    "Implement a thread-safe LRU cache class in Python with get and put in O(1).",
    "Given a list of integers, return the indices of the two numbers that add up to a target.",
    "Refactor this code to be more readable and explain each change you make step by step.",
    "Describe how a hash map handles collisions and analyze the worst-case lookup complexity.",
    "Write a SQL query to find the second highest salary in an employees table.",
    "What are the tradeoffs between depth-first and breadth-first search on a large graph?",
    "Implement binary search over a sorted array and prove its loop invariant holds.",
    "Summarize the SOLID principles of object-oriented design with a short example for each.",
]
# CALIB_DS: default "NeelNanda/pile-10k" (a HF text dataset AutoRound tokenizes + chunks to UNIFORM seqlen).
# The in-script text list (CALIB_DS=list) has VARIABLE tokenized lengths -> AutoRound's batched calib cat throws
# "Sizes of tensors must match ... Expected 23 but got 24" at nsamples>=batch_size (crashed the full run). Use pile-10k.
_DS = os.environ.get("CALIB_DS", "NeelNanda/pile-10k")
if _DS.lower() == "list":
    calib = [(_BASE[i % len(_BASE)] + f"\n\n(sample {i})") for i in range(max(NSAMPLES, 8))]
    print("calib: in-script text list, n=", len(calib))
else:
    calib = _DS
    print("calib: HF dataset =", calib)

# ---- layer_config: keep vision tower, MTP head, DeltaNet linear_attn, lm_head at bf16 (bits=16) ----
# AutoRound keys layer_config by exact module name -> enumerate the loaded model's nn.Linear modules.
IGNORE = re.compile(r"(visual|vision_tower|mtp|linear_attn)")
def is_ignored(name):
    return name == "lm_head" or name.endswith(".lm_head") or bool(IGNORE.search(name))

def build_layer_config(model):
    cfg = {}
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and is_ignored(name):
            cfg[name] = {"bits": 16}
    print(f"layer_config: {len(cfg)} modules forced to bf16 (vision/mtp/linear_attn/lm_head). sample:", list(cfg)[:6])
    return cfg

# ---- load the model TEXT-ONLY across both cards (+cpu overflow) so AutoRound never touches the vision processor ----
from transformers import AutoModelForCausalLM
mm = {0: "30GiB", 1: "30GiB", "cpu": "120GiB"}
print("loading model (AutoModelForCausalLM, device_map=auto, max_memory=%s) ..." % mm)
model = AutoModelForCausalLM.from_pretrained(
    SRC, torch_dtype=torch.bfloat16, device_map="auto", max_memory=mm, trust_remote_code=True)
layer_config = build_layer_config(model)

# ---- AutoRound 0.13.x auto-detects MLLM mode for this VLM ("Using MLLM mode") and REQUIRES a processor
# (quantize() else asserts "processor should not be None"). Provide it; quant_nontext_module=False -> quantize
# ONLY the language-model text modules (vision stays bf16 anyway via layer_config). ----
proc = None
for loader in ("AutoProcessor", "AutoImageProcessor"):
    try:
        import transformers
        proc = getattr(transformers, loader).from_pretrained(SRC, trust_remote_code=True)
        print(f"loaded {loader}: {type(proc).__name__}"); break
    except Exception as e:
        print(f"{loader} load failed:", str(e)[:200])

have_mllm = False
try:
    from auto_round import AutoRoundMLLM
    have_mllm = True
    try: print("AutoRoundMLLM signature:", inspect.signature(AutoRoundMLLM))
    except Exception as e: print("mllm sig err", e)
except Exception as e:
    print("AutoRoundMLLM import failed:", str(e)[:200])

common = dict(model=model, tokenizer=tok, iters=ITERS, nsamples=NSAMPLES, seqlen=SEQLEN, layer_config=layer_config)
# attempts: (label, class, extra_kwargs). MLLM class + processor + quant_nontext_module=False is the primary path;
# text-only calib via dataset=calib (list[str]); fall back across kwarg spellings AutoRound's version may want.
attempts = []
if have_mllm:
    attempts += [
        ("MLLM W4A16 proc nontext=False txt",   "mllm", dict(scheme="W4A16", processor=proc, quant_nontext_module=False, dataset=calib)),
        ("MLLM W4A16 proc nontext=False",       "mllm", dict(scheme="W4A16", processor=proc, quant_nontext_module=False)),
        ("MLLM b4/gs128/sym proc nontext=False","mllm", dict(bits=4, group_size=128, sym=True, processor=proc, quant_nontext_module=False, dataset=calib)),
    ]
attempts += [
    ("base W4A16 proc nontext=False txt",   "base", dict(scheme="W4A16", processor=proc, quant_nontext_module=False, dataset=calib)),
    ("base W4A16 nontext=False txt",        "base", dict(scheme="W4A16", quant_nontext_module=False, dataset=calib)),
]
ar = None
for label, cls, kw in attempts:
    try:
        print(f"-- construct: {label}")
        ar = (AutoRoundMLLM if cls == "mllm" else AutoRound)(**common, **kw)
        print("   constructed OK via", cls)
        break
    except TypeError as e:
        print("   TypeError:", str(e)[:300])
    except Exception as e:
        print("   ERR:", type(e).__name__, str(e)[:300])
if ar is None:
    print("FATAL: could not construct AutoRound(MLLM) with any form"); sys.exit(3)

print("=== quantize() ===")
ar.quantize()

print(f"=== save_quantized(format=auto_round) -> {OUT} ===")
saved = False
for fmt in ("auto_round", "auto_round:auto_gptq", "auto_gptq"):
    try:
        ar.save_quantized(output_dir=OUT, format=fmt)
        print("SAVED ok format=", fmt); saved = True; break
    except Exception as e:
        print(f"save format={fmt} ERR:", type(e).__name__, str(e)[:300])
if not saved:
    # last resort: quantize_and_save one-shot (some versions only expose this)
    try:
        ar.quantize_and_save(output_dir=OUT, format="auto_round"); print("SAVED via quantize_and_save"); saved = True
    except Exception as e:
        print("quantize_and_save ERR:", type(e).__name__, str(e)[:300])
print("RESULT_Q8:", "DONE" if saved else "SAVE_FAILED")
