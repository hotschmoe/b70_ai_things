"""Experimental: can a model load across BOTH Intel B70s (xpu:0 + xpu:1)?
Test A: transformers/accelerate multi-XPU device_map placement + a cross-card forward pass.
Test B: AutoRound on top of the multi-XPU model (no public precedent on Intel XPU).
The user only asked to *try* to get it loaded on two GPUs -- Test A is the deliverable.
"""
import os, traceback
from collections import Counter
import torch

print("=== ENV ===")
print("torch", torch.__version__)
print("xpu available:", torch.xpu.is_available(), "device_count:", torch.xpu.device_count())
for i in range(torch.xpu.device_count()):
    try:
        print(f"  xpu:{i} ->", torch.xpu.get_device_name(i))
    except Exception as e:
        print(f"  xpu:{i} name err: {e}")

PATH = os.environ.get("MODELP", "/models/Qwen_Qwen3-0.6B")
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(PATH, trust_remote_code=True)

print("\n=== TEST A: force a 2-way XPU split (tiny per-card cap so layers MUST spread) ===")
# Cap each XPU small so accelerate is forced to place layers on xpu:0 AND xpu:1 (+ cpu overflow).
mm = {0: "0.35GiB", 1: "0.35GiB", "cpu": "16GiB"}
print("from_pretrained(device_map='auto', max_memory=%s)" % mm)
placement_ok = False
model = None
try:
    model = AutoModelForCausalLM.from_pretrained(
        PATH, torch_dtype=torch.bfloat16, device_map="auto", max_memory=mm, trust_remote_code=True)
    dm = getattr(model, "hf_device_map", {})
    print("HF DEVICE MAP (module -> device):")
    for k, v in dm.items():
        print(f"  {k:40s} -> {v}")
    hist = Counter(str(v) for v in dm.values())
    print("DEVICE HISTOGRAM:", dict(hist))
    xpu_devs = sorted({str(v) for v in dm.values() if "xpu" in str(v)})
    print("DISTINCT XPU DEVICES USED:", xpu_devs)
    placement_ok = len(xpu_devs) >= 2
    # cross-card forward/generate
    ids = tok("The capital of France is", return_tensors="pt").input_ids
    first_dev = next((str(v) for v in dm.values() if "xpu" in str(v)), "xpu:0")
    ids = ids.to(first_dev)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=16, do_sample=False)
    print("CROSS-CARD GEN:", tok.decode(out[0], skip_special_tokens=True))
    print("RESULT_A:", "MULTI_XPU_PLACEMENT_OK" if placement_ok else "ONLY_ONE_XPU_USED")
except Exception as e:
    print("TEST_A_ERROR:", type(e).__name__, str(e)[:600])
    traceback.print_exc()

print("\n=== TEST B: AutoRound on the multi-XPU model (experimental) ===")
try:
    import auto_round
    print("auto_round version:", getattr(auto_round, "__version__", "?"))
    from auto_round import AutoRound
    # Strategy 1: hand AutoRound the comma device_map (its CUDA multi-GPU form) and see if it maps to XPU.
    for strat, kw in [
        ("device_map='0,1'", dict(device_map="0,1")),
        ("device_map='auto'", dict(device_map="auto")),
    ]:
        try:
            print(f"-- AutoRound strat: {strat}")
            ar = AutoRound(model=PATH, tokenizer=tok, bits=4, group_size=128,
                           iters=1, nsamples=8, seqlen=128, **kw)
            print("   AutoRound constructed OK; running a minimal quantize()...")
            ar.quantize()
            print("   RESULT_B:", strat, "QUANTIZE_RAN")
            break
        except Exception as e:
            print("   ", strat, "ERR:", type(e).__name__, str(e)[:400])
except Exception as e:
    print("AUTOROUND_IMPORT_ERR:", type(e).__name__, str(e)[:400])

print("\n=== DONE ===")
