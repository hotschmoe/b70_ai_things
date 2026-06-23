#!/usr/bin/env python3
# Compare body-weight key layout of W8A8 (garbage) vs W4A8 (coherent) vs W4A16 (coherent) + the text-only configs.
import glob, os, json
from collections import Counter
from safetensors import safe_open

DIRS = {
 "W8A8(garbage)": "/models/Qwen3.6-27B-W8A8-sqgptq-mtp-graft",
 "W4A8(coherent)": "/models/Qwen3.6-27B-W4A8-sqgptq-prepacked",
 "W4A16(coherent)": "/models/Qwen3.6-27B-W4A16",
}
def prefixes(d):
    c = Counter()
    for f in sorted(glob.glob(os.path.join(d,"*.safetensors"))):
        with safe_open(f, framework="pt", device="cpu") as sf:
            for k in sf.keys():
                # bucket by top-2 segments
                parts = k.split(".")
                c[".".join(parts[:2])] += 1
    return c
for name, d in DIRS.items():
    if not os.path.isdir(d):
        print(f"\n## {name}: DIR MISSING {d}"); continue
    print(f"\n## {name}  ({d})")
    pc = prefixes(d)
    for p,n in sorted(pc.items(), key=lambda x:-x[1])[:12]:
        print(f"   {n:6d}  {p}")
    cfg = os.path.join(d,"config.json")
    if os.path.exists(cfg):
        j = json.load(open(cfg))
        arch = j.get("architectures")
        mt = j.get("model_type")
        has_text = "text_config" in j
        has_lang = "language_model" in str(j.get("architectures",""))
        print(f"   config: arch={arch} model_type={mt} text_config_present={has_text}")
        # is it text-only flattened or VLM-nested?
        for kk in ("vision_config","mtp_num_hidden_layers","tie_word_embeddings"):
            if kk in j: print(f"     {kk}={j[kk] if kk!='vision_config' else '<present>'}")
print("\nDONE")
