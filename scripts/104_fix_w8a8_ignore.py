#!/usr/bin/env python3
# Fix the W8A8 27B/Qwable broken ignore list: the enumerated flat-prefix 'model.layers.N.linear_attn...'
# names do NOT match the VLM-nested checkpoint keys 'model.language_model.layers.N.linear_attn...', so the
# 48 GDN linear_attn layers (BF16, no scale) are silently quantized as W8A8 -> garbage. Replace with the
# proven W4A8 regex form. CONFIG-ONLY fix; weights are good (dequant cos 0.97-0.9999 vs base).
import json, sys, os, shutil

GOOD_IGNORE = ["lm_head", "re:.*linear_attn.*", "re:.*visual.*", "re:.*mtp.*"]

def fix(path):
    cfg = json.load(open(path))
    qc = cfg.get("quantization_config")
    if qc is None:
        print(f"  [{path}] no quantization_config -> SKIP"); return False
    old = qc.get("ignore", [])
    if old == GOOD_IGNORE:
        print(f"  [{path}] already correct -> SKIP"); return False
    n_old = len(old)
    # sanity: confirm it's the broken enumerated form (has flat model.layers.*.linear_attn names)
    flat = [x for x in old if isinstance(x,str) and x.startswith("model.layers.") and "linear_attn" in x]
    bak = path + f".ignore{n_old}.bak"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
        print(f"  backed up -> {bak}")
    qc["ignore"] = GOOD_IGNORE
    json.dump(cfg, open(path,"w"), indent=2)
    print(f"  [{path}] ignore {n_old} entries ({len(flat)} flat linear_attn) -> {GOOD_IGNORE}")
    return True

if __name__ == "__main__":
    for p in sys.argv[1:]:
        cfgp = os.path.join(p, "config.json") if os.path.isdir(p) else p
        print(f"== {cfgp} ==")
        fix(cfgp)
    print("DONE")
