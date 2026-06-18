#!/usr/bin/env bash
# Why does google/gemma-4-12B-it route to the Transformers fallback instead of native
# vLLM gemma4? Show its declared architecture + attention dims (head_dim=256 vs reshape 8).
python3 - <<'PY'
import json
c=json.load(open('/mnt/vm_8tb/b70/models/google_gemma-4-12B-it/config.json'))
print("architectures:", c.get("architectures"))
print("model_type:", c.get("model_type"))
tc=c.get("text_config", c)
for k in ("num_attention_heads","num_key_value_heads","head_dim","hidden_size","sliding_window","num_hidden_layers"):
    print(f"  text.{k}:", tc.get(k))
print("top-level keys:", list(c.keys()))
PY
echo "=== vLLM native gemma4 archs (for comparison) ==="
echo "Gemma4ForCausalLM (gemma4.py), Gemma4ForConditionalGeneration (gemma4_mm.py)"
echo "=== DONE ==="
