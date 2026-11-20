#!/usr/bin/env python3
"""Real-text token-id prompt files for bench_cache.py --prompt-file (JSON array of ids), with the checkpoint's tokenizer.
usage: make_prompt_ids.py TOKENIZER_JSON CORPUS_FILE OUT_DIR N [N ...]   (CPU only)"""
import json, sys
from tokenizers import Tokenizer
tok = Tokenizer.from_file(sys.argv[1]); text = open(sys.argv[2], errors="ignore").read(); out = sys.argv[3]
ids = tok.encode(text).ids
print("corpus tokens", len(ids))
import os; os.makedirs(out, exist_ok=True)
for n in map(int, sys.argv[4:]):
    if len(ids) < n: raise SystemExit(f"corpus too short for {n}")
    json.dump(ids[:n], open(f"{out}/prompt_{n}.json", "w")); print("wrote", f"{out}/prompt_{n}.json")
