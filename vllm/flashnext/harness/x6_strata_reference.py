"""Generate raw-token references exclusively through Strata's engine API.

Run in the Strata image with the pair GPU lease; never starts a service.
The config is a research copy, with its own log and output paths.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import threading

p = argparse.ArgumentParser()
p.add_argument('--config', type=Path, required=True)
p.add_argument('--prompts', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--check-tokenizer', action='store_true')
a = p.parse_args()
cfg = json.loads(a.config.read_text())
src = Path(cfg['cwd'])
sys.path[:0] = [str(src), str(src / 'tools')]
from serve.server import StrataEngine, engine_args, child_env
import strata_tokenizer as ST

tpath = Path(cfg['tokenizer'])
vocab = json.loads((tpath / 'vocab.json').read_text())
tokens = [None] * len(vocab)
for text, index in vocab.items():
    tokens[index] = text
tok = ST.Tokenizer(tokens, (tpath / 'merges.txt').read_text().split('\n'),
                   json.loads((tpath / 'token_type.json').read_text()))
fixtures = json.loads(a.prompts.read_text())
for f in fixtures:
    assert list(tok.encode(f['prompt'])) == f['token_ids'], 'Strata/HF tokenizer mismatch'
if a.check_tokenizer:
    print('X6_STRATA_TOKENIZER_OK', [len(f['token_ids']) for f in fixtures])
    raise SystemExit(0)
engine = StrataEngine(cfg['exe'], engine_args(cfg), cwd=cfg['cwd'], log=cfg['log'], env=child_env(cfg))
rows = []
try:
    assert engine.alive() and engine.max_context >= 32768
    for f in fixtures:
        generated = [i for i in engine.generate(f['token_ids'], f['max_tokens'], {'temperature': 0}, threading.Event()) if i is not None]
        visible = []
        for i in generated:
            if i in (248044, 248046):
                break
            visible.append(i)
        text = tok.decode(visible)
        row = dict(prompt_sha256=f['prompt_sha256'], prompt_tokens=len(f['token_ids']),
                   text=text, output_ids=visible, raw_output_ids=generated,
                   text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                   engine_info=engine.info, engine_last=engine.last,
                   config=str(a.config), exe_sha256=hashlib.sha256(Path(cfg['exe']).read_bytes()).hexdigest())
        rows.append(row)
        a.out.write_text(json.dumps(rows, indent=2, ensure_ascii=True) + '\n')
        print('X6_STRATA_REFERENCE', row['prompt_tokens'], len(visible), row['text_sha256'], flush=True)
finally:
    engine.close()
print('X6_STRATA_REFERENCES_DONE', flush=True)
