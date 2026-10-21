#!/usr/bin/env python3
"""L1: turn a measured lmx_vllm.py packet into the final submission payload.

usage: l1_finalize.py <run-dir> <kind: 35b|flashnext> <dest-dir>
Adds contextLength (live max_model_len), modelRevision, engineFlags.mtpEnabled=false,
and notes (gate outcome + provenance). Refuses packets whose validation.json is not valid.
Never submits.
"""
import json
import sys
from pathlib import Path

run, kind, dest = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
dest.mkdir(parents=True, exist_ok=True)
config = json.loads((run / 'localmaxxing/server-config.json').read_text())
max_len = int(config['model_config']['max_model_len'])
gate = json.loads((run / 'gate.json').read_text())
checks = gate['checks']
failed = [k for k, v in checks.items() if not v['pass']]
gate_text = ('promotion gate PASS (identity, greedy 17x19/23x29, exact token repeat C1+C2 x96 tokens, '
             '8K/32K three-needle recall)' if not failed else
             'promotion gate: all checks pass except ' + ', '.join(failed) + ' (detail follows)')
extra = (run / 'gate-note.txt').read_text().strip() if (run / 'gate-note.txt').exists() else ''
commit = (run / 'engine.txt').read_text().split('=')[1].strip()

if kind == '35b':
    revision = '5bc3e238d916f48a861bac2f8a1990a0e9b7e98d'
    head = ('One Intel Arc Pro B70 (32 GB, Xe2, kernel 7.1, Compute Runtime 26.22). vLLM 0.32 main + thin '
            'XPU fork (local branch b70/lmx-20261021, commit ' + commit[:12] + ', based on vLLM main 5fd62688, '
            'not pushed) with our out-of-tree GGUF XPU plugin (custom SYCL MoE/GEMV kernels for Q4_K/IQ '
            'formats, BF16 GDN kernel, Q8 linear path), BF16 activations, full-decode XPU graphs. '
            'Unsloth Dynamic GGUF loaded by the plugin (MTP head present in the file but unused). ')
else:
    revision = 'main'
    head = ('Two Intel Arc Pro B70 (32 GB each, Xe2, kernel 7.1, Compute Runtime 26.22). vLLM 0.32 main + thin '
            'XPU fork (local branch b70/lmx-20261021, commit ' + commit[:12] + ', based on vLLM main 5fd62688, '
            'not pushed) with out-of-tree plugins: GGUF IQ/Q2_0 SYCL kernels and an expert-parallel sidecar '
            '(card 0 runs the model and 138 of 512 experts per layer, card 1 the other 374; one vLLM worker, '
            'no oneCCL), fused decode hyper-connection/PLE/top-k kernels, HC weights as Q8_0 copies, '
            'PLE table in pinned host RAM, full-decode XPU graphs, BF16 activations. ')
common = ('No speculative decoding: no --speculative-config, server config speculative_config=null, '
          'zero draft tokens in the server counters, one token per forward pass. '
          f'max_model_len {max_len}, prefix cache off, greedy (temperature 0), thinking on (server default; '
          'reasoning tokens counted as output), 512 max tokens, 1 warmup + 3 timed iterations with a new '
          'leading cache-bust nonce, median decode-rate sample. Client-observed HTTP streaming '
          '(lmx remote method); engineTimingsRaw holds vLLM usage plus isolated Prometheus counter deltas '
          '(vLLM has no per-request timings object). ' + gate_text + '. ')
notes = head + common + (extra + ' ' if extra else '')
for packet in sorted((run / 'localmaxxing').glob('no-spec-*/payload.json')):
    suite = packet.parent.name.removeprefix('no-spec-')
    validation = json.loads((packet.parent / 'validation.json').read_text())
    if validation.get('valid') is not True:
        raise SystemExit(f'{packet}: validation.json not valid')
    payload = json.loads(packet.read_text())
    payload['contextLength'] = max_len
    payload['modelRevision'] = revision
    payload['engineFlags']['mtpEnabled'] = False
    payload['notes'] = notes[:2000]
    out = dest / f'payload_{suite}.json'
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=True) + '\n', encoding='ascii')
    print(out, payload['tokSOut'], payload['ttftMs'], payload['outputTokens'], len(payload['notes']))
