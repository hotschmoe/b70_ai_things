"""Collect canonical local evidence for the owned proxy, without submission."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.request import urlopen


def main():
    out = Path(sys.argv[1])
    repo = Path(__file__).resolve().parents[3]
    quant = os.environ['QUANT']
    depth = int(os.environ['MTP_TOKENS'])
    port = os.environ['PORT']
    base = f'http://127.0.0.1:{port}'
    with urlopen(base + '/version', timeout=30) as response:
        version = json.load(response)['version']
    manifest = out / 'source-before.json'
    identity = json.loads(manifest.read_text())
    command = out / 'lmx-command.txt'
    flags = ('QUANT', 'MTP_TOKENS', 'CARD', 'PORT', 'B70_X5_GDN_BF16',
             'B70_X5_EVENTLESS', 'B70_X5_EVENTLESS_LAZY', 'B70_X5_PACK_LINEAR',
             'B70_X5_Q8_LINEAR', 'B70_X5_CANONICAL', 'B70_X5_BATCH_LINEAR',
             'B70_X5_Q8_FAST', 'B70_X5_GDN_MIXED', 'B70_X5_GDN_RAGGED',
             'B70_D3_GDN_SPEC', 'X5_LMX')
    text = ' '.join(f'{k}={os.environ.get(k, "0")}' for k in flags)
    text += ' B70_FUSE_SHARED=1 B70_D3_GDN=0 DTYPE=bfloat16 MAX_LEN=4096'
    text += ' B70_GGUF_SO=' + str(repo / 'vllm/flashnext/plugins/b70_gguf_xpu/b70_gguf_xpu/_C.o3g.so')
    text += ' ' + str(repo / 'vllm/flashnext/harness/x5_serve_proxy.sh')
    text += '\n# FULL_DECODE_ONLY sizes1,2,3,4,6,8; FLASH_ATTN; no prefix cache; TP1.'
    text += '\n# Source/library SHA256 manifest: ' + str(manifest) + '\n'
    command.write_text(text, encoding='ascii')
    lmx = repo / 'vllm/flashnext/localmaxxing'
    args = [sys.executable, str(lmx / 'lmx_vllm.py'), '--measure-local',
            '--base-url', base, '--served-model', 'hotschmoe-dd',
            '--alias', f'qwen3.6-35b-a3b-{quant.lower()}-x5-mtp{depth}',
            '--hf-id', 'unsloth/Qwen3.6-35B-A3B-MTP-GGUF',
            '--quantization', 'Unsloth-Dynamic-' + quant.removeprefix('UD-'),
            '--hardware', str(lmx / 'hardware_card1.json'),
            '--engine-version', version, '--engine-commit', identity['fork_head'],
            '--engine-build', 'x5-dirty-' + hashlib.sha256(manifest.read_bytes()).hexdigest()[:16],
            '--command-file', str(command), '--mode', 'mtp' if depth else 'no-spec',
            '--out', str(out / 'localmaxxing')]
    if depth:
        args += ['--spec-tokens', str(depth)]
    subprocess.run(args, check=True)
    config = json.loads((out / 'localmaxxing/server-config.json').read_text())
    max_len = int(config['model_config']['max_model_len'])
    for payload_file in (out / 'localmaxxing').glob('*/payload.json'):
        payload = json.loads(payload_file.read_text())
        payload['contextLength'] = max_len
        payload['modelRevision'] = '5bc3e238d916f48a861bac2f8a1990a0e9b7e98d'
        payload['engineFlags']['mtpEnabled'] = bool(depth)
        payload['notes'] = (
            'X5 single physical card 1, BF16, FULL_DECODE_ONLY, FLASH_ATTN. '
            'GGUF revision pinned; HF config/tokenizer revision '
            '995ad96eacd98c81ed38be0c5b274b04031597b0. '
            'Dirty source/library identity in source-before.json; '
            'canonical local measurement, no submission.')
        payload_file.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + '\n', encoding='ascii')


if __name__ == '__main__':
    main()
