#!/usr/bin/env python3
"""Extract the pinned author's standalone R276 invocation without shell execution."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

SOURCE_COMMIT = '54aefaf067b422d507f51f1e362efeecc58668ba'
IMAGE = 'sha256:521eb277c0733f8c2ce47aea1bb98ed576c6f1ad63bf5baf22d38fc07abf54ad'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--profile', choices=['strict', 'depth', 'daily', 'daily-prefix-off'], default='strict')
    p.add_argument('--prefill-batch', type=int, choices=[4096, 8192, 32768],
                   help='Daily-profile prefill budget; default 32768')
    args = p.parse_args()
    if args.prefill_batch is not None and args.profile not in ('daily', 'daily-prefix-off'):
        p.error('--prefill-batch is only supported for daily profiles')
    commit = subprocess.check_output(['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip()
    assert commit == SOURCE_COMMIT, commit
    guide = args.source / 'repro/qwen38-27b-autoround-int4-b70/README.md'
    content = guide.read_text()
    section = content.split('### Standalone `docker run` (no lab scripts)', 1)[1]
    block = section.split('docker run --rm', 1)[1].split('\n```', 1)[0]
    words = shlex.split('docker run --rm' + block.replace('\\\n', ''), comments=True)
    image_index = next(i for i, word in enumerate(words) if word.startswith('ghcr.io/'))
    assert words[image_index].endswith('@' + IMAGE)
    env = [words[i + 1] for i, word in enumerate(words[:image_index]) if word == '--env']
    cmd = words[image_index + 1:]
    assert not any('$' in word for word in cmd + env)
    changes = {}
    def setarg(key, value):
        i = cmd.index(key) + 1
        changes[key] = {'published': cmd[i], 'local': str(value)}
        cmd[i] = str(value)
    if args.profile == 'depth':
        setarg('--max-model-len', 33024)
        setarg('--max-num-batched-tokens', 4096)
    elif args.profile in ('daily', 'daily-prefix-off'):
        setarg('--max-model-len', 200000)
        setarg('--max-num-seqs', 4)
        setarg('--max-num-batched-tokens', args.prefill_batch or 32768)
        setarg('--gpu-memory-utilization', .96)
        if args.profile == 'daily':
            cmd[cmd.index('--no-enable-prefix-caching')] = '--enable-prefix-caching'
            changes['prefix-cache'] = {'published': False, 'local': True}
        cmd += ['--enable-auto-tool-choice', '--tool-call-parser', 'qwen3_coder',
                '--reasoning-parser', 'qwen3', '--default-chat-template-kwargs',
                '{"enable_thinking":true,"reasoning_effort":"xhigh"}']
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'Config.json').write_text(json.dumps({'Env': env, 'Cmd': cmd}, indent=2) + '\n')
    mounts = [{'Source': str(args.model.resolve()), 'Destination': '/model', 'RW': False},
              {'Source': '/unused-campaign-cache', 'Destination': '/root/.cache/vllm', 'RW': True}]
    (args.out / 'Mounts.json').write_text(json.dumps(mounts, indent=2) + '\n')
    manifest = {'source_commit': commit, 'guide_sha256': hashlib.sha256(guide.read_bytes()).hexdigest(),
                'image': IMAGE, 'profile': args.profile, 'changes': changes,
                'published_memory_gib': 12, 'published_memory_swap_gib': 16,
                'local_endpoint_port': 18125, 'source': str(args.source.resolve())}
    (args.out / 'recipe-identity.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
