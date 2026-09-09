#!/usr/bin/env python3
"""Leased startup gate for the user-requested full-feature FP8 trial."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

REPO = Path(os.environ.get('B70_REPO', Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(REPO / 'vllm/fp8'))
from kv_campaign_probe import request


def check_features(manifest):
    a = manifest['args']
    if (a['mtp'] != 3 or a['eager'] or a['prefix_off'] or
            a['kv_dtype'] != 'fp8_e4m3' or a['offload_gib'] != 0):
        raise RuntimeError('requested serving features are not all enabled')
    command = manifest['command']
    if 'VLLM_XPU_ENABLE_XPU_GRAPH=1' not in command:
        raise RuntimeError('XPU graphs disabled')
    graph = json.loads(command[command.index('--compilation-config') + 1])
    if graph.get('mode') in (0, 'NONE') or graph.get('cudagraph_mode') in (0, 'NONE', None):
        raise RuntimeError('graph compilation configuration disabled')


def counter(text, metric):
    rows = re.findall(r'^' + re.escape(metric) + r'(?:\{[^\n]*\})? ([0-9.eE+\-]+)$', text, re.M)
    if not rows:
        raise RuntimeError('required metric missing: ' + metric)
    return sum(float(v) for v in rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--server-root', type=Path, required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--base', default='http://127.0.0.1:18124')
    p.add_argument('--prior-validation', type=Path)
    args = p.parse_args()
    root = args.server_root
    manifest = json.loads((root / 'manifest.json').read_text())
    check_features(manifest)
    startup = (root / 'server.log').read_text()
    if 'Graph capturing finished' not in startup:
        raise RuntimeError('no completed graph capture recorded')
    sizes = re.findall(r'GPU KV cache size: ([0-9,]+) tokens', startup)
    if not sizes or int(sizes[-1].replace(',', '')) < 800000:
        raise RuntimeError('actual logical KV pool is below the requested 800K')
    capacity = int(sizes[-1].replace(',', ''))
    loaded = [r for path in root.glob('kv-load-*.json')
              for r in json.loads(path.read_text()).values()]
    if len(loaded) != 34 or any(r['query_quantized'] for r in loaded):
        raise RuntimeError('expected 17 calibrated layers per rank and FP16 queries')
    out = root / 'trial-validation'
    out.mkdir(exist_ok=False)
    results = {}
    if args.prior_validation:
        prior = args.prior_validation
        old = json.loads((prior / 'manifest.json').read_text())
        check_features(old)
        assert old['image'] == manifest['image'] and old['model'] == manifest['model']
        assert old['source_sha256'] == manifest['source_sha256']
        assert (prior / 'exit.rc').read_text().strip() == '0'
        for flag in ('--compilation-config', '--max-model-len', '--max-num-seqs',
                     '--max-num-batched-tokens', '--gpu-memory-utilization', '--dtype',
                     '--tensor-parallel-size', '--quantization', '--speculative-config'):
            assert old['command'][old['command'].index(flag) + 1] == manifest['command'][manifest['command'].index(flag) + 1]
        old_loaded = [r for path in prior.glob('kv-load-*.json') for r in json.loads(path.read_text()).values()]
        assert {r['artifact_sha256'] for r in old_loaded} == {r['artifact_sha256'] for r in loaded}
        old_results = json.loads((prior / 'trial-validation/results.json').read_text())
        assert all(old_results[name]['rc'] == 0 for name in ('01-quality', '03-tools', '04-long4'))
        assert old_results['02-guides']['trial_coherence_gate']
        for name in ('01-quality', '04-long4'):
            assert json.loads((prior / 'trial-validation' / name / 'summary.json').read_text())['passed']
        old_before = (prior / 'trial-validation/metrics-before.txt').read_text()
        old_after = (prior / 'trial-validation/metrics-after.txt').read_text()
        assert counter(old_after, 'vllm:num_preemptions_total') == counter(old_before, 'vllm:num_preemptions_total')
        results['prior_validation'] = dict(root=str(prior), checks=old_results,
            note='Same image, sources, scales and serving configuration; prior workload evidence retained.')
    def run(name, script, flags, timeout, guide=False):
        dest = out / name
        cmd = ['python3', str(REPO / 'vllm/fp8' / script), '--base', args.base,
               '--model', args.model, '--out', str(dest), *flags]
        with (out / (name + '.log')).open('w') as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=timeout).returncode
        results[name] = dict(rc=rc)
        if guide and (dest / 'summary.json').exists():
            summary = json.loads((dest / 'summary.json').read_text())
            results[name]['summary'] = summary
            # The preserved MTP3 reference also changes long wording. Keep
            # that failure visible; coherent nonidentical prose is allowed
            # for this user trial, not a deterministic shelf qualification.
            accepted = summary['rows_passed'] and summary['rows'] == 2
            results[name]['trial_coherence_gate'] = accepted
        else:
            accepted = rc == 0
        (out / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
        if not accepted:
            raise RuntimeError('trial validation failed: ' + name)
    before = request(args.base, '/metrics')
    run('01-quality', 'kv_campaign_probe.py', ['--concurrency', '4'], 900)
    if not args.prior_validation:
        run('02-guides', 'kv_campaign_probe.py',
            ['--mode', 'decode', '--rounds', '2', '--concurrency', '1'], 1500, guide=True)
        run('03-tools', 'kv_campaign_agent_probe.py',
            ['--shared-cache', '--stream', '--turns', '4', '--records', '180'], 1800)
        run('04-long4', 'kv_campaign_probe.py',
            ['--mode', 'long', '--tokens', '196000', '--rounds', '1',
             '--concurrency', '4', '--timeout', '1800'], 2000)
    run('05-prefix-reuse', 'kv_campaign_probe.py',
        ['--mode', 'reuse', '--tokens', '32000', '--reuse-sequence', '0,1,0'], 1800)
    after = request(args.base, '/metrics')
    (out / 'metrics-before.txt').write_text(before)
    (out / 'metrics-after.txt').write_text(after)
    preemptions = counter(after, 'vllm:num_preemptions_total') - counter(before, 'vllm:num_preemptions_total')
    hits = counter(after, 'vllm:prefix_cache_hits_total') - counter(before, 'vllm:prefix_cache_hits_total')
    accepted = counter(after, 'vllm:spec_decode_num_accepted_tokens_total') - counter(before, 'vllm:spec_decode_num_accepted_tokens_total')
    if preemptions != 0 or hits <= 0:
        raise RuntimeError('unexpected preemption or no observed prefix reuse')
    if accepted <= 0:
        raise RuntimeError('no accepted MTP draft tokens observed')
    (out / 'PASSED.json').write_text(json.dumps(dict(
        user_trial=True, production_qualified=False, preemptions=preemptions,
        prefix_hit_tokens=hits, accepted_mtp_tokens=accepted,
        calibrated_rank_layers=len(loaded), logical_kv_tokens=capacity,
        manifest_sha256=hashlib.sha256((root / 'manifest.json').read_bytes()).hexdigest(),
        note='Bounded startup gate; four long retrievals, not four sustained 8K continuations.'), indent=2) + '\n')


if __name__ == '__main__':
    main()
