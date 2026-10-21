"""Q1 Torch packed-key selector and populated real-shape QSA chain. GPU lease required."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from oracle import CANARY, cases, reference

HERE = Path(__file__).resolve().parent
BUILD = Path(os.environ.get('Q1_TOPK_ROOT', '/mnt/vm_8tb/b70/build/vllm-flashnext/q1/torch-r1'))
REPO = Path('/mnt/vm_8tb/github/b70_ai_things')
UT = '/mnt/vm_8tb/b70/build/strata-ab/pti-gpu/tools/unitrace/build/unitrace'

def log(msg):
    print(f'Q1_TOPK {time.time_ns()} {msg}', flush=True)

def ut(cmd):
    session = os.environ.get('Q1_UT_SESSION')
    if session:
        r = subprocess.run([UT, '--' + cmd, session], capture_output=True)
        if r.returncode:
            raise RuntimeError(f'unitrace {cmd} rc={r.returncode}')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--repeats', type=int, default=12)
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    import torch
    from b70_qsa_xpu import ops
    from b70_qsa_xpu.batch import QsaBatch
    from b70_qsa_xpu.state import Geom
    torch.set_num_threads(2)
    assert torch.xpu.device_count() == 1, 'one-card pin required'
    native = ops.topk_batch
    from selector import PackedTopk
    selectors = {}
    def parallel(scores, ids, step):
        key = (scores.shape[1], ids.shape[1], str(scores.device))
        if key not in selectors:
            selectors[key] = PackedTopk(*key)
        return selectors[key](scores, ids, step)
    variants = {'native': native, 'torch': parallel}
    result = {'device': torch.xpu.get_device_name(0), 'torch': torch.__version__,
              'candidate_sha256': hashlib.sha256((HERE/'selector.py').read_bytes()).hexdigest(),
              'scope': 'deterministic populated component; no model or loaded-model VRAM pressure',
              'correctness': [], 'shapes': []}
    ut('pause')

    # Includes raw NaN payloads, signed zero, infinities, clamp and inactive rows.
    for name, bits, n, width in cases():
        host = np.zeros((4, max(2051, len(bits))), dtype=np.uint32)
        host[:, :len(bits)] = bits
        scores = torch.from_numpy(host.view(np.float32).copy()).to('xpu')
        step_host = np.zeros((4, 4), dtype=np.int32)
        step_host[:2, 1] = min(n, len(bits))
        step_host[:2, 3] = width
        step = torch.from_numpy(step_host).to('xpu')
        expected = np.full((4, 2051), CANARY, dtype=np.int32)
        expected[:2] = reference(bits, n, width)
        for variant, fn in variants.items():
            ids = torch.full((4, 2051), CANARY, dtype=torch.int32, device='xpu')
            fn(scores, ids, step)
            np.testing.assert_array_equal(ids.cpu().numpy(), expected,
                                          err_msg=f'{variant}/{name}')
        result['correctness'].append(name)
    log(f"NUMERICAL_PASS cases={len(result['correctness'])} variants=2")

    # Real serving capacity=4 even for active M1/M2. Full legal page bytes,
    # nonzero signed KV codes, positive fp16 scales, populated f32 index rows.
    geom = Geom(24, 2, 256, 128, 4, 128, 4, 2048)
    for context in (4096, 16384, 32768):
        for active in (1, 2):
            nblocks = (context + 1 + 2751)//2752
            table = np.arange(4*nblocks, dtype=np.int32).reshape(4, nblocks)
            b = QsaBatch(geom, table, 2752, torch.device('xpu:0'), max_cells=147456)
            rng = np.random.default_rng(context + active)
            pages = np.zeros(tuple(b.pages.shape), dtype=np.uint8)
            lay = b.layout
            pages[:, :lay['v_scale_off'] + 2*2*2752*4] = 0
            for off in (lay['k_codes_off'], lay['v_codes_off']):
                pages[:, off:off + 2*2752*256] = rng.integers(
                    -127, 128, (len(pages), 2*2752*256), dtype=np.int16).astype(np.int8).view(np.uint8)
            for off in (lay['k_scale_off'], lay['v_scale_off']):
                scales = rng.uniform(.001, .02, (len(pages), 2*2752*4)).astype(np.float16)
                pages[:, off:off + scales.shape[1]*2] = scales.view(np.uint8)
            pooled = rng.normal(0, .2, (len(pages), lay['n_states']*128)).astype(np.float32)
            pages[:, lay['pooled_off']:] = pooled.view(np.uint8)
            b.pages.copy_(torch.from_numpy(pages))
            del pages, pooled
            for field in ('q', 'k', 'v', 'q_idx', 'raw', 'q_full', 'tail', 'dead'):
                tensor = getattr(b, field)
                tensor.copy_(torch.from_numpy(rng.normal(0, .1, tuple(tensor.shape)).astype(np.float32)))
            b.ids.fill_(CANARY)
            b.pos[:active] = context - 1
            log(f'SHAPE context={context} active={active} capacity=4 pages={tuple(b.pages.shape)}')

            def capture(fn):
                for _ in range(2):
                    fn()
                torch.xpu.synchronize()
                graph = torch.xpu.XPUGraph()
                with torch.xpu.graph(graph):
                    fn()
                graph.replay()
                torch.xpu.synchronize()
                return graph

            def timed(graph, repeats):
                start = torch.xpu.Event(enable_timing=True)
                end = torch.xpu.Event(enable_timing=True)
                start.record()
                for _ in range(repeats):
                    graph.replay()
                end.record()
                end.synchronize()
                return start.elapsed_time(end)/repeats

            row = {'context': context, 'active': active, 'capacity': 4,
                   'page_shape': list(b.pages.shape), 'block_size': 2752,
                   'geometry': vars(geom), 'timing_ms': {}, 'op_events_ms': {},
                   'op_boundaries_ms': {}}
            native_snap = None
            full_graphs = {}
            for variant, topk in variants.items():
                ops.topk_batch = topk
                full = capture(b.run_decode)
                full_graphs[variant] = full
                ids0 = b.ids.cpu().numpy().copy()
                score_bits = b.scores.cpu().numpy().view(np.uint32).copy()
                steps = b.step.cpu().numpy()
                expected = np.full((4, 2051), CANARY, dtype=np.int32)
                for qi in range(active):
                    expected[qi] = reference(score_bits[qi], int(steps[qi,1]), int(steps[qi,3]))
                np.testing.assert_array_equal(ids0, expected)
                snap = b.gated.cpu().numpy().copy()
                if native_snap is None:
                    native_snap = snap
                else:
                    np.testing.assert_array_equal(snap, native_snap)
                top = capture(lambda: topk(b.scores, b.ids, b.step))
                row['timing_ms'][variant] = {'topk': timed(top, a.repeats),
                                             'chain': timed(full, a.repeats)}
                # Changed-input replay uses the already captured topk graph.
                changed = np.flip(score_bits, axis=1).copy()
                b.scores.copy_(torch.from_numpy(changed.view(np.float32)))
                b.ids.fill_(CANARY)
                top.replay()
                torch.xpu.synchronize()
                for qi in range(active):
                    expected[qi] = reference(changed[qi], int(steps[qi,1]), int(steps[qi,3]))
                np.testing.assert_array_equal(b.ids.cpu().numpy(), expected)
                # Restore chain-produced scores and trace uninstrumented graph.
                full.replay()
                torch.xpu.synchronize()
                log(f'TRACE_BEGIN context={context} active={active} variant={variant}')
                ut('resume')
                for _ in range(3):
                    full.replay()
                torch.xpu.synchronize()
                ut('pause')
                log(f'TRACE_END context={context} active={active} variant={variant}')

                # Event boundaries are an auxiliary diagnostic. No host wait
                # between ops. Native trace above has no per-op event overlay.
                marks = []
                saved = {}
                def wrap(name, fn):
                    def call(*args, **kwargs):
                        start = torch.xpu.Event(enable_timing=True)
                        end = torch.xpu.Event(enable_timing=True)
                        start.record()
                        value = fn(*args, **kwargs)
                        end.record()
                        marks.append((name, start, end))
                        return value
                    return call
                names = ('write_step_batch', 'kv_append_batch', 'indexer_append_batch',
                         'index_scores_batch', 'topk_batch', 'kv_gather_batch',
                         'attend_batch', 'gate_batch')
                try:
                    for name in names:
                        saved[name] = getattr(ops, name)
                        setattr(ops, name, wrap(name, saved[name]))
                    saved_commit = b._commit_length
                    b._commit_length = wrap('commit_length', saved_commit)
                    chain_start = torch.xpu.Event(enable_timing=True)
                    chain_end = torch.xpu.Event(enable_timing=True)
                    chain_start.record()
                    b.run_decode()
                    chain_end.record()
                    torch.xpu.synchronize()
                finally:
                    for name, fn in saved.items():
                        setattr(ops, name, fn)
                    b._commit_length = saved_commit
                row['op_events_ms'][variant] = {
                    name: start.elapsed_time(end) for name, start, end in marks}
                row['op_boundaries_ms'][variant] = {
                    'chain': chain_start.elapsed_time(chain_end),
                    'events': [{'op': name,
                                'start': chain_start.elapsed_time(start),
                                'end': chain_start.elapsed_time(end),
                                'prior_gap': (chain_start if i == 0 else marks[i-1][2]).elapsed_time(start)}
                               for i, (name, start, end) in enumerate(marks)]}
                log(f'MEASURE context={context} active={active} variant={variant} '+
                    json.dumps(row['timing_ms'][variant], sort_keys=True))
                del full, top
            # Same captured graphs, updated live inputs and a new append slot.
            # All three variants must observe the changes variants must agree exactly.
            b.q.mul_(-.5)
            b.q_idx.add_(.3)
            b.k.add_(.02)
            b.v.mul_(.75)
            b.raw.add_(.125)
            b.pos[:active] = context
            changed_snap = None
            for variant, graph in full_graphs.items():
                graph.replay()
                torch.xpu.synchronize()
                scores = b.scores.cpu().numpy().view(np.uint32)
                steps = b.step.cpu().numpy()
                expected = np.full((4, 2051), CANARY, dtype=np.int32)
                for qi in range(active):
                    expected[qi] = reference(scores[qi], int(steps[qi,1]), int(steps[qi,3]))
                np.testing.assert_array_equal(b.ids.cpu().numpy(), expected)
                snap = b.gated.cpu().numpy().copy()
                if changed_snap is None:
                    changed_snap = snap
                else:
                    np.testing.assert_array_equal(snap, changed_snap)
            row['changed_chain_replay_exact'] = True
            del full_graphs, graph, changed_snap
            ops.topk_batch = native
            result['shapes'].append(row)
            (out/'result.json').write_text(json.dumps(result, indent=2) + '\n', encoding='ascii')
            del b, native_snap
            gc.collect()
            torch.xpu.empty_cache()
    result['verdict'] = 'COMPONENT_PASS; no serving/fidelity/loaded-liveness claim'
    (out/'result.json').write_text(json.dumps(result, indent=2) + '\n', encoding='ascii')
    log('PASS all numerical IDs/canaries, changed-input replay, exact gated output')

if __name__ == '__main__':
    main()
