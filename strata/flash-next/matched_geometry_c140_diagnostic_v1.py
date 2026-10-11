#!/usr/bin/env python3
"""Explicit new diagnostic profiles; original C140 functions/globals unchanged."""
import argparse
import ast
import copy
from pathlib import Path
import sys
import c1_serve_controller_combined_v140_v3 as original
from strict_registry_yaml_v3 import parse

ORIGINAL_SHA = '5daa1f2d0c9d8c147202724314bc013889dfaf1a102fc04fb6ea021c454d78b4'
ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT/'evals/configs/models.yaml'
OLD_REGISTRY = ROOT/'strata/flash-next/matched-geometry-registry171-snapshot-v1.yaml'
OLD_REGISTRY_SHA = 'fec00c10a7fc29f882217dc8ceb0309616966a13dd81e0cef62da4b0c9c5a50c'
DELTA = ROOT/'strata/flash-next/matched-geometry-model-registry-proposal-v1.yaml'
SUFFIX = '-srcintegrated25-40-api22v2-prefix30-mirror31-currentPLE32-input33-lastwindow34-prompt35-hosttrace36-batchstamp37-fullcache38-memory39-qsa40-OFF-'


def profile(name):
    original.require(name in ('matched-one', 'matched-pair'), 'Explicit matched diagnostic profile required')
    pair = name == 'matched-pair'
    return {'cards': [0, 1] if pair else [0], 'context': 2048, 'prefill': 64, 'ple_rows': 65536,
        'segmented': True, 'one_card_profile': 'one-card-segmented',
        'env': {'STRATA_STAGE_MIRRORS': '1', 'STRATA_STAGE_MIRROR_SEGMENT_MIB': '1024'},
        'extra': (['--layer-split', '32', '--split-device', '1', '--trim-stage-weights'] if pair else [])+
                 ['--adapt-every', '0', '--no-prefill-borrow'],
        'alias': 'qwen3.8-flash-next-Unsloth-UD-Q4_K_XL-strata-native-source-hc-ple-mtp0-c1-fp16kv-ctx2048-prefill64-ple65536-seg1024-adapt0-borrow0-matchedgeometry1'+SUFFIX+
                 ('gpu0-1-split32-16' if pair else 'gpu0')}


def config_gate(name, cfg, runtime):
    p = profile(name)
    common = original.read(ROOT/'strata/flash-next/native-hc-launch-plan.json')
    args = list(common['engine_arguments_common'])
    for key, value in (('--max-context', 2048), ('--prefill', 64), ('--ple-row-cache', 65536)):
        args[args.index(key)+1] = str(value)
    args += p['extra']
    env = dict(runtime['inherited_math_env'])
    env.update(common['runtime_environment_common'])
    env.update(p['env'])
    env.update(ZE_AFFINITY_MASK=','.join(map(str, p['cards'])), STRATA_DEBUG='1', STRATA_REQUEST_LINES='1')
    original.require(cfg['args'] == args and cfg['env'] == env, 'Complete matched argument/environment reconstruction differs')
    original.require(cfg['model_name'] == 'hotschmoe-dd' and cfg['aliases'] == [p['alias']], 'Exact stable-first matched alias required')
    original.baseline_profile_gate(args, env)


def registry_gate(name):
    registry_association(REGISTRY.read_text())
    value = parse(REGISTRY.read_text())
    rows = [r for r in value['models'] if r.get('served_model_id') == profile(name)['alias']]
    original.require(len(rows) == 1, 'Matched alias not registered exactly once; canonical registry append is pending')
    return original.sha(REGISTRY)


def registry_association(current_bytes):
    original.require(original.sha(OLD_REGISTRY) == OLD_REGISTRY_SHA, 'Exact historical171 registry snapshot changed')
    old = parse(OLD_REGISTRY.read_text()); delta = parse(DELTA.read_text()); current = parse(current_bytes)
    original.require(type(delta) is list and len(old['models']) == 171 and len(delta) == 2 and len(current['models']) == 173,
                     'Exact171+2=173 list-only registry append required; original171 must remain live until root authorizes')
    original.require(list(old) == list(current) and
                     {k:v for k,v in old.items() if k != 'models'} == {k:v for k,v in current.items() if k != 'models'} and
                     current['models'] == old['models']+delta, 'Every old registry key/value/row/order must remain intact')
    ids = [r['served_model_id'] for r in current['models']]
    original.require(len(set(ids)) == 173 and [r['served_model_id'] for r in delta] == [profile(n)['alias'] for n in ('matched-one','matched-pair')],
                     'Exact two new truthful aliases and uniqueness required')
    return {'schema': 1, 'historical_registry_sha256': OLD_REGISTRY_SHA, 'old_rows_preserved': 171,
            'new_rows': 2, 'current_rows': 173, 'old_current_registry_gate_passed': False,
            'old_runtime_or_math_proof_transferred': False}


def derive(node):
    """Three exact profile lookups plus one historical prerequisite validator."""
    expected = {'prepare': (1, 0), 'validate_prepared': (1, 0), 'metadata_admission_gate': (0, 0),
                'launch': (1, 1), 'screen': (0, 0)}
    original.require(isinstance(node, ast.FunctionDef) and node.name in expected, 'Only five named original functions may be derived')
    counts = [0, 0]
    class Rewrite(ast.NodeTransformer):
        def visit_Call(self, n):
            n = self.generic_visit(n)
            if isinstance(n.func, ast.Name) and n.func.id == 'validate_prepared' and any(
                    isinstance(x, ast.Attribute) and x.attr == 'one_card_receipt' for x in ast.walk(n)):
                original.require(node.name == 'launch' and ast.unparse(n) == 'validate_prepared(Path(a.one_card_receipt).resolve().parent)', 'Exact historical onecard prerequisite call required')
                counts[1] += 1
                n.func = ast.Attribute(value=ast.Name(id='original', ctx=ast.Load()), attr='validate_prepared', ctx=ast.Load())
            return n
        def visit_Subscript(self, n):
            n = self.generic_visit(n)
            if isinstance(n.value, ast.Name) and n.value.id == 'PROFILES':
                original.require(ast.unparse(n.slice) == ('a.profile' if node.name == 'prepare' else "m['profile']"), 'Exact named profile lookup required')
                counts[0] += 1
                return ast.copy_location(ast.Call(func=ast.Name(id='profile', ctx=ast.Load()), args=[n.slice], keywords=[]), n)
            return n
    result = ast.fix_missing_locations(Rewrite().visit(copy.deepcopy(node)))
    original.require(tuple(counts) == expected[node.name], 'Exact declared AST change counts required')
    return result


def ports():
    original.require(sys.flags.optimize == 0 and original.sha(original.__file__) == ORIGINAL_SHA,
                     'Exact original C140 source and ordinary Python required')
    tree = ast.parse(Path(original.__file__).read_text())
    names = ('prepare', 'validate_prepared', 'metadata_admission_gate', 'launch', 'screen')
    selected = [derive(n) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = dict(vars(original))
    namespace.pop('PROFILES', None)
    namespace.update(__file__=__file__, __name__=__name__, profile=profile, original=original)
    exec(compile(ast.Module(body=selected, type_ignores=[]), __file__, 'exec'), namespace)
    return namespace


_PORTS = ports()
_VALIDATE = _PORTS['validate_prepared']


def validate_prepared(directory):
    m = _VALIDATE(directory)
    original.require(m['registry_sha256'] == registry_gate(m['profile']), 'New matched preparation current registry changed')
    cfg = original.read(Path(directory)/'server-config.json')
    config_gate(m['profile'], cfg, m['runtime'])
    return m


_PORTS['validate_prepared'] = validate_prepared


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    a = sub.add_parser('prepare')
    a.add_argument('--profile', choices=('matched-one', 'matched-pair'), required=True)
    a.add_argument('--output', type=Path, required=True)
    a.add_argument('--engine-root', type=Path, required=True)
    for name in ('runtime-receipt', 'upload-lifecycle', 'oracle-receipt'):
        a.add_argument('--'+name, type=Path, required=True)
    a.add_argument('--verify-model-shards', action='store_true')
    a.add_argument('--port', type=int, default=18082)
    a = sub.add_parser('launch')
    for name in ('prepared', 'pre-health'): a.add_argument('--'+name, type=Path, required=True)
    a.add_argument('--one-card-receipt', type=Path)
    a.add_argument('--ready-deadline', type=int, default=600)
    a.add_argument('--max-runtime', type=int, default=7200)
    for action in ('screen', 'stop'):
        a = sub.add_parser(action); a.add_argument('--prepared', type=Path, required=True)
        if action == 'screen': a.add_argument('--request-timeout', type=int, default=180)
    args = p.parse_args()
    if args.action == 'prepare':
        registry_gate(args.profile)  # Refuse pending alias before expensive payload scans.
        _PORTS['prepare'](args)
        original.write(args.output/'matched-profile.json', {'schema': 1, 'scope': 'new matched startup diagnostic, not original C140 qualification',
            'profile': profile(args.profile), 'registry_association': registry_association(REGISTRY.read_text()),
            'controller_sha256': original.sha(__file__), 'original_controller_sha256': ORIGINAL_SHA,
            'hardware_only_speedup_qualified': False, 'original_math_qualified': False})
    elif args.action == 'stop':
        receipt = original.owned_stop(args.prepared.resolve())
        original.require(receipt['clean_exit'] and receipt['engine_clean_exit_proven'], 'Original owned stop failed')
    else:
        if args.action == 'launch':
            m = validate_prepared(args.prepared)
            if m['cards'] == [0, 1]:
                original.require(args.one_card_receipt is not None, 'Actual original C140 onecard qualification required')
                cfg = original.read(args.one_card_receipt.resolve().parent/'server-config.json')
                for key, value in (('--max-context', '2048'), ('--prefill', '64'), ('--ple-row-cache', '65536')):
                    original.require(cfg['args'][cfg['args'].index(key)+1] == value, 'Onecard prerequisite geometry differs')
        _PORTS[args.action](args)


if __name__ == '__main__': main()
