# NEW explicit pack epoch admission port of adjudicate_c137_missing_started_v3_v1.py
from c137_prepared_pack47_v1 import validate_prepared
'Narrow read-only adjudication of actual90362 producer metadata omission.'
import copy, json, math, sys
from pathlib import Path
from unittest.mock import patch
import qualify_c1_serving_combined_v137_v3 as old
from c137_journal_binding_v3 import parent_binding, journal_binding, health_binding
c = old.c1
HERE = Path('/mnt/vm_8tb/github/b70_ai_things/strata/flash-next/adjudicate_c137_missing_started_v3_v1.py').resolve().parent
RUN = Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/c137-onecard-segmented-prepared-v2')
PINS = {'parent-qualification.json': '788464563dec4ba5d761ce4e75280f4543d461b04f3c4accc55f1612f0ef7c36', 'parent-before-proof.json': 'fde7784cb6f7f3f00912962fb7ccbab62bcf9a8a6d0a89939fb347fd6b5d9ed5', 'c1-source-identity-proof-v137-v3.json': 'f7b885993cb1aa2312c8fdfbe50076e90df1138f0049ca244e48345181183db6', 'qualification-controller-v137-v3.json': 'a90790402ee74d2cd0a99bed6a2e0e6a515330258f6e58f11c31f8a29895c18a', 'screen.json': '06953e5e4f29cb6469dba523b57e85d43cb9389420a39b9b6b8e10d971013c0b', 'c1-post-model-identity-v137-v3.json': '466fe646048f82067d919b93438706d0f7f9e33c992a6180a8bfe531e241290e'}
OLD_SHA = '46e348d958242587dda704ff8b61c32a04aaf7269e77dc277910bebd48443778'
OLD_HELPER_SHA = 'f7794b4cb26189acb5449276c23677af1fa03316a4cf0c98bb356ccd3679ef36'
OLD_PLAN_SHA = 'cdf2a877c3d2d3c3e9b3d7a74b08a36c683ade49907762860d194099e4f258ce'

def source_binding():
    c.require(c.sha(old.__file__) == OLD_SHA and c.sha(HERE / 'c137_journal_binding_v3.py') == OLD_HELPER_SHA and (c.sha(HERE / 'c1-combined-v137-parent-source-plan-v3.json') == OLD_PLAN_SHA), 'Exact frozen producer/reader/helper proposal required')
    return {'old_parent_sha256': OLD_SHA, 'old_helper_sha256': OLD_HELPER_SHA, 'old_plan_sha256': OLD_PLAN_SHA}

def tree_state(root):
    root = Path(root).resolve()
    rows = {}
    for p in sorted(root.rglob('*')):
        c.require(not p.is_symlink(), 'Original run symlink refused')
        if p.is_file():
            rows[str(p.relative_to(root))] = {'sha256': c.sha(p), 'stat5': c.stat_signature(p)}
    return rows

def metadata_views(parent, before, proof):
    """Single explicit missingfield repair from original producer proof, not a waiver."""
    c.require(parent.get('passed') is False and parent.get('parent_generation') == 1373 and (parent.get('post_error') == "'started_epoch'") and ({k for k in parent if k.endswith('error')} == {'post_error'}), 'Only exact original missingstarted producer failure admitted')
    c.require('started_epoch' not in before and 'started_epoch' not in parent and (set(parent) - set(before) == {'post_error', 'finished_epoch', 'page_guard_calls', 'post_error_source_pages_observation'}) and all((parent[k] == v for k, v in before.items())), 'Original before/final parent mustmatch except exact failure-finalization fields')
    epoch = proof['started_epoch']
    c.require(type(epoch) in (int, float) and math.isfinite(epoch) and (epoch > 0), 'Original saved producer start mustbe finite')
    c.require(parent['owned_terminal'] is True and parent['launch_supervisor_exit_code'] == 0 and (parent['pre_health_passed'] is True) and (parent['post_health_passed'] is True) and (parent['kernel_fault_gate_passed'] is True) and (parent['post_model_identity']['passed'] is True), 'Original actual terminal/health/full4 prerequisites absent')
    c.require(parent['finished_epoch'] >= proof['journal_binding_epoch'] and parent['post_error_source_pages_observation']['passed'] is True, 'Actual finalfailed-parent/sourceguard chronology differs')
    a = copy.deepcopy(parent)
    b = copy.deepcopy(before)
    a['started_epoch'] = b['started_epoch'] = epoch
    return (a, b)

def raw_controller_binding(root, raw, prepared):
    root = Path(root).resolve()
    screen = c.read(root / 'screen.json')
    stop = c.read(root / 'stop.json')
    supervisor = c.read(root / 'launch-supervisor-exit.json')
    launch = c.read(root / 'launch.json')
    closed = c.read(root / 'launch-supervisor.json')
    for field, name in [('screen_sha256', 'screen.json'), ('stop_sha256', 'stop.json'), ('launch_supervisor_exit_sha256', 'launch-supervisor-exit.json'), ('post_health_sha256', 'post-health.json')]:
        c.require(raw[field] == c.sha(root / name), 'Original raw controller artifact changed ' + name)
    c.require(raw['passed'] is True and screen['passed'] is True and all((screen[k] is True for k in ('api_identity_checked', 'token_transport_and_consumption_passed', 'one_engine', 'repeat_passed', 'coherence_passed'))), 'Actual bounded source/identity/transport screen absent')
    c.require(raw['profile'] == prepared['profile'] and raw['engine_receipt_sha256'] == prepared['engine_receipt_sha256'] and (screen['prepared_sha256'] == c.sha(root / 'prepared.json')) and (launch['prepared_sha256'] == stop['prepared_sha256'] == c.sha(root / 'prepared.json')), 'Actual controller/screen/owned prepared association differs')
    c.require(supervisor['return_code'] == 0 and supervisor['passed'] is True and (closed['passed'] is True) and (stop['removed'] is True) and (stop['clean_exit'] is True) and (stop['engine_clean_exit_proven'] is True) and (launch['container'] == stop['container']), 'Actual zeroexit/owned native/container lifecycle absent')
    state = stop['terminal']
    c.require(state['Running'] is False and state['ExitCode'] == 0 and (not state.get('OOMKilled')) and (not state.get('Error')), 'Actual saved ownedcontainer terminal failed')
    lines = (root / 'engine-token-trace.jsonl').read_bytes().splitlines(keepends=True)
    n = screen['trace_rows']
    c.require(type(n) is int and 0 < n <= len(lines), 'Actual screened trace prefix count differs')
    import hashlib
    c.require(hashlib.sha256(b''.join(lines[:n])).hexdigest() == screen['trace_sha256'] and all((json.loads(line)['kind'] == 'engine_close' for line in lines[n:])) and c.native_close_proven(root), 'Actual screen trace prefix/afterprefix nativeclose changed')
    c.require(len(screen['cases']) == 6 and all((row['coherent'] is True and row['content'] == row['expected'] for row in screen['cases'])), 'Actual six bounded expected/repeat results failed')
    return {'cases': [{'case': row['case'], 'content': row['content'], 'expected': row['expected']} for row in screen['cases']], 'screen_sha256': c.sha(root / 'screen.json'), 'owned_native_closed': True, 'full_model_math_qualified': False}

def finalized_binding(run_root, *, pack_epoch=None):
    root = Path(run_root).resolve()
    c.require(root == RUN, 'Only independently observed actual90362 root admitted')
    source = source_binding()
    for name, digest in PINS.items():
        c.require(c.sha(root / name) == digest, 'Pinned original actual failure artifact changed ' + name)
    before_tree = tree_state(root)
    parent = c.read(root / 'parent-qualification.json')
    before = c.read(root / 'parent-before-proof.json')
    proof = c.read(root / 'c1-source-identity-proof-v137-v3.json')
    raw = c.read(root / 'qualification-controller-v137-v3.json')
    identity = c.read(root / 'c1-post-model-identity-v137-v3.json')
    prepared = validate_prepared(root, pack_epoch=pack_epoch)
    c.require(not (root / 'qualification.json').exists(), 'Original failure mustnot be overwritten by completedqualification')
    parent_view, before_view = metadata_views(parent, before, proof)
    bounded = raw_controller_binding(root, raw, prepared)
    try:
        parent_binding(root, proof, identity)
    except KeyError as exc:
        c.require(exc.args == ('started_epoch',), 'Unexpected original failure cannotbe adjudicated')
    else:
        raise ValueError('Exact original missingstarted reader failure absent')
    final = copy.deepcopy(raw)
    final.update(c1_parent_generation=1373, c1_parent_controller_sha256=OLD_SHA, post_full4_source_qualified=True, c1_source_identity_proof={'path': str(root / 'c1-source-identity-proof-v137-v3.json'), 'sha256': PINS['c1-source-identity-proof-v137-v3.json']}, controller_qualification_sha256=PINS['qualification-controller-v137-v3.json'])
    read_original = c.read

    def read_view(path):
        return before_view if Path(path).resolve() == root / 'parent-before-proof.json' else read_original(path)
    with patch.object(c, 'read', side_effect=read_view):
        old.validate_final_source_proof(root, prepared, candidate_final=final)
        parent_binding(root, proof, identity, parent_view)
    journal_binding(root, proof)
    health_binding(root, proof, 'pre')
    health_binding(root, proof, 'post')
    c.require(raw['passed'] is True and raw['teardown_passed'] is True and (raw['post_health_passed'] is True) and (raw['engine_receipt_sha256'] == parent['engine_receipt_sha256'] == prepared['engine_receipt_sha256']), 'Actual raw controller terminal/SDK qualification absent')
    c.require(before_tree == tree_state(root) and source_binding() == source, 'Original actual run/source changed during readonly adjudication')
    return {'schema': 1, 'adjudication_kind': 'C137_actual90362_missing_started_v3_v1', 'passed': True, 'scoped_C137_baseline_adjudicated': True, 'original_parent_passed': False, 'original_post_error': parent['post_error'], 'original_reports_modified': False, 'original_run_root': str(root), 'original_artifact_sha256': dict(PINS), 'original_tree_binding': before_tree, 'source_binding': source, 'adjudicator_sha256': c.sha(Path('/mnt/vm_8tb/github/b70_ai_things/strata/flash-next/adjudicate_c137_missing_started_v3_v1.py')), 'profile': raw['profile'], 'engine_receipt_sha256': prepared['engine_receipt_sha256'], 'prepared_sha256': c.sha(root / 'prepared.json'), 'teardown_passed': True, 'post_health_passed': True, 'bounded_actual_screen': bounded, 'source_proof': proof, 'derived_final_metadata': final, 'named_view_fields': ['parent-before-proof.started_epoch', 'parent-qualification.started_epoch'], 'source_epoch_origin': 'Exact original saved sourceproof and every original journal command since-floor; no time fitted', 'full_model_math_qualified': False, 'full_logits_parity_qualified': False, 'prefix_state_qualified': False, 'concurrency_qualified': False, 'latency_qualified': False}
