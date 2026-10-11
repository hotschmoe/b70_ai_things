"""Read-only final admission of genuine source35/P30V4 evidence.

Composes frozen public component validators; never rewrites a native report.
Execution may probe source sentinels through the C111 manifest gate. Callers
must authorize that read separately. This module grants no model-math proof.
"""
import json
from pathlib import Path
import prefix_residual30_qualification_v4 as ctrl
import qualify_prefix_residual30_v4 as parent_ctrl
from audit_prefix30_lifecycle_v1 import audit_text

HERE = Path(__file__).resolve().parent
DRIVER_SHA = '56e9c34dd2aa5d444487da7b0955f46925e8b20c270e65c72c6caa6fe0f442c2'
PARENT_SHA = '18faf1f414680103d3e14b9a0e414de46d9d7dbc7a32e1dec6c8e61d82df282e'
sha, read, require = ctrl.sha, ctrl.read, ctrl.require


def same_json(left, right):
    return json.loads(json.dumps(left)) == json.loads(json.dumps(right))


def same_current_identity(current, recorded):
    # The frozen current-page guard timestamps each new observation. Compare
    # all identity/content/stat fields, retaining each observation's own epoch.
    def stable(value):
        value = json.loads(json.dumps(value))
        value.get('current_known_pages', {}).pop('epoch', None)
        return value
    return stable(current) == stable(recorded)


def historical_result(run_root,model_association):
    root = Path(run_root).resolve()
    parent = read(root / 'parent-qualification.json')
    child = read(root / 'child/report.json')
    plan = read(root / 'child/plan.snapshot.json')
    health = read(root / 'post-health.json')
    require(sha(Path(ctrl.__file__)) == DRIVER_SHA == parent['controller_sha256'] == plan['driver_sha256'] == child['driver_sha256'], 'Actual P30V4 driver differs')
    require(sha(Path(parent_ctrl.__file__)) == PARENT_SHA == parent['wrapper_sha256'], 'Actual P30V4 parent differs')
    binding = sha(root / 'child/plan.snapshot.json')
    require(binding == parent['plan_sha256'] == child['plan_sha256'] == sha(root / 'input-plan.snapshot.json'), 'Final native plan/report binding differs')
    require(parent.get('passed') is True and not parent['errors'] and parent.get('child_return_code') == 0 and parent.get('owned_containers_terminal') is True and parent.get('interrupted') is False and parent.get('forced_cleanup') is False and parent.get('kernel_fault_gate_passed') is True and parent.get('pre_health_passed') is True and parent.get('post_health_passed') is True, 'Final native parent lifecycle/source/health failed')
    required = ('passed', 'numerical_and_teardown_passed', 'post_health_passed', 'P30_logical_lifecycle_qualified', 'actual_original_input33_bitwise', 'actual_P30_lastrow_SFD48_bitwise', 'whole_prefix_rows_qualified')
    require(all(child.get(key) is True for key in required) and child.get('error') is None and child.get('full_model_math_qualified') is False and child.get('complete_layer_math_qualified') is False and child.get('original_own_state_reference_qualified') is False and child.get('graph_retirement_runtime_handle_association_observed') is False, 'Final source33 native observation scope differs')
    require(child['engine_receipt_sha256'] == plan['engine_receipt_sha256'] and child['cards'] == plan['cards'] and len(child['results']) == 2, 'Final native engine/arm roster differs')
    require(parent['finished_epoch'] >= health['finished_epoch'] >= child['finished_epoch'] and parent['child_terminal_epoch'] >= child['finished_epoch'], 'Final native terminal/health chronology differs')
    for stage in ('pre', 'post'):
        h = read(root / (stage + '-health.json'))
        require(h['passed'] is True and h['cards'] == [0, 1] and len(h['files']) == 2 and h['health_image'] == parent_ctrl.HEALTH, 'Actual strict/compiled health roster differs')
        for row in h['files']:
            require(row['return_code'] == 0 and row['error'] is None and sha(row['path']) == row['sha256'], 'Actual health producer log changed')
    from postboot_original_native_manifest_v1 import manifest_binding,validate_prepared
    chain = manifest_binding(ctrl,plan,model_association)['original_binding']
    require(chain['generation'].get('prompt_verifier_P30_capture35') is True, 'Earlier prompt verifier capture35 SDK marker absent')
    require(chain['generation'].get('final_window_PLE_observer_fix34') is True, 'Final accepted prompt observer34 SDK marker absent')
    require(same_json(chain, parent['prepared_chain']), 'Actual source35/C113 prepared chain differs')
    post = child['post_model_identity']
    require(Path(post['path']).resolve() == root / 'post-model-identity.json' and sha(post['path']) == post['sha256'], 'Final native complete source4 association differs')
    prepared = validate_prepared(Path(plan['prepared']),model_association)
    from postboot_original_identity_reader_v1 import native_identity
    identity_admission = native_identity(root / 'post-model-identity.json',read(HERE / 'model-lock.json') if 'HERE' in globals() else read(ctrl.ROOT / 'strata/flash-next/model-lock.json'),[Path(row['path']) for row in prepared['model_shards']],post,model_association=model_association)
    identity = identity_admission['original_binding']
    require(same_current_identity(identity, post), 'Current native source4 proof differs')
    ctrl.numeric.post_hash_boundary(read(root / 'post-model-identity.json'), child, health)
    require(read(root / 'post-model-identity.json')['started'] >= parent['child_terminal_epoch'], 'Source4 scan predates actual child terminal')
    life_path = root / 'prefix30-logical-lifecycle.json'
    life = read(life_path)
    require(sha(life_path) == child['lifecycle_receipt_sha256'] and life['passed'] is True and life['plan_sha256'] == binding and life['graph_retirement_runtime_handle_association_observed'] is False, 'Native P30 lifecycle association differs')
    all_rows = {}
    for name, enabled, result in zip(('p30_off', 'p30_on'), (False, True), child['results']):
        directory = root / 'child' / name
        require(same_json(read(directory / 'result.json'), result) and result['passed'] is True and result['error'] is None and result['removed'] is True and result['engine_rc'] == 0 and result['state']['ExitCode'] == 0 and not result['state'].get('OOMKilled') and not result['state']['Running'] and result['complete_four_prefix_roster'] is True, 'Native arm terminal/result association differs')
        log = directory / 'engine.combined.log'
        fresh = audit_text(log.read_text(), dict(enumerate(ctrl.expected_stage_ranges(plan['args']))), enabled)
        fresh['log_sha256'] = sha(log)
        require(same_json(fresh, life['arms'][name]), 'Current native owning-free/producer log differs')
        rows = read(directory / 'requests.json')
        require([row['prefix'] for row in rows] == ctrl.PREFIXES, 'Current four-prefix roster differs')
        for row in rows:
            require(row['raw']['ids'] == plan['prefixes'][str(row['prefix'])] and row['raw']['fresh'] == 1, 'Current accepted GEN prefix/fresh differs')
            meta = ctrl.base.extract(row['raw'], directory / 'captures', True, True, ctrl.expected_stage_ranges(plan['args']))
            external = ctrl.strengthen(row['raw'], ctrl.expected_stage_ranges(plan['args']), True, True)
            require(same_json(meta, row['meta']) and same_json(external, row['external']), 'Current SFD vectors/producer association differs')
        all_rows[name] = rows
        if not enabled:
            require(not list((directory / 'p30').glob('*')) and not any('P30 ' in line for line in log.read_text().splitlines()), 'P30 OFF produced observer output')
    comparisons = [{'prefix': left['prefix'], **ctrl.numeric.compare_numeric(left, right)} for left, right in zip(all_rows['p30_off'], all_rows['p30_on'])]
    require(same_json(comparisons, child['comparisons']), 'Current matched OFF/ON raw comparison differs')
    on = root / 'child/p30_on'
    requests = {int(row['meta']['logits'][0]['request']): row['raw']['ids'] for row in all_rows['p30_on']}
    capture = ctrl.collect(on / 'p30', requests, dict(enumerate(ctrl.expected_stage_ranges(plan['args']))), binding, on / 'engine.combined.log')
    require(same_json(capture, read(on / 'capture.json')), 'Current whole-prefix P30 data differs')
    mapping = ctrl.cross_sfd(capture, all_rows['p30_on'])
    require(same_json(mapping, read(on / 'lastrow-mapping.json')), 'Current P30/SFD raw mapping differs')
    input_path = root / 'original-input33.json'
    require(sha(input_path) == child['input33_receipt_sha256'], 'Original source33 input receipt changed')
    ctrl.validate_original_input33(input_path, root / 'child', plan, binding, root / 'post-model-identity.json')
    return parent, child, plan, {'parent_sha256': sha(root / 'parent-qualification.json'), 'child_sha256': sha(root / 'child/report.json'), 'plan_sha256': binding, 'source_plan_sha256': ctrl.PLAN_SOURCE_SHA, 'chain': chain, 'native_terminal_postidentity': post, 'posthealth_finished': health['finished_epoch'], 'current_raw_recollection': True, 'capture': capture}


def finalized_binding(run_root,model_association):
    from postboot_original_association_recheck_v1 import recheck
    recheck(model_association)
    original=historical_result(run_root,model_association)
    return {'original_result':original,'historical_evidence_only':True,'old_current_stat_gate_passed':False,'current_runtime_qualified':False,'historical_GPU_health_transferred':False,'current_model_identity_sha256':model_association['current_identity_sha256']}
