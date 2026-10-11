"""Read-only finalized source35/C113 NUM10 admission; no receipt mutation.
Explicit use reads source sentinels/known pages. This grants native observations,
not independently owned mathematics or actual GPU mixed-operator attribution.
"""
import json
from pathlib import Path
import layer0_numerical_qualification_v10 as ctrl
import qualify_layer0_numerical_v10 as parent_ctrl
from audit_layer0_capture_lifecycle_v2 import audit_text
from verify_layer0_packets_v3 import verify as verify_packets
from validate_prefix_residual30_final_v4 import same_current_identity
sha, read, require = ctrl.sha, ctrl.read, ctrl.require
DRIVER_SHA = '5951972606ed0f05a5f3251d8e04bd8d8827c97473462a3f0baf25e248e5c74e'
PARENT_SHA = '858301e4b7a7e027c9d5ed952a2f774f1dfbd8e66d4c7970712a06018926488f'


def same_json(a, b):
    return json.loads(json.dumps(a)) == json.loads(json.dumps(b))


def recollect(child_root, child, plan, life, packet):
    root = Path(child_root)
    rows = {}
    for name, enabled, result in zip(('candidatecombined_off', 'candidatecombined_on'), (False, True), child['results']):
        directory = root / name
        require(same_json(read(directory / 'result.json'), result) and result['passed'] is True and result['error'] is None and result['removed'] is True and result['engine_rc'] == 0 and result['state']['ExitCode'] == 0 and not result['state'].get('OOMKilled') and not result['state']['Running'] and result['complete_four_prefix_roster'] is True, 'NUM10 native arm terminal differs')
        require(sha(directory / 'engine.combined.log') == result['canonical_combined_log_sha256'], 'NUM10 merged producer log changed')
        current = read(directory / 'requests.json')
        require([r['prefix'] for r in current] == ctrl.PREFIXES, 'NUM10 raw prefix roster differs')
        for row in current:
            require(row['raw']['ids'] == plan['prefixes'][str(row['prefix'])] and row['raw']['fresh'] == 1, 'NUM10 accepted IDs/fresh differ')
            meta = ctrl.base.extract(row['raw'], directory / 'captures', True, True, ctrl.expected_stage_ranges(plan['args']))
            external = ctrl.strengthen(row['raw'], ctrl.expected_stage_ranges(plan['args']), True, True)
            require(same_json(meta, row['meta']) and same_json(external, row['external']), 'NUM10 current SFD/request association differs')
            if enabled:
                require(same_json(ctrl.collect_frame(row['raw'], directory, child['plan_sha256']), row['layer0']), 'NUM10 raw33 fields/nonce/live source association changed')
            else:
                require(row['layer0'] is None and not any(line.startswith('L0Q8 frame ') for line in row['raw']['stderr']) and not list((directory / 'layer0').glob('*')), 'NUM10 OFF published observer data')
        rows[name] = current
    comparisons = [{'prefix': a['prefix'], 'same_source35_off_on': ctrl.compare_numeric(a, b)} for a, b in zip(*rows.values())]
    require(same_json(comparisons, child['comparisons']), 'NUM10 matched OFF/ON proof changed')
    on = root / 'candidatecombined_on'
    fresh_packet = verify_packets(rows['candidatecombined_on'])
    fresh_packet.update(plan_sha256=child['plan_sha256'], requests_sha256=sha(on / 'requests.json'))
    require(same_json(fresh_packet, packet), 'NUM10 actual packet recollection differs')
    fresh_life = audit_text((on / 'engine.combined.log').read_text())
    fresh_life.update(plan_sha256=child['plan_sha256'], log_sha256=sha(on / 'engine.combined.log'))
    require(same_json(fresh_life, life), 'NUM10 actual logical owner/free recollection differs')
    return rows


def historical_result(run_root,model_association):
    root = Path(run_root).resolve()
    parent, child, plan = (read(root / name) for name in ('parent-qualification.json', 'child/report.json', 'child/plan.snapshot.json'))
    binding = sha(root / 'child/plan.snapshot.json')
    require(sha(Path(ctrl.__file__)) == DRIVER_SHA == parent['controller_sha256'] == plan['driver_sha256'] == child['driver_sha256'], 'NUM10 current driver binding differs')
    require(sha(Path(parent_ctrl.__file__)) == PARENT_SHA == parent['wrapper_sha256'], 'NUM10 current parent binding differs')
    require(binding == parent['plan_sha256'] == child['plan_sha256'] == sha(root / 'input-plan.snapshot.json'), 'NUM10 parent/child/plan differs')
    require(parent.get('passed') is True and not parent['errors'] and parent.get('child_return_code') == 0 and parent.get('owned_containers_terminal') is True and parent.get('interrupted') is False and parent.get('forced_cleanup') is False and parent.get('kernel_fault_gate_passed') is True and parent.get('pre_health_passed') is True and parent.get('post_health_passed') is True, 'NUM10 finalized native lifecycle failed')
    require(child.get('schema') == 10 and all(child.get(k) is True for k in ('passed', 'numerical_and_teardown_passed', 'post_health_passed', 'packet_contract_qualified', 'layer0_logical_lifecycle_qualified')) and child.get('error') is None, 'NUM10 finalized observation gates failed')
    require(all(child.get(k) is False for k in ('full_model_math_qualified', 'reference29_or21_runtime_transferred', 'complete_layer_math_qualified', 'original_own_state_fp64_qualified')), 'NUM10 observation scope differs')
    require(child['engine_receipt_sha256'] == plan['engine_receipt_sha256'] and child['cards'] == plan['cards'] and len(child['results']) == 2, 'NUM10 engine/arm roster differs')
    from postboot_original_native_manifest_v1 import manifest_binding,validate_prepared
    chain = manifest_binding(ctrl,plan,model_association)['original_binding']
    require(same_json(chain, parent['prepared_chain']), 'NUM10 actual source35/C113 parent chain differs')
    health = read(root / 'post-health.json')
    for stage in ('pre', 'post'):
        h = read(root / (stage + '-health.json'))
        require(h['passed'] is True and h['cards'] == [0, 1] and len(h['files']) == 2 and h['health_image'] == parent_ctrl.HEALTH, 'NUM10 strict/compiled health roster differs')
        for row in h['files']:
            require(row['return_code'] == 0 and row['error'] is None and sha(row['path']) == row['sha256'], 'NUM10 health producer log changed')
    require(parent['finished_epoch'] >= health['finished_epoch'] >= child['finished_epoch'] and parent['child_terminal_epoch'] >= child['finished_epoch'], 'NUM10 terminal/health chronology differs')
    post = child['post_model_identity']
    require(Path(post['path']).resolve() == root / 'post-model-identity.json' and sha(post['path']) == post['sha256'], 'NUM10 complete source4 association differs')
    prepared = validate_prepared(Path(plan['prepared']),model_association)
    from postboot_original_identity_reader_v1 import native_identity
    identity_admission = native_identity(root / 'post-model-identity.json',read(HERE / 'model-lock.json') if 'HERE' in globals() else read(ctrl.ROOT / 'strata/flash-next/model-lock.json'),[Path(row['path']) for row in prepared['model_shards']],post,model_association=model_association)
    identity = identity_admission['original_binding']
    require(same_current_identity(identity, post), 'NUM10 current original4/pages changed')
    ctrl.post_hash_boundary(read(root / 'post-model-identity.json'), child, health)
    require(read(root / 'post-model-identity.json')['started'] >= parent['child_terminal_epoch'], 'NUM10 source4 scan precedes native terminal')
    life_path, packet_path = root / 'layer0-logical-lifecycle.json', root / 'packet-contract.json'
    require(sha(life_path) == child['lifecycle_receipt_sha256'] and sha(packet_path) == child['packet_receipt_sha256'], 'NUM10 lifecycle/packet receipt changed')
    rows = recollect(root / 'child', child, plan, read(life_path), read(packet_path))
    return parent, child, plan, {'parent_sha256': sha(root / 'parent-qualification.json'), 'child_sha256': sha(root / 'child/report.json'), 'plan_sha256': binding, 'source_plan_sha256': ctrl.PLAN_SOURCE_SHA, 'chain': chain, 'native_terminal_postidentity': post, 'posthealth_finished': health['finished_epoch'], 'current_raw_recollection': True, 'requests': rows['candidatecombined_on'], 'full_model_math_qualified': False}


def finalized_binding(run_root,model_association):
    from postboot_original_association_recheck_v1 import recheck
    recheck(model_association)
    original=historical_result(run_root,model_association)
    return {'original_result':original,'historical_evidence_only':True,'old_current_stat_gate_passed':False,'current_runtime_qualified':False,'historical_GPU_health_transferred':False,'current_model_identity_sha256':model_association['current_identity_sha256']}
