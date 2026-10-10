"""Root-owned independent source40 GEN1 controls; no cached-state inputs."""
import os
import subprocess
import time
import traceback
import sys
from pathlib import Path
import full_cache_shared_batch0_fresh40_v3 as ctrl
from merged_numerical_protocol_v2 import MergedProtocol
from serial_prefix_qualification_v8 import Protocol as AbsentPin
from extract_serial_cacheoff_numerical_v1 import extract


class FreshProtocol(MergedProtocol):
    request = AbsentPin.request


def run(plan, output, pre_health,admitted_plan_bytes,handoff,post_ack_seal):
    ctrl.source_binding()
    ctrl.c1.leased([0, 1])
    health = ctrl.read(pre_health)
    ctrl.require(health['passed'] is True and set(plan['cards']) <= set(health['cards'])
                 and 0 <= time.time() - health['finished_epoch'] <= 300,
                 'Actual original parent health must still be fresh before source40 fresh actor')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'captures').mkdir()
    (output / 'ARM').write_text('Independent actual source40 fresh GEN1\n', encoding='ascii')
    __import__('full_cache_shared_plan_snapshot_v3').write_runtime(output / 'plan.snapshot.json',plan,admitted_plan_bytes)
    command = ctrl.command_recipe(plan, output, os.getpid())
    ctrl.write(output / 'command.json', command)
    protocol = None
    rows = []
    error = None
    state = None
    removed = False
    result = {'health_handoff':handoff,'post_ACK_byte_seal':post_ack_seal,'passed': False, 'started_epoch': time.time(), 'producer_pid': os.getpid(),
              'producer_interpreter': str(Path(sys.executable).resolve()),
              'producer_interpreter_sha256': ctrl.sha(Path(sys.executable).resolve()),
              'cached_state_or_activation_used': False, 'full_cache_runtime_qualified': False}
    name = command[command.index('--name') + 1]
    from full_cache_shared_fresh_retirement_v3 import controlled_signals,restore_signals
    previous_signals = controlled_signals(result)
    try:
        from full_cache_shared_fresh_retirement_v3 import original_protocol
        protocol = original_protocol(FreshProtocol, command, output)
        FreshProtocol.__init__(protocol, command, output)
        for job in plan['jobs']:
            raw = protocol.request('own-fresh-' + job['input_sha256_le32'], job['ids'], fresh=1, max_new=1)
            meta = extract(raw, output / 'captures', plan['args'], plan['env'], ctrl.expected_stage_ranges(plan['args']))
            ctrl.require(raw['pin'] is None and raw['fresh'] == 1 and raw['ids'] == job['ids'], 'Actual independent fresh command changed')
            vectors = {'-1': meta['logits'][0]}
            for row in meta['residuals']:
                key = str(row['layer'])
                ctrl.require(key not in vectors, 'Duplicate actual independent fresh layer')
                vectors[key] = row
            ctrl.require(set(vectors) == {str(l) for l in range(-1, 48)}, 'Actual independent complete49 required')
            rows.append({'job': job, 'raw': raw, 'meta': meta, 'vectors': vectors})
            ctrl.write(output / 'requests.json', rows)
    except BaseException as exc:
        error = type(exc).__name__ + ': ' + str(exc)
        result['failure_traceback'] = traceback.format_exc()
    finally:
        def owned():
            obj = ctrl.c1.inspected(name)
            ctrl.require(obj['Name'] == '/' + name and obj['Image'] == plan['image']
                         and obj['Config']['Labels'].get('b70.prefix.plan') == ctrl.sha(output / 'plan.snapshot.json'),
                         'Fresh actor inspection must belong to original recipe')
            from full_cache_shared_memory_capture_v3 import recipe_binding
            recipe_binding(obj, command, plan['image'])
            return obj
        from full_cache_shared_fresh_retirement_v3 import finish
        rc, retirement, ledger, cleanup_errors = finish(protocol, name, owned, subprocess.run, ctrl.c1.absent, ctrl.write, output)
        state = retirement['state']; removed = retirement['removed']; result['actor_retirement'] = ledger
        if cleanup_errors or ledger['failure_count'] or ledger['interruption_signals'] or not retirement['normal_terminal']:
            error = (error + '; ' if error else '') + 'owned retirement: ' + str(cleanup_errors or ledger)
        result.update(error=error, engine_rc=rc, state=state, removed=removed,
                      requests=rows, plan_sha256=ctrl.sha(output / 'plan.snapshot.json'),
                      finished_epoch=time.time())
        result['collection_and_teardown_passed'] = (result['interruption_signals'] == [] and error is None and rc == 0 and removed
                                                   and state['Running'] is False and state['ExitCode'] == 0
                                                   and not state.get('OOMKilled') and len(rows) == len(plan['jobs']))
        result['passed'] = result['collection_and_teardown_passed']
        result['artifact_bindings'] = {str(path.relative_to(output)): {'sha256': ctrl.sha(path), 'bytes': path.stat().st_size}
                                       for path in output.rglob('*') if path.is_file() and path.name != 'report.json'}
        ctrl.write(output / 'report.json', result)
        restore_signals(previous_signals)
    return result
