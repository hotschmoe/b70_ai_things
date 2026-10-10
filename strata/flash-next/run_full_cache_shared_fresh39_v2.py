"""Root-owned independent source39 GEN1 controls; no cached-state inputs."""
import os
import subprocess
import time
import traceback
import sys
from pathlib import Path
import full_cache_shared_fresh39_v2 as ctrl
from merged_numerical_protocol_v2 import MergedProtocol
from serial_prefix_qualification_v8 import Protocol as AbsentPin
from extract_serial_cacheoff_numerical_v1 import extract


class FreshProtocol(MergedProtocol):
    request = AbsentPin.request


def run(plan, output, pre_health):
    ctrl.manifest_binding(plan)
    ctrl.c1.leased([0, 1])
    health = ctrl.read(pre_health)
    ctrl.require(health['passed'] is True and set(plan['cards']) <= set(health['cards'])
                 and 0 <= time.time() - health['finished_epoch'] <= 300,
                 'Actual original parent health must still be fresh before source39 fresh actor')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'captures').mkdir()
    (output / 'ARM').write_text('Independent actual source39 fresh GEN1\n', encoding='ascii')
    ctrl.write(output / 'plan.snapshot.json', plan)
    command = ctrl.command_recipe(plan, output, os.getpid())
    ctrl.write(output / 'command.json', command)
    protocol = None
    rows = []
    error = None
    state = None
    removed = False
    result = {'passed': False, 'started_epoch': time.time(), 'producer_pid': os.getpid(),
              'producer_interpreter': str(Path(sys.executable).resolve()),
              'producer_interpreter_sha256': ctrl.sha(Path(sys.executable).resolve()),
              'cached_state_or_activation_used': False, 'full_cache_runtime_qualified': False}
    name = command[command.index('--name') + 1]
    try:
        protocol = FreshProtocol(command, output)
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
        try:
            rc = protocol.close() if protocol is not None else None
        except BaseException as exc:
            rc = None
            error = (error + '; ' if error else '') + str(exc)
        try:
            obj = ctrl.c1.inspected(name)
            ctrl.require(obj['Name'] == '/' + name and obj['Image'] == plan['image']
                         and obj['Config']['Labels'].get('b70.prefix.plan') == ctrl.sha(output / 'plan.snapshot.json'),
                         'Fresh actor terminal inspection must belong to exact original recipe')
            state = obj['State']
            ctrl.write(output / 'owned-terminal-inspection.json', obj)
            if state['Running'] is False:
                subprocess.run(['docker', 'rm', name], check=True, capture_output=True, timeout=30)
                removed = ctrl.c1.absent(name)
        except BaseException as exc:
            error = (error + '; ' if error else '') + 'owned terminal: ' + str(exc)
        result.update(error=error, engine_rc=rc, state=state, removed=removed,
                      requests=rows, plan_sha256=ctrl.sha(output / 'plan.snapshot.json'),
                      finished_epoch=time.time())
        result['collection_and_teardown_passed'] = (error is None and rc == 0 and removed
                                                   and state['Running'] is False and state['ExitCode'] == 0
                                                   and not state.get('OOMKilled') and len(rows) == len(plan['jobs']))
        result['passed'] = result['collection_and_teardown_passed']
        result['artifact_bindings'] = {str(path.relative_to(output)): {'sha256': ctrl.sha(path), 'bytes': path.stat().st_size}
                                       for path in output.rglob('*') if path.is_file() and path.name != 'report.json'}
        ctrl.write(output / 'report.json', result)
    return result
