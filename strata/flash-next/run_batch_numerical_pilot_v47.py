"""Actual native initial BGEN pilot; parent owns lease/health/stop/full source gate."""
import argparse, copy, hashlib, json, os, shlex, subprocess, time, traceback
from pathlib import Path
import layer0_numerical_qualification_v9 as numerical
from batch_numerical_protocol_v2 import NativeStream
from batch_numerical_prefixes_v2 import serial_jobs
from audit_batch_fidelity_coverage_v5 import coverage
from batch_numerical_proofs_v47 import genuine_baseline, engine_binding, providers, owner_proofs, native_row_event, exact_producer_counters, artifact_bindings, source_observers_off
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
from batch47_plan_snapshot_v1 import Snapshot, write_snapshot

def require(ok, message):
    if not ok:
        raise ValueError(message)

def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n', encoding='ascii')

def set_arg(args, key, value):
    if key in args:
        args[args.index(key) + 1] = str(value)
    else:
        args.extend([key, str(value)])

def diagnose_failure(exc, stage, stream=None):
    return {'exception_type': type(exc).__name__, 'message': str(exc), 'repr': repr(exc), 'traceback': traceback.format_exc(), 'stage': stage, 'roster_at_failure': copy.deepcopy(stream.roster.requests) if stream else None, 'completed_events_at_failure': len(stream.roster.events) if stream else None}

def run(a, *, pack_epoch=None):
    c1 = providers('source37')[0]
    c1.leased([0, 1])
    admitted = getattr(a, 'admitted', None) or Snapshot(a.plan)
    plan = admitted.plan
    admitted.verify(plan)
    require(a.diagnostic == plan['diagnostic'], 'Native diagnostic command differs admitted plan')
    source_observers_off(plan['env'])
    require(plan.get('native_driver_sha256', plan['driver_sha256']) == c1.sha(Path(__file__)), 'Pilot source changed')
    prepared, _ = genuine_baseline(Path(plan['prepared']), plan['lane'], plan.get('baseline_adjudication_receipt'), pack_epoch=pack_epoch)
    c1 = providers(plan['lane'])[0]
    require(c1.sha(Path(plan['prepared']) / 'prepared.json') == plan['prepared_sha256'] and prepared['engine_receipt_sha256'] == plan['engine_receipt_sha256'], 'Genuine preparation changed')
    engine_binding(Path(plan['engine_root']), plan['lane'])
    health = c1.read(a.pre_health)
    require(health['passed'] and set(plan['cards']) <= set(health['cards']) and (0 <= time.time() - health['finished_epoch'] <= 300), 'Fresh parent actual prehealth required')
    require(pack_epoch is not None,'Explicit H47 process-local pack epoch required');pack_epoch.seal_predevice()
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / 'captures').mkdir()
    admitted.verify(plan)
    write_snapshot(a.output / 'plan.snapshot.json', plan, admitted.raw)
    binding = c1.sha(a.output / 'plan.snapshot.json')
    container = 'b70-prefix-' + str(os.getpid()) + '-native' + str(plan['slots'])
    env = dict(plan['env'])
    env['STRATA_ARTIFACT_IDENTITY_SHA256'] = binding
    env['STRATA_BATCH_FIDELITY_DIAG'] = str(a.diagnostic)
    engine = Path(plan['engine_root'])
    model = ROOT / c1.read(HERE / 'model-lock.json')['destination']
    command = ['docker', 'run', '-i', '--name', container, '--label', 'b70.prefix.plan=' + binding, '--network', 'none', '--device', '/dev/dri', '--user', '1000:1000', '--memory', '105g', '--memory-swap', '105g', '--group-add', str(os.stat('/dev/dri/renderD128').st_gid), '-v', str(engine / 'build') + ':/build:ro', '-v', str(engine / 'source') + ':/src:ro', '-v', plan['pack'] + ':/pack:ro', '-v', str(model) + ':/model:ro', '-v', str(a.output.resolve()) + ':/results']
    for key, value in sorted(env.items()):
        command.extend(['-e', key + '=' + str(value)])
    command.extend([plan['image'], 'exec 2>&1; cd /src; source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; exec ' + shlex.join(['/build/strata', '--serve'] + plan['args'])])
    write(a.output / 'command.json', command)
    leaf_launch_epoch=None
    stream = None
    error = None
    report = None
    failure = None
    stage = 'native stream setup'
    try:
        leaf_launch_epoch=time.time()
        stream = NativeStream(command, a.output, plan['slots'])
        stage = 'warm submit/drain'
        warm = [1001 + i for i in range(plan['slots'])]
        for i, rid in enumerate(warm):
            stream.submit(rid, i, plan['tokens']['warm'][i], 32)
        stream.drain(warm)
        if a.diagnostic:
            stream.arm(a.output / 'ARM')
        else:
            require(len(stream.roster.bt_slots) >= 2, 'Off warmup lacks actual private-slot decode traffic')
            stream.roster.bt_slots.clear()
        stage = 'target submit/drain'
        targets = [2001 + i for i in range(plan['slots'])]
        for i, rid in enumerate(targets):
            stream.submit(rid, i, plan['tokens']['target'][i], 32)
        stream.drain(targets, cancel_rid=targets[plan['cancel_index']] if plan['cancel_index'] is not None else None, diagnostic=bool(a.diagnostic))
        trace = '\n'.join(stream.lines)
        stages = [(i, lo, hi) for i, (lo, hi) in enumerate(numerical.expected_stage_ranges(plan['args']))]
        if a.diagnostic:
            stage = 'armed strict raw coverage'
            raw = coverage(trace, targets, stages, a.output / 'captures')
            require(any((int(e['rows']) == plan['slots'] for e in stream.roster.events[stream.arm_event_index:])), 'Actual completed requested-N-row event absent')
            for line in trace.splitlines():
                if line.startswith('SBF resume '):
                    f = dict((item.split('=', 1) for item in line.split()[2:] if '=' in item))
                    if int(f['rid']) in targets:
                        require({k: int(f[k]) for k in ['reused', 'read_from', 'reread_to']} == {'reused': 0, 'read_from': 0, 'reread_to': -1}, 'Actual target reused unexpectedly; cache0 is not fresh proof')
            stage = 'native row event/fresh counters/serial jobs'
            row_event = native_row_event(trace, plan['slots'])
            policy = {(rid, 'admission'): {'reused': 0, 'read_from': 0, 'reread_to': -1} for rid in targets}
            counters = exact_producer_counters(trace, set(targets), policy)
            jobs = serial_jobs(trace, stream.roster)
            write(a.output / 'serial-jobs.json', jobs)
            write(a.output / 'requests.json', stream.roster.requests)
            report = {'initial_native_raw_coverage_completed': True, 'raw': raw, 'serial_jobs': jobs, 'actual_row_event': row_event, 'exact_counters': counters, 'full_model_math_qualified': False, 'actual_serial_comparison_completed': False, 'actual_cache_or_migration_qualified': False}
        else:
            require(not any((line.startswith(('SBF allocation ', 'SBF replay ', 'SBF vector ')) for line in trace.splitlines())), 'Disabled observer emitted allocation/copy/replay')
            write(a.output / 'requests.json', stream.roster.requests)
            report = {'disabled_observer_actual_collection_completed': True, 'full_model_math_qualified': False}
    except BaseException as exc:
        error = type(exc).__name__ + ': ' + str(exc)
        failure = diagnose_failure(exc, stage, stream)
    finally:
        try:
            rc = stream.close() if stream else None
        except BaseException as exc:
            rc = None
            error = (error + '; ' if error else '') + 'close: ' + type(exc).__name__ + ': ' + str(exc)
            if failure is None:
                failure = diagnose_failure(exc, 'owned close', stream)
        state = c1.inspected(container)['State']
        removed = False
        if not state['Running']:
            subprocess.run(['docker', 'rm', container], capture_output=True, text=True, check=True)
            removed = c1.absent(container)
        owners = owner_proofs((a.output / 'engine.combined.log').read_text(), stages, plan['slots'], plan['lane'], snapshots_observed=bool(a.diagnostic)) if error is None else None
        try:
            admitted.verify(plan)
        except Exception as exc:
            error = (error + '; ' if error else '') + 'admitted plan: ' + str(exc)
        owned_leaf_terminal_epoch=time.time()
        try:write(a.output/'pack-operation-witness.json',pack_epoch.finalize())
        except Exception as exc:error=(error+'; ' if error else '')+'pack post-byte validation: '+str(exc)
        result = {'owned_leaf_terminal_epoch':owned_leaf_terminal_epoch,'leaf_launch_started_epoch':leaf_launch_epoch,'passed': False, 'logical_owner_proofs': owners, 'initial_native_collection_and_teardown_passed': error is None and rc == 0 and (state['ExitCode'] == 0) and (not state.get('OOMKilled')) and removed, 'error': error, 'failure_diagnostic': failure, 'planned_max_new_by_request': plan.get('max_new_by_request'), 'actual_warm_max_new': 32, 'actual_target_max_new': 32, 'constant32_vs_plan_maxnew_contract_unresolved': plan.get('max_new_by_request') != [32] * plan['slots'], 'report': report, 'engine_rc': rc, 'state': state, 'removed': removed, 'finished_epoch': time.time(), 'plan_sha256': binding, 'source_and_posthealth_fullhash_qualified': False, 'scope': 'parent must retain lease/watch2pages/stop/health/final4hash then actualserial/fullvectors; no serving/math claim'}
        result['artifact_bindings'] = artifact_bindings(a.output)
        write(a.output / 'report.json', result)
    return 0 if result['initial_native_collection_and_teardown_passed'] else 1

def main(*, pack_epoch=None):
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--pre-health', type=Path, required=True)
    p.add_argument('--diagnostic', type=int, choices=[0, 1], required=True)
    p.add_argument('--output', type=Path, required=True)
    return run(p.parse_args(), pack_epoch=pack_epoch)
if __name__ == '__main__':
    raise SystemExit(main())
