"""Separate source40 fresh GEN1 actors for actual cached consumed prefixes."""
import argparse
import copy
import hashlib
import os
import shlex
from pathlib import Path
import live_cache_identity_controller_v1 as shared
from serial37_canonical_json_v3 import canonical
from extract_serial_cacheoff_numerical_v1 import profile
HERE = shared.HERE
ROOT = shared.ROOT
c1 = shared.c1
sha, read, write, require = shared.sha, shared.read, shared.write, shared.require


def input_digest(ids):
    require(type(ids) is list and 1 <= len(ids) <= 2048
            and all(type(t) is int and 0 <= t < 248320 for t in ids), 'Exact actual consumed IDs required')
    return hashlib.sha256(b''.join(t.to_bytes(4, 'little') for t in ids)).hexdigest()


def job_roster(groups):
    require(type(groups) is list and groups, 'Actual complete cached groups required')
    by_input = {}
    associations = []
    for group in groups:
        require(set(group['vectors']) == {str(l) for l in range(-1, 48)}, 'Actual full cached49 required')
        key = input_digest(group['input_ids'])
        if key in by_input:
            require(by_input[key]['ids'] == group['input_ids'], 'Consumed-prefix digest collision')
        else:
            by_input[key] = {'input_sha256_le32': key, 'ids': group['input_ids'],
                             'max_new': 1, 'fresh': 1, 'pin': None}
        associations.append({'rid': group['rid'], 'role': group['role'], 'input_sha256_le32': key})
    jobs = list(by_input.values())
    require(len(jobs) <= 48, 'Full actor bounded to at most48 complete consumed-prefix jobs')
    return {'jobs': jobs, 'cached_group_associations': associations,
            'groups': [jobs[i:i + 6] for i in range(0, len(jobs), 6)],
            'cached_state_imported': False, 'actual_fresh_execution_observed': False}


def fresh_recipe(actor_plan):
    args = list(actor_plan['args'])
    env = dict(actor_plan['env'])
    for option in ('--batch', '--prompt-cache', '--conversation-cache-mib'):
        shared.set_arg(args, option, 0)
    env.update(STRATA_FULL_CACHE_OBSERVER38='0', STRATA_CRITICAL_PATH_TRACE='0',
               STRATA_BATCH_FIDELITY_DIAG='0', STRATA_BATCH_FULL_STATE_CHAIN='0',
               STRATA_BATCH_PUBLIC_PREFIX='0', STRATA_FIDELITY_DIAG='1',
               STRATA_FIDELITY_DIAG_ACTIVATIONS='1', STRATA_PREFIX_DIAG='1',
               STRATA_PREFIX_LIFECYCLE_DIAG='1', STRATA_LAYER0_Q8_DIAG='0',
               STRATA_PREFIX30='0', STRATA_PLE_INPUT33='0',
               STRATA_FIDELITY_DIAG_ARM='/results/ARM',
               STRATA_FIDELITY_DIAG_DIR='/results/captures', STRATA_PREFIX_DIAG_ARM='/results/ARM')
    env.pop('STRATA_FULL_CACHE_CAPTURE_RIDS38', None)
    profile(args, env)
    return args, env


ADMITTED_PLAN_BYTES=True
ENGINE_PLAN=shared.ENGINE_PLAN
ENGINE_SHA=shared.ENGINE_SHA

def source_binding():
    return shared.source_binding()


def expected_stage_ranges(args):
    return shared.expected_stage_ranges(args)


def verify_model_identity(*args, **kwargs):
    return shared.verify_model_identity(*args, **kwargs)


def lane_contract(lane):
    return shared.lane_contract(lane)


def actual_groups(actor_root):
    import validate_live_cache_identity_v1 as actor
    actor.finalized_binding(actor_root)
    child = read(Path(actor_root) / 'child/report.json')
    return [group for row in child['phase_receipts'] if row['raw49'] is not None
            for group in row['raw49']['groups']]


def manifest_binding(plan):
    require(plan['schema'] == 'full-cache-shared-fresh40-v3' and plan['kind'] == 'serial'
            and plan['lane'] == 'source40' and plan['driver_sha256'] == sha(__file__), 'Separate current39 fresh producer required')
    source = source_binding()
    require(plan['source_binding'] == source, 'Fresh current source changed')
    actor_root = Path(plan['cached_parent'])
    groups = actual_groups(actor_root)
    actor_plan = read(actor_root / 'input-plan.snapshot.json')
    require(sha(actor_root / 'parent-qualification.json') == plan['cached_parent_sha256']
            and sha(actor_root / 'input-plan.snapshot.json') == plan['cached_plan_sha256'], 'Actual closed cached parent changed')
    require(canonical(job_roster(groups)) == canonical(plan['job_roster']), 'Actual complete fresh49 roster changed')
    require(type(plan['group_index']) is int and 0 <= plan['group_index'] < len(plan['job_roster']['groups']), 'Exact declared fresh actor group required')
    prepared, proof = shared.baseline.finalized_binding(Path(plan['prepared']))
    require(prepared['engine_receipt_sha256'] == plan['engine_receipt_sha256'] == actor_plan['engine_receipt_sha256']
            and sha(Path(plan['prepared']) / 'prepared.json') == plan['prepared_sha256'], 'Fresh/cached actual39 SDK mismatch')
    args, env = fresh_recipe(actor_plan)
    require(canonical(plan['args']) == canonical(args) and canonical(plan['env']) == canonical(env)
            and plan['cards'] == prepared['cards'] and plan['image'] == prepared['runtime']['image']
            and plan['pack'] == prepared['pack'], 'Fresh exact independent source40 profile differs')
    require(plan['jobs'] == plan['job_roster']['groups'][plan['group_index']] and len(plan['jobs']) <= 6,
            'All jobs of current bounded fresh actor required')
    require(canonical(plan['baseline_binding']) == canonical(proof) and plan['model_identity'] == actor_plan['model_identity'], 'Fresh actual baseline/identity binding differs')
    return {'source': source, 'current_baseline': proof, 'fresh_cacheOFF_scope': True,
            'original_cached_state_or_activations_imported': False, 'full_cache_runtime_qualified': False}


def command_recipe(plan, output, pid):
    output = Path(output).resolve()
    prepared = read(Path(plan['prepared']) / 'prepared.json')
    engine = Path(prepared['engine_receipt']).parent
    name = 'b70-prefix-' + str(pid) + '-fullcache-fresh40-v3'
    command = ['docker', 'run', '-i', '--name', name, '--label', 'b70.prefix.plan=' + sha(output / 'plan.snapshot.json'),
               '--network', 'none', '--device', '/dev/dri', '--entrypoint', '/bin/bash', '--user', '1000:1000',
               '--memory', '105g', '--memory-swap', '105g', '--group-add', str(os.stat('/dev/dri/renderD128').st_gid)]
    for path, target, mode in ((engine / 'build', '/build', 'ro'), (engine / 'source', '/src', 'ro'),
                               (Path(plan['pack']), '/pack', 'ro'),
                               (ROOT / read(HERE / 'model-lock.json')['destination'], '/model', 'ro'),
                               (output, '/results', 'rw')):
        command += ['-v', str(path) + ':' + target + ':' + mode]
    for key, value in sorted(plan['env'].items()):
        command += ['-e', key + '=' + str(value)]
    return command + [plan['image'], '-lc', 'set -e; if [[ ${SETVARS_COMPLETED:-0} != 1 ]]; then source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; fi; cd /src; exec ' + shlex.join(['/build/strata', '--serve'] + plan['args'])]


def prepare(args):
    require(not args.output.exists(), 'New actual fresh control plan required')
    actor_root = args.cached_parent.resolve()
    groups = actual_groups(actor_root)
    actor_plan = read(actor_root / 'input-plan.snapshot.json')
    roster = job_roster(groups)
    require(type(args.group_index) is int and 0 <= args.group_index < len(roster['groups']), 'Preregistered complete fresh group required')
    actual_args, env = fresh_recipe(actor_plan)
    plan = {'schema': 'full-cache-shared-fresh40-v3', 'kind': 'serial', 'lane': 'source40',
            'driver_sha256': sha(__file__), 'source_binding': source_binding(),
            'cached_parent': str(actor_root), 'cached_parent_sha256': sha(actor_root / 'parent-qualification.json'),
            'cached_plan_sha256': sha(actor_root / 'input-plan.snapshot.json'),
            'group_index': args.group_index, 'job_roster': roster, 'jobs': roster['groups'][args.group_index],
            'args': actual_args, 'env': env, 'state_initialized_from_fresh': True,
            'actual_fresh_execution_observed': False, 'full_cache_runtime_qualified': False}
    for key in ('prepared', 'prepared_sha256', 'engine_receipt_sha256', 'cards', 'image', 'pack', 'model_identity', 'baseline_binding'):
        plan[key] = copy.deepcopy(actor_plan[key])
    manifest_binding(plan)
    write(args.output, plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='mode', required=True)
    p = actions.add_parser('prepare')
    p.add_argument('--cached-parent', type=Path, required=True)
    p.add_argument('--group-index', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = actions.add_parser('run')
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--expected-plan-sha256', required=True)
    p.add_argument('--pre-health', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args)
        return 0
    from run_live_cache_identity_fresh40_v1 import run
    from full_cache_shared_plan_snapshot_v8 import Snapshot
    snapshot=Snapshot(args.plan,args.expected_plan_sha256);plan=snapshot.plan
    manifest=manifest_binding(plan);snapshot.verify()
    c1.leased([0, 1])
    from full_cache_shared_health_handoff_v8 import child_wait,post_ack_seal
    import sys
    handoff=child_wait(plan,manifest,args.output);seal=post_ack_seal(plan,sys.modules[__name__],handoff);snapshot.verify()
    return 0 if run(plan, args.output,args.output.parent/'leaf-health.json',snapshot.raw,handoff,seal)['collection_and_teardown_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
