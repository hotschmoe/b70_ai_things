"""Current source40 closed parent admission shared by actor/fresh controls."""
import hashlib
from pathlib import Path
from serial37_canonical_json_v3 import canonical
import full_cache_shared_runtime_v10 as shared
from api_journal_exact_admission_v5 import journal_gate
from validate_batch_api_cache_positive_buffered_v5 import exact_health_gate
from source_page_watchdog_v3 import KNOWN_PAGES


def admit(root, controller, wrapper):
    root = Path(root).resolve()
    read, sha, require = shared.read, shared.sha, shared.require
    parent = read(root / 'parent-qualification.json')
    plan = read(root / 'input-plan.snapshot.json')
    child = read(root / 'child/report.json')
    before = controller.manifest_binding(plan)
    require(parent['passed'] is True and parent['errors'] == [] and parent['child_return_code'] == 0
            and parent['owned_containers_terminal'] is True and parent['forced_cleanup'] is False
            and parent['interrupted'] is False and canonical(parent['prepared_chain']) == canonical(before),
            'Actual independently closed source40 parent required')
    require(parent['wrapper_sha256'] == sha(wrapper) and parent['controller_sha256'] == sha(controller.__file__)
            and parent['child_report_sha256'] == sha(root / 'child/report.json'), 'Actual source40 parent/controller/report bytes differ')
    require((root / 'wrapper.py').read_bytes() == Path(wrapper).read_bytes()
            and (root / 'controller.py').read_bytes() == Path(controller.__file__).read_bytes(), 'Original producer snapshots differ')
    external = Path(parent['plan'])
    child_plan = root / 'child/plan.snapshot.json'
    require(sha(external) == parent['plan_sha256'] == child['plan_sha256'] == sha(root / 'input-plan.snapshot.json') == sha(child_plan)
            and canonical(read(external)) == canonical(plan) == canonical(read(child_plan)), 'Actual external/input/child snapshots differ')
    command = read(root / 'child.command.json')
    expected=[str(Path(controller.__file__)), 'run', '--plan', str(external.resolve()), '--pre-health', str(root / 'pre-health.json'), '--output', str(root / 'child')]
    if getattr(controller,'ADMITTED_PLAN_BYTES',False) is True:expected=[str(Path(controller.__file__)), 'run', '--plan', str(root / 'input-plan.snapshot.json'), '--expected-plan-sha256', parent['plan_sha256'], '--pre-health', str(root / 'pre-health.json'), '--output', str(root / 'child')]
    require(len(command)==1+len(expected) and Path(command[0]).resolve() == Path(child['producer_interpreter']).resolve()
            and command[1:] == expected
            and child['producer_pid'] == parent['child_pid'] and sha(child['producer_interpreter']) == child['producer_interpreter_sha256'],
            'Actual original child CLI/PID/interpreter differs')
    require(child['collection_and_teardown_passed'] is True and child['error'] is None and child['removed'] is True
            and child['state']['Running'] is False and child['state']['ExitCode'] == 0 and not child['state'].get('OOMKilled'),
            'Actual normal source40 actor terminal absent')
    require(set(child['artifact_bindings'])=={str(p.relative_to(root/'child'))for p in (root/'child').rglob('*')if p.is_file() and p.name!='report.json'},'Exact complete original child artifact roster required')
    for name, binding in child['artifact_bindings'].items():
        path = root / 'child' / name
        require(not Path(name).is_absolute() and '..' not in Path(name).parts and path.is_file() and not path.is_symlink()
                and path.resolve().is_relative_to(root / 'child') and path.stat().st_size == binding['bytes'] and sha(path) == binding['sha256'],
                'Actual original child artifact extent/path/SHA differs')
    from full_cache_shared_health_handoff_v10 import final_binding as leaf_health_binding
    leaf_health_binding(root,parent,plan,child,controller)
    for stage in ('pre', 'post'):
        exact_health_gate(root, stage)
    journal = journal_gate(root, parent)
    lock = read(shared.HERE / 'model-lock.json')
    identity_path = root / 'post-model-identity.json'
    identity = read(identity_path)
    require(parent['post_model_identity']['path'] == str(identity_path)
            and parent['post_model_identity']['sha256'] == sha(identity_path) and identity['passed'] is True
            and identity['lock_sha256'] == sha(shared.HERE / 'model-lock.json') and identity['model_revision'] == lock['revision'],
            'Actual source40 mandatory new4 publisher receipt differs')
    originals = [row for row in lock['files'] if row['path'].startswith('UD-Q4_K_XL/')]
    require(len(identity['rows']) == len(originals) == 4, 'Actual whole ordered publisher4 required')
    for row, expected in zip(identity['rows'], originals):
        path = shared.ROOT / lock['destination'] / expected['path']
        require(row['path'] == str(path) and row['passed'] is True and row['bytes'] == expected['size']
                and row['sha256'] == row['expected_sha256'] == expected['sha256']
                and row['stat_before'] == row['stat_after'] == shared.c1.stat_signature(path), 'Current ordered original publisher/stat5 differs')
    target = identity['rows'][2]
    for label in ('known_pages_before_hash', 'known_pages_after_hash'):
        page = parent[label]
        require(page['passed'] is True and page['path'] == target['path']
                and page['stat_before'] == page['stat_after'] == target['stat_after']
                and [(r['offset'], r['expected_sha256']) for r in page['rows']] == list(KNOWN_PAGES),
                'Exact current source40 third-shard page roster differs')
        for row in page['rows']:
            preserved = Path(row['preserved_path'])
            raw = preserved.read_bytes()
            require(preserved.resolve().parent == root and not preserved.is_symlink() and len(raw) == row['bytes'] == 4096
                    and hashlib.sha256(raw).hexdigest() == row['sha256'] == row['expected_sha256'] and row['passed'] is True
                    and preserved.read_bytes() == raw, 'Actual preserved source page changed')
    boundary = max(child['finished_epoch'], parent['child_terminal_epoch'], parent['post_health_finished_epoch'],
                   journal['actual_original_journal_rows']['post-kernel-journal']['observed_epoch'])
    require(boundary <= parent['known_pages_before_hash']['epoch'] <= identity['started'] <= identity['finished']
            <= parent['known_pages_after_hash']['epoch'] <= parent['finished_epoch'], 'Actual terminal/posthealth/journal/new4/page ordering differs')
    require(canonical(controller.manifest_binding(plan)) == canonical(before), 'Source39 prerequisites changed during read-only admission')
    return parent, plan, child
