"""Read-only actual source40 cache-off/absentPIN fresh49 actor admission."""
import copy
import hashlib
from pathlib import Path
import full_cache_shared_batch0_fresh40_v8 as ctrl
from full_cache_shared_closed_source_v8 import admit as source_admit
from extract_serial_cacheoff_numerical_v1 import extract
from serial37_canonical_json_v3 import canonical


def finalized_binding(root):
    root = Path(root).resolve()
    parent, plan, child = source_admit(root, ctrl, Path(__file__).with_name('qualify_full_cache_shared_batch0_fresh40_v8.py'))
    original = root / 'child'
    command = ctrl.read(original / 'command.json')
    ctrl.require(command == ctrl.command_recipe(plan, original, parent['child_pid']), 'Exact original source40 fresh native command differs')
    from full_cache_shared_actor_retirement_v8 import saved_binding
    saved_binding(original, child, command[command.index('--name') + 1], ctrl.read)
    ctrl.require(child['actor_retirement']['original_protocol_CLI_and_pipe_EOF_joined'] is True, 'Original attached Docker CLI and pump EOF must be joined')
    from full_cache_shared_memory_capture_v8 import recipe_binding
    recipe_binding(ctrl.read(original / 'owned-terminal-inspection.json'), command, plan['image'])
    rows = ctrl.read(original / 'requests.json')
    ctrl.require(canonical(rows) == canonical(child['requests']) and [r['job'] for r in rows] == plan['jobs'], 'Actual complete independently submitted job roster differs')
    groups = {}
    for row in rows:
        job = row['job']
        raw = copy.deepcopy(row['raw'])
        individual = original / ('own-fresh-' + job['input_sha256_le32'] + '.json')
        meta = extract(raw, original / 'captures', plan['args'], plan['env'], ctrl.expected_stage_ranges(plan['args']))
        copied = copy.deepcopy(ctrl.read(individual))
        copied_meta = extract(copied, original / 'captures', plan['args'], plan['env'], ctrl.expected_stage_ranges(plan['args']))
        ctrl.require(canonical(raw) == canonical(row['raw']) == canonical(copied)
                     and canonical(meta) == canonical(row['meta']) == canonical(copied_meta), 'Actual raw individual request/source numeric proof changed')
        ctrl.require(raw['ids'] == job['ids'] and raw['pin'] is None and type(raw['fresh']) is int and raw['fresh'] == 1,
                     'Actual fresh whole-prefix absentPIN request differs')
        vectors = {'-1': meta['logits'][0]}
        for value in meta['residuals']:
            key = str(value['layer'])
            ctrl.require(key not in vectors, 'Duplicate actual fresh output layer')
            vectors[key] = value
        ctrl.require(set(vectors) == {str(l) for l in range(-1, 48)}
                     and canonical(vectors) == canonical(row['vectors']), 'Actual full fresh49 vector roster differs')
        normalized = {}
        for layer, value in vectors.items():
            path = Path(value['path'])
            data = path.read_bytes()
            ctrl.require(path.resolve().parent == original / 'captures' and not path.is_symlink()
                         and len(data) == int(value['bytes']) and hashlib.sha256(data).hexdigest() == value['sha256']
                         and path.read_bytes() == data, 'Actual fresh raw consumed bytes/extent/SHA changed')
            normalized[layer] = {'path': str(path), 'sha256': value['sha256'], 'bytes': len(data),
                                 'position': int(value['pos']), 'token': int(value['token'])}
        key = ctrl.input_digest(job['ids'])
        ctrl.require(key == job['input_sha256_le32'] and key not in groups, 'Duplicate/foreign actual independent input')
        groups[key] = {'input_ids': job['ids'], 'vectors': normalized}
    ctrl.require(len(groups) == len(plan['jobs']), 'Incomplete actual independent fresh49 controls')
    source_admit(root, ctrl, Path(__file__).with_name('qualify_full_cache_shared_batch0_fresh40_v8.py'))
    return {'source_lane': 'source40', 'engine_receipt_sha256': plan['engine_receipt_sha256'],
            'cached_parent_sha256': plan['cached_parent_sha256'], 'cached_plan_sha256': plan['cached_plan_sha256'],
            'group_index': plan['group_index'], 'groups_required': len(plan['job_roster']['groups']),
            'job_roster': plan['job_roster'], 'groups': groups,
            'actual_absentPIN_cacheOFF_fresh49_observed': True, 'cached_state_imported': False,
            'actual_parent_sha256': ctrl.sha(root / 'parent-qualification.json'),
            'full_cache_runtime_qualified': False, 'full_model_math_qualified': False}


def complete_controls(actor_root, control_roots):
    actor_root = Path(actor_root).resolve()
    actor_plan = ctrl.read(actor_root / 'input-plan.snapshot.json')
    jobs = ctrl.job_roster(ctrl.actual_groups(actor_root))
    ctrl.require(type(control_roots) is list and len(control_roots) == len(jobs['groups'])
                 and len(set(map(str, control_roots))) == len(control_roots), 'Every actual fresh control actor exactly once required')
    groups = {}
    bindings = []
    for index, root in enumerate(control_roots):
        binding = finalized_binding(root)
        ctrl.require(binding['group_index'] == index and canonical(binding['job_roster']) == canonical(jobs)
                     and binding['engine_receipt_sha256'] == actor_plan['engine_receipt_sha256']
                     and binding['cached_parent_sha256'] == ctrl.sha(actor_root / 'parent-qualification.json')
                     and binding['cached_plan_sha256'] == ctrl.sha(actor_root / 'input-plan.snapshot.json'), 'Fresh control belongs to another actor/source/input roster')
        ctrl.require(not set(groups) & set(binding['groups']), 'Actual fresh input controls duplicated')
        groups.update(binding['groups'])
        bindings.append(binding)
    ctrl.require(set(groups) == {j['input_sha256_le32'] for j in jobs['jobs']}, 'Actual fresh comparison inputs incomplete')
    return groups, bindings
