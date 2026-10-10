"""Pure recollection of source39 metadata, without payload or qualification."""
import hashlib
import json
import struct
from full_cache_memory39_contract_v1 import recollect as logical_recollect
from full_cache_observer38_contract_v1 import victim_binding


def require(ok, message):
    if not ok:
        raise ValueError(message)


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate metadata JSON field')
        result[key] = value
    return result


def records(trace, pid):
    require(type(pid) is int and pid > 0, 'Actual owned engine PID required')
    result = []
    previous = 0
    for index, line in enumerate(trace.splitlines()):
        if not line.startswith(('FC38 ', 'FC39 ')):
            continue
        row = json.loads(line[5:], object_pairs_hook=unique,
                         parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite metadata')))
        require(type(row) is dict and type(row.get('kind')) is str,
                'Actual typed metadata record required')
        require('observer' not in row and 'original_line_index' not in row,
                'Original-source metadata cannot supply reader attribution')
        require(type(row.get('pid')) is int and row['pid'] == pid,
                'Foreign metadata process')
        if 'event' in row:
            require(type(row['event']) is int and row['event'] > previous,
                    'Actual global observer event chronology differs')
            previous = row['event']
        result.append(dict(row, observer=line[:4], original_line_index=index))
    require(result, 'Actual source39 metadata absent')
    return result


def request_binding(rows, native_command, selected_rids):
    """One actual GEN/BGEN main body; later native windows are separate."""
    from full_cache_shared_prefix_jobs_v2 import command
    parsed = command(native_command)
    require(parsed is not None, 'Actual submitted native command required')
    rid = parsed['rid']
    digest = hashlib.sha256(b''.join(struct.pack('<I', t) for t in parsed['ids'])).hexdigest()
    slot = -1 if parsed['mode'] == 'GEN' else parsed['slot']
    begin = [r for r in rows if r['kind'] == 'request_begin' and r['rid'] == rid
             and r['slot'] == slot and r['source_normal_dispatch'] == parsed['mode']
             and r['input_sha256_le32'] == digest]
    require(len(begin) == 1, 'Unique actual submitted main-body begin required')
    begin = begin[0]
    work = [r for r in rows if r['kind'] == 'request_work' and r['rid'] == rid
            and r['command'] == begin['command']]
    require(len(work) == 1, 'Unique actual command-local work required')
    work = work[0]
    require(begin['original_line_index'] < work['original_line_index'], 'Main-body work precedes begin')
    for key in ('pid', 'command', 'rid', 'slot', 'slotgen'):
        require(type(begin[key]) is int and type(work[key]) is int and begin[key] == work[key],
                'Actual main-body ownership changed ' + key)
    require(begin['command'] > 0 and begin['slot'] == (-1 if parsed['mode'] == 'GEN' else parsed['slot'])
            and begin['source_normal_dispatch'] == parsed['mode'], 'Actual normal dispatch differs')
    require(begin['tokens'] == len(parsed['ids']) and begin['input_sha256_le32'] == digest,
            'Actual native input digest differs')
    import re
    flags = dict(re.findall(r'(\w+)=([^ ]+)', native_command))
    require(flags.get('fresh') in ('0', '1') and type(begin['fresh']) is bool
            and begin['fresh'] == (flags['fresh'] == '1'), 'Actual fresh flag differs')
    require(type(begin['pin_present']) is bool and begin['pin_present'] == ('pin' in flags)
            and type(begin['pin']) is int and begin['pin'] == int(flags.get('pin', '-1')),
            'Actual pin presence/value differs')
    require(type(begin['raw_requested']) is bool and begin['raw_requested'] == (rid in selected_rids),
            'Actual bounded RID selection differs')
    for key in ('reused', 'read_from', 'reread_to', 'prompt_reached', 'prompt', 'generated'):
        require(type(work[key]) is int, 'Actual main-body integer changed ' + key)
    require(0 <= work['reused'] == work['read_from'] <= work['prompt_reached'] <= work['prompt'] == len(parsed['ids'])
            and work['reread_to'] == -1 and 0 <= work['generated'] <= parsed['max_new'],
            'Actual main-body work bounds differ')
    require(type(work['cancelled']) is bool and work['finish'] in ('stop', 'length', 'cancel'),
            'Actual main-body finish differs')
    require(work['cancelled'] == (work['finish'] == 'cancel'), 'Actual main-body cancel/finish disagreement')
    if work['prompt_reached'] != work['prompt']:
        require(work['cancelled'] and work['finish'] == 'cancel' and work['generated'] == 0,
                'Partial prompt work requires actual prefill cancellation')
    return {'rid': rid, 'begin': begin, 'work': work,
            'actual_new_prompt_tokens': work['prompt_reached'] - work['read_from'],
            'main_body_only': True, 'actual_client_terminal_qualified': False,
            'actual_raw_capture_qualified': False, 'full_cache_runtime_qualified': False}


def resource_binding(rows):
    """Aggregate per-cache samples, never per-slot or physical ownership."""
    logical = [r for r in rows if r['kind'] == 'logical_memory']
    result = logical_recollect(logical)
    victims = []
    pending = {}
    for row in rows:
        if row['kind'] not in ('checkpoint_inventory', 'parked_inventory'):
            continue
        key = (row['kind'], row['pid'], row['command'], row['rid'])
        if row['phase'] == 'before_eviction':
            require(key not in pending, 'Overlapping actual eviction observation')
            pending[key] = row
        elif row['phase'] == 'after_eviction':
            require(key in pending, 'Actual removal lacks original before inventory')
            victims.append(victim_binding(pending.pop(key), row))
    require(not pending, 'Actual removal observation incomplete')
    result.update(victims=victims, resource_context='current_main_body_command_context_not_slot_owner',
                  real_checkpoint_victim_observed=any(v['actual_victim'].get('point_capacity_bytes') is not None for v in victims),
                  real_parked_victim_observed=any(v['actual_victim'].get('snapshot_bytes') is not None for v in victims),
                  actual_full_cache_qualified=False, physical_device_memory_qualified=False)
    return result
