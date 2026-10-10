"""CPU admission primitives for proposed source38 metadata; no qualification."""
import json
import re


def capture_rids(flag, value):
    if flag in (None, '', '0'):
        return ()
    if flag != '1' or not isinstance(value, str):
        raise ValueError('Exact observer1 and canonical RID list required')
    if not re.fullmatch(r'[1-9][0-9]*(?:,[1-9][0-9]*){0,5}', value):
        raise ValueError('One through six canonical positive RIDs required')
    result = tuple(map(int, value.split(',')))
    if len(set(result)) != len(result) or any(x >= 2**64 for x in result):
        raise ValueError('Distinct uint64 RIDs required')
    return result


def records(text):
    result = []
    for index, line in enumerate(text.splitlines()):
        if line.startswith('FC38 '):
            row = json.loads(line[5:])
            if not isinstance(row, dict) or not isinstance(row.get('kind'), str):
                raise ValueError('Malformed FC38 record')
            row = dict(row, original_line_index=index)
            result.append(row)
    return result


def victim_binding(before, after):
    """Bind actual removal, without choosing a victim or assuming LRU policy."""
    kind = before['kind']
    if kind not in ('checkpoint_inventory', 'parked_inventory'):
        raise ValueError('Actual inventory kind required')
    for key in ('kind', 'pid', 'command', 'rid'):
        if after[key] != before[key]:
            raise ValueError('Victim owner changed')
    if before['phase'] != 'before_eviction' or after['phase'] != 'after_eviction':
        raise ValueError('Exact before/after eviction pair required')
    if not 0 < before['event'] < after['event']:
        raise ValueError('Eviction chronology changed')
    key = 'points' if kind == 'checkpoint_inventory' else 'entries'
    rows = before[key]
    index = before['actual_victim']
    if type(index) is not int or not 0 <= index < len(rows):
        raise ValueError('Actual selected victim missing')
    if rows[index]['pinned'] is not False:
        raise ValueError('Actual victim unexpectedly pinned')
    expected = [dict(row, index=i) for i, row in enumerate(rows[:index]+rows[index+1:])]
    if after[key] != expected or after['actual_victim'] != -1:
        raise ValueError('Actual removal/inventory changed')
    byte_key = 'point_capacity_bytes' if kind == 'checkpoint_inventory' else 'snapshot_bytes'
    total_key = 'chain_capacity_bytes' if kind == 'checkpoint_inventory' else 'retained_bytes'
    for inventory in (before, after):
        if inventory[total_key] != sum(row[byte_key] for row in inventory[key]):
            raise ValueError('Logical inventory byte sum changed')
        if inventory['physical_allocator_reclamation_qualified'] is not False:
            raise ValueError('Logical inventory cannot qualify physical reclamation')
    return {'pid': before['pid'], 'rid': before['rid'], 'command': before['command'],
            'actual_victim': rows[index], 'logical_bytes_removed': rows[index][byte_key],
            'physical_allocator_reclamation_qualified': False,
            'actual_full_cache_qualified': False}
