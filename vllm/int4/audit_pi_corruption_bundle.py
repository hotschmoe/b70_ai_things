#!/usr/bin/env python3
"""CPU-only audit of extracted Pi diagnostics; never execute transcript text."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def audit(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    events = json.loads((root / 'events.json').read_text())
    incidents, groups = [], {}
    for directory in sorted(root.glob('agent_*')):
        sessions = [p for p in directory.glob('*.jsonl') if p.name != 'rpc.jsonl']
        if len(sessions) != 1:
            raise ValueError('Expected one session per agent')
        records = read_rows(sessions[0])
        messages = [r for r in records if r.get('message', {}).get('role') == 'assistant']
        rpc = read_rows(directory / 'rpc.jsonl')
        trips = [r for r in rpc if r.get('type') == 'extension_ui_request'
                 and r.get('message') == ('bang-guard: aborted after 32 consecutive ! characters. '
                                          'Corrupted output is excluded from future model context.')]
        local_429 = 0
        bangs = 0
        for index, record in enumerate(messages):
            message = record['message']
            local_429 += message.get('errorMessage', '').startswith('429 ')
            blocks = []
            for block in message.get('content', []):
                content = block.get('thinking', block.get('text'))
                if content is None:
                    content = json.dumps(block.get('arguments', {}), ensure_ascii=True)
                match = re.search('!{32,}', re.sub(r'\\u0021', '!', content))
                if match:
                    blocks.append({'type': block['type'], 'first_run_offset': match.start(),
                                   'characters': len(content)})
            if not blocks:
                continue
            bangs += 1
            started = message['timestamp'] / 1000
            ended = datetime.fromisoformat(record['timestamp'].replace('Z', '+00:00')).timestamp()
            following = messages[index + 1]['message'] if index + 1 < len(messages) else {}
            next_429 = following.get('errorMessage', '').startswith('429 ')
            incidents.append({'agent': directory.name, 'message_id': record['id'],
                'response_id': message.get('responseId'), 'request_timestamp_ms': message['timestamp'],
                'start_utc': datetime.fromtimestamp(started, timezone.utc).isoformat(),
                'session_record_utc': record['timestamp'],
                'start_to_record_seconds': ended - started, 'blocks': blocks,
                'stop_reason': message.get('stopReason'),
                'next_attempt_is_local_429': next_429,
                'next_attempt_after_record_ms': following.get('timestamp', ended * 1000) - ended * 1000,
                'world_agents_alive_at_start': [a['name'] for a in manifest['agents']
                    if a['born'] <= started < a['died']],
                'usage_total_tokens': message.get('usage', {}).get('totalTokens')})
        declared = {e['request_timestamp_ms'] for e in events
                    if e['agent'] == directory.name and e['event'] == 'bang_guard_trip'}
        observed = {i['request_timestamp_ms'] for i in incidents if i['agent'] == directory.name}
        if declared != observed or len(trips) != bangs:
            raise ValueError('Session, RPC and declared incident counts disagree')
        groups[directory.name] = {'assistant_attempt_records': len(messages),
            'local_429': local_429, 'bang_attempts': bangs,
            'non_429_attempt_records': len(messages) - local_429,
            'stop_reasons': dict(Counter(m['message'].get('stopReason') for m in messages)),
            'errors': dict(Counter(m['message'].get('errorMessage') for m in messages
                                  if m['message'].get('stopReason') == 'error'))}
    total = sum(g['assistant_attempt_records'] for g in groups.values())
    rejected = sum(g['local_429'] for g in groups.values())
    return {'groups': groups, 'assistant_attempt_records': total, 'local_429': rejected,
        'bang_attempts': len(incidents), 'non_429_attempt_records': total - rejected,
        'observed_bang_fraction_excluding_local_429': len(incidents) / (total - rejected),
        'bangs_followed_by_local_429': sum(i['next_attempt_is_local_429'] for i in incidents),
        'bangs_with_one_world_agent_alive': sum(len(i['world_agents_alive_at_start']) == 1 for i in incidents),
        'bang_block_types': dict(Counter(b['type'] for i in incidents for b in i['blocks'])),
        'incidents': sorted(incidents, key=lambda i: i['request_timestamp_ms']),
        'limits': ['Selected affected sessions, not a representative long-run rate.',
                   'Non-429 records include interrupted streams; not all completed requests.',
                   'Session record time is not an exact network abort timestamp.',
                   'World agent count does not exclude other endpoint clients.',
                   'No wire requests: effective cache salt and full payload cannot be verified.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=True) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'incidents'}, indent=2))
