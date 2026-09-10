#!/usr/bin/env python3
"""Offline repetition/structure review of captured Pi replay results.

Flags are diagnostic heuristics, not a semantic correctness certification.
Never execute generated tools. Preserve the original inference outcome.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


def strings(value, path='message'):
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from strings(item, path + '.' + key)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from strings(item, path + '.' + str(index))


def audit(root):
    source = root / 'results.json'
    rows = []
    for result in json.loads(source.read_text()):
        case = result['case']
        name = '%s-%s-%s' % (case['agent'], case['phase'], result['repeat'])
        response = result.get('response') or {}
        flags = []
        for choice in response.get('choices', []):
            message = choice.get('message', {})
            for path, text in strings(message):
                for tag in ('</function>', '</parameter>', '</tool_call>'):
                    count = text.count(tag)
                    if count >= 16:
                        flags.append(dict(kind='repeated_tool_markup', path=path,
                                          tag=tag, count=count))
                # Bounded units; repeated punctuation, XML, escaped newlines,
                # prose and code all remain visible, including in tool args.
                for match in re.finditer(r'(.{1,80}?)\1{15,}', text, re.DOTALL):
                    unit = match.group(1)
                    if unit.strip():
                        flags.append(dict(kind='repeated_unit', path=path,
                                          unit=unit, repeats=len(match[0]) // len(unit),
                                          offset=match.start()))
            for call in message.get('tool_calls', []):
                try:
                    decoded = json.loads(call['function']['arguments'])
                    if not isinstance(decoded, dict):
                        flags.append(dict(kind='non_object_tool_arguments'))
                except (ValueError, KeyError, TypeError):
                    flags.append(dict(kind='invalid_json_tool_arguments'))
        request = json.loads((root / (name + '-request.json')).read_text())
        usage = response.get('usage', {})
        if usage.get('completion_tokens', 0) >= request['max_tokens']:
            flags.append(dict(kind='output_budget_exhausted'))
        events = [json.loads(line) for line in (root / (name + '.sse.jsonl')).read_text().splitlines()]
        stamps = [event['elapsed_s'] for event in events]
        rows.append(dict(name=name, bang=result['bang'], error=result['error'],
                         usage=usage, flags=flags,
                         largest_sse_gap_s=max((b-a for a, b in zip(stamps, stamps[1:])), default=0)))
    return dict(source=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                disclaimer='Heuristic flags need review; absence of flags does not establish coherence.',
                requests=len(rows), flagged_requests=sum(bool(r['flags']) for r in rows), rows=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('replay', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.replay)
    with args.out.open('x') as output:
        output.write(json.dumps(result, ensure_ascii=True, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('requests', 'flagged_requests')}))
