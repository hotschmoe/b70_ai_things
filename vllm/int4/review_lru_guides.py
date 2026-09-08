#!/usr/bin/env python3
"""Bounded functional review of the campaign's specific generated LRU guide."""
import json
from pathlib import Path
import re
import subprocess
import sys

IMAGE = 'sha256:c0522083adc7557aab54abff248a4a4a9f7c32b4d9d66ddf9af05200ae5a3339'
CHECK = '''
import json, random, sys
from collections import OrderedDict
blocks = json.loads(sys.stdin.readline())
checks = 0
for block in blocks:
    namespace = {}
    exec(block, namespace)
    names = [name for name in ('LRUCache', 'LRUCacheCustom') if name in namespace]
    assert len(names) == 1
    cls = namespace[names[0]]
    for capacity in (1, 2, 7):
        actual = cls(capacity)
        expected = OrderedDict()
        rng = random.Random(739251)
        for step in range(1000):
            key = rng.randrange(12)
            op = rng.randrange(10)
            if op < 5:
                value = rng.randrange(10000)
                actual.put(key, value)
                expected[key] = value
                expected.move_to_end(key)
                if len(expected) > capacity: expected.popitem(last=False)
            elif op < 9:
                assert actual.get(key) == expected.get(key)
                if key in expected: expected.move_to_end(key)
            else:
                actual.clear()
                expected.clear()
            assert len(actual) == len(expected)
            assert all((k in actual) == (k in expected) for k in range(12))
            checks += 1
    for invalid in (0, -1):
        try: cls(invalid)
        except ValueError: pass
        else: raise AssertionError('invalid capacity accepted')
print(json.dumps({'passed': True, 'blocks': len(blocks), 'reference_operations': checks}))
'''


def review(root):
    rows = [json.loads(s) for s in (root / 'responses.jsonl').read_text().splitlines()]
    assert len(rows) == 3 and all(r['passed'] and not r['error'] for r in rows)
    snippets = [re.findall(r'```python\n(.*?)```', r['text'], re.S) for r in rows]
    assert all(len(parts) == 2 for parts in snippets), 'Unexpected guide structure'
    blocks = [b for parts in snippets for b in parts]
    result = subprocess.run(['docker', 'run', '--rm', '-i', '--network', 'none',
        '--read-only', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
        '--memory', '256m', '--pids-limit', '64', '--entrypoint', 'python3',
        IMAGE, '-c', CHECK], input=json.dumps(blocks)+'\n', text=True,
        capture_output=True, timeout=90)
    (root / 'functional-review.log').write_text(result.stdout + result.stderr)
    result.check_returncode()
    data = json.loads(result.stdout)
    data.update(image=IMAGE, byte_exact=False,
        scope='Generated LRU guide only: reference-model sequential operations; not proof of concurrency or full model quality')
    (root / 'functional-review.json').write_text(json.dumps(data, indent=2)+'\n')
    return data


if __name__ == '__main__':
    print(json.dumps(review(Path(sys.argv[1])), indent=2))
