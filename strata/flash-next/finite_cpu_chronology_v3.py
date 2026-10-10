"""Actual sixteen original case receipts define the CPU source boundary."""
import math
from pathlib import Path
from serial37_canonical_json_v3 import canonical, read_unique


def require(ok, message):
    if not ok:
        raise ValueError(message)


def epoch(value):
    require(type(value) in (int, float) and math.isfinite(value) and value > 0,
            'Finite typed actual CPU/source epoch required')
    return value


def admit(root, report):
    root = Path(root)
    rows = report['cases']
    require(len(rows) == 16, 'Complete actual all16 original cases required')
    roster = [(case, repeat) for case in range(8) for repeat in (0, 1)]
    actual = []
    for row, (case, repeat) in zip(rows, roster):
        require(type(row['case']) is int and type(row['repeat']) is int
                and (row['case'], row['repeat']) == (case, repeat), 'Exact actual sequential case roster required')
        saved = read_unique(root / ('case' + str(case) + '-repeat' + str(repeat)) / 'case-receipt.json')
        require(canonical(saved) == canonical(row), 'Parent case contradicts original typed receipt')
        require(saved['passed'] is True and saved['container_removed'] is True
                and saved['container_terminal_observed'] is True and not saved['errors']
                and not saved['memory_errors'], 'Original complete case lifecycle required')
        start, terminal = epoch(saved['started_epoch']), epoch(saved['finished_epoch'])
        require(start <= terminal, 'Original case terminal precedes start')
        pre = read_unique(root / ('case' + str(case) + 'repeat' + str(repeat) + '-pre-source-pages.json'))
        post = read_unique(root / ('case' + str(case) + 'repeat' + str(repeat) + '-post-source-pages.json'))
        require(pre['passed'] is True and post['passed'] is True
                and epoch(pre['epoch']) <= start <= terminal <= epoch(post['epoch']),
                'Original case source pages must bracket actual case')
        actual.append((epoch(pre['epoch']), start, terminal, epoch(post['epoch'])))
    pre_hash = read_unique(root / 'pre-full-four.json')
    post_hash = read_unique(root / 'post-terminal-full-four.json')
    before_pre = read_unique(root / 'pre-hash-source-pages.json')
    after_pre = read_unique(root / 'pre-inference-source-pages.json')
    before_post = read_unique(root / 'post-terminal-pre-hash-source-pages.json')
    after_post = read_unique(root / 'post-terminal-post-hash-source-pages.json')
    require(all(v['passed'] is True for v in (pre_hash, post_hash, before_pre, after_pre, before_post, after_post)),
            'Actual complete pre/post source/page receipts required')
    order = [epoch(report['started_epoch']), epoch(before_pre['epoch']),
             epoch(pre_hash['started_epoch']), epoch(pre_hash['finished_epoch']), epoch(after_pre['epoch'])]
    for bounds in actual:
        order.extend(bounds)
    last = max(bounds[2] for bounds in actual)
    require(type(report['last_case_terminal_epoch']) in (int, float)
            and epoch(report['last_case_terminal_epoch']) == last, 'Saved terminal must equal actual16 original receipt maximum')
    order.extend([epoch(before_post['epoch']), epoch(post_hash['started_epoch']),
                  epoch(post_hash['finished_epoch']), epoch(after_post['epoch']), epoch(report['finished_epoch'])])
    require(all(a <= b for a, b in zip(order, order[1:])), 'Actual pre/all16/post hash/page chronology changed')
    return {'actual_original_case_count': 16, 'derived_last_case_terminal_epoch': last,
            'original_receipts_and_page_brackets_verified': True,
            'parent_terminal_alone_is_authority': False}
