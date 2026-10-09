"""Pristine source34 and sequential verifier-window controls; no native/GPU proof."""
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class LastWindow34(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = json.loads((HERE / 'current-ple-lastwindow34-engine-build-plan-v1.json').read_text())
        cls.oldplan = json.loads((HERE / 'current-ple-input33-engine-build-plan-v1.json').read_text())
        cls.temp = tempfile.TemporaryDirectory(prefix='ple-lastwindow34-test-')
        cls.source = Path(cls.temp.name)
        needed = set(cls.plan['source_file_sha256'])
        for item in cls.plan['patches']:
            text = (ROOT / item['path']).read_text()
            needed.update(re.findall(r'^\+\+\+ b/(.*)$', text, re.M))
            needed.update(re.findall(r'^--- a/(.*)$', text, re.M))
        base = Path(cls.plan['source_root'])
        for rel in needed:
            path = base / rel
            if path.is_file():
                dest = cls.source / rel;dest.parent.mkdir(parents=True, exist_ok=True);shutil.copy2(path, dest)
        for rel, digest in cls.plan['source_file_sha256'].items():
            assert sha(cls.source / rel) == digest, rel
        for index, item in enumerate(cls.plan['patches']):
            path = ROOT / item['path'];assert sha(path) == item['sha256']
            result = subprocess.run(['patch', '--batch', '--forward', '-p1', '-i', str(path)], cwd=cls.source, capture_output=True, text=True)
            assert result.returncode == 0, result.stdout + result.stderr
            if index == 32:
                cls.oldverify = (cls.source / 'sycl/src/core/verify.cpp').read_text()
        cls.verify = (cls.source / 'sycl/src/core/verify.cpp').read_text()
        match = re.search(r'if\((r\.active && pos0 \+ T == static_cast<int64_t>\(r\.ids\.size\(\)\))\) \{', cls.verify)
        assert match, 'Actual final-window source predicate absent'
        cls.expression = match[1].replace('r.active', 'active').replace('&&', 'and').replace('static_cast<int64_t>(r.ids.size())', 'nids')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def reached(self, active, pos, count, window, enabled=True, precollected=True):
        return enabled and precollected and eval(self.expression, {'__builtins__': {}}, {'active': active, 'pos0': pos, 'T': window, 'nids': count})

    def windows(self, count, chunks, enabled=True, active=True, precollected=True):
        events = [];position = 0;frames = 0
        for window in chunks:
            # Source32 gather is outside the new publication predicate.
            if precollected:events.extend([('gather', position), ('fences', position)])
            if self.reached(active, position, count, window, enabled, precollected):
                frames += 1;events.extend([('published', position), ('submit_begin', position)])
            events.append(('graph', position))
            if self.reached(active, position, count, window, enabled, precollected):events.append(('submit_returned', position))
            position += window
        self.assertEqual(position, count)
        if frames:events.append(('terminal', position - chunks[-1]))
        return events, frames

    def test_pristine34_all63_source27_header_hashes(self):
        self.assertEqual(len(self.plan['patches']), 34)
        self.assertEqual(len(self.plan['expected_patched_source_sha256']), 63)
        self.assertEqual(len(self.plan['added_header_payloads']), 27)
        self.assertEqual(len(self.plan['build_targets']), 8)
        self.assertEqual(len(self.plan['runtime_python_sources']), 6)
        for rel, digest in self.plan['expected_patched_source_sha256'].items():self.assertEqual(sha(self.source / rel), digest, rel)
        for item in self.plan['added_header_payloads']:self.assertEqual(sha(self.source / item['path']), item['sha256'])

    def test_only_observer_publication_body_changes(self):
        changed = [name for name, digest in self.oldplan['expected_patched_source_sha256'].items() if self.plan['expected_patched_source_sha256'][name] != digest]
        self.assertEqual(changed, ['sycl/src/core/verify.cpp'])
        start = self.verify.index('    // Only after the actual successful32')
        end = self.verify.index('    if (trace_h_', start)
        oldstart = self.oldverify.index('    // Only after the actual successful32')
        oldend = self.oldverify.index('    if (trace_h_', oldstart)
        self.assertEqual(self.verify[:start] + self.verify[end:], self.oldverify[:oldstart] + self.oldverify[oldend:])

    def test_prefix1_2_4_8_sequentialT1_publish_once_after_final_gather(self):
        for count in (1, 2, 4, 8):
            events, frames = self.windows(count, [1] * count)
            self.assertEqual(frames, 1)
            final = count - 1
            self.assertEqual([pos for event, pos in events if event == 'published'], [final])
            last = [event for event, pos in events if pos == final]
            self.assertEqual(last, ['gather', 'fences', 'published', 'submit_begin', 'graph', 'submit_returned', 'terminal'])
            self.assertEqual(sum(event == 'gather' for event, _ in events), count)
            self.assertEqual(sum(event == 'graph' for event, _ in events), count)

    def test_earlier_windows_gather_and_graph_without_markers(self):
        events, _ = self.windows(8, [1] * 8)
        for position in range(7):self.assertEqual([event for event, pos in events if pos == position], ['gather', 'fences', 'graph'])

    def test_four_request_frame_quota_not_spent_by_earlier_windows(self):
        self.assertEqual(sum(self.windows(count, [1] * count)[1] for count in (1, 2, 4, 8)), 4)

    def test_defaultOFF_preserves_all_gathers_graphs_zero_publications(self):
        for count in (1, 2, 4, 8):
            events, frames = self.windows(count, [1] * count, enabled=False)
            self.assertEqual(frames, 0)
            self.assertEqual(len(events), count * 3)
            self.assertTrue(all(event in ('gather', 'fences', 'graph') for event, _ in events))

    def test_unarmed_warmup_zero_publications(self):
        for count in (1, 2, 4, 8):self.assertEqual(self.windows(count, [1] * count, active=False)[1], 0)

    def test_other_route_no_new_publication_or_gather(self):
        events, frames = self.windows(8, [1] * 8, precollected=False)
        self.assertEqual(frames, 0);self.assertEqual(events, [('graph', n) for n in range(8)])

    def test_actual_source_order_and_empty_publication_submit_noops(self):
        block = self.verify[self.verify.index('const bool ple_precollected'):self.verify.index('trace_ev("LAUNCHED"')]
        order = ['gather_batch', '_mm_sfence', 'ple_input33::publication', 'r.active && pos0 + T', 'ple_input33::publish', 'ple_input33::submit_begin', 'ext_oneapi_graph', 'ple_input33::submit_returned']
        self.assertEqual(sorted(order, key=block.index), order)
        header = (self.source / 'include/strata/core/ple_input_observer33.hpp').read_text()
        self.assertIn('struct publication {uint64_t epoch=0;', header)
        self.assertIn('submit_begin(const publication&p){if(!p.epoch)return;', header)
        self.assertIn('submit_returned(const publication&p,int rc){if(!p.epoch)return;', header)
        self.assertEqual(sha(self.source / 'include/strata/core/ple_input_observer33.hpp'), self.oldplan['expected_patched_source_sha256']['include/strata/core/ple_input_observer33.hpp'])

    def test_final_multirow_window_allowed_earlier_suppressed(self):
        for count, chunks in ((2, [2]), (4, [2, 2]), (8, [3, 3, 2])):
            events, frames = self.windows(count, chunks)
            self.assertEqual(frames, 1)
            self.assertEqual([pos for event, pos in events if event == 'published'], [count - chunks[-1]])

    def test_no_past_end_or_empty_SFD_publication(self):
        for active, pos, count, window in ((True, 2, 2, 1), (True, 0, 0, 1), (False, 0, 1, 1)):
            self.assertFalse(self.reached(active, pos, count, window))


if __name__ == '__main__':unittest.main()
