"""Source reconstruction and CPU metadata controls; no native compilation."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import full_cache_observer38_contract_v1 as contract
import prepare_full_cache_observer38_v1 as source


class SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old, cls.new = source.reconstruct()

    def test_exact_pristine_patch_reconstruction(self):
        with tempfile.TemporaryDirectory(prefix='source38-cpu-', dir='/tmp') as d:
            root = Path(d)
            for name, text in self.old.items():
                p = root/name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(text)
            patch = source.patch_bytes()
            self.assertEqual(patch, source.PATCH.read_bytes())
            run = subprocess.run(['patch', '--batch', '--fuzz=0', '-p1'], input=patch,
                                 cwd=root, capture_output=True)
            self.assertEqual(run.returncode, 0, run.stderr.decode('ascii'))
            for name, text in self.new.items():
                self.assertEqual((root/name).read_bytes(), text.encode('utf-8'), name)

    def test_exact_plan_and_caps(self):
        plan = source.build_plan()
        self.assertEqual(len(plan['expected_patched_source_sha256']), 65)
        self.assertEqual(len(plan['added_header_payloads']), 29)
        self.assertEqual(len(plan['patches']), 38)
        self.assertEqual(set(plan['overlay_files']), set(plan['expected_patched_source_sha256']))
        self.assertFalse(plan['actual_full_cache38_runtime_qualified'])
        c = self.new['sycl/include/strata/core/batch_fidelity_contract.hpp']
        self.assertIn('REQUESTS=6, ROWS=6', c)
        self.assertIn('BYTE_LIMIT=64*1024*1024', c)
        b = self.new['sycl/include/strata/core/batch_fidelity_observer.hpp']
        self.assertIn('GRAPH_LIMIT=16,EPOCHS=ROWS*(LAYERS+1)', b)

    def test_default_off_has_no_snapshot_or_proc_work(self):
        h = self.new[source.H]
        self.assertIn('if(!flag||!*flag||!std::strcmp(flag,"0"))return;', h)
        self.assertIn('capture(const Entries& entries){if(!enabled())return {};', h)
        self.assertLess(h.index('if(!enabled())return {};'), h.index('std::make_unique<registry_capture>'))
        self.assertIn('host_memory(){if(!enabled())return;FILE*', h)
        self.assertIn('if(!enabled())return;std::lock_guard', h)

    def test_global_include_and_no_snapshot_payload_read(self):
        g = self.new['sycl/src/program/generate.cpp']
        self.assertLess(g.index('#include "strata/core/full_cache_observer38.hpp"'), g.index('namespace {'))
        h = self.new[source.H]
        self.assertNotIn('sycl::', h)
        self.assertNotIn('memcpy', h)
        self.assertNotIn('wait_and_throw', h)
        self.assertIn('raw_requested', h)
        self.assertNotIn('raw_selected', h)
        self.assertIn('full_cache38::enabled() && !strata::core::batch_fidelity::settings().enabled', g)

    def test_direct_gen_separate_handle_lifetime_and_truthful_phase(self):
        v = self.new['sycl/src/core/verify.cpp']
        self.assertEqual(v.count('batch_phase_==3?batch_direct_exec_'), 2)
        self.assertLess(v.index('for(auto& p:batch_direct_exec_)'), v.index('batch_snapshot_.reset()'))
        c = self.new['sycl/include/strata/core/batch_fidelity_contract.hpp']
        self.assertIn('phase>3', c)
        self.assertIn('id.slot!=ROWS||id.slotgen!=0||id.event!=0', c)
        self.assertIn('return !e->direct_done;', c)
        self.assertIn('else e->direct_done=true;', c)
        b = self.new['sycl/include/strata/core/batch_fidelity_observer.hpp']
        self.assertIn('normal_dispatch=GEN observed_only=1', b)
        self.assertIn('direct_gen_residual', b)
        self.assertIn('if(full_cache38::enabled()&&!full_cache38::selected(rid))return {};', b)

    def test_victim_instrumentation_does_not_choose_victim(self):
        c = self.new['include/strata/core/conversation_cache.hpp']
        old = self.old['include/strata/core/conversation_cache.hpp']
        for line in old.splitlines():
            if 'std::find_if' in line or 'conversation_prefix(' in line:
                self.assertIn(line, c)
        self.assertLess(c.index('full_cache38::parked("before_eviction"'), c.index('entries_.erase(victim);'))
        self.assertGreater(c.index('full_cache38::parked("after_eviction"'), c.index('entries_.erase(victim);'))
        self.assertIn('cancelled?pp_reached:n,n,produced_n,cancelled||std::strcmp(finish,"cancel")==0', self.new['sycl/src/program/generate.cpp'])


class MetadataTests(unittest.TestCase):
    def test_default_off_ignores_unused_selector(self):
        for flag in (None, '', '0'):
            self.assertEqual(contract.capture_rids(flag, 'garbage'), ())

    def test_canonical_selection(self):
        self.assertEqual(contract.capture_rids('1', '1,2,3,4,5,18446744073709551615'), (1,2,3,4,5,2**64-1))
        for flag, ids in [('2','1'),('1',''),('1','0'),('1','01'),('1','+1'),('1','1,1'),('1','1,'),('1','1,2,3,4,5,6,7'),('1',str(2**64))]:
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                contract.capture_rids(flag, ids)

    def pair(self):
        rows = [dict(index=i, instance=i+10, snapshot_bytes=64*(i+1), pinned=False) for i in range(3)]
        before = dict(kind='parked_inventory', phase='before_eviction', event=2, pid=44,
                      command=1, rid=6, actual_victim=1, entries=rows, retained_bytes=384,
                      physical_allocator_reclamation_qualified=False)
        after = dict(before, phase='after_eviction', event=3, actual_victim=-1,
                     entries=[rows[0], dict(rows[2], index=1)], retained_bytes=256)
        return before, after

    def test_real_victim_and_bytes_are_observations(self):
        before, after = self.pair()
        result = contract.victim_binding(before, after)
        self.assertEqual(result['actual_victim']['instance'], 11)
        self.assertEqual(result['logical_bytes_removed'], 128)
        self.assertFalse(result['actual_full_cache_qualified'])

    def test_victim_mutation_rejected(self):
        before, after = self.pair()
        for key, value in [('pid',45),('command',2),('event',1),('retained_bytes',0),('actual_victim',0),('physical_allocator_reclamation_qualified',True)]:
            changed = dict(after, **{key:value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                contract.victim_binding(before, changed)
        changed = copy.deepcopy(after)
        changed['entries'][1]['instance'] = 999
        with self.assertRaises(ValueError):
            contract.victim_binding(before, changed)

    def test_no_fabricated_pinned_victim(self):
        before, after = self.pair()
        before['entries'][1]['pinned'] = True
        with self.assertRaises(ValueError):
            contract.victim_binding(before, after)

    def test_original_global_log_indices_and_malformed_fail(self):
        text = 'engine line\nFC38 '+json.dumps(dict(kind='request_begin'))+'\nUR line\n'
        self.assertEqual(contract.records(text)[0]['original_line_index'], 1)
        with self.assertRaises(json.JSONDecodeError):
            contract.records('FC38 {truncated')


if __name__ == '__main__':
    unittest.main()
