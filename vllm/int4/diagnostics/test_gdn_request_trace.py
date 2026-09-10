"""CPU-only metadata fixtures and source-shape checks; no torch import."""
import ast
import copy
import importlib.util
import json
import pathlib
import tempfile
import types
import unittest

ROOT = pathlib.Path(__file__).parent


class CPU:
    device = types.SimpleNamespace(type="cpu")

    def __init__(self, values):
        self.values = values

    def tolist(self):
        return list(self.values)


class TraceTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("trace", ROOT / "gdn_request_trace.py")
        self.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.m)
        self.tmp = tempfile.TemporaryDirectory()
        self.m.TRACE_DIR = self.tmp.name
        self.events = []
        self.m._emit = self.events.append

    def tearDown(self):
        self.tmp.cleanup()

    def prepare(self, computed=50047, prompt=50048, lengths=None, synced=False):
        lengths = lengths or [1]
        n = len(lengths)
        ids = ["request-%d" % i for i in range(n)]
        batch = types.SimpleNamespace(num_reqs=n, req_ids=ids,
                                      num_computed_tokens_cpu=[computed]*n,
                                      num_prompt_tokens=[prompt]*n)
        runner = types.SimpleNamespace(input_batch=batch, use_async_spec_decode=True,
            num_accepted_tokens=types.SimpleNamespace(np=[1]*n),
            cache_config=types.SimpleNamespace(mamba_cache_mode="align"))
        scheduler = types.SimpleNamespace(
            scheduled_new_reqs=[types.SimpleNamespace(req_id=x, num_computed_tokens=computed) for x in ids],
            finished_req_ids=set(), num_scheduled_tokens=dict(zip(ids,lengths)))
        self.m.snapshot(runner,scheduler,lengths,synced)
        offsets=[0]
        for length in lengths: offsets.append(offsets[-1]+length)
        meta=types.SimpleNamespace(query_start_loc_cpu=CPU(offsets),
            is_prefilling=CPU([computed<prompt]*n))
        self.m.bind_metadata(runner,meta,False)
        return runner,scheduler,meta

    def test_long_prefix_one_token_correlates_to_actual_bad_route(self):
        _,_,meta=self.prepare()
        self.m.classification(meta,1,0,0,None,0)
        e=self.events[-1];row=e['rows'][0]
        self.assertTrue(e['mapped_target'])
        self.assertEqual(row['initial_computed_host'],50047)
        self.assertEqual(row['remaining_prompt_host'],1)
        self.assertEqual(row['req_id'],'request-0')
        self.assertEqual(row['route'],'recurrent-decode')
        self.assertIn('not-authoritative',e['accepted_source'])

    def test_fixed_route_and_shallow_group_copy(self):
        _,_,meta=self.prepare(synced=True)
        self.m.classification(copy.copy(meta),0,1,0,None,0)
        self.assertEqual(self.events[-1]['rows'][0]['route'],'initializing-prefill')
        self.assertIn('synchronized',self.events[-1]['accepted_source'])

    def test_decode_and_long_prefill_do_not_emit(self):
        _,_,meta=self.prepare(computed=50048)
        self.m.classification(meta,1,0,0,None,0)
        _,_,meta=self.prepare(computed=0,lengths=[128])
        self.m.classification(meta,0,1,0,None,0)
        self.assertEqual(self.events,[])

    def test_draft_or_split_copy_is_unmapped(self):
        _,_,meta=self.prepare()
        meta.query_start_loc_cpu=CPU([0,1])
        self.m.classification(meta,1,0,0,None,0)
        self.assertFalse(self.events[-1]['mapped_target'])
        self.assertNotIn('req_id',self.events[-1]['rows'][0])

    def test_spec_mixed_prefill_route(self):
        _,_,meta=self.prepare(lengths=[4,1])
        self.m.classification(meta,0,1,1,CPU([True,False]),0)
        self.assertEqual(self.events[-1]['rows'][0]['route'],'initializing-prefill')

    def test_capture_does_not_inherit_target_ids(self):
        runner,_,meta=self.prepare()
        self.m.bind_metadata(runner,meta,True)
        self.m.classification(meta,1,0,0,None,0)
        self.assertFalse(self.events[-1]['mapped_target'])

    def test_boundary_origin_survives_chunk_then_cleanup(self):
        runner,sched,_=self.prepare(computed=49984,prompt=50048,lengths=[63])
        sched.scheduled_new_reqs=[]
        runner.input_batch.num_computed_tokens_cpu=[50047]
        sched.num_scheduled_tokens={'request-0':1}
        self.m.snapshot(runner,sched,[1],False)
        meta=types.SimpleNamespace(query_start_loc_cpu=CPU([0,1]),is_prefilling=CPU([True]))
        self.m.bind_metadata(runner,meta,False)
        self.m.classification(meta,1,0,0,None,0)
        self.assertEqual(self.events[-1]['rows'][0]['initial_computed_host'],49984)
        sched.finished_req_ids={'request-0'}
        self.m.snapshot(runner,sched,[1],False)
        self.assertEqual(self.m._origins,{})

    def test_non_cpu_rejected_without_transfer(self):
        _,_,meta=self.prepare()
        meta.query_start_loc_cpu.device=types.SimpleNamespace(type='xpu')
        self.m.classification(meta,1,0,0,None,0)
        self.assertEqual(self.events[-1]['event'],'trace-disabled')

    def test_no_device_or_synchronization_calls_in_trace(self):
        tree=ast.parse((ROOT/'gdn_request_trace.py').read_text())
        forbidden={'item','cpu','cuda','xpu','to','synchronize','query','numpy'}
        calls={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
        self.assertFalse(calls & forbidden)
        self.assertNotIn('prompt_token_ids',(ROOT/'gdn_request_trace.py').read_text())


if __name__=='__main__':
    unittest.main()
