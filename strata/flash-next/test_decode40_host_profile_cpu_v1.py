import copy
from pathlib import Path
import unittest
import prepare_decode40_host_profile_v1 as p
import prefill_chunk64_512_control_v2 as c

class Controls(unittest.TestCase):
    def fixture(self):
        cfg=c.expected_config('chunk64',{'inherited_math_env':{}},18082)
        return {'cards':[0],'alias':cfg['aliases'][0]},cfg,{'short':[1,2],'prefill_about512':[3]*524}
    def test_exact_only_host_observation_changes(self):
        m,cfg,ids=self.fixture();a=p.build(m,cfg,ids,'a'*64,'OFF');b=p.build(m,cfg,ids,'a'*64,'ON')
        self.assertEqual(a['native_argv'],b['native_argv']);self.assertEqual(a['requests'],b['requests']);self.assertEqual(a['protocol_sha256'],b['protocol_sha256'])
        self.assertEqual(set(b['native_env'])-set(a['native_env']),{'STRATA_DECODE_TIMING','STRATA_CRITICAL_PATH_TRACE','STRATA_CRITICAL_PATH_TRACE_DIR','STRATA_CRITICAL_PATH_TRACE_BINDING_SHA256'})
        self.assertEqual(len(a['requests']),4);self.assertTrue(all(r['max_new']==64 for r in a['requests']))
    def test_pair_and_bool_cards_rejected(self):
        for cards in ([False],[0,1]):
            m,cfg,ids=self.fixture();m['cards']=cards
            with self.assertRaises(ValueError):p.build(m,cfg,ids,'a'*64,'ON')
    def test_heavy_profile_and_math_rejected(self):
        for k,v in [('STRATA_VERIFY_PROFILE','0'),('STRATA_VERIFY_EAGER','0'),('STRATA_HC_Q8','1'),('SYCL_UR_TRACE','2')]:
            m,cfg,ids=self.fixture();cfg['env'][k]=v
            with self.assertRaises(ValueError):p.build(m,cfg,ids,'a'*64,'ON')
    def test_input_limits_and_types(self):
        for ids in ({'short':[True],'prefill_about512':[2]}, {'short':[1]*1985,'prefill_about512':[2]}):
            m,cfg,_=self.fixture()
            with self.assertRaises(ValueError):p.build(m,cfg,ids,'a'*64,'ON')

    def test_collector_exact_single_predicate_scope_change(self):
        import ast
        import collect_decode40_host_trace_v1 as reader
        text=Path(reader.__file__).read_text()
        self.assertIn('original.require(count==1',text)
        self.assertIn('namespace=dict(vars(original))',text)
        # Original raw token/graph/ownership/printf/footer checks stay unchanged.
        old=Path(reader.original.__file__).read_text()
        for phrase in ('Actual token printf end/ownership coverage differs','Emitted token coverage differs','Missing/failed terminal host tracer ownership/free boundary'):
            self.assertIn(phrase,old)
        ast.parse(text)

    def test_actual_decode_collector_accepts_scoped_batched_prompt(self):
        import json,tempfile
        import collect_decode40_host_trace_v1 as reader
        from test_host_critical_path36_cpu_v1 import HostTests,K
        with tempfile.TemporaryDirectory() as t:
            path,log,request=HostTests().fixture(Path(t));rows=[json.loads(x)for x in path.read_text().splitlines()]
            events=[e for e in rows[1:-1]if not(e['dev']==0 and e['pos']in(0,1))]
            for seq,e in enumerate(events):
                e['seq']=seq;e['ns']=100+seq*100
                if e['kind']==K['GraphHit']:e['kind']=K['GraphPreexisting']
            rows[0]['events']=len(events);rows[-1]['records']=len(events)
            def save():path.write_text('\n'.join(json.dumps(x)for x in [rows[0],*events,rows[-1]])+'\n')
            save()
            with self.assertRaises(ValueError):reader.original.collect([path],log,'a'*64,[request],[[0,0,48]])
            value=reader.collect([path],log,'a'*64,[request],[[0,0,48]])
            self.assertTrue(value['passed']);self.assertFalse(value['prompt_internal_device_time_observed'])
            next(e for e in events if e['kind']==K['TEmit'])['value']=17;save()
            with self.assertRaises(ValueError):reader.collect([path],log,'a'*64,[request],[[0,0,48]])

if __name__=='__main__':unittest.main()
