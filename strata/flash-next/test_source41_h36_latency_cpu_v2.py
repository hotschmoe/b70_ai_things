import copy,hashlib,json,tempfile,unittest
from pathlib import Path
import analyze_source41_h36_latency_v2 as a
import test_source41_host_trace_contract_cpu_v2 as fixture
class Controls(unittest.TestCase):
    def test_mutated_trace_and_refreshed_local_coverage_refused_by_original_roster(self):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp);phase=root/'source41-host-ON';phase.mkdir();trace=phase/'host-trace36-1.jsonl';coverage=phase/'host-coverage.json'
            trace.write_bytes(b'{"ns":100}\n');coverage.write_bytes(b'{"trace_sha256":"original"}\n')
            report={'artifacts':{str(p.relative_to(root)):a.bound(p)['sha256']for p in(trace,coverage)}}
            self.assertEqual(a.original_artifact(root,trace,report)['bytes'],11)
            trace.write_bytes(b'{"ns":101}\n');coverage.write_text(json.dumps({'trace_sha256':a.bound(trace)['sha256']})+'\n')
            self.assertEqual(json.loads(coverage.read_text())['trace_sha256'],a.bound(trace)['sha256'])
            for path in(trace,coverage):
                with self.assertRaisesRegex(ValueError,'closed report artifact'):a.original_artifact(root,path,report)
    def test_plan_roster_cannot_replace_separate_original_plan_sha(self):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp);path=root/'plan.snapshot.json';path.write_bytes(b'{"source":41}\n');old=a.bound(path)['sha256'];report={'artifacts':{'plan.snapshot.json':old},'plan_sha256':old};a.original_plan(root,report)
            path.write_bytes(b'{"source":40}\n');report['artifacts']['plan.snapshot.json']=a.bound(path)['sha256']
            with self.assertRaisesRegex(ValueError,'parent plan SHA'):a.original_plan(root,report)
    def test_analysis_own_source_plan_admitted(self):
        proof=a.source_binding();self.assertEqual(proof['sha256'],hashlib.sha256(a.PLAN.read_bytes()).hexdigest())
    def test_overlap_union_not_nested_sum(self):
        self.assertEqual(a.union([(10,30),(15,25),(25,40),(50,60)])['union_ns'],40)
        self.assertEqual(a.union([(10,30),(15,25)],(20,35))['union_ns'],10)
    def test_nearest_rank_small_and_empty(self):
        self.assertEqual(a.quantile([1,2,3,100],50),2);self.assertEqual(a.quantile([1,2,3,100],95),100);self.assertIsNone(a.quantile([],95))
    def fixture(self):
        def e(ns,kind,edge,span=0):return {'ns':ns,'kind':a.K[kind],'edge':edge,'span':span}
        events=[e(10,'Request',1,1),e(20,'ReplayWait',1,2),e(30,'ReplayWait',2,2),e(35,'TEmit',0),e(40,'LPReadback',1,3),e(50,'LPReadback',2,3),e(50,'LPCPU',1,4),e(60,'LPCPU',2,4),e(75,'TEmit',0),e(80,'DoneEmit',1,5),e(85,'DoneEmit',2,5),e(90,'Request',2,1)]
        for i,e in enumerate(events):e['seq']=i
        return [{},*events,{}]
    def test_native_ttft_completion_and_token_segmentation(self):
        v=a.metrics(self.fixture());self.assertEqual(v['native_TTFT_ns'],25);self.assertEqual(v['request_completion_ns'],80);self.assertEqual(v['TEmit_gaps_ns'],[40]);self.assertEqual(v['last_T_to_request_end_ns'],15)
        self.assertEqual(v['token_segments'][1]['LP_combined']['union_ns'],20);self.assertEqual(v['token_segments'][0]['LP_combined']['union_ns'],0);self.assertEqual(v['kind_unions']['ReplayWait']['union_ns'],10)
    def test_span_crossing_token_boundary_clipped_without_loss(self):
        intervals=[(20,80)];pieces=[a.union(intervals,w)['union_ns']for w in((10,35),(35,75),(75,90))];self.assertEqual(pieces,[15,40,5]);self.assertEqual(sum(pieces),a.union(intervals)['union_ns'])
    def test_clock_and_partial_span_refusals(self):
        rows=self.fixture();rows[3]['ns']=19
        with self.assertRaises(ValueError):a.metrics(rows)
        rows=self.fixture();rows[-2]['edge']=0
        with self.assertRaises(ValueError):a.metrics(rows)
        with self.assertRaises(ValueError):a.union([(True,4)])
    def test_fresh_original_strict_collector_before_actual_fixture_metrics(self):
        with tempfile.TemporaryDirectory()as root:
            p,l,r,rows=fixture.StrictControls().fixture(Path(root),True);coverage=a.strict.collect([p],l,'a'*64,[r],[[0,0,48]],1);self.assertTrue(coverage['passed']);admitted=a.strict.typed_rows(p,1)[0];v=a.metrics(admitted);self.assertEqual(v['token_ns'],coverage['requests'][0]['token_ns']);self.assertGreater(v['native_TTFT_ns'],0)
    def test_input_hash_stat_mutation_refused_by_boundary(self):
        with tempfile.TemporaryDirectory()as root:
            p=Path(root)/'raw';p.write_bytes(b'abc');old=a.bound(p);p.write_bytes(b'abd');self.assertNotEqual(old,a.bound(p))
if __name__=='__main__':unittest.main()
