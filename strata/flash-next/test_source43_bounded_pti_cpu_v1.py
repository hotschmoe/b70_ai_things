import unittest,copy,ast,inspect
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path
import run_source43_bounded_pti_v1 as a
import collect_source43_sampled_model_pti_v1 as c
import source43_sampled_pti_runtime_v1 as runtime
import pti_sampled_census_exec_v1 as census
class Tests(unittest.TestCase):
    def test_actual_recipe_and_old_untouched(self):
        ids={'short':list(range(54)),'prefill_about512':list(range(524))}
        original={'native_env':{},'native_argv':['/build/strata','--serve'],'requests':[dict(ordinal=i+1,ids=ids[k],fresh=1,pin=None,max_new=64,role=k)for i,k in enumerate(('short','prefill_about512','prefill_about512','short'))]}
        tool=SimpleNamespace(inherited_env=lambda env:None,CLOCK_ENV={'UNITRACE_CLOCK_QUERY_SAMPLE_512':'1'},NATIVE_ENV={'STRATA_DIAGNOSTIC_PTI_FINALIZE':'1'},OPTIONS=['--output-dir-path','/results/unitrace'])
        with patch.object(a,'tool',tool),patch.object(a,'_RUNTIME',{'image':'same'}),patch.object(a.base,'phase_recipe',side_effect=lambda *x:({},copy.deepcopy(original))):
            _,off=a.phase_recipe(a.PHASES[0],{'runtime':{'image':'same'}},{'env':{}},ids,'binding')
            _,on=a.phase_recipe(a.PHASES[1],{'runtime':{'image':'same'}},{'env':{}},ids,'binding')
        self.assertEqual([x['max_new']for x in off['requests']],[4]*4);self.assertEqual(off['requests'],on['requests']);self.assertNotIn('UNITRACE_CLOCK_QUERY_SAMPLE_512',off['native_env']);self.assertEqual(on['native_env']['UNITRACE_CLOCK_QUERY_SAMPLE_512'],'1');self.assertEqual(original['requests'][0]['max_new'],64)
    def test_original_normalize_code_unchanged(self):
        inherited=c.inherited.original
        self.assertIs(c.inherited.original.normalize_clock,inherited.normalize_clock)
        fn=c.clock_port({});self.assertIs(fn.__globals__['normalize_clock'],inherited.normalize_clock)
        self.assertIs(fn.__globals__['bridge_grammar'],c.sampled_bridge)
    def test_no_original_globals_mutated(self):
        old=c.inherited.original;before=dict(vars(old));c.clock_port({});self.assertEqual(before,dict(vars(old)))
    def test_exact_new_summary_key_extension(self):
        old=c.inherited.original;node=copy.deepcopy(next(n for n in ast.parse(Path(old.__file__).read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name=='clock_binding'));new=c.extend_clock(node)
        keys=next(n.value for n in ast.walk(new)if isinstance(n,ast.Assign)and isinstance(n.targets[0],ast.Name)and n.targets[0].id=='keys');self.assertIn('query_sampling_summary',[k.value for k in keys.keys])
    def test_complete_actual_recipe_sampling_flag_native_pid(self):
        source=inspect.getsource(a.native_pid);self.assertIn("'tool':tool",source);self.assertIn('STRATA_DIAGNOSTIC_PTI_FINALIZE=1',source)
    def test_fresh_btrfs_mount_exact(self):
        command=census.command('a'*64,'after_READY','b'*64);self.assertIn(census.EXPECTED_BIND_ROOT,command);self.assertNotIn(census.old.EXPECTED_BIND_ROOT,command)
    def test_wrong_intake_sha_refuses_first(self):
        with patch.object(runtime,'regular',side_effect=ValueError('wrong intake'))as check:
            with self.assertRaises(ValueError):runtime.binding('/tmp/intake','wrong')
        self.assertEqual(check.call_args.args, (Path('/tmp/intake'),'wrong',1<<20))
    def test_cli_requires_intake_hash(self):
        self.assertIn("--pti-intake-sha256",Path(a.__file__).read_text());self.assertIn('set_runtime(a.pti_intake,a.pti_intake_sha256)',Path(a.__file__).read_text())
    def bridge_fixture(self):
        rows=[dict(kind='host_clock_origin'),dict(kind='host_clock_bridge',site=1,raw_before_ns=1,raw_after_ns=2,monotonic_ns=3),dict(kind='global_timestamp_query_bracket',query_site=1,device_handle=9,return_code=0,returned_pair_present=1,host_ns=100,device_ticks=4,raw_before_ns=10,raw_after_ns=20),dict(kind='host_clock_bridge',site=101,raw_before_ns=30,raw_after_ns=35,monotonic_ns=40),dict(kind='device_origin',device_handle=9,host_ns=100,device_ticks=4),dict(kind='instance_normalization',instance_id=77)]
        for site in range(1,7):
            seen=1 if site==1 else 0
            rows.append(dict(kind='query_sampling_summary',query_site=site,sample_every=1 if site<=2 else 512,original_query_calls=seen,selected_brackets=seen,completed_brackets=seen,suppressed_brackets=0))
        rows.append(dict(kind='host_clock_bridge',site=2,raw_before_ns=50,raw_after_ns=55,monotonic_ns=60))
        return [dict(r,pid=1,writer_tid=1,sequence=i+1)for i,r in enumerate(rows)]
    def test_sampled_origin_and_thread_predicates(self):
        rows=self.bridge_fixture();result=c.sampled_bridge(rows);self.assertFalse(result['every_operation_submit_pair_bracketed']);self.assertEqual(len(result['device_origin_query_joins']),1)
        rows[4]['host_ns']=101
        with self.assertRaises(ValueError):c.sampled_bridge(rows)
    def test_postquery_bridge_retains_same_thread(self):
        rows=self.bridge_fixture();rows[3]['writer_tid']=2
        with self.assertRaises(ValueError):c.sampled_bridge(rows)
if __name__=='__main__':unittest.main()
