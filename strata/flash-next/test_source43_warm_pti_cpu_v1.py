import unittest,tempfile,struct,copy,os,json
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import pti_owned_temporal_control_v1 as h
import pti_warm_gen4_contract_v1 as w
import run_source43_warm_pti_v1 as a
import validate_source43_warm_pti_v1 as v
class Tests(unittest.TestCase):
    def test_real_ipc_fixture_and_mutation(self):
        name=h.session_name(99,123,'a'*64)
        with tempfile.TemporaryDirectory()as d:
            p=Path(d)/('uctrl'+name);p.write_bytes(struct.pack('<i',1)+bytes(1020));before=h.snapshot(name,Path(d));p.write_bytes(struct.pack('<i',0)+bytes(1020));after=h.snapshot(name,Path(d));h.check_transition(before,after,'resume')
            after['raw_hex']=(struct.pack('<i',0)+b'x'+bytes(1019)).hex()
            with self.assertRaises(ValueError):h.check_transition(before,after,'resume')
    def test_ipc_alias_and_geometry(self):
        name=h.session_name(99,123,'a'*64)
        with tempfile.TemporaryDirectory()as d:
            p=Path(d)/('uctrl'+name);target=Path(d)/'target';target.write_bytes(bytes(1024));p.symlink_to(target)
            with self.assertRaises(ValueError):h.snapshot(name,Path(d))
            p.unlink();p.write_bytes(bytes(4))
            with self.assertRaises(ValueError):h.snapshot(name,Path(d))
    def test_unique_owner_and_types(self):
        self.assertNotEqual(h.session_name(9,1,'a'*64),h.session_name(9,2,'a'*64))
        with self.assertRaises(ValueError):h.session_name(True,1,'a'*64)
    def graph_rows(self,selected=False):
        K=v.h36.K;context=dict(dev=0,lb=0,le=48,T=1,queue=1,graph=2)
        return [{},dict(context,kind=K['GraphHit'if selected else 'GraphPreexisting'],edge=0),dict(context,kind=K['ReplaySubmit'],edge=1),{}]
    def test_actual_graph_identity(self):
        self.assertTrue(w.graph_reuse(self.graph_rows(),self.graph_rows(True),v.h36.K)['actual_same_process_graph_reuse'])
        changed=self.graph_rows(True);changed[2]['graph']=3
        with self.assertRaises(ValueError):w.graph_reuse(self.graph_rows(),changed,v.h36.K)
    def test_selected_recapture_refused(self):
        selected=self.graph_rows(True);selected[1]['kind']=v.h36.K['GraphPreexisting']
        with self.assertRaises(ValueError):w.graph_reuse(self.graph_rows(),selected,v.h36.K)
    def test_two_actual_case_rows_short_only(self):
        ids={'short':list(range(54)),'prefill_about512':list(range(524))};old=dict(native_env={},native_argv=['/build/strata','--serve'],requests=[dict(ordinal=i+1,role=k,ids=ids[k],fresh=1,pin=None,max_new=64)for i,k in enumerate(('short','prefill_about512','prefill_about512','short'))]);tool=SimpleNamespace(inherited_env=lambda e:None,CLOCK_ENV={'UNITRACE_CLOCK_QUERY_SAMPLE_512':'1'},NATIVE_ENV={},OPTIONS=[])
        with patch.object(a,'tool',tool),patch.object(a,'_RUNTIME',{'image':'same'}),patch.object(a.base,'phase_recipe',side_effect=lambda *args:({},copy.deepcopy(old))):
            a.set_case('short',99,123);_,recipe=a.phase_recipe(a.PHASES[1],{'runtime':{'image':'same'}},{'env':{}},ids,'a'*64)
        self.assertEqual([len(r['ids'])for r in recipe['requests']],[54,54]);self.assertEqual([r['max_new']for r in recipe['requests']],[4,4]);self.assertEqual(recipe['native_env']['UNITRACE_StartPaused'],'1');self.assertIn('--start-paused',recipe['native_argv'])
        with self.assertRaises(ValueError):a.set_case('prefill_about512',99,123)
    def test_real_two_request_contract(self):
        expected=[dict(ordinal=i,ids=[1],fresh=1,pin=None,max_new=4)for i in(1,2)];raw=dict(ids=[1],fresh=1,pin=None,cancel_requested=None,stop_sent=False,command='GEN 4 fresh=1 ckpt=1 temperature=0 1',stdout=['T 2','DONE 1 1 0.1 0.2 stop'],output_ids=[2],stdout_seconds=[.1,.2],LP=[]);rows=[dict(ordinal=i,raw=copy.deepcopy(raw))for i in(1,2)];self.assertEqual(len(w.requests(rows,expected)),2)
        rows[1]['raw']['output_ids']=[3]
        with self.assertRaises(ValueError):w.requests(rows,expected)
    def test_resume_before_selected_is_source_hook(self):
        s=Path(a.__file__).read_text();self.assertIn("row['ordinal']==2:warm_control",s);self.assertIn('warmup-DONE-observed.json',s);self.assertIn('selected-DONE-observed.json',s)
if __name__=='__main__':unittest.main()
