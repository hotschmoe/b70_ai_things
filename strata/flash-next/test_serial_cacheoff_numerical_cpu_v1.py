#!/usr/bin/env python3
"""Actual preserved-output recollection +tiny mockproducer controls, no real model."""
import copy,json,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import extract_serial_cacheoff_numerical_v1 as q
import run_batch_serial_cacheoff_v8 as adapter
ROOT=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f16-source35-20261009/batch-v7-onecard-native2-serial0-run/child/serial-0')

class CacheOffTests(unittest.TestCase):
 def setUp(self):
  self.raw=json.loads((ROOT/'serial-2001-admission.json').read_text());plan=json.loads((ROOT.parent/'plan.snapshot.json').read_text());self.args,self.env=adapter.config(plan)
 def parse(self,raw=None):return q.extract(self.raw if raw is None else raw,ROOT/'captures',self.args,self.env,[(0,48)],allow_legacy_pin_zero=True)
 def test_actual_first49_readonly_correct_cacheOFF_tuple_andwork(self):
  m=self.parse();self.assertEqual(len(m['vectors']),49);self.assertEqual(m['ledger']['evaluated_prompt_rows'],59);self.assertEqual(m['ledger']['generated'],1);self.assertFalse(m['numerical_cacheOFF_lifecycle']['cache_qualification_granted']);self.assertFalse(m['numerical_cacheOFF_lifecycle']['commit']['published']);self.assertTrue(m['numerical_cacheOFF_lifecycle']['commit']['chain_updated']);self.assertFalse(m['numerical_cacheOFF_lifecycle']['commit']['live_reusable'])
 def test_legacyzero_not_newlive_admission(self):
  with self.assertRaises(ValueError):q.extract(self.raw,ROOT/'captures',self.args,self.env,[(0,48)])
  raw=copy.deepcopy(self.raw);raw['pin']=None;raw['command']=adapter.protocol.request_command(raw['ids'],1,1,None);self.assertFalse(self.parse(raw)['numerical_cacheOFF_lifecycle']['historical_legacy_pin_zero_replay'])
 def test_reusable_or_unupdated_or_published_flags_are_not_falsified(self):
  for key,value in [('live_reusable',True),('chain_updated',False),('published',True)]:
   raw=copy.deepcopy(self.raw)
   for i,line in enumerate(raw['stderr']):
    if line.startswith('PCL ') and json.loads(line[4:])['event']=='committed_live':event=json.loads(line[4:]);event[key]=value;raw['stderr'][i]='PCL '+json.dumps(event)
   with self.assertRaises(ValueError):self.parse(raw)
 def test_cacheON_eager0_reread_source_flags_rejected(self):
  for key,value in [('--prompt-cache','1'),('--batch','2'),('--conversation-cache-mib','512')]:
   args=list(self.args);args[args.index(key)+1]=value
   with self.assertRaises(ValueError):q.profile(args,self.env)
  for key,value in [('STRATA_VERIFY_EAGER','0'),('STRATA_STATE_HASH','0'),('STRATA_BATCH_FULL_STATE_CHAIN','1'),('STRATA_FIDELITY_DIAG_ACTIVATIONS','0')]:
   with self.assertRaises(ValueError):q.profile(self.args,dict(self.env,**{key:value}))
 def test_nread_nout_fresh_and_ids_mismatch_rejected(self):
  for change in ({'fresh':0},{'output_ids':[16,17]},{'done':self.raw['done'].replace('DONE 1 59','DONE 1 58')},{'ids':self.raw['ids'][:-1]}):
   raw=copy.deepcopy(self.raw);raw.update(change)
   with self.assertRaises(ValueError):self.parse(raw)
 def test_missing_stage_return_missing_head_and_cacheevent_rejected(self):
  for mode in ('return','head','cache'):
   raw=copy.deepcopy(self.raw)
   if mode=='head':raw['stderr']=[line for line in raw['stderr'] if 'phase=first_logits_before_sampler' not in line]
   elif mode=='return':
    i=next(i for i,line in enumerate(raw['stderr']) if line.startswith('PCL ') and json.loads(line[4:]).get('event')=='stage_span' and json.loads(line[4:]).get('complete'));raw['stderr'].pop(i)
   else:
    begin=next(json.loads(line[4:]) for line in raw['stderr'] if line.startswith('PCL ') and json.loads(line[4:])['event']=='begin');raw['stderr'].append('PCL '+json.dumps({'event':'cache','pid':begin['pid'],'request':begin['request']}))
   with self.assertRaises(ValueError):self.parse(raw)
 def test_real_tiny_singleFD_fakeproducer_keeps_diagnostics_and_absentpin(self):
  # Only a tiny Python text producer, no device/Docker/model payload.
  with tempfile.TemporaryDirectory(prefix='cacheOFF-pipe-CPU-') as name:
   root=Path(name);(root/'ARM').write_text('ARM');script=root/'producer.py';script.write_text('''import sys
print('READY 2048 stop',flush=True)
for line in sys.stdin:
 if line.startswith('QUIT'):break
 print('SFD request pid=1 request=1 tokens=2 ids=1,2 shape=4x2560',file=sys.stderr,flush=True)
 print('PCL {"event":"begin","pid":1,"request":1}',file=sys.stderr,flush=True)
 print('PREFIX_DIAG {"event":"finish"}',file=sys.stderr,flush=True)
 print('T 16',flush=True)
 print('DONE 1 2 0 0 length 0 0 0 0 0 0 0 0 2 0',flush=True)
''')
   producer=adapter.CacheOffMergedProtocol([sys.executable,str(script)],root)
   try:raw=producer.request('tiny',[1,2],fresh=1,max_new=1);self.assertTrue(any(line.startswith('PCL ') for line in raw['stderr']));self.assertTrue(any(line.startswith('SFD ') for line in raw['stderr']));self.assertNotIn('pin=',raw['command']);self.assertEqual(raw['output_ids'],[16]);self.assertIsNone(raw['pin'])
   finally:
    self.assertEqual(producer.close(),0);producer.p.stdin.close();producer.p.stdout.close()
   self.assertTrue((root/'engine.combined.log').is_file());self.assertTrue((root/'engine.protocol.log').is_file())
 def test_pin_metadata_command_mismatch_false_or_foreignPCL_identity_refused(self):
  for value in (None,False):
   raw=copy.deepcopy(self.raw);raw['pin']=value
   with self.assertRaises(ValueError):self.parse(raw)
  raw=copy.deepcopy(self.raw);raw['command']=adapter.protocol.request_command(raw['ids'],1,1,None)
  with self.assertRaises(ValueError):self.parse(raw)
  raw=copy.deepcopy(self.raw)
  for i,line in enumerate(raw['stderr']):
   if line.startswith('PCL '):event=json.loads(line[4:]);event['request']=2;raw['stderr'][i]='PCL '+json.dumps(event)
  with self.assertRaisesRegex(ValueError,'PCL and SFD'):self.parse(raw)
 def test_processlocal_runtime_adapter_restores_shared_V7_helpers(self):
  oldnum=adapter.shared.numerical;oldproto=adapter.shared.MergedProtocol
  with patch.object(adapter.shared,'run',return_value={'CPU_MOCK':True}) as run:
   plan=json.loads((ROOT.parent/'plan.snapshot.json').read_text());result=adapter.run(plan,ROOT,ROOT,ROOT,0);self.assertEqual(result,{'CPU_MOCK':True});run.assert_called_once()
  self.assertIs(adapter.shared.numerical,oldnum);self.assertIs(adapter.shared.MergedProtocol,oldproto)
if __name__=='__main__':unittest.main()
