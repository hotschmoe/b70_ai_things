"""Synthetic own arithmetic/protocol/lifecycle controls. No model/GPU/compiler."""
import ast,copy,hashlib,inspect,os,struct,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import owned_hc_device_rs_protocol_v1 as protocol
import full48_owned_hc_device_rs_v1 as reference
import owned_hc_device_rs_experiment_v1 as experiment
import owned_hc_device_rs_bridge_v1 as bridge
import qualify_owned_hc_device_rs_v1 as q
def qualification():return {'helper_source_sha256':hashlib.sha256(Path(protocol.__file__).with_name('owned_hc_rsqrt37_service_v1.cpp').read_bytes()).hexdigest(),'operation':'sycl::rsqrt','captured_operands_used':False,'ULP_adjustment_or_lookup_used':False,'actual_owned_device_started':False,'synthetic_only':True,'actual_source_flags_and_recipe_qualified':False}
def reply(command,rs=(1.,1.,1.,1.)):
 tokens=command.split();return 'OWNRS37_RESPONSE seq='+tokens[1]+''.join(' arg='+v for v in tokens[2:])+''.join(' rs='+v for v in protocol.words(struct.pack('<4f',*rs)))+' direct_replays_bitwise=1\n'
class Tests(unittest.TestCase):
 def test_owned_argument_and_actual_result_echo(self):
  a=np.asarray([1.,2.,3.,4.],dtype='<f4');line=reply(protocol.request(1,a),(2.,3.,4.,5.));self.assertEqual(protocol.response(line,1,a).tolist(),[2.,3.,4.,5.])
 def test_stale_substituted_nonfinite_missing_newline_negative(self):
  a=np.ones(4,dtype='<f4');good=reply(protocol.request(1,a))
  for line in (good.replace('seq=1','seq=2'),good.replace('arg=3f800000','arg=40000000',1),good.replace('rs=3f800000','rs=7fc00000',1),good[:-1],good+'OWNRS37_ERROR hidden\n'):
   with self.assertRaises(ValueError):protocol.response(line,1,a)
 def test_shape_budget_role_and_actual_source_controls(self):
  for a in ([0.,1.,1.,1.],[float('inf')]*4,[1.]*3):
   with self.assertRaises(ValueError):protocol.request(1,a)
  for n in (0,513,True):
   with self.assertRaises(ValueError):protocol.request(n,np.ones(4))
  c=protocol.OwnDeviceRsClient(reply,qualification())
  with self.assertRaises(ValueError):c.require_actual()
  with self.assertRaises(ValueError):c.rsqrt_owned(np.ones(4),'blk.48.hc_attn_')
 def test_only_HC_read_RS_line_changes_math_body(self):
  old=inspect.getsource(reference.prior.OwnedHcFmaRefinement.hc_read);expected=old.replace("rs=np.asarray([hc.rsqrt_candidate(hc.rms_argument(row,self.epsilon_contract['epsilon'])['rsqrt_argument_f32']) for row in r],dtype='<f4');","rs=self.resolve_owned_rs(r,stem);");self.assertEqual(inspect.getsource(reference.OwnedHcDeviceRsReference.hc_read),expected)
  methods={n for n,v in reference.OwnedHcDeviceRsReference.__dict__.items() if callable(v)};self.assertEqual(methods,{'__init__','resolve_owned_rs','hc_read','tokens'})
 def test_own_residual_generates_arguments_and_device_RS_normalizes(self):
  captured=[]
  def exchange(command):captured.append(command);return reply(command,(2.,3.,4.,5.))
  model=object.__new__(reference.OwnedHcDeviceRsReference);model.device_rs=protocol.OwnDeviceRsClient(exchange,qualification());model.owned_rms_records=[];model.epsilon_contract={'epsilon':reference.hc.EPSILON};model.hc_calls=0;model.projector=types.SimpleNamespace(vector=lambda name:np.ones(10240,dtype='<f4'));model.projection=lambda name,x:np.zeros(320 if 'down.' in name else 4 if 'inject.' in name else 10240,dtype='<f4');model.invoke=lambda op,k,m,mode,a,b,c:np.asarray(a*b+c,dtype='<f4');r=np.arange(10240,dtype='<f4').reshape(4,2560)/np.float32(100);out=model.hc_read(r,'blk.0.hc_attn_');expected=np.asarray(r*np.asarray([2.,3.,4.,5.],dtype='<f4')[:,None],dtype='<f4');self.assertEqual(np.asarray(out['normalized'],dtype='<f4').tobytes(),expected.tobytes());arguments=np.asarray([reference.hc.rms_argument(v,reference.hc.EPSILON)['rsqrt_argument_f32'] for v in r],dtype='<f4');self.assertEqual(captured,[protocol.request(1,arguments)]);self.assertFalse(model.owned_rms_records[0]['captured_operand_used'])
 def test_prefix8_rejected_before_parent_model_work(self):
  model=object.__new__(reference.OwnedHcDeviceRsReference)
  with patch.object(reference.prior.OwnedHcFmaRefinement,'tokens',side_effect=AssertionError('model touched')):
   with self.assertRaises(ValueError):model.tokens([1]*8)
 def fake(self,root):
  p=root/'fake.py';p.write_text("import os,struct,sys\nprint('OWNRS37_READY pid=%d backend=level_zero affinity=0 width=4 maximum=512'%os.getpid(),flush=True)\nn=0\nfor line in sys.stdin:\n if line=='QUIT\\n':break\n t=line.split();n+=1;print('OWNRS37_RESPONSE seq='+t[1]+''.join(' arg='+x for x in t[2:])+(' rs=3f800000'*4)+' direct_replays_bitwise=1',flush=True)\nprint('OWNRS37_DONE requests=%d graph_retired=1 owned_allocations_freed=1'%n,flush=True)\n",encoding='ascii');return p
 def test_real_CPU_pipe_lifetime_and_no_real_device_proof(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);active=[];cleaned=[]
   def work(client):client.rsqrt_owned(np.ones(4),'blk.0.hc_attn_');return {'synthetic_only':True}
   row,phase=bridge.run_phase([sys.executable,str(self.fake(root))],root/'out.log',work,lambda:cleaned.append(True),lambda ready:qualification(),active,timeout=5);self.assertTrue(row['passed']);self.assertTrue(row['eof']);self.assertTrue(row['reader_retired']);self.assertEqual(len(phase['records']),1);self.assertFalse(phase['qualification']['actual_owned_device_started']);self.assertEqual(cleaned,[])
 def test_CPU_work_error_requires_cleanup_and_cannot_pass(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);cleaned=[]
   def work(client):raise ValueError('MODEL_SYNTHETIC_ERROR')
   row,phase=bridge.run_phase([sys.executable,str(self.fake(root))],root/'out.log',work,lambda:cleaned.append(True),lambda ready:qualification(),[],timeout=5);self.assertFalse(row['passed']);self.assertTrue(row['reader_retired']);self.assertIn('MODEL_SYNTHETIC_ERROR',row['command_error']);self.assertEqual(cleaned,[True])
 def test_actual_response_roster_own_CPU_argument_join(self):
  a=np.ones(4,dtype='<f4');client=protocol.OwnDeviceRsClient(reply,qualification());client.rsqrt_owned(a,'blk.0.hc_attn_');record={'role':'blk.0.hc_attn_','device_sequence':1,'source_FMA_XOR_arguments':[{'rsqrt_argument_f32':1.}]*4,'argument_LE_F32_hex':a.tobytes().hex(),'captured_operand_used':False};phase={'ready':'OWNRS37_READY pid=1 backend=level_zero affinity=0 width=4 maximum=512\n','done':'OWNRS37_DONE requests=1 graph_retired=1 owned_allocations_freed=1\n','records':client.records};work={'device_RS_records':client.records,'first_RMS_arguments':[record],'prefix4_attempted':False,'first_gate':{'passed':True}}
  with tempfile.TemporaryDirectory() as t:
   path=Path(t)/'log';path.write_text(phase['ready']+client.records[0]['response']+phase['done']);self.assertEqual(experiment.recollect(path,phase,work)['request_count'],1)
   for kind in ('args','seq','role','duplicate'):
    bad=copy.deepcopy(work)
    if kind=='args':bad['first_RMS_arguments'][0]['source_FMA_XOR_arguments'][0]['rsqrt_argument_f32']=2.
    elif kind=='seq':bad['first_RMS_arguments'][0]['device_sequence']=2
    elif kind=='role':bad['device_RS_records'][0]['role']='blk.1.hc_attn_'
    else:path.write_text(phase['ready']+client.records[0]['response']*2+phase['done'])
    with self.assertRaises(ValueError):experiment.recollect(path,phase,bad)
 def test_inside_dispatch_and_actual_recipe_service_selection(self):
  s=Path(q.__file__).read_text();ast.parse(s);self.assertEqual(s.count("'/leaf/owned_hc_rsqrt37_service_v1.cpp'"),2);self.assertEqual(s.count('--maximum-requests 512 --maps-prefix /out/own-reference'),2);self.assertEqual(s.count("insert(2,'-i')"),2);self.assertLess(s.index('if args.inside_runtime:'),s.index('header(args.prior_rms_root)'));self.assertNotIn('compare_all_variants',s)
 def test_first_exact_gate_before_prefix_work_source(self):
  import owned_hc_device_rs_work_v1 as work
  s=Path(work.__file__).read_text();self.assertIn("if self.prefix_root and self.report['first_gate']['passed']:",s);self.assertLess(s.index("self.report['first_gate']="),s.index('prefix_model.tokens(ids)'));self.assertNotIn('conditional_gdn(',s);self.assertIn('386',s);self.assertIn('385',s)
if __name__=='__main__':unittest.main()
