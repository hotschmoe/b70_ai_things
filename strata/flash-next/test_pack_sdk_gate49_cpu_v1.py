"""Independent source equivalence/callee routing; tiny payloads only."""
import ast,copy,importlib,inspect,json,tempfile,unittest,hashlib
from pathlib import Path
from unittest.mock import patch
import prepare_pack_gate_ports49_v1 as producer
import operation_pack_hash_witness_v2 as witness
import c137_prepared_pack49_v1 as prepared
import pack_witness_binding47_v1 as binding
HERE=Path(__file__).resolve().parent
class Undo(ast.NodeTransformer):
 def __init__(self,original):self.original=original
 def visit_Constant(self,n):
  if n.value==str(HERE/(self.original+'.py')):return ast.Name(id='__file__',ctx=ast.Load())
  return n
 def visit_Import(self,n):
  reverse={v:k for k,v in producer.MAP.items()}
  for a in n.names:
   if a.name in reverse:
    a.name=reverse[a.name]
    if a.asname==a.name:a.asname=None
  return n
 def visit_ImportFrom(self,n):
  n.module={v:k for k,v in producer.MAP.items()}.get(n.module,n.module);return n
 def visit_FunctionDef(self,n):
  while n.args.kwonlyargs and n.args.kwonlyargs[-1].arg in ('pack_epoch','sdk_epoch'):n.args.kwonlyargs.pop();n.args.kw_defaults.pop()
  n.body=[x for x in n.body if 'Only actual historical paired2 corpus' not in ast.unparse(x) and not (isinstance(x,ast.Expr) and isinstance(x.value,ast.Call) and isinstance(x.value.func,ast.Name) and x.value.func.id in ('require_prepared','require_engine'))];return self.generic_visit(n)
 def visit_Call(self,n):
  self.generic_visit(n);n.keywords=[k for k in n.keywords if k.arg not in ('pack_epoch','sdk_epoch')]
  if isinstance(n.func,ast.Name) and n.func.id=='__import__' and n.args and isinstance(n.args[0],ast.Constant):n.args[0].value={v:k for k,v in producer.MAP.items()}.get(n.args[0].value,n.args[0].value)
  if isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='sdk_epoch' and n.func.attr=='digest_for':return ast.Call(func=ast.Name(id='sha',ctx=ast.Load()),args=[n.args[1]],keywords=[])
  return n
class Controls(unittest.TestCase):
 def test_static_port_bytes_reproduce_without_mutating_repo(self):
  for name,row in producer.produce(write=False).items():self.assertEqual(hashlib.sha256((HERE/(name+'.py')).read_bytes()).hexdigest(),row['generated_sha256'])
 def test_all_ported_functions_keep_original_gate_logic(self):
  for original,new in producer.MAP.items():
   old=ast.parse((HERE/(original+'.py')).read_text());current=ast.parse((HERE/(new+'.py')).read_text());functions={f.name:f for f in old.body if isinstance(f,ast.FunctionDef)}
   for f in current.body:
    if not isinstance(f,ast.FunctionDef):continue
    transformed=Undo(original).visit(copy.deepcopy(f))
    if original=='batch_numerical_proofs_v40' and f.name=='engine_binding':
     old=functions[f.name];original_with=next(n for n in ast.walk(old) if isinstance(n,ast.With));loop=next(n for n in transformed.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='target');loop.body=[copy.deepcopy(original_with) if 'sdk_epoch.ELF_magic' in ast.unparse(n) else n for n in loop.body]
     for n in ast.walk(transformed):
      if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='combined_generation_gate':n.func=ast.Attribute(value=ast.Name(id='c1',ctx=ast.Load()),attr='combined_generation_gate',ctx=ast.Load())
    if original=='c137_baseline_admission_v4':
     class PreparedCall(ast.NodeTransformer):
      def visit_Call(self,n):
       self.generic_visit(n)
       if isinstance(n.func,ast.Name) and n.func.id=='validate_prepared':n.func=ast.Attribute(value=ast.Name(id='c',ctx=ast.Load()),attr='validate_prepared',ctx=ast.Load())
       return n
     transformed=PreparedCall().visit(transformed)
    if original=='adjudicate_c137_missing_started_v3_v1':
     class PreparedCall(ast.NodeTransformer):
      def visit_Call(self,n):
       self.generic_visit(n)
       if isinstance(n.func,ast.Name) and n.func.id=='validate_prepared':n.func=ast.Attribute(value=ast.Name(id='c',ctx=ast.Load()),attr='validate_prepared',ctx=ast.Load())
       return n
     transformed=PreparedCall().visit(transformed)
    if original=='paired_source37_private_v4_control_v1':
     class ReaderFile(ast.NodeTransformer):
      def visit_Constant(self,n):
       if n.value==str(HERE/'validate_private_native_offon_source37_v4.py'):return ast.Attribute(value=ast.Name(id='reader',ctx=ast.Load()),attr='__file__',ctx=ast.Load())
       return n
     transformed=ReaderFile().visit(transformed)
    if original=='batch_numerical_execution_v40' and f.name=='manifest_binding':
     branch=next(n for n in functions[f.name].body if isinstance(n,ast.If) and ast.unparse(n.test)=="plan['slots'] > 2");index=next(i for i,n in enumerate(transformed.body) if 'Actual newcase source generation metadata differs' in ast.unparse(n));transformed.body.insert(index+1,copy.deepcopy(branch))
    with self.subTest(module=new,function=f.name):self.assertEqual(ast.dump(transformed,include_attributes=False),ast.dump(functions[f.name],include_attributes=False))
 def test_prepared_gate_changes_only_pack_loop_and_explicit_parameter(self):
  old=next(f for f in ast.parse((HERE/'c1_serve_controller_combined_v137.py').read_text()).body if isinstance(f,ast.FunctionDef) and f.name=='validate_prepared');new=next(f for f in ast.parse(Path(prepared.__file__).read_text()).body if isinstance(f,ast.FunctionDef));new=Undo('c1_serve_controller_combined_v137').visit(new)
  loop=next(n for n in old.body if isinstance(n,ast.For) and 'pack_receipt' in ast.unparse(n.iter));new.body=[copy.deepcopy(loop) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='consume_prepared_pack' else n for n in new.body];self.assertEqual(ast.dump(new,include_attributes=False),ast.dump(old,include_attributes=False))
 def test_every_pack_keyword_resolves_to_explicit_port_signature(self):
  names=list(producer.MAP.values())
  for name in names:
   module=importlib.import_module(name);tree=ast.parse(Path(module.__file__).read_text());aliases={}
   for n in ast.walk(tree):
    if isinstance(n,ast.Import):
     for a in n.names:aliases[a.asname or a.name]=a.name
    if isinstance(n,ast.ImportFrom):
     for a in n.names:
      if a.name!='*':aliases[a.asname or a.name]=(n.module,a.name)
   for n in ast.walk(tree):
    if not isinstance(n,ast.Call):continue
    call_name=n.func.id if isinstance(n.func,ast.Name) else n.func.attr if isinstance(n.func,ast.Attribute) else None
    if not any(k.arg=='pack_epoch' for k in n.keywords) and call_name not in producer.CALLS:continue
    self.assertTrue(any(k.arg=='pack_epoch' for k in n.keywords),'Missing epoch propagation '+name+' '+ast.unparse(n.func))
    f=n.func
    if isinstance(f,ast.Name):
     obj=getattr(module,f.id,None)
     if f.id in aliases and isinstance(aliases[f.id],tuple):m,key=aliases[f.id];obj=getattr(importlib.import_module(m),key)
    elif isinstance(f,ast.Attribute) and isinstance(f.value,ast.Name):
     obj=getattr(importlib.import_module(aliases[f.value.id]),f.attr) if f.value.id in aliases and isinstance(aliases[f.value.id],str) else getattr(importlib.import_module('batch_numerical_execution_v47') if f.value.id=='ctrl' and name=='batch47_prelease_v1' else getattr(module,f.value.id),f.attr)
    elif isinstance(f,ast.Attribute) and isinstance(f.value,ast.Call) and isinstance(f.value.func,ast.Name) and f.value.func.id=='__import__':obj=getattr(importlib.import_module(f.value.args[0].value),f.attr)
    else:self.fail('Unresolved explicit epoch callee '+name+' '+ast.unparse(f))
    with self.subTest(module=name,callee=ast.unparse(f)):self.assertIn('pack_epoch',inspect.signature(obj).parameters)
 def test_cleanup_never_receives_pack_keyword_and_dynamic_selection_is_ported(self):
  for name in ('run_batch_numerical_pilot_v47.py','run_batch_serial_controls_v47.py'):
   tree=ast.parse((HERE/name).read_text())
   calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='subprocess']
   self.assertTrue(calls);self.assertTrue(all(not any(k.arg=='pack_epoch' for k in n.keywords) for n in calls))
  self.assertIn("__import__('serial_selection_pack49_v1')",(HERE/'serial_reader_pack49_v1.py').read_text());self.assertNotIn('validate_private_native_offon_source37_v1',(HERE/'audit_batch_pack49_v1.py').read_text())
 def test_historical_ports_reject_other_scopes_before_baseline(self):
  for name in ('batch40_manifest_pack49_v1','serial_manifest_pack49_v1'):
   m=importlib.import_module(name)
   with self.assertRaises(ValueError):m.manifest_binding({'slots':4},pack_epoch=None,sdk_epoch=None)
 def tiny(self,root):
  p=root/'tiny';p.write_bytes(b'CPUbytes');return [(p,hashlib.sha256(p.read_bytes()).hexdigest())]
 def test_postseal_semantic_consumer_still_current_roster_and_fresh_post(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);roster=self.tiny(r);receipt=r/'receipt';receipt.write_text(json.dumps({'RESULT':{'files':{'tiny':{'sha256':roster[0][1]}}}}));p={'pack':str(r),'pack_receipt':str(receipt),'pack_receipt_sha256':hashlib.sha256(receipt.read_bytes()).hexdigest()};epoch=witness.PackEpoch(roster);witness.consume_prepared_pack(p,epoch);epoch.seal_predevice();witness.consume_prepared_pack(p,epoch);roster[0][0].write_bytes(b'changed');self.assertRaises(ValueError,epoch.finalize)
 def test_current_consumer_roster_changes_after_seal_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);roster=self.tiny(r);epoch=witness.PackEpoch(roster);epoch.seal_predevice();self.assertRaises(ValueError,epoch.require_roster,[(roster[0][0],'0'*64)])
 def test_saved_witness_not_admitted_without_current_byte_epoch(self):
  with tempfile.TemporaryDirectory() as d:
   roster=self.tiny(Path(d));epoch=witness.PackEpoch(roster);epoch.digest_for(*roster[0]);epoch.seal_predevice();proof=epoch.finalize();current=witness.PackEpoch(roster);self.assertEqual(binding.binding(proof,current,proof['owner_pid'],proof['boundaries'][1]['finished_epoch'],proof['boundaries'][2]['started_epoch'])['complete_byte_boundaries'],3)
   for key,value in [('owner_pid',True),('stat_only_validation',True),('semantic_digest_calls',0)]:
    bad=copy.deepcopy(proof);bad[key]=value;self.assertRaises(ValueError,binding.binding,bad,current,proof['owner_pid'],float('inf'),0)
   roster[0][0].write_bytes(b'changed');self.assertRaises(ValueError,witness.PackEpoch,roster)
if __name__=='__main__':unittest.main()
