"""Static semantic equivalence and tiny shared-scope controls; no SDK reads."""
import ast,copy,hashlib,importlib,inspect,tempfile,unittest,json
from pathlib import Path
from unittest.mock import patch
import prepare_sdk37_gate_ports_v1 as producer
import sdk37_witness_scope_v1 as scope
import operation_sdk37_byte_witness_v1 as w
HERE=Path(__file__).resolve().parent
class Undo(ast.NodeTransformer):
 def visit_FunctionDef(self,n):
  if n.args.kwonlyargs[-1].arg=='sdk_epoch':n.args.kwonlyargs.pop();n.args.kw_defaults.pop()
  n.body=[x for x in n.body if not (isinstance(x,ast.Expr) and isinstance(x.value,ast.Call) and isinstance(x.value.func,ast.Name) and x.value.func.id in ('require_engine','require_prepared','require_oracle'))];return self.generic_visit(n)
 def visit_Call(self,n):
  self.generic_visit(n);n.keywords=[k for k in n.keywords if k.arg!='sdk_epoch']
  if isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='sdk_epoch' and n.func.attr=='digest_for':return ast.Call(func=ast.Name(id='sha',ctx=ast.Load()),args=[n.args[1]],keywords=[])
  return n
class Controls(unittest.TestCase):
 def test_generated_static_bytes_reproduce_without_writing(self):
  for name,digest in producer.produce(write=False).items():self.assertEqual(hashlib.sha256((HERE/name).read_bytes()).hexdigest(),digest)
 def test_combined_upload_and_prepared_gate_logic_unchanged(self):
  originals={f.name:f for f in ast.parse((HERE/'c1_serve_controller_combined_v137.py').read_text()).body if isinstance(f,ast.FunctionDef)}
  for filename in ('c137_sdk37_digest_ports_v1.py','prepared_sdk37_digest_port_v1.py'):
   for fn in ast.parse((HERE/filename).read_text()).body:
    if not isinstance(fn,ast.FunctionDef):continue
    old=originals[fn.name] if fn.name!='validate_prepared' else next(f for f in ast.parse((HERE/'c137_prepared_pack48_v1.py').read_text()).body if isinstance(f,ast.FunctionDef));new=Undo().visit(copy.deepcopy(fn));self.assertEqual(ast.dump(new,include_attributes=False),ast.dump(old,include_attributes=False),fn.name)
 def test_current_and_historical_engine_gates_preserve_all_gates_and_magic(self):
  for source,target in [('batch_numerical_proofs_v48.py','current_engine_sdk37_digest_port_v1.py'),('batch_proofs_pack48_v1.py','historical_engine_sdk37_digest_port_v1.py')]:
   old=next(f for f in ast.parse((HERE/source).read_text()).body if isinstance(f,ast.FunctionDef) and f.name=='engine_binding');new=next(f for f in ast.parse((HERE/target).read_text()).body if isinstance(f,ast.FunctionDef));new=Undo().visit(new)
   original_with=next(n for n in ast.walk(old) if isinstance(n,ast.With));loop=next(n for n in new.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='target');loop.body=[copy.deepcopy(original_with) if 'sdk_epoch.ELF_magic' in ast.unparse(n) else n for n in loop.body]
   for call in ast.walk(new):
    if isinstance(call,ast.Call) and isinstance(call.func,ast.Name) and call.func.id=='combined_generation_gate':call.func=ast.Attribute(value=ast.Name(id='c1',ctx=ast.Load()),attr='combined_generation_gate',ctx=ast.Load())
   self.assertEqual(ast.dump(new,include_attributes=False),ast.dump(old,include_attributes=False),target)
 def test_every_routed_semantic_call_has_required_explicit_sdk_keyword(self):
  modules=['c137_sdk37_digest_ports_v1','current_engine_sdk37_digest_port_v1','historical_engine_sdk37_digest_port_v1','prepared_sdk37_digest_port_v1']
  for name in modules:
   m=importlib.import_module(name);tree=ast.parse(Path(m.__file__).read_text())
   for node in ast.walk(tree):
    if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Name):continue
    fn=getattr(m,node.func.id,None)
    if not callable(fn):continue
    try:sig=inspect.signature(fn)
    except (ValueError,TypeError):continue
    if 'sdk_epoch' in sig.parameters:self.assertTrue(any(k.arg=='sdk_epoch' for k in node.keywords),name+' '+node.func.id)
 def tiny(self,r):
  rows=[];receipts=[]
  for i,role in enumerate(w.ROLES):
   p=r/('elf'+str(i));p.write_bytes(b'\x7fELF tiny '+str(i).encode());rows.append((role,p,hashlib.sha256(p.read_bytes()).hexdigest()))
  for i in range(3):
   p=r/('meta'+str(i));p.write_text('{}');receipts.append((p,hashlib.sha256(p.read_bytes()).hexdigest()))
  return rows,receipts
 def test_multiple_declared_prepared_files_share_union_not_single_prepared_cache(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);rows,receipts=self.tiny(r);one,other=receipts[:2]
   with patch.object(scope,'metadata_roster',side_effect=lambda p:(rows,[one,receipts[2]]) if Path(p)==one[0] else (rows,[other,receipts[2]])):
    roster,union=scope.scope_metadata([one[0],other[0]]);self.assertEqual(len(union),3);epoch=w.SDK37Epoch(roster,union);self.assertTrue(scope.require_prepared(epoch,one[0]));self.assertTrue(scope.require_prepared(epoch,other[0]));self.assertRaises(ValueError,scope.require_prepared,epoch,r/'undeclared');epoch.seal_predevice();epoch.finalize()
 def test_changed_sdk_or_oracle_across_baselines_refused(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);rows,receipts=self.tiny(r);other=list(rows);other[0]=(other[0][0],other[0][1],'0'*64)
   with patch.object(scope,'metadata_roster',side_effect=[(rows,receipts),(other,receipts)]):self.assertRaises(ValueError,scope.scope_metadata,[receipts[0][0],receipts[1][0]])
 def test_h48_and_original_helpers_do_not_import_new_ports(self):
  for name in ('batch_numerical_execution_v48.py','c137_prepared_pack48_v1.py','c1_serve_controller_combined_v137.py'):
   text=(HERE/name).read_text();self.assertNotIn('sdk37_digest_port',text);self.assertNotIn('sdk37_witness_scope',text)
if __name__=='__main__':unittest.main()
