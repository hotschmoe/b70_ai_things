"""New integration mandatory keyword routing; original historical ports retained."""
import ast,importlib,inspect,unittest
from pathlib import Path
import prepare_pack_gate_ports49_v1 as producer
class Controls(unittest.TestCase):
 def test_every_reachable_sdk_callee_required_keyword_no_missing_or_foreign(self):
  names=list(producer.MAP.values())+['c137_prepared_pack49_v1','c137_sdk37_digest_ports_v1','batch_numerical_proofs_v54','batch_numerical_execution_v54','audit_batch_numerical_suite_v54','run_batch_serial_controls_v54','run_batch_serial_controls_v49','batch54_prelease_v1']
  for name in names:
   module=importlib.import_module(name);tree=ast.parse(Path(module.__file__).read_text());aliases={}
   for node in ast.walk(tree):
    if isinstance(node,ast.Import):
     for a in node.names:aliases[a.asname or a.name]=a.name
    if isinstance(node,ast.ImportFrom):
     for a in node.names:
      if a.name!='*':aliases[a.asname or a.name]=(node.module,a.name)
   for node in ast.walk(tree):
    if not isinstance(node,ast.Call):continue
    fn=node.func;obj=None
    if isinstance(fn,ast.Name):
     obj=getattr(module,fn.id,None)
     if fn.id in aliases and isinstance(aliases[fn.id],tuple):m,key=aliases[fn.id];obj=getattr(importlib.import_module(m),key)
    elif isinstance(fn,ast.Attribute) and isinstance(fn.value,ast.Name):
     if fn.value.id in aliases and isinstance(aliases[fn.value.id],str):obj=getattr(importlib.import_module(aliases[fn.value.id]),fn.attr,None)
     elif fn.value.id=='ctrl' and name=='batch54_prelease_v1':obj=getattr(importlib.import_module('batch_numerical_execution_v54'),fn.attr)
     elif hasattr(module,fn.value.id):obj=getattr(getattr(module,fn.value.id),fn.attr,None)
    elif isinstance(fn,ast.Attribute) and isinstance(fn.value,ast.Call) and isinstance(fn.value.func,ast.Name) and fn.value.func.id=='__import__':obj=getattr(importlib.import_module(fn.value.args[0].value),fn.attr)
    provided=any(k.arg=='sdk_epoch' for k in node.keywords)
    try:signature=inspect.signature(obj)
    except (TypeError,ValueError):
     self.assertFalse(provided,'Unresolved SDK callee '+name+' '+ast.unparse(fn));continue
    if 'sdk_epoch' in signature.parameters:self.assertTrue(provided,'Missing mandatory SDK propagation '+name+' '+ast.unparse(fn))
    else:self.assertFalse(provided,'Foreign SDK keyword '+name+' '+ast.unparse(fn))
 def test_native_only_cheap_prelease_rejects_other_profiles_before_source_read(self):
  import batch54_prelease_v1 as pre
  from types import SimpleNamespace
  def require(ok,message):
   if not ok:raise ValueError(message)
  ctrl=SimpleNamespace(require=require)
  for key,value in [('kind','native'),('kind','api'),('slots',2),('slots',4.0),('schema',4.0),('harness_generation',54.0)]:
   plan={'kind':'serial','slots':4,'schema':4,'harness_generation':54};plan[key]=value;self.assertRaises(ValueError,pre.cheap,plan,ctrl)
if __name__=='__main__':unittest.main()
