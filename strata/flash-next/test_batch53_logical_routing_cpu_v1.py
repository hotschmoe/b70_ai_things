"""Explicit logical path routing, foreign keyword and lifecycle source controls."""
import ast,importlib,inspect,unittest
from pathlib import Path
from unittest.mock import patch
import prepare_upload_logical_gate_ports_v2 as ports
HERE=Path(__file__).resolve().parent
class Controls(unittest.TestCase):
 def test_all_resolved_metadata_calls_have_logical_keywords_no_foreign_kwargs(self):
  names=list(ports.MAP.values())+['batch_numerical_execution_v53','batch_numerical_proofs_v53','audit_batch_numerical_suite_v53','run_batch_serial_controls_v53']
  for name in names:
   m=importlib.import_module(name);tree=ast.parse(Path(m.__file__).read_text());aliases={}
   for n in ast.walk(tree):
    if isinstance(n,ast.Import):
     for a in n.names:aliases[a.asname or a.name]=a.name
    if isinstance(n,ast.ImportFrom):
     for a in n.names:
      if a.name!='*':aliases[a.asname or a.name]=(n.module,a.name)
   for n in ast.walk(tree):
    if not isinstance(n,ast.Call):continue
    fn=n.func;obj=None
    if isinstance(fn,ast.Name):
     obj=getattr(m,fn.id,None)
     if isinstance(aliases.get(fn.id),tuple):module,attr=aliases[fn.id];obj=getattr(importlib.import_module(module),attr)
    elif isinstance(fn,ast.Attribute)and isinstance(fn.value,ast.Name):
     if isinstance(aliases.get(fn.value.id),str):obj=getattr(importlib.import_module(aliases[fn.value.id]),fn.attr,None)
     elif hasattr(m,fn.value.id):obj=getattr(getattr(m,fn.value.id),fn.attr,None)
    elif isinstance(fn,ast.Attribute)and isinstance(fn.value,ast.Call)and isinstance(fn.value.func,ast.Name)and fn.value.func.id=='__import__':obj=getattr(importlib.import_module(fn.value.args[0].value),fn.attr)
    provided={k.arg for k in n.keywords if k.arg in ports.KEYWORDS}
    try:sig=inspect.signature(obj)
    except (TypeError,ValueError):self.assertFalse(provided,name+' unresolved '+ast.unparse(fn));continue
    wanted=set(ports.KEYWORDS)&set(sig.parameters)
    # prepare/run CLI entry constructs its own fresh scope if absent.
    if wanted and all(sig.parameters[k].default is None for k in wanted)and not provided:continue
    self.assertEqual(provided,wanted,name+' '+ast.unparse(fn))
 def test_parent_child_own_entry_ready_predevice_post_logical_bytes(self):
  parent=(HERE/'qualify_batch_numerical_v53.py').read_text();child=(HERE/'run_batch_serial_controls_v53.py').read_text();ctrl=(HERE/'batch_numerical_execution_v53.py').read_text()
  self.assertIn('logical_for_plan(preflight_plan)',parent);self.assertIn("parent['logical_operation_witness']=logical_epoch.finalize",parent);self.assertLess(child.index('logical_ready('),child.index('ready_row = ready'));self.assertLess(child.index('ack_row = wait_ack'),child.index('logical_epoch.seal_predevice('));self.assertIn("'logical-operation-witness.json'",ctrl);self.assertIn("'prepare-logical-operation-witness.json'",ctrl)
 def test_saved_logical_reader_binds_parent3_child4_and_ACK_before_reseal(self):
  s=(HERE/'audit_batch_numerical_suite_v53.py').read_text();self.assertIn("logical_saved_binding(parent['logical_operation_witness']",s);self.assertIn("ready_boundary=ready['logical_ready_boundary']",s);self.assertIn('ack_epoch',s)
  from logical_witness_binding_v1 import binding
  self.assertRaises(ValueError,binding,{},None,123,1,2)
if __name__=='__main__':unittest.main()
