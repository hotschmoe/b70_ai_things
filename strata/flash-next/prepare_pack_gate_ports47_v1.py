"""Source-only static gate port producer. No pack/model/runtime actions.
Ports explicit pack_epoch signatures/calls, preserving historical producer
provenance as original filenames. Generated static modules are separately pinned.
"""
import ast,copy,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
MAP={'c137_baseline_admission_v4':'c137_baseline_pack47_v1','adjudicate_c137_missing_started_v3_v1':'adjudicate_c137_pack47_v1','batch_numerical_proofs_v40':'batch_proofs_pack47_v1','batch_numerical_execution_v40':'batch40_manifest_pack47_v1','audit_batch_numerical_suite_v40':'audit_batch_pack47_v1','adjudicate_serial37_cacheoff_first_v1':'first49_pack47_v1','serial37_selection_v3':'serial_selection_pack47_v1','batch_serial_source37_v3':'serial_manifest_pack47_v1','validate_batch_serial_source37_v3':'serial_reader_pack47_v1','validate_private_native_offon_source37_v4':'private_reader_pack47_v1','paired_source37_private_v4_control_v1':'paired_control_pack47_v1'}
CALLS={'validate_prepared','manifest_binding','genuine_baseline','topology_baselines','admit_adjudication','finalized_binding','parent_arm','origin_arm','final_source_join','selected_jobs','serial_binding','paired_two_control'}
class Port(ast.NodeTransformer):
 def __init__(self,original):self.original=original;self.functions=[]
 def visit_Import(self,node):
  for row in node.names:
   if row.name in MAP:
    old=row.name;row.name=MAP[old];row.asname=row.asname or old
  return node
 def visit_ImportFrom(self,node):
  if node.module in MAP:node.module=MAP[node.module]
  return node
 def visit_Name(self,node):
  if node.id=='__file__':return ast.copy_location(ast.Constant(str(HERE/(self.original+'.py'))),node)
  return node
 def visit_FunctionDef(self,node):
  needs=node.name in CALLS or any(isinstance(n,ast.Call) and ((isinstance(n.func,ast.Name) and n.func.id in CALLS) or (isinstance(n.func,ast.Attribute) and n.func.attr in CALLS)) for n in ast.walk(node))
  self.functions.append(needs);node=self.generic_visit(node);self.functions.pop()
  if needs:node.args.kwonlyargs.append(ast.arg(arg='pack_epoch'));node.args.kw_defaults.append(ast.Constant(None))
  return node
 def visit_Call(self,node):
  node=self.generic_visit(node)
  if isinstance(node.func,ast.Name) and node.func.id=='__import__' and node.args and isinstance(node.args[0],ast.Constant) and node.args[0].value in MAP:node.args[0].value=MAP[node.args[0].value]
  name=node.func.id if isinstance(node.func,ast.Name) else node.func.attr if isinstance(node.func,ast.Attribute) else None
  if name in CALLS and self.functions and self.functions[-1]:node.keywords.append(ast.keyword(arg='pack_epoch',value=ast.Name(id='pack_epoch',ctx=ast.Load())))
  if self.original=='adjudicate_c137_missing_started_v3_v1' and isinstance(node.func,ast.Attribute) and node.func.attr=='validate_prepared':node.func=ast.Name(id='validate_prepared',ctx=ast.Load())
  return node

def produce(write=True):
 results={}
 for original,new in MAP.items():
  path=HERE/(original+'.py');tree=ast.parse(path.read_text())
  # Historical CLI/model execution is not copied into admission-only ports.
  if original in ('batch_numerical_execution_v40','batch_serial_source37_v3'):
   functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='manifest_binding'];tree=ast.Module(body=[ast.ImportFrom(module=original,names=[ast.alias(name='*')],level=0)]+functions,type_ignores=[])
  else:tree.body=[n for n in tree.body if not isinstance(n,ast.If) and not (isinstance(n,ast.FunctionDef) and n.name in ('main','prepare','run','prepare_serial_roster') or original=='audit_batch_numerical_suite_v40' and isinstance(n,ast.FunctionDef) and n.name in ('paired_two_control','audit'))]
  port=Port(original);tree=port.visit(tree)
  if original in ('batch_numerical_execution_v40','batch_serial_source37_v3'):
   tree.body[0]=ast.ImportFrom(module=original,names=[ast.alias(name='*')],level=0)
   additions='from batch_proofs_pack47_v1 import genuine_baseline,topology_baselines' if original=='batch_numerical_execution_v40' else 'import batch40_manifest_pack47_v1 as origin\nfrom serial_selection_pack47_v1 import selected_jobs'
   tree.body[1:1]=ast.parse(additions).body
   if original=='batch_numerical_execution_v40':functions[0].body=[n for n in functions[0].body if not isinstance(n,ast.If) or ast.unparse(n.test)!="plan['slots'] > 2"]
   functions[0].body.insert(0,ast.parse("require(plan['slots']==2,'Only actual historical paired2 corpus is a prerequisite; no other source scope transfer')").body[0])
  if original=='adjudicate_c137_missing_started_v3_v1':tree.body.insert(0,ast.ImportFrom(module='c137_prepared_pack47_v1',names=[ast.alias(name='validate_prepared')],level=0))
  if original=='c137_baseline_admission_v4':
   # Explicit prepared-gate API; c remains original runtime producer identity.
   class Prepared(ast.NodeTransformer):
    def visit_Call(self,node):
     self.generic_visit(node)
     if isinstance(node.func,ast.Attribute) and node.func.attr=='validate_prepared':node.func=ast.Name(id='validate_prepared',ctx=ast.Load())
     return node
   tree=Prepared().visit(tree);tree.body.insert(0,ast.ImportFrom(module='c137_prepared_pack47_v1',names=[ast.alias(name='validate_prepared')],level=0))
  if original=='paired_source37_private_v4_control_v1':
   # Expected original reader source remains actual historical proof provenance.
   class ReaderFile(ast.NodeTransformer):
    def visit_Attribute(self,node):
     if isinstance(node.value,ast.Name) and node.value.id=='reader' and node.attr=='__file__':return ast.Constant(str(HERE/'validate_private_native_offon_source37_v4.py'))
     return self.generic_visit(node)
   tree=ReaderFile().visit(tree)
  ast.fix_missing_locations(tree);text='# NEW explicit pack epoch admission port of '+original+'.py\n'+ast.unparse(tree)+'\n';target=HERE/(new+'.py');
  if write:target.write_text(text)
  results[new]={'original':str(path),'original_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'generated_sha256':hashlib.sha256(text.encode()).hexdigest()}
 return results
if __name__=='__main__':print(json.dumps(produce(),indent=2))
