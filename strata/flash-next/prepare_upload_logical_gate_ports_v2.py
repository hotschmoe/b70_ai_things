"""Source-only admission port generator; no payload/runtime reads or execution."""
import ast,copy,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
MAP={
 'c137_sdk37_digest_ports_v1':'c137_sdk37_logical_ports_v2',
 'c137_prepared_pack49_v1':'c137_prepared_logical_ports_v2',
 'c137_baseline_pack49_v1':'c137_baseline_logical_ports_v2',
 'adjudicate_c137_pack49_v1':'adjudicate_c137_logical_ports_v2',
 'batch_proofs_pack49_v1':'batch_proofs_logical_ports_v2',
 'batch40_manifest_pack49_v1':'batch40_manifest_logical_ports_v2',
 'audit_batch_pack49_v1':'audit_batch_logical_ports_v2',
 'first49_pack49_v1':'first49_logical_ports_v2',
 'serial_selection_pack49_v1':'serial_selection_logical_ports_v2',
 'serial_manifest_pack49_v1':'serial_manifest_logical_ports_v2',
 'serial_reader_pack49_v1':'serial_reader_logical_ports_v2',
 'private_reader_pack49_v1':'private_reader_logical_ports_v2',
 'paired_control_pack49_v1':'paired_control_logical_ports_v2',
 'batch_numerical_proofs_v50':'batch_numerical_proofs_logical_ports_v2',
 'batch_numerical_execution_v50':'batch50_manifest_logical_ports_v2',
 'audit_batch_numerical_suite_v50':'audit_batch50_logical_ports_v2'}
KEYWORDS=('logical_epoch','logical_roster')
def functions(original):
 tree=ast.parse((HERE/(original+'.py')).read_text())
 return [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name not in ('main','prepare','run','prepare_serial_roster') and (original!='batch_numerical_execution_v50' or n.name=='manifest_binding')]
def calls():
 names={'original_upload_gate','upload_gate'}
 while True:
  previous=set(names)
  for original in MAP:
   tree=ast.parse((HERE/(original+'.py')).read_text())
   for node in ast.walk(tree):
    if isinstance(node,ast.ImportFrom)and node.module in MAP:
     for alias in node.names:
      if alias.name in names and alias.asname:names.add(alias.asname)
   for fn in functions(original):
    if any(isinstance(n,ast.Call) and (n.func.id if isinstance(n.func,ast.Name)else n.func.attr if isinstance(n.func,ast.Attribute)else None)in names for n in ast.walk(fn)):names.add(fn.name)
  if names==previous:return names
CALLS=calls()
def producer_name(name):return ast.Call(func=ast.Name(id='original_producer_file',ctx=ast.Load()),args=[ast.Constant(name)],keywords=[])
class Port(ast.NodeTransformer):
 def __init__(self,original):self.original=original;self.stack=[];self.aliases={}
 def visit_Import(self,node):
  for row in node.names:
   if row.name in MAP:
    old=row.name;self.aliases[row.asname or old]=old;row.name=MAP[old];row.asname=row.asname or old
  return node
 def visit_ImportFrom(self,node):
  if node.module in MAP:node.module=MAP[node.module]
  return node
 def visit_Name(self,node):
  if node.id=='__file__':return ast.copy_location(producer_name(self.original),node)
  return node
 def visit_Attribute(self,node):
  if node.attr=='__file__'and isinstance(node.value,ast.Name)and node.value.id in self.aliases:return ast.copy_location(ast.Call(func=ast.Name(id='original_module_file',ctx=ast.Load()),args=[ast.Constant(self.aliases[node.value.id]),ast.Constant(node.value.id)],keywords=[]),node)
  return self.generic_visit(node)
 def visit_FunctionDef(self,node):
  needs=node.name in CALLS;self.stack.append(needs);node=self.generic_visit(node);self.stack.pop()
  if needs:
   for name in KEYWORDS:node.args.kwonlyargs.append(ast.arg(arg=name));node.args.kw_defaults.append(None)
  if self.original=='c137_sdk37_digest_ports_v1'and node.name=='original_upload_gate':
   loop=next(n for n in node.body if isinstance(n,ast.For)and isinstance(n.target,ast.Name)and n.target.id=='row');start=next(i for i,n in enumerate(loop.body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='logical_path'for t in n.targets));loop.body=loop.body[:start]+ast.parse('logical_case(logical_epoch,Path(path).parent,row,current_roster=logical_roster)').body
   node.body=[n for n in node.body if not(isinstance(n,ast.ImportFrom)and n.module=='parse_usm_logical_free_trace')]
   node.body.insert(0,ast.parse("require(type(logical_epoch)is LogicalEpoch,'Explicit same-operation logical parser epoch required');logical_epoch.owned(logical_roster)").body[0]);node.body.insert(1,ast.parse('logical_epoch.owned(logical_roster)').body[0])
  return node
 def visit_Call(self,node):
  node=self.generic_visit(node)
  if isinstance(node.func,ast.Name)and node.func.id=='__import__'and node.args and isinstance(node.args[0],ast.Constant)and node.args[0].value in MAP:node.args[0].value=MAP[node.args[0].value]
  if isinstance(node.func,ast.Attribute)and node.func.attr=='import_module'and node.args and isinstance(node.args[0],ast.Constant)and node.args[0].value in MAP:node.args[0].value=MAP[node.args[0].value]
  name=node.func.id if isinstance(node.func,ast.Name)else node.func.attr if isinstance(node.func,ast.Attribute)else None
  if name in CALLS and self.stack and self.stack[-1]:
   for key in KEYWORDS:node.keywords.append(ast.keyword(arg=key,value=ast.Name(id=key,ctx=ast.Load())))
  return node

def original_tree(original):
 tree=ast.parse((HERE/(original+'.py')).read_text())
 if original=='batch_numerical_execution_v50':
  return ast.Module(body=[ast.ImportFrom(module=original,names=[ast.alias(name='*')],level=0),*functions(original)],type_ignores=[])
 tree.body=[n for n in tree.body if not isinstance(n,ast.If)and not(isinstance(n,ast.FunctionDef)and n.name in ('main','prepare','run','prepare_serial_roster'))];return tree

def produce(write=True):
 results={}
 for original,target in MAP.items():
  tree=Port(original).visit(original_tree(original))
  if original=='batch_numerical_execution_v50':
   tree.body[0]=ast.ImportFrom(module=original,names=[ast.alias(name='*')],level=0);tree.body[1:1]=ast.parse('from batch_numerical_proofs_logical_ports_v2 import genuine_baseline,topology_baselines').body
  tree.body[0:0]=ast.parse('from upload_logical_port_identity_v2 import original_producer_file,original_module_file').body
  if original=='c137_sdk37_digest_ports_v1':tree.body[1:1]=ast.parse('from logical_free_require_case_epoch_v2 import RequireLogicalEpoch as LogicalEpoch\nfrom original_upload_logical_requirement_v2 import logical_case').body
  ast.fix_missing_locations(tree);text='# NEW mandatory immutable logical-byte port of '+original+'.py\n'+ast.unparse(tree)+'\n';path=HERE/(target+'.py')
  if write:path.write_text(text)
  results[target]={'original':original+'.py','original_sha256':hashlib.sha256((HERE/(original+'.py')).read_bytes()).hexdigest(),'generated_sha256':hashlib.sha256(text.encode()).hexdigest()}
 return results
if __name__=='__main__':produce()
