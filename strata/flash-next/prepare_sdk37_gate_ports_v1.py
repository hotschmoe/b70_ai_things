"""Static source-only proposed digest ports; no runtime integration/payload work."""
import ast,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
C1='c1_serve_controller_combined_v137'

def extract(module,names):
 tree=ast.parse((HERE/(module+'.py')).read_text());return [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
def required(fn):fn.args.kwonlyargs.append(ast.arg(arg='sdk_epoch'));fn.args.kw_defaults.append(None);return fn
class Identity(ast.NodeTransformer):
 def visit_Name(self,n):return ast.Constant(str(HERE/(C1+'.py'))) if n.id=='__file__' else n

def produce(write=True):
 functions=extract(C1,{'combined_generation_gate','original_upload_gate','upload_gate'})
 class Rewrite(ast.NodeTransformer):
  def __init__(self,name):self.name=name
  def visit_Call(self,n):
   self.generic_visit(n)
   if isinstance(n.func,ast.Name) and n.func.id=='sha':
    arg=ast.unparse(n.args[0])
    if self.name=='combined_generation_gate' and arg=='path':return ast.parse("sdk_epoch.digest_for('SDK37:'+name,path,build['binary_sha256'][str(path)])",mode='eval').body
    if self.name=='original_upload_gate' and arg=="oracle_root / 'source-upload-oracle'":return ast.parse("sdk_epoch.digest_for('source37-upload-oracle',oracle_root/'source-upload-oracle',oracle['binary_sha256'])",mode='eval').body
   if isinstance(n.func,ast.Name) and n.func.id=='original_upload_gate':n.keywords.append(ast.keyword(arg='sdk_epoch',value=ast.Name(id='sdk_epoch',ctx=ast.Load())))
   return n
 for fn in functions:
  required(fn);fn=Identity().visit(fn);Rewrite(fn.name).visit(fn)
  if fn.name=='combined_generation_gate':fn.body.insert(0,ast.parse('require_engine(sdk_epoch,engine)').body[0])
  if fn.name in ('original_upload_gate','upload_gate'):fn.body.insert(0,ast.parse('require_oracle(sdk_epoch,engine_receipt,oracle_path)').body[0])
 module=ast.Module(body=functions,type_ignores=[]);ast.fix_missing_locations(module)
 text='"""Proposed explicit SDK37 digest ports; not wired into frozen H48."""\nfrom c1_serve_controller_combined_v137 import *\nfrom sdk37_witness_scope_v1 import require_engine,require_oracle\n'+ast.unparse(module)+'\n';result={'c137_sdk37_digest_ports_v1.py':text}
 # Both current and historical engine_binding implementations keep all gates.
 for source,target in [('batch_numerical_proofs_v48','current_engine_sdk37_digest_port_v1.py'),('batch_proofs_pack48_v1','historical_engine_sdk37_digest_port_v1.py')]:
  fn=extract(source,{'engine_binding'})[0];required(fn)
  class Engine(ast.NodeTransformer):
   def visit_Call(self,n):
    self.generic_visit(n)
    if isinstance(n.func,ast.Name) and n.func.id=='sha' and ast.unparse(n.args[0])=='binary':return ast.parse("sdk_epoch.digest_for('SDK37:'+target,binary,r['binary_sha256'][str(binary)])",mode='eval').body
    if isinstance(n.func,ast.Attribute) and n.func.attr=='combined_generation_gate':n.func=ast.Name(id='combined_generation_gate',ctx=ast.Load());n.keywords.append(ast.keyword(arg='sdk_epoch',value=ast.Name(id='sdk_epoch',ctx=ast.Load())))
    return n
   def visit_With(self,n):
    if "binary.open('rb')" in ast.unparse(n.items):return ast.parse("require(sdk_epoch.ELF_magic('SDK37:'+target,binary,r['binary_sha256'][str(binary)])==b'\\x7fELF','Mock file is not actual rebuilt ELF')").body[0]
    return self.generic_visit(n)
  fn=Engine().visit(fn);fn.body.insert(0,ast.parse('require_engine(sdk_epoch,engine)').body[0]);module=ast.Module(body=[fn],type_ignores=[]);ast.fix_missing_locations(module);result[target]='"""Proposed explicit SDK37 hash/magic port, all other gates retained."""\nfrom '+source+' import *\nfrom c137_sdk37_digest_ports_v1 import combined_generation_gate\nfrom sdk37_witness_scope_v1 import require_engine\n'+ast.unparse(module)+'\n'
 # Explicit prepared gate adds SDK epoch without changing existing pack epoch.
 fn=extract('c137_prepared_pack48_v1',{'validate_prepared'})[0];required(fn)
 class Prepared(ast.NodeTransformer):
  def visit_Call(self,n):
   self.generic_visit(n)
   if isinstance(n.func,ast.Name) and n.func.id=='sha' and ast.unparse(n.args[0])=="m['executable']":return ast.parse("sdk_epoch.digest_for('SDK37:strata',m['executable'],m['executable_sha256'])",mode='eval').body
   if isinstance(n.func,ast.Name) and n.func.id in ('combined_generation_gate','upload_gate'):n.keywords.append(ast.keyword(arg='sdk_epoch',value=ast.Name(id='sdk_epoch',ctx=ast.Load())))
   return n
 fn=Prepared().visit(fn);fn.body.insert(0,ast.parse("require_prepared(sdk_epoch,Path(directory)/'prepared.json')").body[0]);module=ast.Module(body=[fn],type_ignores=[]);ast.fix_missing_locations(module);result['prepared_sdk37_digest_port_v1.py']='"""Proposed explicit SDK+pack prepared admission port; unintegrated."""\nfrom c137_prepared_pack48_v1 import *\nfrom c137_sdk37_digest_ports_v1 import combined_generation_gate,upload_gate\nfrom sdk37_witness_scope_v1 import require_prepared\n'+ast.unparse(module)+'\n'
 if write:
  for name,text in result.items():(HERE/name).write_text(text)
 return {name:hashlib.sha256(text.encode()).hexdigest() for name,text in result.items()}
if __name__=='__main__':print(json.dumps(produce(),indent=2))
