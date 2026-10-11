"""Named exact host file-remount/PATH observation; never old runtime PASS."""
import ast,copy,hashlib,os,types
from pathlib import Path
from serial37_canonical_json_v3 import canonical
import qualify_hc35_host_runtime_v1 as scalar
import postboot_original_native_manifest_v1 as model_port

HOST_SOURCE_SHA={'qualify_hc35_host_runtime_v1': 'bc40f0546d570dc2741edb2c6cfc346ab8dc1af3605df95387839c8144a02aa5', 'qualify_hc35_host_bulk_runtime_v1': 'b88e940681e04be88edc2bd100bbdbb1377506585dd2cabd6dc5096b7946231c'}
SCALAR_SOURCE_SHA='bc40f0546d570dc2741edb2c6cfc346ab8dc1af3605df95387839c8144a02aa5'
def require(ok,msg):
 if not ok:raise ValueError(msg)
def file_rows(value):
 rows={}
 def walk(x):
  if type(x)is dict:
   if set(x)=={'path','sha256','stat'}:
    key=x['path'];require(key not in rows or canonical(rows[key])==canonical(x),'Historical file has contradictory bindings');rows[key]=x
   else:
    for v in x.values():walk(v)
  elif type(x)is list:
   for v in x:walk(v)
 walk(value);return rows

class Association:
 def __init__(self,historical,model_association):
  self.model=model_association;self.allowed=file_rows(historical);self.files={};self.paths=[];self.runtime=[]
 def compare(self,left,right):
  if type(left)is dict and set(left)=={'path','sha256','stat'}:
   require(type(right)is dict and set(right)==set(left) and left['path'] in self.allowed,'Explicit historical file binding/roster required')
   expected=self.allowed[left['path']]
   if canonical(left)!=canonical(expected):
    require(canonical(right)==canonical(expected),'Neither file binding matches declared historical roster');left,right=right,left
   require(left['path']==right['path']==str(Path(left['path']).resolve()) and left['sha256']==right['sha256'],'Host file path/content differs')
   a,b=left['stat'],right['stat'];require(type(a)is list and type(b)is list and len(a)==len(b)==5 and all(type(v)is int for v in a+b) and a[1:]==b[1:],'Host inode/size/timestamps differ')
   if a[0]!=b[0]:require(a[0]in {x['historical_stat5'][0]for x in self.model['mapping']}and b[0]in {x['current_stat5'][0]for x in self.model['mapping']},'Only exact associated remount device mapping allowed')
   require(0<right['stat'][2]<=32<<20 and Path(left['path']).stat().st_size==right['stat'][2],'Bounded declared host-only file required');actual=scalar.file_binding(left['path']);require(canonical(actual)==canonical(right),'Current host file bytes changed during association');self.files[left['path']]={'historical':left,'current':right,'old_current_stat_gate_passed':False};return True
  if type(left)is dict:
   require(type(right)is dict and set(left)==set(right),'Historical host object fields differ')
   for k in left:self.compare(left[k],right[k])
   return True
  if type(left)is list:
   require(type(right)is list and len(left)==len(right),'Host sequence shape differs')
   for a,b in zip(left,right):self.compare(a,b)
   return True
  require(canonical(left)==canonical(right),'Non-stat host evidence changed');return True
 def runtime_compare(self,old,current):
  require(type(old)is dict and type(current)is dict and set(old)==set(current),'Complete current host runtime fields required')
  for key in old:
   if key!='execution_environment':self.compare(old[key],current[key])
  a,b=old['execution_environment'],current['execution_environment'];require('PATH'in a and 'PATH'in b and canonical({k:v for k,v in a.items()if k!='PATH'})==canonical({k:v for k,v in b.items()if k!='PATH'}),'NonPATH host environment changed')
  require(all(a.get(k)==b.get(k)=='1'for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')),'Exact actual host thread pins required')
  self.paths.append({'historical_PATH':a['PATH'],'current_PATH':b['PATH'],'PATH_was_not_patched':True,'all_other_environment_exact':True});self.runtime.append(current);return True
 def evidence(self):
  for path,row in self.files.items():require(canonical(scalar.file_binding(path))==canonical(row['current']),'Host bytes changed before final association')
  return {'schema':1,'files':self.files,'PATH_observations':self.paths,'current_host_runtime':self.runtime,'absolute_ldd':scalar.file_binding('/usr/bin/ldd'),'old_current_runtime_gate_passed':False,'stat_read_or_environment_monkeypatch_used':False,'historical_GPU_health_transferred':False,'device_intrinsics_qualified':False,'full_model_math_qualified':False}

def current_runtime(helper):
 # Same pinned original runtime predicates; only tool lookup becomes absolute.
 source=Path(scalar.__file__).read_bytes();require(hashlib.sha256(source).hexdigest()==SCALAR_SOURCE_SHA,'Exact original host runtime source required');tree=ast.parse(source);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='runtime_binding');node=copy.deepcopy(node)
 class Absolute(ast.NodeTransformer):
  def visit_List(self,n):
   n=self.generic_visit(n)
   if n.elts and isinstance(n.elts[0],ast.Constant)and n.elts[0].value=='ldd':n.elts[0]=ast.Constant('/usr/bin/ldd')
   return n
 ns=dict(vars(scalar));exec(compile(ast.fix_missing_locations(ast.Module([Absolute().visit(node)],type_ignores=[])),'[current-absolute-host-ldd]', 'exec'),ns);return ns['runtime_binding'](helper)

class Port(ast.NodeTransformer):
 def visit_Name(self,n):
  if n.id=='__file__':return ast.copy_location(ast.Name('_ORIGINAL_SOURCE',ast.Load()),n)
  return n
 def visit_Call(self,n):
  n=self.generic_visit(n)
  if isinstance(n.func,ast.Attribute)and isinstance(n.func.value,ast.Name)and n.func.value.id=='old':
   if n.func.attr=='finalized_binding':n.func=ast.Name('_scalar_history',ast.Load())
   elif n.func.attr=='runtime_binding':n.func=ast.Name('_current_runtime',ast.Load())
   elif n.func.attr=='validate_fixture':n.func=ast.Name('_validate_fixture',ast.Load())
  if isinstance(n.func,ast.Name)and n.func.id=='runtime_binding':n.func=ast.Name('_current_runtime',ast.Load())
  return n
 def visit_Compare(self,n):
  n=self.generic_visit(n)
  parts=[n.left,*n.comparators]
  if len(parts)==3 and isinstance(parts[2],ast.Name)and parts[2].id=='current':
   return ast.copy_location(ast.BoolOp(ast.And(),[ast.Compare(parts[0],[ast.Eq()],[parts[1]]),ast.Call(ast.Name('_runtime_associate',ast.Load()),[parts[1],parts[2]],[])]),n)
  if len(parts)==2 and all(isinstance(x,ast.Eq)for x in n.ops):
   for x in parts:
    if isinstance(x,ast.Call)and ((isinstance(x.func,ast.Name)and x.func.id in ('file_binding','build_binding'))or(isinstance(x.func,ast.Attribute)and x.func.attr=='file_binding')):return ast.copy_location(ast.Call(ast.Name('_file_associate',ast.Load()),parts,[]),n)
  return n
 def visit_Return(self,n):
  n=self.generic_visit(n)
  if isinstance(n.value,ast.Dict):
   for i,key in enumerate(n.value.keys):
    if isinstance(key,ast.Constant)and key.value=='current_host_runtime':n.value.values[i]=ast.Name('_historical_runtime',ast.Load())
  return n

def derived(original,names,context,extra=None):
 source=Path(original.__file__).read_bytes();require(hashlib.sha256(source).hexdigest()==HOST_SOURCE_SHA[original.__name__],'Exact original scalar/bulk source required');nodes=[copy.deepcopy(n)for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name in names];require({n.name for n in nodes}==set(names),'Exact declared readonly host predicates required');port=Port();ns={**vars(original),'__file__':__file__,'_ORIGINAL_SOURCE':original.__file__,'_file_associate':context.compare,'_runtime_associate':context.runtime_compare,'_current_runtime':current_runtime};ns.update(extra or {});exec(compile(ast.fix_missing_locations(ast.Module([port.visit(n)for n in nodes],type_ignores=[])),original.__file__+'[explicit-postboot-history]','exec'),ns);return ns
