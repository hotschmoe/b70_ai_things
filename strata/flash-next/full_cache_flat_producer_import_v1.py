"""Exact production local import roster for flat /controller bootstrap smoke."""
import ast,hashlib,importlib.abc,importlib.util,json,sys
from pathlib import Path
ENTRY='full_cache_shared_api_trace_v9'

def roster(here):
 here=Path(here);pending=[ENTRY];rows={}
 while pending:
  name=pending.pop()
  if name in rows:continue
  path=here/(name+'.py');raw=path.read_bytes();raw.decode('ascii');tree=ast.parse(raw);rows[name]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':raw.decode('ascii')}
  for node in ast.walk(tree):
   names=[]
   if isinstance(node,ast.Import):names=[n.name.split('.')[0]for n in node.names]
   if isinstance(node,ast.ImportFrom)and node.module:names=[node.module.split('.')[0]]
   if isinstance(node,ast.Call)and isinstance(node.func,ast.Name)and node.func.id=='__import__'and node.args and isinstance(node.args[0],ast.Constant)and isinstance(node.args[0].value,str):names=[node.args[0].value]
   if isinstance(node,ast.Constant)and type(node.value)is str and node.value.endswith('.py')and '/'not in node.value:names.append(node.value[:-3])
   pending.extend(n for n in names if (here/(n+'.py')).is_file()and n not in rows)
 return dict(sorted(rows.items()))

class Flat(importlib.abc.MetaPathFinder,importlib.abc.Loader):
 def __init__(self,rows):self.rows=rows
 def find_spec(self,fullname,path=None,target=None):return importlib.util.spec_from_loader(fullname,self,origin='/controller/'+fullname+'.py')if fullname in self.rows else None
 def create_module(self,spec):return None
 def exec_module(self,module):
  module.__file__='/controller/'+module.__name__+'.py';exec(compile(self.rows[module.__name__]['bytes'],module.__file__,'exec'),module.__dict__)

def smoke(rows):
 # Neighbor source reads must see the same mounted files as imports.
 original_bytes=Path.read_bytes;original_text=Path.read_text
 def mounted_bytes(path):
  if str(path).startswith('/controller/'):
   name=path.name[:-3] if path.suffix=='.py' else None
   if name in rows:return rows[name]['bytes'].encode('ascii')
  return original_bytes(path)
 Path.read_bytes=mounted_bytes
 Path.read_text=lambda path,*args,**kwargs:mounted_bytes(path).decode(kwargs.get('encoding')or 'utf-8') if str(path).startswith('/controller/') else original_text(path,*args,**kwargs)
 sys.meta_path.insert(0,Flat(rows))
 for name in rows:__import__(name)
 return {'complete':True,'flat_mount':'/controller','modules':{n:r['sha256']for n,r in rows.items()},'actual_server_bootstrap_executed':False}
if __name__=='__main__':print(json.dumps(smoke(json.loads(sys.stdin.read())),sort_keys=True))
