# Reuse pinned prior ELF class and the S104 exact section+addend binding functions only.
helper=R/'attn-value-binding-s92-build-audit/host_binding.py'
node=next(n for n in ast.parse(helper.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='ELF')
def cstr(buf,start):
 end=buf.find(b'\0',start);need(end>=start,'unterminated string');return buf[start:end]
class Bound(ast.NodeTransformer):
 replacements=0
 def visit_Subscript(self,n):
  n=self.generic_visit(n);c=n.value
  if isinstance(n.slice,ast.Constant) and n.slice.value==0 and isinstance(c,ast.Call) and isinstance(c.func,ast.Attribute) and c.func.attr=='split' and isinstance(c.func.value,ast.Subscript):
   s=c.func.value
   if ((isinstance(s.value,ast.Name) and s.value.id=='strings') or (isinstance(s.value,ast.Attribute) and s.value.attr=='data' and isinstance(s.value.value,ast.Name) and s.value.value.id=='self')) and isinstance(s.slice,ast.Slice) and s.slice.upper is None:
    self.replacements+=1;return ast.copy_location(ast.Call(func=ast.Name(id='cstr',ctx=ast.Load()),args=[s.value,s.slice.lower],keywords=[]),n)
  return n
transform=Bound();node=transform.visit(node);need(transform.replacements==4,'reviewed bounded string transforms');ast.fix_missing_locations(node)
code=ast.unparse(node).replace('self.data = path.read_bytes()',"self._file=path.open('rb'); self.data=mmap.mmap(self._file.fileno(),0,access=mmap.ACCESS_READ)").replace('self.sha = hashlib.sha256(self.data).hexdigest()',"self.sha=EXPECTED if path==EXE else ('064b63ce75d7568db541af5cf24eee07f41d0e54ea49813af9958a816d75d5ca' if path==BASE else hashlib.sha256(self.data).hexdigest())")
exec(compile(code,str(helper)+':class-only-bounded','exec'),globals())
prior=R/'q6-a4-s104-compiled-audit-prep/audit.py'
nodes=[n for n in ast.parse(prior.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in {'refs_to','bind','sections'}]
need(len(nodes)==3,'prior helper functions');exec(compile(ast.Module(body=nodes,type_ignores=[]),str(prior)+':three-functions-only','exec'),globals())
