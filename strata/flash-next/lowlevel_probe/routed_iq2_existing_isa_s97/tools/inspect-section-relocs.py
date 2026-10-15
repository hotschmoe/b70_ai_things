from pathlib import Path
import ast,hashlib,json,mmap,re,struct,subprocess
P=Path(__file__).resolve().parent;R=P.parent;O=R/'routed-iq2-existing-isa-s97';EXE=R/'attn-value-resident-s94-build/build/strata-attn-value-resident-s94';EXPECTED='d94e96ca399d8ecf1f15403db754b645697e82eaa72c052a2d18f4b29defc247';SRC=Path('/mnt/vm_8tb/b70/build/strata-ab/src-ep');objpath=SRC/'build-sycl-ep/CMakeFiles/strata_kernels.dir/src/kernels/cuda/iq_kernels.dp.cpp.o';sha=lambda b:hashlib.sha256(b).hexdigest()
s96=json.loads((R/'routed-dot-lowering-s96-source/source-pins.json').read_text());proof=R/'attn-value-resident-s94-build/build/root-build-closure.json';assert json.loads(proof.read_text())['binary_sha256']==EXPECTED
for path,row in s96['pins'].items():assert sha(Path(path).read_bytes())==row['sha256'],path
with EXE.open('rb')as stream:actual_binary_sha256=hashlib.file_digest(stream,'sha256').hexdigest()
assert actual_binary_sha256==EXPECTED
stats={str(f):{'inode':f.stat().st_ino,'bytes':f.stat().st_size,'mtime_ns':f.stat().st_mtime_ns}for f in [EXE,objpath]}
# Reuse only the already audited parser class; bounded C strings/mmap as S93.
helper=R/'attn-value-binding-s92-build-audit/host_binding.py';assert sha(helper.read_bytes())=='37ecf55c43300319f0a3baaf15bede814934bd3ee1b0a11b32a38dc6066d4aa6'
node=next(n for n in ast.parse(helper.read_text()).body if isinstance(n,ast.ClassDef)and n.name=='ELF')
def bounded_cstring(buf,start):
 end=buf.find(b'\0',start);assert end>=start;return buf[start:end]
class Bound(ast.NodeTransformer):
 replacements=0
 def visit_Subscript(self,item):
  item=self.generic_visit(item);call=item.value
  if isinstance(item.slice,ast.Constant)and item.slice.value==0 and isinstance(call,ast.Call)and isinstance(call.func,ast.Attribute)and call.func.attr=='split'and isinstance(call.func.value,ast.Subscript):
   suffix=call.func.value
   if ((isinstance(suffix.value,ast.Name)and suffix.value.id=='strings')or(isinstance(suffix.value,ast.Attribute)and suffix.value.attr=='data'and isinstance(suffix.value.value,ast.Name)and suffix.value.value.id=='self'))and isinstance(suffix.slice,ast.Slice)and suffix.slice.upper is None:
    self.replacements+=1;return ast.copy_location(ast.Call(func=ast.Name(id='bounded_cstring',ctx=ast.Load()),args=[suffix.value,suffix.slice.lower],keywords=[]),item)
  return item
transform=Bound();node=transform.visit(node);assert transform.replacements==4;ast.fix_missing_locations(node);code=ast.unparse(node).replace('self.data = path.read_bytes()',"self._file=path.open('rb'); self.data=mmap.mmap(self._file.fileno(),0,access=mmap.ACCESS_READ)").replace('self.sha = hashlib.sha256(self.data).hexdigest()',"self.sha=EXPECTED if path==EXE else hashlib.sha256(self.data).hexdigest()")
exec(compile(code,str(helper)+':class-only-mmap','exec'));obj=ELF(objpath);final=ELF(EXE)
def wanted(name):return 'native_gu_port'in name and re.search(r'dpct_kernel_scalarILi17EE.*ILi8EE.*ILi1EE',name)is not None
infos=[x for x in obj.symbols if x['section']and'CompileTimeKernelInfoI16dpct_kernel_name'in x['name']and wanted(x['name'])];assert len(infos)==1,len(infos);info=infos[0]
related=[x for x in obj.symbols if x['section']and'getDeviceKernelInfo'in x['name']and wanted(x['name'])];assert len(related)==2
names=[]
for sec in obj.ss:
 if sec[1]!=4 or sec[7]!=info['section']:continue
 for at in range(sec[4],sec[4]+sec[5],sec[9]):
  pos,ri,add=struct.unpack_from('<QQq',obj.data,at)
  if pos!=info['offset']:continue
  z=obj.symbols[ri>>32];dst=obj.ss[z['section']];names.append(bounded_cstring(obj.data,dst[4]+z['offset']+add).decode())
assert len(names)==1;name=names[0];assert wanted(name)

print(json.dumps({'info':info,'related':related,'selected_sections':{str(x['section']):obj.names[x['section']]for x in [info]+related}},indent=2))
for sec in obj.ss:
 if sec[1]!=4:continue
 for at in range(sec[4],sec[4]+sec[5],sec[9]):
  pos,ri,add=struct.unpack_from('<QQq',obj.data,at);z=obj.symbols[ri>>32]
  for dst in [info]+related:
   if z['section']==dst['section']and abs(z['offset']+add-dst['offset'])<=8:
    functions=[x['name']for x in obj.symbols if x['type']==2 and x['section']==sec[7]and x['offset']<=pos<x['offset']+x['size']]
    print(json.dumps({'destination':dst['name'],'target_symbol':z,'addend':add,'type':ri&0xffffffff,'source_section':obj.names[sec[7]],'containing_functions':functions}))
