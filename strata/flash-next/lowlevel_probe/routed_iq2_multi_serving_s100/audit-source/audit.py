"""Future ROOT CPU-only selected MAIN audit. No build, decoder or GPU execution."""
from pathlib import Path
import ast,hashlib,json,mmap,re,struct,sys,subprocess,shutil
from compare import compare_images,raw_inventory
assert len(sys.argv)==6,'audit.py EXE ROOT_EXE_SHA IQ_OBJECT ROOT_OBJECT_SHA ROOT_BUILD_AUDIT_JSON'
P=Path(__file__).resolve().parent;R=P.parent;EXE=Path(sys.argv[1]).resolve();EXPECTED=sys.argv[2];OB=Path(sys.argv[3]).resolve();OBJECT_SHA=sys.argv[4];BUILD=Path(sys.argv[5]).resolve()
assert all(re.fullmatch('[0-9a-f]{64}',x)for x in [EXPECTED,OBJECT_SHA])
O=R/'routed-iq2-multi-s100-main-audit';O.mkdir(exist_ok=False)
sha=lambda b:hashlib.sha256(b).hexdigest()
proof=BUILD.read_bytes();json.loads(proof)
SRC=Path('/mnt/vm_8tb/b70/build/strata-ab/src-ep');iq=SRC/'sycl/src/kernels/cuda/iq_kernels.dp.cpp'
assert sha(iq.read_bytes())=='8bf1489cb53a2de442c051a8a64f577d330eadedbe200fd36ee967f33a29bb6d'
assert iq.read_bytes()==(R/'routed-iq2-multi-s100-source/iq_kernels.candidate.dp.cpp').read_bytes()
with EXE.open('rb')as f:assert hashlib.file_digest(f,'sha256').hexdigest()==EXPECTED
assert sha(OB.read_bytes())==OBJECT_SHA
# Exact reused class-only bounded mmap parser from S97; no module top-level execution.
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
exec(compile(code,str(helper)+':class-only-mmap','exec'))
public=ELF(OB);final=ELF(EXE)
# Member identity only; root build closure binds actual archive to final link.
archive=SRC/'build-sycl-ep/libstrata_kernels.a'
with archive.open('rb')as f:
 assert f.read(8)==b'!<arch>\n';longnames=b'';member=None
 while True:
  h=f.read(60)
  if not h:break
  assert len(h)==60 and h[58:]==b'`\n';nm=h[:16].decode().strip();n=int(h[48:58]);pos=f.tell()
  if nm=='//':longnames=f.read(n)
  else:
   actual=longnames[int(nm[1:]):].split(b'/\n')[0].decode()if nm.startswith('/')and nm[1:].isdigit()else nm.rstrip('/')
   if actual=='iq_kernels.dp.cpp.o':member=f.read(n);break
  f.seek(pos+n+(n&1))
assert member is not None and sha(member)==OBJECT_SHA
# Resolve named OR section+addend metadata references, with actual host owners saved.
def refs_to(obj,targets):
 refs=[]
 for sec in obj.ss:
  if sec[1]!=4:continue
  for at in range(sec[4],sec[4]+sec[5],sec[9]):
   pos,ri,add=struct.unpack_from('<QQq',obj.data,at);z=obj.symbols[ri>>32];rt=ri&0xffffffff
   if rt not in [1,2,10]:continue
   destination=z['offset']+add+(4 if rt==2 else 0)
   for target in targets:
    if z['section']==target['section']and destination==target['offset']:
     functions=[x['name']for x in obj.symbols if x['type']==2 and x['section']==sec[7]and x['offset']<=pos<x['offset']+x['size']]
     refs.append({'target_metadata':target['name'],'target_symbol':z['name'],'STT_SECTION':z['type']==3,'relocation_type':rt,'addend':add,'normalized_destination':destination,'source_section':obj.names[sec[7]],'containing_functions':functions})
 return refs

def bind(obj,predicate):
 infos=[x for x in obj.symbols if x['section']and'CompileTimeKernelInfoI'in x['name']and predicate(x['name'])];assert len(infos)==1,infos;info=infos[0]
 related=[x for x in obj.symbols if x['section']and'getDeviceKernelInfo'in x['name']and predicate(x['name'])];assert len(related)==2,related
 names=[]
 for sec in obj.ss:
  if sec[1]!=4 or sec[7]!=info['section']:continue
  for at in range(sec[4],sec[4]+sec[5],sec[9]):
   pos,ri,add=struct.unpack_from('<QQq',obj.data,at)
   if pos!=info['offset']:continue
   z=obj.symbols[ri>>32];dst=obj.ss[z['section']];names.append(bounded_cstring(obj.data,dst[4]+z['offset']+add).decode())
 assert len(names)==1;refs=refs_to(obj,[info]+related)
 assert any(x['target_metadata']==info['name']and x['containing_functions']for x in refs),'No actual hostfunction->selected metadata reference'
 linked=final.symbol(info['name']);pointer=final.nameptr(linked);assert pointer['name']==names[0]
 return {'object_info':info,'object_cache_guard':related,'actual_host_relocations':refs,'final_info':linked,'final_cache_guard':[final.symbol(x['name'])for x in related],'registered_name_pointer':pointer}

def off_pred(name):return'native_gu_port'in name and re.search(r'dpct_kernel_scalarILi17EE.*ILi8EE.*ILi1EE',name)is not None
hosts={'OFF':bind(public,off_pred),'ON':bind(public,lambda n:'native_gu_iq2xs_multi_s100'in n)}
s97=json.loads((R/'routed-iq2-existing-isa-s97/selection.json').read_text());assert hosts['OFF']['registered_name_pointer']['name']==s97['selected']['kernel'],'OFF registered name must remain unchanged'
assert len({h['final_info']['offset']for h in hosts.values()})==2
assert len({h['registered_name_pointer']['pointer']for h in hosts.values()})==2
assert len({h['registered_name_pointer']['name']for h in hosts.values()})==2
assert len({x['offset']for h in hosts.values()for x in h['final_cache_guard']})==4
def elf_at(base):
 data=final.data;assert data[base:base+6]==b'\x7fELF\x02\x01';so=struct.unpack_from('<Q',data,base+40)[0];se,sn,si=struct.unpack_from('<HHH',data,base+58);assert se==64 and 0<sn<1000 and si<sn
 ss=[struct.unpack_from('<IIQQQQIIQQ',data,base+so+k*se)for k in range(sn)];st=ss[si];strings=data[base+st[4]:base+st[4]+st[5]];sections={bounded_cstring(strings,z[0]).decode():z for z in ss};size=max(so+se*sn,max(z[4]+z[5]for z in ss if z[1]!=8));assert base+size<=len(data);return sections,size
selected={}
for arm,host in hosts.items():
 name=host['registered_name_pointer']['name'];pattern=re.compile(rb'(?m)^  - name:[ \t]+'+re.escape(name.encode())+rb'[ \t]*\r?$');unique={}
 for match in pattern.finditer(final.data):
  at=match.start();base=final.data.rfind(b'\x7fELF\x02\x01',max(0,at-4*1024*1024),at);assert base>=0;ss,size=elf_at(base);z=ss['.ze_info'];assert base+z[4]<=at<base+z[4]+z[5];unique[base]=(ss,size)
 assert len(unique)==1,(arm,len(unique));base,(ss,size)=next(iter(unique.items()));image=final.data[base:base+size];z=ss['.ze_info'];ze=image[z[4]:z[4]+z[5]].decode();names=re.findall(r'^  - name:\s+(\S+)',ze.split('kernels_misc_info:')[0],re.M);assert names.count(name)==1
 t=ss['.text.'+name];text=image[t[4]:t[4]+t[5]];opt=ss['.misc.buildOptions'];options=image[opt[4]:opt[4]+opt[5]].decode().strip('\0');assert options=='-cl-fp32-correctly-rounded-divide-sqrt'
 blocks=re.split(r'(?m)^  - name:\s+',ze.split('kernels_misc_info:')[0])[1:];block=next(x for x in blocks if x.splitlines()[0].strip()==name)
 fields={k:re.findall(r'(?m)^\s*'+re.escape(k)+r':\s*(.+)$',block)for k in ['grf_count','simd_size','slm_size','barrier_count','spill_size','scratch_size','eu_thread_count']}
 (O/(arm+'.zebin')).write_bytes(image);(O/(arm+'.ze_info')).write_text(ze,encoding='ascii')
 selected[arm]={'kernel':name,'offset':base,'image_bytes':size,'image_sha256':sha(image),'text_bytes':len(text),'text_sha256':sha(text),'all_kernel_names_in_this_selected_image':names,'extra_text_sections':{n:ss[n][5]for n in ss if n.startswith('.text.')and n!='.text.'+name},'execution_fields':fields,'full_kernel_metadata':block,'missing_resource_fields_mean':'absent, not measured0','build_options':options}
assert selected['OFF']['offset']!=selected['ON']['offset'],'Selected targets coalesced: HOLD actual enumeration; no blind retry'
v={'status':'HOST_BINDINGS_AND_SELECTED_IMAGES_SAVED_COMPARISON_PENDING','binary_sha256':EXPECTED,'IQ_object_sha256':OBJECT_SHA,'IQ_archive_member_exact':True,'source_sha256':sha(iq.read_bytes()),'root_build_audit':str(BUILD),'root_build_audit_sha256':sha(proof),'hosts':hosts,'selected':selected}
(O/'selection.json').write_text(json.dumps(v,indent=2)+'\n',encoding='ascii')
assert shutil.which('objdump')
functions=sorted({fn for h in hosts.values()for ref in h['actual_host_relocations']for fn in ref['containing_functions']})
assert functions,functions
for index,fn in enumerate(functions):(O/('host-function-'+str(index)+'.asm')).write_bytes(subprocess.check_output(['objdump','-drw','--disassemble='+fn,str(OB)]))
(O/'host-functions.json').write_text(json.dumps(functions,indent=2)+'\n',encoding='ascii')
refs={'OFF':(R/'routed-iq2-existing-isa-s97/qt17.zebin','37cb80cd3034716da7cb4916cbe2379a35abc375169c171568f8525fe780cd78',7296),'ON':(R/'routed-iq2-multi-s98-build-audit-v2/candidate.zebin','94b7bfa94926bc987e7f86eb1d79b17158a48912db7d1f7d74a7d300cf960c8e',22800)}
# Raw complete selected-section inventories survive even a strict HOLD.
inventories={arm:raw_inventory(ref,O/(arm+'.zebin'),selected[arm]['kernel'])for arm,(ref,expected,prefix)in refs.items()}
(O/'section-inventory.json').write_text(json.dumps(inventories,indent=2)+'\n',encoding='ascii')
comparisons={}
for arm,(ref,expected,prefix)in refs.items():
 try:comparisons[arm]=compare_images(ref,O/(arm+'.zebin'),expected,selected[arm]['kernel'],prefix)
 except Exception as error:
  (O/'comparison-HOLD.json').write_text(json.dumps({'arm':arm,'error':str(error),'selected':selected,'completed_comparisons':comparisons},indent=2)+'\n',encoding='ascii');raise
(O/'comparison.json').write_text(json.dumps(comparisons,indent=2)+'\n',encoding='ascii')
v.update({'status':'PASS_TWO_DISTINCT_MAIN_HOST_BINDINGS_PRIOR_COMPUTATIONAL_BODIES_EXACT_ROOT_HOST_BRANCH_READ_PENDING','comparisons':comparisons,'actual_host_functions':functions,'all_image_walk':False,'compiler_decoder_GPU_invocations':0,'limits':['No new ISA decode only if prior computational-body/metadata/codebook/relocation equality passes.','Raw full-image/text identity retained independently.','Root reads selected actual hostfunction refs/branches and exact source guard before GPU.','Source exactly8bf preserves generic FP ledger; matched emitted prefix corroborates previous compiled arithmetic.','No runtime numerical/latency claim; absent resource fields are not measured0.']})
(O/'facts.json').write_text(json.dumps(v,indent=2)+'\n',encoding='ascii');print(json.dumps({'status':v['status'],'resources':{k:x['execution_fields']for k,x in selected.items()}},indent=2))
