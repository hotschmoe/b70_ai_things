"""ROOT future CPU-only post-build audit. No compiler, decoder, GPU or leaf execution."""
from pathlib import Path
import ast,hashlib,json,mmap,re,struct,sys
P=Path(__file__).resolve().parent;R=P.parent;S=R/'routed-iq2-multi-s98-prep';B=S/'build';EXE=B/'routed_iq2_leaf_s98';EXPECTED=sys.argv[1];assert re.fullmatch('[0-9a-f]{64}',EXPECTED)
O=R/'routed-iq2-multi-s98-build-audit-v2';O.mkdir(exist_ok=False)
sha=lambda b:hashlib.sha256(b).hexdigest()
assert(B/'build.exit').read_text().strip()==(B/'tee.exit').read_text().strip()=='0'
assert json.loads((B/'image.json').read_text())[0]['Id']=='sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2'
records=dict(line.split(maxsplit=1)[::-1]for line in(B/'sha256.txt').read_text().splitlines());assert records['routed_iq2_leaf_s98']==EXPECTED
for file,key in [(S/'routed_iq2_leaf.cpp','/src/routed_iq2_leaf.cpp'),(S/'private_iq2_gu.hpp','/src/private_iq2_gu.hpp'),(B/'routed_iq2_leaf.o','routed_iq2_leaf.o'),(B/'leaf.images.o','leaf.images.o')]:assert sha(file.read_bytes())==records[key],str(file)
with EXE.open('rb')as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==EXPECTED
assert ' -fsycl-link 'not in(B/'host-link.log').read_text()and 'ocloc 'not in(B/'host-link.log').read_text(),'Ordinary final host link required'
S97=R/'routed-iq2-existing-isa-s97';s97=json.loads((S97/'selection.json').read_text());assert s97['actual_binary_sha256_checked_under_CPU_lease']=='d94e96ca399d8ecf1f15403db754b645697e82eaa72c052a2d18f4b29defc247'
SRC=Path('/mnt/vm_8tb/b70/build/strata-ab/src-ep');iq=SRC/'sycl/src/kernels/cuda/iq_kernels.dp.cpp';assert sha(iq.read_bytes())==s97['source_sha256']=='acaa7da63733af9f3831b3a74b99e6176bf46a06e498e0a33c0591f7f4a29851'
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
public=ELF(SRC/'build-sycl-ep/CMakeFiles/strata_kernels.dir/src/kernels/cuda/iq_kernels.dp.cpp.o');private=ELF(B/'routed_iq2_leaf.o');final=ELF(EXE)
assert public.sha==s97['object_sha256']and public.sha.startswith('a61c'),public.sha
# The public host object must equal the actual frozen archive member used by the leaf.
archive=R/'iq4xs-rpw2-s59-prep/frozen-inputs/libstrata_kernels.a'
assert records['/frozen/libstrata_kernels.a']=='f2b53eb482814b2c9a046e524a1f76dee717fddc6fc45073a492880bb153e9a8'
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
assert member is not None and sha(member)==public.sha
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

def public_pred(name):return'native_gu_port'in name and re.search(r'dpct_kernel_scalarILi17EE.*ILi8EE.*ILi1EE',name)is not None
hosts={'public':bind(public,public_pred),'original':bind(private,lambda n:'PrivateGu17ILb0EE'in n),'candidate':bind(private,lambda n:'PrivateGu17ILb1EE'in n)}
assert hosts['public']['registered_name_pointer']['name']==s97['selected']['kernel']
assert len({h['final_info']['offset']for h in hosts.values()})==3
assert len({h['registered_name_pointer']['pointer']for h in hosts.values()})==3
assert len({h['registered_name_pointer']['name']for h in hosts.values()})==3
assert len({x['offset']for h in hosts.values()for x in h['final_cache_guard']})==6
# Locate ONLY each registered name; no all-image metadata or text walk.
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
# Additive qualified identity after the preserved v1 full-image failure.
# No altered image or source; compare actual selected public bytes directly.
def sections_of(image):
 so=struct.unpack_from('<Q',image,40)[0];se,sn,si=struct.unpack_from('<HHH',image,58);assert se==64
 rows=[struct.unpack_from('<IIQQQQIIQQ',image,so+k*se)for k in range(sn)];z=rows[si];strings=image[z[4]:z[4]+z[5]]
 return {bounded_cstring(strings,z[0]).decode():(z,image[z[4]:z[4]+z[5]])for z in rows}
prior=(S97/'qt17.zebin').read_bytes();current=(O/'public.zebin').read_bytes();before=sections_of(prior);after=sections_of(current);assert before.keys()==after.keys()
public_raw_image_exact=sha(current)==sha(prior);public_raw_text_exact=selected['public']['text_sha256']==s97['selected']['text_sha256'];assert not public_raw_image_exact and not public_raw_text_exact
textname='.text.'+hosts['public']['registered_name_pointer']['name'];helpername='.text.Intel_Symbol_Table_Void_Program';post_eot={textname:(7296,[7308,7324]),helpername:(80,[92,108])};section_comparison=[]
for name,(old_header,old_bytes)in before.items():
 new_header,new_bytes=after[name];row={'name':name,'s97_sha256':sha(old_bytes),'s98_sha256':sha(new_bytes),'s97_bytes':len(old_bytes),'s98_bytes':len(new_bytes),'exact':old_bytes==new_bytes}
 # Offset changes are from the smaller embedded SPIR-V section; no hidden change
 # to section types, flags, addressing, link/info/alignment or entry layout.
 assert all(old_header[k]==new_header[k]for k in [0,1,2,3,6,7,8,9]),name
 if name in post_eot:
  prefix,words=post_eot[name];expected=[q for at in words for q in range(at,at+4)];actual=[i for i,(x,y)in enumerate(zip(old_bytes,new_bytes))if x!=y]
  assert len(old_bytes)==len(new_bytes)and actual==expected and old_bytes[:prefix]==new_bytes[:prefix],name
  # Retained S97 actual ASM binds these exact four-byte fields to the two
  # MOV-to-null immediates AFTER the EOT instruction, not computational data.
  assert [struct.unpack_from('<I',old_bytes,x)[0]for x in words]==[0xe307351b,0xf3758ade]
  assert [struct.unpack_from('<I',new_bytes,x)[0]for x in words]==[0xde39f167,0x976af49f]
  row.update({'exact_prefix_through_EOT_and_alignment_bytes':prefix,'only_changed_byte_offsets':actual,'classification':'two post-EOT MOV-to-null immediate fields; no broader trailer semantics asserted'})
 elif name=='.spv':
  assert len(old_bytes)==24212 and len(new_bytes)==24180 and old_bytes!=new_bytes
  row['classification']='embedded SPIR-V differs; not the emitted AOT instruction stream; no universal semantic-equivalence claim'
 else:assert old_bytes==new_bytes,name
 section_comparison.append(row)
qualified_public_identity={'raw_full_image_exact_S97':False,'raw_complete_text_exact_S97':False,'computational_prefix_through_EOT_plus_alignment_exact':True,'target_exact_prefix_bytes':7296,'helper_exact_prefix_bytes':80,'other_sections_exact_except_embedded_spirv':True,'sections':section_comparison,'preserved_v1_failure':str(R/'routed-iq2-multi-s98-build-audit'),'basis_actual_S97_ASM':[str(S97/'disasm/.text.s97_qt17.asm'),str(S97/'disasm/.text.Intel_Symbol_Table_Void_Program.asm')]}
assert len({x['kernel']for x in selected.values()})==3 and len({x['offset']for x in selected.values()})==3
v={'status':'PASS_THREE_DISTINCT_HOST_BINDINGS_QUALIFIED_PUBLIC_COMPUTATIONAL_IDENTITY_ROOT_RESOURCE_REVIEW_PENDING','binary_sha256':EXPECTED,'public_object_sha256':public.sha,'private_object_sha256':private.sha,'source_header_sha256':records['/src/private_iq2_gu.hpp'],'source_leaf_sha256':records['/src/routed_iq2_leaf.cpp'],'public_archive_member_match':True,'public_full_image_exact_S97':False,'public_raw_text_exact_S97':False,'qualified_public_identity':qualified_public_identity,'hosts':hosts,'selected':selected,'all_image_walk':False,'ocloc_or_compiler_invocations':0,'GPU_or_leaf_execution':0,'limits':['Actual host-reference/cache/guard/namepointer/image ownership, not runtime dispatch.','Resource fields/absence require root review; no physical occupancy/traffic/speed claim.','No private output correctness inferred from names or distinct images.','No decoder required for this binding audit; S97 computational prefix has actual ISA proof; full public image/text identity failed and is explicitly qualified.']}
(O/'facts.json').write_text(json.dumps(v,indent=2,ensure_ascii=True)+'\n',encoding='ascii')
print(json.dumps({'status':v['status'],'selected':{k:{z:x[z]for z in ['image_sha256','text_sha256','execution_fields']}for k,x in selected.items()}},indent=2))
