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
# The IQ object uses local STT_SECTION+addend references, unlike S92's named weak refs.
refs=[]
for sec in obj.ss:
 if sec[1]!=4:continue
 for at in range(sec[4],sec[4]+sec[5],sec[9]):
  pos,ri,add=struct.unpack_from('<QQq',obj.data,at);z=obj.symbols[ri>>32];rt=ri&0xffffffff
  # For the observed x86 PC32 instruction fields, effective symbolic destination
  # includes the four-byte relocation field width. Absolute 32/64 refs do not.
  if rt not in [1,2,10]:continue
  destination=z['offset']+add+(4 if rt==2 else 0)
  for target in [info]+related:
   if z['section']==target['section']and destination==target['offset']:
    functions=[x['name']for x in obj.symbols if x['type']==2 and x['section']==sec[7]and x['offset']<=pos<x['offset']+x['size']]
    refs.append({'target_metadata':target['name'],'relocation_target_symbol':z['name'],'section_symbol':z['type']==3,'relocation_type':rt,'addend':add,'normalized_section_destination':destination,'source_section':obj.names[sec[7]],'containing_functions':functions})
assert any(x['target_metadata']==info['name']and any('launch_guILi17EE'in fn for fn in x['containing_functions'])for x in refs)
hostasm=(O/'host-launch-gu17.asm').read_text();assert 'R_X86_64_32\t.rodata+0x12538'in hostasm and'getDeviceKernelInfoERKNS1_27compile_time_kernel_info_v123CompileTimeKernelInfoTyE'in hostasm

linked=final.symbol(info['name']);pointer=final.nameptr(linked);assert pointer['name']==name;linked_related=[final.symbol(x['name'])for x in related]
# Compare only the relevant existing archive member, not a full archive digest.
archive=SRC/'build-sycl-ep/libstrata_kernels.a';member=None
with archive.open('rb')as f:
 assert f.read(8)==b'!<arch>\n';longnames=b''
 while True:
  h=f.read(60)
  if not h:break
  assert len(h)==60 and h[58:]==b'`\n';nm=h[:16].decode().strip();n=int(h[48:58]);pos=f.tell()
  if nm=='//':longnames=f.read(n)
  else:
   actual=longnames[int(nm[1:]):].split(b'/\n')[0].decode()if nm.startswith('/')and nm[1:].isdigit()else nm.rstrip('/')
   if actual=='iq_kernels.dp.cpp.o':member=f.read(n);break
  f.seek(pos+n+(n&1))
assert member is not None and sha(member)==obj.sha
# Direct registered-name search, then validate enclosing .ze_info and ELF.
# No full image walk; only selected names/section metadata and one image body.
def elf_at(base):
 data=final.data;assert data[base:base+6]==b'\x7fELF\x02\x01';so=struct.unpack_from('<Q',data,base+40)[0];se,sn,si=struct.unpack_from('<HHH',data,base+58);assert se==64 and 0<sn<1000 and si<sn
 sections=[struct.unpack_from('<IIQQQQIIQQ',data,base+so+k*se)for k in range(sn)];z=sections[si];strings=data[base+z[4]:base+z[4]+z[5]];named={bounded_cstring(strings,z[0]).decode():z for z in sections};size=max(so+se*sn,max(z[4]+z[5]for z in sections if z[1]!=8));assert base+size<=len(data);return named,size
pattern=re.compile(rb'(?m)^  - name:[ \t]+'+re.escape(name.encode())+rb'[ \t]*\r?$');unique={}
for match in pattern.finditer(final.data):
 at=match.start();base=final.data.rfind(b'\x7fELF\x02\x01',max(0,at-4*1024*1024),at);assert base>=0;sections,size=elf_at(base);z=sections['.ze_info'];assert base+z[4]<=at<base+z[4]+z[5];unique[base]=(sections,size)
assert len(unique)==1,len(unique);base,(sections,size)=next(iter(unique.items()));image=final.data[base:base+size];z=sections['.ze_info'];ze=image[z[4]:z[4]+z[5]].decode();target_names=re.findall(r'^  - name:\s+(\S+)',ze.split('kernels_misc_info:')[0],re.M);assert target_names==[name,'Intel_Symbol_Table_Void_Program'],target_names
t=sections['.text.'+name];text=image[t[4]:t[4]+t[5]];options=sections['.misc.buildOptions'];opts=image[options[4]:options[4]+options[5]].decode().strip('\0');assert opts=='-cl-fp32-correctly-rounded-divide-sqrt'
(O/'qt17.zebin').write_bytes(image);(O/'qt17.ze_info').write_text(ze,encoding='ascii')
# Always alias the selected text section on a COPY: long kernel filename exceeds NAME_MAX.
alias=bytearray(image);so=struct.unpack_from('<Q',image,40)[0];se,sn,si=struct.unpack_from('<HHH',image,58);ss=struct.unpack_from('<IIQQQQIIQQ',image,so+si*se);offset=ss[4]+t[0];old=bounded_cstring(alias,offset);new=b'.text.s97_qt17';assert old==('.text.'+name).encode()and len(new)<len(old);alias[offset:offset+len(old)]=new+b'\0'*(len(old)-len(new));assert alias[t[4]:t[4]+t[5]]==text
changed=[i for i,(a,b)in enumerate(zip(image,alias))if a!=b];assert all(ss[4]<=i<ss[4]+ss[5]for i in changed);(O/'qt17-shortname.zebin').write_bytes(alias)
for f in [EXE,objpath]:assert stats[str(f)]=={'inode':f.stat().st_ino,'bytes':f.stat().st_size,'mtime_ns':f.stat().st_mtime_ns}
demangled=subprocess.check_output(['c++filt',name],text=True).strip().removeprefix('typeinfo name for ');observed=s96['observed_primary_S95_exact_names']['QT17'].split('[SIMD32')[0];assert demangled==observed,(demangled,observed)
v={'status':'PASS_EXACT_ROUTED_QT17_HOST_IMAGE_BINDING','root_recorded_binary_sha256':EXPECTED,'actual_binary_sha256_checked_under_CPU_lease':actual_binary_sha256,'root_build_proof_sha256':sha(proof.read_bytes()),'binary_object_stats':stats,'source_sha256':s96['pins'][str(SRC/'sycl/src/kernels/cuda/iq_kernels.dp.cpp')]['sha256'],'object_sha256':obj.sha,'archive_member_byte_match':True,'archive_whole_hash_not_repeated':True,'host':{'object_CompileTimeKernelInfo':info,'object_related_cache_guard':related,'object_host_relocations':refs,'host_disassembly':str(O/'host-launch-gu17.asm'),'host_metadata_reference_kind':'STT_SECTION plus normalized addend, not named symbol references','final_CompileTimeKernelInfo':linked,'final_related_cache_guard':linked_related,'registered_name_pointer':pointer},'observed_exact_demangled_name':observed,'selected':{'kernel':name,'executable_offset':base,'image_bytes':size,'image_sha256':sha(image),'text_bytes':len(text),'text_sha256':sha(text),'ze_info_sha256':sha(ze.encode()),'build_options':opts,'unique_selected_full_image':True,'all_kernel_names_in_selected_image':target_names,'extra_text_sections':{k:sections[k][5]for k in sections if k.startswith('.text.')and k!='.text.'+name}},'alias_on_copy_only':{'reason':'avoid NAME_MAX output filename failure','original_section':old.decode(),'alias':new.decode(),'changed_offsets_count':len(changed),'only_shstrtab_changed':True,'unchanged_text_sha256':sha(text),'alias_image_sha256':sha(alias)},'selected_images':1,'all_image_walk':False,'GPU':0,'build':0}
(O/'selection.json').write_text(json.dumps(v,indent=2,ensure_ascii=True)+'\n',encoding='ascii');print(json.dumps({'status':v['status'],'selected':v['selected'],'host_relocations':len(refs),'mmap_binary_stat_unchanged':True},indent=2))
