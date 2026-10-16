"""FROZEN SOURCE ONLY. ROOT GO and cpu-run required; no decoder/GPU."""
from pathlib import Path
import argparse,ast,hashlib,json,mmap,re,struct,subprocess,shutil
P=Path(__file__).resolve().parent;R=P.parent
p=argparse.ArgumentParser()
for k in ['exe','object','archive','build-record','build-proof','out']:p.add_argument('--'+k,type=Path,required=True)
for k in ['exe-sha256','object-sha256','archive-sha256']:p.add_argument('--'+k,required=True)
a=p.parse_args();EXE=a.exe.resolve();OBJ=a.object.resolve();O=a.out.resolve();EXPECTED=a.exe_sha256
sha=lambda b:hashlib.sha256(b).hexdigest()
assert all(re.fullmatch('[0-9a-f]{64}',x)for x in [EXPECTED,a.object_sha256,a.archive_sha256])
assert not O.exists(),'Preserve prior audit; no blind rerun'
O.mkdir()
records={line.split(maxsplit=1)[1]:line.split(maxsplit=1)[0]for line in a.build_record.read_text().splitlines()}
assert records['/out/strata-q6-a4-s107']==EXPECTED
assert records['CMakeFiles/strata_kernels.dir/src/kernels/cuda/native_mmvq.dp.cpp.o']==a.object_sha256
assert records['libstrata_kernels.a']==a.archive_sha256
source=Path('/mnt/vm_8tb/b70/build/strata-ab/src-ep/sycl/src/kernels/cuda/native_mmvq.dp.cpp')
assert sha(source.read_bytes())=='ae47a632d0cb3a0fc5a992c335bdd1145d4bbb9210c5b481dd45e53cc3835252'
stats={str(f):[f.stat().st_ino,f.stat().st_size,f.stat().st_mtime_ns]for f in [EXE,OBJ,a.archive,source]}
with EXE.open('rb')as f:assert hashlib.file_digest(f,'sha256').hexdigest()==EXPECTED
object_bytes=OBJ.read_bytes();assert sha(object_bytes)==a.object_sha256
# Selected archive member only, not an archive-wide hash.
with a.archive.open('rb')as f:
 assert f.read(8)==b'!<arch>\n';longnames=b'';member=None
 while True:
  h=f.read(60)
  if not h:break
  assert len(h)==60 and h[58:]==b'`\n';nm=h[:16].decode().strip();n=int(h[48:58]);pos=f.tell()
  if nm=='//':longnames=f.read(n)
  else:
   actual=longnames[int(nm[1:]):].split(b'/\n')[0].decode()if nm.startswith('/')and nm[1:].isdigit()else nm.rstrip('/')
   if actual=='native_mmvq.dp.cpp.o':member=f.read(n);break
  f.seek(pos+n+(n&1))
assert member==object_bytes,'Rebuilt archive member differs from selected object'
(O/'build-proof.json').write_bytes(a.build_proof.read_bytes())
# Reuse ONLY the audited ELF class, bounded as in S97; no helper top-level execution.
helper = R / 'attn-value-binding-s92-build-audit/host_binding.py'
assert sha(helper.read_bytes()) == '37ecf55c43300319f0a3baaf15bede814934bd3ee1b0a11b32a38dc6066d4aa6'
node = next(n for n in ast.parse(helper.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'ELF')
def cstr(buf, start):
    end = buf.find(b'\0', start)
    assert end >= start
    return buf[start:end]
class Bound(ast.NodeTransformer):
    replacements = 0
    def visit_Subscript(self, n):
        n = self.generic_visit(n)
        c = n.value
        if isinstance(n.slice, ast.Constant) and n.slice.value == 0 and isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr == 'split' and isinstance(c.func.value, ast.Subscript):
            s = c.func.value
            if ((isinstance(s.value, ast.Name) and s.value.id == 'strings') or (isinstance(s.value, ast.Attribute) and s.value.attr == 'data' and isinstance(s.value.value, ast.Name) and s.value.value.id == 'self')) and isinstance(s.slice, ast.Slice) and s.slice.upper is None:
                self.replacements += 1
                return ast.copy_location(ast.Call(func=ast.Name(id='cstr', ctx=ast.Load()), args=[s.value, s.slice.lower], keywords=[]), n)
        return n
transform = Bound()
node = transform.visit(node)
assert transform.replacements == 4
ast.fix_missing_locations(node)
code = ast.unparse(node).replace('self.data = path.read_bytes()', "self._file=path.open('rb'); self.data=mmap.mmap(self._file.fileno(),0,access=mmap.ACCESS_READ)").replace('self.sha = hashlib.sha256(self.data).hexdigest()', 'self.sha=EXPECTED if path==EXE else hashlib.sha256(self.data).hexdigest()')
exec(compile(code, str(helper) + ':class-only-bounded-mmap', 'exec'))

obj,final=ELF(OBJ),ELF(EXE)
assert obj.sha==a.object_sha256
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
   z=obj.symbols[ri>>32];dst=obj.ss[z['section']];names.append(cstr(obj.data,dst[4]+z['offset']+add).decode())
 assert len(names)==1;refs=refs_to(obj,[info]+related)
 assert any(x['target_metadata']==info['name']and x['containing_functions']for x in refs),'No actual hostfunction->selected metadata reference'
 linked=final.symbol(info['name']);pointer=final.nameptr(linked);assert pointer['name']==names[0]
 return {'object_info':info,'object_cache_guard':related,'actual_host_relocations':refs,'final_info':linked,'final_cache_guard':[final.symbol(x['name'])for x in related],'registered_name_pointer':pointer}


hosts={'original':bind(obj,lambda n:'native_mmvq_q6k_wide_a2E18dpct_kernel_scalarILi6EE'in n),
       'center':bind(obj,lambda n:'native_mmvq_q6k_center6E'in n),
       'candidate':bind(obj,lambda n:'native_mmvq_q6k_a4_c6E'in n)}
refbindings=json.loads((P/'references.json').read_text())
for arm in ['original','center']:assert hosts[arm]['registered_name_pointer']['name']==refbindings[arm]['expected_registered_main_name']
assert len({h['final_info']['offset']for h in hosts.values()})==3
assert len({h['registered_name_pointer']['pointer']for h in hosts.values()})==3
assert len({h['registered_name_pointer']['name']for h in hosts.values()})==3
assert len({s['offset']for h in hosts.values()for s in h['final_cache_guard']})==6
assert shutil.which('objdump')
for arm,host in hosts.items():
 assert 'launch_q6k_wideILi6EE'in host['registered_name_pointer']['name']
 functions=sorted({fn for row in host['actual_host_relocations']if row['target_metadata']==host['object_info']['name']for fn in row['containing_functions']})
 assert functions,(arm,'No live hostfunction metadata reference')
 host['actual_callers']=functions
 for i,fn in enumerate(functions):(O/(arm+'-host-'+str(i)+'.asm')).write_bytes(subprocess.check_output(['objdump','-drw','--disassemble='+fn,str(OBJ)]))
def sections(buf, base=0):
    assert buf[base:base + 6] == b'\x7fELF\x02\x01'
    so = struct.unpack_from('<Q', buf, base + 40)[0]
    se, sn, si = struct.unpack_from('<HHH', buf, base + 58)
    assert se == 64 and 0 < sn < 1000 and si < sn
    ss = [struct.unpack_from('<IIQQQQIIQQ', buf, base + so + k * se) for k in range(sn)]
    st = ss[si]
    strings = buf[base + st[4]:base + st[4] + st[5]]
    named = {cstr(strings, s[0]).decode(): s for s in ss}
    size = max(so + se * sn, max(s[4] + s[5] for s in ss if s[1] != 8))
    assert base + size <= len(buf)
    return named, size

selected={}
get=lambda buf,tab,k:buf[tab[k][4]:tab[k][4]+tab[k][5]]if tab[k][1]!=8 else b''
for arm,host in hosts.items():
 name=host['registered_name_pointer']['name'];unique={}
 for match in re.compile(rb'(?m)^  - name:[ \t]+'+re.escape(name.encode())+rb'[ \t]*\r?$').finditer(final.data):
  at=match.start();base=final.data.rfind(b'\x7fELF\x02\x01',max(0,at-4*1024*1024),at);assert base>=0
  ss,size=sections(final.data,base);z=ss['.ze_info'];assert base+z[4]<=at<base+z[4]+z[5];unique[base]=(ss,size)
 assert len(unique)==1,(arm,'Registered target must occur in one actual full image',len(unique))
 base,(ss,size)=next(iter(unique.items()));image=bytes(final.data[base:base+size]);ze=get(image,ss,'.ze_info').decode();text=get(image,ss,'.text.'+name)
 names=re.findall(r'^  - name:\s+(\S+)',ze.split('kernels_misc_info:')[0],re.M);assert names.count(name)==1
 block=next(x for x in re.split(r'(?m)^  - name:\s+',ze.split('kernels_misc_info:')[0])[1:]if x.splitlines()[0].strip()==name)
 fields={k:re.findall(r'(?m)^\s*'+re.escape(k)+r':\s*(.+)$',block)for k in ['grf_count','simd_size','slm_size','barrier_count','spill_size','scratch_size','eu_thread_count']}
 options=get(image,ss,'.misc.buildOptions');assert options.rstrip(b'\0')==b'-cl-fp32-correctly-rounded-divide-sqrt'
 (O/(arm+'.zebin')).write_bytes(image);(O/(arm+'.ze_info')).write_text(ze,encoding='ascii')
 selected[arm]={'kernel':name,'executable_offset':base,'image_bytes':size,'image_sha256':sha(image),'text_bytes':len(text),'text_sha256':sha(text),'all_kernel_names':names,'execution_fields':fields,'full_kernel_metadata':block,'missing_fields':'absent, not measured0','inventory':{k:{'type':z[1],'flags':z[2],'offset':z[4],'bytes':z[5],'sha256':sha(get(image,ss,k))}for k,z in ss.items()}}
# Save actual selected identities before any reuse gate; failed comparisons stay evidence.
(O/'selection.json').write_text(json.dumps({'hosts':hosts,'selected':selected},indent=2)+'\n',encoding='ascii')
references=json.loads((P/'references.json').read_text());comparisons={}
for arm,ref in references.items():
 old=Path(ref['image']).read_bytes();assert sha(old)==ref['image_sha256'];os,_=sections(old)
 asm=Path(ref['ASM']).read_text();assert sha(asm.encode())==ref['ASM_sha256']
 end=ref['EOT_end'];assert re.findall(r'\{EOT,[^\n]*\nL(\d+):',asm)==[str(end)]
 assert all(int(x)<end for line in asm.splitlines()if re.search(r'\b(goto|jmpi|join)\b',line)for x in re.findall(r'\bL(\d+)\b',line))
 oldname=ref['name'];oldtext=get(old,os,'.text.'+oldname);oldze=get(old,os,'.ze_info').decode()
 image=(O/(arm+'.zebin')).read_bytes();ss,_=sections(image);name=selected[arm]['kernel'];text=get(image,ss,'.text.'+name);ze=get(image,ss,'.ze_info').decode()
 prefix=len(text)>=end and text[:end]==oldtext[:end]
 metadata=ze.split('kernels_misc_info:')[0].replace(name,'TARGET')==oldze.split('kernels_misc_info:')[0].replace(oldname,'TARGET')
 options=get(image,ss,'.misc.buildOptions')==get(old,os,'.misc.buildOptions')
 keys={k for tab in [ss,os]for k in tab if k.startswith(('.data','.bss','.rodata'))}
 constants=all(k in ss and k in os and all(ss[k][j]==os[k][j]for j in [1,2,5,8])and get(image,ss,k)==get(old,os,k)for k in keys)
 norm=lambda n,k:n.replace(k,'TARGET')
 rels=lambda data,tab,k:{norm(n,k):get(data,tab,n)for n in tab if n.startswith(('.rel.','.rela.'))}
 relocs=rels(image,ss,name)==rels(old,os,oldname)
 differences=[i for i,(x,y)in enumerate(zip(text,oldtext))if x!=y]
 inv=lambda data,tab:{k:{'type':z[1],'flags':z[2],'bytes':z[5],'sha256':sha(get(data,tab,k))}for k,z in tab.items()}
 rawnew,rawold=inv(image,ss),inv(old,os)
 sectiondiff={k:{'current':rawnew.get(k),'reference':rawold.get(k)}for k in sorted(set(rawnew)|set(rawold))if rawnew.get(k)!=rawold.get(k)}
 comparisons[arm]={'reference':ref,'raw_full_image_equal':image==old,'raw_whole_target_text_equal':text==oldtext,'computational_prefix_bytes':end,'computational_prefix_exact':prefix,'normalized_execution_payload_metadata_equal':metadata,'options_equal':options,'constant_sections_equal':constants,'relocation_sections_equal':relocs,'target_text_differing_offsets_common_length':differences,'target_text_length_delta':len(text)-len(oldtext),'all_post_EOT_common_differences':all(i>=end for i in differences),'reference_inventory':rawold,'all_raw_section_differences':sectiondiff,'PASS_reuse':prefix and metadata and options and constants and relocs}
# Short selected section-name alias on COPY only, for possible later ROOT decoder GO.
name=selected['candidate']['kernel'];image=(O/'candidate.zebin').read_bytes();ss,_=sections(image);t=ss['.text.'+name];alias=bytearray(image)
so=struct.unpack_from('<Q',image,40)[0];se,sn,si=struct.unpack_from('<HHH',image,58);st=struct.unpack_from('<IIQQQQIIQQ',image,so+si*se);at=st[4]+t[0];old=cstr(alias,at);new=b'.text.s107_a4_candidate';assert old==('.text.'+name).encode()and len(new)<len(old)
alias[at:at+len(old)]=new+b'\0'*(len(old)-len(new));changed=[i for i,(x,y)in enumerate(zip(image,alias))if x!=y];assert all(st[4]<=i<st[4]+st[5]for i in changed);assert bytes(alias[t[4]:t[4]+t[5]])==get(image,ss,'.text.'+name)
(O/'candidate-shortname.zebin').write_bytes(alias)
for f in [EXE,OBJ,a.archive,source]:assert stats[str(f)]==[f.stat().st_ino,f.stat().st_size,f.stat().st_mtime_ns]
passed=all(c['PASS_reuse']for c in comparisons.values())
result={'status':'PASS_SELECTED_MAIN_HOST_IMAGE_COMPUTATIONAL_REUSE_PENDING_ROOT_READ'if passed else'HOLD_COMPILED_DIFFERENCE_ROOT_INSPECT_BEFORE_TARGET_ONLY_DECODER','binary_sha256':EXPECTED,'object_sha256':a.object_sha256,'archive_sha256_from_root_build_record':a.archive_sha256,'archive_selected_member_equals_object':True,'source_sha256':sha(source.read_bytes()),'build_record_sha256':sha(a.build_record.read_bytes()),'build_proof_sha256':sha(a.build_proof.read_bytes()),'hosts':hosts,'selected':selected,'comparison':comparisons,'target_images_unique_and_distinct':len({x['executable_offset']for x in selected.values()})==3,'candidate_alias_copy':{'sha256':sha(alias),'only_section_name_changed':True,'selected_text_unchanged':True},'host_guard_nin_nout_argument_flow':'Exact launch<6> containing-function assembly saved per arm for root manual guard and n_in/n_out forwarding review; no automatic machine equivalence claim.','limits':['Full raw image/text/section differences retained, never normalized to raw identity.','Prefix references end at decoded EOT with known branch targets inside; no dynamic execution proof.','Normalized execution/payload excludes misc type-name inventory which remains raw.','Resource fields exact; absence not measuredzero.','Class-only bounded mmap ELF reuse and metadata-name seek, not an all-image extraction.','No compiler, decoder, GPU, timing or numerical invocation.','Root must accept actual main guard/argument flow and root build provenance.','Post-EOT byte differences are located, not assigned an undocumented meaning.']}
(O/'facts.json').write_text(json.dumps(result,indent=2)+'\n',encoding='ascii')
print(json.dumps({'status':result['status'],'resources':{k:v['execution_fields']for k,v in selected.items()},'reuse':{k:v['PASS_reuse']for k,v in comparisons.items()}}))
if not passed:raise SystemExit(5)
