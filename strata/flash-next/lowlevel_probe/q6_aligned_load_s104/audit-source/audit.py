"""FROZEN SOURCE ONLY until ROOT GO. Selected S104 host/image audit; no decoder/GPU."""
from pathlib import Path
import ast,hashlib,json,mmap,re,struct,subprocess,shutil
P=Path(__file__).resolve().parent;R=P.parent;S=R/'q6-a4-private-prep';B=S/'build';O=R/'q6-a4-s104-compiled-audit'
EXE=B/'q6_a4_leaf_s104';EXPECTED='632f9f86ec2bea2b030286da62c75aa3581c01190a2ecb203f6b1f6f4c7e8e8c'
OBJ=B/'q6_a4_leaf.o';OBJECT_SHA='b3189f26698503340aa92461d5733271ed9a8a5b5665a5e0407c718b55dee55c'
sha=lambda b:hashlib.sha256(b).hexdigest()
assert not O.exists(),'Preserve any prior result; no blind rerun'
O.mkdir()
assert(B/'build.exit').read_text().strip()==(B/'tee.exit').read_text().strip()=='0'
assert json.loads((B/'image.json').read_text())[0]['Id']=='sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2'
records={line.split(maxsplit=1)[1]:line.split(maxsplit=1)[0]for line in(B/'sha256.txt').read_text().splitlines()}
assert records['q6_a4_leaf_s104']==EXPECTED and records['q6_a4_leaf.o']==OBJECT_SHA
assert records['leaf.images.o']=='f5cd80478f7b5174ad424a5c78fc355a618145edb06f0ece57df63a1857ba7e8'
assert records['/common/mmvq-core.images.o']=='8f5a7efea1c815b497edfaa1bd33e02d86d4cf9ceea38fd9190d44bacfd7cd21'
assert records['/frozen/libstrata_kernels.a']=='f2b53eb482814b2c9a046e524a1f76dee717fddc6fc45073a492880bb153e9a8'
assert records['/frozen/libstrata_core.a']=='34e91c40f01b7aa0e652e195228dfc1fadba93e40006c12788c81426567b0286'
for filename,key,expected in [('q6_a4_leaf.cpp','/src/q6_a4_leaf.cpp','eb6b595aea3a6ab35f3e7e51f6995628319ec3a06119dab925554a060a94eb82'),('private_q6_a4.hpp','/src/private_q6_a4.hpp','a5c69ed76b35530f600e278dba019660723c5c67789d69dde4c5b67dc55548e9')]:assert sha((S/filename).read_bytes())==records[key]==expected
with EXE.open('rb')as f:assert hashlib.file_digest(f,'sha256').hexdigest()==EXPECTED
assert sha(OBJ.read_bytes())==OBJECT_SHA
assert sha((B/'leaf.images.o').read_bytes())==records['leaf.images.o']
hostlink=(B/'host-link.log').read_text();assert ' -fsycl-link 'not in hostlink and 'ocloc 'not in hostlink
linkmap=(B/'link.map').read_text()
for required in ['q6_a4_leaf.o','leaf.images.o','/common/mmvq-core.images.o','/frozen/libstrata_kernels.a','/frozen/libstrata_core.a','native_mmvq.dp.cpp.o']:assert required in linkmap,required
# Extract only the actual public native object from the same frozen input archive.
archive=R/'iq4xs-rpw2-s59-prep/frozen-inputs/libstrata_kernels.a'
with archive.open('rb')as f:
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
assert member is not None
PUBLIC=O/'public-native-archive-member.o';PUBLIC.write_bytes(member)
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

private,public,final=ELF(OBJ),ELF(PUBLIC),ELF(EXE)
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


def public_pred(n):return'N_123native_mmvq_q6k_wide_a2E18dpct_kernel_scalarILi6EE'in n
hosts={'public':bind(public,public_pred),'original':bind(private,lambda n:'PrivateC6A4ILb0EE'in n),'candidate':bind(private,lambda n:'PrivateC6A4ILb1EE'in n)}
assert len({h['final_info']['offset']for h in hosts.values()})==3
assert len({h['registered_name_pointer']['pointer']for h in hosts.values()})==3
assert len({h['registered_name_pointer']['name']for h in hosts.values()})==3
assert len({s['offset']for h in hosts.values()for s in h['final_cache_guard']})==6
assert 'launch_q6k_wideILi6EE'in hosts['public']['registered_name_pointer']['name']and'native_mmvq_q6k_wide_a2'in hosts['public']['registered_name_pointer']['name']
s102facts=json.loads((R/'q6-a4-s102-audit/facts.json').read_text());assert hosts['public']['registered_name_pointer']['name']==s102facts['selected']['kernel'],'Public C6A2 registered target must match S102 actual binding'
assert shutil.which('objdump')
for arm,host in hosts.items():
 obj=public if arm=='public'else private
 functions=sorted({fn for row in host['actual_host_relocations']if row['target_metadata']==host['object_info']['name']for fn in row['containing_functions']})
 assert functions,(arm,'No live containing host function')
 host['actual_callers']=functions
 for index,fn in enumerate(functions):(O/(arm+'-host-'+str(index)+'.asm')).write_bytes(subprocess.check_output(['objdump','-drw','--disassemble='+fn,str(obj.path)]))
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
# Alias ONLY a copy's selected text name for ONE later candidate decode.
name=selected['candidate']['kernel'];image=(O/'candidate.zebin').read_bytes();ss,_=sections(image);t=ss['.text.'+name];alias=bytearray(image)
so=struct.unpack_from('<Q',image,40)[0];se,sn,si=struct.unpack_from('<HHH',image,58);st=struct.unpack_from('<IIQQQQIIQQ',image,so+si*se);at=st[4]+t[0];old=cstr(alias,at);new=b'.text.s104_a4_candidate';assert old==('.text.'+name).encode()and len(new)<len(old)
alias[at:at+len(old)]=new+b'\0'*(len(old)-len(new));changed=[i for i,(a,b)in enumerate(zip(image,alias))if a!=b];assert all(st[4]<=i<st[4]+st[5]for i in changed);assert bytes(alias[t[4]:t[4]+t[5]])==get(image,ss,'.text.'+name)
(O/'candidate-shortname.zebin').write_bytes(alias)
selection={'status':'HOST_BINDINGS_SELECTED_IMAGES_SAVED_ORIGINAL_COMPARISON_PENDING','binary_sha256':EXPECTED,'private_object_sha256':OBJECT_SHA,'public_archive_member_sha256':sha(member),'build_sha_file_sha256':sha((B/'sha256.txt').read_bytes()),'link_map_sha256':sha((B/'link.map').read_bytes()),'hosts':hosts,'selected':selected,'selected_target_images_distinct':len({x['executable_offset']for x in selected.values()})==3,'alias_copy':{'sha256':sha(alias),'section_name':new.decode(),'only_section_name_bytes_changed':True,'changed_bytes':len(changed),'text_unchanged':True},'all_image_walk':False,'compiler_decoder_GPU_invocations':0}
(O/'selection.json').write_text(json.dumps(selection,indent=2)+'\n',encoding='ascii')
# Original compiled body must meet the already proven S85 reference gate.
OLD=R/'q6-center-s85-isa';old=(OLD/'original.zebin').read_bytes();assert sha(old)=='6bc2acb286c7c00eea4678f404681b329d42e6dac05d268bd995b5e6feb2d6b8'
os,_=sections(old);oldname='_ZTSN3s859PrivateC6ILb0EEE';oldtext=get(old,os,'.text.'+oldname);oldze=get(old,os,'.ze_info').decode()
image=(O/'original.zebin').read_bytes();ss,_=sections(image);name=selected['original']['kernel'];text=get(image,ss,'.text.'+name);ze=get(image,ss,'.ze_info').decode()
asm_path=OLD/'original-disasm/.text._ZTSN3s859PrivateC6ILb0EEE.asm';asm=asm_path.read_text();assert sha(asm.encode())=='668ee30e1417c8c70c2aedd99660f8a94bb215653531281e4c5a492d02711a09'
assert re.search(r'\{EOT,[^\n]*\nL16384:',asm)and all(int(x)<16384 for line in asm.splitlines()if re.search(r'\b(goto|jmpi|join)\b',line)for x in re.findall(r'\bL(\d+)\b',line))
prefix=text[:16384]==oldtext[:16384]and len(text)>=16384
metadata=ze.split('kernels_misc_info:')[0].replace(name,'TARGET')==oldze.split('kernels_misc_info:')[0].replace(oldname,'TARGET')
options=get(image,ss,'.misc.buildOptions')==get(old,os,'.misc.buildOptions')
constant_keys={k for tab in [ss,os]for k in tab if k.startswith(('.data','.bss','.rodata'))};constants=all(k in ss and k in os and all(ss[k][field]==os[k][field]for field in [1,2,5,8])and get(image,ss,k)==get(old,os,k)for k in constant_keys)
norm=lambda n,k:n.replace(k,'TARGET');newrels={norm(n,name):get(image,ss,n)for n in ss if n.startswith(('.rel.','.rela.'))};oldrels={norm(n,oldname):get(old,os,n)for n in os if n.startswith(('.rel.','.rela.'))};relocs=newrels==oldrels
comparison={'raw_full_image_exact':image==old,'raw_complete_text_exact':text==oldtext,'computational_prefix_bytes':16384,'computational_prefix_exact':prefix,'normalized_execution_payload_metadata_equal':metadata,'options_equal':options,'constant_data_equal':constants,'raw_relocation_sections_equal':relocs,'all_text_changed_byte_offsets_common_length':[i for i,(a,b)in enumerate(zip(text,oldtext))if a!=b],'text_length_delta':len(text)-len(oldtext),'reference_inventory':{k:{'bytes':z[5],'sha256':sha(get(old,os,k))}for k,z in os.items()},'current_inventory':selected['original']['inventory'],'reference_ASM':str(asm_path),'reference_ASM_sha256':sha(asm.encode())}
(O/'original-comparison.json').write_text(json.dumps(comparison,indent=2)+'\n',encoding='ascii')
passed=prefix and metadata and options and constants and relocs
selection.update({'status':'PASS_THREE_HOST_BINDINGS_ORIGINAL_S85_COMPUTATIONAL_REUSE_CANDIDATE_ISA_PENDING_ROOT_GO'if passed else'HOLD_ORIGINAL_DIFFERENCE_BEFORE_ANY_DECODER_ROOT_INSPECTION','original_comparison':comparison,'limits':['Actual static name/pointer/hostcall binding, not GPU dispatch or numerical proof.','Candidate load/DP4A/conversion/FP/scratch analysis is pending later authorized one selected decoder.','Resource fields reported exactly; absence not measuredzero.','Build recorded frozen full archive/common identity is reused; only public member read, no archive-wide rehash.','Embedded SPIR-V and misc argument type-name differences stay raw inventory; full image/text identity never silently asserted.']})
(O/'facts.json').write_text(json.dumps(selection,indent=2)+'\n',encoding='ascii');print(json.dumps({'status':selection['status'],'resources':{k:v['execution_fields']for k,v in selected.items()},'comparison':comparison},indent=2))
if not passed:raise SystemExit(5)
