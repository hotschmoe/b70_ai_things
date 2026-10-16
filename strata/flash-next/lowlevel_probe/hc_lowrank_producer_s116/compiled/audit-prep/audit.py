"""SOURCE ONLY. Future root CPU_AUDIT_GO: six selected S116 static bindings; no decoder/GPU."""
from pathlib import Path
import argparse,ast,hashlib,json,mmap,re,struct,subprocess
P=Path(__file__).resolve().parent;R=P.parent
ap=argparse.ArgumentParser()
ap.add_argument('--prep',type=Path,required=True)
for key in ['expected-prep-manifest-sha256','expected-exe-sha256','expected-object-sha256','expected-images-sha256']:
 ap.add_argument('--'+key,required=True)
a=ap.parse_args();S=a.prep.resolve();B=S/'build';O=R/(S.name+'-built-audit')
assert S.parent==R and(S.name=='hc-producer-s116-oracle-prep'or S.name.startswith('hc-producer-s116-oracle-prep-v'))
EXE=B/'hc_producer_s116';OBJ=B/'oracle.o';IMAGES=B/'private.images.o';EXPECTED=a.expected_exe_sha256
for h in [a.expected_prep_manifest_sha256,EXPECTED,a.expected_object_sha256,a.expected_images_sha256]:assert re.fullmatch('[0-9a-f]{64}',h)
sha=lambda b:hashlib.sha256(b).hexdigest()
def save(n,x):(O/n).write_text(json.dumps(x,indent=2,ensure_ascii=True)+'\n',encoding='ascii')
def stat(p):
 s=p.stat();return {'inode':s.st_ino,'size':s.st_size,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns}
def require(b,m):
 if not b:raise RuntimeError('HOLD: '+m)
require(not O.exists(),'preserve prior audit; no automatic retry');O.mkdir()
try:
 pins=json.loads((P/'pins.json').read_text())
 for x in pins['small_files']:
  p=Path(x['path']);b=p.read_bytes();require(len(b)==x['bytes']and sha(b)==x['sha256'],'small helper/reference pin '+str(p))
 manifest_raw=(S/'MANIFEST.json').read_bytes();require(sha(manifest_raw)==a.expected_prep_manifest_sha256,'root-selected exact source manifest')
 manifest=json.loads(manifest_raw)
 for row in manifest['files']:
  p=S/row['path'];require(p.parent==S,'source manifest basename only');b=p.read_bytes();require(len(b)==row['bytes']and sha(b)==row['sha256'],'frozen reviewed source payload '+row['path'])
 projection=json.loads((S/'source-projection.json').read_text())
 require(projection['all_six_source_chunks_inverse_exact']and projection['candidate_reduce_inverse_exact']and projection['candidate_up_inverse_exact'],'source inverse projection')
 require((B/'build.exit').read_text().strip()=='0','private build terminal0')
 require(json.loads((B/'image.json').read_text())[0]['Id']==pins['image'],'build image')
 require('2026.1'in(B/'compiler-version.txt').read_text(),'DPC++2026.1 compiler record')
 require((B/'status.txt').read_text().strip()=='BUILD_COMPLETE_SELECTED_CONTROL_AND_CANDIDATE_IMAGE_AUDIT_REQUIRED_BEFORE_GPU','build completion status')
 records={line.split(maxsplit=1)[1]:line.split(maxsplit=1)[0]for line in(B/'build-sha256.txt').read_text().splitlines()}
 for filename in ['oracle.cpp','private_hc.hpp','anchor_names.hpp']:require(sha((S/filename).read_bytes())==records['/src/'+filename],'build source binding '+filename)
 expected_artifacts={EXE:(EXPECTED,'hc_producer_s116'),OBJ:(a.expected_object_sha256,'oracle.o'),IMAGES:(a.expected_images_sha256,'private.images.o')}
 before={str(p):stat(p)for p in expected_artifacts}
 for p,(expected,name)in expected_artifacts.items():
  require(records[name]==expected,'root-selected build artifact record '+name)
  with p.open('rb')as f:require(hashlib.file_digest(f,'sha256').hexdigest()==expected,'actual private artifact '+name)
 buildlog=(B/'build.log').read_text()
 for line in(S/'expected-container.sha256').read_text().splitlines():
  _,name=line.split(maxsplit=1);require(buildlog.count(name+': OK')==2,'pre/post small build input binding '+name)
 hostlink=(B/'host-link.log').read_text();require(' -fsycl-link 'not in hostlink and 'ocloc 'not in hostlink,'host link did not AOT again')
 # Do not read/scan/hash any model, frozen main, full archive or original object.
 # Reuse only pinned class/functions from the already reviewed ELF idiom.
 helper=R/'attn-value-binding-s92-build-audit/host_binding.py'
 node=next(n for n in ast.parse(helper.read_text()).body if isinstance(n,ast.ClassDef)and n.name=='ELF')
 def cstr(buf,start):
  end=buf.find(b'\0',start);require(end>=start,'unterminated ELF string');return buf[start:end]
 class Bound(ast.NodeTransformer):
  replacements=0
  def visit_Subscript(self,n):
   n=self.generic_visit(n);c=n.value
   if isinstance(n.slice,ast.Constant)and n.slice.value==0 and isinstance(c,ast.Call)and isinstance(c.func,ast.Attribute)and c.func.attr=='split'and isinstance(c.func.value,ast.Subscript):
    s=c.func.value
    if ((isinstance(s.value,ast.Name)and s.value.id=='strings')or(isinstance(s.value,ast.Attribute)and s.value.attr=='data'and isinstance(s.value.value,ast.Name)and s.value.value.id=='self'))and isinstance(s.slice,ast.Slice)and s.slice.upper is None:
     self.replacements+=1;return ast.copy_location(ast.Call(func=ast.Name(id='cstr',ctx=ast.Load()),args=[s.value,s.slice.lower],keywords=[]),n)
   return n
 transform=Bound();node=transform.visit(node);require(transform.replacements==4,'four bounded string transforms');ast.fix_missing_locations(node)
 code=ast.unparse(node).replace('self.data = path.read_bytes()',"self._file=path.open('rb'); self.data=mmap.mmap(self._file.fileno(),0,access=mmap.ACCESS_READ)").replace('self.sha = hashlib.sha256(self.data).hexdigest()',"self.sha=EXPECTED if path==EXE else a.expected_object_sha256")
 exec(compile(code,str(helper)+':class-only-bounded','exec'),globals())
 prior=R/'q6-a4-s104-compiled-audit-prep/audit.py'
 funcs=[n for n in ast.parse(prior.read_text()).body if isinstance(n,ast.FunctionDef)and n.name in {'refs_to','bind','sections'}]
 require(len(funcs)==3,'three exact reused helper functions');exec(compile(ast.Module(body=funcs,type_ignores=[]),str(prior)+':functions-only','exec'),globals())
 final=ELF(EXE);obj=ELF(OBJ)
 tags={'norm':'HCNormS116','down':'HCDownS116','control_reduce':'HCControlReduceS116','control_up':'HCControlUpS116','candidate_reduce':'HCCandidateReduceS116','candidate_up':'HCCandidateUpS116'}
 hosts={arm:bind(obj,lambda n,tag=tag:tag in n)for arm,tag in tags.items()}
 require(len({h['final_info']['offset']for h in hosts.values()})==6,'six unique linked Info')
 require(len({h['registered_name_pointer']['pointer']for h in hosts.values()})==6,'six unique linked name pointers')
 require(len({h['registered_name_pointer']['name']for h in hosts.values()})==6,'six unique registered kernel names')
 require(len({x['offset']for h in hosts.values()for x in h['final_cache_guard']})==12,'twelve unique cache/guard cells')
 # Do not accept tag presence as the complete materialized-HC class inventory.
 observed_infos=[z['name']for z in obj.symbols if z['section']and 'CompileTimeKernelInfoI'in z['name']and's116_private'in z['name']]
 require(set(observed_infos)=={h['object_info']['name']for h in hosts.values()},'no extra materialized S116 HC Info')
 functions=sorted({fn for h in hosts.values()for row in h['actual_host_relocations']if row['target_metadata']==h['object_info']['name']for fn in row['containing_functions']})
 require(bool(functions),'selected actual host callers');hostfiles={}
 for i,fn in enumerate(functions):
  name=f'host-{i}.asm';raw=subprocess.check_output(['objdump','-drw','--disassemble='+fn,str(OBJ)]);(O/name).write_bytes(raw);hostfiles[fn]={'path':name,'bytes':len(raw),'sha256':sha(raw)}
 get=lambda buf,tab,k:buf[tab[k][4]:tab[k][4]+tab[k][5]]if tab[k][1]!=8 else b''
 def extra_inventory(image,ss):
  # Decode actual symbols/REL(A) records as well as retaining raw section hashes.
  require(len(ss)==struct.unpack_from('<H',image,60)[0],'unique ELF section names before numerical section-index mapping')
  syms={}
  for key,sec in ss.items():
   if sec[1]!=2:continue
   table=list(ss.values());st=table[sec[6]];strings=image[st[4]:st[4]+st[5]];rows=[]
   for at in range(sec[4],sec[4]+sec[5],sec[9]):
    no,info,other,idx,value,size=struct.unpack_from('<IBBHQQ',image,at);rows.append({'name':cstr(strings,no).decode(),'type':info&15,'bind':info>>4,'section':idx,'value':value,'size':size})
   syms[list(ss).index(key)]=rows
  relocs=[];table=list(ss.values());names=list(ss)
  for key,sec in ss.items():
   if sec[1]not in[4,9]:continue
   for at in range(sec[4],sec[4]+sec[5],sec[9]):
    pos,ri=struct.unpack_from('<QQ',image,at)
    relocs.append({'section':key,'destination_section':names[sec[7]],'offset':pos,'type':ri&0xffffffff,'symbol':syms[sec[6]][ri>>32],'explicit_addend':struct.unpack_from('<q',image,at+16)[0]if sec[1]==4 else None})
  return {'data_sections':{k:{'type':s[1],'flags':s[2],'bytes':s[5],'alignment':s[8],'sha256':sha(get(image,ss,k))}for k,s in ss.items()if k.startswith(('.data','.bss','.rodata'))},'relocations':relocs,'relocation_type_scope':'Raw numeric type and actual symbol/offset only, no unverified type-name semantics.'}
 selected={};raw_images={}
 for arm,host in hosts.items():
  name=host['registered_name_pointer']['name'];unique={}
  for m in re.compile(rb'(?m)^  - name:[ \t]+'+re.escape(name.encode())+rb'[ \t]*\r?$').finditer(final.data):
   at=m.start();base=final.data.rfind(b'\x7fELF\x02\x01',max(0,at-4*1024*1024),at);require(base>=0,'selected ELF base')
   ss,size=sections(final.data,base);z=ss['.ze_info'];require(base+z[4]<=at<base+z[4]+z[5],'registered name in actual selected .ze_info');unique[base]=(ss,size)
  require(len(unique)==1,'unique selected image '+arm);base,(ss,size)=next(iter(unique.items()));image=bytes(final.data[base:base+size]);raw_images[arm]=image
  ze=get(image,ss,'.ze_info').decode('ascii');names=re.findall(r'^  - name:\s+(\S+)',ze.split('kernels_misc_info:')[0],re.M);require(names==[name],'one selected kernel/image '+arm)
  block=next(x for x in re.split(r'(?m)^  - name:\s+',ze.split('kernels_misc_info:')[0])[1:]if x.splitlines()[0].strip()==name)
  fields={k:re.findall(r'(?m)^\s*'+re.escape(k)+r':\s*(.+)$',block)for k in ['grf_count','simd_size','slm_size','barrier_count','spill_size','scratch_size','eu_thread_count','subgroup_independent_forward_progress']}
  options=get(image,ss,'.misc.buildOptions');require(options.rstrip(b'\0')==b'-cl-fp32-correctly-rounded-divide-sqrt','precise actual options '+arm)
  text=get(image,ss,'.text.'+name);(O/(arm+'.zebin')).write_bytes(image);(O/(arm+'.ze_info')).write_text(ze,encoding='ascii')
  selected[arm]={'kernel':name,'executable_offset':base,'image_bytes':size,'image_sha256':sha(image),'text_bytes':len(text),'text_sha256':sha(text),'execution_fields':fields,'missing_fields':'absent/UNKNOWN, not measuredzero; barrier_count is not static instruction count','full_kernel_metadata':block,'options_ascii':options.rstrip(b'\0').decode('ascii'),'inventory':{k:{'type':s[1],'flags':s[2],'bytes':s[5],'sha256':sha(get(image,ss,k))}for k,s in ss.items()},'data_and_relocations':extra_inventory(image,ss)}
 require(len({x['executable_offset']for x in selected.values()})==6,'six distinct selected images')
 selected_stats={str(p):stat(p)for p in expected_artifacts};require(before==selected_stats,'private artifact stat changed during selected extraction')
 save('selection.json',{'artifact_stat_before':before,'artifact_stat_after':selected_stats,'status':'SIX_ACTUAL_HOST_IMAGE_BINDINGS_CONTROL_COMPARISON_PENDING','hosts':hosts,'host_function_files':hostfiles,'selected':selected,'source_manifest_sha256':sha(manifest_raw),'private_artifact_hashes':{'exe':EXPECTED,'object':a.expected_object_sha256,'images':a.expected_images_sha256},'GPU':0,'decoder_invocations':0})
 baseline_dir=R/'hc-selected-old-s115-audit/selected';comparisons={}
 for arm,baseline_arm in [('control_up','up'),('control_reduce','reduce')]:
  old=(baseline_dir/(baseline_arm+'.zebin')).read_bytes();ref=pins['baseline'][baseline_arm];require(sha(old)==ref['image_sha256'],'frozen S115 selected reference '+baseline_arm)
  os,_=sections(old);oldze=get(old,os,'.ze_info').decode('ascii');oldname=re.findall(r'^  - name:\s+(\S+)',oldze.split('kernels_misc_info:')[0],re.M);require(len(oldname)==1,'single original kernel');oldname=oldname[0]
  image=raw_images[arm];ss,_=sections(image);name=selected[arm]['kernel'];text=get(image,ss,'.text.'+name);oldtext=get(old,os,'.text.'+oldname);eot=ref['computational_EOT_end']
  require(len(oldtext)==ref['text_bytes']and sha(oldtext)==ref['text_sha256'],'S115 selected complete original text')
  norm=lambda k,target:k.replace(target,'TARGET')
  metadata=get(image,ss,'.ze_info').decode().split('kernels_misc_info:')[0].replace(name,'TARGET')==oldze.split('kernels_misc_info:')[0].replace(oldname,'TARGET')
  constants={k for tab in[ss,os]for k in tab if k.startswith(('.data','.bss','.rodata'))}
  data=all(k in ss and k in os and all(ss[k][i]==os[k][i]for i in[1,2,5,8])and get(image,ss,k)==get(old,os,k)for k in constants)
  relnew={norm(k,name):get(image,ss,k)for k in ss if k.startswith(('.rel.','.rela.'))};relold={norm(k,oldname):get(old,os,k)for k in os if k.startswith(('.rel.','.rela.'))}
  prefix=len(text)>=eot and text[:eot]==oldtext[:eot]
  checks={'computational_prefix_exact':prefix,'normalized_execution_argument_metadata_exact':metadata,'options_exact':get(image,ss,'.misc.buildOptions')==get(old,os,'.misc.buildOptions'),'constant_data_exact':data,'raw_normalized_relocation_sections_exact':relnew==relold,'decoded_normalized_relocations_exact':json.dumps(selected[arm]['data_and_relocations']['relocations'],sort_keys=True).replace(name,'TARGET')==json.dumps(extra_inventory(old,os)['relocations'],sort_keys=True).replace(oldname,'TARGET')}
  comparisons[arm]={'checks':checks,'PASS':all(checks.values()),'computational_prefix_bytes':eot,'raw_full_image_exact':image==old,'raw_complete_text_exact':text==oldtext,'new_text_bytes':len(text),'original_text_bytes':len(oldtext),'first_changed_offset_common_length':next((i for i,(a,b)in enumerate(zip(text,oldtext))if a!=b),None),'reference_ASM':str(baseline_dir/f'{baseline_arm}-isa/disasm/.text.hc_{baseline_arm}.asm'),'reference_ASM_sha256_reused':ref['ASM_sha256'],'reference_scope':'Exact S115 selected computational body/resources/options/constants/relocations; raw trailing image/text equality is separately reported. Norm/down retained exact source, no new original-object/archive audit.'}
 save('control-comparison.json',comparisons)
 require(all(x['PASS']for x in comparisons.values()),'original control compiled mismatch: root inspects retained facts; no automatic decode or gate relaxation')
 # Alias only candidate text SECTION NAME on copies. All executable/data/reloc bytes exact.
 aliases={}
 for arm in ['candidate_reduce','candidate_up']:
  image=raw_images[arm];ss,_=sections(image);name=selected[arm]['kernel'];t=ss['.text.'+name];alias=bytearray(image)
  so=struct.unpack_from('<Q',image,40)[0];se,sn,si=struct.unpack_from('<HHH',image,58);st=struct.unpack_from('<IIQQQQIIQQ',image,so+si*se);at=st[4]+t[0];old=cstr(alias,at);new=('.text.s116_'+arm).encode();require(old==('.text.'+name).encode()and len(new)<len(old),'short alias fits')
  alias[at:at+len(old)]=new+b'\0'*(len(old)-len(new));changed=[i for i,(a,b)in enumerate(zip(image,alias))if a!=b]
  require(all(at<=i<at+len(old)for i in changed),'alias only selected section-name bytes');require(bytes(alias[t[4]:t[4]+t[5]])==get(image,ss,'.text.'+name),'candidate text byte exact under alias')
  (O/(arm+'-shortname.zebin')).write_bytes(alias);aliases[arm]={'sha256':sha(alias),'section_name':new.decode(),'text_byte_exact':True,'only_section_name_changed':True,'changed_bytes':len(changed)}
 after={str(p):stat(p)for p in expected_artifacts};require(before==after,'private artifact stat changed during audit')
 save('facts.json',{'status':'PASS_SIX_STATIC_BINDINGS_STRICT_ORIGINAL_CONTROL_REUSE_CANDIDATE_MANUAL_ISA_PENDING','hosts':hosts,'host_function_files':hostfiles,'selected':selected,'original_control_comparison':comparisons,'candidate_aliases':aliases,'artifact_stat_before':before,'artifact_stat_after':after,'source_manifest_sha256':sha(manifest_raw),'exe_sha256':EXPECTED,'object_sha256':a.expected_object_sha256,'private_images_sha256':a.expected_images_sha256,'GPU':0,'decoder_invocations':0,'full_main_archive_or_model_reads':False,'limits':['Static host/image binding is not GPU dispatch or numerical proof.','Norm/down are exact source projections only; no redundant original-object/archive sweep.','Candidate raw-store operand/extra+0/div/nativeexp/finaldiv/CR0 and unchanged UP downstream order require actual selected ISA and root review before GPU.','Missing fields UNKNOWN; static barrier instructions must not be inferred from barrier_count.','Unexpected original-control mismatch HOLD; no automatic weakened equivalence or decode retry.']})
 print('PASS_SIX_BINDINGS_STRICT_CONTROL_REUSE_CANDIDATE_ISA_REVIEW_REQUIRED')
except BaseException as e:
 save('HOLD.json',{'status':'HOLD_STOP_FIRST_AUDIT_FAILURE','type':type(e).__name__,'error':str(e),'GPU':0,'decoder_invocations':0});raise
