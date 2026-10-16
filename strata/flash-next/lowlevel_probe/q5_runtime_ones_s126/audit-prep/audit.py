"""S126 selected private/public CPU audit. ROOT executes after build; no decoder/GPU."""
from pathlib import Path
import argparse,ast,hashlib,json,mmap,re,struct,subprocess
P=Path(__file__).resolve().parent;R=P.parent
ap=argparse.ArgumentParser();ap.add_argument('--prep',type=Path,required=True);ap.add_argument('--expected-manifest-sha256',required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
S=a.prep.resolve();B=S/'build';O=a.out.resolve();sha=lambda b:hashlib.sha256(b).hexdigest()
def require(ok,msg):
 if not ok:raise RuntimeError('HOLD: '+msg)
def save(n,x):(O/n).write_text(json.dumps(x,indent=2,ensure_ascii=True)+'\n',encoding='ascii')
def stat(p):
 z=p.stat();return dict(device=z.st_dev,inode=z.st_ino,size=z.st_size,mtime_ns=z.st_mtime_ns,ctime_ns=z.st_ctime_ns)
require(S.parent==R and S.name.startswith('q5-c2-runtime-ones-s126-prep'),'explicit S126 source path')
require(re.fullmatch('[0-9a-f]{64}',a.expected_manifest_sha256),'explicit root-frozen source manifest SHA')
require(not O.exists(),'preserve old output/no automatic retry');O.mkdir()
try:
 pins=json.loads((P/'pins.json').read_text())
 for x in pins['small_files']:
  raw=Path(x['path']).read_bytes();require(len(raw)==x['bytes']and sha(raw)==x['sha256'],'small helper/reference '+x['path'])
 mr=(R/(S.name+'-manifest.json')).read_bytes();require(sha(mr)==a.expected_manifest_sha256,'root-frozen source manifest');manifest=json.loads(mr)
 for n,v in manifest['files'].items():
  require(Path(n).name==n,'source basename only');raw=(S/n).read_bytes();require(len(raw)==v['bytes']and sha(raw)==v['sha256'],'source manifest '+n)
 require(sha((S/'private_q5_runtime_ones.hpp').read_bytes())==pins['header_sha256'],'unchanged exact kernel header')
 for n in ['build.sh','build-container.sh']:
  require(sha((S/n).read_bytes())==pins['build_scripts'][n],'unchanged reviewed build flags/link '+n)
 ledger=json.loads((S/'source-ledger.json').read_text());require(all(ledger[k]for k in ['control_kernel_inverse_name_equals_native','candidate_inverse_runtime_apply_then_kernel_equals_native','runtime_apply_inverse_equals_native']),'source inverse projections')
 require((B/'build.exit').read_text().strip()=='0'and(B/'tee.exit').read_text().strip()=='0','actual build/tee terminal0')
 require(json.loads((B/'image.json').read_text())[0]['Id']==pins['image'],'actual pinned build image')
 require('2026.1'in(B/'compiler-version.txt').read_text(),'actual DPC++2026.1')
 records={line.split(maxsplit=1)[1]:line.split(maxsplit=1)[0]for line in(B/'sha256.txt').read_text().splitlines()}
 for n in ['private_q5_runtime_ones.hpp','q5_runtime_ones_leaf.cpp']:require(records['/src/'+n]==sha((S/n).read_bytes()),'build exact source '+n)
 buildlog=(B/'build.log').read_text()
 for line in(S/'expected-container.sha256').read_text().splitlines():
  _,n=line.split(maxsplit=1);require(buildlog.count(n+': OK')==2,'actual pre/post build input '+n)
 link=(B/'host-link.log').read_text();require(' -fsycl-link 'not in link and 'ocloc 'not in link,'host link does not AOT')
 for n,v in pins['frozen_link_inputs'].items():
  require(records[n]==v['sha256']and stat(Path(v['path']))==v['stat'],'actual build identity/stat continuity '+n);require(n in link,'actual host link input '+n)
 EXE=B/'q5_runtime_ones_s126';OBJ=B/'q5_runtime_ones_leaf.o';IMAGES=B/'leaf.images.o'
 private={p:records[p.name]for p in[EXE,OBJ,IMAGES]};require({p.name:h for p,h in private.items()}==pins['expected_built_artifacts'],'root-provided exact V3 build hashes');before={str(p):stat(p)for p in private}
 for p,h in private.items():
  with p.open('rb')as f:require(hashlib.file_digest(f,'sha256').hexdigest()==h,'actual private artifact '+p.name)
 # ONE old public host member from the linked frozen S59 archive, never S107's TU.
 ARCH=Path(pins['frozen_link_inputs']['/frozen/libstrata_kernels.a']['path']);archive_before=stat(ARCH)
 toc=subprocess.check_output(['ar','t',str(ARCH)],text=True).splitlines();members=[n for n in toc if Path(n).name=='native_mmvq.dp.cpp.o'];require(len(members)==1,'one exact native MMVQ archive member')
 PUBLIC=O/'public-native_mmvq.dp.cpp.o';member=subprocess.check_output(['ar','p',str(ARCH),members[0]]);PUBLIC.write_bytes(member)
 require(archive_before==stat(ARCH),'archive continuity around one-member extraction')
 # Reuse audited bounded ELF class; no helper top-level execution/all-image walk.
 helper=Path(pins['elf_helper']);node=next(n for n in ast.parse(helper.read_text()).body if isinstance(n,ast.ClassDef)and n.name=='ELF')
 def cstr(buf,start):
  end=buf.find(b'\0',start);require(end>=start,'bounded C string');return buf[start:end]
 class Bound(ast.NodeTransformer):
  replacements=0
  def visit_Subscript(self,n):
   n=self.generic_visit(n);c=n.value
   if isinstance(n.slice,ast.Constant)and n.slice.value==0 and isinstance(c,ast.Call)and isinstance(c.func,ast.Attribute)and c.func.attr=='split'and isinstance(c.func.value,ast.Subscript):
    s=c.func.value
    if ((isinstance(s.value,ast.Name)and s.value.id=='strings')or(isinstance(s.value,ast.Attribute)and s.value.attr=='data'and isinstance(s.value.value,ast.Name)and s.value.value.id=='self'))and isinstance(s.slice,ast.Slice)and s.slice.upper is None:
     self.replacements+=1;return ast.copy_location(ast.Call(func=ast.Name(id='cstr',ctx=ast.Load()),args=[s.value,s.slice.lower],keywords=[]),n)
   return n
 t=Bound();node=t.visit(node);require(t.replacements==4,'four existing bounded-string adaptations');ast.fix_missing_locations(node)
 known={str(p):h for p,h in private.items()};known[str(PUBLIC)]=sha(member)
 code=ast.unparse(node).replace('self.data = path.read_bytes()',"self._file=path.open('rb'); self.data=mmap.mmap(self._file.fileno(),0,access=mmap.ACCESS_READ)").replace('self.sha = hashlib.sha256(self.data).hexdigest()','self.sha=known[str(path)]')
 exec(compile(code,str(helper)+':bounded-class-only','exec'),globals())
 prior=Path(pins['binding_helper']);nodes=[n for n in ast.parse(prior.read_text()).body if isinstance(n,ast.FunctionDef)and n.name in{'refs_to','bind','sections'}];require(len(nodes)==3,'three existing binding functions');exec(compile(ast.Module(body=nodes,type_ignores=[]),str(prior)+':three-functions-only','exec'),globals())
 prior=Path(pins['inventory_helper']);nodes=[n for n in ast.walk(ast.parse(prior.read_text()))if isinstance(n,ast.FunctionDef)and n.name=='extra_inventory'];require(len(nodes)==1,'one existing inventory function');exec(compile(ast.Module(body=nodes,type_ignores=[]),str(prior)+':inventory-only','exec'),globals())
 final=ELF(EXE);leaf=ELF(OBJ);pub=ELF(PUBLIC)
 def demangle(names):
  require(bool(names),'nonempty actual symbol list');out=subprocess.check_output(['c++filt'],input=('\n'.join(names)+'\n').encode()).decode().splitlines();require(len(out)==len(names),'demangle line count');return dict(zip(names,out))
 def metadata(obj):
  names=[z['name']for z in obj.symbols if z['section']and z['type']==1 and('CompileTimeKernelInfoI'in z['name']or'getDeviceKernelInfo'in z['name'])];return demangle(names)
 def object_registered_name(obj,info):
  found=[]
  for sec in obj.ss:
   if sec[1]!=4 or sec[7]!=info['section']:continue
   for at in range(sec[4],sec[4]+sec[5],sec[9]):
    pos,ri,add=struct.unpack_from('<QQq',obj.data,at)
    if pos==info['offset']:
     z=obj.symbols[ri>>32];found.append(cstr(obj.data,obj.ss[z['section']][4]+z['offset']+add).decode())
  require(len(found)==1,'one actual Info/name relocation');return found[0]
 def observed_bind(obj,ds,info):
  # Cache/guard use the ACTUAL Info template type, not a guessed registered-name spelling.
  marker='CompileTimeKernelInfo<';text=ds[info['name']];require(marker in text and text.endswith('>'),'actual demangled Info type');typ=text.split(marker,1)[1][:-1]
  related=[z for z in obj.symbols if z['section']and z['type']==1 and'getDeviceKernelInfo'in z['name']and('getDeviceKernelInfo<'+typ+'>')in ds.get(z['name'],'')];require(len(related)==2,'two observed cache/guard symbols')
  exact={info['name']}|{z['name']for z in related};bound=bind(obj,lambda n:n in exact)
  return bound,{'info_type':typ,'actual_symbols':{n:ds[n]for n in sorted(exact)}}
 dleaf=metadata(leaf);dpub=metadata(pub);hosts={};discovery={};owners={}
 specs={'original':'s126::OriginalQ5C2','candidate':'s126::RuntimeOnesQ5C2','raw16':'s126::Raw16Sums','raw16_wrapper':'sycl::_V1::detail::__pf_kernel_wrapper<s126::Raw16Sums>'}
 normtype=lambda t:re.sub(r'\s+>', '>',t).strip()
 for arm,typ in specs.items():
  infos=[z for z in leaf.symbols if z['section']and'CompileTimeKernelInfoI'in z['name']and normtype(dleaf.get(z['name'],'').split('CompileTimeKernelInfo<',1)[-1][:-1])==typ];require(len(infos)==1,'one actual private class/form '+typ)
  hosts[arm],discovery[arm]=observed_bind(leaf,dleaf,infos[0]);name=hosts[arm]['registered_name_pointer']['name'];require(normtype(demangle([name])[name])=='typeinfo name for '+typ,'exact actual registered private class/form '+typ);owners[arm]=leaf
 oldfacts=json.loads(Path(pins['baseline_facts']).read_text());oldname=oldfacts['selected']['kernel']
 candidates=[z for z in pub.symbols if z['section']and'CompileTimeKernelInfoI'in z['name']and'WideQ5K'in z['name']and'native_mmvq_wide'in z['name']]
 infos=[z for z in candidates if object_registered_name(pub,z)==oldname];require(len(infos)==1,'exact retained public Q5C2 registered name')
 hosts['public'],discovery['public']=observed_bind(pub,dpub,infos[0]);owners['public']=pub
 require(len({v['final_info']['offset']for v in hosts.values()})==5,'five distinct LIVE linked Info')
 require(len({v['registered_name_pointer']['pointer']for v in hosts.values()})==5,'five distinct linked names')
 require(len({z['offset']for v in hosts.values()for z in v['final_cache_guard']})==10,'ten distinct linked cache/guard cells')
 live=[z['name']for z in leaf.symbols if z['section']and'CompileTimeKernelInfoI'in z['name']and's126::'in dleaf.get(z['name'],'')];require(set(live)=={hosts[k]['object_info']['name']for k in specs},'exact four private materialized forms (Raw16 direct/wrapper)')
 save('symbol-discovery.json',discovery)
 # Actual selected Info users, main's argv parse, public dispatcher/try_wide branches.
 functions={str(OBJ):set(),str(PUBLIC):set()};objects={str(OBJ):leaf,str(PUBLIC):pub}
 for arm,h in hosts.items():
  obj=owners[arm];functions[str(obj.path)].update(fn for row in h['actual_host_relocations']if row['target_metadata']==h['object_info']['name']for fn in row['containing_functions'])
 require(all(functions.values()),'actual private/public host users')
 require(any(z['name']=='main'and z['section']and z['type']==2 for z in leaf.symbols),'actual main');functions[str(OBJ)].add('main')
 for api in ['native_mmvq','native_q5_k_mmvq']:
  xs=[z['name']for z in pub.symbols if z['section']and z['type']==2 and z['name'].startswith('_ZN6strata7kernels'+str(len(api))+api+'E')];require(len(xs)==1,'actual public API '+api);functions[str(PUBLIC)].update(xs)
 xs=[z['name']for z in pub.symbols if z['section']and z['type']==2 and'try_wide'in z['name']and'WideQ5K'in z['name']];require(len(xs)==1,'actual Q5 try_wide');functions[str(PUBLIC)].update(xs)
 hostfiles={};edges={}
 for path,names in functions.items():
  obj=objects[path];edges[path]=obj.relocs(set(names)|{z['name']for z in leaf.symbols if'strtoul'in z['name']or'runtime_ones'in z['name']})
  for fn in sorted(names):
   raw=subprocess.check_output(['objdump','-drw','--disassemble='+fn,path]);require(re.search(rb'(?m)^0+[0-9a-f]+ <'+re.escape(fn.encode())+rb'>:',raw)is not None,'actual selected function body');n='host-'+str(len(hostfiles))+'.asm';(O/n).write_bytes(raw);hostfiles[path+'::'+fn]={'path':n,'bytes':len(raw),'sha256':sha(raw),'symbol':obj.symbol(fn),'linked_symbol':final.symbol(fn)if fn in{z['name']for z in final.symbols}else None}
 get=lambda buf,tab,k:buf[tab[k][4]:tab[k][4]+tab[k][5]]if tab[k][1]!=8 else b''
 def user_args(block):
  require('    payload_arguments:\n'in block,'actual payload metadata');part=block.split('    payload_arguments:\n',1)[1].split('\n    per_thread_payload_arguments:',1)[0];out=[]
  for piece in re.split(r'(?m)^      - arg_type:\s*',part)[1:]:
   row={'arg_type':piece.splitlines()[0].strip()}
   for k in ['arg_index','offset','size','addrmode','addrspace','access_type','source_offset','is_ptr']:
    m=re.search(r'(?m)^\s*'+k+r':\s*(\S+)',piece)
    if m:row[k]=int(m[1])if k in['arg_index','offset','size','source_offset']else m[1]
   if'arg_index'in row:out.append(row)
  return sorted(out,key=lambda x:x['arg_index'])
 selected={};images={}
 for arm,h in hosts.items():
  name=h['registered_name_pointer']['name'];found={}
  for m in re.compile(rb'(?m)^  - name:[ \t]+'+re.escape(name.encode())+rb'[ \t]*\r?$').finditer(final.data):
   at=m.start();base=final.data.rfind(b'\x7fELF\x02\x01',max(0,at-4*1024*1024),at);require(base>=0,'selected ELF base');ss,size=sections(final.data,base);z=ss['.ze_info'];require(base+z[4]<=at<base+z[4]+z[5],'name in selected metadata');found[base]=(ss,size)
  require(len(found)==1,'one exact selected image '+arm);base,(ss,size)=next(iter(found.items()));image=bytes(final.data[base:base+size]);images[arm]=image;ze=get(image,ss,'.ze_info').decode('ascii')
  require(re.findall(r'^  - name:\s+(\S+)',ze.split('kernels_misc_info:')[0],re.M)==[name],'one selected kernel/image');block=next(x for x in re.split(r'(?m)^  - name:\s+',ze.split('kernels_misc_info:')[0])[1:]if x.splitlines()[0].strip()==name)
  fields={k:re.findall(r'(?m)^\s*'+k+r':\s*(.+)$',block)for k in['grf_count','simd_size','slm_size','barrier_count','spill_size','scratch_size','eu_thread_count','subgroup_independent_forward_progress']};u=user_args(block)
  if arm in['public','original','candidate']:
   count=6 if arm=='candidate'else 5;require(len(u)==count and[x['arg_index']for x in u]==list(range(count)),'actual projection argument count/index '+arm)
   require([(x['arg_type'],x['offset'],x['size'])for x in u]==[('arg_bypointer',0,8),('arg_bypointer',8,8),('arg_bypointer',16,8),('arg_byvalue',24,4),('arg_byvalue',28,4)]+([('arg_byvalue',32,4)]if arm=='candidate'else[]),'projection capture ABI '+arm)
  elif arm=='raw16':require(len(u)==3 and sorted((x['arg_type'],x['size'])for x in u)==[('arg_bypointer',8),('arg_bypointer',8),('arg_byvalue',4)],'direct raw16 captures two pointers/ones only; range outside lambda')
  else:
   # Actual SDK wrapper: range8B +closure24B; closure ones/padding is split.
   require([(x['arg_type'],x['offset'],x['size'],x['arg_index'],x.get('source_offset'),x.get('is_ptr'))for x in u]==[('arg_byvalue',24,8,0,0,None),('arg_byvalue',32,5,1,0,None),('arg_byvalue',40,1,1,5,None),('arg_byvalue',44,1,1,6,None),('arg_byvalue',48,1,1,7,None),('arg_byvalue',56,8,1,8,'true'),('arg_byvalue',64,8,1,16,'true')],'exact actual SDK raw16 wrapper aggregate ABI')
  opt=get(image,ss,'.misc.buildOptions');require(opt.rstrip(b'\0')==b'-cl-fp32-correctly-rounded-divide-sqrt','actual precise options '+arm);text=get(image,ss,'.text.'+name)
  (O/(arm+'.zebin')).write_bytes(image);(O/(arm+'.ze_info')).write_text(ze,encoding='ascii')
  selected[arm]={'kernel':name,'executable_offset':base,'image_bytes':len(image),'image_sha256':sha(image),'text_bytes':len(text),'text_sha256':sha(text),'execution_fields':fields,'missing_metadata':'UNKNOWN, not zero; barrier_count is not an instruction count','payload_user_arguments':u,'full_metadata':block,'options_ascii':opt.rstrip(b'\0').decode(),'data_and_relocations':extra_inventory(image,ss)}
 require(len({x['executable_offset']for x in selected.values()})==5,'five distinct selected images')
 save('selection.json',{'status':'FIVE_ACTUAL_BINDINGS_CONTROL_COMPARISON_PENDING','hosts':hosts,'selected':selected,'host_function_files':hostfiles,'host_edges':edges})
 old=Path(pins['baseline_image']).read_bytes();require(sha(old)==oldfacts['selected']['image_sha256'],'exact S122 selected baseline image');os,_=sections(old);oldtext=get(old,os,'.text.'+oldname);require(len(oldtext)==8320 and sha(oldtext)==oldfacts['selected']['text_sha256'],'exact old complete text');eot=8128;comparisons={}
 for arm in['public','original']:
  image=images[arm];ss,_=sections(image);name=selected[arm]['kernel'];text=get(image,ss,'.text.'+name)
  constants={k for tab in[ss,os]for k in tab if k.startswith(('.data','.bss','.rodata'))};data=all(k in ss and k in os and all(ss[k][i]==os[k][i]for i in[1,2,5,8])and get(image,ss,k)==get(old,os,k)for k in constants)
  rel=lambda buf,tab,n:{k.replace(n,'TARGET'):get(buf,tab,k)for k in tab if k.startswith(('.rel.','.rela.'))}
  checks={'computational_prefix_8128_exact':len(text)>=eot and text[:eot]==oldtext[:eot],'normalized_metadata_exact':get(image,ss,'.ze_info').decode().split('kernels_misc_info:')[0].replace(name,'TARGET')==get(old,os,'.ze_info').decode().split('kernels_misc_info:')[0].replace(oldname,'TARGET'),'options_exact':get(image,ss,'.misc.buildOptions')==get(old,os,'.misc.buildOptions'),'constants_exact':data,'raw_normalized_relocations_exact':rel(image,ss,name)==rel(old,os,oldname),'decoded_normalized_relocations_exact':json.dumps(extra_inventory(image,ss)['relocations'],sort_keys=True).replace(name,'TARGET')==json.dumps(extra_inventory(old,os)['relocations'],sort_keys=True).replace(oldname,'TARGET')}
  comparisons[arm]={'PASS':all(checks.values()),'checks':checks,'raw_full_image_exact':image==old,'raw_complete_text_exact':text==oldtext,'first_changed_text_byte':next((i for i,(x,y)in enumerate(zip(text,oldtext))if x!=y),None),'new_text_bytes':len(text),'old_text_bytes':len(oldtext),'computational_EOT_end':eot,'reference_ASM_sha256':pins['baseline_ASM_sha256'],'limits':'Prefix equality includes through EOT; full image/text equality reported separately, no trailer relaxation of prefix/metadata/data/relocation gates.'}
 save('control-comparison.json',comparisons);require(all(x['PASS']for x in comparisons.values()),'public/private original drift; inspect facts and STOP, no automatic decoder/retry')
 image=images['candidate'];ss,_=sections(image);name=selected['candidate']['kernel'];t=ss['.text.'+name];alias=bytearray(image);so=struct.unpack_from('<Q',image,40)[0];se,sn,si=struct.unpack_from('<HHH',image,58);st=struct.unpack_from('<IIQQQQIIQQ',image,so+si*se);at=st[4]+t[0];old_section=cstr(alias,at);new=b'.text.s126_candidate';require(old_section==('.text.'+name).encode()and len(new)<len(old_section),'candidate alias fits')
 alias[at:at+len(old_section)]=new+b'\0'*(len(old_section)-len(new));changed=[i for i,(x,y)in enumerate(zip(image,alias))if x!=y];require(all(at<=i<at+len(old_section)for i in changed),'section-name-only alias');require(bytes(alias[t[4]:t[4]+t[5]])==get(image,ss,'.text.'+name),'alias executable bytes exact');(O/'candidate-shortname.zebin').write_bytes(alias)
 after={str(p):stat(p)for p in private};require(before==after,'private artifact stat continuity')
 for v in pins['frozen_link_inputs'].values():require(stat(Path(v['path']))==v['stat'],'frozen input end stat continuity')
 save('facts.json',{'status':'PASS_FIVE_STATIC_BINDINGS_STRICT_PUBLIC_CONTROL_REUSE_CANDIDATE_MANUAL_ISA_PENDING','source_prep':str(S),'source_manifest_sha256':sha(mr),'private_artifact_hashes':{p.name:h for p,h in private.items()},'artifact_stat_before':before,'artifact_stat_after':after,'public_member':{'archive':str(ARCH),'archive_sha256_reused':pins['frozen_link_inputs']['/frozen/libstrata_kernels.a']['sha256'],'archive_stat':archive_before,'member':members[0],'object_sha256':sha(member),'object_bytes':len(member),'substituted_S107_object':False},'hosts':hosts,'host_function_files':hostfiles,'host_edges':edges,'selected':selected,'original_comparison':comparisons,'candidate_alias':{'sha256':sha(alias),'only_section_name_changed':True,'text_data_relocations_unchanged':True,'changed_bytes':len(changed)},'manual_host_obligation':'Read actual main argv4/strtoul/validation/runtime_ones assignment and selected private call/capture/submit branch including scalar offset32. A 4B arg does not itself prove signed value or runtime use. Read public type13/try_wide/C2/RPW1 branch. Actual pointers/relocations are static binding, not observed GPU dispatch.','manual_ISA_obligation':'Candidate scalar origin, packed correction dots, removal of four expanded chains, weighted operand/FP MAD/conversion/accumulation/XOR/store ledger, output write scope, CR0 and actual resources before any GPU. No source-only universal equivalence. Raw16 validation payload/source separate from timed projection.','ISA':'UNKNOWN until one separate root-selected candidate decode/read, never automatic','all_image_walk':False,'main_archive_model_fresh_hash':False,'GPU_build_decoder_invocations':0})
 print('PASS_FIVE_BINDINGS_STRICT_PUBLIC_CONTROL_REUSE_CANDIDATE_MANUAL_ISA_PENDING')
except BaseException as e:
 save('HOLD.json',{'status':'HOLD_FIRST_AUDIT_FAILURE_STOP_NO_RETRY','type':type(e).__name__,'error':str(e),'GPU_build_decoder_invocations':0});raise
