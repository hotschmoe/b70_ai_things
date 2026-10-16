"""S122 ONE existing Q5 C2 image; CPU-only when root runs it. No decoder/GPU."""
from pathlib import Path
import argparse,ast,hashlib,json,mmap,re,struct,subprocess
P=Path(__file__).resolve().parent;R=P.parent;SRC=Path('/mnt/vm_8tb/b70/build/strata-ab/src-ep')
a=argparse.ArgumentParser();a.add_argument('out',type=Path);args=a.parse_args();O=args.out.resolve()
assert not O.exists(),'Preserve prior result; no automatic retry';O.mkdir()
sha=lambda b:hashlib.sha256(b).hexdigest()
def save(n,v):(O/n).write_text(json.dumps(v,indent=2,ensure_ascii=True)+'\n',encoding='ascii')
def stat(p):
 s=p.stat();return dict(device=s.st_dev,inode=s.st_ino,size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
def require(b,m):
 if not b:raise AssertionError(m)
try:
 pins=json.loads((P/'pins.json').read_text())
 for row in pins['small_inputs']:
  b=Path(row['path']).read_bytes();require(len(b)==row['bytes']and sha(b)==row['sha256'],'small input identity '+row['path'])
 EXE=Path(pins['artifacts']['main']['path']);OBJ=Path(pins['artifacts']['object']['path']);ARCHIVE=Path(pins['artifacts']['archive']['path']);EXPECTED=pins['artifacts']['main']['sha256']
 before={str(Path(v['path'])):stat(Path(v['path']))for v in pins['artifacts'].values()}
 for v in pins['artifacts'].values():require(before[v['path']]==v['stat'],'frozen artifact stat '+v['path'])
 old=json.loads((R/'hc-selected-old-s115-audit/selected/facts.json').read_text())
 require(old['main']['stat_after']==before[str(EXE)]and old['archive']['stat_after']==before[str(ARCHIVE)],'reuse closed S115 main/archive continuity')
 build=json.loads((R/'q6-a4-main-s107-build/build/root-build-audit.json').read_text());require(build['actual_exit']==build['tee_exit']==0,'successful exact S107 build')
 keys={'main':'/out/strata-q6-a4-s107','object':'CMakeFiles/strata_kernels.dir/src/kernels/cuda/native_mmvq.dp.cpp.o','archive':'libstrata_kernels.a'}
 for k,v in keys.items():require(build['build_sha256'][v]==pins['artifacts'][k]['sha256'],'frozen build identity '+k)
 require(build['source_sha256']==pins['native_source_sha256'],'exact native source binding')
 # ONE selected object hash, not another 92 MB main / whole archive hash.
 with OBJ.open('rb')as f:require(hashlib.file_digest(f,'sha256').hexdigest()==pins['artifacts']['object']['sha256'],'selected native object hash')
 cfg=json.loads((R/'q6-a4-main-s107-prep/mode0/config.json').read_text());require(cfg['exe']==str(EXE),'retained configured binary')
 for k,v in {'STRATA_HEAD_Q5K_RPW2':'0','STRATA_MMVQ_WIDE_K':'1','STRATA_MMVQ_WIDE_K_RPW':'1','STRATA_BATCH_MTP_PAIR_HEAD':'1'}.items():require(cfg['env'][k]==v,'retained source-route config '+k)
 # Same bounded class-only mmap adaptation used by S107, without helper top-level execution.
 helper=R/'attn-value-binding-s92-build-audit/host_binding.py'
 node=next(n for n in ast.parse(helper.read_text()).body if isinstance(n,ast.ClassDef)and n.name=='ELF')
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
 transform=Bound();node=transform.visit(node);require(transform.replacements==4,'four audited bounded-string replacements');ast.fix_missing_locations(node)
 known={str(EXE):EXPECTED,str(OBJ):pins['artifacts']['object']['sha256']}
 code=ast.unparse(node).replace('self.data = path.read_bytes()',"self._file=path.open('rb'); self.data=mmap.mmap(self._file.fileno(),0,access=mmap.ACCESS_READ)").replace('self.sha = hashlib.sha256(self.data).hexdigest()','self.sha=known[str(path)]')
 exec(compile(code,str(helper)+':bounded-class-only','exec'),globals())
 prior=R/'q6-a4-main-s107-audit-prep/audit.py'
 nodes=[n for n in ast.parse(prior.read_text()).body if isinstance(n,ast.FunctionDef)and n.name in {'refs_to','bind','sections'}];require(len(nodes)==3,'three existing S107 functions')
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(prior)+':three-functions-only','exec'),globals())
 final=ELF(EXE);obj=ELF(OBJ)
 def target(n):return n in ('_ZN4sycl3_V16detail27compile_time_kernel_info_v121CompileTimeKernelInfoI16dpct_kernel_nameIJN6strata7kernels12_GLOBAL__N_116native_mmvq_wideENS7_7WideQ5KE18dpct_kernel_scalarILi2EEEEEE', '_ZGVZN4sycl3_V16detail19getDeviceKernelInfoI16dpct_kernel_nameIJN6strata7kernels12_GLOBAL__N_116native_mmvq_wideENS6_7WideQ5KE18dpct_kernel_scalarILi2EEEEEERNS1_16DeviceKernelInfoEvE4Info', '_ZZN4sycl3_V16detail19getDeviceKernelInfoI16dpct_kernel_nameIJN6strata7kernels12_GLOBAL__N_116native_mmvq_wideENS6_7WideQ5KE18dpct_kernel_scalarILi2EEEEEERNS1_16DeviceKernelInfoEvE4Info')
 host=bind(obj,target);name=host['registered_name_pointer']['name']
 require('WideQ5K'in name and 'E16native_mmvq_wide'in name and 'scalarILi2EE'in name and 'rpw'not in name,'one Q5 C2 original registered name')
 owners={fn for row in host['actual_host_relocations']if row['target_metadata']==host['object_info']['name']for fn in row['containing_functions']};require(bool(owners),'selected actual host owner')
 public=[]
 for api in ['native_mmvq_head','native_mmvq','native_q5_k_mmvq']:
  xs=[z for z in obj.symbols if z['section']and z['type']==2 and z['name'].startswith('_ZN6strata7kernels'+str(len(api))+api+'E')];require(len(xs)==1,'unique actual public API '+api);public.append(xs[0]);owners.add(xs[0]['name'])
 hostfiles={}
 for i,fn in enumerate(sorted(owners)):
  raw=subprocess.check_output(['objdump','-drw','--disassemble='+fn,str(OBJ)]);require(bool(re.search(rb'(?m)^0+[0-9a-f]+ <'+re.escape(fn.encode())+rb'>:',raw)),'actual containing function ASM '+fn)
  fname=f'host-{i}.asm';(O/fname).write_bytes(raw);hostfiles[fn]={'path':fname,'bytes':len(raw),'sha256':sha(raw)}
 hostedges=obj.relocs(set(owners));unique={}
 # ONE exact registered-name seek, not enumeration/walking of all embedded images.
 for m in re.compile(rb'(?m)^  - name:[ \t]+'+re.escape(name.encode())+rb'[ \t]*\r?$').finditer(final.data):
  at=m.start();base=final.data.rfind(b'\x7fELF\x02\x01',max(0,at-4*1024*1024),at);require(base>=0,'selected ELF start')
  ss,size=sections(final.data,base);z=ss['.ze_info'];require(base+z[4]<=at<base+z[4]+z[5],'selected name within selected metadata');unique[base]=(ss,size)
 require(len(unique)==1,'exactly ONE registered selected image');base,(ss,size)=next(iter(unique.items()));image=bytes(final.data[base:base+size])
 get=lambda buf,tab,k:buf[tab[k][4]:tab[k][4]+tab[k][5]]if tab[k][1]!=8 else b''
 ze=get(image,ss,'.ze_info').decode('ascii');names=re.findall(r'^  - name:\s+(\S+)',ze.split('kernels_misc_info:')[0],re.M);require(names==[name],'one kernel in selected per-kernel image')
 block=next(x for x in re.split(r'(?m)^  - name:\s+',ze.split('kernels_misc_info:')[0])[1:]if x.splitlines()[0].strip()==name)
 fields={k:re.findall(r'(?m)^\s*'+re.escape(k)+r':\s*(.+)$',block)for k in ['grf_count','simd_size','slm_size','barrier_count','spill_size','scratch_size','eu_thread_count','required_work_group_size','subgroup_independent_forward_progress']}
 options=get(image,ss,'.misc.buildOptions');require(options.rstrip(b'\0')==b'-cl-fp32-correctly-rounded-divide-sqrt','actual exact FP option')
 # Reuse S116's already-reviewed selected data/REL(A) decoder, no new parser framework.
 prior_inventory=R/'hc-producer-s116-compiled-audit-prep/audit.py';nodes=[n for n in ast.walk(ast.parse(prior_inventory.read_text()))if isinstance(n,ast.FunctionDef)and n.name=='extra_inventory'];require(len(nodes)==1,'one retained inventory helper')
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(prior_inventory)+':one-function-only','exec'),globals())
 text=get(image,ss,'.text.'+name);inventory=extra_inventory(image,ss)
 (O/'q5-c2.zebin').write_bytes(image);(O/'q5-c2.ze_info').write_text(ze,encoding='ascii')
 # Optional decoder input is a byte-exact executable/data/reloc alias COPY.
 alias=bytearray(image);t=ss['.text.'+name];so=struct.unpack_from('<Q',image,40)[0];se,sn,si=struct.unpack_from('<HHH',image,58);st=struct.unpack_from('<IIQQQQIIQQ',image,so+si*se);at=st[4]+t[0];oldname=cstr(alias,at);new=b'.text.s122_q5_c2'
 require(oldname==('.text.'+name).encode()and len(new)<len(oldname),'short section alias fits');alias[at:at+len(oldname)]=new+b'\0'*(len(oldname)-len(new));changed=[i for i,(x,y)in enumerate(zip(image,alias))if x!=y]
 require(all(at<=i<at+len(oldname)for i in changed),'only selected section-name bytes change');require(bytes(alias[t[4]:t[4]+t[5]])==text,'all text bytes unchanged')
 (O/'q5-c2-shortname.zebin').write_bytes(alias)
 after={str(Path(v['path'])):stat(Path(v['path']))for v in pins['artifacts'].values()};require(before==after,'frozen artifact continuity after audit')
 save('facts.json',{'status':'PASS_ONE_Q5_C2_STATIC_BINDING_HOST_REVIEW_PENDING_ISA_UNKNOWN','host':host,'host_function_files':hostfiles,'actual_public_and_selected_host_relocations':hostedges,'selected':{'kernel':name,'main_offset':base,'image_bytes':len(image),'image_sha256':sha(image),'text_bytes':len(text),'text_sha256':sha(text),'execution_fields':fields,'missing_fields':'UNKNOWN, not zero; metadata barrier_count is not instruction count','full_execution_payload_metadata':block,'options_ascii':options.rstrip(b'\0').decode(),'data_and_relocations':inventory,'sections':{k:{'type':v[1],'flags':v[2],'bytes':v[5],'sha256':sha(get(image,ss,k))}for k,v in ss.items()}},'alias':{'sha256':sha(alias),'only_section_name_changed':True,'text_data_relocations_unchanged':True,'changed_bytes':len(changed),'section_name':new.decode()},'known_artifact_hashes_reused':{k:v['sha256']for k,v in pins['artifacts'].items()},'main_archive_fresh_hash':False,'selected_object_hash_verified':True,'artifact_stat_before':before,'artifact_stat_after':after,'source_config_conditions':{'N':2560,'M':248320,'NCOLS':2,'source_WG_groups':62080,'source_local_range_xyz':[32,4,1],'head_rpw2':0,'generic_RPW':1},'manual_host_obligation':'Root reads saved public and selected launch ASM for actual flag/type/N/M/ncols forwarding and branch. Static name binding is not runtime dispatch. Source geometry is not decoded runtime scalar payload.','ISA':'UNKNOWN; no exact retained selected ISA established. Root may use separate one-image decoder after actual binding review; never launched automatically.','limits':['RPW1 row owners are separate subgroups/workgroups; no cross-workgroup compiler CSE. Different columns have distinct input.','Only local lowering/strength reduction and cost of two distinct ones-dot half sums are open, not duplication within one j.','Encoded size/service ratio is not physical bandwidth; instruction/sends/waits are not critical-path cycles.','Rawsum integer equality is not universal compiled FP/output equivalence; no candidate prototype or numeric claim.'],'all_image_walk':False,'compiler_decoder_GPU_invocations':0})
 print('PASS_ONE_Q5_C2_STATIC_BINDING_HOST_REVIEW_PENDING_ISA_UNKNOWN')
except BaseException as e:
 save('HOLD.json',{'status':'HOLD_FIRST_AUDIT_FAILURE_NO_RETRY','type':type(e).__name__,'error':str(e),'GPU':0,'decoder_invocations':0});raise
