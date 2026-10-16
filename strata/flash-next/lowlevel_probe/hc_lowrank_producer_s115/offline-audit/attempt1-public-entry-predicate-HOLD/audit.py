"""Root-authorized CPU-only selected S107 HC extraction; no compiler/decoder/GPU."""
from pathlib import Path
import ast, hashlib, json, mmap, re, struct, subprocess
P=Path(__file__).resolve().parent;R=P.parent;O=P/'selected'
EXE=R/'q6-a4-main-s107-build/build/strata-q6-a4-s107'
EXPECTED='064b63ce75d7568db541af5cf24eee07f41d0e54ea49813af9958a816d75d5ca'
LIVE=Path('/mnt/vm_8tb/b70/build/strata-ab/src-ep')
ARCHIVE=LIVE/'build-sycl-ep/libstrata_kernels.a'
ORIGINAL=LIVE/'build-sycl-ep/CMakeFiles/strata_kernels.dir/src/kernels/cuda/fused_gr.dp.cpp.o'
sha=lambda b:hashlib.sha256(b).hexdigest()
def save(n,x):(O/n).write_text(json.dumps(x,indent=2,ensure_ascii=True)+'\n',encoding='ascii')
def stat(p):
 s=p.stat();return {'device':s.st_dev,'inode':s.st_ino,'size':s.st_size,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns}
assert not O.exists(),'Preserve existing extraction; no automatic retry'
O.mkdir()
try:
 pins=json.loads((P/'pins.json').read_text())
 for row in pins['small_inputs']:
  p=Path(row['path']);b=p.read_bytes();assert len(b)==row['bytes'] and sha(b)==row['sha256'],p
 root=json.loads((R/'s113-post-hc-source-reviews-20261016/root-post-decision.json').read_text())
 assert root['root_HC_selected_source_read'] and root['root_read_both_full_assistant_texts']
 build=json.loads((R/'q6-a4-main-s107-build/build/root-build-audit.json').read_text())
 assert build['actual_exit']==0 and build['build_sha256']['/out/strata-q6-a4-s107']==EXPECTED
 assert build['build_sha256']['libstrata_kernels.a']==pins['archive_sha256_reused']
 before={str(p):stat(p)for p in [EXE,ARCHIVE,ORIGINAL]}
 # Read only one selected archive member. No main/archive full rehash.
 member=None
 with ARCHIVE.open('rb')as f:
  assert f.read(8)==b'!<arch>\n';longnames=b''
  while True:
   h=f.read(60)
   if not h:break
   assert len(h)==60 and h[58:]==b'`\n';name=h[:16].decode().strip();n=int(h[48:58]);pos=f.tell()
   if name=='//':longnames=f.read(n)
   else:
    actual=longnames[int(name[1:]):].split(b'/\n')[0].decode()if name.startswith('/')and name[1:].isdigit()else name.rstrip('/')
    if actual=='fused_gr.dp.cpp.o':member=f.read(n);break
   f.seek(pos+n+(n&1))
 assert member is not None and member==ORIGINAL.read_bytes(),'Selected archive/object mismatch'
 OBJ=O/'fused_gr.dp.cpp.o';OBJ.write_bytes(member)
 # Reuse exactly the audited ELF class, with its four bounded-string transforms.
 helper=R/'attn-value-binding-s92-build-audit/host_binding.py'
 assert sha(helper.read_bytes())=='37ecf55c43300319f0a3baaf15bede814934bd3ee1b0a11b32a38dc6066d4aa6'
 node=next(n for n in ast.parse(helper.read_text()).body if isinstance(n,ast.ClassDef)and n.name=='ELF')
 def cstr(buf,start):
  end=buf.find(b'\0',start);assert end>=start;return buf[start:end]
 class Bound(ast.NodeTransformer):
  replacements=0
  def visit_Subscript(self,n):
   n=self.generic_visit(n);c=n.value
   if isinstance(n.slice,ast.Constant)and n.slice.value==0 and isinstance(c,ast.Call)and isinstance(c.func,ast.Attribute)and c.func.attr=='split'and isinstance(c.func.value,ast.Subscript):
    s=c.func.value
    if ((isinstance(s.value,ast.Name)and s.value.id=='strings')or(isinstance(s.value,ast.Attribute)and s.value.attr=='data'and isinstance(s.value.value,ast.Name)and s.value.value.id=='self'))and isinstance(s.slice,ast.Slice)and s.slice.upper is None:
     self.replacements+=1;return ast.copy_location(ast.Call(func=ast.Name(id='cstr',ctx=ast.Load()),args=[s.value,s.slice.lower],keywords=[]),n)
   return n
 transform=Bound();node=transform.visit(node);assert transform.replacements==4;ast.fix_missing_locations(node)
 code=ast.unparse(node).replace('self.data = path.read_bytes()',"self._file=path.open('rb'); self.data=mmap.mmap(self._file.fileno(),0,access=mmap.ACCESS_READ)").replace('self.sha = hashlib.sha256(self.data).hexdigest()',"self.sha=EXPECTED if path==EXE else hashlib.sha256(self.data).hexdigest()")
 exec(compile(code,str(helper)+':bounded-class-only','exec'),globals())
 prior=R/'q6-a4-s104-compiled-audit-prep/audit.py'
 nodes=[n for n in ast.parse(prior.read_text()).body if isinstance(n,ast.FunctionDef)and n.name in {'refs_to','bind','sections'}]
 assert len(nodes)==3
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(prior)+':three-functions-only','exec'),globals())
 final=ELF(EXE);obj=ELF(OBJ)
 hosts={'up':bind(obj,lambda n:'gr_up_multi_cols' in n and 'grf256'not in n and 'ilp'not in n and 'dpct_kernel_scalarILi16EE'in n),
        'reduce':bind(obj,lambda n:'gr_down_reduce_k' in n)}
 assert len({h['final_info']['offset']for h in hosts.values()})==2
 assert len({h['registered_name_pointer']['pointer']for h in hosts.values()})==2
 assert len({z['offset']for h in hosts.values()for z in h['final_cache_guard']})==4
 wanted={fn for h in hosts.values()for row in h['actual_host_relocations']if row['target_metadata']==h['object_info']['name']for fn in row['containing_functions']}
 fused=[z for z in obj.symbols if z['type']==2 and z['section']and 'fused_gr_read_multi'in z['name']and '.cold'not in z['name']]
 assert len(fused)==1;wanted.add(fused[0]['name'])
 host_files={}
 for i,fn in enumerate(sorted(wanted)):
  name=f'host-{i}.asm';raw=subprocess.check_output(['objdump','-drw','--disassemble='+fn,str(OBJ)]);(O/name).write_bytes(raw)
  host_files[fn]={'path':name,'bytes':len(raw),'sha256':sha(raw)}
 host_calls=obj.relocs({z['name']for z in obj.symbols if any(t in z['name']for t in ['launch_gr_up','gr_down_sliced','gr_more_groups','gr_large_grf','gr_up_cols','gr_up_ilp'])})
 get=lambda buf,tab,k:buf[tab[k][4]:tab[k][4]+tab[k][5]]if tab[k][1]!=8 else b''
 selected={}
 for arm,h in hosts.items():
  name=h['registered_name_pointer']['name'];unique={}
  for match in re.compile(rb'(?m)^  - name:[ \t]+'+re.escape(name.encode())+rb'[ \t]*\r?$').finditer(final.data):
   at=match.start();base=final.data.rfind(b'\x7fELF\x02\x01',max(0,at-4*1024*1024),at);assert base>=0
   ss,size=sections(final.data,base);z=ss['.ze_info'];assert base+z[4]<=at<base+z[4]+z[5];unique[base]=(ss,size)
  assert len(unique)==1,(arm,len(unique))
  base,(ss,size)=next(iter(unique.items()));image=bytes(final.data[base:base+size]);ze=get(image,ss,'.ze_info').decode('ascii');text=get(image,ss,'.text.'+name)
  names=re.findall(r'^  - name:\s+(\S+)',ze.split('kernels_misc_info:')[0],re.M);assert names==[name],names
  block=next(x for x in re.split(r'(?m)^  - name:\s+',ze.split('kernels_misc_info:')[0])[1:]if x.splitlines()[0].strip()==name)
  fields={k:re.findall(r'(?m)^\s*'+re.escape(k)+r':\s*(.+)$',block)for k in ['grf_count','simd_size','slm_size','barrier_count','spill_size','scratch_size','eu_thread_count','has_global_atomics','subgroup_independent_forward_progress']}
  opts=get(image,ss,'.misc.buildOptions');assert opts.rstrip(b'\0')==b'-cl-fp32-correctly-rounded-divide-sqrt'
  (O/(arm+'.zebin')).write_bytes(image);(O/(arm+'.ze_info')).write_text(ze,encoding='ascii')
  # Short text-section alias on a copy. All non-section-name bytes stay exact.
  alias=bytearray(image);t=ss['.text.'+name];so=struct.unpack_from('<Q',image,40)[0];se,sn,si=struct.unpack_from('<HHH',image,58);st=struct.unpack_from('<IIQQQQIIQQ',image,so+si*se)
  at=st[4]+t[0];old=cstr(alias,at);new=('.text.hc_'+arm).encode();assert old==('.text.'+name).encode()and len(new)<len(old)
  alias[at:at+len(old)]=new+b'\0'*(len(old)-len(new));changed=[i for i,(a,b)in enumerate(zip(image,alias))if a!=b]
  assert all(at<=i<at+len(old)for i in changed)
  assert bytes(alias[t[4]:t[4]+t[5]])==text
  (O/(arm+'-shortname.zebin')).write_bytes(alias)
  selected[arm]={'kernel':name,'main_image_offset':base,'image_bytes':len(image),'image_sha256':sha(image),'text_bytes':len(text),'text_sha256':sha(text),'execution_fields':fields,'absent_fields':'unknown, not zero','full_kernel_metadata':block,'options_ascii':opts.rstrip(b'\0').decode('ascii'),'inventory':{k:{'type':s[1],'flags':s[2],'bytes':s[5],'sha256':sha(get(image,ss,k))}for k,s in ss.items()},'short_alias':{'bytes':len(alias),'sha256':sha(alias),'section_name':new.decode(),'only_selected_section_name_changed':True,'text_byte_exact':True}}
 runtime=[]
 for stage in ['attn-value-fixed-policy-s113-prep','q6-a4-main-s107-prep']:
  for arm in [0,1]:
   d=R/stage/f'mode{arm}';native=json.loads((d/'native-process-identity.json').read_text())[0];cfg=json.loads((d/'config.json').read_text());ci=json.loads((d/'container-ready.json').read_text())
   assert native['sha256']==EXPECTED and ci['Image']==pins['image']
   runtime.append({'stage':stage,'arm':arm,'exe_sha256':native['sha256'],'image':ci['Image'],'configured_HC_flags':{k:v for k,v in cfg['env'].items()if k.startswith('STRATA_GR_')},'recorded_effective_HC_flags':{k:v for k,v in native['effective_config_env'].items()if k.startswith('STRATA_GR_')},'container_HC_environment':[k for k in ci['Config']['Env']if k.startswith('STRATA_GR_')],'effective_env_scope':'Native identity is a configured-key projection; absence there alone cannot prove full process absence. Container env and child_env source inherit/overlay strengthen source-conditioned normal-default selection, not dynamic dispatch proof.'})
 after={str(p):stat(p)for p in [EXE,ARCHIVE,ORIGINAL]};assert before==after,'Input changed during selected extraction'
 save('facts.json',{'status':'PASS_SELECTED_TWO_STATIC_BINDINGS_DECODE_REQUIRED','main':{'path':str(EXE),'known_sha256_reused':EXPECTED,'fresh_complete_hash':False,'stat_before':before[str(EXE)],'stat_after':after[str(EXE)]},'archive':{'path':str(ARCHIVE),'known_sha256_reused':pins['archive_sha256_reused'],'fresh_complete_hash':False,'selected_member_equals_object':True,'member_bytes':len(member),'member_sha256':sha(member),'stat_before':before[str(ARCHIVE)],'stat_after':after[str(ARCHIVE)]},'source_pin':next(x for x in pins['small_inputs']if x['path'].endswith('/fused_gr.dp.cpp')),'hosts':hosts,'host_function_files':host_files,'actual_host_helper_relocations':host_calls,'selected':selected,'historical_runtime_controls':runtime,'all_image_walk':False,'decoder_invocations':0,'GPU':0,'limits':['Static host Info/cache/guard/name/exact selected image binding is not runtime dispatch proof.','No current main/archive full hash; root-authorized existing identities reused with pre/post stat continuity and exact selected archive member/object comparison.','Normal-default source branch and historical selected S95 event name are separate evidence from static host binding.','Absent resource fields unknown; raw byte identity and ordered floating-point dataflow require separate claims.']})
 print(json.dumps({'status':'PASS_SELECTED_TWO_STATIC_BINDINGS_DECODE_REQUIRED','resources':{k:v['execution_fields']for k,v in selected.items()},'selected_image_bytes':{k:v['image_bytes']for k,v in selected.items()}}))
except BaseException as e:
 save('HOLD.json',{'status':'HOLD_STOP_FIRST_FAILURE','type':type(e).__name__,'error':str(e),'GPU':0});raise
