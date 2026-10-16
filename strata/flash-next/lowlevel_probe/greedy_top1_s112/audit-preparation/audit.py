"""SOURCE ONLY. Root GO required; CPU-only selected host/image audit, no decoder/GPU."""
from pathlib import Path
import argparse, ast, hashlib, json, mmap, re, struct, subprocess
P=Path(__file__).resolve().parent; R=P.parent
ap=argparse.ArgumentParser()
ap.add_argument('--prep',type=Path,required=True)
ap.add_argument('--expected-prep-manifest-sha256',required=True)
ap.add_argument('--expected-exe-sha256',required=True)
a=ap.parse_args();S=a.prep.resolve();B=S/'build';O=R/(S.name+'-built-audit')
assert S.parent==R and S.name.startswith('greedy-top1-s112-oracle-prep'), 'Only reviewed S112 prep under R'
assert re.fullmatch('[0-9a-f]{64}',a.expected_prep_manifest_sha256) and re.fullmatch('[0-9a-f]{64}',a.expected_exe_sha256)
EXE=B/'greedy_top1_s112'; OBJ=B/'oracle.o'
LIVE=Path('/mnt/vm_8tb/b70/build/strata-ab/src-ep')
sha=lambda b:hashlib.sha256(b).hexdigest()
def save(name,value): (O/name).write_text(json.dumps(value,indent=2)+'\n',encoding='ascii')
def need(condition,message):
 if not condition: raise RuntimeError('HOLD: '+message)
need(not O.exists(),'preserve existing audit, no automatic retry');O.mkdir()
try:
 pins=json.loads((P/'pins.json').read_text())
 manifest_bytes=(S/'manifest.json').read_bytes();need(sha(manifest_bytes)==a.expected_prep_manifest_sha256,'root-selected prep manifest')
 prep_manifest=json.loads(manifest_bytes)
 for name,row in prep_manifest['files'].items():
  f=S/name;need(f.resolve().parent==S,'manifest basename');b=f.read_bytes();need(len(b)==row['bytes'] and sha(b)==row['sha256'],'reviewed prep payload '+name)
 need(sha((S/'private_argmax.hpp').read_bytes())==pins['private_hpp_sha256'],'unchanged reviewed private-kernel body')
 projection=json.loads((S/'source-projection.json').read_text())
 need(projection['inverse_exact'] is True and projection['arithmetic_changed'] is False,'source arithmetic/index inverse proof')
 need(projection['source_sha256']=='5faf60ccd6e8d0665281253374bd97d4d874deb86f0aad479ce94f8565b2673e','source inverse baseline')
 for row in pins['small_files']:
  f=Path(row['path']);b=f.read_bytes();need(len(b)==row['bytes'] and sha(b)==row['sha256'],'source/helper/record pin '+str(f))
 need((B/'build.exit').read_text().strip()==(B/'tee.exit').read_text().strip()=='0','build not terminal0')
 need(json.loads((B/'image.json').read_text())[0]['Id']==pins['image_id'],'build image identity')
 need((B/'status.txt').read_text().strip()=='BUILD_PASS_IMAGE_BINDING_AUDIT_REQUIRED_BEFORE_GPU','build status')
 records={line.split(maxsplit=1)[1]:line.split(maxsplit=1)[0] for line in (B/'build-sha256.txt').read_text().splitlines()}
 for file,key in [(S/'oracle.cpp','/src/oracle.cpp'),(S/'private_argmax.hpp','/src/private_argmax.hpp'),(OBJ,'oracle.o'),(EXE,'greedy_top1_s112')]:
  with file.open('rb') as stream: actual=hashlib.file_digest(stream,'sha256').hexdigest()
  need(actual==records[key],'build artifact '+str(file))
 EXPECTED=records['greedy_top1_s112'];need(EXPECTED==a.expected_exe_sha256,'root-selected actual executable hash')
 buildlog=(B/'build.log').read_text(); expected=(S/'expected-container.sha256').read_text().splitlines()
 for line in expected:
  _,name=line.split(maxsplit=1);need(buildlog.count(name+': OK')==2,'pre/post build input closure '+name)
 s107=json.loads((R/'q6-a4-main-s107-build/build/root-build-audit.json').read_text())
 need(s107['actual_exit']==0 and s107['build_sha256']['libstrata_kernels.a']==pins['archive_sha256'],'S107 identity record')
 need(pins['archive_sha256']+'  /work/src-ep/build-sycl-ep/libstrata_kernels.a' in expected,'S112 same pinned S107 archive')
 # One archive directory walk, reading only the TWO selected members. Full archive hash reused from build closure.
 archive=LIVE/'build-sycl-ep/libstrata_kernels.a'; wanted={'sampler.dp.cpp.o','verify_kernels.dp.cpp.o'};members={}
 with archive.open('rb') as f:
  need(f.read(8)==b'!<arch>\n','archive magic');longnames=b''
  while wanted-set(members):
   h=f.read(60);need(len(h)==60 and h[58:]==b'`\n','selected archive member absent/malformed')
   nm=h[:16].decode().strip();n=int(h[48:58]);pos=f.tell()
   if nm=='//':longnames=f.read(n)
   else:
    actual=longnames[int(nm[1:]):].split(b'/\n')[0].decode() if nm.startswith('/') and nm[1:].isdigit() else nm.rstrip('/')
    if actual in wanted:
     need(actual not in members,'duplicate member');members[actual]=f.read(n)
   f.seek(pos+n+(n&1))
 original={}
 original_records={line.split(maxsplit=1)[1]:line.split(maxsplit=1)[0] for line in (B/'original-objects.sha256').read_text().splitlines()}
 for name,data in members.items():
  bound=B/'bound-objects'/name;live=LIVE/'build-sycl-ep/CMakeFiles/strata_kernels.dir/src/kernels/cuda'/name
  need(data==bound.read_bytes()==live.read_bytes(),'archive/bound/original object equality '+name)
  need(sha(data)==original_records['bound-objects/'+name],'actual build original-object hash '+name)
  (O/name).write_bytes(data);original[name]={'bytes':len(data),'sha256':sha(data),'archive':str(archive),'archive_sha256_reused':pins['archive_sha256'],'source':'S107 archive identity plus S112 exact selected-member comparisons'}
 save('build-binding.json',{'status':'PASS_BUILD_AND_TWO_ORIGINAL_MEMBER_BINDINGS','exe_sha256':EXPECTED,'object_sha256':records['oracle.o'],'original_objects':original,'full_archive_rehash':False,'build_hash_record_sha256':sha((B/'build-sha256.txt').read_bytes()),'limits':'Pre/post archive hashes reused from terminal0 frozen build; selected bytes actually compared. No historical individual-object hash invented.'})
 # Reuse pinned prior ELF class and the S104 exact section+addend binding functions only.
 helper=R/'attn-value-binding-s92-build-audit/host_binding.py'
 node=next(n for n in ast.parse(helper.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='ELF')
 def cstr(buf,start):
  end=buf.find(b'\0',start);need(end>=start,'unterminated string');return buf[start:end]
 class Bound(ast.NodeTransformer):
  replacements=0
  def visit_Subscript(self,n):
   n=self.generic_visit(n);c=n.value
   if isinstance(n.slice,ast.Constant) and n.slice.value==0 and isinstance(c,ast.Call) and isinstance(c.func,ast.Attribute) and c.func.attr=='split' and isinstance(c.func.value,ast.Subscript):
    s=c.func.value
    if ((isinstance(s.value,ast.Name) and s.value.id=='strings') or (isinstance(s.value,ast.Attribute) and s.value.attr=='data' and isinstance(s.value.value,ast.Name) and s.value.value.id=='self')) and isinstance(s.slice,ast.Slice) and s.slice.upper is None:
     self.replacements+=1;return ast.copy_location(ast.Call(func=ast.Name(id='cstr',ctx=ast.Load()),args=[s.value,s.slice.lower],keywords=[]),n)
   return n
 transform=Bound();node=transform.visit(node);need(transform.replacements==4,'reviewed bounded string transforms');ast.fix_missing_locations(node)
 code=ast.unparse(node).replace('self.data = path.read_bytes()',"self._file=path.open('rb'); self.data=mmap.mmap(self._file.fileno(),0,access=mmap.ACCESS_READ)").replace('self.sha = hashlib.sha256(self.data).hexdigest()',"self.sha=EXPECTED if path==EXE else hashlib.sha256(self.data).hexdigest()")
 exec(compile(code,str(helper)+':class-only-bounded','exec'),globals())
 prior=R/'q6-a4-s104-compiled-audit-prep/audit.py'
 nodes=[n for n in ast.parse(prior.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in {'refs_to','bind','sections'}]
 need(len(nodes)==3,'prior helper functions');exec(compile(ast.Module(body=nodes,type_ignores=[]),str(prior)+':three-functions-only','exec'),globals())
 final=ELF(EXE);private=ELF(OBJ);sampler=ELF(O/'sampler.dp.cpp.o');verify=ELF(O/'verify_kernels.dp.cpp.o')
 targets={'candidate':(private,'GreedyTop1S112'),'sampler':(sampler,'sampler_greedy_kernel_5b2c3f'),'probability':(verify,'row_top_prob_split_kernel_9495b0'),'select':(verify,'mtp_select_kernel_4611fe')}
 hosts={arm:bind(obj,lambda n,tag=tag:tag in n) for arm,(obj,tag) in targets.items()}
 need(len({h['final_info']['offset'] for h in hosts.values()})==4,'distinct linked Info')
 need(len({h['registered_name_pointer']['pointer'] for h in hosts.values()})==4,'distinct linked names')
 need(len({s['offset'] for h in hosts.values() for s in h['final_cache_guard']})==8,'distinct linked caches/guards')
 # Actual original host wrapper calls from private oracle, rather than selected-name existence alone.
 calls=private.relocs({z['name'] for z in private.symbols if any(t in z['name'] for t in ['sample_tokens','row_top_prob_split','mtp_select'])})
 for token in ['sample_tokens','row_top_prob_split','mtp_select']:need(any(token in x['target_symbol'] and x['containing_functions'] for x in calls),'oracle wrapper relocation '+token)
 for arm,h in hosts.items():
  obj=targets[arm][0];functions=sorted({fn for row in h['actual_host_relocations'] if row['target_metadata']==h['object_info']['name'] for fn in row['containing_functions']});need(bool(functions),'host owner '+arm);h['actual_callers']=functions
  for i,fn in enumerate(functions):(O/(arm+'-host-'+str(i)+'.asm')).write_bytes(subprocess.check_output(['objdump','-drw','--disassemble='+fn,str(obj.path)]))
 selected={};get=lambda buf,tab,k:buf[tab[k][4]:tab[k][4]+tab[k][5]] if tab[k][1]!=8 else b''
 for arm,h in hosts.items():
  name=h['registered_name_pointer']['name'];unique={}
  for match in re.compile(rb'(?m)^  - name:[ \t]+'+re.escape(name.encode())+rb'[ \t]*\r?$').finditer(final.data):
   at=match.start();base=final.data.rfind(b'\x7fELF\x02\x01',max(0,at-4*1024*1024),at);need(base>=0,'selected zebin base');ss,size=sections(final.data,base);z=ss['.ze_info'];need(base+z[4]<=at<base+z[4]+z[5],'name in selected ze_info');unique[base]=(ss,size)
  need(len(unique)==1,'unique selected image '+arm);base,(ss,size)=next(iter(unique.items()));image=bytes(final.data[base:base+size]);ze=get(image,ss,'.ze_info').decode();text=get(image,ss,'.text.'+name)
  names=re.findall(r'^  - name:\s+(\S+)',ze.split('kernels_misc_info:')[0],re.M);need(names.count(name)==1,'selected kernel metadata uniqueness')
  block=next(x for x in re.split(r'(?m)^  - name:\s+',ze.split('kernels_misc_info:')[0])[1:] if x.splitlines()[0].strip()==name)
  options=get(image,ss,'.misc.buildOptions');need(options.rstrip(b'\0')==b'-cl-fp32-correctly-rounded-divide-sqrt','actual compile options '+arm)
  fields={k:re.findall(r'(?m)^\s*'+re.escape(k)+r':\s*(.+)$',block) for k in ['grf_count','simd_size','slm_size','barrier_count','spill_size','scratch_size','eu_thread_count']}
  (O/(arm+'.zebin')).write_bytes(image);(O/(arm+'.ze_info')).write_text(ze,encoding='ascii')
  selected[arm]={'kernel':name,'executable_offset':base,'image_bytes':size,'image_sha256':sha(image),'text_bytes':len(text),'text_sha256':sha(text),'all_kernel_names':names,'full_kernel_metadata':block,'execution_fields':fields,'missing_fields':'unknown/absent, not measuredzero','options_ascii':options.rstrip(b'\0').decode('ascii'),'inventory':{k:{'type':s[1],'flags':s[2],'bytes':s[5],'sha256':sha(get(image,ss,k))} for k,s in ss.items()}}
 # Alias only selected candidate section NAME on a copy, preserving every executable text byte.
 image=(O/'candidate.zebin').read_bytes();ss,_=sections(image);name=selected['candidate']['kernel'];t=ss['.text.'+name];alias=bytearray(image)
 so=struct.unpack_from('<Q',image,40)[0];se,sn,si=struct.unpack_from('<HHH',image,58);st=struct.unpack_from('<IIQQQQIIQQ',image,so+si*se);at=st[4]+t[0];old=cstr(alias,at);new=b'.text.s112_top1_candidate'
 need(old==('.text.'+name).encode() and len(new)<len(old),'selected alias fits');alias[at:at+len(old)]=new+b'\0'*(len(old)-len(new));changed=[i for i,(a,b) in enumerate(zip(image,alias)) if a!=b]
 need(all(st[4]<=i<st[4]+st[5] for i in changed),'alias section-name bytes only');need(bytes(alias[t[4]:t[4]+t[5]])==get(image,ss,'.text.'+name),'alias text unchanged');(O/'candidate-shortname.zebin').write_bytes(alias)
 save('facts.json',{'status':'PASS_STATIC_BINDINGS_SELECTED_IMAGES_CANDIDATE_ISA_PENDING_ROOT_GO','exe_sha256':EXPECTED,'object_sha256':records['oracle.o'],'hosts':hosts,'oracle_original_wrapper_calls':calls,'selected':selected,'original_objects':original,'alias_sha256':sha(alias),'alias_text_unchanged':True,'all_image_walk':False,'decoder_invocations':0,'GPU':0,'limits':['Static hostfunction/Info/cache/guard/name/image binding is not runtime dispatch or numerical proof.','Original archive kernels are exact member bindings to S107; no redundant ISA decode or universal binary equivalence claim.','Candidate acquire after atomic before WG barrier MUST be checked in one later selected ISA decode and source ledger; metadata alone cannot establish instruction ordering.','Arithmetic/index/tie/NaN source projection was checked independently before build; compiled arithmetic proof remains pending.']})
 print('PASS_STATIC_BINDINGS_CANDIDATE_ISA_PENDING_ROOT_GO')
except BaseException as e:
 save('HOLD.json',{'status':'HOLD_STOP_FIRST_FAILURE','type':type(e).__name__,'error':str(e),'GPU':0,'decoder':0});raise
