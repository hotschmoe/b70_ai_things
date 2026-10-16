"""FROZEN PREPARATION ONLY; ROOT separately authorizes CPU audit. No decoder/GPU."""
from pathlib import Path
import argparse,ast,hashlib,json,mmap,re,struct,subprocess,shutil
P=Path(__file__).resolve().parent;R=P.parent
ap=argparse.ArgumentParser()
ap.add_argument('--expected-exe-sha256',required=True)
ap.add_argument('--expected-mtp-object-sha256',required=True)
a=ap.parse_args()
for v in vars(a).values():assert re.fullmatch('[0-9a-f]{64}',v)
S=R/'mtp-parallel-argmax-main-s114-build';B=S/'build';O=R/'mtp-parallel-argmax-main-s114-audit'
LIVE=Path('/mnt/vm_8tb/b70/build/strata-ab/src-ep');LB=LIVE/'build-sycl-ep'
EXE=B/'strata-mtp-argmax-s114';OBJ=LB/'CMakeFiles/strata_engine.dir/src/core/mtp.cpp.o'
BASE=R/'q6-a4-main-s107-build/build/strata-q6-a4-s107'
EXPECTED=a.expected_exe_sha256
sha=lambda b:hashlib.sha256(b).hexdigest()
def save(n,v):(O/n).write_text(json.dumps(v,indent=2)+'\n',encoding='ascii')
def need(c,m):
 if not c:raise RuntimeError('HOLD: '+m)
def digest(p):
 with p.open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
need(O.is_dir() and not any(O.iterdir()),'fresh empty output, preserve all failed audit attempts')
(O/'tmp').mkdir()
try:
 for row in json.loads((P/'reference-pins.json').read_text())['files']:
  f=Path(row['path']);need(f.stat().st_size==row['bytes'] and digest(f)==row['sha256'],'pinned small reference '+str(f))
 need((B/'build.exit').read_text().strip()==(B/'tee.exit').read_text().strip()=='0','terminal0 build and tee')
 need(json.loads((B/'image.json').read_text())[0]['Id']=='sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2','build image')
 rec={x.split(maxsplit=1)[1]:x.split()[0]for x in(B/'build-sha256.txt').read_text().splitlines()}
 need(rec['/out/strata-mtp-argmax-s114']==EXPECTED==digest(EXE),'actual new frozen executable')
 need(rec['CMakeFiles/strata_engine.dir/src/core/mtp.cpp.o']==a.expected_mtp_object_sha256==digest(OBJ),'actual MTP object')
 need(rec['libstrata_kernels.a']=='cf710683aec404599d1cc6173bf8364e6d30731bda82b315df286bf58aed9b38','unchanged S107 kernel archive receipt')
 need(digest(BASE)=='064b63ce75d7568db541af5cf24eee07f41d0e54ea49813af9958a816d75d5ca','actual frozen S107 baseline, one CPU-lease hash')
 # Reuse terminal0 build's pre/post archive identity; never rehash archive/model/raw trace here.
 for pinfile in ['expected-applied-source.sha256','expected-unchanged-source.sha256']:
  for line in(S/pinfile).read_text().splitlines():
   expected,path=line.split(maxsplit=1);f=Path(path.replace('/work',str(LIVE.parent),1));need(digest(f)==expected,'current source '+str(f))
 log=(B/'build.log').read_text()
 for line in(S/'expected-unchanged-build-inputs.sha256').read_text().splitlines():
  _,name=line.split(maxsplit=1);need(log.count(name+': OK')==2,'pre/post unchanged build receipt '+name)
 plan=json.loads((B/'root-validated-plan.json').read_text())
 need(plan=={'compiled':[{'source':'/work/src-ep/sycl/src/core/mtp.cpp','object':'CMakeFiles/strata_engine.dir/src/core/mtp.cpp.o'}],'archives':['libstrata_engine.a'],'links':1},'one MTP compile/archive/link')
 # Only TWO exact existing archive members read, using previous selected-member identities.
 expected_members={'sampler.dp.cpp.o':'1199c7addeeb64dc3cd51f5d6218d1379696d6674e386b83144787ebf8fe97e7','verify_kernels.dp.cpp.o':'584e694fc1ece4740fe4550e2c34fafd28a77921d81960039d01f3c27dbb6613'}
 members={}
 with(LB/'libstrata_kernels.a').open('rb')as f:
  need(f.read(8)==b'!<arch>\n','archive magic');longnames=b''
  while set(members)!=set(expected_members):
   h=f.read(60);need(len(h)==60 and h[58:]==b'`\n','selected member absent');nm=h[:16].decode().strip();n=int(h[48:58]);pos=f.tell()
   if nm=='//':longnames=f.read(n)
   else:
    name=longnames[int(nm[1:]):].split(b'/\n')[0].decode()if nm.startswith('/')and nm[1:].isdigit()else nm.rstrip('/')
    if name in expected_members:need(name not in members,'duplicate selected member');members[name]=f.read(n)
   f.seek(pos+n+(n&1))
 for name,b in members.items():
  need(sha(b)==expected_members[name],'unchanged actual archive member '+name)
  need(b==(LB/'CMakeFiles/strata_kernels.dir/src/kernels/cuda'/name).read_bytes(),'archive/live selected member '+name)
  (O/name).write_bytes(b)
 # AST-only reuse of established bounded ELF and section/addend metadata binding helpers.
 exec(compile((P/'helpers.py').read_text(),str(P/'helpers.py'),'exec'),globals())
 private=ELF(OBJ);sampler=ELF(O/'sampler.dp.cpp.o');verify=ELF(O/'verify_kernels.dp.cpp.o')
 tags={'candidate':(private,'GreedyTop1S112'),'sampler':(sampler,'sampler_greedy_kernel_5b2c3f'),'probability':(verify,'row_top_prob_split_kernel_9495b0'),'select':(verify,'mtp_select_kernel_4611fe')}
 final=ELF(EXE);hosts={arm:bind(obj,lambda n,t=tag:t in n)for arm,(obj,tag)in tags.items()}
 need(len({h['final_info']['offset']for h in hosts.values()})==4,'four distinct final Info')
 need(len({h['registered_name_pointer']['pointer']for h in hosts.values()})==4,'four distinct final names')
 need(len({z['offset']for h in hosts.values()for z in h['final_cache_guard']})==8,'eight distinct caches/guards')
 for arm,h in hosts.items():
  funcs=sorted({fn for row in h['actual_host_relocations']if row['target_metadata']==h['object_info']['name']for fn in row['containing_functions']});need(funcs,'actual containing host function '+arm);h['actual_callers']=funcs
  for i,fn in enumerate(funcs):(O/(arm+'-host-'+str(i)+'.asm')).write_bytes(subprocess.check_output(['objdump','-drw','--disassemble='+fn,str(tags[arm][0].path)]))
 # Save exact MTP host call forwarding and record_post_head code, not merely name existence.
 calls=private.relocs({z['name']for z in private.symbols if any(t in z['name']for t in ['sample_tokens','row_top_prob_split','mtp_select'])})
 for token in ['sample_tokens','row_top_prob_split','mtp_select']:need(any(token in x['target_symbol']and x['containing_functions']for x in calls),'main MTP forwarding '+token)
 post=[z for z in private.symbols if z['type']==2 and z['section']and'record_post_head'in z['name']]
 need(post,'compiled record_post_head symbol')
 for i,z in enumerate(post):(O/('record-post-head-'+str(i)+'.asm')).write_bytes(subprocess.check_output(['objdump','-drw','--disassemble='+z['name'],str(OBJ)]))
 get=lambda buf,tab,k:buf[tab[k][4]:tab[k][4]+tab[k][5]]if tab[k][1]!=8 else b''
 def selected(exe,host,label):
  name=host['registered_name_pointer']['name'];unique={}
  for m in re.compile(rb'(?m)^  - name:[ \t]+'+re.escape(name.encode())+rb'[ \t]*\r?$').finditer(exe.data):
   at=m.start();base=exe.data.rfind(b'\x7fELF\x02\x01',max(0,at-4*1024*1024),at);need(base>=0,'selected ELF base');ss,size=sections(exe.data,base);z=ss['.ze_info'];need(base+z[4]<=at<base+z[4]+z[5],'selected metadata section');unique[base]=(ss,size)
  need(len(unique)==1,'one selected image '+label);base,(ss,size)=next(iter(unique.items()));image=bytes(exe.data[base:base+size]);ze=get(image,ss,'.ze_info').decode();text=get(image,ss,'.text.'+name)
  names=re.findall(r'^  - name:\s+(\S+)',ze.split('kernels_misc_info:')[0],re.M);need(names.count(name)==1,'one selected metadata '+label)
  block=next(x for x in re.split(r'(?m)^  - name:\s+',ze.split('kernels_misc_info:')[0])[1:]if x.splitlines()[0].strip()==name)
  options=get(image,ss,'.misc.buildOptions');need(options.rstrip(b'\0')==b'-cl-fp32-correctly-rounded-divide-sqrt','actual AOT options')
  fields={k:re.findall(r'(?m)^\s*'+k+r':\s*(.+)$',block)for k in['grf_count','simd_size','slm_size','barrier_count','spill_size','scratch_size','eu_thread_count']}
  (O/(label+'.zebin')).write_bytes(image);(O/(label+'.ze_info')).write_text(ze,encoding='ascii')
  info={'kernel':name,'executable_offset':base,'image_bytes':size,'image_sha256':sha(image),'text_bytes':len(text),'text_sha256':sha(text),'all_kernel_names':names,'full_kernel_metadata':block,'execution_fields':fields,'absent_fields':'unknown, not measured zero','inventory':{k:{'bytes':z[5],'type':z[1],'flags':z[2],'sha256':sha(get(image,ss,k))}for k,z in ss.items()}}
  return info,image,ss
 current={k:selected(final,h,k)for k,h in hosts.items()}
 final=ELF(BASE);basehosts={k:bind(tags[k][0],lambda n,t=tags[k][1]:t in n)for k in ['sampler','probability','select']}
 baseline={k:selected(final,h,'S107-'+k)for k,h in basehosts.items()}
 # Restore global final after baseline binding to avoid any later accidental cross-final use.
 final=ELF(EXE)
 save('selection.json',{'status':'ACTUAL_HOST_BINDINGS_AND_SELECTED_IMAGES_SAVED_COMPARISON_PENDING','exe_sha256':EXPECTED,'mtp_object_sha256':a.expected_mtp_object_sha256,'hosts':hosts,'S107_hosts':basehosts,'MTP_forwarding_calls':calls,'record_post_head_symbols':post,'selected':{k:v[0]for k,v in current.items()},'S107_selected':{k:v[0]for k,v in baseline.items()},'all_image_walk':False,'decoder':0,'GPU':0})
 def compare(new,old,prefix=None):
  ni,nb,ns=new;oi,ob,os=old;nn=ni['kernel'];on=oi['kernel'];nt=get(nb,ns,'.text.'+nn);ot=get(ob,os,'.text.'+on)
  keys={k for tab in[ns,os]for k in tab if k.startswith(('.data','.bss','.rodata'))}
  constants=all(k in ns and k in os and all(ns[k][f]==os[k][f]for f in[1,2,5,8])and get(nb,ns,k)==get(ob,os,k)for k in keys)
  rel=lambda b,s,n:{k.replace(n,'TARGET'):get(b,s,k)for k in s if k.startswith(('.rel.','.rela.'))}
  meta=get(nb,ns,'.ze_info').decode().split('kernels_misc_info:')[0].replace(nn,'TARGET')==get(ob,os,'.ze_info').decode().split('kernels_misc_info:')[0].replace(on,'TARGET')
  textok=nt==ot if prefix is None else len(nt)>=prefix and len(ot)>=prefix and nt[:prefix]==ot[:prefix]
  comp={'raw_full_image_exact':nb==ob,'raw_complete_text_exact':nt==ot,'computational_prefix_bytes':prefix,'computational_prefix_exact':textok,'execution_payload_metadata_equal':meta,'build_options_equal':get(nb,ns,'.misc.buildOptions')==get(ob,os,'.misc.buildOptions'),'constant_data_equal':constants,'relocation_sections_equal':rel(nb,ns,nn)==rel(ob,os,on),'text_size_delta':len(nt)-len(ot),'text_changed_offsets':[i for i,(x,y)in enumerate(zip(nt,ot))if x!=y],'section_diffs':{k:{'new':ni['inventory'].get(k),'reference':oi['inventory'].get(k)}for k in set(ni['inventory'])|set(oi['inventory'])if ni['inventory'].get(k)!=oi['inventory'].get(k)}}
  comp['PASS']=textok and meta and comp['build_options_equal']and constants and comp['relocation_sections_equal'];return comp
 comparisons={k:compare(current[k],baseline[k])for k in baseline}
 oldfacts=json.loads((R/'greedy-top1-s112-oracle-prep-v2-built-audit/facts.json').read_text());oi=oldfacts['selected']['candidate'];ob=(R/'greedy-top1-s112-oracle-prep-v2-built-audit/candidate.zebin').read_bytes();os,_=sections(ob)
 asm=(R/'greedy-top1-s112-oracle-prep-v2-built-audit/candidate-isa/disasm/.text.s112_top1_candidate.asm').read_text()
 need(sha(ob)==oi['image_sha256']=='2d411054e0940bb92eb55c35d3d2671c455ca08aa9245127a32a7226b0ac1f46','actual S112 candidate image')
 need(re.search(r'\{EOT,[^\n]*\nL9728:',asm)and all(int(v)<9728 for line in asm.splitlines()if re.search(r'\b(goto|jmpi|join)\b',line)for v in re.findall(r'\bL(\d+)\b',line)),'actual reference EOT and bounded branch targets')
 comparisons['candidate']=compare(current['candidate'],(oi,ob,os),9728)
 comparisons['candidate']['reference_EOT_end']=9728;comparisons['candidate']['reference_ASM_sha256']=sha(asm.encode())
 save('comparison.json',comparisons)
 passed=all(x['PASS']for x in comparisons.values())
 # Preserve exact source gate/owner receipts for required root manual host-forwarding review.
 proof=json.loads((R/'mtp-parallel-argmax-integration-patch-prep/source-proof.json').read_text())
 save('source-owner-proof.json',proof)
 save('facts.json',{'status':'PASS_COMPILED_REUSE_PENDING_ROOT_HOST_GUARD_FORWARDING_READ'if passed else'HOLD_COMPILED_DIFFERENCES_ROOT_SELECTED_INSPECTION_REQUIRED','exe_sha256':EXPECTED,'mtp_object_sha256':a.expected_mtp_object_sha256,'comparison':comparisons,'selection':'selection.json','GPU_GO':False,'decoder_invocations':0,'all_image_walk':False,'limits':['Static actual callsite/metadata/name/image binding is not independent runtime dispatch proof.','Root must read compiled record_post_head guard/argument forwarding against reviewed source/owner ledger before GPU GO.','Candidate computational-prefix identity reuses S112 publication-chain ISA; suffix/SPIR-V/section differences remain raw and are not silently called equal.','Any meaningful instruction/metadata/data/relocation difference is HOLD for root-selected inspection, no automatic decoder or rebuild.','Resource absence is unknown; no runtime scratch/spill/traffic/latency claim.']})
 if not passed:raise SystemExit(5)
 print('PASS_COMPILED_REUSE_PENDING_ROOT_HOST_GUARD_FORWARDING_READ')
except BaseException as e:
 save('HOLD.json',{'status':'HOLD_PRESERVE_FIRST_RESULT','type':type(e).__name__,'error':str(e),'GPU':0,'decoder':0});raise
