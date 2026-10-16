"""ROOT GO only: three tiny offline ocloc calls, no source compile/link/GPU."""
from pathlib import Path
import hashlib,json,os,re,struct,subprocess,time
P=Path('/prep');O=Path('/out');sha=lambda b:hashlib.sha256(b).hexdigest()
def save(n,v):(O/n).write_text(json.dumps(v,indent=2)+'\n',encoding='ascii')
def need(c,m):
 if not c:raise RuntimeError('HOLD: '+m)
def sections(b):
 need(b[:6]==b'\x7fELF\x02\x01','ELF64 little endian')
 at=struct.unpack_from('<Q',b,40)[0];sz,n,ix=struct.unpack_from('<HHH',b,58);need(sz==64 and 0<n<1000 and ix<n,'bounded section table')
 ss=[struct.unpack_from('<IIQQQQIIQQ',b,at+i*sz)for i in range(n)];st=ss[ix];names=b[st[4]:st[4]+st[5]]
 return {names[x[0]:names.find(b'\0',x[0])].decode():x for x in ss}
def get(b,s,k):return b[s[k][4]:s[k][4]+s[k][5]]if s[k][1]!=8 else b''
def inventory(p):return {str(f.relative_to(p)):{'bytes':f.stat().st_size,'sha256':sha(f.read_bytes())}for f in sorted(p.rglob('*'))if f.is_file()}
def stats(label):
 cmd=['/usr/bin/ocloc','cache','-show-stats','-dir',str(O/'offline-cache')]
 r=subprocess.run(cmd,capture_output=True);(O/(label+'-stats.stdout')).write_bytes(r.stdout);(O/(label+'-stats.stderr')).write_bytes(r.stderr)
 save(label+'-stats-command.json',{'argv':cmd,'exit':r.returncode});need(r.returncode==0,'supported stats command')
 # Pinned SDK supports this table; fail closed if output/format differs.
 m=re.search(rb'(?m)^Version\s+Hits\s+Misses\s+Hit Rate\s*\n\s*(\d+)\s+(\d+)\s+(\d+)\s+[0-9.]+%',r.stdout)
 need(m is not None,'genuine supported hit/miss counters unavailable; stop before next compile')
 return {'version':int(m[1]),'hits':int(m[2]),'misses':int(m[3])}
def compare(output,row):
 candidates=[]
 for f in output.iterdir():
  if f.is_file()and f.read_bytes()[:6]==b'\x7fELF\x02\x01':candidates.append(f)
 need(len(candidates)==1,'one produced actual zebin')
 new=candidates[0].read_bytes();old=Path(row['source_path']).read_bytes();need(sha(old)==row['source_sha256'],'bound selected reference image')
 ns=sections(new);ss=sections(old);name=row['source_registered_kernel'];target='.text.'+name
 need(target in ns and target in ss,'same selected registered name')
 nt=get(new,ns,target);ot=get(old,ss,target);end=row['native_EOT_prefix_bytes']
 prefix=len(nt)>=end and nt[:end]==ot[:end];meta=get(new,ns,'.ze_info')==get(old,ss,'.ze_info');opts=get(new,ns,'.misc.buildOptions')==get(old,ss,'.misc.buildOptions')
 keys={k for s in[ns,ss]for k in s if k.startswith(('.data','.bss','.rodata'))};data=all(k in ns and k in ss and all(ns[k][i]==ss[k][i]for i in[1,2,5,8])and get(new,ns,k)==get(old,ss,k)for k in keys)
 rel=lambda b,s:{k:get(b,s,k)for k in s if k.startswith(('.rel.','.rela.'))};rels=rel(new,ns)==rel(old,ss)
 comp={'output':str(candidates[0]),'raw_full_image_exact':new==old,'raw_complete_text_exact':nt==ot,'computational_prefix_bytes':end,'computational_prefix_exact':prefix,'full_metadata_equal':meta,'options_equal':opts,'constant_data_equal':data,'relocations_equal':rels,'raw_text_changed_offsets':[i for i,(x,y)in enumerate(zip(nt,ot))if x!=y],'text_size_delta':len(nt)-len(ot),'new_inventory':{k:{'bytes':v[5],'sha256':sha(get(new,ns,k))}for k,v in ns.items()}}
 save(output.name+'-compiled-comparison.json',comp)
 need(prefix and meta and opts and data and rels,'direct-input compilation differs from selected source image; no relaxed or alternative flags')
 return comp
need(os.getuid()==1000 and not Path('/dev/dri').exists(),'CPU-only UID1000/noDRI')
need({x.name for x in O.iterdir()}=={'tmp','offline-cache'} and all((O/n).is_dir()and not any((O/n).iterdir())for n in['tmp','offline-cache']),'fresh precreated empty TMP/cache only; preserve prior attempts')
need(Path.cwd()==O and os.environ.get('NEO_CACHE_PERSISTENT')=='1'and os.environ.get('NEO_CACHE_DIR')==str(O/'offline-cache'),'explicit cwd/enabled absolute env cache path')
cache=O/'offline-cache';st=cache.stat()
need(st.st_uid==1000 and os.access(O,os.W_OK|os.X_OK)and os.access(cache,os.W_OK|os.X_OK),'owned writable cwd and existing cache before ANY ocloc CLI')
need(cache.resolve()==Path(os.environ['NEO_CACHE_DIR']).resolve(),'environment/CLI/statistics same cache path')
save('cache-path-preflight.json',{'actual_cwd':str(Path.cwd()),'uid':os.getuid(),'gid':os.getgid(),'cache_path':str(cache),'cache_resolved':str(cache.resolve()),'cache_uid':st.st_uid,'cache_gid':st.st_gid,'cache_mode':oct(st.st_mode&511),'cwd_writable':os.access(O,os.W_OK|os.X_OK),'cache_writable':os.access(cache,os.W_OK|os.X_OK),'cache_preexists_before_any_ocloc_CLI':True,'NEO_CACHE_PERSISTENT':os.environ['NEO_CACHE_PERSISTENT'],'NEO_CACHE_DIR':os.environ['NEO_CACHE_DIR'],'NEO_CACHE_STATS':os.environ.get('NEO_CACHE_STATS'),'CLI_cache_dir':str(cache),'stats_cache_dir':str(cache),'source_supported_config_path':'26.22.38646.4 getDefaultCompilerCacheConfig reads existing NEO_CACHE_DIR with persistent=1 before compile CLI fallback; actual cache selection/hits still require later counters'})
records=[];pins=json.loads((P/'input-pins.json').read_text());before={n:sha((P/'inputs'/n).read_bytes())for n in pins}
try:
 need(os.environ.get('NEO_CACHE_STATS')=='1','supported counter collection enabled')
 need(all(before[n]==v['sha256']for n,v in pins.items()),'frozen actual SPIR-V inputs')
 save('tool.json',{'ocloc_sha256':sha(Path('/usr/bin/ocloc').read_bytes()),'version':subprocess.check_output(['/usr/bin/ocloc','--version'],text=True),'ldd':subprocess.check_output(['ldd','/usr/bin/ocloc'],text=True)})
 save('environment.json',{k:v for k,v in os.environ.items()if k.startswith(('NEO_','IGC_','SYCL_'))or k in['TMPDIR','OMP_NUM_THREADS','MKL_NUM_THREADS']})
 for label,filename in [('cold','S107-sampler.spv'),('warm','S107-sampler.spv'),('changed','S114-sampler.spv')]:
  output=O/label;output.mkdir()
  cmd=['/usr/bin/ocloc','-output','sampler','-file',str(P/'inputs'/filename),'-output_no_suffix','-spirv_input','-device','bmg-g31','-options','-cl-fp32-correctly-rounded-divide-sqrt','-allow_caching','-cache_dir',str(O/'offline-cache'),'-out_dir',str(output)]
  t=time.monotonic();r=subprocess.run(cmd,capture_output=True);elapsed=time.monotonic()-t
  (O/(label+'.stdout')).write_bytes(r.stdout);(O/(label+'.stderr')).write_bytes(r.stderr);record={'label':label,'argv':cmd,'exit':r.returncode,'elapsed_s':elapsed};records.append(record);save('calls.json',records)
  need(r.returncode==0,'one finite ocloc compilation '+label)
  record['compiled_comparison']=compare(output,pins[filename]);record['stats']=stats(label);record['cache_inventory']=inventory(O/'offline-cache');save('calls.json',records)
  current=record['stats']
  if label=='cold':need(current['hits']==0 and current['misses']>0,'fresh exact input must miss with supported counters')
  else:
   prior=records[-2]['stats'];need(current['version']==prior['version'],'same counter ABI')
   record['stats_delta']={k:current[k]-prior[k]for k in['hits','misses']};delta=record['stats_delta'];save('calls.json',records)
   if label=='warm':need(delta['hits']>0 and delta['misses']==0,'genuine exact-input hit proof required before changed input')
   else:need(delta['misses']>0 and delta['hits']==0,'changed actual IR must independently miss; no input-equivalence assumption')
 need(before=={n:sha((P/'inputs'/n).read_bytes())for n in pins},'immutable input after three calls')
 save('result.json',{'status':'PASS_TINY_EXACT_INPUT_OFFLINE_CACHE_DISCRIMINATOR','calls':records,'actual_compiler_calls':3,'GPU':0,'source_compiles':0,'full_main_links':0,'limits':['One selected module, one process/container and one cold/warm/changed sequence; no population timing or whole-main gain.','Stats enabled only for this CPU diagnostic and may add overhead.','Selected embedded IR need not equal every main backend input; no module-ID normalization or cache fabrication.','Raw full-image/text differences remain explicit; native-prefix reuse is scoped to prior verified EOT boundary.']})
 print('PASS_TINY_EXACT_INPUT_OFFLINE_CACHE_DISCRIMINATOR')
except BaseException as e:
 save('HOLD.json',{'status':'HOLD_STOP_NO_RETRY_OR_FURTHER_COMPILATION','type':type(e).__name__,'reason':str(e),'completed_calls':records,'GPU':0,'no_tolerance_or_flag_relaxation':True});raise
