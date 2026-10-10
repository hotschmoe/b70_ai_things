"""Actual selected graph row provenance and complete49 comparison, no raw synthesis."""
import hashlib,math,re,struct
from pathlib import Path
from full_cache_shared_prefix_jobs_v2 import prefix_jobs
from full_cache_shared_phase_contract_v2 import require
def fields(line):return dict(re.findall(r'(\w+)=([^ ]+)',line))
def vector_role(phase):
 for prefix,role in (('direct_gen_','direct_gen'),('admission_','admission'),('batch_step_','later'),('solo_migration_','solo_migration')):
  if phase.startswith(prefix):return role
 raise ValueError('Unknown actual source39 vector phase')
def replay_role(phase):
 require(type(phase)is int and phase in (0,1,2,3),'Exact source39 phase required')
 return ('later','admission','solo_migration','direct_gen')[phase]
def replay_geometry(f):
 n=int(f['rows']);mask=int(f['active_mask']);selected=int(f['selected_mask']);phase=int(f['admission']);event=int(f['event'])
 require(1<=n<=2 and 0<mask<128 and mask.bit_count()==n and 0<selected<(1<<n) and phase in (0,1,2,3),'Actual bounded complete replay masks invalid')
 require((n==1 and event==0) if phase==1 else (n==1 and mask==64 and event==0) if phase==3 else (n==1 and mask==64 and event>0) if phase==2 else (n==2 and event>0),'Actual admission/body/solo/direct replay geometry differs')
 return phase

def required_roles(phase,work,terminals):
 out={r['rid']:[('direct_gen' if r['native_first_command']['line'].startswith('GEN ') else 'admission')]+(['later'] if phase['native_multirow_required'] else []) for r in work}
 if phase['cancel_kind']=='decode':
  ends=list(terminals.values());require(len(ends)==2 and sum(e['actual_client_cancelled'] is True for e in ends)==1,'Exactly one actual client cancellation and surviving owner required')
  survivors=[e for e in ends if e['actual_client_cancelled'] is False];require(len(survivors)==1 and survivors[0]['rid'] in out,'Actual surviving request RID missing');out[survivors[0]['rid']].append('solo_migration')
 return out

def collect(trace,events,work,stages,capture_root,required_roles):
 require(not Path(capture_root).is_symlink(),'Raw root symlink');root=Path(capture_root).resolve();jobs=prefix_jobs(events);calls={r['call'] for r in work};byrid={r['rid']:r for r in work};jobs=[j for j in jobs['jobs'] if j['call'] in calls];jobmap={(j['rid'],j['role']):j for j in jobs};replays={};rows={};vectors={};seen=set();used=0;epochs=set();owners={l:s for s,lo,hi in stages for l in range(lo,hi)};requests={};resumes={};spans=[]
 for number,line in enumerate(trace.splitlines(),1):
  if not line.startswith('SBF '):continue
  kind=line.split()[1];f=fields(line)
  if kind=='bound':raise ValueError('Actual source observer bound exceeded')
  if kind in ('request','direct_gen_begin'):
   rid=int(f['rid']);require(rid in byrid and rid not in requests and int(f['pid'])==byrid[rid]['pid'] and int(f['prompt'])==len(byrid[rid]['input_ids']) and int(f['slotgen'])==byrid[rid]['slotgen'],'Actual current request descriptor mismatch');requests[rid]=f;continue
  if kind in ('resume','span'):
   rid=int(f['rid']);require(rid in requests and all(f[k]==requests[rid][k] for k in ('pid','enginegen','requestgen','slotgen')),'Actual source span/resume request ownership differs')
   if kind=='resume' and (f['slot']!='6' or requests[rid].get('normal_dispatch')=='GEN'):require(rid not in resumes,'Duplicate actual admission resume');resumes[rid]=f
   if kind=='span':require((int(f['stage']),int(f['lb']),int(f['le'])) in stages and f['completed']=='1' and 0<=int(f['begin'])<int(f['end']),'Actual source stage span invalid');spans.append(f)
   continue
  if kind=='replay':
   key=(int(f['pid']),int(f['stage']),int(f['epoch']));require(key not in replays and key[2]>0 and (key[1],int(f['lb']),int(f['le'])) in stages,'Unique actual replay/stage ownership required');require(f['full_roster']==f['input_output_verified']==f['stage_context_verified']=='1' and f['geometry']=='48x4x2560x248320','Actual native graph row/stage seal missing');require(0<=int(f['graph'])<16 and int(f['enginegen'])>0,'Actual graph/engine identity invalid');replay_geometry(f);replays[key]=(number,f);continue
  if kind=='row':
   key=(int(f['pid']),int(f['stage']),int(f['epoch']));require(key in replays,'Actual row missing sealed replay');proof=replays[key][1];row=int(f['row']);require(0<=row<int(f['batchrows'])==int(proof['rows']) and int(f['selected']) in (0,1),'Actual row geometry invalid');require(all(f[k]==proof[k] for k in ('enginegen','graph','event','admission')),'Actual row/replay ownership mismatch');require(bool(int(proof['selected_mask'])&(1<<row))==bool(int(f['selected'])) and bool(int(proof['active_mask'])&(1<<int(f['slot']))),'Actual selected/active row mask mismatch');require(key+(row,) not in rows,'Duplicate actual graph row');rows[key+(row,)]=f;continue
  if kind!='vector':continue
  rid=int(f['rid'])
  if rid not in byrid:raise ValueError('Foreign actual vector RID within phase')
  role=vector_role(f['phase']);require(role is not None and (rid,role) in jobmap,'Actual vector lacks consumed-prefix job');job=jobmap[rid,role];owner=byrid[rid];layer=int(f['layer']);stage=int(f['stage']);key=(int(f['pid']),stage,int(f['epoch']));row=int(f['row']);require(key in replays and key+(row,) in rows,'Actual vector replay/row not observed');proof=replays[key][1];rowproof=rows[key+(row,)];require(all(f[k]==rowproof[k] for k in ('rid','enginegen','requestgen','slot','slotgen','pos','token','row','batchrows','event','graph')) and rowproof['selected']=='1','Actual selected GPU row/vector identity differs')
  require(rid in requests and all(f[k]==requests[rid][k] for k in ('pid','enginegen','requestgen','slotgen')) and int(f['pid'])==owner['pid'] and int(f['slotgen'])==owner['slotgen'] and int(f['enginegen'])>0 and int(f['requestgen'])>0,'Actual vector stale process/request/slot generation');require(-1<=layer<48 and stage==(stages[-1][0] if layer==-1 else owners[layer]) and (stage,int(f['lb']),int(f['le'])) in stages,'Actual layer/head stage owner differs');require(f['phase'] in ('admission_residual','admission_logits_before_sampler','batch_step_residual','batch_step_logits_before_sampler','solo_migration_residual','solo_migration_logits_before_sampler','direct_gen_residual','direct_gen_logits_before_sampler') and f['phase'].endswith('residual')==(layer>=0),'Actual phase/head role differs');require(int(f['pos'])==job['position'] and int(f['token'])==job['token'] and job['ids'][-1]==job['token'],'Actual consumed-prefix position/token differs');count=248320 if layer==-1 else 10240;require(f['canonical']=='le_f32' and int(f['bytes'])==count*4 and int(f['floats'])==count,'Actual canonical matrix/head extent differs')
  path=root/Path(f['file']).name;require(path.resolve().parent==root and not path.is_symlink() and path not in seen,'Actual raw path alias/duplication');before=path.stat();raw=path.read_bytes();require(path.read_bytes()==raw and path.stat()==before and len(raw)==count*4 and all(math.isfinite(x[0]) for x in struct.iter_unpack('<f',raw)),'Actual full raw matrix/head invalid');seen.add(path);used+=len(raw);require(used<=64<<20,'Perphase existing64MiB raw bound exceeded');group=vectors.setdefault((rid,role),{});require(layer not in group,'Actual duplicate layer/head');group[layer]={'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'source_line':number,'replay':key,'row':row,'position':job['position'],'token':job['token'],'ids':job['ids']}
 for key,group in vectors.items():require(set(group)==set(range(-1,48)),'Actual partial49 field roster');require(len({(v['position'],v['token'],v['row']) for v in group.values()})==1,'Actual crossstage selected row changed')
 for rid,roles in required_roles.items():
  for role in roles:require((rid,role) in vectors,'Actual required full49 role missing')
  require(rid in resumes,'Actual source admission counters missing');owner=byrid[rid];resume=resumes[rid];require(int(resume['reused'])==int(resume['read_from'])==owner['actual_reused'] and resume['reread_to']=='-1','Actual source read/reuse/reread counter differs')
  for stage,lo,hi in stages:
   intervals=[(int(s['begin']),int(s['end'])) for s in spans if int(s['rid'])==rid and int(s['stage'])==stage and (s['slot']!='6' or requests[rid].get('normal_dispatch')=='GEN') and s['phase'] in ('prefill_chunk','admission_target_verify','direct_gen_target_verify')];covered=[i for a,b in intervals for i in range(max(a,owner['actual_reused']),min(b,len(owner['input_ids'])))];require(sorted(covered)==list(range(owner['actual_reused'],len(owner['input_ids']))),'Actual complete source new-prompt stage spans overlap/missing')
 # Every selected row in each observed replay must have its complete owned
 # layer partition (and the final stage head), even if no policy requested it.
 for key,(_,proof) in replays.items():
  selected=int(proof['selected_mask']);actual={r for k,r in ((k[:3],k[3]) for k in rows) if k==key};require(actual==set(range(int(proof['rows']))),'Actual replay complete physical row roster missing');physical=[int(rows[key+(r,)]['slot']) for r in actual];require(len(set(physical))==len(physical) and sum(1<<r for r in physical)==int(proof['active_mask']),'Actual physical slot/mask roster differs')
  for row in range(int(proof['rows'])):
   if not selected&(1<<row):continue
   rp=rows[key+(row,)];group=vectors.get((int(rp['rid']),replay_role(int(proof['admission']))),{});expected=set(range(int(proof['lb']),int(proof['le'])))|({-1} if key[1]==stages[-1][0] else set());require({l for l,v in group.items() if tuple(v['replay'])==key and v['row']==row}==expected,'Actual selected replay partial stage outputs')
 return {'groups':[{'rid':rid,'role':role,'vectors':{str(k):v for k,v in group.items()},'input_ids':jobmap[rid,role]['ids']} for (rid,role),group in sorted(vectors.items())],'jobs':jobs,'actual_raw_fields':sum(len(g) for g in vectors.values()),'all_required_49_fields_observed':True,'fresh_numerical_control_still_required':True,'full_cache_runtime_qualified':False}
def compare49(cached,fresh):
 require(cached['input_ids']==fresh['input_ids'] and set(cached['vectors'])==set(fresh['vectors'])=={str(l) for l in range(-1,48)},'Actual cached/fresh complete49 or consumed prefix differs');rows=[]
 for layer in sorted(cached['vectors'],key=int):
  a,b=cached['vectors'][layer],fresh['vectors'][layer];require(a['bytes']==b['bytes'] and a['position']==b['position'] and a['token']==b['token'],'Actual cached/fresh selected position/token differs');pa,pb=Path(a['path']),Path(b['path']);sa,sb=pa.stat(),pb.stat();rawa=pa.read_bytes();rawb=pb.read_bytes();require(pa.read_bytes()==rawa and pb.read_bytes()==rawb and pa.stat()==sa and pb.stat()==sb and len(rawa)==a['bytes'] and len(rawb)==b['bytes'] and hashlib.sha256(rawa).hexdigest()==a['sha256'] and hashlib.sha256(rawb).hexdigest()==b['sha256'],'Actual comparison data changed during fresh read');rows.append({'layer':int(layer),'bitwise_equal':rawa==rawb})
 return {'rows':rows,'all49_bitwise_equal':all(r['bitwise_equal'] for r in rows),'tolerance_gate':None,'full_model_math_qualified':False}
