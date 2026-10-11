"""Owned input fixture and bounded direct/replay storage admission."""
import hashlib,json,re,struct,os
from pathlib import Path
import numpy as np
from serial37_canonical_json_v3 import read_unique,canonical
from owned_layer3_qsa_contract_v1 import require
from owned_layer3_qsa_helper_contract_v3 import load_own
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
BASE=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010')
OWN=BASE/'owned-layer3-qsa-producer-v4-run-v2'
ADMISSION=BASE/'postboot-original-qsa-reference-admission-v4-current173.json'
STEP_BYTES=struct.pack('<16i',0,1,0,1,1,2,0,2,2,3,0,3,3,4,1,4)
MODES=('single_raw','single_materialized','steps_raw','steps_materialized','batch_raw','batch_materialized')
FIELDS={'input.f32':2048,'gamma.f32':512,'materialized.f16':1024,'expanded.f32':2048,'steps.i32':64,'wrong_raw_store.f32':1536,'before_states.bin':9228,'after_states.bin':9228}
CONFIG=('INDEXER40_MATERIALIZED_V1\nids 248045 8678 198 15666\ngeometry 24 2 256 4 128 64 4 2048\nrope none 10000000 1 0 262144 0 1 32 1\nepsilon source_qsa_rms_eps\nwindows 0:2 2:1 3:1\norigin independent_original_projection_only\n').encode('ascii')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def consume(path,size):
 path=Path(path);require(path.is_file()and not path.is_symlink()and path.resolve()==path,'Exact regular source/output required')
 def signature():
  st=path.stat();return (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
 before=signature();require(before[2]==size and size<=64<<20,'Exact bounded output required');raw=path.read_bytes();require(len(raw)==size and signature()==before and path.read_bytes()==raw and signature()==before,'Consumed source/output bytes or identity changed');return raw

def _full_producer_binding(root):
 root=Path(root).resolve();require(root==OWN,'Exact original four-row own producer required');saved=read_unique(ADMISSION)
 from admit_postboot_original_qsa_reference_v4 import finalized_binding
 a=saved['model_identity_association'];current=finalized_binding(root,a['historical_identity_path'],a['current_identity_path'],a['current_identity_sha256'],saved['fresh_native_runtime_authorization']['baseline_root']);require(canonical(current)==canonical(saved),'Authentic current postboot original admission changed')
 require(current['original_reference']['original_binding']['report_sha256']=='61041a2990c6f1b172ff1e8de8f9187805ac4279349e9082fd48136816c31ab3','Exact original producer report required')
 return {'root':str(root),'admission_path':str(ADMISSION),'admission_sha256':sha(ADMISSION),'current_postboot_reference':current,'captured_operands_used':False,'full_model_math_qualified':False}

_operation=None
def producer_binding(root,role=None):
 """One full semantic admission per owned process, current full bytes per reuse."""
 global _operation
 root=Path(root).resolve();require(root==OWN,'Exact original own producer required');require(role in(None,'predevice','postrun','postproof'),'Exact current-use role required')
 from native_qsa40_byte_epoch_v9 import ByteEpoch
 from qualify_indexer_materialized41_masked_v2 import PLAN
 if _operation is None:
  from admit_postboot_original_qsa_reference_v4 import source_binding as nested_source_preflight
  nested_source_preflight()
  require(role is None,'Role boundary requires original entry/ready admission');saved=read_unique(ADMISSION);current_root=Path(saved['fresh_native_runtime_authorization']['baseline_root']);prepared=read_unique(current_root/'prepared.json');model_lock=read_unique(HERE/'model-lock.json');shards=[ROOT/model_lock['destination']/f['path']for f in model_lock['files']if f['path'].startswith('UD-Q4_K_XL/')]
  model_plan={'prepared':str(current_root),'prepared_sha256':sha(current_root/'prepared.json'),'engine_root':str(Path(prepared['engine_receipt']).parent),'pack':prepared['pack'],'own_producer_root':str(root),'own_producer_binding':{'report_sha256':sha(root/'report.json')},'current_admission':str(ADMISSION)}
  source=[ROOT/p if not Path(p).is_absolute()else Path(p)for p in read_unique(PLAN)['files']]+[PLAN]
  from source41_sdk_binding_v1 import SDK
  sdk_receipt=read_unique(SDK/'receipt.json');sdk_plan=read_unique(SDK/'plan.snapshot.json');source +=[SDK/'receipt.json',SDK/'plan.snapshot.json',SDK/'build/CMakeCache.txt']+[SDK/'source'/n for n in sdk_plan['expected_patched_source_sha256']]+[Path(n)for n in sdk_receipt['binary_sha256']]+[SDK/'build'/('libstrata_'+n+'.a')for n in('engine','prefill','spec','kernels','core','kernels_cpu')]+[SDK/'build/ggml/src/libggml-cpu.a',SDK/'build/ggml/src/libggml-base.a']
  epoch=ByteEpoch(model_plan,source,shards);proof=_full_producer_binding(root);epoch.ready();_operation={'owner_pid':os.getpid(),'root':root,'proof':proof,'epoch':epoch,'calls':1}
 else:
  require(_operation['owner_pid']==os.getpid()and _operation['root']==root,'No previous actor semantic witness can transfer');_operation['calls']+=1;_operation['epoch'].fresh('current-use-'+str(_operation['calls'])+(':'+role if role else ''))
 return _operation['proof']

def operation_witness():
 require(_operation is not None and _operation['owner_pid']==os.getpid(),'Actual owned source operation witness required');epoch=_operation['epoch'];return {'schema':1,'owner_pid':os.getpid(),'one_full_semantic_admission_per_owned_process':True,'whole_current_input_bytes_on_each_use':True,'generic_persistent_cached_PASS_used':False,'calls':_operation['calls'],'roots':[str(p)for p in epoch.roots],'files':[str(p)for p in epoch.files],'boundaries':epoch.boundaries,'full_model_math_qualified':False}

def admit_operation_witness(root,report):
 root=Path(root).resolve();path=root/'producer-operation-witness.json';digest=sha(path)
 require(digest==report['producer_operation_witness_sha256']==report['artifact_sha256'].get('producer-operation-witness.json'),'Original operation witness artifact/hash missing');saved=read_unique(path)
 require(type(saved['schema'])is int and saved['schema']==1 and type(saved['owner_pid'])is int and type(report['producer_pid'])is int and saved['owner_pid']==report['producer_pid'] and saved['owner_pid']>0 and saved['one_full_semantic_admission_per_owned_process']is True and saved['whole_current_input_bytes_on_each_use']is True and saved['generic_persistent_cached_PASS_used']is False and saved['full_model_math_qualified']is False,'Original process-scoped semantic witness scope differs')
 boundaries=saved['boundaries'];require(type(saved['calls'])is int and saved['calls']==6 and type(boundaries)is list and len(boundaries)==saved['calls']+1 and [b['label']for b in boundaries[:2]]==['entry','ready'],'Exact original semantic entry/ready/count bracket missing')
 require([b['label']for b in boundaries]==['entry','ready','current-use-2','current-use-3:predevice','current-use-4:postrun','current-use-5','current-use-6:postproof'],'Exact successful original six-call byte seal roster required')
 require(_operation is not None and _operation['owner_pid']==os.getpid(),'New reader must independently admit same original predicates once')
 require(saved['roots']==[str(p)for p in _operation['epoch'].roots]and saved['files']==[str(p)for p in _operation['epoch'].files],'Exact original complete byte roster differs');last=0.;roles={}
 def epoch(value):
  require(type(value)in(int,float)and np.isfinite(value)and value>=0,'Finite typed actual epoch required');return value
 for index,b in enumerate(boundaries):
  require(type(b['owner_pid'])is int and b['owner_pid']==report['producer_pid'],'Original byte owner differs')
  begin=epoch(b['started_epoch']);finish=epoch(b['finished_epoch']);require(last<=begin<=finish,'Original byte chronology differs');last=finish
  require(canonical(b['view'])==canonical(boundaries[0]['view'])==canonical(_operation['epoch'].initial),'Original/current complete evidence bytes differ')
  if index>=2:
   prefix='current-use-'+str(index);label=b['label'];require(type(label)is str and (label==prefix or label in[prefix+':'+r for r in('predevice','postrun','postproof')]),'Exact ordered current-use label required')
   if ':'in label:
    role=label.split(':',1)[1];require(role not in roles,'Duplicate original current-use role');roles[role]=b
 require(set(roles)=={'predevice','postrun','postproof'} and boundaries[-1]is roles['postproof'],'Complete original predevice/postrun/final-postproof roster required')
 require(epoch(boundaries[1]['finished_epoch'])<=epoch(report['started_epoch'])and last<=epoch(report['finished_epoch']),'Semantic bracket/actual report chronology differs')
 before=roles['predevice'];after=roles['postrun'];final=roles['postproof']
 require(epoch(report['compile_command']['finished_epoch'])<=before['started_epoch']<=before['finished_epoch']<=epoch(report['run_command']['started_epoch']),'Actual compile/predevice/runtime byte bracket differs')
 require(epoch(report['run_command']['finished_epoch'])<=after['started_epoch']<=after['finished_epoch']<=epoch(report['GPU_terminal_epoch']),'Actual runtime EOF/postrun/GPU terminal byte bracket differs')
 require(epoch(report['post_full4']['finished'])<=epoch(report['post_pages']['epoch'])<=final['started_epoch']<=final['finished_epoch']<=epoch(report['finished_epoch']),'Actual newfour/pages/final byte bracket differs')
 return {'original_operation_byte_roster_recollected':True,'original_predevice_postrun_postproof_recollected':True,'generic_persistent_cached_PASS_used':False,'full_model_math_qualified':False}

def own_params(root):
 work=read_unique(Path(root)/'reference-work/work-report.json');metadata=work['own_layer3_projection']['metadata_binding'];expected={'type':'none','freq_base':1e7,'factor':1.,'freq_scale_in':0.,'orig_ctx':262144.,'ext_factor':0.,'attn_factor':1.,'beta_fast':32.,'beta_slow':1.}
 require(metadata['effective_epsilon_F32_hex']==struct.pack('<f',1e-6).hex()and canonical(metadata['effective_RopeScaling'])==canonical(expected),'Original resolved epsilon/RopeScaling differs from actual leaf defaults')
 return {'effective_epsilon_F32_hex':metadata['effective_epsilon_F32_hex'],'effective_RopeScaling':metadata['effective_RopeScaling'],'original_metadata_binding':metadata,'source':'original metadata plus pinned native source defaults; not CONFIG authority'}

def own_inputs(root):
 root=Path(root).resolve();parent=read_unique(root/'report.json');work=read_unique(root/'reference-work/work-report.json');require(canonical(work)==canonical(parent['reference_phase']['work']),'Original work/report association changed');own=work['own_layer3_projection'];identity=sha(parent['original_work_config']['model_identity']);rows=[];gammas=[]
 for pos,row in enumerate(own['records']):
  require(row['position']==pos and row['token']==(248045,8678,198,15666)[pos]and row['captured_inputs_or_states_used']is False,'Exact own four-token source roster required')
  rows.append(load_own(row['projections'][3],own['root'],(128,),'blk.3.indexer.k_proj.weight',pos,identity,True));gammas.append(load_own(row['vectors'][3],own['root'],(128,),'blk.3.indexer.k_norm.weight',pos,identity,True))
 require(len(rows)==4 and all(g.tobytes()==gammas[0].tobytes()for g in gammas),'Complete original rows/shared gamma required')
 raw=np.stack(rows).astype('<f4').tobytes();gamma=gammas[0].astype('<f4').tobytes();half,widen=ieee_rne(raw);require(widen[:1536]!=raw[:1536],'Own fixture must discriminate unrounded storage');return raw,gamma

def ieee_rne(raw):
 require(type(raw)is bytes and len(raw)%4==0,'Exact LE_F32 bytes required');words=struct.unpack('<%dI'%(len(raw)//4),raw);bits=[]
 for b in words:
  require((b>>23)&255!=255,'Finite preregistered input required');sign=(b>>16)&0x8000;e=((b>>23)&255)-112;m=b&0x7fffff
  if e>=31:out=sign|0x7c00
  elif e<=0:
   if e<-10:out=sign
   else:
    m|=1<<23;shift=14-e;q=m>>shift;r=m&((1<<shift)-1);h=1<<(shift-1);q+=int(r>h or(r==h and q&1));out=sign|q
  else:
   q=m>>13;r=m&8191;q+=int(r>4096 or(r==4096 and q&1))
   if q==1024:e+=1;q=0
   out=sign|(e<<10)|q
  bits.append(out)
 half=struct.pack('<%dH'%len(bits),*bits);widen=b''.join(struct.pack('<f',struct.unpack('<e',half[i:i+2])[0])for i in range(0,len(half),2));return half,widen

def prepare(root,out):
 from qualify_indexer_materialized41_masked_v2 import source_binding,PLAN
 before=producer_binding(root);raw,gamma=own_inputs(root);params=own_params(root);out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
 for name,data in [('config.txt',CONFIG),('raw.f32',raw),('gamma.f32',gamma)]:
  with(out/name).open('xb')as f:f.write(data)
 record={'schema':1,'own_producer_root':str(Path(root).resolve()),'own_producer_binding':before,'source_binding':source_binding(),'source_plan_sha256':sha(PLAN),'resolved_own_params':params,'raw_sha256':hashlib.sha256(raw).hexdigest(),'gamma_sha256':hashlib.sha256(gamma).hexdigest(),'captured_operands_used':False,'full_model_math_qualified':False};(out/'source-plan.snapshot.json').write_bytes(PLAN.read_bytes());(out/'input-binding.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n');require(canonical(producer_binding(root))==canonical(before),'Original admission changed during extraction');return record

def fixture_binding(root):
 from qualify_indexer_materialized41_masked_v2 import source_binding,PLAN
 root=Path(root).resolve();require({p.name for p in root.iterdir()}=={'config.txt','raw.f32','gamma.f32','source-plan.snapshot.json','input-binding.json'}and all(p.is_file()and not p.is_symlink()for p in root.iterdir()),'Exact fixture roster required');record=read_unique(root/'input-binding.json');require(type(record['schema'])is int and record['schema']==1 and record['captured_operands_used']is False and record['full_model_math_qualified']is False,'Exact fixture scope required');before=producer_binding(record['own_producer_root']);require(canonical(before)==canonical(record['own_producer_binding'])and record['source_plan_sha256']==sha(PLAN)and(root/'source-plan.snapshot.json').read_bytes()==PLAN.read_bytes()and canonical(record['source_binding'])==canonical(source_binding()),'Current fixture/source/prior differs');raw,gamma=own_inputs(record['own_producer_root']);require(canonical(record['resolved_own_params'])==canonical(own_params(record['own_producer_root'])),'Exact original resolved parameters changed');require(consume(root/'raw.f32',2048)==raw and consume(root/'gamma.f32',512)==gamma and consume(root/'config.txt',len(CONFIG))==CONFIG and record['raw_sha256']==hashlib.sha256(raw).hexdigest()and record['gamma_sha256']==hashlib.sha256(gamma).hexdigest(),'Exact own source operand bytes differ');return {'root':str(root),'fixture_sha256':sha(root/'input-binding.json'),'record':record}

def native_trace(path):
 text=Path(path).read_text();lines=[s for s in text.splitlines(keepends=True)if s.startswith('IDXMAT40_')];require(all(s.endswith('\n')for s in lines),'Complete marker newline required');lines=[s[:-1]for s in lines];expected=[f'IDXMAT40_FRAME mode={m} route={r} graph_replay={r} fields=8 own_zero_restored=1'for m in MODES for r in range(3)];require(len(lines)==22 and re.fullmatch(r'IDXMAT40_DEVICE backend=level_zero affinity=0 fp16=1 vendor=.+ driver=.+ name=.+',lines[0])and re.fullmatch(r'IDXMAT40_FP_CONFIG flags=(?:[0-9]+,)+ observed_device_flags_only=1 compiler_lowering_unobserved=1',lines[1])and lines[2]=='IDXMAT40_CONFIG actual_resolved_checked=1 rows=4 dim=128 windows=2,1,1 source_functions_unchanged=1'and lines[3:21]==expected and lines[21]=='IDXMAT40_RESULT frames=18 graph_retired=1 owned_allocations_freed=1 full_model_math_qualified=0 semantic_model_fix_qualified=0'and not re.search(r'IDXMAT40_ERROR|OWNED_ERROR',text),'Exact source marker/error roster required');return {'frames':18,'source_functions_unchanged':True,'semantic_model_fix_qualified':False}

def expected_tail(raw):
 a=np.frombuffer(raw,dtype='<f4').reshape(4,128);tails=[];state=np.zeros((3,128),dtype='<f4')
 for p,n in ((0,2),(2,1),(3,1)):
  for i in range(p,p+n):
   if i%4<3:state[i%4]=a[i]
  tails.append(state.tobytes())
 return tails

def compare_routes(root,fixture,prior):
 root=Path(root);require(canonical(prior)==canonical(fixture['record']['own_producer_binding']),'Exact independent prior required');raw=consume(Path(fixture['root'])/'raw.f32',2048);gamma=consume(Path(fixture['root'])/'gamma.f32',512);half,widen=ieee_rne(raw);require(widen[:1536]!=raw[:1536],'Preregistered wrong raw-store discrimination required');native_trace(root.parents[1]/'runtime.log');require({p.name for p in root.iterdir()}=={'input_echo',*(m+'-route'+str(r)for m in MODES for r in range(3))},'Exact native output root roster required');echo=root/'input_echo';require({p.name for p in echo.iterdir()}=={'config.txt','raw.f32','gamma.f32'},'Exact input echo roster required');require(consume(echo/'config.txt',len(CONFIG))==CONFIG and consume(echo/'raw.f32',2048)==raw and consume(echo/'gamma.f32',512)==gamma,'Actual consumed source inputs differ');reports={};states={}
 for mode in MODES:
  all_routes=[]
  for route in range(3):
   frame=root/(mode+'-route'+str(route));require(frame.is_dir()and not frame.is_symlink()and {p.name for p in frame.iterdir()}==set(FIELDS),'Exact frame roster required');data={n:consume(frame/n,s)for n,s in FIELDS.items()};require(data['wrong_raw_store.f32']==raw[:1536]and data['wrong_raw_store.f32']!=widen[:1536]and data['input.f32']==raw and data['gamma.f32']==gamma and data['steps.i32']==STEP_BYTES and data['materialized.f16']==half and data['expanded.f32']==widen,'Actual RTE half-store/widen/echo contract failed');before=data['before_states.bin'];after=data['after_states.bin'];
   for state_bytes in (before,after):
    for i in range(3):
     require(np.isfinite(np.frombuffer(state_bytes[i*3076:i*3076+3072],dtype='<f4')).all()and struct.unpack('<i',state_bytes[i*3076+3072:i*3076+3076])[0]==0,'Finite bounded complete state/blockposition required')
   require(before[:3076]==bytes(3076)and before[3076:6152]==after[:3076]and before[6152:]==after[3076:6152],'Exact own zero/chronological state joins required');wanted=expected_tail(widen if'materialized'in mode else raw);actual=[after[i*3076:i*3076+1536]for i in range(3)]
   if'materialized'in mode:require(actual==wanted,'Materialized input must produce exact source-RTE tail')
   all_routes.append(data)
  require(all_routes[0]==all_routes[1]==all_routes[2],'Complete direct/two-replay raw state equality required');states[mode]=all_routes[0]['after_states.bin'];reports[mode]={'tail_matches_RTE':all_routes[0]['after_states.bin'][6152:7688]==expected_tail(widen)[-1],'tail_matches_unrounded':all_routes[0]['after_states.bin'][6152:7688]==expected_tail(raw)[-1],'fields':{n:{'sha256':hashlib.sha256(v).hexdigest(),'bytes':len(v)}for n,v in all_routes[0].items()}}
 require(states['single_materialized']==states['steps_materialized']==states['batch_materialized'],'Complete materialized single/step/batch source equivalence required');return {'modes':reports,'all_direct_two_replays_bitwise':True,'materialized_half_bytes_and_expansion_bitwise_IEEE_RNE':True,'materialized_tail_bitwise_contract':True,'complete_materialized_all3_API_state_bitwise':True,'wrong_raw_store_control_discriminated':all(reports[m]['fields']['wrong_raw_store.f32']['sha256']==hashlib.sha256(raw[:1536]).hexdigest()for m in MODES)and raw[:1536]!=widen[:1536],'captured_operands_used':False,'semantic_model_fix_qualified':False,'full_model_math_qualified':False,'tolerance_used':None,'compiler_cause_qualified':False}
