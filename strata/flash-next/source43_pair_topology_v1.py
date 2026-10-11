"""Exact existing32/16 geometry over fresh43 matched onecard software settings."""
import hashlib,re,time,json,collections
from pathlib import Path
import run_source43_normal_model_cli_v1 as base
from serial37_canonical_json_v3 import canonical
require=base.require;PHASES=('source43-one','source43-pair')

def recipe(phase,metadata,cfg,inputs,binding):
    require(phase in PHASES,'Exact actual43 one/pair topology required');descriptor,plan=base.phase_recipe('source43-off',metadata,cfg,inputs,binding);descriptor=dict(descriptor);descriptor['port']=metadata['port'];env=dict(plan['native_env']);args=list(plan['native_argv']);paired=phase==PHASES[1]
    require(not any(flag in args for flag in('--layer-split','--split-device','--trim-stage-weights','--mtp')),'Base topology/MTP must be absent')
    if paired:args+=['--layer-split','32','--split-device','1','--trim-stage-weights'];env['ZE_AFFINITY_MASK']='0,1';descriptor['cards']=[0,1]
    descriptor['alias']='source43-mode0-'+('split32-16'if paired else'one')+'-native-CLI-diagnostic'
    env.update(STRATA_FIDELITY_DIAG='1',STRATA_FIDELITY_DIAG_ACTIVATIONS='1',STRATA_FIDELITY_DIAG_ARM='/results/ARM',STRATA_FIDELITY_DIAG_DIR='/results/captures',STRATA_PREFIX_DIAG='1',STRATA_PREFIX_LIFECYCLE_DIAG='1',STRATA_PREFIX_DIAG_ARM='/results/ARM',STRATA_ARTIFACT_IDENTITY_SHA256=binding)
    require(env.get('SYCL_UR_TRACE','0')=='0'and env.get('STRATA_QSA3_TARGET','0')=='0'and env.get('STRATA_LAYER0_Q8_DIAG','0')=='0','Extra UR/QSA/NUM10 observers forbidden')
    plan.update(mode=phase,native_argv=args,native_env=env,cards=[0,1]if paired else[0],stages=[[0,0,32],[1,32,48]]if paired else[[0,0,48]],observer_enabled=True,full49_first_head_and48_residuals=True,physical_buffer_release_qualified=False)
    return descriptor,plan

def memory_admission(directory,metadata,plan):
    if plan['cards']==[0]:return base.life.prepared_ctrl.memory_admission(directory,metadata)
    path=Path(directory)/'server-engine.log';require(path.is_file()and not path.is_symlink()and path.stat().st_size<=16<<20,'Original bounded pair startup required');raw=path.read_bytes();text=raw.decode('ascii')
    require(plan['cards']==[0,1]and plan['stages']==[[0,0,32],[1,32,48]],'Exact32/16 pair startup scope required')
    require(text.count('strata generate: layer split: CUDA0 runs layers 0-31\n')==1,'Actual primaryGPU32-layer constructor missing')
    primary=re.findall(r'strata generate: expert cache (\d+) slots,',text);secondary=re.findall(r'strata generate: layer split: CUDA1 runs layers 32-47, expert cache (\d+) slots .*slot 0 verified\n',text);require(len(primary)==len(secondary)==1,'Actual verified secondaryGPU16-layer cache missing')
    mirrors=re.findall(r'stage mirror: layers(\d+)-(\d+) missing(\d+) mirrored(\d+) bytes(\d+) ',text);require(len(mirrors)==2 and [(int(r[0]),int(r[1]))for r in mirrors]==[(0,31),(32,47)],'Actual exact two-stage mirror ownership missing/duplicated')
    details=[]
    for stage,(row,resident,total)in enumerate(zip(mirrors,map(int,[primary[0],secondary[0]]),[16384,8192])):
        _,_,missing,mirrored,size=map(int,row);require(0<resident<=total and missing==mirrored==total-resident and(size==0)==(mirrored==0),'Actual complete stage cache+mirror coverage differs');details.append({'stage':stage,'resident_experts':resident,'mirror_experts':mirrored,'mirror_bytes':size})
    free=re.findall(r'strata serve: (\d+) MiB of VRAM free with everything loaded\n',text);require(len(free)==1 and int(free[0])>=1024,'Current-device startup freeheadroom insufficient')
    require('the prompt path allocates its own buffers'in text and not re.search(r'trying a smaller expert cache|trying a \d+-token chunk| - LOW:',text),'Capacity fallback/borrowing forbidden')
    value={'schema':1,'epoch':time.time(),'log_path':str(path),'log_prefix_sha256':hashlib.sha256(raw).hexdigest(),'stages':details,'observed_current_device_free_VRAM_MiB':int(free[0]),'both_device_headroom_qualified':False,'both_GPU_constructor_ranges_observed':True,'physical_transfer_bytes_qualified':False};base.c.write(Path(directory)/'chunk-memory-admission.json',value);return value

def numeric(raw,directory,stages):
    import serial_prefix_qualification_v6 as original
    expected=[(r[1],r[2])for r in stages];require(expected in([(0,48)],[(0,32),(32,48)]),'Exact fresh43 topology only');require(type(raw['fresh'])is int and raw['fresh']==1 and raw['pin']is None and raw['cancel_requested']is None and raw['stop_sent']is False,'Fresh absentPIN/no cancellation required')
    events=[json.loads(line[4:])for line in raw['stderr']if line.startswith('PCL ')];begin=[e for e in events if e['event']=='begin'];require(len(begin)==1 and begin[0]['tokens']==len(raw['ids'])and begin[0]['input_sha256_le32']==original.le32_digest(raw['ids']),'Actual PCL input/owner absent')
    spans=[e for e in events if e['event']=='stage_span'];require(spans and all(e['pid']==begin[0]['pid']and e['request']==begin[0]['request']and type(e['device'])is int and type(e['complete'])is bool and(e['lb'],e['le'])in expected and e['device']==expected.index((e['lb'],e['le']))for e in spans),'Actual perGPU stage source ownership differs')
    key=lambda e:(e['phase'],e['device'],e['lb'],e['le'],e['lo'],e['hi']);require(collections.Counter(key(e)for e in spans if not e['complete'])==collections.Counter(key(e)for e in spans if e['complete'])and {e['device']for e in spans if e['complete']}==set(range(len(expected))),'Actual GPU stage entry/return missing')
    return original.extract_numeric(raw,Path(directory)/'captures',True,True,expected)
