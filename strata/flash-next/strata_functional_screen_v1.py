#!/usr/bin/env python3
"""Two-prompt source35 native semantic screen; no fullquality/arithmetic authority."""
import argparse,hashlib,importlib.util,json,os,shlex,subprocess,sys,time,traceback
from pathlib import Path
import serial_prefix_qualification_v8 as serial
import batch_numerical_proofs_v7 as proof
from source_page_watchdog_v3 import guard as page_guard
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
c1=serial.c1;sha=proof.sha;read=proof.read;write=proof.write;require=proof.require
lane_contract=proof.lane_contract;expected_stage_ranges=serial.expected_stage_ranges
CPU_RUNNER=ROOT/'llamacpp/flash-next/qualify_cpu_functional_pilot_v3.py'
CPU_PLAN=ROOT/'llamacpp/flash-next/cpu-functional-pilot-source-plan-v3.json'
SOURCE_PLAN=HERE/'strata-functional-screen-source-plan-v1.json'

def criteria():
 plan=read(SOURCE_PLAN);require(sha(CPU_RUNNER)==plan['files'][str(CPU_RUNNER.relative_to(ROOT))],'Frozen CPU semantic checker changed');spec=importlib.util.spec_from_file_location('cpu_functional_criteria_only',CPU_RUNNER);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def fixture_gate(cpu_plan,fixtures):
 criteria().fixture_gate(cpu_plan,fixtures);require(len(fixtures['fixtures'])==2,'Exactly two CPU prompts required')
 require(fixtures.get('actual_GPU_touch') is False and fixtures.get('actual_model_payload_read') is False,'CPU metadata fixture provenance differs')
 for row in fixtures['fixtures']:require(all(type(i) is int and 0<=i<248320 for i in row['ids']),'Fixture outside exact native vocabulary')
 return True

def cpu_run_binding(fixture_path):
 declared=read(SOURCE_PLAN);require(Path(fixture_path).resolve()==Path(declared['actual_CPU_fixture_path']).resolve() and sha(fixture_path)==declared['actual_CPU_fixture_sha256'],'Exact successful CPU tokenfixture path/hash required');root=Path(fixture_path).resolve().parent;report_path=root/'report.json';report=read(report_path);fixture=read(fixture_path);cpu=criteria();fixture_gate(read(CPU_PLAN),fixture)
 require(sha(report_path)==read(SOURCE_PLAN)['actual_CPU_report_sha256'],'Pinned actual successful CPU report changed');require(report.get('passed') is True and report.get('errors')==[] and all(report.get(k) is True for k in ('functional_and_fresh_process_repeat_passed','independent_output_token_decode_passed','post_terminal_full_four_passed','actual_CPU_inference_observed')),'Genuine successful CPU functional report required')
 require(report['source_plan_sha256']==sha(CPU_PLAN)==sha(root/'source-plan.snapshot.json') and len(report['cases'])==4,'CPU source/actual fourcase association differs')
 for row,(case,repeat) in zip(report['cases'],[(0,0),(0,1),(1,0),(1,1)]):
  require(row['case']==case and row['repeat']==repeat and row['passed'] is True and not row['errors'] and row['container_terminal_observed'] is True and row['container_removed'] is True and row['terminal']['ExitCode']==0 and not row['terminal']['Running'] and not row['terminal']['OOMKilled'],'CPU actual freshcase/owned terminal differs')
  directory=root/('case%d-repeat%d'%(case,repeat))
  for name,want in row['file_sha256'].items():require(sha(directory/name)==want,'CPU actual case artifact changed '+name)
  require(read(directory/'completion-response.json')==row['response'],'CPU stored actual response changed');item=fixture['fixtures'][case];require(read(directory/'accepted-input-ids.json')==item['ids'],'CPU actual accepted tokenIDs differ from shared fixture');cpu.response_gate(row['response'],item['ids'],case,item['rendered'])
 for case in (0,1):
  first,second=report['cases'][case*2:case*2+2];require(first['response']['tokens']==second['response']['tokens'] and first['response']['content']==second['response']['content'],'CPU actual deterministic repeat changed')
 require(all(report.get(k) is False for k in ('full48_math_qualified','registered_quality_qualified','native_bitwise_authority','speed_qualified','concurrency_qualified')),'CPU functional evidence scope inflated')
 return {'report_path':str(report_path),'report_sha256':sha(report_path),'fixture_sha256':sha(fixture_path),'input_prompt_lengths':[len(row['ids']) for row in fixture['fixtures']],'functional_fresh_repeats_observed':True,'full_model_math_qualified':False,'native_bitwise_authority':False}

def verify_model_identity(path,lock,shards):
 value=serial.verify_model_identity(path,lock,shards);value['current_known_pages']=page_guard(shards[2]);return value

def config_contract(cfg):
 args=list(cfg['args']);env=dict(cfg['env'])
 for key,value in serial.GEOMETRY.items():serial.option(args,key,value)
 env.update(STRATA_FIDELITY_DIAG='0',STRATA_FIDELITY_DIAG_ACTIVATIONS='0',STRATA_LAYER0_Q8_DIAG='0',STRATA_PLE_INPUT33='0',STRATA_PREFIX30='0')
 return args,env

def native_command(plan,directory,pid):
 directory=Path(directory).resolve();binding=sha(directory.parent/'plan.snapshot.json');env=dict(plan['env']);env.update(STRATA_PREFIX_DIAG='1',STRATA_PREFIX_DIAG_ARM='/results/ARM',STRATA_PREFIX_LIFECYCLE_DIAG='0',STRATA_ARTIFACT_IDENTITY_SHA256=binding);engine=Path(plan['engine_root']);container='b70-prefix-'+str(pid)+'-functional';model=ROOT/read(HERE/'model-lock.json')['destination'];command=['docker','run','-i','--name',container,'--label','b70.prefix.plan='+binding,'--network','none','--device','/dev/dri','--user','1000:1000','--memory','105g','--memory-swap','105g','--group-add',str(os.stat('/dev/dri/renderD128').st_gid),'-v',str(engine/'build')+':/build:ro','-v',str(engine/'source')+':/src:ro','-v',plan['pack']+':/pack:ro','-v',str(model)+':/model:ro','-v',str(directory)+':/results']
 for key,value in sorted(env.items()):command+=['-e',key+'='+value]
 command+=[plan['image'],'cd /src; source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; exec '+shlex.join(['/build/strata','--serve']+plan['args'])]
 return command,container

def manifest_binding(plan):
 require(__debug__ and sys.flags.optimize==0 and os.environ.get('PYTHONOPTIMIZE','0') in ('','0'),'Functional screen requires strict assertions enabled')
 require(plan.get('screen_generation')==1 and plan.get('kind')=='functional' and plan['max_new']==64 and plan['fresh']==1 and plan['primary_model_name']=='hotschmoe-dd','Declared functional natural64/fresh identity differs')
 source=read(SOURCE_PLAN)
 for name,want in source['files'].items():require(sha(ROOT/name)==want,'Frozen functional dependency changed '+name)
 require(plan['driver_sha256']==sha(Path(__file__)) and plan['source_plan_sha256']==proof.PLAN_SHA,'Functional source generation changed')
 require(sha(plan['cpu_fixture_path'])==plan['cpu_fixture_sha256'] and sha(CPU_PLAN)==plan['cpu_source_plan_sha256'],'CPU prompt/fixture identity changed');fixture=read(plan['cpu_fixture_path']);fixture_gate(read(CPU_PLAN),fixture);require(plan['fixtures']==fixture['fixtures'],'Actual functional accepted IDs changed')
 require(cpu_run_binding(plan['cpu_fixture_path'])==plan['CPU_functional_input_binding'],'Completed CPU evidence changed');metadata,binding=proof.genuine_baseline(Path(plan['prepared']),plan['lane']);require(binding==plan['baseline_source_proof'] and metadata['engine_receipt_sha256']==plan['engine_receipt_sha256'] and metadata['cards']==plan['cards'] and Path(metadata['engine_receipt']).resolve().parent==Path(plan['engine_root']).resolve(),'Current genuine source35 C113/topology differs')
 require(sha(Path(plan['prepared'])/'prepared.json')==plan['prepared_sha256'] and fixture['tokenizer_file_sha256']==read(Path(plan['prepared'])/'artifact-identity.json')['tokenizer_files'],'Source35/CPU exported tokenizer identity differs')
 exact_config_gate(plan,metadata,read(Path(plan['prepared'])/'server-config.json'));serial.geometry_contract(plan['args']);serial.observer_env_contract(plan['env']);require(all(plan['env'].get(k,'0')=='0' for k in ('STRATA_FIDELITY_DIAG','STRATA_FIDELITY_DIAG_ACTIVATIONS','STRATA_LAYER0_Q8_DIAG','STRATA_PLE_INPUT33','STRATA_PREFIX30')),'Functional observerOFF route required')
 return binding

def exact_config_gate(plan,metadata,cfg):
 expected_args,expected_env=config_contract(cfg);require(plan['args']==expected_args and plan['env']==expected_env and plan['image']==metadata['runtime']['image'] and plan['pack']==metadata['pack'],'Exact functional prepared argv/env/image/pack recipe differs')
 return True

def prepare(a):
 fixture=read(a.cpu_fixtures);fixture_gate(read(CPU_PLAN),fixture);cpu_binding=cpu_run_binding(a.cpu_fixtures);metadata,binding=proof.genuine_baseline(a.prepared,'source35');cfg=read(a.prepared/'server-config.json');args,env=config_contract(cfg)
 identity=verify_model_identity(a.model_identity,read(HERE/'model-lock.json'),[Path(row['path']) for row in metadata['model_shards']]);plan={'schema':1,'screen_generation':1,'kind':'functional','lane':'source35','driver_sha256':sha(Path(__file__)),'source_plan_sha256':proof.PLAN_SHA,'prepared':str(a.prepared.resolve()),'prepared_sha256':binding['prepared_sha256'],'baseline_source_proof':binding,'engine_root':str(Path(metadata['engine_receipt']).parent),'engine_receipt_sha256':metadata['engine_receipt_sha256'],'cards':metadata['cards'],'args':args,'env':env,'image':metadata['runtime']['image'],'pack':metadata['pack'],'model_identity':identity,'primary_model_name':'hotschmoe-dd','research_alias':metadata['alias']+'-functional-twoCPU-prompts-freshGEN64','max_new':64,'fresh':1,'cpu_fixture_path':str(a.cpu_fixtures.resolve()),'cpu_fixture_sha256':sha(a.cpu_fixtures),'cpu_source_plan_sha256':sha(CPU_PLAN),'fixtures':fixture['fixtures'],'CPU_functional_input_binding':cpu_binding,'full_model_math_qualified':False,'registered_quality_qualified':False,'native_bitwise_authority':False,'registry_edit_applied':False,'fresh_scope':'allstage explicit freshGEN reset in one owned native process; CPU uses freshprocesses'}
 manifest_binding(plan);a.output.mkdir(parents=True,exist_ok=False);write(a.output/'plan.json',plan);print(json.dumps({'prepared':str(a.output/'plan.json'),'GPU_executed':False}))

def functional_request(raw,case,text,ids):
 parts=raw['done'].split();require(raw['ids']==ids and raw['fresh']==1 and raw['cancel_requested'] is None and raw['stop_sent'] is False,'Exact fresh functional source input differs')
 require(len(parts)>=6 and parts[0]=='DONE' and parts[5]=='stop' and int(parts[2])==len(ids),'Natural native stop/full prompt consumption required')
 output=raw['output_ids'];require(0<len(output)<64 and int(parts[1])==len(output) and all(type(t) is int and 0<=t<248320 for t in output),'Natural output below64 with complete token IDs required')
 require(raw['command']==serial.request_command(ids,64,1,None),'Actual greedy freshGEN64 command differs')
 require(criteria().functional(case,text) is True,'CPU functional semantic criterion failed');return {'passed':True,'case':case,'generated_tokens':len(output),'natural_stop':True,'semantic_checker':'exact frozen CPUV3 functional()','registered_quality_qualified':False}

def run(a):
 plan=read(a.plan);manifest_binding(plan);c1.leased([0,1]);health=read(a.pre_health);require(health['passed'] and set(plan['cards'])<=set(health['cards']) and 0<=time.time()-health['finished_epoch']<=300,'Fresh parent strict/compiled health missing');a.output.mkdir(parents=True,exist_ok=False);write(a.output/'plan.snapshot.json',plan);directory=a.output/'native';directory.mkdir();(directory/'captures').mkdir();(directory/'ARM').write_text('ARM workledger only\n');command,container=native_command(plan,directory,os.getpid());
 write(directory/'command.json',command);protocol=None;error=None;failure=None;rows=[];stage='native readiness';rc=None;state=None;removed=False
 try:
  protocol=serial.Protocol(command,directory)
  for case,fixture in enumerate(plan['fixtures']):
   for repeat in range(2):
    stage='case%d repeat%d freshGEN64'%(case,repeat);raw=protocol.request('case%d-repeat%d'%(case,repeat),fixture['ids'],fresh=1,max_new=64);text=serial.decode_output(plan['pack'],raw['output_ids']);gate=functional_request(raw,case,text,fixture['ids']);meta=serial.extract(raw,directory/'captures',False,False,expected_stage_ranges(plan['args']));require(meta['ledger']['actual_reused']==0 and not meta['ledger']['cancelled'],'Functional fresh request reused or canceled');rows.append({'case':case,'repeat':repeat,'raw':raw,'decoded_output':text,'functional':gate,'work_ledger':meta});write(directory/'requests.json',rows)
   first,second=rows[-2:];require(first['raw']['output_ids']==second['raw']['output_ids'] and first['decoded_output']==second['decoded_output'],'Native fresh deterministic repeat differs')
  require(not any(line.startswith(('SFD vector ','L0Q8 frame ','P30 frame ','PLEINPUT33 ')) for line in protocol.stderr),'Functional observerOFF emitted vectors');require(not list((directory/'captures').glob('*')),'ObserverOFF created raw activation files')
 except BaseException as exc:error=type(exc).__name__+': '+str(exc);failure={'type':type(exc).__name__,'stage':stage,'traceback':traceback.format_exc()}
 finally:
  try:rc=protocol.close() if protocol else None
  except BaseException as exc:error=(error+'; ' if error else '')+'close '+type(exc).__name__+': '+str(exc)
  state=c1.inspected(container)['State']
  if not state['Running']:subprocess.run(['docker','rm',container],check=True,capture_output=True,text=True);removed=c1.absent(container)
  terminal=error is None and rc==0 and state['ExitCode']==0 and not state.get('OOMKilled') and removed and len(rows)==4;report={'schema':1,'screen_generation':1,'native_supervisor_pid':os.getpid(),'passed':False,'collection_and_teardown_passed':terminal,'error':error,'failure_diagnostic':failure,'engine_rc':rc,'state':state,'removed':removed,'finished_epoch':time.time(),'plan_sha256':sha(a.output/'plan.snapshot.json'),'requests_sha256':sha(directory/'requests.json') if (directory/'requests.json').exists() else None,'logical_owner_proofs':{},'semantic_cases_and_repeats':rows,'full_model_math_qualified':False,'registered_quality_qualified':False,'natural_API_completion_or_latency_qualified':False,'scope':'twoCPU-prompts semantic/freshdeterministic nativeGEN only; parent source4/pages/health mandatory'};report['artifact_bindings']=proof.artifact_bindings(a.output);write(a.output/'report.json',report)
 return 0 if terminal else 1

def main():
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True);a=sub.add_parser('prepare');a.add_argument('--prepared',type=Path,required=True);a.add_argument('--cpu-fixtures',type=Path,required=True);a.add_argument('--model-identity',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a=sub.add_parser('run');a.add_argument('--plan',type=Path,required=True);a.add_argument('--pre-health',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a=p.parse_args();return prepare(a) if a.mode=='prepare' else run(a)
if __name__=='__main__':raise SystemExit(main())
