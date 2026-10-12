#!/usr/bin/env python3
"""Six natural-EOS requests in one process; existing owned model boundary only."""
import argparse,ast,copy,hashlib,importlib.util,json,os,sys,types
from pathlib import Path
from types import SimpleNamespace
import run_source51_esimd_f16_model_v1 as base
import run_source50_pair24_mirror_t1_model_v1 as pair
from serial37_canonical_json_v3 import canonical
c=base.c;require=base.require;HERE=base.HERE;ROOT=base.ROOT
PLAN=HERE/'natural-p2048-native-source-recipe-v1.json';PHASES=('natural-p2048-native',)
SETTINGS=None;NS=None;fresh=None

def configure(settings):
 global SETTINGS,fresh
 require(settings['generation']in(50,51)and settings['prefill']in(64,512,2048)and settings['cards']in(1,2),'Declared generation/prefill/topology only')
 require(settings['generation']==51 or settings['esimd']==0,'Source50 cannot enable source51 ESIMD')
 require(settings['corpus_sha256']==c.read(PLAN)['input_corpus_sha256']and c.sha(settings['corpus'])==settings['corpus_sha256'],'Authentic prepared corpus SHA differs')
 corpus=c.read(settings['corpus']);require(len(corpus['cases'])==3 and all(len(r['ids'])==2048 and r['max_new_tokens']==512 for r in corpus['cases']),'Three exact P2048 GEN512 cases required')
 require(corpus['actual_model_payload_read']is False and corpus['actual_GPU_touch']is False,'Tokenizer-only corpus required')
 SETTINGS=dict(settings)
 path=HERE/('source%d_model_sdk_binding_v1.py'%settings['generation']);spec=importlib.util.spec_from_file_location('_natural_sdk_binding',path);fresh=importlib.util.module_from_spec(spec);spec.loader.exec_module(fresh)
 if NS is not None:
  NS['fresh']=fresh
  if '_PORTS'in NS:NS['_PORTS']['fresh']=fresh
 return corpus

def closure():
 base.closure()
 for name,want in c.read(PLAN)['files'].items():require(c.sha(ROOT/name)==want,'Natural benchmark source changed '+name)
 require(SETTINGS is not None and c.sha(SETTINGS['corpus'])==SETTINGS['corpus_sha256'],'Prepared corpus changed');return c.sha(PLAN)

def set_sdk(root,receipt_sha):
 value=fresh.configure(root,receipt_sha);NS['intake'].SDK=fresh.SDK
 corpus=c.read(SETTINGS['corpus']);require(corpus['software_sha256']=={'/tools/strata_tokenizer.py':c.sha(fresh.SDK/'source/tools/strata_tokenizer.py'),'/serve/frontend.py':c.sha(fresh.SDK/'source/serve/frontend.py')},'Actual candidate tokenizer/template source differs from prepared corpus');return value

def requests(rows,expected):
 import source41_model_cli_contract_v1 as old
 node=copy.deepcopy(next(n for n in ast.parse(Path(old.__file__).read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name=='requests'))
 for n in ast.walk(node):
  if isinstance(n,ast.Compare)and ast.unparse(n)=='len(rows) == len(expected) == 4':n.comparators[-1]=ast.Constant(6)
  if isinstance(n,ast.Constant)and n.value=='GEN %d fresh=1 ckpt=1 temperature=0 logprobs=20 ':n.value='GEN %d fresh=1 ckpt=1 temperature=0 '
 scope=dict(vars(old));exec(compile(ast.fix_missing_locations(ast.Module([node],type_ignores=[])),__file__+'[six-original-request-predicates]','exec'),scope)
 result=scope['requests'](rows,expected)
 for i in(0,2,4):require(canonical(result[i])==canonical(result[i+1]),'Fresh-state natural output repeat differs')
 require(all(r['raw']['LP']==[]for r in rows),'Logprobs omitted');return result

def equivalence(*arms):
 require(len(arms)==1 and len(arms[0])==6,'One process/six natural requests required');return {'fresh_state_repeats_equal':True,'scope':'native output coherence; no original numerical qualification'}

def render_template(phase,metadata,cfg,inputs,binding):
 scope={**NS,'PHASES':(phase,'unused-no-second-process')};render=types.FunctionType(BASE_PHASE.__code__,scope,BASE_PHASE.__name__,BASE_PHASE.__defaults__,BASE_PHASE.__closure__);return render(phase,metadata,cfg,inputs,binding)

def phase_recipe(phase,metadata,cfg,inputs,binding):
 m,recipe=render_template(phase,metadata,cfg,inputs,binding);s=SETTINGS;argv=recipe['native_argv']
 for flag,value in(('--prefill',s['prefill']),('--max-context',4096),('--spec',2)):
  require(flag in argv,'Missing native option '+flag);argv[argv.index(flag)+1]=str(value)
 argv+=['--stop-eos','--eos-ids','248044,248046']
 recipe['native_env'].update(STRATA_PF_GROUPED_F16='0',STRATA_PF_ESIMD_F16=str(s['esimd']),STRATA_VERIFY_MIRROR_STAGE_T1=str(s['mirror']),ZE_AFFINITY_MASK='0,1'if s['cards']==2 else'0')
 if s['cards']==2:argv+=['--layer-split','24','--split-device','1','--trim-stage-weights'];m['cards']=[0,1]
 else:m['cards']=[0]
 corpus=c.read(s['corpus']);rows=[]
 for case_index,case in enumerate(corpus['cases']):
  for repeat in(0,1):rows.append({'ordinal':len(rows)+1,'ids':case['ids'],'fresh':1,'max_new':512,'pin':None,'role':case['id'],'case_index':case_index,'repeat':repeat,'process_temperature':'cold'if case_index==0 and repeat==0 else'warm','input_temperature':'first'if repeat==0 else'repeat'})
 recipe.update(cards=list(m['cards']),stages=[[0,0,24],[1,24,48]]if s['cards']==2 else[[0,0,48]],requests=rows,engine_generation=s['generation'],bounded_window_schema=2,selected_request_count=6,maximum_total_output_tokens=3072,natural_settings=s,requested_prefill=s['prefill'],requested_context=4096,natural_EOS=True,capture_kind='three P2048 natural-EOS cases and fresh-state repeats in one process',host_trace_enabled=False,both_arms_H36_enabled=False)
 m.update(source=str(fresh.SDK/'source'),executable=str(fresh.SDK/'build/strata'),alias='source%d-natural-P2048-native-CLI'%s['generation']);return m,recipe

def native_command(directory,m,recipe,pid,binding):
 name,cmd=BASE_COMMAND(directory,m,recipe,pid,binding)
 if SETTINGS['cards']==2:
  for card in(0,1):
   for path in('/dev/dri/card'+str(card),'/dev/dri/renderD'+str(128+card)):
    gid=str(os.stat(path).st_gid)
    if gid not in[cmd[i+1]for i,x in enumerate(cmd[:-1])if x=='--group-add']:cmd[cmd.index(m['runtime']['image']):cmd.index(m['runtime']['image'])]=['--group-add',gid]
 return name,cmd

def memory_admission(startup,metadata):
 return pair.memory_admission(startup,metadata)if SETTINGS['cards']==2 else base.original.life.prepared_ctrl.memory_admission(startup,metadata)

def preflight(plan,output):
 m=c.read(Path(plan['template_path'])/'prepared.json');cfg=c.read(Path(plan['template_path'])/'server-config.json');inputs=c.read(plan['inputs_path'])
 descriptor,recipe=phase_recipe(PHASES[0],m,cfg,inputs,c.sha(PLAN));name,command=native_command(Path(output)/PHASES[0],descriptor,recipe,NS['_CASE'][1],c.sha(PLAN))
 require(recipe['native_env']['STRATA_PF_GROUPED_F16']=='0'and recipe['native_env']['STRATA_CRITICAL_PATH_TRACE']=='0'and recipe['PTI_enabled']is False,'Unsafe grouping/tracing prohibited')
 require(len(recipe['requests'])==6 and all(len(r['ids'])==2048 and r['max_new']==512 and r['fresh']==1 for r in recipe['requests']),'Natural geometry differs')
 argv=recipe['native_argv'];require(argv[argv.index('--max-context')+1]=='4096'and argv[argv.index('--spec')+1]=='2'and '--stop-eos'in argv and not any(x in argv for x in('--mtp','--batch-mtp','--keep-canonical')),'Context/native pack/EOS/no-draft gate differs')
 return {'pure_command_construction_passed':True,'commands':[{'phase':PHASES[0],'container_name':name,'native_command':command,'recipe':recipe}],'GPU_launch_performed':False,'current_admission_still_required':True}

NS={**vars(base),**base.NS};NS.update(stat_snapshot=base.stat_snapshot,campaign_stats=base.campaign_stats)
for name,value in list(NS.items()):
 if isinstance(value,types.FunctionType)and (value.__globals__ is base.NS or value.__globals__ is vars(base)):NS[name]=types.FunctionType(value.__code__,NS,value.__name__,value.__defaults__,value.__closure__)
BASE_PHASE=NS['phase_recipe'];BASE_COMMAND=NS['native_command']
NS.update(__file__=__file__,_API=None,_API_SHA=None,PLAN=PLAN,PHASES=PHASES,closure=closure,set_sdk=set_sdk,phase_recipe=phase_recipe,preflight=preflight)
NS['contract']=SimpleNamespace(**{**vars(NS['contract']),'requests':requests,'equivalence':equivalence})
NS['life']=SimpleNamespace(**{**vars(NS['life']),'prepared_ctrl':SimpleNamespace(**{**vars(NS['life'].prepared_ctrl),'memory_admission':memory_admission}),'contract':SimpleNamespace(**{**vars(NS['life'].contract),'native_command':native_command})})
def generic_code(code):
 values=tuple(generic_code(v)if isinstance(v,types.CodeType)else v.replace('SDK51_root','SDK_root').replace('SDK51_receipt_sha256','SDK_receipt_sha256')if isinstance(v,str)else v for v in code.co_consts)
 return code.replace(co_consts=values)
fn=NS['ports'];NS['ports']=types.FunctionType(generic_code(fn.__code__),NS,fn.__name__,fn.__defaults__,fn.__closure__)
# Keep protected setup/pumps/retirement and campaign pre/post full4 verbatim.
NS['_PORTS']=base.boundary_ports(NS)
for name in('contract','set_case','set_runtime','admission','prerequisite','life','warm_control','mapped','native_pid','run_phase','ports','stat_snapshot','campaign_stats'):globals()[name]=NS[name]
def __getattr__(name):
 if name in NS:return NS[name]
 raise AttributeError(name)

def main():
 parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--generation',type=int,choices=(50,51),required=True);parser.add_argument('--corpus',required=True);parser.add_argument('--corpus-sha256',required=True);parser.add_argument('--prefill',type=int,choices=(64,512,2048),required=True);parser.add_argument('--cards',type=int,choices=(1,2),default=1);parser.add_argument('--esimd',type=int,choices=(0,1),required=True);parser.add_argument('--mirror',type=int,choices=(0,1),required=True);parser.add_argument('--construction-only',action='store_true');options,rest=parser.parse_known_args();settings=vars(options);only=settings.pop('construction_only');configure(settings)
 if only:
  q=argparse.ArgumentParser();q.add_argument('--sdk',required=True);q.add_argument('--sdk-receipt-sha256',required=True);q.add_argument('--plan',required=True);q.add_argument('--output',required=True);q.add_argument('--pti-intake',required=True);q.add_argument('--pti-intake-sha256',required=True);a,unused=q.parse_known_args(rest);set_sdk(a.sdk,a.sdk_receipt_sha256);set_case('prefill_about512');set_runtime(a.pti_intake,a.pti_intake_sha256);closure();require(not Path(a.output).exists(),'New external construction receipt required');c.write(a.output,preflight(c.read(a.plan),Path(a.output).parent/'construction-only-no-GPU'));return
 if '--leased'not in rest:os.execv(str(ROOT/'bin/gpu-run'),['gpu-run',sys.executable,__file__,*sys.argv[1:],'--leased'])
 sys.argv=[sys.argv[0],*rest];NS['_PORTS']['main']()
if __name__=='__main__':main()
