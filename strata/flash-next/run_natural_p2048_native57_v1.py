#!/usr/bin/env python3
"""Matched natural-EOS requests in two owned processes; existing owned model boundary only."""
import argparse,ast,copy,hashlib,importlib.util,json,os,sys,types
from pathlib import Path
from types import SimpleNamespace
import run_source51_esimd_f16_model_v1 as base
import run_source50_pair24_mirror_t1_model_v1 as pair
from serial37_canonical_json_v3 import canonical
c=base.c;require=base.require;HERE=base.HERE;ROOT=base.ROOT
PLAN=HERE/'native57-natural-parent-source-recipe-v1.json';PHASES=('natural-p2048-A','natural-p2048-B')
SETTINGS=None;NS=None;fresh=None

def profile_binding(path,expected):
 import struct
 path=Path(path);require(path.is_absolute()and path.resolve()==path and path.is_file()and not path.is_symlink(),'Exact nonaliased expert-profile file required')
 before=c.stat_signature(path);raw=path.read_bytes();after=c.stat_signature(path);require(before==after and len(raw)==196632 and hashlib.sha256(raw).hexdigest()==expected,'Expert-profile bytes/stat changed')
 require(raw[:4]==b'STRP'and struct.unpack_from('<5I',raw,4)==(1,48,512,24576,24576),'Complete full48x512 STRP v1 profile required')
 pairs=list(struct.iter_unpack('<HH',raw[24:98328]));require(set(pairs)=={(l,e)for l in range(48)for e in range(512)},'Exact complete expert ID permutation required');table=struct.unpack_from('<24576i',raw,98328);require(all(table[l*512+e]==i for i,(l,e)in enumerate(pairs)),'Exact profile reverse index required')
 return {'path':str(path),'sha256':expected,'stat5':before,'bytes':len(raw),'layers':48,'experts':512,'ranked_pairs':24576,'model_quantization_identity_encoded':False}

def arm_profile(phase):
 require(fresh is not None and fresh.SDK is not None,'Genuine SDK required before profile selection');s=SETTINGS
 arm='a'if s['experiment']=='copy'or phase==PHASES[0]else'b';path=s.get('profile_'+arm)or str(fresh.SDK/'source/data/expert-profile.bin');expected=s.get('profile_'+arm+'_sha256')or '8f59b4aa8873209dff11c11e37bcda9529a1335b724a1afeea37bf6388975baf'
 return profile_binding(path,expected)

def configure(settings):
 global SETTINGS,fresh
 require(settings['generation']==57 and settings['prefill']in(64,512,2048)and settings['cards']in(1,2),'Declared generation/prefill/topology only')
 require(settings['esimd']==0,'ESIMD remains OFF after actual output mismatch');require(settings['experiment']in('profile','copy')and(settings['experiment']!='copy'or settings['generation']==57),'Copy A/B requires actual source57')
 require(settings['corpus_sha256']==c.read(PLAN)['input_corpus_sha256']and c.sha(settings['corpus'])==settings['corpus_sha256'],'Authentic prepared corpus SHA differs')
 corpus=c.read(settings['corpus']);require(len(corpus['cases'])==3 and all(len(r['ids'])==2048 and r['max_new_tokens']==512 for r in corpus['cases']),'Three exact P2048 GEN512 cases required')
 require(corpus['actual_model_payload_read']is False and corpus['actual_GPU_touch']is False,'Tokenizer-only corpus required')
 SETTINGS=dict(settings)
 if settings['experiment']=='profile':require(settings.get('profile_a')and settings.get('profile_b')and settings.get('profile_a_sha256')and settings.get('profile_b_sha256'),'Two explicit profile files and SHA pins required')
 for arm in('a','b'):
  if settings.get('profile_'+arm):profile_binding(settings['profile_'+arm],settings['profile_'+arm+'_sha256'])
 if settings['experiment']=='copy':require(not settings.get('profile_b'),'Copy A/B uses one shared original profile, not two data arms')
 path=HERE/('source%d_model_sdk_binding_v1.py'%settings['generation']);spec=importlib.util.spec_from_file_location('_natural_sdk_binding',path);fresh=importlib.util.module_from_spec(spec);spec.loader.exec_module(fresh)
 if NS is not None:
  NS['fresh']=fresh
  if '_PORTS'in NS:NS['_PORTS']['fresh']=fresh
 return corpus

def closure():
 base.closure()
 for name,want in c.read(PLAN)['files'].items():require(c.sha(ROOT/name)==want,'Natural benchmark source changed '+name)
 require(SETTINGS is not None and c.sha(SETTINGS['corpus'])==SETTINGS['corpus_sha256'],'Prepared corpus changed')
 if fresh.SDK is not None:
  for phase in PHASES:arm_profile(phase)
 return c.sha(PLAN)

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
 require(len(arms)==2 and all(len(arm)==6 for arm in arms)and canonical(arms[0])==canonical(arms[1]),'Matched natural outputs/finish differ');return {'fresh_state_repeats_equal':True,'matched_arms_outputs_equal':True,'scope':'two processes, same six natural requests; no original numerical qualification'}

def render_template(phase,metadata,cfg,inputs,binding):
 scope={**NS,'PHASES':(phase,'unused-no-second-process')};render=types.FunctionType(BASE_PHASE.__code__,scope,BASE_PHASE.__name__,BASE_PHASE.__defaults__,BASE_PHASE.__closure__);return render(phase,metadata,cfg,inputs,binding)

def phase_recipe(phase,metadata,cfg,inputs,binding):
 m,recipe=render_template(phase,metadata,cfg,inputs,binding);s=SETTINGS;argv=recipe['native_argv']
 for flag,value in(('--prefill',s['prefill']),('--max-context',4096),('--spec',2)):
  require(flag in argv,'Missing native option '+flag);argv[argv.index(flag)+1]=str(value)
 argv+=['--stop-eos','--eos-ids','248044,248046']
 recipe['native_env'].update(STRATA_COPY_ENGINE=str(PHASES.index(phase))if s['experiment']=='copy'else'0',STRATA_PF_GROUPED_F16='0',STRATA_PF_ESIMD_F16=str(s['esimd']),STRATA_VERIFY_MIRROR_STAGE_T1=str(s['mirror']),ZE_AFFINITY_MASK='0,1'if s['cards']==2 else'0')
 if s['cards']==2:argv+=['--layer-split','24','--split-device','1','--trim-stage-weights'];m['cards']=[0,1]
 else:m['cards']=[0]
 asset=arm_profile(phase);require('--expert-profile'in argv,'Original static expert profile required');argv[argv.index('--expert-profile')+1]='/profiles/expert-profile.bin'
 corpus=c.read(s['corpus']);rows=[]
 for case_index,case in enumerate(corpus['cases']):
  for repeat in(0,1):rows.append({'ordinal':len(rows)+1,'ids':case['ids'],'fresh':1,'max_new':512,'pin':None,'role':case['id'],'case_index':case_index,'repeat':repeat,'process_temperature':'cold'if case_index==0 and repeat==0 else'warm','input_temperature':'first'if repeat==0 else'repeat'})
 recipe.update(expert_profile_binding=asset,matched_experiment=s['experiment'],cards=list(m['cards']),stages=[[0,0,24],[1,24,48]]if s['cards']==2 else[[0,0,48]],requests=rows,engine_generation=s['generation'],bounded_window_schema=2,selected_request_count=6,maximum_total_output_tokens=3072,natural_settings=s,requested_prefill=s['prefill'],requested_context=4096,natural_EOS=True,capture_kind='three P2048 natural-EOS cases and fresh-state repeats in one process',host_trace_enabled=False,both_arms_H36_enabled=False)
 m.update(source=str(fresh.SDK/'source'),executable=str(fresh.SDK/'build/strata'),alias='source%d-natural-P2048-native-CLI'%s['generation']);return m,recipe

def native_command(directory,m,recipe,pid,binding):
 name,cmd=BASE_COMMAND(directory,m,recipe,pid,binding)
 if SETTINGS['cards']==2:
  for card in(0,1):
   for path in('/dev/dri/card'+str(card),'/dev/dri/renderD'+str(128+card)):
    gid=str(os.stat(path).st_gid)
    if gid not in[cmd[i+1]for i,x in enumerate(cmd[:-1])if x=='--group-add']:cmd[cmd.index(m['runtime']['image']):cmd.index(m['runtime']['image'])]=['--group-add',gid]
 asset=recipe['expert_profile_binding'];require(profile_binding(asset['path'],asset['sha256'])==asset,'Actual RO expert profile changed');at=cmd.index(m['runtime']['image']);cmd[at:at]=['-v',asset['path']+':/profiles/expert-profile.bin:ro'];return name,cmd

def memory_admission(startup,metadata):
 return pair.memory_admission(startup,metadata)if SETTINGS['cards']==2 else base.original.life.prepared_ctrl.memory_admission(startup,metadata)

def matched_recipe_gate(left,right):
 require(left['native_argv']==right['native_argv']and canonical(left['requests'])==canonical(right['requests'])and canonical(left['natural_settings'])==canonical(right['natural_settings']),'Matched native configuration/input differs')
 le=dict(left['native_env']);re=dict(right['native_env']);kind=left['matched_experiment'];require(kind==right['matched_experiment'],'Matched experiment differs')
 if kind=='copy':
  require(le.pop('STRATA_COPY_ENGINE')=='0'and re.pop('STRATA_COPY_ENGINE')=='1'and canonical(left['expert_profile_binding'])==canonical(right['expert_profile_binding']),'Copy-engine A/B must retain identical profile')
 else:require(le['STRATA_COPY_ENGINE']==re['STRATA_COPY_ENGINE']=='0','Profile A/B cannot change copy dispatch')
 require(canonical(le)==canonical(re),'Unexpected matched environment difference');return True

def preflight(plan,output):
 m=c.read(Path(plan['template_path'])/'prepared.json');cfg=c.read(Path(plan['template_path'])/'server-config.json');inputs=c.read(plan['inputs_path'])
 commands=[]
 for phase in PHASES:
  descriptor,recipe=phase_recipe(phase,m,cfg,inputs,c.sha(PLAN));name,command=native_command(Path(output)/phase,descriptor,recipe,NS['_CASE'][1],c.sha(PLAN));commands.append({'phase':phase,'container_name':name,'native_command':command,'recipe':recipe})
  require(recipe['native_env']['STRATA_PF_GROUPED_F16']=='0'and recipe['native_env']['STRATA_CRITICAL_PATH_TRACE']=='0'and recipe['PTI_enabled']is False,'Unsafe grouping/tracing prohibited')
  require(len(recipe['requests'])==6 and all(len(r['ids'])==2048 and r['max_new']==512 and r['fresh']==1 for r in recipe['requests']),'Natural geometry differs')
  argv=recipe['native_argv'];require(argv[argv.index('--max-context')+1]=='4096'and argv[argv.index('--spec')+1]=='2'and '--stop-eos'in argv and not any(x in argv for x in('--mtp','--batch-mtp','--keep-canonical')),'Context/native pack/EOS/no-draft gate differs')
 matched_recipe_gate(commands[0]['recipe'],commands[1]['recipe'])
 return {'pure_command_construction_passed':True,'commands':commands,'GPU_launch_performed':False,'current_admission_still_required':True}

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
 parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--generation',type=int,choices=(57,),required=True);parser.add_argument('--corpus',required=True);parser.add_argument('--corpus-sha256',required=True);parser.add_argument('--prefill',type=int,choices=(64,512,2048),required=True);parser.add_argument('--cards',type=int,choices=(1,2),default=1);parser.add_argument('--esimd',type=int,choices=(0,1),required=True);parser.add_argument('--mirror',type=int,choices=(0,1),required=True);parser.add_argument('--experiment',choices=('profile','copy'),required=True);parser.add_argument('--profile-a');parser.add_argument('--profile-a-sha256');parser.add_argument('--profile-b');parser.add_argument('--profile-b-sha256');parser.add_argument('--construction-only',action='store_true');options,rest=parser.parse_known_args();settings=vars(options);only=settings.pop('construction_only');configure(settings)
 if only:
  q=argparse.ArgumentParser();q.add_argument('--sdk',required=True);q.add_argument('--sdk-receipt-sha256',required=True);q.add_argument('--plan',required=True);q.add_argument('--output',required=True);q.add_argument('--pti-intake',required=True);q.add_argument('--pti-intake-sha256',required=True);a,unused=q.parse_known_args(rest);set_sdk(a.sdk,a.sdk_receipt_sha256);set_case('prefill_about512');set_runtime(a.pti_intake,a.pti_intake_sha256);closure();require(not Path(a.output).exists(),'New external construction receipt required');c.write(a.output,preflight(c.read(a.plan),Path(a.output).parent/'construction-only-no-GPU'));return
 if '--leased'not in rest:os.execv(str(ROOT/'bin/gpu-run'),['gpu-run',sys.executable,__file__,*sys.argv[1:],'--leased'])
 sys.argv=[sys.argv[0],*rest];NS['_PORTS']['main']()
if __name__=='__main__':main()
