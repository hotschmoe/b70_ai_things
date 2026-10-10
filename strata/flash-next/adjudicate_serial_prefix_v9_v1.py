#!/usr/bin/env python3
"""NEW read-only adjudication of one exact V9 post-reader failure; no GPU rerun.
Run only by the coordinator. Original files/FAIL flags are never modified.
"""
import argparse,ast,copy,hashlib,importlib,json,os,re,shlex,sys,time
from pathlib import Path
import serial_prefix_qualification_v9 as v
import qualify_serial_prefix_v9 as parent_source
from serial_request_recollection_v1 import recollect_request
from source_page_watchdog_v3 import KNOWN_PAGES,PAGE_BYTES,preserve
read,sha,require=v.read,v.sha,v.require

def source_binding():
 path=v.ROOT/'strata/flash-next/serial-prefix-v9-adjudication-source-plan-v1.json';manifest=read(path)
 for name,digest in manifest['files'].items():require(sha(v.ROOT/name)==digest,'Frozen adjudication source changed '+name)
 return sha(path)

def equal(a,b):return json.dumps(a,sort_keys=True)==json.dumps(b,sort_keys=True)

def closed_admission(parent,child,log):
 require(parent['passed'] is False and parent['errors']==['V9 finalize rejected qualification'],'Only exact V9 finalizer failure may be adjudicated')
 require(log.rstrip().endswith('ValueError: Cold raw request copy changed') and 'line 502, in finalize' in log,'Exact frozen reader failure not observed')
 require(parent['child_return_code']==0 and parent['owned_containers_terminal'] is True and parent['forced_cleanup'] is False and parent['interrupted'] is False and all(parent[k] is True for k in ['pre_health_passed','post_health_passed','kernel_fault_gate_passed']),'Original lifecycle gates incomplete')
 require(child['schema']==9 and child['group']=='remaining' and child['passed'] is False and child['post_health_passed'] is False and child['error'] is None and child['numerical_and_teardown_passed'] is True and child['complete_prefix_qualified'] is False,'Original child scope/numerical state differs')
 require([x['name'] for x in child['results']]==list(v.REMAINING_GROUPS),'Complete exact seven-group child roster required')
 for result in child['results']:
  t=result['terminal'];require(result['passed'] is True and result['error'] is None and result['removed'] is True and result['engine_rc']==0 and t['Running'] is False and t['ExitCode']==0 and t['OOMKilled'] is False and not t['Error'],'Actual group healthy owned terminal proof absent')
 return True

def tree_binding(root):
 files={}
 for p in sorted(root.rglob('*')):
  require(not p.is_symlink(),'Symlink in original evidence tree')
  if p.is_file():files[str(p.relative_to(root))]=sha(p)
 return files

def page_receipt(row):
 require(row['passed'] is True and row['stat_before']==row['stat_after'],'Recorded source page state differs')
 require([(x['offset'],x['expected_sha256']) for x in row['rows']]==list(KNOWN_PAGES),'Exact two known page roster differs')
 for item in row['rows']:require(item['passed'] is True and item['bytes']==PAGE_BYTES and item['sha256']==item['expected_sha256']==sha(Path(item['preserved_path'])),'Preserved source page bytes differ')

def health_binding(root,label):
 h=read(root/(label+'-health.json'));require(h['passed'] is True and h['cards']==[0,1] and len(h['files'])==2,'Actual strict/compiled pair health missing')
 require(h['health_image']==parent_source.HEALTH,'Health image changed')
 for path,digest in h['source_sha256'].items():require(sha(Path(path))==digest,'Health source changed')
 for index,row in enumerate(h['files']):
  require(row['return_code']==0 and row['error'] is None and sha(Path(row['path']))==row['sha256'],'Health command/log failure or mutation')
  expected=[str(v.ROOT/'vllm/int4/diagnostics/xpu_health_strict.sh'),'--img',parent_source.HEALTH] if index==0 else [str(v.ROOT/'bin/xpu-collective-health'),'--img',parent_source.HEALTH,'--p2p','0','--timeout','180']
  require(row['command']==expected,'Public actual strict/compiled health command differs')
  command_path=root/(label+('-strict' if index==0 else '-compiled-pair')+'.command.json');require(sha(command_path)==row['command_file_sha256'] and read(command_path)==row['command'],'Health exact command differs')
  text=Path(row['path']).read_text()
  if index==0:require('card 0: OK' in text and 'card 1: OK' in text and 'xpu-health: HEALTHY (cards 0 1)' in text and 'HEALTH_FAIL' not in text,'Strict card sentinels incomplete')
  else:require('COLLECTIVE_HEALTH_OK world_size=2 shape=4x5120 compiled_iterations=10 p2p=0' in text and 'xpu-collective-health: HEALTHY' in text,'Compiled collective marker incomplete')
 return h

def tokenizer(plan):
 source=Path(plan['engine_root'])/'source';sys.path[:0]=[str(source/'tools'),str(source/'serve')]
 token_module=importlib.import_module('strata_tokenizer');front_module=importlib.import_module('frontend')
 require(Path(token_module.__file__).resolve()==(source/'tools/strata_tokenizer.py').resolve() and Path(front_module.__file__).resolve()==(source/'serve/frontend.py').resolve(),'Foreign preloaded tokenizer/frontend module')
 Tokenizer=token_module.Tokenizer;ChatTemplate=front_module.ChatTemplate
 p=Path(plan['pack'])/'tokenizer';vocab=read(p/'vocab.json');tokens=[None]*len(vocab)
 for token,index in vocab.items():tokens[index]=token
 cfg=read(p/'tokenizer.json');tok=Tokenizer(tokens,(p/'merges.txt').read_text().split('\n'),read(p/'token_type.json'),cfg['pre'],cfg['special_ids']);tpl=ChatTemplate(p/'chat_template.jinja')
 require({x.name:sha(x) for x in p.iterdir() if x.is_file()}==plan['tokens']['tokenizer_sha256'],'Exact tokenizer files changed')
 return tok,tpl

def owned_command_binding(root,group,plan,parent,child):
 directory=root/'child'/group;engine=Path(plan['engine_root']);args=list(plan['args']);parking=group in {'parked','eviction'}
 v.option(args,'--conversation-cache-mib',512 if parking else 0);v.option(args,'--conversation-cache-slots',1 if parking else 0)
 env=dict(read(Path(plan['prepared'])/'server-config.json')['env']);v.observer_env_contract(env)
 env.update({'STRATA_FIDELITY_DIAG':'1','STRATA_FIDELITY_DIAG_ACTIVATIONS':'1','STRATA_FIDELITY_DIAG_ARM':'/results/ARM','STRATA_FIDELITY_DIAG_DIR':'/results/captures','STRATA_PREFIX_DIAG':'1','STRATA_PREFIX_LIFECYCLE_DIAG':'1','STRATA_PREFIX_DIAG_ARM':'/results/ARM','STRATA_ARTIFACT_IDENTITY_SHA256':child['plan_sha256']})
 name='b70-prefix-'+str(parent['child_pid'])+'-'+group
 expected=['docker','run','-i','--name',name,'--label','b70.prefix.plan='+child['plan_sha256'],'--network','none','--device','/dev/dri','--user','1000:1000','--memory','105g','--memory-swap','105g','--group-add',str(os.stat('/dev/dri/renderD128').st_gid),'-v',str(engine/'build')+':/build:ro','-v',str(engine/'source')+':/src:ro','-v',plan['pack']+':/pack:ro','-v',str(v.ROOT/read(v.ROOT/'strata/flash-next/model-lock.json')['destination'])+':/model:ro','-v',str(directory)+':/results']
 for key,value in sorted(env.items()):expected+=['-e',key+'='+value]
 expected += [plan['image'],'cd /src; source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; exec '+shlex.join(['/build/strata','--serve']+args)]
 require(read(directory/'command.json')==expected,'Complete actual owned process argv/config differs')
 return True

def recollect_group(root,group,plan,result,tok,tpl):
 directory=root/'child'/group;saved=read(directory/'requests.json');require(len(saved)==v.ARMED_REQUESTS[group],'Complete exact request count differs')
 requests=[];comparisons=[];cursor=0;cases=copy.deepcopy(plan['tokens']['cases']);ranges=v.expected_stage_ranges(plan['args'])
 def req(label,key,fresh=0,pin=None,cancel=None):
  nonlocal cursor
  require(cursor<len(saved),'Production branch exceeds saved request roster');entry=saved[cursor];cursor+=1
  raw,meta=recollect_request(directory/(label+'.json'),entry,directory/'captures',ranges,v.extract)
  require(raw['label']==label and raw['ids']==cases[key] and raw['fresh']==fresh and raw['pin']==pin and raw['cancel_requested']==cancel and raw['command']==v.request_command(cases[key],64,fresh,pin),'Exact production branch protocol/input differs')
  decoded=tok.decode([i for i in raw['output_ids'] if tok.token_types[i] not in (3,4)],errors='strict');require(decoded==raw['decoded_output_without_special_tokens'],'Actual natural output text/token identity differs')
  requests.append({'raw':raw,'meta':meta});return raw,meta
 def continuation(plan,seed,directory):
  c=read(directory/'continuation-token-receipt.json');require(equal(read(directory/'continuation-seed.json'),seed),'Actual continuation seed changed')
  messages=plan['tokens']['seed_messages']+[{'role':'assistant','content':seed['decoded_output_without_special_tokens']},{'role':'user','content':plan['tokens']['continuation_question']}]
  ids=tok.encode(tpl.render(messages,enable_thinking=False),parse_special=True);committed=seed['lifecycle_committed_ids']
  expected={'schema':6,'continuation_ids':ids,'expected_live_reuse':len(committed),'seed_input_ids':seed['ids'],'seed_output_ids':seed['output_ids'],'seed_committed_ids':committed,'messages':messages,'tokenizer_sha256':plan['tokens']['tokenizer_sha256'],'fixture_semantics':'actual natural assistant reply plus new user; exact template, no token substitutions'}
  require(equal(c,expected) and ids[:len(committed)]==committed and len(committed)<len(ids)-1,'Current exact-template continuation recollection differs');return c
 def check_write(path,data):require(equal(read(path),data),'Recollected keyed eviction proof changed')
 tree=ast.parse(Path(v.__file__).read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_process');branch=next(n for n in ast.walk(fn) if isinstance(n,ast.If) and isinstance(n.test,ast.Compare) and ast.unparse(n.test)=="group == 'basic'")
 env={'group':group,'req':req,'plan':plan,'cases':cases,'rows':comparisons,'requests':requests,'directory':directory,'require':require,'compare_requests':v.compare_requests,'keyed_eviction':v.keyed_eviction,'write':check_write,'continue_fixture':continuation}
 exec(compile(ast.Module(body=[branch],type_ignores=[]),'<frozen V9 production request branch>','exec'),env)
 require(cursor==len(saved) and equal(comparisons,result['comparisons']),'Complete production comparisons differ')
 require(equal(read(directory/'result.json'),result),'Saved terminal group report differs from child report')
 return {'group':group,'requests':cursor,'comparisons':len(comparisons),'passed':True,'actual_reused':[x['meta']['ledger']['actual_reused'] for x in requests]}

def adjudicate(run_root,output):
 root=Path(run_root).resolve();output=Path(output).resolve();require(not output.exists() and root not in output.parents,'New evidence output outside original tree required');output.mkdir()
 proof={'schema':1,'passed':False,'started_epoch':time.time(),'original_parent_passed':False,'original_reports_modified':False,'scope':'New closed-evidence bounded serial remaining-suite adjudication; original V9 remains FAILED; no GPU replay/full model/concurrency/API/latency qualification','complete_prefix_qualified':False,'groups':[],'errors':[]}
 before=tree_binding(root)
 try:
  adjudicator_source=source_binding();source_plan=v.source_plan_binding();parent=read(root/'parent-qualification.json');child=read(root/'child/report.json');plan=read(root/'input-plan.snapshot.json');closed_admission(parent,child,(root/'finalize.log').read_text())
  require(equal(plan,read(root/'child/plan.snapshot.json')) and sha(root/'input-plan.snapshot.json')==parent['plan_sha256']==child['plan_sha256'],'Parent/child exact plan association differs')
  require(plan['suite_source_plan_sha256']==source_plan and parent['controller_sha256']==sha(Path(v.__file__)) and parent['wrapper_sha256']==sha(Path(parent_source.__file__)),'Original frozen source closure changed')
  require(sha(root/'controller.py')==parent['controller_sha256'] and sha(root/'wrapper.py')==parent['wrapper_sha256'],'Preserved source snapshots differ')
  command=read(root/'child.command.json');basic=Path(command[command.index('--basic-receipt')+1]);one=Path(command[command.index('--one-card-receipt')+1]) if '--one-card-receipt' in command else None;v.preflight(plan,basic,one);require(equal(parent_source.prepared_chain_binding(plan),parent['prepared_chain']),'Current C113 SDK/identity association differs')
  require(read(root/'preflight.json')['passed'] is True,'Original CPU admission absent')
  require(read(root/'finalize.command.json')==['/usr/bin/python3',str(Path(v.__file__)), 'finalize','--output',str(root/'child'),'--post-health',str(root/'post-health.json'),'--model-identity',str(root/'post-model-identity.json')],'Exact failed finalizer command association differs')
  require(child['cards']==plan['cards']==parent['workload_cards'] and parent['cards_held']==[0,1] and parent['expected_stage_ranges']==[list(x) for x in v.expected_stage_ranges(plan['args'])],'Original device/topology association differs')
  require(child['engine_receipt_sha256']==plan['engine_receipt_sha256'] and child['pre_health_sha256']==sha(root/'pre-health.json'),'Original engine/pre-health association differs')
  pre=health_binding(root,'pre');post=health_binding(root,'post');require(pre['finished_epoch']<=child['finished_epoch']<=parent['child_terminal_epoch']<=post['started_epoch']<=post['finished_epoch'],'Runtime/health chronology differs')
  for label in ['pre','post']:
   kernel=read(root/(label+'-kernel-journal.command.json'));require(kernel==['journalctl','-k','--since','@'+str(int(parent['started_epoch'])),'--no-pager'],'Exact kernel fault interval differs');require(not parent_source.FAULT.search((root/(label+'-kernel-journal.log')).read_text()),'Stored kernel fault gate fails')
  for key,label in [('known_pages_before_hash','before-full4'),('known_pages_after_hash','after-full4')]:require(equal(parent[key],read(root/(label+'-source-pages.json'))),'Recorded page receipt association differs');page_receipt(parent[key])
  identity=read(root/'post-model-identity.json');require(identity['after_child_terminal_epoch']==max(parent['child_terminal_epoch'],child['finished_epoch'],post['finished_epoch']),'New full4 terminal/health bound differs')
  require(parent['known_pages_before_hash']['epoch']<=identity['started']<=identity['finished']<=parent['known_pages_after_hash']['epoch']<=parent['finished_epoch'] and identity['started']>=max(parent['child_terminal_epoch'],post['finished_epoch']),'Original NEW full4/page chronology differs')
  lock_path=v.ROOT/'strata/flash-next/model-lock.json';lock=read(lock_path);shards=[v.ROOT/lock['destination']/f['path'] for f in lock['files'] if f['path'].startswith('UD-Q4_K_XL/')];proof['identity']=v.verify_model_identity(root/'post-model-identity.json',lock,shards)
  proof['current_pages_before']=preserve(shards[2],output,'before-recollection');require(proof['current_pages_before']['passed'],'Current source pages failed')
  tok,tpl=tokenizer(plan)
  proof['CPU_reader_runtime']={'python':sys.executable,'python_sha256':sha(Path(sys.executable).resolve()),'python_version':sys.version,'modules':{name:{'path':str(Path(importlib.import_module(name).__file__).resolve()),'sha256':sha(Path(importlib.import_module(name).__file__).resolve()),'version':getattr(importlib.import_module(name),'__version__',None)} for name in ['regex','jinja2','strata_tokenizer','frontend']}}
  for group,result in zip(v.REMAINING_GROUPS,child['results']):
   owned_command_binding(root,group,plan,parent,child)
   proof['groups'].append(recollect_group(root,group,plan,result,tok,tpl))
  proof['current_pages_after']=preserve(shards[2],output,'after-recollection');require(proof['current_pages_after']['passed'],'Post-recollection current source pages failed');require(equal(parent_source.prepared_chain_binding(plan),parent['prepared_chain']),'Current C113/SDK association changed during recollection')
  require(source_binding()==adjudicator_source and v.source_plan_binding()==source_plan and tree_binding(root)==before,'Original closed evidence/source mutated during adjudication')
  proof.update(passed=True,bounded_serial_remaining_adjudicated=True,original_parent_sha256=before['parent-qualification.json'],original_child_sha256=before['child/report.json'],source_plan_sha256=source_plan,adjudicator_source_plan_sha256=adjudicator_source,evidence_tree_sha256=hashlib.sha256(json.dumps(before,sort_keys=True).encode()).hexdigest(),requests=sum(x['requests'] for x in proof['groups']))
 except Exception as error:proof['errors'].append(str(error))
 proof['finished_epoch']=time.time();v.write(output/'original-evidence-manifest.json',before);v.write(output/'report.json',proof);return proof

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();result=adjudicate(a.run_root,a.output);print(json.dumps({'passed':result['passed'],'errors':result['errors'],'report':str(a.output/'report.json')}));raise SystemExit(0 if result['passed'] else 1)
