"""Read-only finalized synthetic HC composition/current source admission."""
from pathlib import Path
import qualify_hc_composition35_v2 as parent_module
from explore_full48_original_prefix1_v4 import identity_admission
from source_page_watchdog_v3 import guard,KNOWN_PAGES
sha,read,require=parent_module.sha,parent_module.read,parent_module.require
ROOT=parent_module.ROOT;HERE=parent_module.HERE

def journal_receipt_binding(root,parent,stage):
 import math
 root=Path(root).resolve();require(stage in ('pre','post'),'Exact journal stage required');row=parent['kernel_journal_receipts'][stage];require(row==read(root/(stage+'-kernel-receipt.json')),'Actual saved journal receipt changed/missing')
 path=root/(stage+'-kernel-journal.log');cmd=root/(stage+'-kernel-journal.command.json');argv=['journalctl','-k','--since','@'+str(int(parent['started_epoch'])),'--no-pager']
 require(row['return_code']==0 and row['error'] is None and row['path']==str(path) and row['command']==argv==read(cmd) and row['command_file_sha256']==sha(cmd) and row['sha256']==row['stdout_sha256']==sha(path),'Original complete kernel journal command/log/return/error differs')
 require(all(type(row[k]) in (int,float) and math.isfinite(row[k]) for k in ('started_epoch','finished_epoch')) and parent['started_epoch']<=row['started_epoch']<=row['finished_epoch']<=parent['finished_epoch'],'Journal chronology invalid')
 require(row['started_epoch']>=parent[stage+'_health_finished_epoch'],'Journal mustfollow complete stagehealth')
 if stage=='pre':require(row['finished_epoch']<=parent['child_started_epoch']<=parent['child_terminal_epoch'],'Prejournal mustcomplete before leaf start')
 else:require(row['started_epoch']>=parent['child_terminal_epoch'],'Postjournal before leaf terminal')
 require(not parent_module.FAULT.search(path.read_text()),'Actual kernel fault signature present');return row

def leaf_log_binding(root,saved):
 root=Path(root).resolve();path=root/'leaf.log';require(saved['log_sha256']==sha(path),'Original device/FP leaf log changed');text=path.read_text();result={}
 for kind in ('DEVICE','CONFIG','FP_CONFIG'):
  rows=[line for line in text.splitlines() if line.startswith('HC35_COMPOSITION_'+kind+' ')];require(len(rows)==1,'Actual unique device/config/FP observation missing');result[kind]=rows[0]
 require(result['CONFIG']=='HC35_COMPOSITION_CONFIG synthetic=1 compiled_composition=1 shadows_separate=1 subgroup=32 eps=1e-6 graph_and_queue=1 normal_model_graph_qualified=0 device_intrinsics_qualified=0 model_math_qualified=0','Actual source composition configuration differs')
 require('backend=level_zero ' in result['DEVICE'] and ' affinity=0 selector=level_zero:gpu' in result['DEVICE'],'Actual physical-device backend/pin differs')
 require(result['FP_CONFIG'].endswith(' observed_device_flags_only=1 compiler_lowering_unobserved=1'),'FP capability observation scope changed');return {'log_sha256':sha(path),'lines':result,'device_general_FP_mode_qualified':False}

def finalized_binding(run_root):
 root=Path(run_root).resolve();p=read(root/'parent-qualification.json');require(p.get('schema')==2 and p.get('parent_generation')==2,'V1/older parent runtime proof refused');source=parent_module.source_binding();require(p['source_dependencies']==source and p['controller_sha256']==sha(Path(parent_module.__file__)) and (root/'parent.py').read_bytes()==Path(parent_module.__file__).read_bytes(),'Actual immutable parent source snapshot differs')
 require(p['passed'] is True and p['errors']==[] and p['child_return_code']==0 and p['owned_containers_terminal'] is True and p['forced_cleanup'] is False and p['interrupted'] is False and all(p[k] is True for k in ('pre_health_passed','post_health_passed','kernel_fault_gate_passed','post_source_unchanged')),'Actual source/health/owned parent qualification failed')
 require(p['cards_held']==[0,1] and p['workload_cards']==[0] and p['physical_leaf_card']==0 and all(p[k] is False for k in ('full_model_math_qualified','normal_model_graph_qualified','device_intrinsics_qualified','real_model_inputs_observed')),'Actual physicalcard/scope differs')
 receipt,compiled=parent_module.compile_binding(Path(p['compile_receipt_path']));require(compiled==p['compile_binding'],'Current fresh compile source/binary association changed');inputs=Path(p['inputs_root']);manifest=parent_module.fixture.input_binding(inputs);require(manifest==p['input_manifest'] and sha(inputs/'manifest.json')==p['input_manifest_sha256'] and parent_module.producer.corpus_contract(receipt['plan'],manifest)==p['canonical_corpus'],'Current canonical synthetic input/host/count association changed')
 name='b70-hc35comp2-runtime-'+str(p['parent_pid']);require(p['container']==name and p['image']==receipt['plan']['image'] and read(root/'leaf.command.json')==parent_module.launch_command(receipt,root,name,compiled['compile_receipt_sha256'],inputs),'Actual full owned leaf command/PID/image differs');state=p['container_terminal'];require(state['Running'] is False and state['ExitCode']==0 and not state.get('OOMKilled') and not state.get('Error'),'Actual owned leaf terminal state differs')
 require(read(root/'leaf-owned-remove.command.json')==['docker','rm',name],'Actual owned normal removal command differs')
 for stage in ('pre','post'):
  h=read(root/(stage+'-health.json'));require(h['passed'] is True and h['cards']==[0,1] and h['health_image']==parent_module.HEALTH and len(h['files'])==2 and h['finished_epoch']==p[stage+'_health_finished_epoch'],'Actual strict pair health binding differs')
  commands=[[str(ROOT/'vllm/int4/diagnostics/xpu_health_strict.sh'),'--img',parent_module.HEALTH],[str(ROOT/'bin/xpu-collective-health'),'--img',parent_module.HEALTH,'--p2p','0','--timeout','180']]
  for row,command,label in zip(h['files'],commands,('strict','compiled-pair')):
   path=root/(stage+'-'+label+'.log');cmd=root/(stage+'-'+label+'.command.json');require(row['path']==str(path) and row['return_code']==0 and row['error'] is None and row['command']==read(cmd)==command and row['command_file_sha256']==sha(cmd) and row['sha256']==sha(path),'Actual health command/log association differs')
  journal_receipt_binding(root,p,stage)
 lock_path=HERE/'model-lock.json';lock=read(lock_path);shards=[ROOT/lock['destination']/f['path'] for f in lock['files'] if f['path'].startswith('UD-Q4_K_XL/')];path=root/'post-model-identity.json';require(sha(path)==p['post_model_identity_sha256'],'Actual new post4 identity receipt changed');boundary=max(p['child_terminal_epoch'],p['post_health_finished_epoch']);require(read(path)['started']>=p['kernel_journal_receipts']['post']['finished_epoch'],'New full4 mustfollow complete postjournal');identity=identity_admission(path,lock_path,shards,boundary);current=guard(shards[2])
 for key in ('known_pages_before_hash','known_pages_after_hash'):
  pages=p[key];require(pages['passed'] is True and Path(pages['path']).resolve()==shards[2].resolve() and pages['stat_before']==pages['stat_after']==current['stat_after'] and [(r['offset'],r['expected_sha256']) for r in pages['rows']]==list(KNOWN_PAGES),'Exact current known-page stat/offset binding differs')
  for row in pages['rows']:
   raw=Path(row['preserved_path']);require(raw.resolve().parent==root and not raw.is_symlink() and raw.stat().st_size==4096 and sha(raw)==row['sha256']==row['expected_sha256'],'Preserved known page changed/escaped')
 require(p['known_pages_before_hash']['epoch']<=identity['started']<=identity['finished']<=p['known_pages_after_hash']['epoch']<=p['finished_epoch'],'Known-page/new full4 chronology differs')
 raw=parent_module.fixture.collect(inputs,root/'raw-new',root/'leaf.log');require(raw==read(root/'raw-component-proof.json') and raw['numeric_bitwise_passed'] is True and p['component_proof']['numerical_and_teardown_passed'] is True,'Fresh actual raw/shadow/replay recollection differs');free=parent_module.parse_trace((root/'leaf.log').read_text(),False);saved=read(root/'usm-logical-free-proof.json');require(saved['log_sha256']==sha(root/'leaf.log'),'Original leaf log SHA changed');require(free['passed'] is True and free['live']==[] and all(saved[k]==v for k,v in free.items()) and all(parent_module.negative_controls((root/'leaf.log').read_text(),False).values()),'Actual context-matched USM logical frees differ');require(p['component_proof']['raw_proof_sha256']==sha(root/'raw-component-proof.json') and p['component_proof']['usm_proof_sha256']==sha(root/'usm-logical-free-proof.json'),'Actual raw/free proof SHA association differs')
 require(parent_module.source_binding()==source and parent_module.fixture.input_binding(inputs)==manifest and parent_module.compile_binding(Path(p['compile_receipt_path']))[1]==compiled,'Current source/library/corpus changed during readonly work');guard(shards[2])
 observations=leaf_log_binding(root,saved);return p,raw,{'actual_device_configuration_observations':observations,'parent_sha256':sha(root/'parent-qualification.json'),'source':source,'compile_binding':compiled,'current_post4_identity':identity,'current_known_pages':current,'device_intrinsics_qualified':False,'normal_model_graph_qualified':False,'full_model_math_qualified':False,'tolerance_gate':None}
