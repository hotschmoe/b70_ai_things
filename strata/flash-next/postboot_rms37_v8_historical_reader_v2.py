"""Explicit historical reader; original source/math/runtime receipts immutable."""
import qualify_native_rms_rsqrt37_v8 as original
from qualify_native_rms_rsqrt37_v8 import *
from postboot_original_publisher_reader_v1 import publisher
from postboot_original_model_association_v1 import historical_stat
from postboot_original_rms_dependencies_v2 import historical_fixture
def historical_result(root,model_association):
 root=Path(root).resolve();r=read(root/'report.json');require(r['passed'] is True and r['errors']==[] and r['actual_leaf_execution_observed'] is True and r['source_binding']==source_binding(),'Actual owned RMS runtime/source prerequisites failed');require(read(root/'source-plan.snapshot.json')==read(PLAN) and (root/'qualifier.snapshot.py').read_bytes()==Path(original.__file__).read_bytes(),'Actual owned producer/source snapshot changed');from native_rms_health_journal_binding_v3 import admit as health_admit
 health_admit(r,ROOT,HEALTH)
 from native_rms_owned_device_ops_binding_v8 import match_fixture
 from postboot_original_rms_dependencies_v2 import prior_v7
 admit_prior=lambda path:prior_v7(path,model_association)
 prior_proof=admit_prior(r['prior_rms_binding']['root']);require(prior_proof==r['prior_rms_binding'],'Actual prior RMS proof changed')
 fixture=historical_fixture(original,r['fixture_binding']['root'],model_association);argv,compile,runtime=expected_recipes(root,fixture['root'],r['producer_pid']);require(r['compile_argv']==argv and r['compile_command']['command']==compile and r['run_command']['command']==runtime,'Actual complete fresh compile/runtime recipes differ');require(fixture==r['fixture_binding'],'Actual original fixture changed');match_fixture(prior_proof,fixture);chronology(r)
 for n,w in r['artifact_sha256'].items():require(not (root/n).is_symlink() and sha(root/n)==w,'Original owned RMS artifact changed '+n)
 for label,row in [('compile',r['compile_command']),('runtime',r['run_command'])]+[(name,row) for stage in ('pre','post') for name,row in zip((stage+'-strict-health',stage+'-compiled-health'),r[stage+'_health']['rows'])]+[(stage+'-kernel',r[stage+'_journal']) for stage in ('pre','post')]:
  require(label+'.receipt.json' in r['artifact_sha256'],'Original command receipt absent from artifact closure');command_binding(row,root/(label+'.log'),root/(label+'.command.json'),root/(label+'.receipt.json'))
 from native_rms_kernel_journal_v5 import admit as kernel_admit
 kernel_admit(root,r)
 for stage,name in [('compile','compile'),('run','runtime')]:
  require(name+'-inspection.json' in r['artifact_sha256'],'Original terminal inspection absent from artifact closure');obj=r[stage+'_terminal'];terminal_receipt_binding(obj,root/(name+'-inspection.json'));observed_container(obj,obj['Name'][1:],modules()[0].IMAGE if stage=='compile' else RUNTIME_IMAGE,r[stage+'_command']['command']);terminal_binding(obj['State'])
 p,capture,guard,preserve,full4,baseline=modules();from c137_postboot_historical_baseline_v1 import finalized_binding as admit_baseline;_,baseline_scope=admit_baseline(r['baseline_binding']['baseline_root'],model_association,r['baseline_binding'].get('adjudication_receipt'));current=baseline_scope['original_binding'];require(current==r['baseline_binding'],'Current genuine SDK37/C137 baseline changed')
 lock=read(HERE/'model-lock.json');from native_rms_device_ops_lifecycle_v8 import native_trace,docker_epochs,pre_publisher
 native_trace(root/'runtime.log');docker_epochs(r);require('pre-full4.json' in r['artifact_sha256'],'Original pre full4 absent from artifact closure');publisher(root,r,model_association,lock,ROOT,'pre')
 shards=[ROOT/lock['destination']/x['path'] for x in lock['files'] if x['path'].startswith('UD-Q4_K_XL/')];require(len(shards)==4 and r['post_full4']==read(root/'post-full4.json') and r['post_full4']['passed'] is True and len(r['post_full4']['rows'])==4,'Original newfull4 source record changed');require(r['post_full4']['lock_sha256']==sha(HERE/'model-lock.json') and r['post_full4']['model_revision']==lock['revision'],'Original publisher lock/revision differs')
 for row,path in zip(r['post_full4']['rows'],shards):
  require(row['passed'] is True and row['path']==str(path) and row['stat_before']==row['stat_after'],'Original model source changed');historical_stat(model_association,path,row['stat_after'])
 from native_rms_publisher_binding_v2 import publisher_binding
 publisher(root,r,model_association,lock,ROOT,'post')
 require(guard(shards[2])['passed'],'Current source knownpage guard failed')
 require(r['helper_sha256']==sha(root/'build/native-rms-rsqrt37') and read(root/'runtime-before.json')==r['runtime_binding_before'],'Actual fresh helper/runtime-before receipt changed')
 actual_after=read(root/'runtime-after.json');actual_after['transient_unmapped_libraries_observed']=False;require(actual_after==r['runtime_binding_after'],'Actual runtime-after receipt changed')
 runtime_receipt_binding(r['runtime_binding_before'],r['runtime_binding_after'],root/'build/native-rms-rsqrt37')
 for row in r['runtime_binding_after']['libraries'].values():require(row['sha256'] and row['bytes']>0,'Actual intended runtime resolved library record absent')
 require(compare_all_variants(root/'build/native-output',fixture,prior_proof)==r['comparison'],'Actual direct/replay/ALLhypothesis comparison changed');require(source_binding()==r['source_binding'] and historical_fixture(original,fixture['root'],model_association)==fixture,'Source/current original inputs changed during readonly work');require(admit_prior(prior_proof['root'])==prior_proof,'Actual prior V7 changed during readonly comparison')
 return {'report_sha256':sha(root/'report.json'),'actual_RMS_observed':True,'comparison':r['comparison'],'full_model_math_qualified':False,'normal_model_graph_qualified':False,'latency_qualified':False,'mapped_library_roster_independently_reconstructed_from_raw_maps':False}

def finalized_binding(root,model_association):
 from postboot_original_association_recheck_v1 import recheck
 recheck(model_association)
 original_binding=historical_result(root,model_association)
 return {'original_binding':original_binding,'historical_evidence_only':True,'old_current_stat_gate_passed':False,'historical_GPU_health_transferred':False,'current_runtime_qualified':False,'current_identity_sha256':model_association['current_identity_sha256']}
