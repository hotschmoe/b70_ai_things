"""Current52/58/59 existing-hook diagnostic recipe; no runtime admission claim."""
import copy, hashlib, json
from pathlib import Path
from native_qsa40_matched_experiment_v1 import arm_environment

def require(ok,message):
 if not ok:raise ValueError(message)
def set_arg(argv,key,value):
 require(argv.count(key)==1,'Unique native option required '+key);argv=list(argv);argv[argv.index(key)+1]=value;return argv
def recipe(native_argv,native_env,source_binding_sha256,generation):
 require(type(generation)is int and generation in(52,58,59),'Actual current source generation required')
 require(type(source_binding_sha256)is str and len(source_binding_sha256)==64 and all(x in '0123456789abcdef'for x in source_binding_sha256),'Actual frozen capture source binding required')
 require(type(native_argv)is list and native_argv[:2]==['/build/strata','--serve'],'Original current native executable/serve protocol required')
 require('--mtp'not in native_argv and '--batch-mtp'not in native_argv and '--tokens'not in native_argv,'Fresh protocol input only; no stock MTP/captured tokens')
 env=copy.deepcopy(native_env);require(env.get('STRATA_SYCL_NATIVE_HC')=='1' and env.get('STRATA_QSA_INDEXER_MATERIALIZED_HALF')=='0','Current native mode0 required')
 require(all(env.get(k,'0')=='0'for k in ('STRATA_HC_Q8','STRATA_HC_Q8_INJECT','STRATA_QFUSE','STRATA_GR_V3','STRATA_SYCL_NATIVE_GROUP_F16','STRATA_VERIFY_HC_T2_REUSE','STRATA_EXPERT_PAIR_REUSE')),'No arithmetic/operator flags stacked')
 for k in ('STRATA_DECODE_TIMING','STRATA_VERIFY_DEBUG','STRATA_VERIFY_PROFILE','STRATA_VERIFY_EAGER'):env.pop(k,None)
 env.update(STRATA_CRITICAL_PATH_TRACE='0',STRATA_BATCH_FIDELITY_DIAG='0',STRATA_BATCH_PREFILL_LAYER_COOPERATE='0',STRATA_BATCH_PREFILL_SINGLE_WINDOW='0',STRATA_ORIGINAL_MTP53='0')
 # Explicit serial diagnostic: unchanged current math, no concurrent-service claim.
 argv=set_arg(native_argv,'--batch','0');argv=set_arg(argv,'--prompt-cache','0')
 for k in ('--conversation-cache-mib','--conversation-cache-slots','--suffix-draft','--lookup-chain','--adapt-every','--adapt-swaps','--pipeline-windows'):
  argv=set_arg(argv,k,'0')if k in argv else argv+[k,'0']
 require(argv.count('--layer-split')==1 and argv[argv.index('--layer-split')+1]=='24'and argv.count('--split-device')==1 and argv[argv.index('--split-device')+1]=='1','Actual current paired24/24 required')
 require(env.get('ZE_AFFINITY_MASK')=='0,1'and '--no-prefill-borrow'in argv,'Owned pair/noBorrow required')
 env=arm_environment(env,False,source_binding_sha256,'/results/qsa-unused')
 env.update(STRATA_LAYER0_Q8_DIAG='1',STRATA_LAYER0_Q8_DIAG_DIR='/results/layer0',STRATA_LAYER0_Q8_DIAG_ARM='/results/ARM',STRATA_LAYER0_Q8_DIAG_BINDING_SHA256=source_binding_sha256,SYCL_UR_TRACE='2')
 requests=[{'ordinal':i,'ids':[248045],'fresh':1,'pin':None,'max_new':1}for i in range(1,5)]
 return {'schema':1,'engine_generation':generation,'materialized_mode':0,'native_argv':argv,'native_env':env,'requests':requests,'stages':[[0,0,24],[1,24,48]],'raw_layout':'existing NUM33 +P30 input/attention/FFN all48 +SFD full49/head +PLE-input33','observer_enabled':True,'extra44_core_RMS_fields_observed':False,'QSA3_target_enabled':False,'pipe_limits':{'bytes':128<<20,'lines':1000000,'line_bytes':1<<20},'required_directories':['captures','p30','ple-input','layer0'],'required_ARM_policy':'Original producer must arm exact request ordinal/binding before GEN and disarm after complete original raw publish','fresh_zero_state_proof':'NUM33 recurrent/conv before arrays bytezero EACH request, PCL fresh/reused0 +actual owning UR joins','runtime_lifecycle_integrated':False,'launch_authorized':False,'source_only_recipe':True,'full_model_math_qualified':False,'API_qualified':False,'concurrency_qualified':False}
