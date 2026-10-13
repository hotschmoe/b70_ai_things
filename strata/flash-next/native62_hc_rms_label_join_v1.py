"""Named producer-label association for authentic fresh/no-disk HC capture only."""
import json
from pathlib import Path
from serial37_canonical_json_v3 import canonical
from native62_hc_rms_comparison_v2 import sha,require
LABELS=('STRATA_ARTIFACT_IDENTITY_SHA256','STRATA_CRITICAL_PATH_TRACE_BINDING_SHA256','STRATA_LAYER0_Q8_DIAG_BINDING_SHA256','STRATA_PLE_INPUT33_BINDING_SHA256','STRATA_PREFIX30_BINDING_SHA256')
SOURCES={'sycl/include/strata/core/api_state_identity.hpp':['const char* digest=std::getenv("STRATA_ARTIFACT_IDENTITY_SHA256")','out+="|artifact=";out+=digest;'],'sycl/include/strata/program/host_critical_path_trace.hpp':['if(!flag||std::strcmp(flag,"0")==0)return true;','std::strcpy(s.binding,b);'],'sycl/include/strata/core/layer0_numerical_observer.hpp':['binding=std::getenv("STRATA_LAYER0_Q8_DIAG_BINDING_SHA256")','settings().binding'],'include/strata/core/ple_input_observer33.hpp':['binding=std::getenv("STRATA_PLE_INPUT33_BINDING_SHA256")','settings().binding'],'sycl/include/strata/core/prefix_residual_capture.hpp':['binding=std::getenv("STRATA_PREFIX30_BINDING_SHA256")','settings().binding'],'sycl/src/program/generate.cpp':['auto session_identity =','if (!session_identity(id, err,','if (!stages.empty() || multi_gpu)','if (o.prompt_cache <= 0)']}
def fresh_only(recipe,stdin_path):
 require(recipe['materialized_mode']==0 and recipe['cards']==[0,1]and recipe['stages']==[[0,0,24],[1,24,48]]and recipe['native_env']['STRATA_CRITICAL_PATH_TRACE']=='0','Exact H36disabled paired scope required');argv=recipe['native_argv']
 for flag in('--prompt-cache','--conversation-cache-mib','--conversation-cache-slots','--batch'):
  require(argv.count(flag)==1 and argv[argv.index(flag)+1]=='0','No-cache profile required for identity association')
 require(len(recipe['requests'])==4 and all(type(r['fresh'])is int and r['fresh']==1 and r['ids']==[248045]and r['pin']is None and r['max_new']==1 for r in recipe['requests']),'Four exact fresh/noPIN rows required')
 raw=Path(stdin_path).read_bytes();require(raw.endswith(b'\n')and len(raw)<16384,'Original complete boundedstdin required');rows=[json.loads(line)for line in raw.splitlines()];require([r['seq']for r in rows]==list(range(5))and [r['packet']for r in rows]==['GEN 1 fresh=1 ckpt=1 temperature=0 248045\n']*4+['QUIT\n'],'Only original GEN4/QUIT allowed; no SAVE/RESTORE/load/session control')

def source_consumers(sdk,proposal):
 expected=proposal['expected_patched_source_sha256'];result={}
 for n,needles in SOURCES.items():
  p=Path(sdk)/'source'/n;require(sha(p)==expected[n],'Actual compiled label consumer changed '+n);text=p.read_text(encoding='ascii');require(all(x in text for x in needles),'Expected actual consumer semantics changed '+n);result[n]=expected[n]
 return result

def matched(left,right,left_binding,right_binding,left_stdin,right_stdin,sdk,proposal):
 fresh_only(left,left_stdin);fresh_only(right,right_stdin);consumers=source_consumers(sdk,proposal);require(left['native_argv']==right['native_argv']and canonical(left['requests'])==canonical(right['requests'])and canonical(left['expert_profile_binding'])==canonical(right['expert_profile_binding']),'Actual model/config/input/profile differ');a=dict(left['native_env']);b=dict(right['native_env']);pairs={}
 for k in LABELS:
  require(a.get(k)==left_binding and b.get(k)==right_binding,'Each named actual label must match its own authenticated producer source SHA '+k);pairs[k]={'OFF':a[k],'ON':b[k]};a[k]=b[k]='VALIDATED_PRODUCER_SOURCE_BINDING'
 require(a.pop('STRATA_HC_EXPLICIT_SQRT_DIVIDE')=='0'and b.pop('STRATA_HC_EXPLICIT_SQRT_DIVIDE')=='1'and canonical(a)==canonical(b),'Only explicit sqrt/divide arithmetic delta allowed after named source-label joins')
 return {'named_producer_labels':pairs,'compiled_label_consumer_source_sha256':consumers,'H36_initialization_DISABLED':True,'observer_labels_metadata_only':True,'artifact_identity_disk_session_consumer_inactive':True,'actualstdin_GEN4_QUIT_ONLY':True,'math_delta_only':'STRATA_HC_EXPLICIT_SQRT_DIVIDE0/1','global_ENV_ignoring_or_stat_normalization':False}
