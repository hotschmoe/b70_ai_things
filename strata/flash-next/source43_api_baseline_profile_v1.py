"""Complete mode1 API configuration over authenticated matched43 normal intake."""
import copy,json
from pathlib import Path
from serial37_canonical_json_v3 import canonical
import run_source43_normal_model_cli_v1 as normal
require=normal.require
ALIAS='qwen3.8-flash-next-Unsloth-UD-Q4_K_XL-strata-native-hc-ple-mtp0-source43-materializedhalf1-api-baseline-fp16kv-ctx2048-prefill64-ple65536-seg1024-adapt0-borrow0-gpu0'

def expected_config(base_cfg,port):
    require(type(port)is int and 1024<=port<=65535,'Typed dedicated API port required');require(base_cfg['model_name']=='hotschmoe-dd'and base_cfg['backend']=='sycl'and base_cfg['parallel']==1,'Authenticated original single-native API template required')
    cfg=copy.deepcopy(base_cfg);cfg.update(model_name='hotschmoe-dd',aliases=[ALIAS],port=port,artifact_identity_manifest='/results/artifact-identity.json');cfg['env']['STRATA_QSA_INDEXER_MATERIALIZED_HALF']='1'
    args=cfg['args']
    for key,value in(('--max-context','2048'),('--prefill','64'),('--ple-row-cache','65536'),('--batch','0'),('--prompt-cache','0'),('--suffix-draft','0')):require(args.count(key)==1 and args[args.index(key)+1]==value,'Exact normal API geometry/cache argument '+key)
    require('--conversation-cache-mib'not in args or args[args.index('--conversation-cache-mib')+1]=='0','Conversation cache must remain source-default0 or explicit0')
    require(not any(k in args for k in('--mtp','--layer-split','--split-device')),'Initial genuine43 API baseline is singleGPU/noMTP');normal.life.recipe.client.observer_off({k:v for k,v in cfg['env'].items()if k!='STRATA_QSA_INDEXER_MATERIALIZED_HALF'})
    return cfg

def preview(normal_plan,normal_control,port,output):
    """Root metadata/source intake, not an API runtime or registry waiver."""
    output=Path(output);require(not output.exists(),'New preview output required');plan=normal.c.read(normal_plan);metadata,cfg=normal.admission(plan)
    from validate_source43_normal_model_cli_v1 import finalized_binding
    actual=finalized_binding(normal_control);require(actual['source_generation']==43 and actual['materialized_mode']==0 and actual['OFF_equivalence_passed']is True,'Actual freshly admitted43 normal control required')
    final=expected_config(cfg,port);output.mkdir();normal.c.write(output/'server-config.json',final);normal.c.write(output/'preview.json',{'schema':1,'actual43_sdk':plan['source43'],'actual43_upload':plan['fresh43_upload'],'actual43_masked':plan['fresh43_masked_component'],'actual43_normal_control':actual,'normal_plan_path':str(Path(normal_plan).resolve()),'normal_plan_sha256':normal.c.sha(normal_plan),'original_template_sha256':plan['template_config_sha256'],'configuration_sha256':normal.c.sha(output/'server-config.json'),'primary_model_name':'hotschmoe-dd','research_alias':ALIAS,'mode':1,'API_launch_allowed':False,'unresolved':['Signed actual43 artifact identity manifest/new controller generation/source proof','Current174 association and old173 prerequisite consumer routing','Owned C140 launch/screen/2-client/retirement parent and public final source joins'],'full_model_math_qualified':False});return final
