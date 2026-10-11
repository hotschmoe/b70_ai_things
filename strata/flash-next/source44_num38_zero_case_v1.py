"""Four literal prefix1 fresh zero-state NUM38 requests; no old runtime authority."""
import hashlib,json
from pathlib import Path
import source44_sdk_binding_v1 as sdk
from serial37_canonical_json_v3 import canonical
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];IDS=[248045]

def requests():return [{'ordinal':i,'ids':list(IDS),'fresh':1,'pin':None,'max_new':1}for i in range(1,5)]
def recipe(metadata,cfg,binding,enabled):
    from prepare_decode40_host_profile_v1 import build
    sdk.require(type(enabled)is bool,'Typed NUM38 arm required');shape=build(metadata,cfg,{'short':IDS,'prefill_about512':IDS},binding,'OFF');env=dict(shape['native_env']);env.update(STRATA_QSA_INDEXER_MATERIALIZED_HALF='1',STRATA_FIDELITY_DIAG='1',STRATA_FIDELITY_DIAG_ACTIVATIONS='1',STRATA_FIDELITY_DIAG_ARM='/results/ARM',STRATA_FIDELITY_DIAG_DIR='/results/captures',STRATA_PREFIX_DIAG='1',STRATA_PREFIX_LIFECYCLE_DIAG='1',STRATA_PREFIX_DIAG_ARM='/results/ARM',STRATA_LAYER0_Q8_DIAG=str(int(enabled)),STRATA_LAYER0_Q8_DIAG_DIR='/results/layer0',STRATA_LAYER0_Q8_DIAG_ARM='/results/ARM',STRATA_LAYER0_Q8_DIAG_BINDING_SHA256=binding,STRATA_ARTIFACT_IDENTITY_SHA256=binding,STRATA_BATCH_FIDELITY_DIAG='0',SYCL_UR_TRACE='2')
    sdk.require(env.get('STRATA_QSA3_TARGET','0')=='0'and env.get('STRATA_CRITICAL_PATH_TRACE','0')=='0','Simultaneous QSA/H36 observer forbidden');return {'engine_generation':44,'mode':'source44-NUM38-'+('ON'if enabled else'OFF'),'materialized_mode':1,'NUM38_enabled':enabled,'observer_enabled':enabled,'native_env':env,'native_argv':['/build/strata','--serve',*cfg['args']],'requests':requests(),'pipe_limits':{'bytes':128<<20,'lines':1000000,'line_bytes':1<<20},'API_qualified':False,'full_model_math_qualified':False}

def collect(raw,directory,binding):
    from source44_num38_fields_v1 import collect_frame
    sdk.require(raw['ids']==IDS and type(raw['fresh'])is int and raw['fresh']==1 and raw['pin']is None,'Actual prefix1 fresh absent-PIN required');record=collect_frame(raw,Path(directory),binding);sdk.require(all(type(record['frame'][k])is int for k in ('schema','pid','request','stage','layer','rows','position','token','reused','completed_nonce','graph_key')),'Exact typed native frame identities required');fields={f['name']:f for f in record['observed']}
    for name,size in [('gdn_state_before',3145728),('gdn_conv_before',122880)]:
        field=fields[name];data=Path(field['path']).read_bytes();sdk.require(field['bytes']==len(data)==size and not any(data),'Actual each-request native recurrent/conv state must be bytezero')
    sdk.require(record['frame']['position']==0 and record['frame']['token']==248045 and record['frame']['reused']==0,'Actual first-row work differs');return record
