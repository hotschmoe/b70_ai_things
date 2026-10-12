"""Root-only independent original firstHC/GDN arithmetic vs admitted current59 NUM33."""
import argparse,hashlib,json,signal,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from serial37_canonical_json_v3 import read_unique,canonical
from full48_owned_composition_storage_v1 import BoundOriginalFull48Rows,OriginalStorageProjector,OwnedGdn
from original_first_gdn_layer_v1 import Geometry
from original_native_hc_bulk_f32_v1 import RowsDot32
import original_native_hc_f32_v1 as hc
from original_q80_q81_wide32_reference_v1 import OwnFirstGdnArithmeticProjector
from current_first_gdn_bulk_observation_v1 import independent_first
from source43_num10_original_reference_v3 import SHAPES,PACKETS,consume
from compare_current_full48_num10_v1 import metrics
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PLAN=HERE/'current59-first-gdn-reference-source-plan-v1.json'
ELF_SHA='83c289ccd11ef6fed8cb9c0563328795f1fbd665110edace7efcfb4fc710fb47'
FIRST_HC_SHA='936e75f64ad8384501f76e4a267b47fa5abdb1ff5f44cbf38c0d4496649fc38e'
SDK_SHA='df52f1b799d013e84438dc50e77c4b81b7d5f655fa04ba14c0af05b6d0ea74d0'
def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='ascii')
def source_binding():
 before=sha(PLAN);p=read_unique(PLAN)
 for name,want in p['files'].items():require(sha(ROOT/name)==want,'Independent firstGDN source changed '+name)
 require(sha(PLAN)==before,'Independent firstGDN plan changed');return before

def targets(root,producer_sha,public_path,public_sha):
 root=Path(root).resolve();require(sha(root/'report.json')==producer_sha and sha(public_path)==public_sha,'Exact current producer/public target reports required');report=read_unique(root/'report.json');public=read_unique(public_path)
 require(producer_sha=='2efcaa63076d6421ea3476e29aea62c44e38809d2199c42ca23d6a54ae947bc8' and report['passed']is False and report['errors']==['ValueError: Actual exact fresh4/MAX1/absentPIN request required'] and report['SDK_receipt_sha256']==SDK_SHA and public['passed']is True and public['source_generation']==59 and type(public['materialized_mode'])is int and public['materialized_mode']==0 and public['original_producer_report_sha256']==producer_sha and public['original_producer_passed']is False and public['producer_status_rewritten']is False and public['original_V3_public_gate_passed']is False and public['original_normal_native_rc']==0 and public['original_posthealth_full4_validated']is True,'Exact admitted recovery component/failedproducer current59 mode0 targets required')
 raw=public['current_mode0_targets'];require(raw['source_generation']==59 and raw['materialized_mode']==0 and raw['native_numerical_observer_field_count']==33 and raw['extra44_core_RMS_fields_observed']is False and len(raw['requests'])==4,'Exact currentNUM33/no invented core targets required')
 directory=root/'source59-mode0-fidelity';recipe=read_unique(directory/'recipe.json')
 require(sha(directory/'recipe.json')==report['artifacts']['source59-mode0-fidelity/recipe.json'] and recipe['engine_generation']==59 and recipe['materialized_mode']==0 and recipe['native_env']['STRATA_QFUSE']=='0' and recipe['native_env'].get('STRATA_Q8_PACKED','0')=='0' and recipe['native_env'].get('STRATA_MMVQ_WIDE_32','1')=='1','Exact current unpacked wide32 source-expression dispatch required')
 require(recipe['native_argv'][recipe['native_argv'].index('--batch')+1]=='0' and recipe['native_argv'][recipe['native_argv'].index('--prompt-cache')+1]=='0' and '--mtp'not in recipe['native_argv'],'Independent first row requires current fresh serial/noMTP scope')
 output=[];pins={root/'report.json':producer_sha,Path(public_path):public_sha,directory/'recipe.json':sha(directory/'recipe.json')}
 for r in raw['requests']:
  require(r['raw']['ids']==[248045] and r['raw']['fresh']==1 and r['raw']['pin']is None and r['layer0']['frame']['position']==0 and r['layer0']['frame']['token']==248045 and r['layer0']['frame']['pid']==1,'Current literal/zero-position target differs')
  f={v['name']:v for v in r['layer0']['observed']};values={};packets={}
  for name,shape in SHAPES.items():
   p=Path(f[name]['path']);require(p.resolve()==p and p.parent==directory/'layer0','Original target path escaped current capture');data=consume(p,int(np.prod(shape))*4,f[name]['sha256']);require(report['artifacts'][str(p.relative_to(root))]==f[name]['sha256'],'Original producer target artifact mismatch');pins[p]=f[name]['sha256'];values[name]=np.frombuffer(data,dtype='<f4').reshape(shape).copy()
  for name,size in PACKETS.items():
   p=Path(f[name]['path']);require(p.resolve()==p and p.parent==directory/'layer0','Original target packet escaped');data=consume(p,size,f[name]['sha256']);require(report['artifacts'][str(p.relative_to(root))]==f[name]['sha256'],'Original packet artifact mismatch');pins[p]=f[name]['sha256'];packets[name]=data
  require(not np.any(values['gdn_state_before'])and not np.any(values['gdn_conv_before']),'Each native target before-state must bytezero')
  output.append({'ordinal':r['ordinal'],'fields':values,'packets':packets})
 require([v['ordinal']for v in output]==[1,2,3,4],'Exactly current four fresh original ordinals required')
 return output,pins

def compute_original(provider,bulk_elf):
 # NEW own-state instance. No executed full48 module/global is patched.
 require(provider.actual_source is True,'Actual original-weight provider required');g=Geometry();base=OriginalStorageProjector(provider);projector=OwnFirstGdnArithmeticProjector(base);rows=RowsDot32(bulk_elf,ELF_SHA)
 model=SimpleNamespace(p=provider,g=g,gdn={0:OwnedGdn(provider,projector,0,g,{'STRATA_SYCL_NATIVE_HC':'1'})},hc_read=lambda residual,stem:rows.hc_read_original(base,residual,stem,g.eps),hc_write=hc.hc_write_original)
 result=independent_first(model);rows.recheck();result.update(explicit_bulk_calls=rows.calls,first_GDN_projection_events=projector.events,reference_lane='original-own-prefix1-HC-bulk-plus-Q80Q81-wide32-and-BF16-GDNAB-FMA-candidate',captured_inputs_used=False,full_model_math_qualified=False,mode0_first_layer_GDN_no_QSA_indexer_executed=True,device_intrinsic_or_contraction_qualified=False)
 return result

def execute(a):
 require(__debug__,'Unoptimized independent reference required');out=a.output.resolve();require(not out.exists(),'New independent reference output required');out.mkdir();report={'schema':1,'passed':False,'errors':[],'started_epoch':time.time(),'captured_inputs_used':False,'numeric_pass_assigned':False,'full_model_math_qualified':False,'current_mode0_full48_reference_implemented':False};pins={};original_error=None
 try:
  source=source_binding();require(sha(a.model_identity)==a.expected_identity_sha256,'Exact fresh original full4 receipt required');native,pins=targets(a.native_root,a.producer_report_sha256,a.public_binding,a.public_binding_sha256)
  # FirstHC transport proof only; no historical GPU/model authority transfer.
  from current_full48_own_state_bulk_v1 import first_HC_gate
  first_HC_gate(a.first_HC_report,FIRST_HC_SHA,ELF_SHA)
  provider=BoundOriginalFull48Rows(HERE/'original-gguf-reference-foundation-plan-v1.json',a.model_identity);owned=compute_original(provider,a.bulk_elf);report['computation_terminal_epoch']=time.time()
  arrays={}
  for group in('fields','own_intermediates'):
   for name,value in owned[group].items():
    x=np.asarray(value,dtype='<f4');require(np.isfinite(x).all(),'Independent arithmetic nonfinite '+name);p=out/(name+'.f32');p.write_bytes(x.tobytes());arrays[name]={'shape':list(x.shape),'bytes':x.nbytes,'sha256':sha(p),'current_native_target_observed':name in SHAPES}
  for name,data in owned['packets'].items():(out/(name+'.q81')).write_bytes(data)
  comparisons=[{'request':r['ordinal'],'fields':{name:metrics(value,r['fields'][name])for name,value in owned['fields'].items()},'packet_bitwise':{name:data==r['packets'][name]for name,data in owned['packets'].items()}}for r in native]
  report.update(arrays=arrays,comparisons=comparisons,source_plan_sha256=source,native_producer_report_sha256=a.producer_report_sha256,native_public_binding_sha256=a.public_binding_sha256,model_identity_sha256=a.expected_identity_sha256,source_generation=59,materialized_mode=0,accepted_ids=[248045],own_before_state_zero=True,independent_computations=1,current_native_comparisons=4,reference_lane=owned['reference_lane'],projection_events=owned['first_GDN_projection_events'],bulk_calls=owned['explicit_bulk_calls'],extra_native_core_RMS_fields_observed=False,device_intrinsic_or_contraction_qualified=False,first_layer_only=True)
 except BaseException as exc:report['errors'].append(type(exc).__name__+': '+str(exc));original_error=exc
 finally:
  # All actual provider/target reads have ended BEFORE the fresh closing scan,
  # including failures. Do not recursively replay current capture/model admission.
  try:
   require(source_binding()==locals().get('source',source_binding()),'Closing source changed')
   for p,want in pins.items():require(sha(p)==want,'Consumed original target metadata/bytes changed '+str(p))
   require(sha(a.model_identity)==a.expected_identity_sha256,'Caller fresh identity bytes changed')
  except BaseException as exc:report['errors'].append('read-closure: '+type(exc).__name__+': '+str(exc))
  try:
   from run_source44_num38_original_reference_v2 import closing_identity
   report['postCPU_full4']=closing_identity(out,time.time())
  except BaseException as exc:report['errors'].append('fresh-closing: '+type(exc).__name__+': '+str(exc))
  report['finished_epoch']=time.time();report['passed']=not report['errors'];report['artifacts']={str(p.relative_to(out)):sha(p)for p in out.rglob('*')if p.is_file()and p.name!='report.json'};write(out/'report.json',report)
 require(report['passed'],'Independent current firstGDN computation/source/integrity failed');return report

def main():
 p=argparse.ArgumentParser()
 for name in('native-root','public-binding','model-identity','bulk-elf','first-HC-report','output'):p.add_argument('--'+name,type=Path,required=True)
 for name in('producer-report-sha256','public-binding-sha256','expected-identity-sha256'):p.add_argument('--'+name,required=True)
 a=p.parse_args()
 for sig in(signal.SIGHUP,signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:(_ for _ in()).throw(InterruptedError('Controlled independent original CPU interruption')))
 execute(a)
if __name__=='__main__':main()
