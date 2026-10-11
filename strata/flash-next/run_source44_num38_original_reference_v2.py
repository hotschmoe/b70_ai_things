#!/usr/bin/env python3
"""Root-only original FP64 L0 reference; no native/GPU execution by this CLI."""
import argparse,hashlib,json,os,signal,time
from pathlib import Path
import numpy as np
import source44_num38_original_reference_v2 as reference
from serial37_canonical_json_v3 import canonical,read_unique
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];PLAN=HERE/'source44-num38-original-cli-source-plan-v2.json';MANIFEST=HERE/'original-gguf-reference-foundation-plan-v1.json'
require=reference.require

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def source_binding():
    record=read_unique(PLAN)
    for name,want in record['files'].items():require(sha(ROOT/name)==want,'Exact original CLI source changed '+name)
    return sha(PLAN)
def array_binding(value):
    a=np.asarray(value,dtype='<f4');require(np.isfinite(a).all(),'Finite diagnostic owned field required');raw=a.tobytes()
    return {'shape':list(a.shape),'encoding':'LE_F32 narrowing of original FP64 diagnostic only','bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def capture_binding(value):
    public=value['public_actor_admission']
    return {'capture_root':value['capture_root'],'source_generation':value['source_generation'],'materialized_mode':value['materialized_mode'],'engine_pid':value['engine_pid'],'plan_sha256':value['plan_sha256'],'binding_sha256':value['binding_sha256'],'public_report_sha256':public['report_sha256'],'public_source_plan_sha256':public['source_plan_sha256'],'full_descriptor_sha256':hashlib.sha256(canonical(value).encode('ascii')).hexdigest(),'requests':[{'ordinal':row['ordinal'],'input_ids':row['raw']['ids'],'output_ids':row['raw']['output_ids'],'raw_request_sha256':hashlib.sha256(canonical(row['raw']).encode('ascii')).hexdigest(),'metadata':row['layer0']['metadata'],'metadata_sha256':row['layer0']['metadata_sha256'],'raw38':row['layer0']['observed']}for row in value['requests']]}
def compact(result,targets):
    by_request={t['request']:t for t in targets};primary=[];seams=[]
    for row in result['independent']:
        primary.append({'request':row['request'],'lane':row['lane'],'captured_inputs_used':False,'owned_fields':{n:array_binding(v)for n,v in row['fields_owned'].items()},'comparisons':row['comparisons'],'packet_exact':row['packet_exact']})
    for row in result['conditional']:
        target=by_request[row['request']];v=target['values'];lanes=row['lanes'];mixed=lanes['native_HC_mixed_to_original_zero_GDN'];gated=lanes['native_gated_to_original_Q81_projection'];packet=gated['packet']
        seams.append({'request':row['request'],'captured_native_inputs_used_for_local_diagnostic':True,'reference_replaced':False,'captured_state_used':False,'native_HC_mixed_to_original_zero_GDN':{'owned_block':array_binding(mixed['block']),'owned_gated':array_binding(mixed['gated']),'gated_comparison':reference.metrics(mixed['gated'],v['gdn_output_gated']),'block_comparison':reference.metrics(mixed['block'],v['gdn_block_output']),'state_comparison':reference.metrics(mixed['state']['recurrent'].transpose(1,0,2),v['gdn_state_after']),'conv_comparison':reference.metrics(mixed['state']['conv'].T,v['gdn_conv_after'])},'native_gated_to_original_Q81_projection':{'packet_sha256':hashlib.sha256(packet).hexdigest(),'packet_bytes':len(packet),'packet_bitwise_equal_native':packet==target['packets']['gdn_output_q81'],'owned_output':array_binding(gated['output']),'output_comparison':reference.metrics(gated['output'],v['gdn_block_output']),'stored_sum_used':False,'projection':gated['projection']},'full_model_math_qualified':False})
    core_seams=[{'request':r['request'],**r['lanes']['native_normalized_qkv_to_cpu_F32_core']}for r in result['conditional']]
    return {'conditional_F32_core':core_seams,'primary':primary,'conditional':seams,'lane':'original_first_gdn_layer_v1 FP64 equations; distinct from current F32 wholemodel reference','full_model_math_qualified':False,'numeric_pass_claim':False,'tolerance_fitted':False,'reference_replaced':False,'existing_F32_full48_reference_replaced':False}

def closing_identity(output,boundary):
    from run_source_upload_oracle_full_v2 import full_buffered_identity
    from source44_num38_intake_v2 import guard
    lock=read_unique(HERE/'model-lock.json');shards=[ROOT/lock['destination']/r['path']for r in lock['files']if r['path'].startswith('UD-Q4_K_XL/')];errors=[];before=after=identity=None
    try:before=guard(shards,output,'postCPU-before')
    except BaseException as exc:errors.append('page-before: '+type(exc).__name__+': '+str(exc))
    try:identity=full_buffered_identity(HERE/'model-lock.json',lock,shards,output/'postCPU-full4.json',boundary)
    except BaseException as exc:errors.append('full4: '+type(exc).__name__+': '+str(exc))
    finally:
        try:after=guard(shards,output,'postCPU-after')
        except BaseException as exc:errors.append('page-after: '+type(exc).__name__+': '+str(exc))
    if not errors:
        try:
            require(before['epoch']<=identity['started']<=identity['finished']<=after['epoch'],'Actual postCPU page/full4 brackets differ');require(identity['passed']is True,'Actual postCPU complete publisher identity failed')
        except BaseException as exc:errors.append('closing-predicate: '+type(exc).__name__+': '+str(exc))
    record={'boundary_epoch':boundary,'before_pages':before,'identity':identity,'after_pages':after,'errors':errors,'full4_path':str(output/'postCPU-full4.json'),'full4_sha256':sha(output/'postCPU-full4.json')if(output/'postCPU-full4.json').is_file()else None}
    (output/'closing-receipt.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n',encoding='ascii');require(not errors,'Actual publisher/page closure failed: '+'; '.join(errors));return identity

def execute(actor_root,expected_report_sha256,identity,expected_identity_sha256,output,conditional=False):
    actor_root=Path(actor_root).resolve();identity=Path(identity).resolve();output=Path(output).resolve();require(not output.exists()and output not in(actor_root,identity)and actor_root not in output.parents,'Fresh external original CPU output required');output.mkdir(parents=True)
    report={'schema':1,'producer_pid':os.getpid(),'started_epoch':time.time(),'passed':False,'errors':[],'actor_root':str(actor_root),'expected_actor_report_sha256':expected_report_sha256,'identity_path':str(identity),'expected_identity_sha256':expected_identity_sha256,'actual_reference_computation_attempted':False,'full_model_math_qualified':False,'GPU_execution':False,'API_qualified':False}
    targets=[];source=None
    def actor_admit(root):
        from validate_source44_num38_zero_v2 import captured_binding
        require(sha(Path(root)/'report.json')==expected_report_sha256,'Actual closed actor report differs');binding=captured_binding(root);targets[:]=reference.load_targets(binding,binding['requests']);return binding
    def model_factory():
        from original_first_gdn_layer_v1 import OriginalTensorRows,FirstGdnLayer
        require(identity.is_file()and not identity.is_symlink()and sha(identity)==expected_identity_sha256,'Exact current full4 identity receipt required');report['actual_reference_computation_attempted']=True;return FirstGdnLayer(OriginalTensorRows(MANIFEST,identity))
    try:
        source=source_binding();report['source_plan_sha256']=source;report['foundation_manifest_sha256']=sha(MANIFEST)
        (output/'source-plan.snapshot.json').write_bytes(PLAN.read_bytes());(output/'producer.snapshot.py').write_bytes(Path(__file__).read_bytes())
        result=reference.enclosed(actor_root,model_factory,actor_admit,lambda boundary:closing_identity(output,boundary),conditional)
        report['diagnostics']=compact(result,targets);report['current44_capture_before']=capture_binding(result['current44_capture_before']);report['current44_capture_after']=capture_binding(result['current44_capture_after']);report['closing_capture_read_finished_epoch']=result['closing_capture_read_finished_epoch'];report['publisher_closing_boundary_epoch']=result['publisher_closing_boundary_epoch'];report['postCPU_identity']=result['postCPU_identity']
        require(sha(actor_root/'report.json')==expected_report_sha256 and sha(identity)==expected_identity_sha256,'Input actor/identity receipt changed during reference');require(source_binding()==source,'Final original CLI source changed');report['passed']=True
    except BaseException as exc:report['errors'].append(type(exc).__name__+': '+str(exc))
    finally:
        report['finished_epoch']=time.time();report['artifacts']={str(p.relative_to(output)):sha(p)for p in sorted(output.rglob('*'))if p.is_file()and p.name!='report.json'};(output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='ascii')
    require(report['passed'],'Original FP64 diagnostic or actual closing failed; original artifacts retained');return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--actor-root',type=Path,required=True);p.add_argument('--expected-actor-report-sha256',required=True);p.add_argument('--model-identity',type=Path,required=True);p.add_argument('--expected-identity-sha256',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--conditional',action='store_true');a=p.parse_args();require(__debug__,'Ordinary Python required')
    for sig in(signal.SIGTERM,signal.SIGHUP,signal.SIGINT):signal.signal(sig,lambda number,frame:(_ for _ in()).throw(InterruptedError('Controlled original CPU interruption '+str(number))))
    execute(a.actor_root,a.expected_actor_report_sha256,a.model_identity,a.expected_identity_sha256,a.output,a.conditional)
if __name__=='__main__':main()
