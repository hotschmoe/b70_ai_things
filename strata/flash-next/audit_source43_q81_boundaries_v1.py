#!/usr/bin/env python3
"""Weight-free captured-input packet diagnostic; never an original reference."""
import argparse,hashlib,json,os,re,time
from pathlib import Path
import audit_source43_t1_conv_commit_v2 as prior
from independent_q8_1_activation_v1 import encode,require_packet_exact,NATIVE
from serial37_canonical_json_v3 import canonical
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PLAN=HERE/'source43-q81-boundary-audit-source-plan-v1.json'
SEAMS=(('attn_mixed','attn_input_q81',2560),('gdn_output_gated','gdn_output_q81',6144))
require=prior.require
def source_binding():
 raw=PLAN.read_bytes();plan=prior.parse(raw)
 for name,want in plan['files'].items():prior.consumed(ROOT/name,want,16<<20)
 require(PLAN.read_bytes()==raw,'Auditor plan changed');return hashlib.sha256(raw).hexdigest()
def packet_comparison(raw,packet,cols):
 require(type(cols)is int and cols in(2560,6144)and len(raw)==cols*4 and len(packet)==cols//32*36,'Exact declared packet seam bounds required')
 expected=encode(raw,1,cols,NATIVE);exact=True
 try:require_packet_exact(raw,packet,1,cols)
 except ValueError:
  require(expected!=packet,'Unexpected packet predicate failure');exact=False
 different=[i for i,(a,b)in enumerate(zip(expected,packet))if a!=b];first=different[0]if different else None
 return {'packet_exact':exact,'different_bytes':len(different),'first_differing_byte':first,'first_differing_block':None if first is None else first//36,'first_differing_field':None if first is None else 'scale_half'if first%36<2 else 'sum_half'if first%36<4 else 'code','expected_packet_sha256':hashlib.sha256(expected).hexdigest(),'actual_packet_sha256':hashlib.sha256(packet).hexdigest(),'raw_input_sha256':hashlib.sha256(raw).hexdigest(),'rows':1,'cols':cols,'contract':NATIVE,'captured_native_input_used_for_local_diagnostic':True,'original_reference_replaced':False,'numeric_tolerance_used':False}
def audit(root,report,report_sha,reference,consume):
 require(reference['passed']is True and reference['errors']==[]and reference['source_plan_sha256']==prior.CLI2_SOURCE and reference['actor_root']==str(root)and reference['expected_actor_report_sha256']==report_sha and reference['actual_reference_computation_attempted']is True,'Exact closed original CLI2 context required')
 compact=reference['current43_capture_before'];require(canonical(compact)==canonical(reference['current43_capture_after']),'Sealed compact capture changed')
 capture=root/'source43-num10-ON';pid=compact['engine_pid'];require(type(pid)is int and pid>0 and type(compact['source_generation'])is int and compact['source_generation']==43 and type(compact['materialized_mode'])is int and compact['materialized_mode']==1 and compact['capture_root']==str(capture),'Exact current43 capture owner required')
 require(report['passed']is True and report['errors']==[]and compact['public_report_sha256']==report_sha and compact['public_source_plan_sha256']==report['source_plan_sha256']==compact['binding_sha256']==prior.ACTOR_SOURCE and compact['plan_sha256']==report['plan_sha256'],'Original source/corpus/report association required')
 require(re.fullmatch('[0-9a-f]{64}',compact['full_descriptor_sha256'])and len(compact['requests'])==4,'Four sealed requests/full-descriptor digest required')
 contract=[(n,e,int(b))for n,e,b in re.findall(r'\{"([^"]+)","([^"]+)",(\d+)\}',(HERE/'layer0_numerical_contract_v1.hpp').read_text())];require(len(contract)==33,'Exact frozen raw33 layout required');results=[];seen=[]
 for index,row in enumerate(compact['requests'],1):
  require(type(row['ordinal'])is int and row['ordinal']==index and type(row['input_ids'])is list and all(type(v)is int for v in row['input_ids'])and row['input_ids']==[248045],'Exact accepted first token/ordinal required')
  meta=Path(row['metadata']);require(meta.parent==capture/'layer0'and report['artifacts'][str(meta.relative_to(root))]==row['metadata_sha256'],'Original metadata artifact association required');frame=prior.parse(consume(meta,row['metadata_sha256'],1<<20))
  wanted={'pid':pid,'token':248045,'position':0,'rows':1,'layer':0,'stage':0,'reused':0}
  require(all(type(frame[k])is int and frame[k]==v for k,v in wanted.items())and type(frame['gen_ids'])is list and all(type(v)is int for v in frame['gen_ids'])and frame['gen_ids']==[248045],'Typed current native T1 boundary required');request=frame['request'];require(type(request)is int and request>0 and request not in seen and type(frame['completed_nonce'])is int and frame['completed_nonce']==((pid<<32)|request)and frame['binding_sha256']==compact['binding_sha256'],'Exact native request/nonce/source required');seen.append(request)
  fields=row['raw33'];require(len(fields)==33 and [(f['name'],f['encoding'],f['bytes'])for f in fields]==contract and all(type(f['bytes'])is int for f in fields),'Exact source-shaped raw33 roster required')
  require([(f['name'],f['encoding'],f['bytes'])for f in frame['fields']]==contract,'Metadata raw33 layout differs');byname={f['name']:f for f in fields};checks=[]
  for raw_name,packet_name,cols in SEAMS:
   values=[]
   for name,size in((raw_name,cols*4),(packet_name,cols//32*36)):
    f=byname[name];path=Path(f['path']);require(path.parent==capture/'layer0'and f['provenance']=='actual_buffer'and report['artifacts'][str(path.relative_to(root))]==f['sha256'],'Original selected field ownership/artifact required');values.append(consume(path,f['sha256'],size,size))
   checks.append({'input_field':raw_name,'packet_field':packet_name,**packet_comparison(*values,cols)})
  results.append({'ordinal':index,'native_request':request,'comparisons':checks})
 return {'requests':results,'all_eight_packets_exact':all(c['packet_exact']for r in results for c in r['comparisons']),'saved_full_descriptor_sha256':compact['full_descriptor_sha256'],'full_descriptor_reconstructed':False,'fresh_request_scope':'sealed original CLI2/reader4 context; no new public admission'}
def run(root,report_sha,reference_path,reference_sha,out):
 root=Path(root);out=Path(out);reference_path=Path(reference_path)
 require(root.is_absolute()and root.resolve()==root and root.is_dir()and not any(p.is_symlink()for p in(root,*root.parents)),'Canonical nonaliased original actor root required')
 require(out.is_absolute()and out.resolve()==out and not any(p.is_symlink()for p in(out,*out.parents))and not out.exists()and root!=out and root not in out.parents and reference_path!=out and out not in reference_path.parents,'New external output outside original evidence required');out.mkdir(parents=True)
 result={'schema':1,'passed':False,'errors':[],'producer_pid':os.getpid(),'started_epoch':time.time(),'weights_read':False,'GPU_execution':False,'captured_native_inputs_used_for_local_diagnostic':True,'original_reference_replaced':False,'current_model_or_semantic_admission_performed':False,'full_model_math_qualified':False};bindings=[];source=None
 def consume(path,want,limit,size=None):
  raw,binding=prior.consumed(path,want,limit,size);bindings.append(binding);return raw
 try:
  source=source_binding();result['source_plan_sha256']=source;report=prior.parse(consume(root/'report.json',report_sha,prior.MAX_METADATA));reference=prior.parse(consume(reference_path,reference_sha,1<<20));result['audit']=audit(root,report,report_sha,reference,consume);result['passed']=True
 except BaseException as error:result['errors'].append(type(error).__name__+': '+str(error))
 finally:
  for binding in bindings:
   try:
    _,fresh=prior.consumed(binding['path'],binding['sha256'],binding['bytes'],binding['bytes']);require(fresh==binding,'Original evidence changed across audit')
   except BaseException as error:result['passed']=False;result['errors'].append('Closing: '+str(error))
  try:require(source is not None and source_binding()==source,'Auditor source changed')
  except BaseException as error:result['passed']=False;result['errors'].append('Source closing: '+str(error))
  result['consumed_evidence']=bindings;result['finished_epoch']=time.time();(out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='ascii')
 require(result['passed'],'Packet boundary audit failed; original evidence preserved');return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--actor-root',type=Path,required=True);p.add_argument('--expected-actor-report-sha256',required=True);p.add_argument('--sealed-reference-report',type=Path,required=True);p.add_argument('--expected-reference-report-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(__debug__,'Ordinary Python required');run(a.actor_root,a.expected_actor_report_sha256,a.sealed_reference_report,a.expected_reference_report_sha256,a.output)
