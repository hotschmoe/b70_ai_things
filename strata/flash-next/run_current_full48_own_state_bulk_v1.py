#!/usr/bin/env python3
"""Root-only original full48 prefix1 CPU candidate, no native qualification."""
import argparse,hashlib,json,signal,time
from pathlib import Path
import numpy as np
import current_full48_own_state_bulk_v1 as candidate
from full48_owned_composition_storage_v1 import BoundOriginalFull48Rows
from serial37_canonical_json_v3 import read_unique
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];PLAN=HERE/'current-full48-own-state-bulk-source-plan-v1.json'
require=candidate.require
ELF_SHA='83c289ccd11ef6fed8cb9c0563328795f1fbd665110edace7efcfb4fc710fb47'
FIRST_SHA='936e75f64ad8384501f76e4a267b47fa5abdb1ff5f44cbf38c0d4496649fc38e'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def source_binding():
 plan=read_unique(PLAN)
 for name,want in plan['files'].items():require(sha(ROOT/name)==want,'Own-state software source changed '+name)
 return sha(PLAN)
def execute(args):
 require(__debug__,'Unoptimized Python required');out=args.output.resolve();require(not out.exists(),'New own-state output required');out.mkdir();report={'schema':1,'passed':False,'errors':[],'started_epoch':time.time(),'full_model_math_qualified':False,'numeric_pass_claim':False,'GPU_execution':False,'captured_inputs_used':False,'state_initialized_from_zero':True}
 try:
  source=source_binding();require(sha(args.config)==args.expected_config_sha256 and sha(args.model_identity)==args.expected_identity_sha256,'Exact caller supplied config/fresh identity bytes required');cfg=read_unique(args.config);require(type(cfg)is dict and type(cfg['args'])is list and type(cfg['env'])is dict,'Actual baseline config args/env required')
  candidate.first_HC_gate(args.first_HC_report,FIRST_SHA,ELF_SHA);candidate.schedule([248045],cfg['args'],cfg['env']);candidate.source_contract(args.generation)
  provider=BoundOriginalFull48Rows(HERE/'original-gguf-reference-foundation-plan-v1.json',args.model_identity)
  model=candidate.CurrentFull48OwnStateBulk(provider,args.expected_identity_sha256,cfg['args'],cfg['env'],generation=args.generation,rows_dot_elf=args.bulk_elf,rows_dot_sha256=ELF_SHA,first_HC_report=args.first_HC_report,first_HC_report_sha256=FIRST_SHA)
  owned=model.tokens([248045]);report['computation_terminal_epoch']=time.time();require(len(owned['trace'])==1 and len(owned['trace'][0]['layers'])==48,'Exact own original prefix1/full48 required')
  def save(name,value):
   a=np.asarray(value,dtype='<f4');require(np.isfinite(a).all(),'Nonfinite own state '+name);(out/name).write_bytes(a.tobytes());return {'shape':list(a.shape),'bytes':a.nbytes,'sha256':sha(out/name)}
  arrays={}
  for layer in owned['trace'][0]['layers']:
   for key in('input','attention','ffn'):arrays['l%d_%s'%(layer['layer'],key)]=save('own-l%d-%s.f32'%(layer['layer'],key),layer[key])
  arrays['head']=save('own-first-head.f32',owned['first_generated_logits'])
  for layer,state in owned['gdn_states_owned'].items():
   for key,value in state.items():arrays['gdn%d_%s'%(layer,key)]=save('own-gdn%d-%s.f32'%(layer,key),value)
  for layer,state in owned['qsa_states_owned'].items():
   for key in('keys','values','tail','dead','pooled'):arrays['qsa%d_%s'%(layer,key)]=save('own-qsa%d-%s.f32'%(layer,key),getattr(state,key))
  arrays['PLE_history']=save('own-PLE-history.f32',owned['ple_history_owned']);report.update(arrays=arrays,source_plan_sha256=source,model_identity_sha256=args.expected_identity_sha256,config_sha256=args.expected_config_sha256,source_generation=args.generation,materialized_mode=1,accepted_ids=[248045],bulk_calls=model.rows_backend.calls,explicit_bulk_ELF_sha256=ELF_SHA,first_HC_report_sha256=FIRST_SHA,declared_schedule=owned['declared_schedule'],packet_events=owned['packet_events'],unsupported=owned['unsupported'],host_rsqrt_expf_device_intrinsics_qualified=False,actual_window_group_rounding_qualified=False,mode0_storage_semantics_qualified=False,current_native_qualification_transferred=False)
  require(source_binding()==source and sha(args.config)==args.expected_config_sha256 and sha(args.model_identity)==args.expected_identity_sha256,'Closing caller/software bytes changed')
 except BaseException as exc:report['errors'].append(type(exc).__name__+': '+str(exc))
 finally:
  try:
   from run_source44_num38_original_reference_v2 import closing_identity
   report['postCPU_full4']=closing_identity(out,time.time())
  except BaseException as exc:report['errors'].append('fresh-closing: '+type(exc).__name__+': '+str(exc))
  try:report['closing_source_plan_sha256']=source_binding()
  except BaseException as exc:report['errors'].append('closing-source: '+type(exc).__name__+': '+str(exc))
  report['finished_epoch']=time.time();report['passed']=not report['errors'];report['artifacts']={str(p.relative_to(out)):sha(p)for p in sorted(out.rglob('*'))if p.is_file()and p.name!='report.json'};(out/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='ascii')
 require(report['passed'],'Own full48 candidate compute/fresh closing failed');return report

def main():
 p=argparse.ArgumentParser();p.add_argument('--generation',type=int,choices=[43,47],required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--expected-config-sha256',required=True);p.add_argument('--model-identity',type=Path,required=True);p.add_argument('--expected-identity-sha256',required=True);p.add_argument('--bulk-elf',type=Path,required=True);p.add_argument('--first-HC-report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 for sig in(signal.SIGTERM,signal.SIGHUP,signal.SIGINT):signal.signal(sig,lambda *_:(_ for _ in()).throw(InterruptedError('Controlled original full48 CPU interruption')))
 execute(a)
if __name__=='__main__':main()
