"""Saved native observations only; neither operand is an independent math reference."""
import argparse,json,hashlib,struct,math
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];PLAN=HERE/'prefill-first-gdn-comparison-source-plan-v1.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def need(v,s):
 if not v:raise ValueError(s)
def stat5(p):
 s=p.stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
def metrics(decoder,prefill):
 x=np.frombuffer(decoder,dtype='<f4');y=np.frombuffer(prefill,dtype='<f4');need(x.size==y.size and np.isfinite(x).all()and np.isfinite(y).all(),'Finite same-shape observed F32 fields required')
 xx=x.astype('f8');yy=y.astype('f8');d=yy-xx;energy=float(np.dot(xx,xx));bits=x.view('<u4')!=y.view('<u4');nz=np.flatnonzero(bits)
 return {'words':int(x.size),'bitwise_equal':decoder==prefill,'different_bit_words':int(bits.sum()),'NMSE_decoder_observation_denominator':float(np.dot(d,d)/energy)if energy else(0.0 if not np.any(d)else None),'max_abs':float(np.max(np.abs(d),initial=0)),'first_different_word':int(nz[0])if nz.size else None,'independent_math_reference':False}
def run(a):
 for n,w in json.loads(PLAN.read_bytes())['files'].items():need(sha(ROOT/n)==w,'Comparison source changed '+n)
 root=a.actor_root;out=a.output;need(root.is_absolute()and root.resolve()==root and out.is_absolute()and out.resolve()==out and not out.exists()and root not in out.parents,'New nonaliased output outside actual capture required')
 parent=root/'report.json';need(sha(parent)==a.report_sha256,'Authentic passed actual capture report required');report=json.loads(parent.read_bytes());need(report['passed']is True and report['first_row_observation_completed']is True and report['source_plan_sha256']==sha(HERE/'prefill-first-gdn-run-source-plan-v2.json'),'Exact actual serial V2 capture/source namespace required')
 need(report['actual_compile_receipt_sha256']=='d65a853afe4d4842f81c455b9cf68a72965df7d193beace849f08bd202b03244'and report['actual_ELF_sha256']=='fdc9855a6a5c144fadb4bb8382b08ca0a61ba5732cc766864b6757624c27ed9c','Actual first-row ELF/build required')
 reads=[]
 def bound(p,size=None,digest=None):
  need(p.resolve()==p and p.is_file()and not p.is_symlink()and p.is_relative_to(root),'Nonaliased actual observed file required');rel=str(p.relative_to(root));before=stat5(p);need(rel in report['artifacts'],'Original actual producer artifact required');raw=p.read_bytes();need(stat5(p)==before and sha(p)==report['artifacts'][rel]and (size is None or len(raw)==size)and (digest is None or hashlib.sha256(raw).hexdigest()==digest),'Current bytes differ from original producer/field binding');need(p.read_bytes()==raw and stat5(p)==before,'Observed field changed across exact reread');reads.append({'path':str(p),'bytes':len(raw),'sha256':report['artifacts'][rel],'stat5':before});return raw
 d=root/'first-row-target';pf=[]
 for p in sorted((d/'prefill_first_gdn').glob('*/frame.json')):
  m=json.loads(bound(p));need(m['schema']==1 and m['source_binding_sha256']==sha(HERE/'prefill-first-gdn-observer-source-plan-v1.json')and m['layer']==0 and m['position']==0 and m['first_token']==248045 and m['before_any_L0_recurrence']is True and m['before_L0_conv']is True,'Actual first-row preconv/recurrence binding required');need(m['qkv_original_quant_type']==m['z_original_quant_type']==8 and m['qkv_shape']==[2560,10240]and m['z_shape']==[2560,6144],'Actual original Q8_0 weight descriptor required')
  need(m['f16_io']is False and type(m['bf16x2_mode'])is int and m['bf16x2_mode']==0 and m['BF16_low_present']is False,'Actual observed single BF16-high contract required')
  fields={f['name']:f for f in m['fields']};need(len(fields)==len(m['fields'])and fields['gdn_alpha_beta_raw']['shape']==[2,48]and fields['gdn_decay']['shape']==fields['gdn_beta']['shape']==[48],'Exact raw AB row layout required')
  pf.append((m,{name:bound(p.parent/(name+'.bin'),f['bytes'],f['sha256'])for name,f in fields.items()}))
 need(len(pf)==2 and [(m['prefill_n'],m['chunk_T'])for m,v in pf]==[(53,53),(116,64)],'Actual two first-chunk routes required')
 meta_paths=list((d/'num10').glob('*.json'));need(len(meta_paths)==1,'One independent decoder frame required');meta=json.loads(bound(meta_paths[0]));need(meta['pid']==pf[0][0]['producer_pid']==pf[1][0]['producer_pid']and meta['request']==3 and meta['gen_ids']==[248045]and meta['token']==248045 and meta['rows']==1 and meta['stage']==0 and meta['layer']==0 and meta['position']==0 and meta['reused']==0 and meta['request_replay_marker_verified']is True and type(meta['completed_nonce'])is int and meta['completed_nonce']>0 and meta['binding_sha256']==sha(HERE/'prefill-first-gdn-observer-source-plan-v1.json'),'Actual independent decoder owner/ordinal/current nonce required')
 selected={'residual_input','attn_mixed','attn_input_q81','gdn_qkv','gdn_z','gdn_decay_beta','gdn_state_before','gdn_conv_before'};decoder={}
 for f in meta['fields']:
  if f['name']not in selected:continue
  cp=Path(f['file']);need(cp.is_relative_to('/results/num10')and '..'not in cp.parts and f['observed']is True,'Actual decoder observed field/prefix required');decoder[f['name']]=bound(d/cp.relative_to('/results'),f['bytes'])
 need(set(decoder)==selected and len(decoder['gdn_decay_beta'])==384 and len(decoder['attn_input_q81'])==80*36,'Source33 pair decay[48], beta[48] and packet[80,36] required')
 zero=[]
 for label,fields in [('decoder',decoder),('prefill53',pf[0][1]),('prefill64',pf[1][1])]:
  for name,extent in [('gdn_state_before',128*48*128*4),('gdn_conv_before',10240*3*4)]:need(len(fields[name])==extent and not any(fields[name]),'Actual independent before-state/conv must be bytezero');zero.append({'route':label,'field':name,'bytezero':True,'bytes':extent,'sha256':hashlib.sha256(fields[name]).hexdigest()})
 comparisons=[]
 for m,fields in pf:
  row={'prefill_n':m['prefill_n'],'chunk_T':m['chunk_T'],'residual_input_bitwise':fields['residual_input']==decoder['residual_input'],'attn_mixed_F32_bitwise':fields['attn_mixed']==decoder['attn_mixed'],'QKV':metrics(decoder['gdn_qkv'],fields['gdn_qkv']),'Z':metrics(decoder['gdn_z'],fields['gdn_z']),'decay':metrics(decoder['gdn_decay_beta'][:192],fields['gdn_decay']),'beta':metrics(decoder['gdn_decay_beta'][192:],fields['gdn_beta']),'raw_AB_decoder_observed':False,'raw_AB_comparison_to_decoder':None,'actual_BF16_low_present':m['BF16_low_present'],'input_views':{name:{'bytes':len(fields[name]),'sha256':hashlib.sha256(fields[name]).hexdigest()}for name in ['attn_mixed_f16','attn_mixed_bf16_high']},'decoder_Q81_input_sha256':hashlib.sha256(decoder['attn_input_q81']).hexdigest()};comparisons.append(row)
 first={name:pf[0][1][name]==pf[1][1][name]for name in pf[0][1]if name!='actual_prefill_input_tokens'}
 need(sha(parent)==a.report_sha256,'Closing producer report changed')
 for v in reads:need(stat5(Path(v['path']))==v['stat5']and sha(v['path'])==v['sha256'],'Closing consumed observation changed')
 record={'schema':1,'source_plan_sha256':sha(PLAN),'capture_report_sha256':a.report_sha256,'actual_source_namespace':'actual58 firstrow d951 / runtimeV2 a6a','comparison_scope':'Two native observed operator paths; neither is an independent original-math reference','before_state_bytezero':zero,'comparisons':comparisons,'prefill53_vs64_bitwise':first,'raw_AB_layout':'Prefill F32 [row][alpha48,beta48], first-row384B; decoder raw pre-nonlinearity AB UNOBSERVED','decay_beta_layout':'Decoder source snapshot.pair copies decay48 then beta48. Prefill exports existing gdn_gates outputs separately. No inverse nonlinear reconstruction.','consumed_files':reads,'model_payload_read':False,'GPU_touched':False,'mathematics_qualified':False,'cause_qualified':False,'speed_qualified':False}
 out.write_text(json.dumps(record,indent=2,ensure_ascii=True)+'\n',encoding='ascii');return record
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--actor-root',type=Path,required=True);p.add_argument('--report-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();result=run(a);print(json.dumps({'output':str(a.output),'sha256':sha(a.output),'comparisons':result['comparisons']},ensure_ascii=True))
