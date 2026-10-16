"""SOURCE ONLY: later selected decoded text summary, never candidate FP/PASS authority."""
from pathlib import Path
import argparse,collections,hashlib,json,re
p=argparse.ArgumentParser();p.add_argument('audit',type=Path);p.add_argument('--expected-facts-sha256',required=True);a=p.parse_args();D=a.audit.resolve()
raw=(D/'facts.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==a.expected_facts_sha256
facts=json.loads(raw);assert facts['status']=='PASS_SIX_STATIC_BINDINGS_STRICT_ORIGINAL_CONTROL_REUSE_CANDIDATE_MANUAL_ISA_PENDING'
O=D/'candidate-isa-summary.json';assert not O.exists(),'Preserve prior summary; no automatic rewrite'
summary={}
for arm in ['candidate_reduce','candidate_up']:
 q=D/'candidate-isa'/arm;assert(q/'decoder.exit').read_text().strip()=='0';assert(q/'input-before.sha256').read_bytes()==(q/'input-after.sha256').read_bytes()
 f=q/'disasm'/('.text.s116_'+arm+'.asm');b=f.read_bytes();rows=[];at=0
 for line,l in enumerate(b.decode('ascii').splitlines(),1):
  m=re.fullmatch(r'L(\d+):',l)
  if m:assert at==int(m[1]),(line,at,l);continue
  if not l.strip():continue
  m=re.match(r'^\s*(?:\([^)]*\)\s*)?([A-Za-z][\w.]*)\b',l);assert m,(line,l)
  rows.append({'line':line,'byte_offset':at,'op':m[1],'text':l.rstrip()});at+=8 if'Compacted'in l else 16
  if'{EOT'in l:break
 else:raise RuntimeError('HOLD selected first EOT absent; do not guess computational bound')
 counter=collections.Counter(x['op']for x in rows);sync=[x for x in rows if x['op']=='sync.bar'];first=sync[0]['byte_offset']if sync else None
 fp=[x for x in rows if x['op'].startswith('math.')or(x['op']in['add','mul','mad','madm','mac','fma']and any(t in x['text']for t in [':f',':df',':hf',':bf']))]
 summary[arm]={'ASM':str(f),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'EOT_end_byte':at,'static_instruction_count':len(rows),'static_opcode_counts':dict(sorted(counter.items())),'CR0_raw_sites':[x for x in rows if'cr0.'in x['text']],'actual_sync_bar_sites':sync,'FP_instruction_sites':fp,'UGM_load_store_sites':[x for x in rows if x['op']=='send.ugm'and('load.ugm'in x['text']or'store.ugm'in x['text'])],'first_sync_bar_byte':first,'before_first_barrier_FP_sites':[x for x in fp if x['byte_offset']<first]if first is not None else None,'selected_execution_fields':facts['selected'][arm]['execution_fields'],'missing_fields':'UNKNOWN; barrier_count does not give static barrier instruction count','scope':'All counts are textual static sites through first EOT, including predicates/refinement/loops, not dynamically executed work/cycles/SFU traffic/physical traffic or critical-path budget. FP site list and store addresses still require manual operand/consumer tracing.'}
report={'status':'SELECTED_ISA_SUMMARIZED_MANUAL_FP_AND_UP_BODY_REVIEW_REQUIRED','facts_sha256':a.expected_facts_sha256,'candidates':summary,'GPU_GO':False,'candidate_numerical_correctness_proved':False,'manual_required':['REDUCE: preserved raw part store value is materialized FP32; extra+0 input is exact same32bit published value, despite possible forwarded load.','REDUCE: original80-slice add chain/inject store retained; then source divide/native exp/denominator+1/final divide lowerings/ordering and CR0 read explicitly.','UP: only old activation/global-lo prologue replaced by correct owner global-lo->same SLM/barrier; all dot/FMA/butterfly/gating/residual/stream-mean FP operations remain in their original per-value order.','Separate source identity, raw image/text identity and ordered FP dataflow; scheduling/register/resource differences explicitly documented; no universal signedzero/subnormal/nonfinite equivalence.','Before GPU, root/actual reviewers close any original-control or candidate ledger/metadata/resource uncertainty; no automatic normalization/tolerance or retest.']}
O.write_text(json.dumps(report,indent=2)+'\n',encoding='ascii');print('SELECTED_ISA_SUMMARY_ONLY_ROOT_FP_ORDER_REVIEW_STILL_REQUIRED')
