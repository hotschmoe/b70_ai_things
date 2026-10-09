#!/usr/bin/env python3
"""Strict no-weight owner/raw plus chronological UR trace qualification only."""
import argparse,hashlib,json
from pathlib import Path
from parse_usm_logical_free_trace import parse_trace,negative_controls
ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def collect(raw_path,log_path,plan_path,case):
 plan=json.loads(plan_path.read_text());raw=json.loads(raw_path.read_text());text=log_path.read_text(errors='replace')
 if plan['logical_free_parser_sha256']!=sha(ROOT/plan['logical_free_parser']):raise ValueError('Logical-free parser changed')
 trace=parse_trace(text);controls=negative_controls(text,True) if trace['passed'] else {name:False for name in ['missing_free','double_free','failed_free']}
 expected=plan['expected_owner_registrations'][case]
 errors=[]
 if not raw.get('owner_and_probe_passed') or raw.get('model_weights_loaded') is not False or raw.get('inference_or_graph_retirement_qualified') is not False:errors.append('Raw owner/probe scope/result incomplete')
 if raw.get('owner_registrations')!=expected or trace['counts']['owners']!=expected:errors.append('Owned arena/host/probe count incomplete')
 if raw.get('cells')!=plan['bounds']['cells']:errors.append('Cell geometry differs')
 profile=next(p for p in plan['cases'] if p['id']==case);devices=[int(s.split(':')[2]) for s in profile['stages']]
 expected_labels={label+'-'+str(dev)+(('-'+str(slot)) if label=='normal' else '') for dev in devices for label in ['normal','partial-zero','partial-throw','probe'] for slot in (range(3) if label=='normal' else [0])}
 owners=trace.get('owners',{});items=list(owners.values()) if isinstance(owners,dict) else owners
 labels={x['stage'] for x in items}
 if labels!=expected_labels or raw.get('stages')!=len(devices):errors.append('Normal/partial/probe stage roster incomplete')
 role_counts={role:sum(x.get('role')==role for x in items) for role in ['slot_arena','host_step','host_pos','probe']}
 pairs=38 if len(devices)==1 else 40
 if role_counts!={'slot_arena':5*len(devices),'host_step':pairs,'host_pos':pairs,'probe':len(devices)}:errors.append('Arena/host/probe allocation role coverage differs')
 if not trace['passed'] or not all(controls.values()):errors.append('Chronological owning-context logical-free gate/negatives failed')
 return {'schema':1,'passed':not errors,'case':case,'errors':errors,'raw_sha256':sha(raw_path),'trace_sha256':sha(log_path),'plan_sha256':sha(plan_path),'logical_free':trace,'negative_controls':controls,'scope':'No-weight slot arena/host ownership, rollback, resize/isolation and fresh-probe only','model_or_concurrent_graph_math_qualified':False,'postfree_unknown_required':False}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',type=Path,required=True);p.add_argument('--log',type=Path,required=True);p.add_argument('--plan',type=Path,default=ROOT/'strata/flash-next/stage-slot-owner-oracle-plan.json');p.add_argument('--case',choices=['owner_card0','owner_card1','owner_pair32_16'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():p.error('New collector output required')
 r=collect(a.raw,a.log,a.plan,a.case);a.output.write_text(json.dumps(r,indent=2)+'\n',encoding='ascii');print(json.dumps({'passed':r['passed'],'case':a.case,'owners':r['logical_free']['counts']['owners']}));return 0 if r['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
