#!/usr/bin/env python3
"""Read-only host intervals after original strict collector; no model admission."""
import argparse,hashlib,json,math
from pathlib import Path
from serial37_canonical_json_v3 import read_unique,canonical
import source41_host_trace_contract_v2 as strict
require=strict.require
K=strict.K
PLAN=Path(__file__).with_name('source41-h36-latency-analysis-source-plan-v2.json')

def source_binding():
    plan=bound(PLAN);value=read_unique(PLAN);root=PLAN.parent.parents[1]
    for name,want in value['files'].items():require(bound(root/name)['sha256']==want,'Analysis source changed '+name)
    require(bound(PLAN)==plan,'Analysis source plan changed during admission')
    return plan

def original_artifact(root,path,report):
    value=bound(path);relative=str(Path(path).relative_to(root))
    require(relative in report['artifacts']and value['sha256']==report['artifacts'][relative],'Original closed report artifact SHA differs '+relative)
    return value

def original_plan(root,report):
    value=original_artifact(root,root/'plan.snapshot.json',report)
    require(value['sha256']==report['plan_sha256'],'Original parent plan SHA differs')
    return value

def union(intervals,window=None):
    rows=[]
    for a,b in intervals:
        require(type(a)is int and type(b)is int and 0<a<=b,'Ordered typed host span required')
        if window is not None:a,b=max(a,window[0]),min(b,window[1])
        if a<b:rows.append((a,b))
    merged=[]
    for a,b in sorted(rows):
        if merged and a<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],b)
        else:merged.append([a,b])
    return {'union_ns':sum(b-a for a,b in merged),'intervals_ns':merged}
def quantile(values,num,den=100):
    require(type(num)is int and type(den)is int and 0<num<=den,'Exact quantile fraction required')
    require(all(type(x)is int and x>=0 for x in values),'Typed nonnegative token gaps required')
    if not values:return None
    return sorted(values)[(num*len(values)+den-1)//den-1]
def metrics(rows):
    events=rows[1:-1];opens={};spans={};tokens=[];last=0
    for seq,e in enumerate(events):
        require(type(e['ns'])is int and e['ns']>0 and e['ns']>=last and e['seq']==seq,'Ordered host clock/sequence required');last=e['ns']
        if e['edge']==1:
            require(e['span']not in opens,'Duplicate open span');opens[e['span']]=e
        elif e['edge']==2:
            require(e['span']in opens,'Unmatched host span');b=opens.pop(e['span']);require(b['kind']==e['kind'],'Host span kind mismatch');spans.setdefault(e['kind'],[]).append((b['ns'],e['ns']))
        elif e['kind']==K['TEmit']:tokens.append(e['ns'])
    require(not opens and len(spans.get(K['Request'],[]))==1 and tokens,'Complete request and token markers required')
    start,end=spans[K['Request']][0];require(start<=tokens[0]<=tokens[-1]<=end,'Tokens outside original request')
    gaps=[b-a for a,b in zip(tokens,tokens[1:])];windows=[(start,tokens[0]),*zip(tokens,tokens[1:]),(tokens[-1],end)]
    kinds={name:union(spans.get(kind,[]),(start,end))for name,kind in K.items()if name!='TEmit'}
    segments=[]
    for index,window in enumerate(windows):
        segments.append({'segment':'request_to_first_T'if index==0 else'after_T','token_ordinal':None if index==0 else index,'bounds_ns':list(window),'LPReadback':union(spans.get(K['LPReadback'],[]),window),'LPCPU':union(spans.get(K['LPCPU'],[]),window),'LP_combined':union(spans.get(K['LPReadback'],[])+spans.get(K['LPCPU'],[]),window)})
    return {'request_start_ns':start,'request_end_ns':end,'native_TTFT_ns':tokens[0]-start,'request_completion_ns':end-start,'last_T_to_request_end_ns':end-tokens[-1],'token_ns':tokens,'TEmit_gaps_ns':gaps,'gap_P50_ns':quantile(gaps,50),'gap_P95_ns':quantile(gaps,95),'quantile':'nearest rank ceil(p*N), independent request gaps; N=0 gives null','kind_unions':kinds,'token_segments':segments,'all_observed_kind_union':union([x for values in spans.values()for x in values],(start,end)),'unions_are_not_additive':True}
def bound(path,limit=64<<20):
    path=Path(path);require(path.is_file()and not path.is_symlink()and path.resolve()==path,'Regular exact analysis input required')
    def sig():
        s=path.stat();return[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
    before=sig();require(before[2]<=limit,'Bounded analysis artifact required');raw=path.read_bytes();require(len(raw)==before[2]and sig()==before,'Analysis input changed');return {'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'stat5':before}
def analyze(root,policy,expected_report_sha256):
    analysis_source=source_binding()
    root=Path(root).resolve();require(policy in('LP20','LPdisabled'),'Explicit LP policy required')
    if policy=='LP20':
        import run_source41_host_latency_v2 as actor
        import validate_source41_host_latency_v2 as reader
    else:
        import run_source41_host_lpdisabled_v1 as actor
        import validate_source41_host_lpdisabled_v1 as reader
    artifacts={}
    report=None
    def save(path):
        result=original_artifact(root,path,report)if report is not None and Path(path)!=root/'report.json'and root in Path(path).parents else bound(path)
        artifacts[str(path)]=result;return result
    require(save(root/'report.json')['sha256']==expected_report_sha256,'Exact closed parent report SHA required');report=read_unique(root/'report.json');require(report['passed']is True and report['errors']==[]and report['phase']=='host-latency','Original closed host control required')
    plan=read_unique(actor.PLAN);require(report['source_plan_sha256']==save(actor.PLAN)['sha256']==save(root/'source-plan.snapshot.json')['sha256'],'Original source plan snapshot differs')
    for name,want in plan['files'].items():require(bound(actor.ROOT/name)['sha256']==want,'Current original source changed '+name)
    for name in('plan.snapshot.json','prepared.snapshot.json','config.snapshot.json','inputs.json','producer.snapshot.py'):save(root/name)
    original_plan(root,report)
    require(artifacts[str(root/'producer.snapshot.py')]['sha256']==bound(actor.__file__)['sha256'],'Original producer snapshot differs')
    m=read_unique(root/'prepared.snapshot.json');cfg=read_unique(root/'config.snapshot.json');inputs=read_unique(root/'inputs.json');views=[];result=[]
    for phase in actor.PHASES:
        directory=root/phase;descriptor,recipe=actor.phase_recipe(phase,m,cfg,inputs,report['source_plan_sha256'])
        for name in('recipe.json','command.json','result.json','requests.json','engine.combined.log','native-namespace.json'):save(directory/name)
        require(canonical(recipe)==canonical(read_unique(directory/'recipe.json')),'Original exact cfg/env/request recipe differs')
        command=reader._PORTS['command'](root,phase,descriptor,recipe,report['producer_pid'],report['source_plan_sha256']);require(canonical(command)==canonical(read_unique(directory/'command.json')),'Original actual command differs')
        terminal=read_unique(directory/'result.json');require(canonical(terminal)==canonical(report['results'][phase])and terminal['passed']is True,'Original phase result differs');pid=actor.collector.native_namespace(terminal['terminal'],command)
        require(canonical(read_unique(directory/'native-namespace.json'))==canonical({'native_pid':pid,'scope':'direct exec bash replacement/no-init/private PID namespace','terminal_command_source_recollected':True}),'Original native namespace record differs')
        rows=read_unique(directory/'requests.json');view=actor.contract.requests(rows,recipe['requests']);views.append(view)
        for row in rows:
            path=directory/('request-'+str(row['ordinal'])+'.json');save(path);require(canonical(row['raw'])==canonical(read_unique(path)),'Original individual request differs')
        traces=sorted((directory/'host-traces').glob('host-trace36-*.jsonl'),key=lambda p:int(p.stem.split('-')[-1]))
        if phase==actor.PHASES[0]:require(not list((directory/'host-traces').iterdir()),'OFF trace unexpectedly present');continue
        for path in traces:save(path)
        coverage=actor.collector.collect(traces,directory/'engine.combined.log',report['source_plan_sha256'],[r['raw']for r in rows],[[0,0,48]],pid);save(directory/'host-coverage.json');require(canonical(coverage)==canonical(read_unique(directory/'host-coverage.json')),'Fresh original strict collector result differs')
        result=[{'request':raw['ordinal'],'input_tokens':len(raw['raw']['ids']),'output_tokens':len(raw['raw']['output_ids']),'finish':raw['raw']['done'].split()[5],'trace':str(path),**metrics(strict.typed_rows(path,pid)[0])}for raw,path in zip(rows,traces)]
    require(canonical(actor.equivalence(*views))==canonical(report['H36_OFF_ON_comparison']),'Original OFF/ON/repeat equivalence differs')
    for name,want in plan['files'].items():require(bound(actor.ROOT/name)['sha256']==want,'Original source changed after analysis '+name)
    for path,old in artifacts.items():require(bound(path)==old,'Analysis artifact changed at end '+path)
    require(source_binding()==analysis_source,'Analysis source changed at final boundary')
    return {'schema':2,'scope':'closed fourGEN64 native serial HOST trace analysis only','LP_policy':policy,'analysis_source_plan':analysis_source,'source_sha256':bound(__file__)['sha256'],'parent_report_sha256':expected_report_sha256,'raw_bindings':artifacts,'fresh_original_collector':coverage,'requests':result,'H36_export_IO_outside_Request_span':True,'TTFT_is_native_Request_to_TEmit_not_client_first_byte':True,'TEmit_is_marker_before_printf_flush_not_stream_delivery':True,'sampling_and_copy_scopes_are_host_intervals_not_device_execution':True,'nested_kinds_are_not_additive':True,'no_new_model_admission_or_full4_scan':True,'device_idle_qualified':False,'bandwidth_qualified':False,'clean_serving_latency_qualified':False,'interactive_concurrency_qualified':False}
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--policy',choices=['LP20','LPdisabled'],required=True);p.add_argument('--expected-report-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists()and a.root.resolve()not in a.output.resolve().parents,'New external analysis file required');value=analyze(a.root,a.policy,a.expected_report_sha256);a.output.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='ascii')
if __name__=='__main__':main()
