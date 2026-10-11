#!/usr/bin/env python3
"""Closed PTI43 raw observations; neither current model nor physical-idle proof."""
import argparse,hashlib,json,math
from pathlib import Path
from serial37_canonical_json_v3 import canonical,read_unique
import analyze_source41_h36_latency_v2 as host
import collect_source41_model_pti_v1 as pti
import source41_host_trace_contract_v2 as strict
require=host.require;HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];PLAN=HERE/'source43-pti-latency-analysis-source-plan-v2.json'
PHASES=('source43-pti-OFF','source43-pti-ON')

def source_binding():
    before=host.bound(PLAN);value=read_unique(PLAN)
    for name,want in value['files'].items():require(host.bound(ROOT/name)['sha256']==want,'Analysis source changed '+name)
    require(host.bound(PLAN)==before,'Analysis plan changed during source check');return before

def generation_gate(report,admission):
    from run_source43_pti_latency_v3 import PLAN as producer_plan
    expected=host.bound(producer_plan)['sha256']
    require(type(admission.get('source_generation'))is int and admission['source_generation']==43 and type(admission.get('materialized_mode'))is int and admission['materialized_mode']==0 and report.get('source_plan_sha256')==admission.get('source_plan_sha256')==expected,'Exact current PTI actor3/public3 source generation required')
    return expected

def raw_device_metrics(instances):
    """Union raw device ticks, without using host-normalized ordering/alignment."""
    groups={};ids=set()
    for row in instances:
        fields=('instance_id','device_handle','timer_frequency_hz','timer_mask','raw_kernel_start_ticks','raw_kernel_end_ticks')
        require(all(type(row[k])is int and row[k]>=0 for k in fields)and row['instance_id']>0 and row['instance_id']not in ids and row['device_handle']>0 and row['timer_frequency_hz']>0 and row['timer_mask']>0,'Typed unique actual device instance required');ids.add(row['instance_id'])
        key=(row['device_handle'],row['timer_frequency_hz'],row['timer_mask']);groups.setdefault(key,[]).append(row)
    output=[]
    for (device,freq,mask),rows in sorted(groups.items()):
        require(mask&(mask+1)==0,'Actual contiguous timer mask required');intervals=[];wrapped=[]
        for row in rows:
            start=row['raw_kernel_start_ticks']&mask;end=row['raw_kernel_end_ticks']&mask
            if end<start:wrapped.append(row['instance_id'])
            else:intervals.append((start,end))
        # No epoch unwrapping from a host-normalized timestamp is inferred.
        if wrapped or (intervals and max(b for a,b in intervals)-min(a for a,b in intervals)>mask//2):
            output.append({'device_handle':device,'timer_frequency_hz':freq,'timer_mask':mask,'instances':len(rows),'raw_linear_epoch_ambiguous':True,'wrapped_instance_ids':wrapped,'observed_busy_union_ticks':None,'observed_gaps_ticks':None,'physical_idle_qualified':False});continue
        merged=[]
        for start,end in sorted(intervals):
            if start==end:continue
            if merged and start<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],end)
            else:merged.append([start,end])
        gaps=[b[0]-a[1]for a,b in zip(merged,merged[1:])];union=sum(b-a for a,b in merged)
        output.append({'device_handle':device,'timer_frequency_hz':freq,'timer_mask':mask,'instances':len(rows),'raw_linear_epoch_ambiguous':False,'raw_intervals_ticks':merged,'observed_busy_union_ticks':union,'observed_union_duration_ns_fraction':{'numerator':union*1000000000,'denominator':freq},'observed_gaps_ticks':gaps,'gap_P50_ticks':host.quantile(gaps,50),'gap_P95_ticks':host.quantile(gaps,95),'covered_interval_hull_ticks':merged[-1][1]-merged[0][0]if merged else 0,'producer_record_start_order_reversals':sum(b[0]<a[0]for a,b in zip(intervals,intervals[1:])),'physical_idle_qualified':False,'gap_scope':'gaps between observed operation unions; unobserved work may occupy them','host_clock_alignment_used':False,'complete_device_busy_or_utilization_qualified':False})
    return output

def analyze(root,public_admission,expected_report_sha256,expected_public_sha256):
    source=source_binding();root=Path(root).resolve();public_admission=Path(public_admission).resolve();bindings={};report=None
    def bind(path,limit=64<<20):
        path=Path(path);value=host.bound(path,limit)
        if report is not None and path!=root/'report.json'and root in path.parents:require(report['artifacts'].get(str(path.relative_to(root)))==value['sha256'],'Original closed artifact SHA changed '+str(path))
        bindings[str(path)]=value;return value
    require(bind(root/'report.json')['sha256']==expected_report_sha256 and bind(public_admission)['sha256']==expected_public_sha256,'Exact closed producer/public admission SHA required');report=read_unique(root/'report.json');admission=read_unique(public_admission)
    generation_gate(report,admission)
    require(report['passed']is True and report['errors']==[]and report['phase']=='host-latency'and admission['passed']is True and admission['root']==str(root)and admission['report_sha256']==expected_report_sha256 and admission['source_generation']==43 and admission['materialized_mode']==0 and admission['actual_finalizer_closed']is True and admission['LPdisabled']is True and admission['H36_OFF_ON_equivalence_passed']is True,'Existing successful PTI43 public admission required')
    require(bind(root/'plan.snapshot.json')['sha256']==report['plan_sha256'],'Actual intake/corpus SHA differs');require(bind(root/'source-plan.snapshot.json')['sha256']==report['source_plan_sha256']==admission['source_plan_sha256'],'Actual producer source-plan association differs')
    from run_source43_pti_latency_v3 import PLAN as producer_plan
    require(bind(producer_plan)['sha256']==report['source_plan_sha256'],'Current original producer source plan differs');producer=read_unique(producer_plan)
    for name,want in producer['files'].items():require(bind(ROOT/name)['sha256']==want,'Current producer dependency changed '+name)
    full_roster={str(p.relative_to(root)):host.bound(p,1<<30)['sha256']for p in root.rglob('*')if p.is_file()and p.name!='report.json'}
    require(not any(p.is_symlink()for p in root.rglob('*'))and canonical(full_roster)==canonical(report['artifacts']),'Original full regular artifact roster changed')
    arms=[];device=None
    for phase in PHASES:
        directory=root/phase;rows=read_unique(directory/'requests.json');bind(directory/'requests.json');require(len(rows)==4,'All actual four requests required');recipe=read_unique(directory/'recipe.json');bind(directory/'recipe.json');require(recipe['engine_generation']==43 and recipe['materialized_mode']==0 and recipe['PTI_enabled']is(phase==PHASES[1]),'Actual43 PTI phase recipe changed');runtime=recipe['tool_runtime'];traces=sorted((directory/'host-traces').glob('host-trace36-*.jsonl'),key=lambda p:int(p.stem.split('-')[-1]));require(len(traces)==4,'All four host traces required')
        from collect_source41_lpdisabled_host_v1 import collect
        bind(directory/'engine.combined.log');coverage=collect(traces,directory/'engine.combined.log',report['source_plan_sha256'],[r['raw']for r in rows],[[0,0,48]],1);bind(directory/'host-coverage.json');require(canonical(coverage)==canonical(read_unique(directory/'host-coverage.json'))==canonical(admission['host_coverage_both_arms'][phase]),'Actual strict host coverage/public result differs')
        observations=[]
        for row,path in zip(rows,traces):
            bind(path);bind(directory/('request-'+str(row['ordinal'])+'.json'));require(canonical(row['raw'])==canonical(read_unique(directory/('request-'+str(row['ordinal'])+'.json'))),'Actual individual request differs');observations.append({'ordinal':row['ordinal'],'input_tokens':len(row['raw']['ids']),'output_tokens':len(row['raw']['output_ids']),'finish':row['raw']['done'].split()[5],**host.metrics(strict.typed_rows(path,1)[0])})
        arms.append({'phase':phase,'requests':observations})
        if phase==PHASES[1]:
            for path in sorted((directory/'unitrace').iterdir()):bind(path,1<<30)
            bind(directory/'leaf.stderr');from validate_source43_pti_latency_v3 import census_binding
            mapped=census_binding(directory,recipe,read_unique(directory/'result.json')['terminal'],rows);require(canonical(mapped)==canonical(admission['mapped_injection']),'Actual strict current mapped backing association/public result differs');fresh=pti.collect(directory,runtime);require(canonical(fresh)==canonical(admission['PTI_observed_model_coverage']),'Actual clock/Chrome audit/public result differs');files=sorted((directory/'unitrace').glob('clock.*.jsonl'));require(len(files)==1,'One actual clock process required');clock=[read_unique_line(line)for line in files[0].read_text().splitlines()];device=raw_device_metrics([row for row in clock if row['kind']=='instance_normalization'])
    comparisons=[]
    for left,right in zip(arms[0]['requests'],arms[1]['requests']):
        require((left['ordinal'],left['input_tokens'],left['output_tokens'],left['finish'])==(right['ordinal'],right['input_tokens'],right['output_tokens'],right['finish']),'Observed OFF/ON request pairing differs');comparisons.append({'ordinal':left['ordinal'],'OFF_request_completion_ns':left['request_completion_ns'],'ON_request_completion_ns':right['request_completion_ns'],'ON_minus_OFF_request_completion_ns':right['request_completion_ns']-left['request_completion_ns'],'OFF_gap_P50_ns':left['gap_P50_ns'],'ON_gap_P50_ns':right['gap_P50_ns'],'scope':'matched H36-bearing native observations; not clean serving latency'})
    require({str(p.relative_to(root)):host.bound(p,1<<30)['sha256']for p in root.rglob('*')if p.is_file()and p.name!='report.json'}==full_roster,'Original raw roster changed during analysis')
    for path,before in bindings.items():require(host.bound(path,1<<30)==before,'Consumed analysis input changed '+path)
    require(source_binding()==source,'Analysis source changed after raw reads')
    return {'schema':1,'scope':'observed closed PTI43 LPdisabled host/device intervals after existing public admission','analysis_source':source,'parent_report_sha256':expected_report_sha256,'public_admission_sha256':expected_public_sha256,'raw_bindings':bindings,'host_arms':arms,'profiler_OFF_ON_observations':comparisons,'raw_device_observed_operation_unions':device,'nested_host_kinds_are_not_additive':True,'H36_export_IO_outside_request_span':True,'TEmit_is_native_marker_not_client_delivery':True,'device_union_uses_raw_ticks_not_host_normalized_reordering':True,'host_device_clock_alignment_qualified':False,'complete_operation_coverage_qualified':False,'physical_idle_qualified':False,'bandwidth_qualified':False,'clean_serving_speed_qualified':False,'current_model_or_health_readmission_performed':False,'no_model_payload_read':True}

def read_unique_line(line):return json.loads(line,object_pairs_hook=__import__('serial37_canonical_json_v3').unique_object,parse_constant=__import__('serial37_canonical_json_v3').finite_constant)
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--public-admission',type=Path,required=True);p.add_argument('--expected-report-sha256',required=True);p.add_argument('--expected-public-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists()and a.root.resolve()not in a.output.resolve().parents,'New external analysis output required');value=analyze(a.root,a.public_admission,a.expected_report_sha256,a.expected_public_sha256);a.output.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='ascii')
if __name__=='__main__':main()
