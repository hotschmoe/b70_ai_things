#!/usr/bin/env python3
"""Closed PTI43 raw observations; neither current model nor physical-idle proof."""
import argparse,hashlib,json,math
from pathlib import Path
from serial37_canonical_json_v3 import canonical,read_unique
import analyze_source41_h36_latency_v2 as host
import collect_source43_sampled_model_pti_v1 as pti
import run_source43_warm_pti_v5 as warm_actor
import validate_source43_warm_pti_v5 as warm_public
import validate_source43_bounded_pti_v1 as cold_public
import ast,copy
import source41_host_trace_contract_v2 as strict
require=host.require;HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];PLAN=HERE/'source43-warm-pti-analysis-source-plan-v4.json'
PHASES=warm_actor.PHASES

def source_binding():
    before=host.bound(PLAN);value=read_unique(PLAN)
    for name,want in value['files'].items():require(host.bound(ROOT/name)['sha256']==want,'Analysis source changed '+name)
    require(host.bound(PLAN)==before,'Analysis plan changed during source check');return before

def generation_gate(report,admission):
    producer_plan=warm_actor.PLAN
    expected=host.bound(producer_plan)['sha256']
    require(type(admission.get('source_generation'))is int and admission['source_generation']==43 and type(admission.get('materialized_mode'))is int and admission['materialized_mode']==0 and report.get('source_plan_sha256')==admission.get('source_plan_sha256')==expected,'Exact warm producer/public source generation required')
    return expected

def runtime_join(runtime):
    require(runtime['injection']['sha256']==warm_actor.fresh_tool.LIB_SHA and runtime['tool']['sha256']==warm_actor.fresh_tool.TOOL_SHA and runtime['source_plan_sha256']==warm_actor.fresh_tool.SOURCE_SHA and runtime['paused_regular_list_retention_required']is True,'Exact fresh paused-list tool source/ELFs required')
    result=warm_actor.fresh_tool.binding(runtime['intake_path'],runtime['exact_intake_sha256'])
    require(result['full_model_coverage_qualified']is False and result['clock_alignment_qualified']is False,'Fixture authority cannot become full-model coverage/calibration')
    return result

def kernel_counts(instances):
    counts={};ids=set()
    for row in instances:
        name=row['kernel_name'];identifier=row['instance_id']
        require(type(name)is str and bool(name)and type(identifier)is int and identifier>0 and identifier not in ids,'Unique actual named GPU instance required');ids.add(identifier);counts[name]=counts.get(name,0)+1
    return [{'kernel_name':name,'instances':count}for name,count in sorted(counts.items())]

def component_join(component,report,admission):
    require(canonical(component)==canonical(report['OFF_component_prerequisite'])==canonical(admission['OFF_component_prerequisite']),'Exact recorded OFF component/public descriptors required')
    require(component['passed']is True and component['historical_metadata_only']is True and component['original_parent_passed']is False and component['whole_parent_qualified']is False and component['current_model_identity_qualified']is False and component['current_GPU_health_transferred']is False and component['fresh_public_full4_rehashed']is False and component['component_original_recorded_phase_gates_passed']is True,'Completed historical OFF component only; no whole-parent/current authority')
    require(component['phase']=='source43-warm-pti-OFF'and type(component['source_generation'])is int and component['source_generation']==43 and type(component['materialized_mode'])is int and component['materialized_mode']==0,'Exact original OFF component role required')
    off_root=Path(component['root']);require(off_root.is_absolute()and off_root.resolve()==off_root,'Canonical original OFF root required');original_path=off_root/'report.json';before=host.bound(original_path);raw=original_path.read_bytes();require(hashlib.sha256(raw).hexdigest()==component['report_sha256']==before['sha256']and host.bound(original_path)==before,'Exact consumed original failed parent SHA changed');original=read_unique_line(raw)
    require(original['passed']is False and canonical(original['errors'])==canonical(component['original_parent_errors'])and original['errors'],'Original failed parent/errors retained')
    return off_root,original

def selected_host_metric(row,trace_rows):
    require(type(row['ordinal'])is int and row['ordinal']==2 and len(row['raw']['ids'])==54,'Exact selected useful54 request required')
    return {'ordinal':2,'input_tokens':54,'output_tokens':len(row['raw']['output_ids']),'finish':row['raw']['done'].split()[5],**host.metrics(trace_rows)}

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

def analyze(root,public_admission,expected_report_sha256,expected_public_sha256,off_component_public,expected_off_component_sha256):
    source=source_binding();root=Path(root).resolve();public_admission=Path(public_admission).resolve();bindings={};report=None;original_reports={}
    def bind(path,limit=64<<20):
        path=Path(path);value=host.bound(path,limit)
        for admitted_root,original_report in original_reports.items():
            if path!=admitted_root/'report.json'and admitted_root in path.parents:require(original_report['artifacts'].get(str(path.relative_to(admitted_root)))==value['sha256'],'Original closed artifact SHA changed '+str(path))
        bindings[str(path)]=value;return value
    require(bind(root/'report.json')['sha256']==expected_report_sha256 and bind(public_admission)['sha256']==expected_public_sha256,'Exact closed producer/public admission SHA required');report=read_unique(root/'report.json');admission=read_unique(public_admission)
    generation_gate(report,admission)
    original_reports[root]=report
    require(bind(off_component_public)['sha256']==expected_off_component_sha256,'Exact recorded OFF component public SHA required');component=read_unique(off_component_public);off_root,off_report=component_join(component,report,admission);require(off_root!=root and off_root not in root.parents and root not in off_root.parents,'Separate original OFF and new ON roots required');require(bind(off_root/'report.json')['sha256']==component['report_sha256'],'Original failed OFF parent SHA changed');original_reports[off_root]=off_report
    require(bind(off_root/'source-plan.snapshot.json')['sha256']==component['source_plan_sha256'],'Original OFF source snapshot changed')
    original_source=read_unique(off_root/'source-plan.snapshot.json')
    for name,want in original_source['files'].items():require(bind(ROOT/name)['sha256']==want,'Original OFF source dependency changed '+name)
    require(report['passed']is True and report['errors']==[]and report['phase']=='host-latency'and admission['passed']is True and admission['root']==str(root)and admission['report_sha256']==expected_report_sha256 and admission['source_generation']==43 and admission['materialized_mode']==0 and admission['actual_finalizer_closed']is True and admission['LPdisabled']is True and admission['H36_OFF_ON_equivalence_passed']is True and report['role_input']=='short' and admission['cold_GEN4_scope']is False and admission['conversation_state_reuse_claimed']is False,'Existing successful warm43 public admission required')
    require(bind(root/'plan.snapshot.json')['sha256']==report['plan_sha256'],'Actual intake/corpus SHA differs');require(bind(root/'source-plan.snapshot.json')['sha256']==report['source_plan_sha256']==admission['source_plan_sha256'],'Actual producer source-plan association differs')
    producer_plan=warm_actor.PLAN
    require(bind(producer_plan)['sha256']==report['source_plan_sha256'],'Current original producer source plan differs');producer=read_unique(producer_plan)
    for name,want in producer['files'].items():require(bind(ROOT/name)['sha256']==want,'Current producer dependency changed '+name)
    require(producer['OFF_native_rerun']is False and producer['fresh_current_model_identity_health_and_EOF_gates_preserved']is True,'ON-only retry must preserve failed-parent historical OFF scope')
    full_rosters={}
    for admitted_root,original_report in original_reports.items():
        full_rosters[admitted_root]={str(p.relative_to(admitted_root)):host.bound(p,1<<30)['sha256']for p in admitted_root.rglob('*')if p.is_file()and p.name!='report.json'}
        require(not any(p.is_symlink()for p in admitted_root.rglob('*'))and canonical(full_rosters[admitted_root])==canonical(original_report['artifacts']),'Original full regular artifact roster changed')
    arms=[];device=None;kernel_summary=None;observed_flow=None
    for arm_root,phase,is_on,arm_report in ((off_root,component['phase'],False,off_report),(root,PHASES[0],True,report)):
        directory=arm_root/phase;rows=read_unique(directory/'requests.json');bind(directory/'requests.json');require(len(rows)==2,'Exactly warmup/selected original requests required');recipe=read_unique(directory/'recipe.json');bind(directory/'recipe.json');require(recipe['engine_generation']==43 and recipe['materialized_mode']==0 and recipe['PTI_enabled']is is_on,'Actual43 PTI phase recipe changed');runtime=recipe['tool_runtime'];require(canonical(runtime_join(runtime))==canonical(runtime),'Actual fresh tool/fixture runtime envelope changed');traces=sorted((directory/'host-traces').glob('host-trace36-*.jsonl'),key=lambda p:int(p.stem.split('-')[-1]));require(len(traces)==2,'Exactly warmup/selected host traces required')
        from collect_source41_lpdisabled_host_v1 import collect
        bind(directory/'engine.combined.log');coverage=collect(traces,directory/'engine.combined.log',arm_report['source_plan_sha256'],[r['raw']for r in rows],[[0,0,48]],1);bind(directory/'host-coverage.json');require(canonical(coverage)==canonical(read_unique(directory/'host-coverage.json')),'Actual strict host coverage differs');require(not is_on or canonical(coverage)==canonical(admission['host_coverage_both_arms'][phase]),'Actual ON strict host/public coverage differs')
        observations=[]
        for row,path in zip(rows,traces):
            bind(path);bind(directory/('request-'+str(row['ordinal'])+'.json'));require(canonical(row['raw'])==canonical(read_unique(directory/('request-'+str(row['ordinal'])+'.json'))),'Actual individual request differs');typed=strict.typed_rows(path,1)[0]
            if row['ordinal']==2:observations.append(selected_host_metric(row,typed))
        graph=warm_public.graph_binding(directory);bind(directory/'warm-graph-proof.json');require(canonical(graph)==canonical(read_unique(directory/'warm-graph-proof.json')),'Original saved warm graph reuse differs');require(not is_on or canonical(graph)==canonical(admission['warm_same_process_graph_reuse'][phase]),'Original admitted ON graph reuse differs');require(len(observations)==1 and observations[0]['ordinal']==2 and observations[0]['input_tokens']==54,'Only actual selected useful54 request host metrics required');arms.append({'phase':phase,'requests':observations,'warmup_host_metrics_included':False})
        if is_on:
            for path in sorted((directory/'unitrace').iterdir()):bind(path,1<<30)
            bind(directory/'leaf.stderr');warm_actor.set_case(report['role_input'],report['producer_pid'],report['producer_start_ticks'])
            node=copy.deepcopy(next(n for n in ast.parse(Path(cold_public.__file__).read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name=='census_binding'));namespace={**vars(cold_public),'actor':warm_actor};exec(compile(ast.fix_missing_locations(ast.Module([node],type_ignores=[])),__file__+'[warm-original-saved-census-admission]','exec'),namespace);census_binding=namespace['census_binding']
            mapped=census_binding(directory,recipe,read_unique(directory/'result.json')['terminal'],rows);require(canonical(mapped)==canonical(admission['mapped_injection']),'Actual strict current mapped backing association/public result differs');fresh=pti.collect(directory,runtime);require(canonical(fresh)==canonical(admission['PTI_observed_model_coverage']),'Actual clock/Chrome audit/public result differs');kernel_summary=kernel_counts(fresh['instances']);observed_flow={'observed_submit_flow_coverage_passed':fresh['observed_submit_flow_coverage_passed'],'flow_gaps':fresh['flow_gaps'],'model_source_semantic_operation_coverage_qualified':fresh['model_source_semantic_operation_coverage_qualified']};files=sorted((directory/'unitrace').glob('clock.*.jsonl'));require(len(files)==1,'One actual clock process required');clock=[read_unique_line(line)for line in files[0].read_text().splitlines()];warm_public.control_binding(directory,recipe,read_unique(directory/'result.json')['terminal'],report);window=warm_public.clock_window(directory);require(canonical(window)==canonical(admission['observed_window']),'Original admitted warm clock window differs');device=raw_device_metrics([row for row in clock if row['kind']=='instance_normalization'])
    comparisons=[]
    for left,right in zip(arms[0]['requests'],arms[1]['requests']):
        require((left['ordinal'],left['input_tokens'],left['output_tokens'],left['finish'])==(right['ordinal'],right['input_tokens'],right['output_tokens'],right['finish']),'Observed OFF/ON request pairing differs');comparisons.append({'ordinal':left['ordinal'],'OFF_request_completion_ns':left['request_completion_ns'],'ON_request_completion_ns':right['request_completion_ns'],'ON_minus_OFF_request_completion_ns':right['request_completion_ns']-left['request_completion_ns'],'OFF_gap_P50_ns':left['gap_P50_ns'],'ON_gap_P50_ns':right['gap_P50_ns'],'scope':'matched selected request2 after actual same-process graph warmup; not clean serving latency'})
    for admitted_root,original_report in original_reports.items():require({str(p.relative_to(admitted_root)):host.bound(p,1<<30)['sha256']for p in admitted_root.rglob('*')if p.is_file()and p.name!='report.json'}==full_rosters[admitted_root],'Original raw roster changed during analysis')
    for path,before in bindings.items():require(host.bound(path,1<<30)==before,'Consumed analysis input changed '+path)
    require(source_binding()==source,'Analysis source changed after raw reads')
    return {'schema':1,'scope':'closed warm43 selected-host and post-resume raw device intervals; teardown may be included','analysis_source':source,'parent_report_sha256':expected_report_sha256,'public_admission_sha256':expected_public_sha256,'OFF_component_public_sha256':expected_off_component_sha256,'original_OFF_whole_parent_passed':False,'OFF_native_rerun':False,'raw_bindings':bindings,'host_arms':arms,'profiler_OFF_ON_observations':comparisons,'raw_device_observed_operation_unions':device,'actual_GPU_kernel_names_and_counts':kernel_summary,'observed_trace_flow_coverage':observed_flow,'device_interval_scope':'all actual post-resume normalized operations, including possible teardown; not host-aligned selected-only device time','host_metrics_scope':'selected request2 only; warmup excluded','conversation_state_reuse_claimed':False,'nested_host_kinds_are_not_additive':True,'H36_export_IO_outside_request_span':True,'TEmit_is_native_marker_not_client_delivery':True,'device_union_uses_raw_ticks_not_host_normalized_reordering':True,'host_device_clock_alignment_qualified':False,'complete_operation_coverage_qualified':False,'physical_idle_qualified':False,'bandwidth_qualified':False,'clean_serving_speed_qualified':False,'current_model_or_health_readmission_performed':False,'no_model_payload_read':True}

def read_unique_line(line):return json.loads(line,object_pairs_hook=__import__('serial37_canonical_json_v3').unique_object,parse_constant=__import__('serial37_canonical_json_v3').finite_constant)
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--public-admission',type=Path,required=True);p.add_argument('--expected-report-sha256',required=True);p.add_argument('--expected-public-sha256',required=True);p.add_argument('--off-component-public',type=Path,required=True);p.add_argument('--expected-off-component-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists()and a.root.resolve()not in a.output.resolve().parents and Path(read_unique(a.off_component_public)['root']).resolve()not in a.output.resolve().parents,'New external analysis output outside both original trees required');value=analyze(a.root,a.public_admission,a.expected_report_sha256,a.expected_public_sha256,a.off_component_public,a.expected_off_component_sha256);a.output.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='ascii')
if __name__=='__main__':main()
