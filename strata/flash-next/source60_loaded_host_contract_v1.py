"""One loaded native owner, exact repeated histories and bounded host-only localization."""
import json
from pathlib import Path
from source60_loaded_fairness_observations_v1 import requests as observe
from source57_native2_boundary_contract_v1 import reference
from source59_native2_pacing_contract_v1 import reference58
from native60_loaded_fairness_protocol_v1 import require,fields,command_for

def host_rows(directory,requests):
    lines=(directory/'engine.combined.log').read_text(encoding='ascii').splitlines();groups={};ends={};owners={x['rid']:x for x in requests}
    for line in lines:
        if not line.startswith('PFH60 '):continue
        require(line.startswith(('PFH60 row ','PFH60 end ')),'Diagnostic run bound/error or unknown record')
        f=fields(line);run=int(f['run']);require(1<=run<=8,'Bounded actual diagnostic run required')
        if line.startswith('PFH60 end '):require(run not in ends,'Duplicate diagnostic end');ends[run]=f;continue
        require(f['rid'].isdigit() and int(f['rid']) in owners,'Foreign diagnostic request owner');rid=int(f['rid']);owner=owners[rid]
        require(int(f['slot'])==owner['slot'] and int(f['slotgen'])==owner['slotgen'] and int(f['pid'])==1 and int(f['tid'])>0 and f['owner'].startswith('0x') and int(f['T'])==512 and int(f['pos'])==24,'Actual native PID/thread/object/owner/shape association differs')
        require(int(f['device']) in (0,1) and int(f['host_steady_ns'])>0,'Actual diagnostic device/host timestamp required');groups.setdefault(run,[]).append(f)
    require(set(groups)==set(ends)==set(range(1,9)),'Exactly4 request owners x2 stage first512 diagnostics required')
    result=[];seen=set()
    for run,rows in sorted(groups.items()):
        end=ends[run];first=rows[0];require(len(rows)<=64 and int(end['rows'])==len(rows) and end['overflow']=='0' and end['deferred']=='1' and end['host_only']=='1','Complete bounded deferred rows required')
        require([int(x['seq']) for x in rows]==list(range(len(rows))) and [int(x['host_steady_ns']) for x in rows]==sorted(int(x['host_steady_ns']) for x in rows),'Actual row sequence/host monotonic order differs')
        require(all(all(x[k]==first[k] for k in ('pid','tid','owner','rid','slotgen','slot','device','pos','T','run')) for x in rows) and all(end[k]==first[k] for k in ('pid','tid','owner','rid','run')),'Diagnostic row/end association differs')
        key=(int(first['rid']),int(first['device']));require(key not in seen,'Duplicate request/stage diagnostic');seen.add(key)
        require(rows[0]['phase']=='run_enter' and rows[-1]['phase']=='chunk_layers_submitted','Selected body boundaries incomplete')
        mandatory={'run_enter','device_scope_ready','run_preparation_ready','chunk_prepare_begin','embedding_submission_end','ple_schedule_end','expert_plan_begin','expert_plan_issuer_start_end','chunk_layers_submitted'}
        require(mandatory <= {x['phase'] for x in rows},'Selected preparation/embedding/expert-plan host coverage incomplete')
        begin=0 if int(first['device'])==0 else 24
        for layer in (begin,begin+1):
            for phase in ('layer_enter','router_existing_wait_begin','router_existing_wait_end'):require(sum(x['phase']==phase and int(x['layer'])==layer for x in rows)==1,'Actual early layer/router wait coverage incomplete')
        if begin==0:require(sum(x['phase']=='ple_future_get_begin' for x in rows)==sum(x['phase']=='ple_future_get_end' for x in rows)==1,'Original PLE get host coverage incomplete')
        spans=[]
        for a,b in zip(rows,rows[1:]):spans.append(dict(begin_phase=a['phase'],end_phase=b['phase'],begin_layer=int(a['layer']),end_layer=int(b['layer']),host_elapsed_ms=(int(b['host_steady_ns'])-int(a['host_steady_ns']))/1e6,begin_ne=int(a['ne']),end_ne=int(b['ne'])))
        result.append(dict(run=run,rid=key[0],device=key[1],owner=first['owner'],tid=int(first['tid']),raw_rows=rows,adjacent_host_spans=spans))
    require(seen=={(rid,dev) for rid in owners for dev in (0,1)},'Complete request/stage census differs')
    return dict(selected_runs=result,host_only=True,GPU_elapsed_qualified=False,host_to_Python_clock_alignment_qualified=False,budget_runtime_ledger_qualified=False,clean_performance_qualified=False)

def requests(rows,expected):
    require(len(rows)==len(expected)==4,'Two sequential original A/C cohorts required');cohorts=[];owned=[]
    for cohort in (0,1):
        pair=rows[2*cohort:2*cohort+2];wants=expected[2*cohort:2*cohort+2];value=observe(pair,wants)
        for current,prior,prior58 in zip(value['streams'],reference(),reference58()):require(all(current[k]==prior[k]==prior58[k] for k in ('role','ids','output_ids','finish')),'Loaded current60 full history differs from admitted52/58ON')
        cohorts.append(value);owned.extend(value['streams'])
    for a,b in zip(cohorts[0]['streams'],cohorts[1]['streams']):require(all(a[k]==b[k] for k in ('role','ids','output_ids','finish')),'Same loaded owner cold/warm exact history differs')
    return dict(cohorts=cohorts,host_diagnostic=host_rows(Path(rows[0]['raw']['phase_directory']),owned),loaded_processes=1,repeated_fresh_histories_equal=True,accepted52_and58_ON_histories_equal=True,actual_row_counts=[x['observed_completed_window_rows'] for x in cohorts],source_budget_math_bound=True,runtime_budget_ledger_qualified=False,native2_simultaneous_coherence_qualified=False,API_qualified=False,clean_performance_qualified=False)
def equivalence(*arms):
    require(len(arms)==1 and arms[0]['loaded_processes']==1 and arms[0]['repeated_fresh_histories_equal'] is True,'One owned loaded context with two bounded cohorts required')
    return dict(bounded60_loaded_host_localization_passed=True,loaded_observation=arms[0],scope='Same fixed cap1/diag1 native process, fresh A512/C64 twice; cold-vs-warm local host spans only, no cap causality/calibrated GPU/clean speed claim',API_qualified=False,budget_runtime_ledger_qualified=False,cancellation_usability_qualified=False,clean_performance_qualified=False)
