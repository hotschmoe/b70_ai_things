"""Actual58 histories/timing plus observed service; no invented overlap or budget ledger."""
import json, math, re
from decimal import Decimal
from pathlib import Path
from source57_native2_boundary_contract_v1 import reference, REFERENCE_SHA
from source58_native2_fairness_observations_v1 import requests as observe_requests
from native2_fairness_protocol_v1 import require, fields, command_for
SERVICE=re.compile(r'^strata cooperate: owner service device=(\d+) layer=(\d+) pos=(\d+) rows=(\d+) host_ms=([0-9]+\.[0-9]{3})$')
def service_observation(directory):
    recipe=json.loads((directory/'recipe.json').read_bytes());flag=recipe['native_env']['STRATA_BATCH_PREFILL_LAYER_COOPERATE'];require(flag in ('0','1'),'Exact cooperative flag required')
    rows=[]
    for line in (directory/'engine.combined.log').read_text(encoding='ascii').splitlines():
        if not line.startswith('strata cooperate:'):continue
        match=SERVICE.fullmatch(line);require(match is not None,'Malformed real cooperative service record')
        dev,layer,pos,count=map(int,match.groups()[:4]);duration=Decimal(match.group(5))
        require(dev in (0,1) and (0<=layer<24 if dev==0 else 24<=layer<48) and 0<count<=512 and pos+count<=2048 and duration>=0,'Actual paired service shape/stage/host duration invalid')
        rows.append(dict(device=dev,layer=layer,prompt_position=pos,prompt_rows=count,reported_host_ms=str(duration)))
    require(flag=='1' or not rows,'DefaultOFF emitted cooperative service')
    summary={str(dev):dict(service_calls=sum(row['device']==dev for row in rows),reported_host_ms=str(sum((Decimal(row['reported_host_ms']) for row in rows if row['device']==dev),Decimal(0)))) for dev in (0,1)}
    return dict(enabled=flag=='1',stage_service=summary,both_stages_service_observed=all(summary[str(dev)]['service_calls']>0 for dev in (0,1)),observed_service_rows=rows,zero_service_ACKs_observable=False,absent_service_record_means='No emitted positive service record; zero-service ACKs are not logged',source_reported_host_ms_precision='.001ms rounding; not calibrated GPU execution',budget_math_source_bound=True,budget_runtime_ledger_available=False,budget_runtime_accounting_qualified=False,cancellation_usability_qualified=False)
def requests(rows,expected):
    value=observe_requests(rows,expected)
    for row,prior in zip(value['streams'],reference()):require(all(row[k]==prior[k] for k in ('role','ids','output_ids','finish')),'Actual58 full history differs from admitted52 serial reference')
    value['accepted52_history_reference_sha256']=REFERENCE_SHA
    value['cooperative_service_observation']=service_observation(Path(rows[0]['raw']['phase_directory']))
    return value
def equivalence(off,on):
    require(off['default_scheduler_observed'] is True and on['default_scheduler_observed'] is True,'Both actual58 bounded native arrival arms required')
    require(off['cooperative_service_observation']['enabled'] is False and on['cooperative_service_observation']['enabled'] is True,'Matched58 OFF/ON flags differ')
    for a,b in zip(off['streams'],on['streams']):require(all(a[k]==b[k] for k in ('role','ids','output_ids','finish')),'Actual58 OFF/ON whole histories differ')
    def arm(value):return dict(stream_timings=value['streams'],arrival_window_gaps=value['arrival_window_gaps'],actual_two_row_events=len(value['actual_two_row_events']),observed_completed_window_rows=value['observed_completed_window_rows'],cooperative_service_observation=value['cooperative_service_observation'])
    return dict(bounded58_cooperative_scheduler_histories_passed=True,actual58_OFF_ON_full_histories_equal=True,accepted52_serial_histories_equal=True,accepted52_history_reference_sha256=REFERENCE_SHA,default_decode_share=.5,same_host_monotonic_domain=True,OFF=arm(off),ON=arm(on),both_stages_service_observed=on['cooperative_service_observation']['both_stages_service_observed'],native2_simultaneous_coherence_qualified=False,coadmission_qualification_open=True,budget_runtime_accounting_qualified=False,cancellation_usability_qualified=False,raw_capture_armed=False,HTTP_latency_qualified=False,fullmath_qualified=False,fairness_stability_qualified=False,clean_performance_qualified=False,speed_promotion_qualified=False,scope='Two owned actual58 processes, matching original52 histories and local host token timings; diagnostic STRATA_TRACE in both arms; observed rows/service only; no runtime budget ledger, cancellation or full4/6/API qualification')
