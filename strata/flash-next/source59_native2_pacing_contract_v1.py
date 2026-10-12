"""Actual59 cap0/1 histories/service census; no synthetic runtime budget ledger."""
import hashlib,json
from pathlib import Path
from source58_native2_cooperative_contract_v1 import requests as prior_requests
from native2_fairness_protocol_v1 import require,fields,command_for
REFERENCE58=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/source58-native2-cooperative-v1-public-v1.json')
REFERENCE58_SHA='81aebab33636fc019d15557432f3341f0830cb3d3a7959b826e7f2408252ded5'
def reference58():
    before=REFERENCE58.stat();raw=REFERENCE58.read_bytes();after=REFERENCE58.stat();sig=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    require(sig(before)==sig(after) and hashlib.sha256(raw).hexdigest()==REFERENCE58_SHA,'Immutable admitted58 public reference changed')
    value=json.loads(raw);require(value['passed'] is True and value['source_generation']==58 and value['bounded58_cooperative_scheduler_histories_passed'] is True and value['cooperative_scheduler_comparison']['actual58_OFF_ON_full_histories_equal'] is True,'Authentic independently admitted58 required')
    return value['cooperative_scheduler_comparison']['ON']['stream_timings']
def service_windows(directory):
    recipe=json.loads((directory/'recipe.json').read_bytes());cap=recipe['native_env']['STRATA_BATCH_PREFILL_SINGLE_WINDOW'];require(cap in ('0','1') and recipe['native_env']['STRATA_BATCH_PREFILL_LAYER_COOPERATE']=='1','Actual59 matched pacing/cooperation flags required')
    lines=(directory/'engine.combined.log').read_text(encoding='ascii').splitlines();calls=[]
    for index,line in enumerate(lines):
        if not line.startswith('strata cooperate:'):continue
        # Original batch_step may finish the long slot inside a service. Its
        # terminal/profile prints precede the service record; they remain raw.
        region=[];j=index-1
        while j>=0 and lines[j].startswith(('SBF batch_event ','BT ','BDONE ','strata batch: ','strata batch GPU stages,')):
            region.append(lines[j]);j-=1
        region.reverse();windows=[fields(row) for row in region if row.startswith('SBF batch_event ')];tokens=[row for row in region if row.startswith('BT ')]
        require(windows and all(x['completed']=='1' and x['rows'] in ('1','2') for x in windows) and sum(int(x['rows']) for x in windows)==len(tokens),'Exact actual service window/token association required')
        require(cap=='0' or len(windows)<=1,'Single-windowON emitted more than one actual batch window per service')
        calls.append(dict(service_record=line,actual_window_count=len(windows),actual_events=[int(x['event']) for x in windows],actual_rows=[int(x['rows']) for x in windows],actual_BT_lines=tokens))
    events=[event for call in calls for event in call['actual_events']];require(len(events)==len(set(events)),'Actual service windows double counted')
    return dict(single_window_enabled=cap=='1',observed_service_calls=len(calls),observed_service_windows=len(events),maximum_windows_per_service=max((x['actual_window_count'] for x in calls),default=None),service_window_records=calls,absence_means='No positive service observed; zero-service ACKs unobservable',runtime_budget_ledger_qualified=False)
def requests(rows,expected):
    value=prior_requests(rows,expected)
    for row,prior in zip(value['streams'],reference58()):require(all(row[k]==prior[k] for k in ('role','ids','output_ids','finish')),'Actual59 full history differs from admitted58 ON')
    value['accepted58_ON_history_reference_sha256']=REFERENCE58_SHA;value['pacing_service_observation']=service_windows(Path(rows[0]['raw']['phase_directory']));return value
def equivalence(off,on):
    require(off['default_scheduler_observed'] is True and on['default_scheduler_observed'] is True,'Both actual59 arrival arms required')
    require(off['cooperative_service_observation']['enabled'] is True and on['cooperative_service_observation']['enabled'] is True and off['pacing_service_observation']['single_window_enabled'] is False and on['pacing_service_observation']['single_window_enabled'] is True,'Cooperation1 and matched cap0/1 required')
    for a,b in zip(off['streams'],on['streams']):require(all(a[k]==b[k] for k in ('role','ids','output_ids','finish')),'Actual59 whole histories differ')
    def arm(value):return dict(stream_timings=value['streams'],arrival_window_gaps=value['arrival_window_gaps'],actual_two_row_events=len(value['actual_two_row_events']),observed_completed_window_rows=value['observed_completed_window_rows'],cooperative_service_observation=value['cooperative_service_observation'],pacing_service_observation=value['pacing_service_observation'])
    return dict(bounded59_pacing_scheduler_histories_passed=True,actual59_OFF_ON_full_histories_equal=True,accepted52_serial_histories_equal=True,accepted58_ON_histories_equal=True,accepted58_ON_history_reference_sha256=REFERENCE58_SHA,default_decode_share=.5,same_host_monotonic_domain=True,OFF=arm(off),ON=arm(on),single_window_ON_observed=bool(on['pacing_service_observation']['observed_service_calls']),single_window_ON_maximum_observed=on['pacing_service_observation']['maximum_windows_per_service'],native2_simultaneous_coherence_qualified=False,coadmission_qualification_open=True,budget_runtime_accounting_qualified=False,cancellation_usability_qualified=False,raw_capture_armed=False,HTTP_latency_qualified=False,fullmath_qualified=False,fairness_stability_qualified=False,clean_performance_qualified=False,speed_promotion_qualified=False,scope='Two owned actual59 processes; cooperation1 fixed, single-window0/1 only delta; complete52+58ON histories, observed service windows/rows and local host latency; budget/cancel/API/4/6/clean-performance qualification open')
